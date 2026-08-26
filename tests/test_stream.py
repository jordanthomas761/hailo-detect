"""The ffmpeg argv, which is the part most likely to be quietly wrong.

ffmpeg exits on an input option the demuxer does not recognise rather than
ignoring it, so passing RTSP-only flags to an HTTP source breaks the stream
outright -- and only at runtime, on hardware. These assert the shape of the
command without needing ffmpeg or a camera.
"""

from __future__ import annotations

import pytest

from hailo_detect.stream import ffmpeg_command, redact

ARGS = {"transport": "tcp", "fps": 5.0, "width": 640, "height": 640}


def test_rtsp_gets_the_rtsp_only_flags():
    command = ffmpeg_command("rtsp://cam.local/stream1", **ARGS)

    assert "-rtsp_transport" in command
    assert command[command.index("-rtsp_transport") + 1] == "tcp"
    # -timeout, not the -stimeout that ffmpeg 7 dropped.
    assert "-timeout" in command
    assert "-stimeout" not in command


@pytest.mark.parametrize(
    "url",
    ["http://ml.k8s.internal:8080/stream", "https://cam/feed.mjpg", "file:///tmp/clip.mp4"],
    ids=["http", "https", "file"],
)
def test_other_inputs_get_no_rtsp_flags(url):
    command = ffmpeg_command(url, **ARGS)

    assert "-rtsp_transport" not in command
    assert "-timeout" not in command
    assert command[command.index("-i") + 1] == url


def test_output_is_always_letterboxed_rawvideo():
    command = ffmpeg_command("rtsp://cam/1", **ARGS)

    video_filter = command[command.index("-vf") + 1]
    assert "fps=5.0" in video_filter
    assert "scale=640:640:force_original_aspect_ratio=decrease" in video_filter
    # Must match imaging._PAD (114,114,114) or an uploaded image and a stream
    # frame are letterboxed differently.
    assert "color=0x727272" in video_filter
    assert command[-5:] == ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def test_credentials_are_stripped_before_logging():
    assert redact("rtsp://user:secret@cam.local:554/stream1") == "rtsp://***@cam.local:554/stream1"
    assert "secret" not in redact("rtsp://user:secret@cam.local/s")


def test_redact_survives_a_url_it_cannot_parse():
    assert redact("not a url") == "<invalid rtsp url>"


class _DummyEngine:
    """StreamWorker only touches the engine from its threads, which never start."""

    @property
    def input_size(self) -> tuple[int, int]:
        return (640, 640)


def worker(**overrides):
    from hailo_detect.config import Settings
    from hailo_detect.stream import StreamWorker

    settings = Settings(stream_url="http://cam.local:8080/stream", **overrides)
    return StreamWorker(settings, _DummyEngine())  # type: ignore[arg-type]


def test_on_demand_keeps_the_source_closed_until_something_watches():
    w = worker()

    assert w._wanted() is False

    w.add_viewer()
    assert w._wanted() is True


def test_the_last_viewer_leaving_starts_a_grace_period():
    # A page reload should not close the camera and reopen it a second later.
    w = worker(stream_idle_timeout_s=60.0)
    w.add_viewer()

    w.remove_viewer()

    assert w._wanted() is True


def test_the_source_closes_once_the_grace_period_expires():
    w = worker(stream_idle_timeout_s=0.0)
    w.add_viewer()

    w.remove_viewer()

    assert w._wanted() is False


def test_concurrent_viewers_are_counted_not_toggled():
    w = worker(stream_idle_timeout_s=0.0)
    w.add_viewer()
    w.add_viewer()

    w.remove_viewer()

    # One viewer left, so the camera must stay open.
    assert w._wanted() is True
    w.remove_viewer()
    assert w._wanted() is False


def test_always_on_never_closes_the_source():
    w = worker(stream_on_demand=False)

    assert w._wanted() is True
    w.add_viewer()
    w.remove_viewer()
    assert w._wanted() is True


def test_closing_the_source_forgets_what_it_captured():
    # The privacy half: a frame taken while someone watched must not still be
    # served from /api/stream/snapshot.jpg after the camera light goes out.
    w = worker()
    w._jpeg.put(b"\xff\xd8jpeg-bytes")
    w._detections = [object()]  # type: ignore[list-item]

    w._forget_frames()

    assert w.latest_jpeg()[0] is None
    assert w.status()["detections"] == []
