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
