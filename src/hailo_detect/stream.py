"""Pull an RTSP source with ffmpeg and run detection on it continuously.

Two threads, on purpose:

* the **reader** owns the ffmpeg process and does nothing but pull frames out
  of the pipe into a one-slot buffer, overwriting whatever was there;
* the **processor** takes whatever is in that slot and runs it through the
  accelerator.

Latest-frame-wins is the whole design. A queue between them would trade
freshness for a growing backlog the moment inference is slower than the
source, and on a live camera an old frame has no value -- you want the most
recent one the device can keep up with, and the rest dropped.

ffmpeg is asked to scale and pad to the network input itself, so the frames
arriving here are already letterboxed and can be detected on, and drawn on,
without a further resize.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

from .config import Settings
from .engine import EngineBusy, EngineNotReady, HailoEngine
from .imaging import draw_detections, encode_jpeg
from .postprocess import Detection

log = logging.getLogger(__name__)


def redact(url: str) -> str:
    """Strip any user:password from an RTSP URL before it reaches a log line."""
    parts = urlsplit(url)
    if not parts.hostname:
        return "<invalid rtsp url>"
    netloc = parts.hostname
    if parts.username:
        netloc = f"***@{netloc}"
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


class _Slot:
    """A one-slot buffer with a sequence number, for latest-frame-wins."""

    def __init__(self) -> None:
        self._new_frame = threading.Condition()
        self._lock = self._new_frame
        self._value: Any = None
        self._seq = 0

    def put(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._seq += 1
            self._new_frame.notify_all()

    def get(self) -> tuple[Any, int]:
        with self._lock:
            return self._value, self._seq

    def wait_for(self, after_seq: int, timeout: float) -> tuple[Any, int] | None:
        """Block until a frame newer than ``after_seq`` lands, or time out."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._new_frame.wait(remaining)
            return self._value, self._seq


class RtspWorker:
    def __init__(self, settings: Settings, engine: HailoEngine) -> None:
        if not settings.rtsp_url:
            raise ValueError("RtspWorker needs settings.rtsp_url")
        self._settings = settings
        self._engine = engine
        self._url = settings.rtsp_url
        self._stopping = threading.Event()

        self._raw = _Slot()
        self._jpeg = _Slot()

        self._reader = threading.Thread(target=self._read_loop, name="rtsp-reader", daemon=True)
        self._processor = threading.Thread(
            target=self._process_loop, name="rtsp-detect", daemon=True
        )
        self._process: subprocess.Popen[bytes] | None = None

        self._lock = threading.Lock()
        self._connected = False
        self._last_error: str | None = None
        self._frames_read = 0
        self._frames_detected = 0
        self._frames_dropped = 0
        self._connects = 0
        self._detections: list[Detection] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if shutil.which("ffmpeg") is None:
            self._note_error("ffmpeg is not on PATH")
            return
        self._reader.start()
        self._processor.start()

    def stop(self) -> None:
        self._stopping.set()
        self._kill_ffmpeg()
        # Unblock the processor's wait_for() so it can see the stop flag.
        self._raw.put(None)
        for thread in (self._reader, self._processor):
            if thread.is_alive():
                thread.join(timeout=5.0)

    # -- state -------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": True,
                "url": redact(self._url),
                "connected": self._connected,
                "last_error": self._last_error,
                "connects": self._connects,
                "frames_read": self._frames_read,
                "frames_detected": self._frames_detected,
                "frames_dropped": self._frames_dropped,
                "target_fps": self._settings.rtsp_fps,
                "detections": [d.as_dict() for d in self._detections],
            }

    def latest_jpeg(self) -> tuple[bytes | None, int]:
        return self._jpeg.get()

    def wait_for_jpeg(self, after_seq: int, timeout: float) -> tuple[bytes, int] | None:
        result = self._jpeg.wait_for(after_seq, timeout)
        if result is None or result[0] is None:
            return None
        return result

    def _note_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._connected = False
        log.warning("rtsp %s: %s", redact(self._url), message)

    # -- reader ------------------------------------------------------------

    def _ffmpeg_command(self, width: int, height: int) -> list[str]:
        settings = self._settings
        # Pad colour 0x727272 == (114, 114, 114), matching imaging._PAD so an
        # uploaded image and a stream frame are letterboxed identically.
        video_filter = (
            f"fps={settings.rtsp_fps},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=0x727272"
        )
        return [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel", "error",
            "-rtsp_transport", settings.rtsp_transport,
            # Socket I/O timeout, in microseconds -- fail the connect
            # rather than hang forever on a camera that is off, and let the
            # reconnect loop below retry. This option is spelled `-timeout`
            # on the RTSP demuxer in ffmpeg >= 5.0; the older `-stimeout`
            # name is gone in the ffmpeg 7.x that Debian trixie ships, and an
            # unknown demuxer option makes ffmpeg exit rather than warn.
            "-timeout", "5000000",
            "-i", self._url,
            "-an", "-sn",
            "-vf", video_filter,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ]

    def _read_loop(self) -> None:
        backoff = self._settings.rtsp_reconnect_min_s
        while not self._stopping.is_set():
            try:
                width, height = self._engine.input_size
            except EngineNotReady:
                # The device is still opening (or never will). Wait rather
                # than connecting to the camera with no way to use its frames.
                time.sleep(1.0)
                continue

            try:
                connected_at = time.monotonic()
                self._stream_once(width, height)
                # A session that lasted a while was healthy: reset the backoff
                # so a nightly camera reboot does not creep towards 30s waits.
                if time.monotonic() - connected_at > 30:
                    backoff = self._settings.rtsp_reconnect_min_s
            except Exception as exc:  # noqa: BLE001 -- reconnecting is the whole job
                self._note_error(f"{type(exc).__name__}: {exc}")

            if self._stopping.is_set():
                return
            time.sleep(backoff)
            backoff = min(backoff * 2, self._settings.rtsp_reconnect_max_s)

    def _stream_once(self, width: int, height: int) -> None:
        frame_bytes = width * height * 3
        log.info("rtsp %s: connecting", redact(self._url))

        process = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            self._ffmpeg_command(width, height),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._process = process
        with self._lock:
            self._connects += 1

        try:
            assert process.stdout is not None
            while not self._stopping.is_set():
                buffer = process.stdout.read(frame_bytes)
                if not buffer or len(buffer) < frame_bytes:
                    break  # ffmpeg exited or the stream ended mid-frame
                with self._lock:
                    self._connected = True
                    self._last_error = None
                    self._frames_read += 1
                self._raw.put(buffer)
        finally:
            self._kill_ffmpeg()
            with self._lock:
                self._connected = False
            stderr = b""
            if process.stderr is not None:
                stderr = process.stderr.read() or b""
            if stderr and not self._stopping.is_set():
                self._note_error(stderr.decode("utf-8", "replace").strip().splitlines()[-1])

    def _kill_ffmpeg(self) -> None:
        process, self._process = self._process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    # -- processor ---------------------------------------------------------

    def _process_loop(self) -> None:
        seq = 0
        while not self._stopping.is_set():
            item = self._raw.wait_for(seq, timeout=1.0)
            if item is None:
                continue
            buffer, seq = item
            if buffer is None:
                continue
            try:
                self._detect_frame(buffer)
            except EngineBusy:
                with self._lock:
                    self._frames_dropped += 1
            except EngineNotReady:
                time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001 -- one bad frame is not fatal
                self._note_error(f"detect: {type(exc).__name__}: {exc}")

    def _detect_frame(self, buffer: bytes) -> None:
        width, height = self._engine.input_size
        frame = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
        result = self._engine.infer(frame)

        # The frame is already letterboxed to the network input, so its own
        # pixel space *is* the space the boxes came back in -- no unmapping.
        annotated = draw_detections(Image.fromarray(frame), result.detections)
        self._jpeg.put(encode_jpeg(annotated, self._settings.jpeg_quality))

        with self._lock:
            self._frames_detected += 1
            self._detections = result.detections
