"""Runtime configuration, all of it from the environment.

Everything here has a working default except RTSP_URL, so the container runs
with no env at all and the stream half stays dormant until a URL is supplied.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _str(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _opt_str(name: str) -> str | None:
    return os.environ.get(name, "").strip() or None


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    # The hailo-models package installs HEFs here. Use the _h8 variants --
    # _h8l is Hailo-8L and _h10 is Hailo-10, and loading the wrong one fails
    # at configure time with an architecture mismatch.
    hef_path: str = "/usr/share/hailo-models/yolov8s_h8.hef"
    score_threshold: float = 0.4
    max_detections: int = 50

    host: str = "0.0.0.0"  # noqa: S104 -- a container port, fronted by a Service
    port: int = 8000

    max_upload_bytes: int = 16 * 1024 * 1024
    jpeg_quality: int = 80

    # One device, one inference thread: every request and every stream frame
    # queues behind the same worker. The queue is short on purpose -- a deep
    # queue converts overload into latency instead of a fast 503.
    infer_queue_size: int = 8
    infer_timeout_s: float = 15.0

    # Any URL ffmpeg can read: rtsp://, http:// (an MJPEG stream), or a file.
    # It is not RTSP-specific -- a USB capture device on the host is easier to
    # publish as MJPEG over HTTP than as RTSP.
    stream_url: str | None = None
    # Applies to rtsp:// only, and is passed only for those URLs; ffmpeg
    # errors on an unknown option rather than ignoring it.
    rtsp_transport: str = "tcp"
    stream_fps: float = 5.0
    # ffmpeg is restarted with this backoff when the source drops. Cameras
    # reboot, and a stream that never comes back on its own is worse than one
    # that reconnects noisily.
    stream_reconnect_min_s: float = 1.0
    stream_reconnect_max_s: float = 30.0

    @classmethod
    def from_env(cls) -> Settings:
        transport = _str("RTSP_TRANSPORT", "tcp")
        if transport not in {"tcp", "udp"}:
            raise ValueError(f"RTSP_TRANSPORT must be 'tcp' or 'udp', got {transport!r}")
        return cls(
            hef_path=_str("HEF_PATH", cls.hef_path),
            score_threshold=_float("SCORE_THRESHOLD", cls.score_threshold),
            max_detections=_int("MAX_DETECTIONS", cls.max_detections),
            host=_str("HOST", cls.host),
            port=_int("PORT", cls.port),
            max_upload_bytes=_int("MAX_UPLOAD_BYTES", cls.max_upload_bytes),
            jpeg_quality=_int("JPEG_QUALITY", cls.jpeg_quality),
            infer_queue_size=_int("INFER_QUEUE_SIZE", cls.infer_queue_size),
            infer_timeout_s=_float("INFER_TIMEOUT_S", cls.infer_timeout_s),
            # RTSP_URL/RTSP_FPS still work: they were the original names and
            # cost nothing to keep.
            stream_url=_opt_str("STREAM_URL") or _opt_str("RTSP_URL"),
            rtsp_transport=transport,
            stream_fps=_float("STREAM_FPS", _float("RTSP_FPS", cls.stream_fps)),
        )
