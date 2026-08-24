"""The Hailo-8 device, owned by exactly one thread.

There is one accelerator in the node and HailoRT's Python objects are not
thread-safe, so every caller -- HTTP upload or RTSP frame -- posts a job onto
a queue that a single worker thread drains. That thread opens the device,
configures the network group once, and keeps the vstream pipeline alive for
the life of the process; per-request configure is what makes naive HailoRT
services slow.

Importing ``hailo_platform`` is deliberately soft. It exists only inside the
container (apt's python3-hailort), and the rest of this package -- the
letterbox maths, the NMS decode, the routes -- has to stay importable and
testable on a laptop that has never seen a Hailo.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import stat
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Settings
from .postprocess import Detection, decode_nms

log = logging.getLogger(__name__)

# Where hailort_service listens. HailoRT has this compiled in, so it is not
# configurable from here -- the pod spec has to mount the host's socket at
# exactly this path.
SERVICE_SOCKET = "/tmp/hailort_uds.sock"

try:  # pragma: no cover -- present only in the image
    import hailo_platform as hpf

    HAILO_IMPORT_ERROR: str | None = None
except Exception as exc:  # noqa: BLE001 -- any import failure is the same story here
    hpf = None
    HAILO_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def check_service_socket(path: str = SERVICE_SOCKET) -> None:
    """Fail with a diagnosable message if hailort_service is not reachable.

    Without this, a missing socket surfaces as a bare HailoRT status code from
    somewhere inside VDevice creation, which says nothing about which of the
    two things went wrong. Both are one-line checks for whoever reads
    /readyz.
    """
    try:
        mode = os.stat(path).st_mode
    except FileNotFoundError:
        raise RuntimeError(
            f"hailort_service socket {path} is not present -- either "
            "hailort.service is not running on this node "
            "(systemctl status hailort.service), or the hostPath mount for it "
            "is missing from the pod spec"
        ) from None
    except OSError as exc:
        raise RuntimeError(f"cannot stat {path}: {exc}") from exc

    if not stat.S_ISSOCK(mode):
        raise RuntimeError(
            f"{path} exists but is not a socket (mode {mode:o}) -- a hostPath "
            "without `type: Socket` creates a directory when the host path is "
            "absent, which is the usual cause"
        )


class EngineNotReady(RuntimeError):
    """The device is not open (yet, or at all)."""


class EngineBusy(RuntimeError):
    """The inference queue is full -- the accelerator is saturated."""


@dataclass(frozen=True)
class InferenceResult:
    detections: list[Detection]
    duration_ms: float


@dataclass
class _Job:
    frame: np.ndarray
    future: Future


class HailoEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._queue: queue.Queue[_Job | None] = queue.Queue(maxsize=settings.infer_queue_size)
        self._thread = threading.Thread(target=self._run, name="hailo-inference", daemon=True)
        self._ready = threading.Event()
        self._stopping = threading.Event()

        self._lock = threading.Lock()
        self._input_size: tuple[int, int] | None = None  # (width, height)
        self._error: str | None = None
        self._frames = 0
        self._failures = 0
        self._last_ms: float | None = None
        self._model_name: str | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self, timeout: float = 60.0) -> None:
        """Start the worker and block until the device is open or has failed.

        Blocking here is the point: it turns "the HEF is missing" or "the
        driver version does not match" into a startup failure with a log line,
        rather than a 503 on the first request.
        """
        self._thread.start()
        if self._ready.wait(timeout):
            return
        # Giving up waiting is not proof of failure: the worker may set the
        # event moments later. _note_timeout drops the error if that has
        # already happened, and the worker clears a stale one if it happens
        # next -- both sides mutate the pair under the same lock, so either
        # ordering ends with the engine reflecting the device.
        self._note_timeout(f"device did not become ready within {timeout:.0f}s")

    def close(self) -> None:
        self._stopping.set()
        # A full queue means the worker is busy and will see _stopping on its
        # next pass, so failing to post the sentinel is not a problem.
        with contextlib.suppress(queue.Full):
            self._queue.put_nowait(None)
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)

    # -- state -------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._error is None

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._error

    @property
    def input_size(self) -> tuple[int, int]:
        with self._lock:
            if self._input_size is None:
                raise EngineNotReady(self._error or "device not open")
            return self._input_size

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": self._ready.is_set() and self._error is None,
                "error": self._error,
                "model": self._model_name,
                "input_size": (
                    {"width": self._input_size[0], "height": self._input_size[1]}
                    if self._input_size
                    else None
                ),
                "queue_depth": self._queue.qsize(),
                "queue_capacity": self._settings.infer_queue_size,
                "frames": self._frames,
                "failures": self._failures,
                "last_inference_ms": (
                    round(self._last_ms, 2) if self._last_ms is not None else None
                ),
            }

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._error = message
            # Unblocks start(); `ready` still reads False because _error is
            # set. Both are written under the lock so no one observes half.
            self._ready.set()

    def _note_timeout(self, message: str) -> None:
        """Record a startup timeout, unless the device opened in the meantime."""
        with self._lock:
            if self._ready.is_set():
                return
            self._error = message
            self._ready.set()

    # -- inference ---------------------------------------------------------

    def infer(self, frame: np.ndarray) -> InferenceResult:
        """Run one frame. ``frame`` must already be HWC uint8 at the input size."""
        if not self.ready:
            raise EngineNotReady(self.error or "device not open")

        job = _Job(frame=frame, future=Future())
        try:
            self._queue.put_nowait(job)
        except queue.Full as exc:
            raise EngineBusy("inference queue is full") from exc

        return job.future.result(timeout=self._settings.infer_timeout_s)

    # -- worker ------------------------------------------------------------

    def _run(self) -> None:
        if hpf is None:
            self._set_error(f"hailo_platform is not importable ({HAILO_IMPORT_ERROR})")
            return
        try:
            self._serve()
        except Exception as exc:  # noqa: BLE001 -- surfaced through /readyz
            log.exception("inference worker stopped")
            self._set_error(f"{type(exc).__name__}: {exc}")
            self._drain(exc)

    def _serve(self) -> None:
        settings = self._settings

        # Checked before touching HailoRT so the common deployment mistakes
        # name themselves rather than arriving as a status code.
        check_service_socket()

        params = hpf.VDevice.create_params()
        # Go through hailort_service rather than opening /dev/hailo0.
        #
        # A container cannot open the device node directly under Kubernetes:
        # a hostPath mount hands over the inode but adds no device-cgroup
        # rule, so open() returns EPERM even though the node is mode 666.
        # (`docker --device` adds that rule for you, which is why the same
        # image works under plain Docker and the README used to claim this
        # was enough.) The alternatives were a privileged device plugin or
        # this.
        #
        # hailort_service already runs on the host as a systemd unit and owns
        # the device, so the client side needs no device access at all --
        # only its Unix socket, mounted in at the path HailoRT expects. It
        # also means several pods can share one accelerator, which direct
        # access cannot do.
        params.multi_process_service = True
        # The service requires the scheduler. That is not a free choice: with
        # ROUND_ROBIN enabled, `network_group.activate()` becomes illegal and
        # the scheduler activates network groups itself -- which is why the
        # activate() call that used to wrap the inference loop is gone.
        params.scheduling_algorithm = hpf.HailoSchedulingAlgorithm.ROUND_ROBIN
        # Clients naming the same group share one underlying VDevice instead
        # of each trying to claim the device for themselves.
        params.group_id = "SHARED"

        with hpf.VDevice(params) as vdevice:
            hef = hpf.HEF(settings.hef_path)
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef, interface=hpf.HailoStreamInterface.PCIe
            )
            network_group = vdevice.configure(hef, configure_params)[0]

            input_info = hef.get_input_vstream_infos()[0]
            output_info = hef.get_output_vstream_infos()[0]
            height, width = int(input_info.shape[0]), int(input_info.shape[1])

            input_params = hpf.InputVStreamParams.make(
                network_group, format_type=hpf.FormatType.UINT8
            )
            # FLOAT32 out: the on-chip NMS emits normalised coordinates and a
            # score, and asking for quantised output here would mean
            # dequantising them again by hand.
            output_params = hpf.OutputVStreamParams.make(
                network_group, format_type=hpf.FormatType.FLOAT32
            )

            with self._lock:
                self._input_size = (width, height)
                self._model_name = network_group.name

            log.info(
                "hailo device open via hailort_service: model=%s input=%dx%d output=%s hef=%s",
                network_group.name,
                width,
                height,
                output_info.name,
                settings.hef_path,
            )

            # No activate() here -- see the scheduler note above.
            with hpf.InferVStreams(network_group, input_params, output_params) as pipeline:
                with self._lock:
                    # The device is open, so any error start() recorded while
                    # it was still opening is stale.
                    self._error = None
                    self._ready.set()
                self._loop(pipeline, input_info.name, output_info.name, (height, width))

    def _loop(
        self,
        pipeline: Any,
        input_name: str,
        output_name: str,
        expected_hw: tuple[int, int],
    ) -> None:
        while not self._stopping.is_set():
            job = self._queue.get()
            if job is None:
                return
            if job.future.cancelled():
                continue
            try:
                job.future.set_result(
                    self._infer_one(pipeline, input_name, output_name, expected_hw, job.frame)
                )
            except Exception as exc:  # noqa: BLE001 -- one bad frame must not end the loop
                with self._lock:
                    self._failures += 1
                job.future.set_exception(exc)

    def _infer_one(
        self,
        pipeline: Any,
        input_name: str,
        output_name: str,
        expected_hw: tuple[int, int],
        frame: np.ndarray,
    ) -> InferenceResult:
        if frame.shape[:2] != expected_hw:
            raise ValueError(
                f"frame is {frame.shape[:2]}, model wants {expected_hw} (height, width)"
            )

        started = time.perf_counter()
        raw = pipeline.infer({input_name: np.expand_dims(frame, axis=0)})[output_name]
        detections = decode_nms(
            raw,
            score_threshold=self._settings.score_threshold,
            max_detections=self._settings.max_detections,
        )
        duration_ms = (time.perf_counter() - started) * 1000.0

        with self._lock:
            self._frames += 1
            self._last_ms = duration_ms

        return InferenceResult(detections=detections, duration_ms=duration_ms)

    def _drain(self, exc: Exception) -> None:
        """Fail everything still queued, so no caller waits out its timeout."""
        while True:
            try:
                job = self._queue.get_nowait()
            except queue.Empty:
                return
            if job is not None and not job.future.done():
                job.future.set_exception(exc)
