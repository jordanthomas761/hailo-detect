"""HTTP surface: upload an image, or watch the RTSP stream being detected on."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .engine import EngineBusy, EngineNotReady, HailoEngine, InferenceResult
from .imaging import (
    decode_image,
    draw_detections,
    encode_jpeg,
    letterbox,
    map_to_source,
    to_input_array,
)
from .postprocess import Detection
from .stream import StreamWorker

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
MJPEG_BOUNDARY = "frame"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = HailoEngine(settings)
        # Opening the device is slow (~seconds) and blocking; keep it off the
        # event loop so the health endpoints answer during startup.
        await asyncio.to_thread(engine.start)
        if engine.error:
            # Deliberately not fatal: the pod stays up, /readyz stays red and
            # names the reason. A CrashLoopBackOff hides the error message
            # behind a restart count.
            log.error("hailo device unavailable: %s", engine.error)

        worker: StreamWorker | None = None
        if settings.stream_url:
            worker = StreamWorker(settings, engine)
            worker.start()

        app.state.settings = settings
        app.state.engine = engine
        app.state.stream = worker
        try:
            yield
        finally:
            if worker is not None:
                worker.stop()
            engine.close()

    app = FastAPI(title="hailo-detect", version="0.1.0", lifespan=lifespan)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    _register_routes(app)
    return app


def _settings(request: Request) -> Settings:
    # Routes read settings off app.state rather than closing over the object
    # create_app was handed, so there is exactly one live copy per app.
    return request.app.state.settings


def _engine(request: Request) -> HailoEngine:
    return request.app.state.engine


def _stream(request: Request) -> StreamWorker:
    worker = request.app.state.stream
    if worker is None:
        raise HTTPException(status_code=404, detail="no stream source configured (set STREAM_URL)")
    return worker


async def _run_inference(engine: HailoEngine, frame: Any) -> InferenceResult:
    """Hand one frame to the device thread, mapping its failures onto HTTP."""
    try:
        return await asyncio.to_thread(engine.infer, frame)
    except EngineNotReady as exc:
        raise HTTPException(status_code=503, detail=f"accelerator not ready: {exc}") from exc
    except EngineBusy as exc:
        raise HTTPException(
            status_code=503,
            detail="accelerator is saturated, retry shortly",
            headers={"Retry-After": "1"},
        ) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="inference timed out") from exc


async def _read_upload(image: UploadFile, limit: int) -> bytes:
    """Read an upload, refusing anything over the limit without buffering it."""
    chunks: list[bytes] = []
    total = 0
    while chunk := await image.read(1 << 20):
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status_code=413, detail=f"image exceeds MAX_UPLOAD_BYTES ({limit} bytes)"
            )
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(status_code=400, detail="empty upload")
    return b"".join(chunks)


async def _detect_upload(
    request: Request, image: UploadFile
) -> tuple[list[Detection], InferenceResult, Any]:
    settings: Settings = request.app.state.settings
    engine = _engine(request)

    data = await _read_upload(image, settings.max_upload_bytes)
    try:
        source = decode_image(data)
    except Exception as exc:  # noqa: BLE001 -- Pillow raises a wide range here
        raise HTTPException(status_code=400, detail=f"could not decode image: {exc}") from exc

    try:
        input_size = engine.input_size
    except EngineNotReady as exc:
        raise HTTPException(status_code=503, detail=f"accelerator not ready: {exc}") from exc

    canvas, box = letterbox(source, input_size)
    result = await _run_inference(engine, to_input_array(canvas))
    detections = [map_to_source(d, box) for d in result.detections]
    return detections, result, source


async def _mjpeg_frames(request: Request, worker: StreamWorker) -> AsyncIterator[bytes]:
    """Yield multipart JPEG parts for as long as the client stays connected."""
    seq = 0
    while not await request.is_disconnected():
        # Each waiter parks a thread from asyncio's default executor for up
        # to the timeout below. That pool is min(32, cpu_count + 4) -- eight
        # on this four-core Pi -- and it is NOT anyio's limiter, which only
        # bounds anyio.to_thread.run_sync. It is shared with the upload path,
        # so enough simultaneous viewers would make /api/detect queue behind
        # them. Fine for a LAN-only preview; worth revisiting before this is
        # ever exposed more widely.
        #
        # A None result is an idle tick: either no frame yet -- on demand, the
        # camera may still be starting -- or the source closed and dropped
        # what it had. Loop and re-check the client either way.
        item = await asyncio.to_thread(worker.wait_for_jpeg, seq, 5.0)
        if item is None:
            continue
        jpeg, seq = item
        yield (
            f"--{MJPEG_BOUNDARY}\r\n"
            f"Content-Type: image/jpeg\r\n"
            f"Content-Length: {len(jpeg)}\r\n\r\n"
        ).encode() + jpeg + b"\r\n"


def _register_routes(app: FastAPI) -> None:
    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        # Mirrors the conditional mount above: if the package was built
        # without its static assets, say so plainly rather than raising from
        # inside FileResponse on every request to the root.
        page = STATIC_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(status_code=404, detail="UI assets are not present in this build")
        return FileResponse(page)

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        # Liveness only: the process is answering. Device state belongs in
        # /readyz -- restarting the pod does not re-seat a PCIe card.
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        engine = _engine(request)
        if engine.ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "unavailable", "error": engine.error}, status_code=503)

    @app.get("/api/status")
    async def status(request: Request) -> dict[str, Any]:
        settings = _settings(request)
        worker = request.app.state.stream
        return {
            "engine": _engine(request).status(),
            "stream": worker.status() if worker else {"enabled": False},
            "config": {
                "score_threshold": settings.score_threshold,
                "max_detections": settings.max_detections,
                "hef_path": settings.hef_path,
            },
        }

    @app.post("/api/detect")
    async def detect(request: Request, image: UploadFile) -> dict[str, Any]:
        detections, result, source = await _detect_upload(request, image)
        return {
            "detections": [d.as_dict() for d in detections],
            "count": len(detections),
            "inference_ms": round(result.duration_ms, 2),
            "image": {"width": source.width, "height": source.height},
        }

    @app.post(
        "/api/detect/annotated",
        response_class=Response,
        responses={200: {"content": {"image/jpeg": {}}}},
    )
    async def detect_annotated(request: Request, image: UploadFile) -> Response:
        detections, result, source = await _detect_upload(request, image)
        annotated = draw_detections(source, detections)
        return Response(
            content=encode_jpeg(annotated, _settings(request).jpeg_quality),
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(detections)),
                "X-Inference-Ms": f"{result.duration_ms:.2f}",
            },
        )

    @app.get("/api/stream/snapshot.jpg", response_class=Response, include_in_schema=False)
    async def snapshot(request: Request) -> Response:
        # Deliberately does not register a viewer, so hitting this cannot open
        # the camera. It serves a frame only while something else is already
        # watching -- a snapshot endpoint that switched the camera on would be
        # the obvious way to defeat on-demand.
        jpeg, _ = _stream(request).latest_jpeg()
        if jpeg is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "no frame available -- the camera is only open while the "
                    "stream is being viewed"
                ),
            )
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/stream/mjpeg", include_in_schema=False)
    async def mjpeg(request: Request) -> StreamingResponse:
        worker = _stream(request)

        async def frames() -> AsyncIterator[bytes]:
            # Registered inside the generator, not around it: this is the only
            # place guaranteed to pair with the finally below, so a viewer
            # cannot be leaked and leave the camera open.
            worker.add_viewer()
            try:
                async for chunk in _mjpeg_frames(request, worker):
                    yield chunk
            finally:
                worker.remove_viewer()

        return StreamingResponse(
            frames(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
        )
