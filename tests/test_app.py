"""Route behaviour, with the accelerator replaced by a stub.

Nothing here touches HailoRT: the point is that the HTTP layer maps a
detection onto JSON and JPEG correctly, and turns each engine failure into
the right status code.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hailo_detect import app as app_module
from hailo_detect.config import Settings
from hailo_detect.engine import EngineBusy, EngineNotReady, InferenceResult
from hailo_detect.postprocess import Detection

INPUT_SIZE = (64, 64)


class StubEngine:
    """Stands in for HailoEngine, returning one box in the middle of the frame."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ready = True
        self.error: str | None = None
        self.raises: Exception | None = None
        self.frames: list[np.ndarray] = []

    @property
    def input_size(self) -> tuple[int, int]:
        if not self.ready:
            raise EngineNotReady(self.error or "device not open")
        return INPUT_SIZE

    def infer(self, frame: np.ndarray) -> InferenceResult:
        if self.raises is not None:
            raise self.raises
        self.frames.append(frame)
        return InferenceResult(
            detections=[
                Detection(
                    class_id=0, label="person", score=0.87, x0=0.25, y0=0.25, x1=0.75, y1=0.75
                )
            ],
            duration_ms=12.5,
        )

    def start(self, timeout: float = 60.0) -> None: ...

    def close(self) -> None: ...

    def status(self) -> dict:
        return {"ready": self.ready, "error": self.error, "frames": len(self.frames)}


@pytest.fixture
def engines(monkeypatch) -> list[StubEngine]:
    """Swap the engine class for the stub, and collect what lifespan builds."""
    created: list[StubEngine] = []

    def factory(settings: Settings) -> StubEngine:
        stub = StubEngine(settings)
        created.append(stub)
        return stub

    monkeypatch.setattr(app_module, "HailoEngine", factory)
    return created


@pytest.fixture
def client(engines: list[StubEngine]):
    with TestClient(app_module.create_app(Settings())) as test_client:
        # Hung off the client so each test can reach the stub the app is
        # actually using without threading another fixture through.
        test_client.engine = engines[0]  # type: ignore[attr-defined]
        yield test_client


def jpeg_bytes(size: tuple[int, int] = (200, 100)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 140, 160)).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(client: TestClient, path: str = "/api/detect", data: bytes | None = None):
    payload = data if data is not None else jpeg_bytes()
    return client.post(path, files={"image": ("frame.jpg", payload, "image/jpeg")})


def test_healthz_is_independent_of_the_device(client):
    client.engine.ready = False

    assert client.get("/healthz").status_code == 200


def test_readyz_reports_the_device_error(client):
    client.engine.ready = False
    client.engine.error = "HailoRTStatusException: HAILO_INVALID_DRIVER_VERSION"

    response = client.get("/readyz")

    assert response.status_code == 503
    assert "HAILO_INVALID_DRIVER_VERSION" in response.json()["error"]


def test_detect_returns_boxes_in_source_coordinates(client):
    response = upload(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["image"] == {"width": 200, "height": 100}

    # The stub's box is the centre half of a *letterboxed* 64x64 frame. The
    # source is 2:1, so the content band is y in [16, 48) -- the box's y
    # therefore stretches to the full height of the source, while x is
    # unchanged. This is the whole reason map_to_source exists.
    box = payload["detections"][0]["box"]
    assert (box["x0"], box["x1"]) == pytest.approx((0.25, 0.75))
    assert (box["y0"], box["y1"]) == pytest.approx((0.0, 1.0))


def test_detect_feeds_the_engine_a_frame_at_the_model_input_size(client):
    upload(client)

    frame = client.engine.frames[0]
    assert frame.shape == (INPUT_SIZE[1], INPUT_SIZE[0], 3)
    assert frame.dtype.name == "uint8"


def test_annotated_returns_a_jpeg_of_the_original_size(client):
    response = upload(client, "/api/detect/annotated")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-detection-count"] == "1"
    assert Image.open(io.BytesIO(response.content)).size == (200, 100)


def test_oversized_upload_is_rejected_before_decoding(client):
    client.app.state.settings = Settings(max_upload_bytes=128)

    response = upload(client, data=jpeg_bytes((900, 900)))

    assert response.status_code == 413


def test_undecodable_upload_is_a_client_error(client):
    response = upload(client, data=b"this is not an image")

    assert response.status_code == 400
    assert "could not decode" in response.json()["detail"]


def test_saturated_accelerator_asks_the_caller_to_retry(client):
    client.engine.raises = EngineBusy("inference queue is full")

    response = upload(client)

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"


def test_inference_timeout_is_a_gateway_timeout(client):
    client.engine.raises = TimeoutError()

    assert upload(client).status_code == 504


def test_stream_endpoints_404_without_an_rtsp_url(client):
    assert client.get("/api/stream/snapshot.jpg").status_code == 404
    assert client.get("/api/status").json()["stream"] == {"enabled": False}
