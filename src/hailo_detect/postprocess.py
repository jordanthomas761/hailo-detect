"""Turn HailoRT's NMS output into plain detections.

The detection HEFs in the hailo-models package are compiled with the NMS
postprocess on-chip, so there is no anchor decoding or IoU suppression to do
here -- the device hands back, per class, a variable-length array of
``[y_min, x_min, y_max, x_max, score]`` rows in **normalised** coordinates.

What varies between HailoRT versions is the *container* those arrays arrive
in: a batch-major list of per-class lists, or one dense
``(batch, classes, max_det, 5)`` array. Both are handled. Anything else raises
with the shape it actually saw, because the alternative -- returning an empty
list -- looks exactly like "the model found nothing" and would send you
hunting the camera instead of the parser.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from .labels import class_name

# HailoRT emits boxes as (y_min, x_min, y_max, x_max, score).
_BOX_COLUMNS = 5


@dataclass(frozen=True)
class Detection:
    """One box, normalised 0..1 against whatever image it was drawn from."""

    class_id: int
    label: str
    score: float
    x0: float
    y0: float
    x1: float
    y1: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "label": self.label,
            "score": round(self.score, 4),
            "box": {
                "x0": round(self.x0, 5),
                "y0": round(self.y0, 5),
                "x1": round(self.x1, 5),
                "y1": round(self.y1, 5),
            },
        }


class NmsFormatError(ValueError):
    """The device returned something this parser does not recognise."""


def _describe(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray{value.shape}"
    if isinstance(value, list | tuple):
        inner = _describe(value[0]) if value else "empty"
        return f"{type(value).__name__}[{len(value)}] of {inner}"
    return type(value).__name__


def _per_class_arrays(raw: Any) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(class_id, rows)`` for the first (only) item in the batch."""
    value = raw

    if isinstance(value, np.ndarray) and value.ndim == 4:
        value = value[0]

    if isinstance(value, list | tuple):
        if not value:
            return
        head = value[0]
        # A batch of per-class lists, or a batch of dense arrays: unwrap once.
        if isinstance(head, list | tuple) or (isinstance(head, np.ndarray) and head.ndim == 3):
            value = value[0]

    if isinstance(value, np.ndarray) and value.ndim == 3:
        value = list(value)

    if not isinstance(value, list | tuple):
        raise NmsFormatError(f"expected a per-class sequence of boxes, got {_describe(raw)}")

    for class_id, rows in enumerate(value):
        rows = np.asarray(rows, dtype=np.float32)
        if rows.size == 0:
            continue
        if rows.ndim != 2 or rows.shape[1] < _BOX_COLUMNS:
            raise NmsFormatError(
                f"class {class_id}: expected (n, {_BOX_COLUMNS}) boxes, "
                f"got {_describe(rows)} (full output {_describe(raw)})"
            )
        yield class_id, rows


def decode_nms(
    raw: Any,
    *,
    score_threshold: float,
    max_detections: int,
) -> list[Detection]:
    """Decode one frame's worth of NMS output, best-scoring detections first."""
    detections: list[Detection] = []

    for class_id, rows in _per_class_arrays(raw):
        for row in rows:
            y0, x0, y1, x1, score = (float(v) for v in row[:_BOX_COLUMNS])
            # A dense (classes, max_det, 5) output pads unused slots with
            # zeros, and those decode to a zero-area box at the origin. `or`,
            # not `and`: a box degenerate in one axis has no area either, and
            # clamping it would turn it into a plausible-looking sliver.
            if score < score_threshold or x1 <= x0 or y1 <= y0:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    label=class_name(class_id),
                    score=score,
                    x0=min(max(x0, 0.0), 1.0),
                    y0=min(max(y0, 0.0), 1.0),
                    x1=min(max(x1, 0.0), 1.0),
                    y1=min(max(y1, 0.0), 1.0),
                )
            )

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections[:max_detections]
