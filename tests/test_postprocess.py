"""The NMS decoder, exercised against both container shapes HailoRT uses."""

from __future__ import annotations

import numpy as np
import pytest

from hailo_detect.postprocess import NmsFormatError, decode_nms

NUM_CLASSES = 80


def dense_output(boxes: dict[int, list[list[float]]], max_det: int = 4) -> np.ndarray:
    """A (batch, classes, max_det, 5) array, zero-padded like the device's."""
    out = np.zeros((1, NUM_CLASSES, max_det, 5), dtype=np.float32)
    for class_id, rows in boxes.items():
        for slot, row in enumerate(rows):
            out[0, class_id, slot] = row
    return out


def ragged_output(boxes: dict[int, list[list[float]]]) -> list[list[np.ndarray]]:
    """A batch of per-class variable-length arrays."""
    per_class = [
        np.asarray(boxes.get(class_id, []), dtype=np.float32).reshape(-1, 5)
        for class_id in range(NUM_CLASSES)
    ]
    return [per_class]


# Rows are (y_min, x_min, y_max, x_max, score).
PERSON = [0.1, 0.2, 0.9, 0.6, 0.91]
DOG = [0.4, 0.5, 0.7, 0.8, 0.55]


@pytest.mark.parametrize("build", [dense_output, ragged_output], ids=["dense", "ragged"])
def test_decodes_either_container_identically(build):
    detections = decode_nms(
        build({0: [PERSON], 16: [DOG]}), score_threshold=0.3, max_detections=10
    )

    assert [(d.label, round(d.score, 2)) for d in detections] == [("person", 0.91), ("dog", 0.55)]
    person = detections[0]
    # x from x_min/x_max, y from y_min/y_max -- the row order is not xyxy.
    assert (person.x0, person.y0, person.x1, person.y1) == pytest.approx((0.2, 0.1, 0.6, 0.9))


def test_drops_low_scores_and_zero_padding():
    detections = decode_nms(
        dense_output({0: [PERSON], 16: [DOG]}), score_threshold=0.6, max_detections=10
    )

    # The dog is under threshold; the 78 all-zero padded slots must not decode
    # into detections at the origin.
    assert [d.label for d in detections] == ["person"]


def test_drops_boxes_degenerate_in_either_axis():
    # Zero width but a perfectly good height, and vice versa. Neither encloses
    # anything, and clamping would turn both into plausible-looking slivers.
    zero_width = [0.1, 0.5, 0.9, 0.5, 0.8]
    zero_height = [0.5, 0.1, 0.5, 0.9, 0.8]

    detections = decode_nms(
        dense_output({0: [zero_width], 16: [zero_height]}),
        score_threshold=0.1,
        max_detections=10,
    )

    assert detections == []


def test_orders_by_score_and_truncates():
    boxes = {i: [[0.1, 0.1, 0.5, 0.5, 0.5 + i / 100]] for i in range(6)}

    detections = decode_nms(dense_output(boxes), score_threshold=0.0, max_detections=3)

    scores = [d.score for d in detections]
    assert scores == sorted(scores, reverse=True)
    assert len(detections) == 3


def test_clamps_boxes_that_run_off_the_edge():
    detections = decode_nms(
        dense_output({0: [[-0.2, -0.1, 1.4, 1.2, 0.8]]}),
        score_threshold=0.1,
        max_detections=5,
    )

    box = detections[0]
    assert (box.x0, box.y0, box.x1, box.y1) == (0.0, 0.0, 1.0, 1.0)


def test_empty_output_is_no_detections():
    assert decode_nms(ragged_output({}), score_threshold=0.1, max_detections=5) == []
    assert decode_nms([], score_threshold=0.1, max_detections=5) == []


def test_unexpected_shape_names_what_it_saw():
    # A four-column output would silently lose the score column if we just
    # unpacked it, so it has to raise -- and say so with the real shape.
    with pytest.raises(NmsFormatError, match=r"ndarray\(1, 4\)"):
        decode_nms(
            [[np.zeros((1, 4), dtype=np.float32)]], score_threshold=0.1, max_detections=5
        )

    with pytest.raises(NmsFormatError, match="per-class sequence"):
        decode_nms("not an array", score_threshold=0.1, max_detections=5)
