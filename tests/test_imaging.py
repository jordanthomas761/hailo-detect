"""Letterboxing and the box mapping that undoes it."""

from __future__ import annotations

import pytest
from PIL import Image

from hailo_detect.imaging import (
    Letterbox,
    decode_image,
    draw_detections,
    encode_jpeg,
    letterbox,
    map_to_source,
    to_input_array,
)
from hailo_detect.postprocess import Detection

INPUT = (640, 640)


def box(x0: float, y0: float, x1: float, y1: float) -> Detection:
    return Detection(class_id=0, label="person", score=0.9, x0=x0, y0=y0, x1=x1, y1=y1)


def test_letterbox_pads_a_wide_image_top_and_bottom():
    canvas, fit = letterbox(Image.new("RGB", (200, 100)), INPUT)

    assert canvas.size == INPUT
    assert fit == Letterbox(
        src_w=200, src_h=100, dst_w=640, dst_h=640, scale=3.2, pad_x=0, pad_y=160
    )


def test_letterbox_pads_a_tall_image_left_and_right():
    _, fit = letterbox(Image.new("RGB", (100, 200)), INPUT)

    assert (fit.pad_x, fit.pad_y) == (160, 0)


def test_letterbox_leaves_a_square_image_unpadded():
    _, fit = letterbox(Image.new("RGB", (320, 320)), INPUT)

    assert (fit.pad_x, fit.pad_y, fit.scale) == (0, 0, 2.0)


def test_map_to_source_undoes_the_padding():
    _, fit = letterbox(Image.new("RGB", (200, 100)), INPUT)

    # The content occupies y in [160, 480) of the 640px canvas -- so this box,
    # which covers exactly that band, is the whole source image.
    mapped = map_to_source(box(0.0, 0.25, 1.0, 0.75), fit)

    assert (mapped.x0, mapped.y0, mapped.x1, mapped.y1) == pytest.approx((0.0, 0.0, 1.0, 1.0))


def test_map_to_source_clamps_boxes_that_reach_into_the_padding():
    _, fit = letterbox(Image.new("RGB", (200, 100)), INPUT)

    mapped = map_to_source(box(0.0, 0.0, 1.0, 1.0), fit)

    assert (mapped.y0, mapped.y1) == (0.0, 1.0)


def test_map_to_source_preserves_metadata():
    _, fit = letterbox(Image.new("RGB", (200, 100)), INPUT)

    mapped = map_to_source(box(0.1, 0.3, 0.4, 0.6), fit)

    assert (mapped.class_id, mapped.label, mapped.score) == (0, "person", 0.9)


def test_to_input_array_is_contiguous_hwc_uint8():
    canvas, _ = letterbox(Image.new("RGB", (200, 100)), INPUT)

    array = to_input_array(canvas)

    assert array.shape == (640, 640, 3)
    assert array.dtype.name == "uint8"
    assert array.flags["C_CONTIGUOUS"]


def test_draw_detections_does_not_resize_or_mutate():
    source = Image.new("RGB", (320, 240), (10, 20, 30))

    annotated = draw_detections(source, [box(0.1, 0.1, 0.5, 0.5)])

    assert annotated.size == source.size
    assert source.getpixel((100, 100)) == (10, 20, 30)


def test_draw_detections_keeps_a_caption_on_a_box_at_the_top_edge():
    # y0 = 0 leaves no room above the box; the caption has to move inside it
    # rather than being drawn off-canvas.
    annotated = draw_detections(Image.new("RGB", (320, 240)), [box(0.0, 0.0, 0.6, 0.6)])

    assert annotated.getpixel((5, 5)) != (0, 0, 0)


def test_jpeg_roundtrip():
    original = Image.new("RGB", (64, 48), (200, 30, 30))

    decoded = decode_image(encode_jpeg(original, quality=90))

    assert decoded.size == (64, 48)
    assert decoded.mode == "RGB"
