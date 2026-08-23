"""Image plumbing: fit to the network input, map boxes back, draw, encode."""

from __future__ import annotations

import colorsys
import io
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .postprocess import Detection

# Ultralytics' padding grey. Nothing depends on the exact value -- it just
# has to be a flat colour the model was not trained to find edges in.
_PAD = (114, 114, 114)


@dataclass(frozen=True)
class Letterbox:
    """How a source image was fitted into the (square) network input."""

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    pad_x: int
    pad_y: int


def letterbox(image: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, Letterbox]:
    """Resize preserving aspect ratio and pad to ``size`` (width, height)."""
    dst_w, dst_h = size
    src_w, src_h = image.size
    scale = min(dst_w / src_w, dst_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    pad_x = (dst_w - new_w) // 2
    pad_y = (dst_h - new_h) // 2

    canvas = Image.new("RGB", (dst_w, dst_h), _PAD)
    canvas.paste(image.resize((new_w, new_h), Image.BILINEAR), (pad_x, pad_y))
    return canvas, Letterbox(src_w, src_h, dst_w, dst_h, scale, pad_x, pad_y)


def to_input_array(image: Image.Image) -> np.ndarray:
    """A contiguous HWC uint8 RGB array, which is what the input vstream wants."""
    return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def map_to_source(detection: Detection, box: Letterbox) -> Detection:
    """Re-express a box given in letterbox space as one in source space.

    Both ends are normalised, so this undoes the pad and the scale in the
    letterboxed image's own pixel units before renormalising against the
    source. Boxes that ran into the padding clamp to the image edge.
    """

    def x(value: float) -> float:
        px = (value * box.dst_w - box.pad_x) / box.scale
        return min(max(px / box.src_w, 0.0), 1.0)

    def y(value: float) -> float:
        px = (value * box.dst_h - box.pad_y) / box.scale
        return min(max(px / box.src_h, 0.0), 1.0)

    return Detection(
        class_id=detection.class_id,
        label=detection.label,
        score=detection.score,
        x0=x(detection.x0),
        y0=y(detection.y0),
        x1=x(detection.x1),
        y1=y(detection.y1),
    )


@lru_cache(maxsize=1)
def _font(size: int = 15) -> ImageFont.ImageFont:
    # fonts-dejavu-core is installed in the image; the default bitmap font is
    # a legible fallback if this ever runs somewhere it is not.
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=128)
def _colour(class_id: int) -> tuple[int, int, int]:
    # Golden-ratio hue stepping: adjacent class ids land far apart on the
    # wheel, so a person next to a bicycle is never two shades of the same.
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    """Return a copy of ``image`` with normalised boxes and labels drawn on."""
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    font = _font()

    for det in detections:
        x0, y0 = det.x0 * width, det.y0 * height
        x1, y1 = det.x1 * width, det.y1 * height
        colour = _colour(det.class_id)
        draw.rectangle((x0, y0, x1, y1), outline=colour, width=3)

        caption = f"{det.label} {det.score:.2f}"
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        # Sit the caption above the box, or inside it when the box is at the
        # very top of the frame and there is no room above.
        label_y = y0 - text_h - 6
        if label_y < 0:
            label_y = y0
        draw.rectangle((x0, label_y, x0 + text_w + 8, label_y + text_h + 6), fill=colour)
        draw.text((x0 + 4, label_y + 3), caption, fill=(0, 0, 0), font=font)

    return out


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def decode_image(data: bytes) -> Image.Image:
    """Decode uploaded bytes, normalising orientation and colour space."""
    from PIL import ImageOps

    image = Image.open(io.BytesIO(data))
    image.load()
    # Phone photos carry rotation in EXIF; without this the boxes are correct
    # and the picture is sideways.
    return ImageOps.exif_transpose(image).convert("RGB")
