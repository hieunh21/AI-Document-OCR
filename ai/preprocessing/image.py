"""Image loading, validation and resizing with an exact way back to original pixels.

Every box that leaves the AI core must be in *original-image* coordinates
(after EXIF orientation). Any transform applied for OCR is recorded in a
:class:`Transform` and undone with :meth:`Transform.to_original`.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 60_000_000  # reject decompression bombs (~60 MP)

MIN_SIDE = 200  # px; smaller images cannot hold readable receipt text
MAX_ASPECT = 12.0  # very long receipts exist, but beyond this it is not a photo of one


class ImageError(ValueError):
    """The input cannot be used for OCR. ``code`` is machine readable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def load_image(source: str | Path | bytes) -> Image.Image:
    """Decode, apply EXIF orientation, convert to RGB and validate."""
    try:
        if isinstance(source, bytes):
            im = Image.open(io.BytesIO(source))
        else:
            im = Image.open(source)
        im.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ImageError("unreadable_image", f"Cannot decode image: {e}") from e
    im = ImageOps.exif_transpose(im).convert("RGB")
    validate_image(im)
    return im


def validate_image(im: Image.Image) -> None:
    w, h = im.size
    if min(w, h) < MIN_SIDE:
        raise ImageError(
            "image_too_small", f"Image is {w}x{h}px; need at least {MIN_SIDE}px per side"
        )
    if max(w, h) / min(w, h) > MAX_ASPECT:
        raise ImageError(
            "bad_aspect_ratio", f"Aspect ratio {max(w, h) / min(w, h):.1f} is too extreme"
        )


@dataclass(frozen=True)
class Transform:
    """Maps working-image coordinates back to the original image.

    Working image = original rotated counter-clockwise by ``rotation`` degrees
    (0/90/180/270), then scaled by ``scale``.
    """

    orig_w: int
    orig_h: int
    scale: float = 1.0
    rotation: int = 0

    def points_to_original(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2) / self.scale
        x, y = pts[:, 0], pts[:, 1]
        W, H = self.orig_w, self.orig_h
        if self.rotation == 0:
            ox, oy = x, y
        elif self.rotation == 90:  # CCW: original (x, y) -> (y, W - x)
            ox, oy = W - y, x
        elif self.rotation == 180:
            ox, oy = W - x, H - y
        elif self.rotation == 270:  # CCW 270 = CW 90: original (x, y) -> (H - y, x)
            ox, oy = y, H - x
        else:
            raise ValueError(f"unsupported rotation {self.rotation}")
        out = np.stack([np.clip(ox, 0, W), np.clip(oy, 0, H)], axis=1)
        return out

    def bbox_to_original(self, pts: np.ndarray) -> list[float]:
        p = self.points_to_original(pts)
        return [
            float(p[:, 0].min()),
            float(p[:, 1].min()),
            float(p[:, 0].max()),
            float(p[:, 1].max()),
        ]


def prepare_for_ocr(
    im: Image.Image, max_side: int | None = None, rotation: int = 0
) -> tuple[Image.Image, Transform]:
    """Rotate (CCW degrees) and downscale so the longer side is <= ``max_side``.

    Never upscales: enlarging adds no information and slows detection.
    """
    orig_w, orig_h = im.size
    if rotation:
        im = im.rotate(rotation, expand=True)
    scale = 1.0
    if max_side and max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return im, Transform(orig_w=orig_w, orig_h=orig_h, scale=scale, rotation=rotation)
