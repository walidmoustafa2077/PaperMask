"""Letterbox preprocessing shared by training, evaluation and the tools.

The model was trained on square frames, but phone photos are anything from
1:1 to 1:2.3. Squashing them into a square stretches the page and moves the
corners to places the model has never seen; centre-cropping throws away up to
half the photo. Letterboxing keeps the geometry: the photo is scaled down
until it fits and the leftover strip is filled with a flat grey.

Every consumer must use these functions so the geometry cannot drift:
`PaperCornerDataset`, `refine_corners.py`, the framing materialiser and
(eventually) the .NET side.
"""

from typing import Tuple

import cv2
import numpy as np

# Flat grey, the same value OpenCV's resize border uses by convention.
PAD_VALUE = 114


def letterbox(img: np.ndarray, size: int, pad_value: int = PAD_VALUE
              ) -> Tuple[np.ndarray, float, float, float]:
    """Scale ``img`` to fit ``size`` x ``size`` and pad the rest.

    Returns ``(out, scale, pad_x, pad_y)`` where a pixel ``(x, y)`` in ``img``
    maps to ``(x * scale + pad_x, y * scale + pad_y)`` in ``out``.
    """
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError("empty image")
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # Recompute the scale from the rounded size so the mapping stays exact.
    scale_w = new_w / w
    scale_h = new_h / h
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype) \
        if img.ndim == 3 else np.full((size, size), pad_value, dtype=img.dtype)
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return out, (scale_w + scale_h) / 2.0, float(pad_x), float(pad_y)


def letterbox_points(pts: np.ndarray, scale: float, pad_x: float, pad_y: float) -> np.ndarray:
    """Map points from the source frame into the letterboxed frame."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    return pts * scale + np.array([pad_x, pad_y], np.float32)


def unletterbox_points(pts: np.ndarray, scale: float, pad_x: float, pad_y: float) -> np.ndarray:
    """Inverse of :func:`letterbox_points`: back to original pixel coordinates."""
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    return (pts - np.array([pad_x, pad_y], np.float32)) / scale


def is_square(img: np.ndarray) -> bool:
    h, w = img.shape[:2]
    return h == w
