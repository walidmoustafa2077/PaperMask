"""Framing augmentation: teach the model that the page can be any size anywhere.

The synthetic set is extremely regular: every frame is square and the page
always covers 60-70 % of it. Real phone photos are 1:1 to 1:2.3 and the page
can be tiny in the middle or run off the edge. When the framing changes the
current model's answer swings by 30-90 px, so framing is a training problem,
not a preprocessing one.

`reframe` recomposes a square sample into another plausible framing by
scaling the scene (0.55x-1.35x), dropping it at a random position on a grey
canvas, and cropping where it overflows. Corners are transformed with the
same affine, so the labels stay exact. Callers reject samples whose corners
leave the frame - never clamp.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.preprocess import PAD_VALUE


def reframe(img: np.ndarray, corners: np.ndarray, size: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
            min_scale: float = 0.55, max_scale: float = 1.35,
            pad_value: int = PAD_VALUE) -> Tuple[np.ndarray, np.ndarray]:
    """Scale + place + crop a sample, returning ``(image, corners)``.

    ``corners`` are pixels in ``img``'s frame and come back in the new frame.
    ``min_scale < 1`` shrinks the scene (grey appears around it), ``max_scale > 1``
    zooms in (the scene is cropped). The caller must validate the result.
    """
    rng = np.random.default_rng() if rng is None else rng
    h, w = img.shape[:2]
    size = w if size is None else int(size)

    scale = float(rng.uniform(min_scale, max_scale))
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    scale_x = new_w / w
    scale_y = new_h / h
    scene = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.full((size, size, img.shape[2]), pad_value, dtype=img.dtype) \
        if img.ndim == 3 else np.full((size, size), pad_value, dtype=img.dtype)

    # Random placement. When the scene is larger than the frame the offset is
    # negative, which is how the overflow gets cropped.
    ox = int(round(rng.uniform(min(0, size - new_w), max(0, size - new_w))))
    oy = int(round(rng.uniform(min(0, size - new_h), max(0, size - new_h))))

    sx0, sy0 = max(0, ox), max(0, oy)
    sx1, sy1 = min(size, ox + new_w), min(size, oy + new_h)
    out[sy0:sy1, sx0:sx1] = scene[sy0 - oy:sy1 - oy, sx0 - ox:sx1 - ox]

    pts = np.asarray(corners, np.float32).reshape(-1, 2)
    pts = pts * np.array([scale_x, scale_y], np.float32) + np.array([ox, oy], np.float32)
    return out, pts
