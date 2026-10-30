"""Corner metrics in native image pixels (no rescaling to 256-space lies)."""
from __future__ import annotations

import math

import numpy as np
import torch
from shapely.geometry import Polygon

from src.utils.corners import quad_props


class CornerMetrics:
    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.reset()

    def reset(self) -> None:
        self._pred: list[np.ndarray] = []
        self._gt: list[np.ndarray] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        self._pred.append(pred.detach().float().cpu().numpy())
        self._gt.append(gt.detach().float().cpu().numpy())

    def compute(self) -> dict:
        keys = ("mae_px", "rmse_px", "nmae", "pck1", "pck2", "pck5", "iou", "invalid_frac")
        if not self._pred:
            return {k: 0.0 for k in keys} | {"per_corner_mae": [0.0] * 4}
        pred = np.concatenate(self._pred)
        gt = np.concatenate(self._gt)
        delta = pred - gt                                # (N,4,2)
        abs_err = np.abs(delta)                          # per coordinate, in pixels
        err = np.linalg.norm(delta, axis=-1)             # (N,4) euclidean, for PCK
        diag = math.hypot(self.img_size, self.img_size)

        ious, invalid = [], 0
        for p, g in zip(pred, gt):
            area, convex, _ = quad_props(torch.from_numpy(p))
            if not convex or area < 64:
                invalid += 1
                ious.append(0.0)
                continue
            try:
                a, b = Polygon(p).buffer(0), Polygon(g).buffer(0)
                ious.append(a.intersection(b).area / a.union(b).area)
            except Exception:
                ious.append(0.0)

        return {
            "mae_px": float(abs_err.mean()),
            "rmse_px": float(np.sqrt((delta ** 2).mean())),
            "nmae": float(abs_err.mean() / self.img_size),
            "pck1": float((err <= 0.01 * diag).mean()),
            "pck2": float((err <= 0.02 * diag).mean()),
            "pck5": float((err <= 0.05 * diag).mean()),
            "iou": float(np.mean(ious)),
            "invalid_frac": invalid / len(pred),
            "per_corner_mae": [float(v) for v in abs_err.mean(axis=(0, 2))],
        }
