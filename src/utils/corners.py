"""Decoding and quad geometry shared by training, evaluation, export checks."""
from __future__ import annotations

import torch


@torch.no_grad()
def decode_corners(heatmaps: torch.Tensor, offsets: torch.Tensor,
                   img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, h, w = heatmaps.shape
    stride = img_size / h
    flat = heatmaps.view(b, n, -1)
    scores, idx = flat.max(dim=-1)
    iy, ix = idx // w, idx % w

    bi = torch.arange(b, device=heatmaps.device).view(b, 1)
    ci = torch.arange(n, device=heatmaps.device).view(1, n)
    ox = offsets[bi, 2 * ci, iy, ix]
    oy = offsets[bi, 2 * ci + 1, iy, ix]

    x = (ix.to(heatmaps.dtype) + ox) * stride
    y = (iy.to(heatmaps.dtype) + oy) * stride
    return torch.stack([x, y], dim=-1), scores


@torch.no_grad()
def quad_props(pts: torch.Tensor) -> tuple[float, bool, bool]:
    """pts: (4,2) float tensor. Returns (area, is_convex, is_clockwise)."""
    p = pts.detach().to(torch.float64).cpu()
    x, y = p[:, 0], p[:, 1]
    area = float(abs((x * y.roll(-1) - y * x.roll(-1)).sum()) / 2)
    cross = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross.append(float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
    convex = all(v > 0 for v in cross) or all(v < 0 for v in cross)
    return area, convex, area > 0 and all(v > 0 for v in cross)


def winding(pts: torch.Tensor) -> int:
    """+1 clockwise in image coordinates (y down), -1 counter-clockwise."""
    p = pts.detach().to(torch.float64).cpu()
    x, y = p[:, 0], p[:, 1]
    signed = float((x * y.roll(-1) - y * x.roll(-1)).sum())
    return 1 if signed > 0 else -1
