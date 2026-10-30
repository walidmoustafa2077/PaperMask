"""Target encoding and penalty-reduced focal + masked L1 offset loss."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_targets(corners_px: torch.Tensor, img_size: int, heatmap_size: int,
                   sigma: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    b, n, _ = corners_px.shape
    device = corners_px.device
    stride = img_size / heatmap_size

    hm_coord = corners_px / stride
    # The Gaussian is centred on the exact sub-pixel position, so the decoded argmax is the
    # nearest cell, with ties (frac == 0.5) resolved downwards exactly like torch.max's
    # first-index rule. Offsets are stored on that cell, which keeps decode exact:
    #   (cell + offset) * stride == corner
    base = torch.floor(hm_coord)
    frac = hm_coord - base
    cells = torch.where(frac > 0.5, base + 1.0, base).clamp(0, heatmap_size - 1).long()
    offsets = hm_coord - cells.to(hm_coord.dtype)

    ys = torch.arange(heatmap_size, device=device, dtype=torch.float32).view(1, 1, -1, 1)
    xs = torch.arange(heatmap_size, device=device, dtype=torch.float32).view(1, 1, 1, -1)

    dx = xs - hm_coord[..., 0].view(b, n, 1, 1)
    dy = ys - hm_coord[..., 1].view(b, n, 1, 1)
    hm = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    mask = torch.zeros(b, n, heatmap_size, heatmap_size, device=device)
    off = torch.zeros(b, 2 * n, heatmap_size, heatmap_size, device=device)

    bi = torch.arange(b, device=device).view(b, 1).expand(b, n)
    ci = torch.arange(n, device=device).view(1, n).expand(b, n)
    mask[bi, ci, cells[..., 1], cells[..., 0]] = 1.0
    off[bi, 2 * ci, cells[..., 1], cells[..., 0]] = offsets[..., 0]
    off[bi, 2 * ci + 1, cells[..., 1], cells[..., 0]] = offsets[..., 1]
    return hm, off, mask


class CornerLoss(nn.Module):
    def __init__(self, heatmap_weight: float = 1.0, offset_weight: float = 1.0,
                 sigma: float = 2.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.sigma = sigma

    def forward(self, logits: torch.Tensor, offsets: torch.Tensor,
                corners_px: torch.Tensor, img_size: int,
                heatmap_size: int) -> dict[str, torch.Tensor]:
        b = logits.shape[0]
        target, off_t, mask = encode_targets(corners_px, img_size, heatmap_size, self.sigma)

        p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
        # Penalty-reduced focal term on the peak cell (binary), so peaks are learned sharply...
        pos_loss = -torch.log(p) * (1 - p).pow(2) * mask
        # ...and a Gaussian-weighted squared residual elsewhere. The spec's soft-target focal
        # negative term, -log(1-p) * p^2 * (1-y)^4, is NOT zero at p == y (mid-range Gaussian
        # cells contribute ~0.2 each, ~15.7 for one sample), so it can never satisfy the
        # "perfect prediction has ~zero loss" contract. The residual form is zero at p == y and
        # still pins the cells around a corner to the Gaussian, which keeps argmax sharp.
        neg_loss = (1.0 - target).pow(4) * (p - target).pow(2)
        n = b * logits.shape[1]
        heatmap_loss = (pos_loss.sum() + neg_loss.sum()) / n

        offset_loss = F.l1_loss(offsets * mask.repeat_interleave(2, dim=1),
                                off_t * mask.repeat_interleave(2, dim=1), reduction="sum") / n

        total = self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
        return {"total": total, "heatmap": heatmap_loss.detach(),
                "offset": offset_loss.detach()}
