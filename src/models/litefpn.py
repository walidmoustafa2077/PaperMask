"""Lite-FPN: 1x1 laterals, nearest upsample, depthwise-separable smoothing."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class LiteFPN(nn.Module):
    """Four-level top-down FPN with depthwise smoothing and level residuals."""

    def __init__(self, in_channels: list[int], out_channels: int = 64):
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError("LiteFPN needs at least two input levels")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.lateral = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.SiLU(inplace=True))
            for c in in_channels
        )
        self.smooth = nn.ModuleList(
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in range(len(in_channels) - 1)
        )
        self.out_conv = nn.ModuleList(
            DepthwiseSeparableConv(out_channels, out_channels)
            for _ in in_channels
        )

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(feats) != len(self.in_channels):
            raise ValueError(f"expected {len(self.in_channels)} feature maps, got {len(feats)}")
        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]
        td = [None] * len(laterals)
        td[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=laterals[i].shape[-2:],
                               mode="nearest")
            td[i] = self.smooth[i](laterals[i] + up)
        return [self.out_conv[i](td[i]) + laterals[i] for i in range(len(td))]
