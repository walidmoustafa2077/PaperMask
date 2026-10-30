"""MobileNetV3-Small -> Lite-FPN -> heatmap + sub-pixel offset head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from src.models.litefpn import DepthwiseSeparableConv, LiteFPN

BACKBONES = {
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, [1, 3, 6, 9]),
}


class PaperCornerModel(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_small", pretrained: bool = False,
                 fpn_channels: int = 64, heatmap_size: int = 64, num_corners: int = 4):
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unsupported backbone: {backbone}")
        factory, indices = BACKBONES[backbone]
        weights = "DEFAULT" if pretrained else None
        features = list(factory(weights=weights).features.children())

        self.stages = nn.ModuleList()
        prev = 0
        for idx in indices:
            self.stages.append(nn.Sequential(*features[prev:idx + 1]))
            prev = idx + 1

        with torch.no_grad():
            probe = torch.zeros(1, 3, 256, 256)
            channels = []
            sizes = []
            for stage in self.stages:
                probe = stage(probe)
                channels.append(probe.shape[1])
                sizes.append(probe.shape[-1])
            spatial = sizes[0]          # the head reads the finest level, not the coarsest

        self.neck = LiteFPN(channels, fpn_channels)
        self.stem = DepthwiseSeparableConv(fpn_channels, fpn_channels)
        self.heatmap_head = nn.Conv2d(fpn_channels, num_corners, 1)
        self.offset_head = nn.Conv2d(fpn_channels, num_corners * 2, 1)

        nn.init.constant_(self.heatmap_head.bias, -4.6)
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)

        self.heatmap_size = heatmap_size
        self.num_corners = num_corners
        if spatial != heatmap_size:
            raise ValueError(f"neck output is {spatial}x{spatial}, expected {heatmap_size}x{heatmap_size}")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        f = self.neck(feats)[0]
        f = self.stem(f)
        logits = self.heatmap_head(f)
        return {
            "logits": logits,
            "heatmaps": torch.sigmoid(logits),
            "offsets": self.offset_head(f),
        }


def build_model(cfg: dict) -> PaperCornerModel:
    m = cfg["model"]
    return PaperCornerModel(
        backbone=m["backbone"],
        pretrained=m["pretrained"],
        fpn_channels=m["fpn_channels"],
        heatmap_size=m["heatmap_size"],
        num_corners=4,
    )
