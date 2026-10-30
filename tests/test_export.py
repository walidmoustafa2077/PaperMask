"""ONNX export + parity gate. Slow: exports a real graph and runs onnxruntime."""
from __future__ import annotations

import json

import pytest
import torch

from src.config import validate
from src.export import export
from src.models.paper_corner import build_model

pytestmark = pytest.mark.slow


def _config():
    return validate(
        {
            "seed": 0,
            "data": {"root": "unused", "img_size": 256, "num_workers": 0, "augment": False},
            "model": {
                "backbone": "mobilenet_v3_small",
                "pretrained": False,
                "fpn_channels": 16,
                "heatmap_size": 64,
                "sigma": 2.0,
            },
            "train": {
                "output_dir": "unused",
                "batch_size": 1,
                "epochs": 1,
                "lr": 1e-3,
                "weight_decay": 0.0,
                "warmup_epochs": 0,
                "grad_clip": 1.0,
                "amp": False,
                "early_stop_patience": 1,
                "save_best_only": True,
                "log_interval": 1,
            },
            "loss": {"heatmap_weight": 1.0, "offset_weight": 1.0},
        }
    )


def test_export_passes_parity_and_writes_metadata(tmp_path):
    cfg = _config()
    torch.manual_seed(0)
    model = build_model(cfg)
    # Perturb the zero-initialised offset head so parity actually exercises it.
    with torch.no_grad():
        for p in model.offset_head.parameters():
            p.add_(torch.randn_like(p) * 0.01)

    ckpt_path = tmp_path / "best.pth"
    torch.save({"arch_version": 2, "config": cfg, "state_dict": model.state_dict()}, ckpt_path)

    onnx_path = export(str(ckpt_path), str(tmp_path / "out"))

    assert onnx_path.exists()
    meta = json.loads((tmp_path / "out" / "model.json").read_text(encoding="utf-8"))
    assert meta["input"]["height"] == 256
    assert meta["output"]["stride"] == 4
    assert meta["output"]["heatmap_size"] == 64
    assert meta["parity"]["max_abs_diff"] <= meta["parity"]["tolerance"]
