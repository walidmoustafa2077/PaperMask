"""Export a trained checkpoint to ONNX (heatmaps + offsets) with parity verification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.config import ConfigError, validate
from src.models.paper_corner import build_model

ARCH_VERSION = 2
OPSET = 17
PARITY_TOL = 1e-4
CORNER_ORDER = ["top_left", "top_right", "bottom_right", "bottom_left"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class _ExportWrapper(torch.nn.Module):
    """ONNX needs plain positional outputs, not the model's dict."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor):
        out = self.model(image)
        return out["heatmaps"], out["offsets"]


def export(checkpoint: str, output_dir: str) -> Path:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("arch_version") != ARCH_VERSION:
        raise SystemExit(f"checkpoint arch_version {ckpt.get('arch_version')} != {ARCH_VERSION}")

    cfg = validate(ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img_size = cfg["data"]["img_size"]
    hm_size = cfg["model"]["heatmap_size"]
    stride = img_size // hm_size

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "paper_corner.onnx"

    wrapper = _ExportWrapper(model)
    wrapper.eval()
    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        wrapper,
        (dummy,),
        str(onnx_path),
        opset_version=OPSET,
        input_names=["image"],
        output_names=["heatmaps", "offsets"],
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}, "offsets": {0: "batch"}},
        do_constant_folding=True,
    )

    # Parity gate: a numerically wrong export is worse than no export at all.
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # torch.onnx.export can leave the module in training mode; BatchNorm-in-train
    # would make the reference output differ from the exported (eval) graph.
    wrapper.eval()
    with torch.no_grad():
        ref_hm, ref_off = wrapper(dummy)
    got = session.run(None, {"image": dummy.numpy()})
    diffs = [float(np.abs(a.detach().numpy() - b).max()) for a, b in zip((ref_hm, ref_off), got)]

    if max(diffs) > PARITY_TOL:
        onnx_path.unlink(missing_ok=True)
        raise SystemExit(f"ONNX parity failed: max abs diff {max(diffs):.2e} > {PARITY_TOL}")

    meta = {
        "arch_version": ARCH_VERSION,
        "created_from": str(checkpoint),
        "corner_order": CORNER_ORDER,
        "confidence_threshold": 0.3,
        "input": {
            "name": "image",
            "height": img_size,
            "width": img_size,
            "layout": "nchw",
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
        "output": {
            "heatmaps": "heatmaps",
            "offsets": "offsets",
            "stride": stride,
            "heatmap_size": hm_size,
        },
        "parity": {"max_abs_diff": max(diffs), "tolerance": PARITY_TOL, "opset": OPSET},
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"wrote {onnx_path} ({size_mb:.2f} MB), parity {max(diffs):.2e}")
    print(f"wrote {out_dir / 'model.json'}")
    return onnx_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Export PaperMask v2 to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="exports")
    args = parser.parse_args(argv)
    try:
        export(args.checkpoint, args.output_dir)
    except ConfigError as exc:
        print(f"ConfigError: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
