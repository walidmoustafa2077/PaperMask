"""Evaluate a checkpoint on a split. Architecture comes from the checkpoint, data root from YAML."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ConfigError, load_config
from src.data.paper_dataset import PaperCornerDataset, resolve_data_root
from src.metrics.corners import CornerMetrics
from src.models.paper_corner import build_model
from src.utils.corners import decode_corners

ARCH_VERSION = 2


@torch.no_grad()
def evaluate(checkpoint: str, config: str, split: str = "test", batch_size: int = 64) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("arch_version") != ARCH_VERSION:
        raise SystemExit(f"checkpoint arch_version {ckpt.get('arch_version')} != {ARCH_VERSION}")

    cfg = ckpt["config"]
    data_cfg = load_config(config)["data"]
    img_size = cfg["data"]["img_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    ds = PaperCornerDataset(str(resolve_data_root(data_cfg["root"])), split,
                            img_size=img_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    metrics = CornerMetrics(img_size)
    for batch in loader:
        images = batch["image"].to(device)
        out = model(images)
        pred, _ = decode_corners(out["heatmaps"], out["offsets"], img_size)
        metrics.update(pred.cpu(), batch["corners"])

    result = metrics.compute()
    result["split"] = split
    result["checkpoint"] = str(checkpoint)
    result["epoch"] = int(ckpt.get("epoch", -1))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a PaperMask v2 checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default=None, help="write metrics.json here")
    args = parser.parse_args(argv)

    try:
        result = evaluate(args.checkpoint, args.config, args.split, args.batch_size)
    except ConfigError as exc:
        print(f"ConfigError: {exc}")
        return 2

    print(f"| metric | value |")
    print(f"| --- | --- |")
    for key in ("split", "mae_px", "rmse_px", "nmae", "pck1", "pck2", "pck5",
                "iou", "invalid_frac"):
        print(f"| {key} | {result[key]:.4f} |" if isinstance(result[key], float)
              else f"| {key} | {result[key]} |")
    print(f"| per_corner_mae | {[round(v, 3) for v in result['per_corner_mae']]} |")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
