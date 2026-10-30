"""Train the v2 corner model: MobileNetV3-Small + Lite-FPN, heatmap + offset head.

Usage:
    python -m src.train --config configs/paper_base.yaml
    python -m src.train --config configs/paper_base.yaml --preflight
    python -m src.train --config configs/paper_base.yaml --resume outputs/paper_base/last.pth
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import ConfigError, load_config
from src.data.paper_dataset import PreflightError, build_datasets, preflight
from src.losses.corner_loss import CornerLoss
from src.metrics.corners import CornerMetrics
from src.models.paper_corner import build_model
from src.utils.corners import decode_corners
from src.utils.seed import seed_everything

ARCH_VERSION = 2
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _worker_init(seed: int):
    def init(worker_id: int) -> None:
        seed_everything(seed + worker_id)
    return init


def _loader(ds, batch_size: int, num_workers: int, seed: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=_worker_init(seed),
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, cfg, device,
                    epoch: int, writer=None) -> dict[str, float]:
    model.train()
    tcfg = cfg["train"]
    img_size = cfg["data"]["img_size"]
    hm_size = cfg["model"]["heatmap_size"]

    running = {"total": 0.0, "heatmap": 0.0, "offset": 0.0}
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        corners = batch["corners"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            out = model(images)
            losses = loss_fn(out["logits"], out["offsets"], corners, img_size, hm_size)
            total = losses["total"]

        if scaler is not None:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()

        n_batches += 1
        for key in running:
            running[key] += float(losses[key].item())

        if step % tcfg["log_interval"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            print(f"  epoch {epoch} step {step}/{len(loader)} "
                  f"loss {total.item():.4f} lr {lr:.2e} {n_batches / elapsed:.2f} it/s",
                  flush=True)

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, loader, loss_fn, cfg, device, collect: int = 4):
    model.eval()
    img_size = cfg["data"]["img_size"]
    hm_size = cfg["model"]["heatmap_size"]
    metrics = CornerMetrics(img_size)
    total_loss = 0.0
    n_batches = 0
    visuals = []

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        corners = batch["corners"].to(device, non_blocking=True)

        out = model(images)
        losses = loss_fn(out["logits"], out["offsets"], corners, img_size, hm_size)
        total_loss += float(losses["total"].item())
        n_batches += 1

        pred, _ = decode_corners(out["heatmaps"], out["offsets"], img_size)
        metrics.update(pred.cpu(), corners.cpu())

        if len(visuals) < collect:
            for i in range(min(len(images), collect - len(visuals))):
                visuals.append((images[i].cpu(), corners[i].cpu(), pred[i].cpu()))

    result = metrics.compute()
    result["loss"] = total_loss / max(n_batches, 1)
    return result, visuals


def _to_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    arr = image_tensor.numpy().transpose(1, 2, 0)
    arr = (arr * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return cv2.cvtColor(np.clip(arr, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def save_visuals(visuals, out_dir: Path, epoch: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (image, gt, pred) in enumerate(visuals):
        canvas = _to_bgr(image)
        gt = gt.numpy()
        pred = pred.numpy()
        cv2.polylines(canvas, [gt.astype(np.int32)], True, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.polylines(canvas, [pred.astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
        for (x, y) in gt:
            cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)
        for (x, y) in pred:
            cv2.circle(canvas, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv2.imwrite(str(out_dir / f"epoch_{epoch:03d}_{i}.jpg"), canvas)


def _make_scheduler(optimizer, cfg):
    epochs = cfg["train"]["epochs"]
    warmup = max(int(cfg["train"]["warmup_epochs"]), 0)

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _summary_writer(out_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:                       # tensorboard is optional at runtime
        return None
    return SummaryWriter(str(out_dir / "logs"))


def build_everything(cfg: dict, device: torch.device):
    train_ds, val_ds, _ = build_datasets(cfg)
    tcfg = cfg["train"]
    train_loader = _loader(train_ds, tcfg["batch_size"], cfg["data"]["num_workers"],
                           cfg["seed"], shuffle=True)
    val_loader = _loader(val_ds, tcfg["batch_size"], cfg["data"]["num_workers"],
                         cfg["seed"], shuffle=False)

    model = build_model(cfg).to(device)
    loss_fn = CornerLoss(cfg["loss"]["heatmap_weight"], cfg["loss"]["offset_weight"],
                         cfg["model"]["sigma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                                  weight_decay=tcfg["weight_decay"])
    scheduler = _make_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda") if (tcfg["amp"] and device.type == "cuda") else None
    return train_ds, train_loader, val_loader, model, loss_fn, optimizer, scheduler, scaler


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the PaperMask v2 corner model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true",
                        help="validate the dataset and exit")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"ConfigError: {exc}")
        return 2

    try:
        counts = preflight(cfg["data"]["root"])
    except PreflightError as exc:
        print(f"PreflightError: {exc}")
        return 3
    print(f"dataset ok: {counts}")
    if args.preflight:
        return 0

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    (train_ds, train_loader, val_loader, model, loss_fn,
     optimizer, scheduler, scaler) = build_everything(cfg, device)

    start_epoch = 0
    best_mae = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if ckpt.get("arch_version") != ARCH_VERSION:
            print(f"checkpoint arch_version {ckpt.get('arch_version')} != {ARCH_VERSION}")
            return 4
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_mae = float(ckpt.get("metrics", {}).get("mae_px", float("inf")))
        print(f"resumed from {args.resume} at epoch {start_epoch} (best mae {best_mae:.3f})")

    writer = _summary_writer(out_dir)
    tcfg = cfg["train"]
    patience = 0

    for epoch in range(start_epoch, tcfg["epochs"]):
        train_ds.set_epoch(epoch)
        print(f"epoch {epoch + 1}/{tcfg['epochs']}")
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                      cfg, device, epoch, writer)
        scheduler.step()

        val_stats, visuals = validate(model, val_loader, loss_fn, cfg, device)
        print(f"  train {train_stats} | val loss {val_stats['loss']:.4f} "
              f"mae {val_stats['mae_px']:.2f}px pck1 {val_stats['pck1']:.3f} "
              f"iou {val_stats['iou']:.4f}")

        if writer is not None:
            for key, value in train_stats.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_stats.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"val/{key}", value, epoch)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        ckpt = {
            "arch_version": ARCH_VERSION,
            "config": cfg,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "metrics": val_stats,
        }
        torch.save(ckpt, out_dir / "last.pth")
        if epoch % 10 == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pth")

        if val_stats["mae_px"] < best_mae:
            best_mae = val_stats["mae_px"]
            patience = 0
            torch.save(ckpt, out_dir / "best.pth")
            save_visuals(visuals, out_dir / "visualizations", epoch)
            print(f"  new best mae {best_mae:.2f}px -> best.pth")
        else:
            patience += 1
            if tcfg["early_stop_patience"] > 0 and patience >= tcfg["early_stop_patience"]:
                print(f"early stopping after {patience} epochs without improvement")
                break

    (out_dir / "metrics.json").write_text(json.dumps({"best_mae_px": best_mae}, indent=2),
                                          encoding="utf-8")
    if writer is not None:
        writer.close()
    print(f"done. best val mae {best_mae:.2f}px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
