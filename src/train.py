"""Train the v2 corner model: MobileNetV3-Small + Lite-FPN, heatmap + offset head.

Usage:
    python -m src.train --config configs/paper_base.yaml
    python -m src.train --config configs/paper_base.yaml --preflight
    python -m src.train --config configs/paper_base.yaml --resume outputs/paper_base/last.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        # Each worker decoding/augmenting images must not spawn its own thread
        # pool: with 4 workers on a 4-vCPU node that oversubscribes the CPU and
        # starves the GPU (observed: 0.6 it/s at batch 64 on Kaggle T4).
        cv2.setNumThreads(0)
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
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
        prefetch_factor=4 if num_workers > 0 else None,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, cfg, device,
                    epoch: int, writer=None, max_steps: int = 0) -> dict[str, float]:
    model.train()
    tcfg = cfg["train"]
    img_size = cfg["data"]["img_size"]
    hm_size = cfg["model"]["heatmap_size"]

    running = {"total": 0.0, "heatmap": 0.0, "offset": 0.0}
    metrics = CornerMetrics(img_size)
    n_batches = 0
    skipped = 0
    t_data = 0.0
    t_compute = 0.0
    last = time.time()
    bar = tqdm(loader, desc=f"train {epoch + 1}", leave=False, dynamic_ncols=True,
               bar_format="{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for step, batch in enumerate(bar):
        t_data += time.time() - last
        images = batch["image"].to(device, non_blocking=True)
        corners = batch["corners"].to(device, non_blocking=True)
        step_t0 = time.time()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=scaler is not None):
            out = model(images)
            losses = loss_fn(out["logits"], out["offsets"], corners, img_size, hm_size)
            total = losses["total"]

        if not torch.isfinite(total):
            # Never step on a non-finite loss: one NaN batch poisons every weight and the
            # whole epoch's reported average.
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            last = time.time()
            continue

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

        t_compute += time.time() - step_t0
        last = time.time()

        n_batches += 1
        for key in running:
            running[key] += float(losses[key].item())

        with torch.no_grad():
            pred, _ = decode_corners(out["heatmaps"].detach(), out["offsets"].detach(),
                                     img_size)
        metrics.update(pred.cpu(), corners.cpu())

        if step % tcfg["log_interval"] == 0:
            lr = optimizer.param_groups[0]["lr"]
            bar.set_postfix(loss=f"{float(total.item()):.4f}", lr=f"{lr:.1e}",
                            data=f"{t_data / n_batches:.3f}s",
                            step=f"{t_compute / n_batches:.3f}s")

        if max_steps and n_batches >= max_steps:
            break

    bar.close()
    bar.clear()
    stats = {k: v / max(n_batches, 1) for k, v in running.items()}
    stats.update(metrics.compute())
    stats["data_s"] = t_data / max(n_batches, 1)
    stats["step_s"] = t_compute / max(n_batches, 1)
    stats["batches"] = float(n_batches)
    stats["skipped"] = float(skipped)
    if skipped:
        print(f"[!] skipped {skipped} batch(es) with non-finite loss", flush=True)
    return stats


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
    parser.add_argument("--no-debug", action="store_false", dest="debug",
                        help="do not write per-epoch prediction overlays")
    parser.add_argument("--epochs", type=int, default=0,
                        help="override train.epochs (smoke runs)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop each epoch after N batches (speed profiling)")
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"[OK] GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("[!] Running on CPU (training will be slow)")
    print(f"[OK] Device: {device}")
    print(f"[OK] Config: {args.config}")
    print(f"[OK] Debug: {args.debug}")

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    (train_ds, train_loader, val_loader, model, loss_fn,
     optimizer, scheduler, scaler) = build_everything(cfg, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[OK] Params: {n_params:,} | train batches/epoch: {len(train_loader)} "
          f"| val batches: {len(val_loader)}")

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
    epochs = args.epochs if args.epochs > 0 else tcfg["epochs"]
    patience = 0
    history: list[dict] = []

    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        t_epoch = time.time()
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                      cfg, device, epoch, writer, args.max_steps)
        scheduler.step()

        val_stats, visuals = validate(model, val_loader, loss_fn, cfg, device)
        epoch_s = time.time() - t_epoch
        it_s = train_stats["batches"] / max(epoch_s, 1e-6)

        print(f"[Train] Epoch {epoch + 1:3d}/{epochs} | "
              f"Loss: {train_stats['total']:.4f} "
              f"(hm={train_stats['heatmap']:.4f}, off={train_stats['offset']:.4f}) | "
              f"MAE: {train_stats['mae_px']:.2f}px | nMAE: {train_stats['nmae']:.4f} | "
              f"PCK1: {train_stats['pck1']:.3f} PCK5: {train_stats['pck5']:.3f} | "
              f"IoU: {train_stats['iou']:.4f} | "
              f"lr: {optimizer.param_groups[0]['lr']:.2e} | {it_s:.2f} it/s | "
              f"data: {train_stats['data_s'] * 1000:.0f}ms step: "
              f"{train_stats['step_s'] * 1000:.0f}ms | {epoch_s:.0f}s")
        improved = val_stats["mae_px"] < best_mae
        status = "[+]" if improved else "[-]"
        print(f"{status} Epoch {epoch + 1:3d}/{epochs} | "
              f"Loss: {val_stats['loss']:.4f} | "
              f"MAE: {val_stats['mae_px']:.2f}px | nMAE: {val_stats['nmae']:.4f} | "
              f"PCK1: {val_stats['pck1']:.3f} PCK2: {val_stats['pck2']:.3f} "
              f"PCK5: {val_stats['pck5']:.3f} | IoU: {val_stats['iou']:.4f} | "
              f"invalid: {val_stats['invalid_frac']:.3f}", flush=True)

        history.append({
            "epoch": epoch + 1,
            "epoch_s": round(epoch_s, 1),
            "it_s": round(it_s, 3),
            **{f"train_{k}": v for k, v in train_stats.items()
               if isinstance(v, (int, float))},
            **{f"val_{k}": v for k, v in val_stats.items()
               if isinstance(v, (int, float))},
            "lr": optimizer.param_groups[0]["lr"],
        })
        (out_dir / "history.json").write_text(json.dumps(history, indent=2),
                                              encoding="utf-8")

        if writer is not None:
            for key, value in train_stats.items():
                if isinstance(value, (int, float)):
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

        if args.debug and visuals:
            save_visuals(visuals, out_dir / "visualizations", epoch)

        if improved:
            best_mae = val_stats["mae_px"]
            patience = 0
            torch.save(ckpt, out_dir / "best.pth")
            print(f"        new best MAE {best_mae:.2f}px -> best.pth")
        else:
            patience += 1
            if tcfg["early_stop_patience"] > 0 and patience >= tcfg["early_stop_patience"]:
                print(f"early stopping after {patience} epochs without improvement")
                break

    if history:
        keys = list(history[0].keys())
        with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            writer_csv = csv.DictWriter(fh, fieldnames=keys)
            writer_csv.writeheader()
            writer_csv.writerows(history)

    (out_dir / "metrics.json").write_text(
        json.dumps({"best_mae_px": best_mae, "epochs": len(history)}, indent=2),
        encoding="utf-8")
    if writer is not None:
        writer.close()
    print(f"[OK] Training complete. Best val MAE {best_mae:.2f}px "
          f"-> {out_dir / 'best.pth'}")
    print(f"[OK] TensorBoard logs: {out_dir / 'logs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
