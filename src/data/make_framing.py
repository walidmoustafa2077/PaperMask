"""Materialise a framing-varied copy of a papermask dataset.

The shipped nexadoc-50k papermask split is square and uniform: page at 60-70 %
of the frame in every sample. Training on that teaches a model that only works
on that one framing. `reframe` in `src/utils/framing.py` breaks the pattern;
this script bakes it into a dataset on disk so the result can be inspected,
counted and reused.

    python -m src.data.make_framing --source data/nexadoc-50k/papermask \
        --dest data/papermask_frames

Splits keep their names and membership - never re-shuffle a papermask split.
Samples whose reframed corners leave the frame are retried and finally left
untouched, so no label is ever clipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from src.data.paper_dataset import SPLITS, order_corners, valid_quad
from src.utils.framing import reframe

RETRIES = 6


def _resolve(source: Path) -> Path:
    return source / "papermask" if (source / "papermask").is_dir() else source


def build_split(src_split: Path, dst_split: Path, seed: int, min_scale: float,
                max_scale: float, jpeg_quality: int = 95) -> dict[str, int]:
    labels = json.loads((src_split / "labels.json").read_text(encoding="utf-8"))
    (dst_split / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out_labels: dict[str, list] = {}
    stats = {"total": 0, "reframed": 0, "kept": 0, "unreadable": 0}

    for name in sorted(labels):
        img = cv2.imread(str(src_split / "images" / name), cv2.IMREAD_COLOR)
        stats["total"] += 1
        if img is None:
            stats["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        corners = np.asarray(labels[name], np.float32) * np.array([w, h], np.float32)

        best = None
        for _ in range(RETRIES):
            out, pts = reframe(img, corners, w, rng=rng,
                               min_scale=min_scale, max_scale=max_scale)
            if valid_quad(order_corners(pts), w):
                best = (out, pts)
                break
        if best is None:                        # keep the original rather than a bad label
            best = (img, corners)
            stats["kept"] += 1
        else:
            stats["reframed"] += 1

        out, pts = best
        cv2.imwrite(str(dst_split / "images" / name), out,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        out_labels[name] = [[round(float(x) / w, 6), round(float(y) / h, 6)]
                            for x, y in order_corners(pts)]

    (dst_split / "labels.json").write_text(json.dumps(out_labels, indent=0), encoding="utf-8")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, help="dataset root (with papermask/ or split dirs)")
    ap.add_argument("--dest", required=True, help="output root; gets train/val/test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-scale", type=float, default=0.55)
    ap.add_argument("--max-scale", type=float, default=1.35)
    ap.add_argument("--only", nargs="*", default=list(SPLITS), choices=list(SPLITS))
    args = ap.parse_args(argv)

    src = _resolve(Path(args.source))
    dst = Path(args.dest)
    if not (src / "train").is_dir():
        print(f"no train/ under {src}", file=sys.stderr)
        return 2

    for split in args.only:
        if not (src / split / "labels.json").exists():
            print(f"skip {split}: no labels.json")
            continue
        stats = build_split(src / split, dst / split, args.seed, args.min_scale,
                            args.max_scale)
        print(f"{split}: {stats['reframed']}/{stats['total']} reframed, "
              f"{stats['kept']} kept as-is, {stats['unreadable']} unreadable -> {dst / split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
