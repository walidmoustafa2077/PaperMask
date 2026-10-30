"""nexadoc-50k papermask dataset: preflight, canonical ordering, rejection-sampled augmentation."""
from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
MIN_AREA = 64.0


class PreflightError(RuntimeError):
    """The dataset on disk does not match the expected contract."""


def resolve_data_root(root: str) -> Path:
    p = Path(root)
    if (p / "papermask").is_dir():
        p = p / "papermask"
    return p


def _fail(msg: str) -> None:
    raise PreflightError(msg)


def preflight(root: str, splits=SPLITS) -> dict[str, int]:
    base = resolve_data_root(root)
    counts: dict[str, int] = {}
    for split in splits:
        d = base / split
        if not (d / "images").is_dir():
            _fail(f"missing split directories under {base}: ['{split}']")
        labels_path = d / "labels.json"
        if not labels_path.is_file():
            _fail(f"missing {labels_path}")
        try:
            labels = json.loads(labels_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            _fail(f"{labels_path} is not valid JSON: {exc}")
        if not isinstance(labels, dict) or any(not isinstance(v, list) for v in labels.values()):
            _fail(f"{labels_path}: not a filename->corners map (legacy [{{'value':..}}] shape?)")

        missing = [k for k in labels if not (d / "images" / k).is_file()]
        if missing:
            _fail(f"{split}: {len(missing)} labels have no image, e.g. {missing[:5]}")

        for name, quad in labels.items():
            if not (isinstance(quad, list) and len(quad) == 4):
                _fail(f"{split}/{name}: expected exactly 4 points, got {len(quad) if isinstance(quad, list) else type(quad)}")
            pts = np.asarray(quad, np.float32)
            if pts.shape != (4, 2):
                _fail(f"{split}/{name}: expected shape (4,2), got {pts.shape}")
            if pts.min() < -0.001 or pts.max() > 1.001:
                _fail(f"{split}/{name}: coordinates outside [0,1]")
            if len(np.unique(pts, axis=0)) != 4:
                _fail(f"{split}/{name}: duplicate corner points")
            if _area(order_corners(pts) * 256.0) < MIN_AREA:
                _fail(f"{split}/{name}: degenerate quad (area < {MIN_AREA} px^2)")
        counts[split] = len(labels)
    return counts


def _area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _cross2(u: np.ndarray, v: np.ndarray) -> float:
    return float(u[0] * v[1] - u[1] * v[0])


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Return TL, TR, BR, BL. Robust for rotations smaller than ~45 degrees."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    d = pts[:, 0] - pts[:, 1]
    return np.stack([pts[np.argmin(s)], pts[np.argmax(d)], pts[np.argmax(s)], pts[np.argmin(d)]])


def valid_quad(pts: np.ndarray, img_size: int) -> bool:
    pts = np.asarray(pts, np.float32)
    if pts.shape != (4, 2) or len(np.unique(pts, axis=0)) != 4:
        return False
    if pts.min() < 0 or pts.max() > img_size:
        return False
    if _area(pts) < MIN_AREA:
        return False
    # convex: all cross products share a sign
    cross = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append(_cross2(b - a, c - b))
    cross = np.array(cross)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def _train_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
            # std_range is a fraction of the pixel range: sqrt(10)/255 .. sqrt(30)/255,
            # i.e. the same noise the legacy var_limit=(10, 30) asked for.
            A.GaussNoise(std_range=(0.0124, 0.0215), p=0.2),
            A.Affine(rotate=(-8, 8), scale=(0.95, 1.05), shear=(-3, 3),
                     border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.Perspective(scale=(0.01, 0.03), p=0.3),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


class PaperCornerDataset(Dataset):
    def __init__(self, root: str, split: str, img_size: int = 256,
                 augment: bool = False, seed: int = 0, repeat: int = 1):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root = resolve_data_root(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.repeat = repeat
        self.images_dir = self.root / split / "images"
        self.labels = json.loads((self.root / split / "labels.json").read_text(encoding="utf-8"))
        self.names = sorted(self.labels)
        self.seed = seed
        self.tf = _train_transform(img_size) if augment else None
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.names) * self.repeat

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> dict:
        name = self.names[index % len(self.names)]
        path = self.images_dir / name
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"unreadable image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (self.img_size, self.img_size):
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        corners = np.asarray(self.labels[name], np.float32) * self.img_size

        if self.tf is not None:
            corners = self._augment(rgb, corners)
            if corners is None:                     # 5 bad draws: fall back to the clean sample
                corners = np.asarray(self.labels[name], np.float32) * self.img_size
            rgb = self._last_image

        corners = order_corners(corners)
        if not valid_quad(corners, self.img_size):
            raise ValueError(f"{self.split}/{name}: invalid quad after augmentation")

        x = rgb.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))),
            "corners": torch.from_numpy(corners),
            "name": name,
        }

    def _augment(self, rgb: np.ndarray, corners: np.ndarray):
        self._last_image = rgb
        for _ in range(5):
            out = self.tf(image=rgb, keypoints=corners.tolist())
            pts = np.asarray(out["keypoints"], np.float32)
            if pts.shape == (4, 2) and valid_quad(order_corners(pts), self.img_size):
                self._last_image = out["image"]
                return pts
        return None


def build_datasets(cfg: dict) -> tuple[PaperCornerDataset, PaperCornerDataset, PaperCornerDataset]:
    data = cfg["data"]
    common = dict(root=data["root"], img_size=data["img_size"])
    train = PaperCornerDataset(split="train", augment=data["augment"],
                               seed=cfg["seed"], repeat=1, **common)
    val = PaperCornerDataset(split="val", augment=False, **common)
    test = PaperCornerDataset(split="test", augment=False, **common)
    return train, val, test
