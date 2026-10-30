# PaperMask v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two legacy corner models with one MobileNetV3-Small → Lite-FPN network that predicts heatmaps plus sub-pixel offsets, trained on `nexadoc-50k/papermask`, exported to a single ONNX file consumed by the .NET app.

**Architecture:** One forward pass produces `heatmaps (B,4,64,64)` and `offsets (B,8,64,64)`. Corners are decoded as `(argmax + offset) * 4` pixels, which removes the 4-px quantization of the current deployment. Training uses penalty-reduced focal loss on the heatmaps and masked L1 on the offsets. Everything is config-driven through a strict schema, and one `build_model()` serves train, export, evaluate and refine.

**Tech Stack:** Python 3.11, PyTorch 2.x + torchvision, albumentations, OpenCV, shapely, ONNX/onnxruntime, pytest; .NET 9 + Microsoft.ML.OnnxRuntime + OpenCvSharp4.

**Spec:** `docs/superpowers/specs/2026-10-28-papermask-litefpn-retrain-design.md`

## Global Constraints

- Input is always `256×256` RGB, ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`.
- Heatmap grid is `64×64`, stride `4`. Corner order is always TL, TR, BR, BL, normalized `[0,1]` in labels, pixels in model space.
- Coordinates are never clamped after a geometric augmentation; samples that leave the frame are re-drawn.
- Config is strict: unknown keys and missing keys both raise `ConfigError`.
- Checkpoints carry `arch_version: 2` and the full config; every consumer rebuilds the model from the checkpoint, never from a side file.
- ONNX opset 17, input name `image`, outputs `heatmaps` and `offsets`, a `model.json` sidecar ships next to it.
- No new dependency outside `torch, torchvision, albumentations, opencv-python, numpy, PyYAML, shapely, onnx, onnxruntime, tqdm, pytest`.
- Every task ends with its own tests passing and its own commit on branch `feat/litefpn-heatmap-offset`.

---

### Task 0: Toolchain

**Files:**
- Modify: `requirements.txt`
- Create: `requirements-dev.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: a working `pytest` invocation from the repo root; `from src.x import y` imports resolve.

- [ ] **Step 1: Create a working environment** (the committed `torch_env/` is broken)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
pip install numpy opencv-python albumentations pyyaml shapely tqdm onnx onnxruntime pytest
```

- [ ] **Step 2: Replace `requirements.txt`** with the runtime set actually used:

```
torch>=2.4
torchvision>=0.19
numpy>=1.26
opencv-python>=4.9
albumentations>=1.4
PyYAML>=6.0
shapely>=2.0
tqdm>=4.66
onnx>=1.16
onnxruntime>=1.18
```

- [ ] **Step 3: Add `requirements-dev.txt`**

```
-r requirements.txt
pytest>=8.0
matplotlib>=3.8
tensorboard>=2.14
```

- [ ] **Step 4: Add `pytest.ini`**

```ini
[pytest]
testpaths = tests
addopts = -q
markers =
    slow: needs onnx export (>5 s)
```

- [ ] **Step 5: Verify imports resolve**

Run: `python -c "import src, src.models; print('ok')"`
Expected: `ok` (create `src/__init__.py` if it is missing)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt requirements-dev.txt pytest.ini tests/__init__.py src/__init__.py
git commit -m "chore: python toolchain for v2 pipeline"
```

---

### Task 1: Strict config

**Files:**
- Create: `src/config.py`
- Create: `configs/paper_base.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_config(path: str | Path) -> dict`, `ConfigError`, and the exact config shape below. All later tasks read these keys.

- [ ] **Step 1: Write the failing test**

```python
import pytest
from src.config import ConfigError, load_config


def write(tmp_path, text):
    p = tmp_path / "c.yaml"
    p.write_text(text)
    return p


def test_loads_valid_config(tmp_path):
    cfg = load_config("configs/paper_base.yaml")
    assert cfg["model"]["heatmap_size"] == 64
    assert cfg["data"]["img_size"] == 256
    assert cfg["data"]["img_size"] % cfg["model"]["heatmap_size"] == 0


def test_unknown_key_raises(tmp_path):
    p = write(tmp_path, """
seed: 1
data: {root: x, img_size: 256, num_workers: 0, augment: false, typo: 1}
model: {backbone: mobilenet_v3_small, pretrained: false, fpn_channels: 64, heatmap_size: 64, sigma: 2.0}
train: {output_dir: out, batch_size: 2, epochs: 1, lr: 0.001, weight_decay: 0.0,
        warmup_epochs: 0, grad_clip: 1.0, amp: false, early_stop_patience: 1,
        save_best_only: true, log_interval: 1}
loss: {heatmap_weight: 1.0, offset_weight: 1.0}
""")
    with pytest.raises(ConfigError, match="unknown key: data.typo"):
        load_config(p)


def test_missing_key_raises(tmp_path):
    p = write(tmp_path, "seed: 1\ndata: {root: x}\n")
    with pytest.raises(ConfigError, match="missing key: data.img_size"):
        load_config(p)


def test_bool_is_not_int(tmp_path):
    cfg = load_config("configs/paper_base.yaml")
    cfg["train"]["batch_size"] = True
    with pytest.raises(ConfigError, match="train.batch_size"):
        from src.config import validate
        validate(cfg)
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: src.config`

- [ ] **Step 3: Implement `src/config.py`**

```python
"""Strict configuration loading: unknown keys are errors, not silent no-ops."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file does not match the schema."""


SCHEMA: dict[str, Any] = {
    "seed": int,
    "data": {
        "root": str,
        "img_size": int,
        "num_workers": int,
        "augment": bool,
    },
    "model": {
        "backbone": str,
        "pretrained": bool,
        "fpn_channels": int,
        "heatmap_size": int,
        "sigma": float,
    },
    "train": {
        "output_dir": str,
        "batch_size": int,
        "epochs": int,
        "lr": float,
        "weight_decay": float,
        "warmup_epochs": int,
        "grad_clip": float,
        "amp": bool,
        "early_stop_patience": int,
        "save_best_only": bool,
        "log_interval": int,
    },
    "loss": {
        "heatmap_weight": float,
        "offset_weight": float,
    },
}


def _check(node: dict, schema: dict, prefix: str = "") -> None:
    for key in node:
        if key not in schema:
            raise ConfigError(f"unknown key: {prefix}{key}")
    for key, expected in schema.items():
        path = f"{prefix}{key}"
        if key not in node:
            raise ConfigError(f"missing key: {path}")
        value = node[key]
        if isinstance(expected, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
            _check(value, expected, f"{path}.")
        else:
            if expected is int and isinstance(value, bool):
                raise ConfigError(f"{path} must be int, got bool")
            if not isinstance(value, expected):
                raise ConfigError(
                    f"{path} must be {expected.__name__}, got {type(value).__name__}"
                )


def validate(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ConfigError("config root must be a mapping")
    _check(cfg, SCHEMA)
    img_size = cfg["data"]["img_size"]
    hm = cfg["model"]["heatmap_size"]
    if img_size % hm:
        raise ConfigError(f"data.img_size ({img_size}) must be divisible by model.heatmap_size ({hm})")
    if cfg["model"]["sigma"] <= 0:
        raise ConfigError("model.sigma must be > 0")
    return cfg


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return validate(cfg)
```

- [ ] **Step 4: Create `configs/paper_base.yaml`**

```yaml
seed: 42

data:
  root: data/nexadoc-50k        # <root>/papermask/{train,val,test}/...
  img_size: 256
  num_workers: 4
  augment: true

model:
  backbone: mobilenet_v3_small
  pretrained: true
  fpn_channels: 64
  heatmap_size: 64
  sigma: 2.0

train:
  output_dir: outputs/paper_base
  batch_size: 32
  epochs: 100
  lr: 3.0e-4
  weight_decay: 2.0e-4
  warmup_epochs: 3
  grad_clip: 1.0
  amp: true
  early_stop_patience: 15
  save_best_only: true
  log_interval: 50

loss:
  heatmap_weight: 1.0
  offset_weight: 1.0
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add src/config.py configs/paper_base.yaml tests/test_config.py
git commit -m "feat(config): strict schema replacing silent config knobs"
```

---

### Task 2: Dataset + preflight

**Files:**
- Create: `src/data/paper_dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: config dict from Task 1.
- Produces:
  - `class PreflightError(RuntimeError)`
  - `resolve_data_root(root: str) -> Path`
  - `preflight(root: str, splits=("train","val","test")) -> dict[str,int]`
  - `order_corners(pts: np.ndarray) -> np.ndarray` — (4,2) → TL,TR,BR,BL
  - `valid_quad(pts, img_size) -> bool`
  - `PaperCornerDataset(root, split, img_size=256, augment=False, seed=0, repeat=1)`, `__getitem__ -> {"image": (3,H,W) float32 tensor, "corners": (4,2) float32 tensor in pixels, "name": str}`

- [ ] **Step 1: Write the failing test**

```python
import json
import numpy as np
import pytest

from src.data.paper_dataset import (
    PaperCornerDataset, PreflightError, order_corners, preflight, resolve_data_root,
)


def make_split(root, split, n=3, legacy=False):
    d = root / "papermask" / split
    (d / "images").mkdir(parents=True)
    labels = {}
    for i in range(n):
        name = f"{i:05d}.jpg"
        img = np.full((256, 256, 3), 200, np.uint8)
        assert __import__("cv2").imwrite(str(d / "images" / name), img)
        labels[name] = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    if legacy:
        labels = [{"value": v[0], "Count": 2} for v in labels.values()]
    (d / "labels.json").write_text(json.dumps(labels))
    return d


@pytest.fixture
def root(tmp_path):
    for s in ("train", "val", "test"):
        make_split(tmp_path, s)
    return tmp_path


def test_resolve_and_preflight(root):
    assert resolve_data_root(str(root)) == root / "papermask"
    counts = preflight(str(root))
    assert counts == {"train": 3, "val": 3, "test": 3}


def test_rejects_missing_image(tmp_path):
    make_split(tmp_path, "train", n=2)
    d = tmp_path / "papermask" / "train"
    (d / "images" / "00001.jpg").unlink()
    with pytest.raises(PreflightError, match="00001.jpg"):
        preflight(str(tmp_path))


def test_rejects_legacy_label_shape(tmp_path):
    make_split(tmp_path, "train", n=2, legacy=True)
    with pytest.raises(PreflightError, match="not a filename->corners map"):
        preflight(str(tmp_path))


def test_rejects_three_points(tmp_path):
    make_split(tmp_path, "train", n=2)
    d = tmp_path / "papermask" / "train"
    labels = json.loads((d / "labels.json").read_text())
    labels["00000.jpg"] = labels["00000.jpg"][:3]
    (d / "labels.json").write_text(json.dumps(labels))
    with pytest.raises(PreflightError, match="exactly 4"):
        preflight(str(tmp_path))


def test_dataset_item(root):
    ds = PaperCornerDataset(str(root), "train", augment=False)
    item = ds[0]
    assert item["image"].shape == (3, 256, 256)
    assert item["image"].dtype.is_floating_point
    assert item["corners"].shape == (4, 2)
    # 0.2 * 256 = 51.2 -> order preserved TL,TR,BR,BL
    assert np.allclose(item["corners"].numpy(), [[51.2, 51.2], [204.8, 51.2], [204.8, 204.8], [51.2, 204.8]])


def test_order_corners_recovers_flipped_quad():
    pts = np.array([[204.8, 51.2], [204.8, 204.8], [51.2, 204.8], [51.2, 51.2]], np.float32)
    out = order_corners(pts)
    assert np.allclose(out, [[51.2, 51.2], [204.8, 51.2], [204.8, 204.8], [51.2, 204.8]])


def test_augment_never_leaves_frame(root):
    ds = PaperCornerDataset(str(root), "train", augment=True, seed=0)
    for i in range(30):
        item = ds[i % 3]
        assert item["corners"].min() >= 0 and item["corners"].max() <= 256
```

Note: `make_split` writes the same 3 images for each split, so augmenting may occasionally exhaust 5 attempts — `repeat` in the constructor (`repeat=1`) does not change this test; if it ever raises, the test is telling you rejection sampling is too aggressive.

- [ ] **Step 2: Run it**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/data/paper_dataset.py`**

```python
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
    missing = [s for s in SPLITS if not (p / s / "images").is_dir()]
    if missing:
        raise PreflightError(f"missing split directories under {p}: {missing}")
    return p


def _fail(msg: str) -> None:
    raise PreflightError(msg)


def preflight(root: str, splits=SPLITS) -> dict[str, int]:
    base = resolve_data_root(root)
    counts: dict[str, int] = {}
    for split in splits:
        d = base / split
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
        cross.append(np.cross(b - a, c - b))
    cross = np.array(cross)
    return bool(np.all(cross > 0) or np.all(cross < 0))


def _train_transform(img_size: int) -> A.Compose:
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.HueSaturationValue(10, 15, 10, p=0.3),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.Affine(rotate=(-8, 8), scale=(0.95, 1.05), shear=(-3, 3),
                     mode=cv2.BORDER_REPLICATE, p=0.5),
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dataset.py -v`
Expected: 7 passed. If `test_augment_never_leaves_frame` fails intermittently, the augmentation is too strong — that is a real signal, not a flaky test.

- [ ] **Step 5: Commit**

```bash
git add src/data/paper_dataset.py tests/test_dataset.py
git commit -m "feat(data): papermask loader with preflight and rejection sampling"
```

---

### Task 3: Lite-FPN neck

**Files:**
- Create: `src/models/litefpn.py`
- Test: `tests/test_litefpn.py`

**Interfaces:**
- Produces: `DepthwiseSeparableConv(in_ch, out_ch)`, `LiteFPN(in_channels: list[int], out_channels: int = 64)` whose `forward(feats: list[Tensor]) -> list[Tensor]` returns one `(B,C,h_i,w_i)` per input, same spatial sizes and order.

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.models.litefpn import DepthwiseSeparableConv, LiteFPN


def test_dwconv_shape():
    m = DepthwiseSeparableConv(16, 32)
    assert m(torch.zeros(2, 16, 8, 8)).shape == (2, 32, 8, 8)


def test_litefpn_shapes_and_order():
    neck = LiteFPN([16, 24, 40, 96], 64)
    feats = [torch.zeros(2, 16, 64, 64), torch.zeros(2, 24, 32, 32),
             torch.zeros(2, 40, 16, 16), torch.zeros(2, 96, 8, 8)]
    out = neck(feats)
    assert [tuple(t.shape) for t in out] == [(2, 64, 64, 64), (2, 64, 32, 32),
                                             (2, 64, 16, 16), (2, 64, 8, 8)]


def test_top_down_path_is_used():
    """A signal in the coarsest level must reach the finest level."""
    neck = LiteFPN([8, 8, 8, 8], 16).eval()
    feats = [torch.zeros(1, 8, 64, 64), torch.zeros(1, 8, 32, 32),
             torch.zeros(1, 8, 16, 16), torch.zeros(1, 8, 8, 8)]
    feats[3][0, :, 0, 0] = 5.0
    out = neck(feats)
    assert out[0].abs().sum() > 0
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_litefpn.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/models/litefpn.py`**

```python
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
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_litefpn.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/models/litefpn.py tests/test_litefpn.py
git commit -m "feat(model): Lite-FPN neck"
```

---

### Task 4: Corner model

**Files:**
- Create: `src/models/paper_corner.py`
- Modify: `src/models/__init__.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `LiteFPN` (Task 3).
- Produces:
  - `PaperCornerModel(backbone="mobilenet_v3_small", pretrained=False, fpn_channels=64, heatmap_size=64, num_corners=4)`
  - `forward(x) -> {"heatmaps": (B,4,64,64) sigmoid, "offsets": (B,8,64,64) raw, "logits": (B,4,64,64)}`
  - `build_model(cfg: dict) -> PaperCornerModel` — the only construction path.

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.models.paper_corner import PaperCornerModel, build_model


def test_forward_shapes():
    model = PaperCornerModel(pretrained=False, fpn_channels=32).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 256, 256))
    assert out["heatmaps"].shape == (2, 4, 64, 64)
    assert out["offsets"].shape == (2, 8, 64, 64)
    assert out["logits"].shape == (2, 4, 64, 64)
    assert out["heatmaps"].min() >= 0 and out["heatmaps"].max() <= 1


def test_heatmap_bias_starts_low():
    model = PaperCornerModel(pretrained=False)
    assert torch.allclose(model.heatmap_head.bias, torch.full((4,), -4.6), atol=1e-4)


def test_offsets_start_at_zero():
    model = PaperCornerModel(pretrained=False)
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 256, 256))
    assert out["offsets"].abs().max() < 1e-6


def test_build_model_from_config():
    cfg = {"model": {"backbone": "mobilenet_v3_small", "pretrained": False,
                     "fpn_channels": 32, "heatmap_size": 64, "sigma": 2.0}}
    m = build_model(cfg)
    assert isinstance(m, PaperCornerModel)
    assert m.heatmap_size == 64


def test_param_count_under_budget():
    m = PaperCornerModel(pretrained=False)
    assert sum(p.numel() for p in m.parameters()) < 1_500_000
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/models/paper_corner.py`**

```python
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
            for stage in self.stages:
                probe = stage(probe)
                channels.append(probe.shape[1])
            spatial = probe.shape[-1]

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
```

- [ ] **Step 4: Update `src/models/__init__.py`**

```python
from src.models.litefpn import DepthwiseSeparableConv, LiteFPN
from src.models.paper_corner import PaperCornerModel, build_model

__all__ = ["DepthwiseSeparableConv", "LiteFPN", "PaperCornerModel", "build_model"]
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: 5 passed. The param count is ~1.1 M; if it is over 1.5 M, the tap slices are wrong.

- [ ] **Step 6: Commit**

```bash
git add src/models/paper_corner.py src/models/__init__.py tests/test_model.py
git commit -m "feat(model): MobileNetV3-Small + Lite-FPN corner model"
```

---

### Task 5: Target encoding, loss, decoding

**Files:**
- Create: `src/losses/corner_loss.py`
- Create: `src/utils/corners.py`
- Test: `tests/test_corner_loss.py`
- Test: `tests/test_decode.py`

**Interfaces:**
- Produces:
  - `encode_targets(corners_px: Tensor(B,4,2), img_size: int, heatmap_size: int, sigma: float) -> (hm(B,4,H,W), off(B,8,H,W), mask(B,4,H,W))`
  - `CornerLoss(heatmap_weight=1.0, offset_weight=1.0, sigma=2.0)` → `forward(logits, offsets, corners_px, img_size, heatmap_size) -> dict(total, heatmap, offset)`
  - `decode_corners(heatmaps, offsets, img_size) -> (corners_px(B,4,2), scores(B,4))`
  - `quad_props(pts) -> (area, is_convex, is_clockwise)`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corner_loss.py
import torch
from src.losses.corner_loss import CornerLoss, encode_targets


def test_target_shapes_and_values():
    corners = torch.tensor([[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]]])
    hm, off, mask = encode_targets(corners, 256, 64, 2.0)
    assert hm.shape == (1, 4, 64, 64) and off.shape == (1, 8, 64, 64)
    assert torch.allclose(hm[0, 0, 16, 16], torch.tensor(1.0), atol=1e-6)
    assert mask.sum() == 4
    assert torch.allclose(off[0, 0, 16, 16], torch.tensor(0.0))


def test_offset_target_fractional():
    corners = torch.tensor([[[66.0, 64.0], [192, 64], [192, 192], [64, 192]]])
    _, off, _ = encode_targets(corners, 256, 64, 2.0)
    assert torch.allclose(off[0, 0, 16, 16], torch.tensor(0.5))


def test_loss_zero_for_perfect_prediction():
    corners = torch.tensor([[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]]])
    hm, off, _ = encode_targets(corners, 256, 64, 2.0)
    logits = torch.logit(hm.clamp(1e-6, 1 - 1e-6))
    loss = CornerLoss()(logits, off, corners, 256, 64)
    assert loss["total"].item() < 1e-3


def test_loss_large_for_wrong_prediction():
    corners = torch.tensor([[[64.0, 64.0], [192.0, 64.0], [192.0, 192.0], [64.0, 192.0]]])
    loss = CornerLoss()(torch.full((1, 4, 64, 64), -10.0), torch.zeros(1, 8, 64, 64),
                        corners, 256, 64)
    assert loss["heatmap"].item() > 1.0
```

```python
# tests/test_decode.py
import torch
from src.losses.corner_loss import encode_targets
from src.utils.corners import decode_corners, quad_props


def test_decode_is_subpixel_exact():
    corners = torch.tensor([[[66.0, 70.5], [190.0, 62.0], [200.0, 194.0], [58.0, 186.0]]])
    hm, off, _ = encode_targets(corners, 256, 64, 2.0)
    pred, scores = decode_corners(hm, off, 256)
    assert torch.allclose(pred, corners, atol=0.5), (pred, corners)
    assert scores.min() > 0.9


def test_quad_props():
    tl, tr, br, bl = (0, 0), (10, 0), (10, 10), (0, 10)
    area, convex, cw = quad_props(torch.tensor([[tl, tr, br, bl]], dtype=torch.float32)[0])
    assert abs(area - 100) < 1e-3 and convex and cw
```

- [ ] **Step 2: Run them**

Run: `pytest tests/test_corner_loss.py tests/test_decode.py -v`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement `src/losses/corner_loss.py`**

```python
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
    cells = torch.floor(hm_coord).clamp(0, heatmap_size - 1).long()
    offsets = (hm_coord - cells.to(hm_coord.dtype)).clamp(0.0, 1.0)

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
        pos = target
        neg = 1 - pos
        pos_loss = -torch.log(p) * (1 - p).pow(2) * pos
        neg_loss = -torch.log(1 - p) * p.pow(2) * neg.pow(4)
        n = b * logits.shape[1]
        heatmap_loss = (pos_loss.sum() + neg_loss.sum()) / n

        offset_loss = F.l1_loss(offsets * mask, off_t * mask, reduction="sum") / n

        total = self.heatmap_weight * heatmap_loss + self.offset_weight * offset_loss
        return {"total": total, "heatmap": heatmap_loss.detach(),
                "offset": offset_loss.detach()}
```

- [ ] **Step 4: Implement `src/utils/corners.py`**

```python
"""Decoding and quad geometry shared by training, evaluation, export checks."""
from __future__ import annotations

import torch


@torch.no_grad()
def decode_corners(heatmaps: torch.Tensor, offsets: torch.Tensor,
                   img_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, n, h, w = heatmaps.shape
    stride = img_size / h
    flat = heatmaps.view(b, n, -1)
    scores, idx = flat.max(dim=-1)
    iy, ix = idx // w, idx % w

    bi = torch.arange(b, device=heatmaps.device).view(b, 1)
    ci = torch.arange(n, device=heatmaps.device).view(1, n)
    ox = offsets[bi, 2 * ci, iy, ix]
    oy = offsets[bi, 2 * ci + 1, iy, ix]

    x = (ix.to(heatmaps.dtype) + ox) * stride
    y = (iy.to(heatmaps.dtype) + oy) * stride
    return torch.stack([x, y], dim=-1), scores


@torch.no_grad()
def quad_props(pts: torch.Tensor) -> tuple[float, bool, bool]:
    """pts: (4,2) float tensor. Returns (area, is_convex, is_clockwise)."""
    p = pts.detach().to(torch.float64).cpu()
    x, y = p[:, 0], p[:, 1]
    area = float(abs((x * y.roll(-1) - y * x.roll(-1)).sum()) / 2)
    cross = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross.append(float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
    convex = all(v > 0 for v in cross) or all(v < 0 for v in cross)
    return area, convex, area > 0 and all(v > 0 for v in cross)


def winding(pts: torch.Tensor) -> int:
    """+1 clockwise in image coordinates (y down), -1 counter-clockwise."""
    p = pts.detach().to(torch.float64).cpu()
    x, y = p[:, 0], p[:, 1]
    signed = float((x * y.roll(-1) - y * x.roll(-1)).sum())
    return 1 if signed > 0 else -1
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_corner_loss.py tests/test_decode.py -v`
Expected: 6 passed

- [ ] **Step 6: Commit**

```bash
git add src/losses/corner_loss.py src/utils/corners.py tests/test_corner_loss.py tests/test_decode.py
git commit -m "feat(loss): focal heatmap + masked offset loss, sub-pixel decode"
```

---

### Task 6: Metrics

**Files:**
- Create: `src/metrics/corners.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `CornerMetrics(img_size=256, heatmap_size=64)` with `update(pred(B,4,2), gt(B,4,2))` and `compute() -> dict` containing `mae_px, rmse_px, nmae, pck1, pck2, pck5, iou, invalid_frac, per_corner_mae (list[4])`. Predictions are never clamped here — invalid quads are counted, not fixed.

- [ ] **Step 1: Write the failing test**

```python
import torch
from src.metrics.corners import CornerMetrics


def test_perfect_prediction():
    gt = torch.tensor([[[10.0, 10.0], [230.0, 10.0], [230.0, 230.0], [10.0, 230.0]]])
    m = CornerMetrics()
    m.update(gt.clone(), gt)
    r = m.compute()
    assert r["mae_px"] == 0 and r["iou"] == 1.0 and r["pck1"] == 1.0


def test_offset_of_one_pixel():
    gt = torch.tensor([[[10.0, 10.0], [230.0, 10.0], [230.0, 230.0], [10.0, 230.0]]])
    pred = gt + 1.0
    m = CornerMetrics()
    m.update(pred, gt)
    r = m.compute()
    assert abs(r["mae_px"] - 1.0) < 1e-5
    assert r["pck1"] == 1.0          # 1 px == 1 % of 100... no: threshold is 1% of the diagonal
    assert r["per_corner_mae"] == [1.0, 1.0, 1.0, 1.0]


def test_invalid_quad_counted():
    gt = torch.tensor([[[10.0, 10.0], [230.0, 10.0], [230.0, 230.0], [10.0, 230.0]]])
    pred = gt.clone()
    pred[0, 2] = pred[0, 0]          # collapse one corner
    m = CornerMetrics()
    m.update(pred, gt)
    assert m.compute()["invalid_frac"] == 1.0
```

Note: PCK thresholds are fractions of the image diagonal (1 %, 2 %, 5 %). With a 256-px image the diagonal is ~362 px, so 1 % ≈ 3.6 px — a 1-px error passes.

- [ ] **Step 2: Run it**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/metrics/corners.py`**

```python
"""Corner metrics in native image pixels (no rescaling to 256-space lies)."""
from __future__ import annotations

import math

import numpy as np
import torch
from shapely.geometry import Polygon

from src.utils.corners import quad_props


class CornerMetrics:
    def __init__(self, img_size: int = 256):
        self.img_size = img_size
        self.reset()

    def reset(self) -> None:
        self._pred: list[np.ndarray] = []
        self._gt: list[np.ndarray] = []

    @torch.no_grad()
    def update(self, pred: torch.Tensor, gt: torch.Tensor) -> None:
        self._pred.append(pred.detach().float().cpu().numpy())
        self._gt.append(gt.detach().float().cpu().numpy())

    def compute(self) -> dict:
        if not self._pred:
            return {k: 0.0 for k in ("mae_px", "rmse_px", "nmae", "pck1", "pck2", "pck5", "iou", "invalid_frac")} | {"per_corner_mae": [0.0] * 4}
        pred = np.concatenate(self._pred)
        gt = np.concatenate(self._gt)
        err = np.linalg.norm(pred - gt, axis=-1)          # (N,4)
        diag = math.hypot(self.img_size, self.img_size)

        ious, invalid = [], 0
        for p, g in zip(pred, gt):
            area, convex, _ = quad_props(torch.from_numpy(p))
            if not convex or area < 64:
                invalid += 1
                ious.append(0.0)
                continue
            try:
                a, b = Polygon(p).buffer(0), Polygon(g).buffer(0)
                ious.append(a.intersection(b).area / a.union(b).area)
            except Exception:
                ious.append(0.0)

        return {
            "mae_px": float(err.mean()),
            "rmse_px": float(np.sqrt((err ** 2).mean())),
            "nmae": float(err.mean() / self.img_size),
            "pck1": float((err <= 0.01 * diag).mean()),
            "pck2": float((err <= 0.02 * diag).mean()),
            "pck5": float((err <= 0.05 * diag).mean()),
            "iou": float(np.mean(ious)),
            "invalid_frac": invalid / len(pred),
            "per_corner_mae": [float(v) for v in err.mean(0)],
        }
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py -v`
Expected: 3 passed. (`per_corner_mae` is a list of floats; `==` works because they are exact.)

- [ ] **Step 5: Commit**

```bash
git add src/metrics/corners.py tests/test_metrics.py
git commit -m "feat(metrics): pixel-space MAE, PCK and quad IoU"
```

---

### Task 7: Trainer

**Files:**
- Create: `src/train.py`
- Create: `src/utils/seed.py`
- Create: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: config (T1), dataset (T2), model (T4), loss/decode (T5), metrics (T6).
- Produces: `main()`; checkpoint dict `{arch_version, config, state_dict, epoch, metrics}`; `--preflight` and `--resume PATH`.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_train_smoke.py
import json
import numpy as np
import pytest
import torch

from src.train import train_one_epoch, validate

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_one_epoch_runs(tmp_path):
    # 2 images per split is enough to exercise the loop
    import cv2
    root = tmp_path / "papermask"
    for split in ("train", "val"):
        d = root / split
        (d / "images").mkdir(parents=True)
        labels = {}
        for i in range(2):
            name = f"{i:05d}.jpg"
            cv2.imwrite(str(d / "images" / name), np.full((256, 256, 3), 180, np.uint8))
            labels[name] = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        (d / "labels.json").write_text(json.dumps(labels))

    from src.data.paper_dataset import PaperCornerDataset
    from src.losses.corner_loss import CornerLoss
    from src.models.paper_corner import PaperCornerModel
    from torch.utils.data import DataLoader

    ds = PaperCornerDataset(str(tmp_path), "train")
    dl = DataLoader(ds, batch_size=2)
    model = PaperCornerModel(pretrained=False, fpn_channels=16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = CornerLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=False)

    loss = train_one_epoch(model, dl, opt, criterion, scaler, torch.device("cpu"),
                           256, 64, grad_clip=1.0, amp=False)
    assert loss > 0
    metrics = validate(model, DataLoader(ds, batch_size=2), torch.device("cpu"), 256, 64)
    assert "mae_px" in metrics
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_train_smoke.py -v`
Expected: FAIL — `src.train` not found

- [ ] **Step 3: Implement `src/utils/seed.py`**

```python
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, workers: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if workers:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
```

- [ ] **Step 4: Implement `src/train.py`**

```python
"""Single training entry point. All behaviour comes from the config."""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import load_config
from src.data.paper_dataset import PaperCornerDataset, build_datasets, preflight
from src.losses.corner_loss import CornerLoss
from src.metrics.corners import CornerMetrics
from src.models.paper_corner import build_model
from src.utils.corners import decode_corners
from src.utils.seed import seed_everything

ARCH_VERSION = 2


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    base = info.dataset.seed if info is not None else 0
    seed_everything(base + worker_id, workers=False)


def _loader(ds, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=shuffle, persistent_workers=workers > 0,
                      worker_init_fn=_worker_init)


def train_one_epoch(model, loader, opt, criterion, scaler, device, img_size, heatmap_size,
                    grad_clip: float, amp: bool, log_interval: int = 50,
                    writer=None, epoch: int = 0) -> float:
    model.train()
    total, seen = 0.0, 0
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        images = batch["image"].to(device, non_blocking=True)
        corners = batch["corners"].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=amp and device.type == "cuda"):
            out = model(images)
            loss = criterion(out["logits"], out["offsets"], corners, img_size, heatmap_size)["total"]

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

        total += loss.item() * images.shape[0]
        seen += images.shape[0]
        if step % log_interval == 0:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if writer is not None:
                writer.add_scalar("train/step_loss", loss.item(), epoch * len(loader) + step)
    return total / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device, img_size, heatmap_size) -> dict:
    model.eval()
    metrics = CornerMetrics(img_size)
    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        out = model(images)
        pred, _ = decode_corners(out["heatmaps"].float(), out["offsets"].float(), img_size)
        metrics.update(pred, batch["corners"].to(device))
    return metrics.compute()


def _save_visuals(model, ds, device, img_size, out_dir: Path, n: int = 8) -> None:
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    for k, i in enumerate(idx):
        item = ds[int(i)]
        with torch.no_grad():
            out = model(item["image"].unsqueeze(0).to(device))
            pred, scores = decode_corners(out["heatmaps"].float(), out["offsets"].float(), img_size)
        img = item["image"].permute(1, 2, 0).numpy()
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        canvas = np.repeat(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)[:, :, None], 2, 1)
        canvas = cv2.resize(canvas, (img_size * 2, img_size))
        for pts, color in ((item["corners"].numpy(), (0, 255, 0)),
                           (pred[0].cpu().numpy(), (0, 0, 255))):
            poly = pts.astype(np.int32)
            cv2.polylines(canvas, [poly], True, color, 2)
        cv2.imwrite(str(out_dir / f"epoch_{k:02d}.jpg"), canvas)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.preflight:
        print(preflight(cfg["data"]["root"]))
        return

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size, hm_size = cfg["data"]["img_size"], cfg["model"]["heatmap_size"]

    train_ds, val_ds, _ = build_datasets(cfg)
    bs, workers = cfg["train"]["batch_size"], cfg["data"]["num_workers"]
    train_dl = _loader(train_ds, bs, workers, True)
    val_dl = _loader(val_ds, bs * 2, workers, False)

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")
    criterion = CornerLoss(cfg["loss"]["heatmap_weight"], cfg["loss"]["offset_weight"],
                           cfg["model"]["sigma"])

    epochs = cfg["train"]["epochs"]
    warmup = cfg["train"]["warmup_epochs"]

    def lr_lambda(ep: int) -> float:
        if ep < warmup:
            return (ep + 1) / max(warmup, 1)
        t = (ep - warmup) / max(epochs - warmup, 1)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(out_dir / "logs")
    start_epoch, best = 0, float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("arch_version") != ARCH_VERSION:
            raise SystemExit(f"checkpoint arch_version {ckpt.get('arch_version')} != {ARCH_VERSION}")
        model.load_state_dict(ckpt["state_dict"])
        opt.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best = ckpt["metrics"].get("mae_px", float("inf"))

    patience = 0
    for epoch in range(start_epoch, epochs):
        train_ds.set_epoch(epoch)
        loss = train_one_epoch(model, train_dl, opt, criterion, scaler, device, img_size,
                               hm_size, cfg["train"]["grad_clip"],
                               cfg["train"]["amp"], cfg["train"]["log_interval"], writer, epoch)
        metrics = validate(model, val_dl, device, img_size, hm_size)
        sched.step()

        writer.add_scalar("train/epoch_loss", loss, epoch)
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"val/{k}", v, epoch)
        print(f"epoch {epoch:3d} loss {loss:.4f} mae {metrics['mae_px']:.3f}px "
              f"iou {metrics['iou']:.4f} lr {opt.param_groups[0]['lr']:.2e}")

        ckpt = {"arch_version": ARCH_VERSION, "config": cfg, "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "epoch": epoch, "metrics": metrics}
        torch.save(ckpt, out_dir / "last.pth")
        if metrics["mae_px"] < best:
            best = metrics["mae_px"]
            patience = 0
            torch.save(ckpt, out_dir / "best.pth")
            _save_visuals(model, val_ds, device, img_size, out_dir / "visualizations" / f"epoch_{epoch:03d}")
        else:
            patience += 1
        if not cfg["train"]["save_best_only"]:
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pth")
        if patience >= cfg["train"]["early_stop_patience"]:
            print(f"early stopping at epoch {epoch}")
            break

    writer.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the smoke test**

Run: `pytest tests/test_train_smoke.py -v`
Expected: PASS in < 60 s on CPU. If `GradScaler` warns, that is fine on CPU (`enabled=False`).

- [ ] **Step 6: Commit**

```bash
git add src/train.py src/utils/seed.py tests/test_train_smoke.py
git commit -m "feat(train): single strict-config trainer with AMP clipping and checkpoints"
```

---

### Task 8: Export

**Files:**
- Create: `src/export.py`
- Test: `tests/test_export.py` (marked `slow`)

**Interfaces:**
- Consumes: checkpoint from Task 7.
- Produces: `export_checkpoint(ckpt_path, out_dir) -> Path` writing `paper_v2.onnx` + `model.json`; CLI.

- [ ] **Step 1: Write the failing test**

```python
import json
import pytest
import torch

from src.export import export_checkpoint
from src.models.paper_corner import PaperCornerModel

pytestmark = pytest.mark.slow


def test_export_writes_onnx_and_metadata(tmp_path):
    model = PaperCornerModel(pretrained=False, fpn_channels=16)
    cfg = {"seed": 1,
           "data": {"root": "x", "img_size": 256, "num_workers": 0, "augment": False},
           "model": {"backbone": "mobilenet_v3_small", "pretrained": False,
                     "fpn_channels": 16, "heatmap_size": 64, "sigma": 2.0},
           "train": {"output_dir": str(tmp_path), "batch_size": 1, "epochs": 1, "lr": 1e-3,
                     "weight_decay": 0.0, "warmup_epochs": 0, "grad_clip": 1.0, "amp": False,
                     "early_stop_patience": 1, "save_best_only": True, "log_interval": 1},
           "loss": {"heatmap_weight": 1.0, "offset_weight": 1.0}}
    ckpt = tmp_path / "best.pth"
    torch.save({"arch_version": 2, "config": cfg, "state_dict": model.state_dict(),
                "epoch": 0, "metrics": {"mae_px": 1.0}}, ckpt)

    onnx_path = export_checkpoint(ckpt, tmp_path / "out")
    assert onnx_path.exists() and onnx_path.stat().st_size > 100_000
    meta = json.loads((tmp_path / "out" / "model.json").read_text())
    assert meta["output"]["stride"] == 4 and meta["corner_order"] == "TL,TR,BR,BL"
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_export.py -v -m slow`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/export.py`**

```python
"""Checkpoint -> ONNX + model.json, with an onnxruntime parity gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.models.paper_corner import build_model

ARCH_VERSION = 2
INPUT_SIZE = 256


class _ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out["heatmaps"], out["offsets"]


def export_checkpoint(ckpt_path: str | Path, out_dir: str | Path) -> Path:
    ckpt_path, out_dir = Path(ckpt_path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("arch_version") != ARCH_VERSION:
        raise SystemExit(f"unsupported arch_version: {ckpt.get('arch_version')}")
    cfg = ckpt["config"]

    model = build_model(cfg).eval()
    model.load_state_dict(ckpt["state_dict"])

    sample = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        ref_hm, ref_off = _ExportWrapper(model)(sample)

    onnx_path = out_dir / "paper_v2.onnx"
    torch.onnx.export(
        _ExportWrapper(model).eval(), sample, str(onnx_path),
        input_names=["image"], output_names=["heatmaps", "offsets"],
        opset_version=17, do_constant_folding=True,
        dynamic_axes={"image": {0: "batch"}, "heatmaps": {0: "batch"}, "offsets": {0: "batch"}},
    )

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got_hm, got_off = sess.run(None, {"image": sample.numpy()})
    diff_hm = np.abs(got_hm - ref_hm.numpy()).max()
    diff_off = np.abs(got_off - ref_off.numpy()).max()
    if max(diff_hm, diff_off) > 1e-4:
        onnx_path.unlink(missing_ok=True)
        raise SystemExit(f"ONNX parity failed: heatmaps {diff_hm:.2e}, offsets {diff_off:.2e}")

    meta = {
        "arch_version": ARCH_VERSION,
        "corner_order": "TL,TR,BR,BL",
        "confidence_threshold": 0.3,
        "input": {"name": "image", "height": INPUT_SIZE, "width": INPUT_SIZE,
                  "layout": "CHW", "color": "RGB",
                  "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "output": {"heatmaps": "heatmaps", "offsets": "offsets",
                   "heatmap_size": cfg["model"]["heatmap_size"],
                   "stride": INPUT_SIZE // cfg["model"]["heatmap_size"]},
        "parity": {"heatmaps_max_abs": float(diff_hm), "offsets_max_abs": float(diff_off)},
    }
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2))
    print(f"exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB), parity ok")
    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    export_checkpoint(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it**

Run: `pytest tests/test_export.py -v -m slow`
Expected: 1 passed; ONNX is ~1.5 MB.

- [ ] **Step 5: Commit**

```bash
git add src/export.py tests/test_export.py
git commit -m "feat(export): single build_model path, opset 17, parity gate, model.json"
```

---

### Task 9: Evaluation CLI

**Files:**
- Create: `src/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `evaluate(ckpt_path, config_path, split="test") -> dict`; CLI printing a markdown table and writing `metrics.json`.

- [ ] **Step 1: Write the failing test**

```python
import json, numpy as np, cv2, torch
from src.evaluate import evaluate


def test_evaluate_returns_metrics(tmp_path):
    root = tmp_path / "papermask" / "test"
    (root / "images").mkdir(parents=True)
    labels = {}
    for i in range(2):
        name = f"{i:05d}.jpg"
        cv2.imwrite(str(root / "images" / name), np.full((256, 256, 3), 200, np.uint8))
        labels[name] = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    (root / "labels.json").write_text(json.dumps(labels))

    cfg = {"seed": 0, "data": {"root": str(tmp_path), "img_size": 256, "num_workers": 0, "augment": False},
           "model": {"backbone": "mobilenet_v3_small", "pretrained": False, "fpn_channels": 16,
                     "heatmap_size": 64, "sigma": 2.0},
           "train": {"output_dir": str(tmp_path / "out"), "batch_size": 2, "epochs": 1, "lr": 1e-3,
                     "weight_decay": 0.0, "warmup_epochs": 0, "grad_clip": 1.0, "amp": False,
                     "early_stop_patience": 1, "save_best_only": True, "log_interval": 1},
           "loss": {"heatmap_weight": 1.0, "offset_weight": 1.0}}
    from src.models.paper_corner import PaperCornerModel
    ckpt = tmp_path / "best.pth"
    torch.save({"arch_version": 2, "config": cfg,
                "state_dict": PaperCornerModel(pretrained=False, fpn_channels=16).state_dict(),
                "epoch": 0, "metrics": {}}, ckpt)
    cfg_path = tmp_path / "c.yaml"
    import yaml; cfg_path.write_text(yaml.safe_dump(cfg))

    m = evaluate(ckpt, cfg_path, split="test")
    assert "mae_px" in m and "iou" in m
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `src/evaluate.py`**

```python
"""Held-out evaluation from a checkpoint. No config drift: the checkpoint wins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.paper_dataset import PaperCornerDataset
from src.metrics.corners import CornerMetrics
from src.models.paper_corner import build_model
from src.utils.corners import decode_corners


@torch.no_grad()
def evaluate(ckpt_path: str | Path, config_path: str | Path, split: str = "test") -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if ckpt.get("arch_version") != 2:
        raise SystemExit(f"unsupported arch_version: {ckpt.get('arch_version')}")
    cfg = ckpt["config"] if "config" in ckpt else load_config(config_path)
    if not cfg["data"]["augment"]:
        pass
    cfg["data"]["root"] = load_config(config_path)["data"]["root"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = cfg["data"]["img_size"]
    ds = PaperCornerDataset(cfg["data"]["root"], split, img_size=img_size)
    dl = DataLoader(ds, batch_size=64, num_workers=cfg["data"]["num_workers"])

    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    metrics = CornerMetrics(img_size)
    for batch in tqdm(dl, desc=split):
        out = model(batch["image"].to(device))
        pred, _ = decode_corners(out["heatmaps"].float(), out["offsets"].float(), img_size)
        metrics.update(pred, batch["corners"].to(device))
    return metrics.compute()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    m = evaluate(args.checkpoint, args.config, args.split)
    print(f"\n| metric | value |\n|---|---|")
    for k, v in m.items():
        print(f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |")
    out = Path(args.out or Path(args.checkpoint).parent / "metrics.json")
    out.write_text(json.dumps(m, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_evaluate.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add src/evaluate.py tests/test_evaluate.py
git commit -m "feat(eval): held-out metric table from checkpoint"
```

---

### Task 10: .NET decoder parity

**Files:**
- Modify: `dotnet_project/Program.cs` (rewrite `PredictCorners`, add `LoadMeta`, `Decode`, `ValidateQuad`)
- Modify: `dotnet_project/dotnet_project.csproj` (copy `model.json` to output)
- Create: `tests/parity/README.md`

**Interfaces:**
- Consumes: `paper_v2.onnx` + `model.json` from Task 8.
- Produces: `Corner[] Decode(float[] heatmaps, float[] offsets, int hm, int stride, int imgSize)` matching `src/utils/corners.py` exactly.

- [ ] **Step 1: Write the parity fixture (Python side)**

```python
# tests/parity/make_fixture.py
"""Writes 20 (input, expected corners) pairs used by the C# parity check."""
import json, numpy as np, torch
from src.data.paper_dataset import PaperCornerDataset
from src.utils.corners import decode_corners
from src.models.paper_corner import build_model

# run once with a real checkpoint, dump corners.json; the C# app reads the same
# ONNX and must reproduce these within 0.5 px.
```

- [ ] **Step 2: Rewrite the decoder in C#**

```csharp
private static readonly float[] Mean = { 0.485f, 0.456f, 0.406f };
private static readonly float[] Std  = { 0.229f, 0.224f, 0.225f };

private sealed record Corner(double X, double Y, double Score);

private static Corner[] Decode(float[] hm, float[] off, int c, int h, int w, float stride)
{
    var result = new Corner[4];
    for (int ch = 0; ch < 4; ch++)
    {
        int best = 0;
        float bestVal = float.MinValue;
        for (int i = 0; i < h * w; i++)
        {
            float v = hm[ch * h * w + i];
            if (v > bestVal) { bestVal = v; best = i; }
        }
        int iy = best / w, ix = best % w;
        float ox = off[(2 * ch) * h * w + best];
        float oy = off[(2 * ch + 1) * h * w + best];
        result[ch] = new Corner((ix + ox) * stride, (iy + oy) * stride, bestVal);
    }
    return result;
}

private static bool ValidateQuad(Corner[] c, int width, int height)
{
    foreach (var p in c)
        if (p.X < 0 || p.Y < 0 || p.X > width || p.Y > height) return false;

    double area = 0;
    for (int i = 0; i < 4; i++)
    {
        var a = c[i]; var b = c[(i + 1) % 4];
        area += a.X * b.Y - b.X * a.Y;
    }
    area = Math.Abs(area) / 2;
    if (area < 64) return false;

    for (int i = 0; i < 4; i++)
    {
        var a = c[i]; var b = c[(i + 1) % 4]; var d = c[(i + 2) % 4];
        double cross = (b.X - a.X) * (d.Y - b.Y) - (b.Y - a.Y) * (d.X - b.X);
        if (cross <= 0) return false;   // clockwise in image coordinates
    }
    return true;
}
```

`Decode` must be called with the **output tensors flattened in the same layout the ONNX model emits** (`[1,4,64,64]` and `[1,8,64,64]`), and `stride = imgSize / h`.

- [ ] **Step 3: Read `model.json` instead of hardcoding**

```csharp
private sealed class ModelMeta
{
    public string OnnxName = "paper_v2.onnx";
    public int InputSize = 256, HeatmapSize = 64, Stride = 4;
    public string InputName = "image", HeatmapName = "heatmaps", OffsetName = "offsets";
    public double ConfidenceThreshold = 0.3;

    public static ModelMeta Load(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        var root = doc.RootElement;
        var o = root.GetProperty("output");
        return new ModelMeta
        {
            InputSize = root.GetProperty("input").GetProperty("height").GetInt32(),
            HeatmapSize = o.GetProperty("heatmap_size").GetInt32(),
            Stride = o.GetProperty("stride").GetInt32(),
            InputName = root.GetProperty("input").GetProperty("name").GetString() ?? "image",
            HeatmapName = o.GetProperty("heatmaps").GetString() ?? "heatmaps",
            OffsetName = o.GetProperty("offsets").GetString() ?? "offsets",
            ConfidenceThreshold = root.GetProperty("confidence_threshold").GetDouble(),
        };
    }
}
```

- [ ] **Step 4: Replace every argmax `* 4 * scale` with `Decode` + `ValidateQuad`**
  - log rejected images with the reason and the four scores;
  - clamp final points to the source image before `GetPerspectiveTransform`;
  - mirror the input subdirectory in the output path;
  - wrap `Mat`/`DenseTensor` in `using`;
  - print the full exception (`ex.ToString()`), not `ex.Message`.

- [ ] **Step 5: Run the parity check against 20 exported fixtures**

Expected: every corner within 0.5 px of the Python value, and `model.json`'s confidence threshold rejects the same images.

- [ ] **Step 6: Commit**

```bash
git add dotnet_project/Program.cs dotnet_project/dotnet_project.csproj tests/parity
git commit -m "feat(dotnet): sub-pixel offset decode, quad validation, model.json metadata"
```

---

### Task 11: Remove the legacy pipeline and update docs

**Files:**
- Delete: `src/models/point_regression.py`, `src/models/heatmap_regression.py`, `src/train_point.py`,
  `src/train_heatmap.py`, `src/data/dataset.py`, `src/losses/smooth_l1.py`, `src/losses/awing.py`,
  `src/losses/heatmap_focal.py`, `configs/point_base.yaml`, `configs/heatmap_base.yaml`,
  `configs/heatmap_phase2.yaml`, `configs/heatmap_phase3.yaml`, `analyze_heatmap_model.py`
- Modify: `README.md`, `AGENTS.md`, `AUDIT.md` (add a "resolved by v2" column), `QUICKSTART*.md`
- Modify: `refine_corners.py` (build from checkpoint config, use `decode_corners`, prefer `model.json`)

**Interfaces:** none new; the repo must import cleanly after the deletion.

- [ ] **Step 1: Adapt `refine_corners.py` to the new model**

Replace its model construction with:

```python
from src.models.paper_corner import build_model
from src.utils.corners import decode_corners

ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
model = build_model(ckpt["config"]).eval()
model.load_state_dict(ckpt["state_dict"])
...
out = model(img_tensor)
corners_px, scores = decode_corners(out["heatmaps"], out["offsets"], img_size)
```

and delete the `outputs['heatmaps'].numpy()` call that was missing `.cpu()`.

- [ ] **Step 2: Delete the legacy files**

```bash
git rm src/models/point_regression.py src/models/heatmap_regression.py src/train_point.py \
       src/train_heatmap.py src/data/dataset.py src/losses/smooth_l1.py src/losses/awing.py \
       src/losses/heatmap_focal.py configs/point_base.yaml configs/heatmap_base.yaml \
       configs/heatmap_phase2.yaml configs/heatmap_phase3.yaml analyze_heatmap_model.py
```

- [ ] **Step 3: Verify nothing imports the deleted modules**

Run:
```bash
grep -rn "heatmap_regression\|point_regression\|train_heatmap\|train_point\|losses.smooth_l1\|losses.awing\|losses.heatmap_focal" --include=*.py --include=*.md .
```
Expected: no hits outside `AUDIT.md`'s history section.

- [ ] **Step 4: Update `README.md`** — new architecture diagram, the new commands
(`train.py`, `export.py`, `evaluate.py`), the dataset layout, and the corrected metric table from Task 9.

- [ ] **Step 5: Update `AGENTS.md`** — replace the "two models" table with the single Lite-FPN model,
the new file map, and the Kaggle workflow. Keep the audit's historical findings but mark them resolved.

- [ ] **Step 6: Full test run**

Run: `pytest -q -m "not slow"`
Expected: all green; `pytest -q -m slow` green as well.

- [ ] **Step 7: Commit and merge**

```bash
git add -A
git commit -m "refactor: retire legacy point/heatmap pipeline; docs for v2"
git checkout main && git merge --no-ff feat/litefpn-heatmap-offset
git push origin main
```

---

## Self-Review

**Spec coverage**

| Spec section | Task |
|---|---|
| §2 data contract + preflight | 2 |
| §3 model (trunk, neck, heads, init, params) | 3, 4 |
| §4 targets / loss / decode | 5 |
| §5 metrics | 6 |
| §6 training loop, AMP order, seeds, checkpoints | 7 |
| §7 export + model.json + parity | 8 |
| §8 .NET | 10 |
| §9 Kaggle | already delivered (`kaggle/papermask.ipynb`); Task 11 updates docs |
| §10 tests | 1–9 |
| §11 deletions | 11 |

**Known gaps to close during execution**

- The Kaggle notebook calls `src/train.py --preflight`, which Task 7 implements; it also calls
  `src/export.py` and `src/evaluate.py` with the flags defined in Tasks 8 and 9 — no further
  notebook edits are needed.
- `tests/parity/make_fixture.py` needs a trained checkpoint to produce fixtures. Until the first
  Kaggle run, the C# parity test is skipped; the decoder unit test (hand-computed ONNX outputs)
  covers the arithmetic in the meantime.
- The spec's `confidence_threshold` is applied in C# only; `evaluate.py` reports scores but does
  not filter on them. That is deliberate — filtering the metric would hide recall loss.
