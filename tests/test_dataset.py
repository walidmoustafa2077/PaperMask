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
