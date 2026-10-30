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
