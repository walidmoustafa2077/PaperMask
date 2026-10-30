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
