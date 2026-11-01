"""The documented entry points must be runnable as `python -m src.<name>`.

`python src/train.py` puts `src/` on `sys.path`, so the `from src.…` imports in
every module fail with `ModuleNotFoundError: No module named 'src'`. The README,
QUICKSTART, AGENTS.md and kaggle/papermask.ipynb all use the `-m` form; this
test keeps that promise honest.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Each case pays a fresh torch import (~5 s), so keep them out of the fast loop.
pytestmark = pytest.mark.slow


@pytest.mark.parametrize("module", ["src.train", "src.evaluate", "src.export"])
def test_entrypoint_runs_as_module(module):
    proc = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage:" in proc.stdout
