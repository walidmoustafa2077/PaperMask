"""Single seeding entry point so every process in a run starts identically."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
