"""Central configuration for paths and training defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TERRASHIFT_DATA_DIR", ROOT / "data"))
ARTIFACT_DIR = Path(os.getenv("TERRASHIFT_ARTIFACT_DIR", ROOT / "artifacts"))

TILE_SIZE = 64
SEED = 42


@dataclass
class CNNTrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.2
    patience: int = 4
    pretrained: bool = True
    num_workers: int = 0
    device: str = field(default_factory=lambda: os.getenv("TERRASHIFT_DEVICE", "auto"))


@dataclass
class RFTrainConfig:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 2
    class_weight: str | None = "balanced"
    n_jobs: int = -1
