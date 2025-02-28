"""Train and evaluate the Random Forest baseline."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import SEED, RFTrainConfig
from ..data.bands import class_index
from ..data.dataset import train_val_split
from ..data.io import TileRecord, load_pair
from ..features import extract_features
from ..models.random_forest import RandomForestChangeModel
from .metrics import evaluate


def build_xy(root: Path, records: list[TileRecord]) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([extract_features(*load_pair(root, r)) for r in records])
    y = np.array([class_index(r.label) for r in records])
    return X, y


def train_random_forest(
    root: str | Path,
    records: list[TileRecord],
    out_path: str | Path,
    cfg: RFTrainConfig | None = None,
    val_fraction: float = 0.2,
) -> tuple[RandomForestChangeModel, dict]:
    root = Path(root)
    train_recs, val_recs = train_val_split(records, val_fraction, SEED)
    X_tr, y_tr = build_xy(root, train_recs)
    model = RandomForestChangeModel(cfg).fit(X_tr, y_tr)
    metrics: dict = {}
    if val_recs:
        X_va, y_va = build_xy(root, val_recs)
        y_hat = model.predict_proba_features(X_va).argmax(axis=1)
        metrics = evaluate(y_va, y_hat)
    metrics["top_features"] = model.feature_importances()
    model.save(out_path)
    return model, metrics
