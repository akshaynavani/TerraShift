"""Random Forest change classifier over spectral change features."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import SEED, RFTrainConfig
from ..data.bands import CHANGE_CLASSES
from ..features import extract_features, feature_names


class RandomForestChangeModel:
    name = "random_forest"

    def __init__(self, cfg: RFTrainConfig | None = None, pipeline: Pipeline | None = None):
        cfg = cfg or RFTrainConfig()
        self.pipeline = pipeline or Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=cfg.n_estimators,
                        max_depth=cfg.max_depth,
                        min_samples_leaf=cfg.min_samples_leaf,
                        class_weight=cfg.class_weight,
                        n_jobs=cfg.n_jobs,
                        random_state=SEED,
                    ),
                ),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestChangeModel":
        self.pipeline.fit(X, y)
        return self

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        """Return probabilities over all CHANGE_CLASSES, even if some were unseen in training."""
        proba = self.pipeline.predict_proba(X)
        full = np.zeros((X.shape[0], len(CHANGE_CLASSES)), dtype=np.float32)
        for col, cls in enumerate(self.pipeline.classes_):
            full[:, int(cls)] = proba[:, col]
        return full

    def predict_proba(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        return self.predict_proba_features(extract_features(before, after)[None])[0]

    def feature_importances(self, top_k: int = 15) -> list[tuple[str, float]]:
        imp = self.pipeline.named_steps["rf"].feature_importances_
        names = feature_names()
        order = np.argsort(imp)[::-1][:top_k]
        return [(names[i], float(imp[i])) for i in order]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestChangeModel":
        return cls(pipeline=joblib.load(path))
