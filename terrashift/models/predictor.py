"""Unified inference interface over the RF, ResNet-50, or an ensemble of both."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..data.bands import CHANGE_CLASSES
from ..data.dataset import stack_pair
from .random_forest import RandomForestChangeModel
from .resnet import BiTemporalResNet50


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    model: str

    @property
    def entropy(self) -> float:
        p = np.clip(np.array(list(self.probabilities.values())), 1e-9, 1.0)
        return float(-(p * np.log(p)).sum())


def _to_prediction(proba: np.ndarray, model_name: str) -> Prediction:
    idx = int(proba.argmax())
    return Prediction(
        label=CHANGE_CLASSES[idx],
        confidence=float(proba[idx]),
        probabilities={c: float(p) for c, p in zip(CHANGE_CLASSES, proba)},
        model=model_name,
    )


class ChangePredictor:
    """Loads whichever model artifacts exist and predicts on (before, after) pairs."""

    def __init__(
        self,
        rf: RandomForestChangeModel | None = None,
        cnn: BiTemporalResNet50 | None = None,
        cnn_weight: float = 0.6,
    ):
        if rf is None and cnn is None:
            raise ValueError("ChangePredictor needs at least one model")
        self.rf, self.cnn, self.cnn_weight = rf, cnn, cnn_weight

    @classmethod
    def from_artifacts(cls, artifact_dir: str | Path) -> "ChangePredictor":
        artifact_dir = Path(artifact_dir)
        rf_path, cnn_path = artifact_dir / "random_forest.joblib", artifact_dir / "resnet50.pt"
        rf = RandomForestChangeModel.load(rf_path) if rf_path.exists() else None
        cnn = BiTemporalResNet50.load(cnn_path) if cnn_path.exists() else None
        return cls(rf=rf, cnn=cnn)

    @property
    def available(self) -> list[str]:
        out = []
        if self.rf is not None:
            out.append("random_forest")
        if self.cnn is not None:
            out.append("resnet50")
        if len(out) == 2:
            out.append("ensemble")
        return out

    def _rf_proba(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        assert self.rf is not None
        return self.rf.predict_proba(before, after)

    @torch.no_grad()
    def _cnn_proba(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        assert self.cnn is not None
        x = torch.from_numpy(stack_pair(before, after))[None]
        return torch.softmax(self.cnn(x), dim=1)[0].numpy()

    def predict(self, before: np.ndarray, after: np.ndarray, model: str = "auto") -> Prediction:
        if model == "auto":
            model = "ensemble" if "ensemble" in self.available else self.available[0]
        if model not in self.available:
            raise ValueError(f"Model '{model}' not loaded; available: {self.available}")
        if model == "random_forest":
            proba = self._rf_proba(before, after)
        elif model == "resnet50":
            proba = self._cnn_proba(before, after)
        else:
            w = self.cnn_weight
            proba = w * self._cnn_proba(before, after) + (1 - w) * self._rf_proba(before, after)
        return _to_prediction(proba, model)
