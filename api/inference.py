"""Lazily loaded model service shared by the API process."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from terrashift.data.io import load_array
from terrashift.models.predictor import ChangePredictor, Prediction

from .settings import get_settings


class InferenceService:
    def __init__(self, artifact_dir: Path, imagery_root: Path):
        self.artifact_dir = artifact_dir
        self.imagery_root = imagery_root
        self._predictor: ChangePredictor | None = None
        self._lock = threading.Lock()

    @property
    def predictor(self) -> ChangePredictor:
        if self._predictor is None:
            with self._lock:
                if self._predictor is None:
                    self._predictor = ChangePredictor.from_artifacts(self.artifact_dir)
        return self._predictor

    def set_predictor(self, predictor: ChangePredictor | None) -> None:
        self._predictor = predictor

    def available(self) -> list[str]:
        try:
            return self.predictor.available
        except ValueError:
            return []

    def resolve(self, rel_path: str) -> Path:
        path = (self.imagery_root / rel_path).resolve()
        root = self.imagery_root.resolve()
        if root not in path.parents and path != root:
            raise ValueError(f"Path escapes imagery root: {rel_path}")
        return path

    def load_pair(self, before_path: str, after_path: str) -> tuple[np.ndarray, np.ndarray]:
        return load_array(self.resolve(before_path)), load_array(self.resolve(after_path))

    def predict(self, before_path: str, after_path: str, model: str = "auto") -> Prediction:
        before, after = self.load_pair(before_path, after_path)
        return self.predictor.predict(before, after, model)

    def reload(self) -> list[str]:
        with self._lock:
            self._predictor = None
        return self.available()


_service: InferenceService | None = None


def get_inference() -> InferenceService:
    global _service
    if _service is None:
        s = get_settings()
        _service = InferenceService(s.terrashift_artifact_dir, s.terrashift_imagery_root)
    return _service
