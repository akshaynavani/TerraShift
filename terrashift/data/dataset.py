"""PyTorch dataset for bi-temporal change classification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .bands import BAND_MEAN, BAND_STD, class_index
from .io import TileRecord, load_pair


def normalize(img: np.ndarray) -> np.ndarray:
    mean = np.asarray(BAND_MEAN, dtype=np.float32)[:, None, None]
    std = np.asarray(BAND_STD, dtype=np.float32)[:, None, None]
    return (img - mean) / std


def stack_pair(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Early-fusion input: normalised before and after concatenated on channels."""
    return np.concatenate([normalize(before), normalize(after)], axis=0).astype(np.float32)


def augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Geometric augmentation applied identically to both dates (they share channels)."""
    k = int(rng.integers(0, 4))
    x = np.rot90(x, k, axes=(1, 2))
    if rng.random() < 0.5:
        x = x[:, :, ::-1]
    if rng.random() < 0.5:
        x = x[:, ::-1, :]
    return np.ascontiguousarray(x)


class ChangePairDataset(Dataset):
    def __init__(self, root: str | Path, records: list[TileRecord], train: bool = False, seed: int = 0):
        self.root = Path(root)
        self.records = [r for r in records if r.label is not None]
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        rec = self.records[idx]
        before, after = load_pair(self.root, rec)
        x = stack_pair(before, after)
        if self.train:
            x = augment(x, self.rng)
        return torch.from_numpy(x), class_index(rec.label)


def train_val_split(records: list[TileRecord], val_fraction: float, seed: int) -> tuple[list[TileRecord], list[TileRecord]]:
    """Stratified split by label."""
    rng = np.random.default_rng(seed)
    by_label: dict[str, list[TileRecord]] = {}
    for r in records:
        if r.label is not None:
            by_label.setdefault(r.label, []).append(r)
    train, val = [], []
    for recs in by_label.values():
        idx = rng.permutation(len(recs))
        n_val = max(1, int(round(len(recs) * val_fraction))) if len(recs) > 1 else 0
        val += [recs[i] for i in idx[:n_val]]
        train += [recs[i] for i in idx[n_val:]]
    return train, val
