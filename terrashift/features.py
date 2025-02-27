"""Hand-crafted spectral change features for the Random Forest baseline."""
from __future__ import annotations

import numpy as np

from .data.bands import BANDS
from .data.indices import INDEX_FUNCS

STATS = ("mean", "std", "p10", "p90")
CHANGE_THRESHOLDS = (0.1, 0.2, 0.3)


def _stats(x: np.ndarray) -> list[float]:
    flat = x.ravel()
    return [float(flat.mean()), float(flat.std()), float(np.percentile(flat, 10)), float(np.percentile(flat, 90))]


def feature_names() -> list[str]:
    names: list[str] = []
    for b in BANDS:
        names += [f"{b}_t0_mean", f"{b}_t1_mean", f"{b}_delta_mean", f"{b}_absdelta_mean"]
    for idx in INDEX_FUNCS:
        names += [f"{idx}_t0_{s}" for s in STATS]
        names += [f"{idx}_t1_{s}" for s in STATS]
        names += [f"{idx}_delta_{s}" for s in STATS]
        names += [f"{idx}_frac_up_{t}" for t in CHANGE_THRESHOLDS]
        names += [f"{idx}_frac_down_{t}" for t in CHANGE_THRESHOLDS]
    names += ["spectral_cva_mean", "spectral_cva_p90", "spectral_cva_frac_gt_0.1"]
    return names


def extract_features(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    feats: list[float] = []
    delta = after - before
    for i in range(len(BANDS)):
        feats += [
            float(before[i].mean()),
            float(after[i].mean()),
            float(delta[i].mean()),
            float(np.abs(delta[i]).mean()),
        ]
    for fn in INDEX_FUNCS.values():
        i0, i1 = fn(before), fn(after)
        d = i1 - i0
        feats += _stats(i0) + _stats(i1) + _stats(d)
        feats += [float((d > t).mean()) for t in CHANGE_THRESHOLDS]
        feats += [float((d < -t).mean()) for t in CHANGE_THRESHOLDS]
    # Change Vector Analysis magnitude across all bands.
    cva = np.sqrt((delta**2).sum(axis=0))
    feats += [float(cva.mean()), float(np.percentile(cva, 90)), float((cva > 0.1).mean())]
    return np.asarray(feats, dtype=np.float32)


def extract_batch(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.stack([extract_features(b, a) for b, a in pairs])
