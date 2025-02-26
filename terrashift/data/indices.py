"""Spectral indices computed from (bands, H, W) reflectance arrays."""
from __future__ import annotations

import numpy as np

from .bands import BAND_INDEX

EPS = 1e-6


def _band(img: np.ndarray, name: str) -> np.ndarray:
    return img[BAND_INDEX[name]].astype(np.float32)


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a - b) / (a + b + EPS)


def ndvi(img: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red)."""
    return normalized_difference(_band(img, "nir"), _band(img, "red"))


def ndwi(img: np.ndarray) -> np.ndarray:
    """McFeeters NDWI (open water): (Green - NIR) / (Green + NIR)."""
    return normalized_difference(_band(img, "green"), _band(img, "nir"))


def ndbi(img: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index: (SWIR1 - NIR) / (SWIR1 + NIR)."""
    return normalized_difference(_band(img, "swir1"), _band(img, "nir"))


def nbr(img: np.ndarray) -> np.ndarray:
    """Normalized Burn Ratio: (NIR - SWIR2) / (NIR + SWIR2)."""
    return normalized_difference(_band(img, "nir"), _band(img, "swir2"))


INDEX_FUNCS = {"ndvi": ndvi, "ndwi": ndwi, "ndbi": ndbi, "nbr": nbr}


def all_indices(img: np.ndarray) -> dict[str, np.ndarray]:
    return {name: fn(img) for name, fn in INDEX_FUNCS.items()}
