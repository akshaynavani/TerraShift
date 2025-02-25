"""Band layout and change-class taxonomy."""
from __future__ import annotations

# Sentinel-2-like subset, in channel order.
BANDS: tuple[str, ...] = ("blue", "green", "red", "nir", "swir1", "swir2")
BAND_INDEX = {name: i for i, name in enumerate(BANDS)}
NUM_BANDS = len(BANDS)

CHANGE_CLASSES: tuple[str, ...] = (
    "no_change",
    "vegetation_loss",
    "urban_expansion",
    "water_change",
    "agricultural_change",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(CHANGE_CLASSES)}
NUM_CLASSES = len(CHANGE_CLASSES)

# Approximate per-band reflectance statistics (0-1 scale) for normalisation.
BAND_MEAN = (0.08, 0.10, 0.11, 0.28, 0.22, 0.15)
BAND_STD = (0.04, 0.05, 0.06, 0.10, 0.08, 0.07)


def class_index(name: str) -> int:
    try:
        return CLASS_TO_IDX[name]
    except KeyError as exc:
        raise ValueError(f"Unknown change class '{name}'. Expected one of {CHANGE_CLASSES}") from exc
