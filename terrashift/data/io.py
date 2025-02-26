"""Tile I/O: bi-temporal tiles stored as .npy pairs plus a JSON manifest.

Layout::

    <root>/manifest.json            [{"tile_id", "before", "after", "label"?}, ...]
    <root>/tiles/<tile_id>_t0.npy   float32 (bands, H, W) reflectance in [0, 1]
    <root>/tiles/<tile_id>_t1.npy
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .bands import NUM_BANDS


@dataclass
class TileRecord:
    tile_id: str
    before: str
    after: str
    label: str | None = None
    lat: float | None = None
    lon: float | None = None


def load_array(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix in {".tif", ".tiff"}:
        arr = read_geotiff(path)
    else:
        arr = np.load(path)
    arr = arr.astype(np.float32)
    if arr.ndim != 3 or arr.shape[0] != NUM_BANDS:
        raise ValueError(f"{path}: expected shape ({NUM_BANDS}, H, W), got {arr.shape}")
    return arr


def read_geotiff(path: Path) -> np.ndarray:
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Reading GeoTIFFs requires `pip install terrashift[geo]`") from exc
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    # Sentinel-2 L2A reflectance is scaled by 10000.
    if arr.max() > 1.5:
        arr /= 10000.0
    return np.clip(arr, 0.0, 1.0)


def load_pair(root: Path, rec: TileRecord) -> tuple[np.ndarray, np.ndarray]:
    return load_array(root / rec.before), load_array(root / rec.after)


def read_manifest(root: str | Path) -> list[TileRecord]:
    root = Path(root)
    with open(root / "manifest.json") as f:
        rows = json.load(f)
    return [TileRecord(**row) for row in rows]


def write_manifest(root: str | Path, records: list[TileRecord]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / "manifest.json", "w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


def to_rgb_uint8(img: np.ndarray, gain: float = 3.0) -> np.ndarray:
    """Return an (H, W, 3) uint8 true-colour preview with a simple brightness stretch."""
    from .bands import BAND_INDEX

    rgb = np.stack([img[BAND_INDEX[b]] for b in ("red", "green", "blue")], axis=-1)
    return (np.clip(rgb * gain, 0.0, 1.0) * 255).astype(np.uint8)
