"""Synthetic bi-temporal multispectral tile generator.

Each tile is a mixture of land-cover endmembers (spectral signatures) arranged with
smooth random fields. The "after" image applies a class-specific change inside a
random blob, plus radiometric noise and a small illumination shift, so models must
learn the change pattern rather than raw brightness.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .bands import CHANGE_CLASSES, NUM_BANDS
from .io import TileRecord, write_manifest

# Reflectance signatures in band order (blue, green, red, nir, swir1, swir2).
ENDMEMBERS: dict[str, np.ndarray] = {
    "forest": np.array([0.03, 0.06, 0.04, 0.40, 0.18, 0.08]),
    "grass": np.array([0.05, 0.09, 0.07, 0.32, 0.24, 0.13]),
    "crop": np.array([0.04, 0.08, 0.05, 0.45, 0.22, 0.10]),
    "bare_soil": np.array([0.12, 0.15, 0.19, 0.25, 0.33, 0.27]),
    "urban": np.array([0.14, 0.15, 0.17, 0.20, 0.27, 0.24]),
    "water": np.array([0.07, 0.06, 0.04, 0.02, 0.01, 0.01]),
}
BASE_COVERS = ("forest", "grass", "crop", "bare_soil", "urban", "water")


def _smooth_field(rng: np.random.Generator, size: int, scale: int = 8) -> np.ndarray:
    coarse = rng.random((size // scale + 2, size // scale + 2))
    # bilinear upsample
    ys = np.linspace(0, coarse.shape[0] - 1.001, size)
    xs = np.linspace(0, coarse.shape[1] - 1.001, size)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = coarse[y0][:, x0]
    b = coarse[y0][:, x0 + 1]
    c = coarse[y0 + 1][:, x0]
    d = coarse[y0 + 1][:, x0 + 1]
    return a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + d * fy * fx


def _blob_mask(rng: np.random.Generator, size: int) -> np.ndarray:
    field = _smooth_field(rng, size, scale=rng.choice([8, 16]))
    cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
    yy, xx = np.mgrid[:size, :size]
    radius = rng.uniform(0.15, 0.35) * size
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / radius
    return ((1.0 - dist) + 0.4 * (field - 0.5)) > 0.0


def _render(rng: np.random.Generator, size: int, covers: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Render a scene; returns (image, cover-index map)."""
    fields = np.stack([_smooth_field(rng, size) for _ in covers])
    cover_map = fields.argmax(axis=0)
    sig = np.stack([ENDMEMBERS[c] for c in covers])  # (k, bands)
    img = sig[cover_map].transpose(2, 0, 1)  # (bands, H, W)
    return img.astype(np.float32), cover_map


def _paint(img: np.ndarray, mask: np.ndarray, cover: str, rng: np.random.Generator) -> None:
    sig = ENDMEMBERS[cover] * rng.uniform(0.9, 1.1, NUM_BANDS)
    img[:, mask] = sig[:, None]


def _apply_change(before: np.ndarray, label: str, rng: np.random.Generator) -> np.ndarray:
    size = before.shape[-1]
    after = before.copy()
    if label == "no_change":
        return after
    mask = _blob_mask(rng, size)
    if label == "vegetation_loss":
        _paint(after, mask, rng.choice(["bare_soil", "grass"]) if rng.random() < 0.7 else "bare_soil", rng)
    elif label == "urban_expansion":
        _paint(after, mask, "urban", rng)
        # add a street-grid texture
        grid = np.zeros_like(mask)
        grid[:: rng.integers(4, 8), :] = True
        grid[:, :: rng.integers(4, 8)] = True
        _paint(after, mask & grid, "bare_soil", rng)
    elif label == "water_change":
        water_present = (before[3] < 0.05)[mask].mean() > 0.5
        _paint(after, mask, "bare_soil" if water_present else "water", rng)
    elif label == "agricultural_change":
        # stripes of alternating crop / bare soil (harvest or new fields)
        stripes = (np.arange(size)[None, :] // rng.integers(3, 7)) % 2 == 0
        stripes = np.broadcast_to(stripes, mask.shape)
        _paint(after, mask & stripes, "crop", rng)
        _paint(after, mask & ~stripes, "bare_soil", rng)
    else:
        raise ValueError(label)
    return after


def _radiometry(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = rng.uniform(0.92, 1.08)
    offset = rng.uniform(-0.01, 0.01)
    noise = rng.normal(0, 0.008, img.shape)
    return np.clip(img * gain + offset + noise, 0.0, 1.0).astype(np.float32)


def generate_pair(label: str, size: int = 64, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    k = rng.integers(2, 4)
    covers = list(rng.choice(BASE_COVERS, size=k, replace=False))
    if label == "vegetation_loss" and "forest" not in covers:
        covers[0] = "forest"
    before, _ = _render(rng, size, covers)
    after = _apply_change(before, label, rng)
    return _radiometry(before, rng), _radiometry(after, rng)


def generate_dataset(root: str | Path, n_per_class: int = 200, size: int = 64, seed: int = 42) -> list[TileRecord]:
    root = Path(root)
    tiles_dir = root / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    records: list[TileRecord] = []
    for label in CHANGE_CLASSES:
        for i in range(n_per_class):
            tile_id = f"{label}_{i:05d}"
            before, after = generate_pair(label, size, rng)
            b_rel, a_rel = f"tiles/{tile_id}_t0.npy", f"tiles/{tile_id}_t1.npy"
            np.save(root / b_rel, before)
            np.save(root / a_rel, after)
            records.append(
                TileRecord(
                    tile_id=tile_id,
                    before=b_rel,
                    after=a_rel,
                    label=label,
                    lat=float(rng.uniform(-50, 60)),
                    lon=float(rng.uniform(-120, 150)),
                )
            )
    rng.shuffle(records)
    write_manifest(root, records)
    return records
