import numpy as np
import pytest

from terrashift.data.bands import CHANGE_CLASSES, NUM_BANDS, class_index
from terrashift.data.dataset import ChangePairDataset, augment, stack_pair, train_val_split
from terrashift.data.indices import ndbi, ndvi, ndwi
from terrashift.data.io import read_manifest, to_rgb_uint8
from terrashift.data.synthetic import ENDMEMBERS, generate_pair


def _uniform(sig):
    return np.broadcast_to(np.asarray(sig, dtype=np.float32)[:, None, None], (NUM_BANDS, 4, 4)).copy()


def test_indices_have_expected_sign_for_endmembers():
    forest, water, urban = (_uniform(ENDMEMBERS[k]) for k in ("forest", "water", "urban"))
    assert ndvi(forest).mean() > 0.5
    assert ndwi(water).mean() > 0.3
    assert ndbi(urban).mean() > 0
    assert ndbi(forest).mean() < 0


def test_indices_bounded():
    img = np.random.default_rng(0).random((NUM_BANDS, 8, 8)).astype(np.float32)
    for fn in (ndvi, ndwi, ndbi):
        v = fn(img)
        assert v.min() >= -1 and v.max() <= 1


@pytest.mark.parametrize("label", CHANGE_CLASSES)
def test_generate_pair_shapes_and_range(label, rng):
    before, after = generate_pair(label, size=32, rng=rng)
    assert before.shape == after.shape == (NUM_BANDS, 32, 32)
    assert before.dtype == np.float32
    assert 0 <= before.min() and after.max() <= 1


def test_change_classes_differ_more_than_no_change(rng):
    def mean_abs_delta(label):
        return np.mean([np.abs(np.subtract(*generate_pair(label, 32, rng))).mean() for _ in range(10)])

    assert mean_abs_delta("urban_expansion") > mean_abs_delta("no_change")


def test_manifest_and_dataset(synthetic_dir):
    records = read_manifest(synthetic_dir)
    assert len(records) == 12 * len(CHANGE_CLASSES)
    ds = ChangePairDataset(synthetic_dir, records, train=True)
    x, y = ds[0]
    assert x.shape == (2 * NUM_BANDS, 32, 32)
    assert y == class_index(records[0].label)


def test_stratified_split_covers_every_class(synthetic_dir):
    train, val = train_val_split(read_manifest(synthetic_dir), 0.25, seed=0)
    assert {r.label for r in val} == set(CHANGE_CLASSES)
    assert not {r.tile_id for r in train} & {r.tile_id for r in val}


def test_augment_preserves_shape(rng):
    x = stack_pair(*generate_pair("water_change", 32, rng))
    assert augment(x, rng).shape == x.shape


def test_rgb_preview(rng):
    before, _ = generate_pair("no_change", 16, rng)
    rgb = to_rgb_uint8(before)
    assert rgb.shape == (16, 16, 3) and rgb.dtype == np.uint8


def test_unknown_class_raises():
    with pytest.raises(ValueError):
        class_index("deforestation")
