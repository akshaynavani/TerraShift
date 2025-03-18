import numpy as np
import pytest
import torch

from terrashift.data.bands import CHANGE_CLASSES, NUM_BANDS, NUM_CLASSES
from terrashift.data.io import load_pair, read_manifest
from terrashift.data.synthetic import generate_pair
from terrashift.features import extract_features, feature_names
from terrashift.models.predictor import ChangePredictor
from terrashift.models.random_forest import RandomForestChangeModel
from terrashift.models.resnet import BiTemporalResNet50
from terrashift.training.metrics import evaluate, format_report
from terrashift.training.train_rf import train_random_forest


def test_feature_vector_matches_names(rng):
    f = extract_features(*generate_pair("vegetation_loss", 32, rng))
    assert f.shape == (len(feature_names()),)
    assert np.isfinite(f).all()


@pytest.fixture(scope="module")
def trained_rf(synthetic_dir, tmp_root, small_rf_cfg):
    model, metrics = train_random_forest(
        synthetic_dir, read_manifest(synthetic_dir), tmp_root / "rf_test.joblib", small_rf_cfg, val_fraction=0.25
    )
    return model, metrics


def test_rf_beats_chance(trained_rf):
    _, metrics = trained_rf
    assert metrics["accuracy"] > 1.5 / NUM_CLASSES
    assert len(metrics["top_features"]) > 0


def test_rf_roundtrip(trained_rf, tmp_root, synthetic_dir):
    model, _ = trained_rf
    loaded = RandomForestChangeModel.load(tmp_root / "rf_test.joblib")
    rec = read_manifest(synthetic_dir)[0]
    b, a = load_pair(synthetic_dir, rec)
    np.testing.assert_allclose(model.predict_proba(b, a), loaded.predict_proba(b, a))
    assert loaded.predict_proba(b, a).sum() == pytest.approx(1.0, abs=1e-5)


def test_resnet_forward_and_save(tmp_root):
    model = BiTemporalResNet50(pretrained=False).eval()
    x = torch.randn(2, 2 * NUM_BANDS, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, NUM_CLASSES)
    path = tmp_root / "resnet_test.pt"
    model.save(path)
    reloaded = BiTemporalResNet50.load(path)
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), out)


def test_predictor_ensemble(trained_rf, rng):
    model, _ = trained_rf
    cnn = BiTemporalResNet50(pretrained=False).eval()
    predictor = ChangePredictor(rf=model, cnn=cnn)
    assert set(predictor.available) == {"random_forest", "resnet50", "ensemble"}
    pred = predictor.predict(*generate_pair("water_change", 32, rng))
    assert pred.model == "ensemble"
    assert pred.label in CHANGE_CLASSES
    assert sum(pred.probabilities.values()) == pytest.approx(1.0, abs=1e-4)
    assert pred.entropy >= 0


def test_predictor_rejects_missing_model(trained_rf, rng):
    predictor = ChangePredictor(rf=trained_rf[0])
    with pytest.raises(ValueError):
        predictor.predict(*generate_pair("no_change", 32, rng), model="resnet50")


def test_metrics_perfect_and_report():
    y = np.arange(NUM_CLASSES).repeat(3)
    m = evaluate(y, y)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0 and m["change_detection_f1"] == 1.0
    assert "confusion matrix" in format_report(m)
