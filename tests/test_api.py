import pytest
from fastapi.testclient import TestClient

from terrashift.data.bands import CHANGE_CLASSES
from terrashift.data.io import read_manifest
from terrashift.models.predictor import ChangePredictor
from terrashift.training.train_rf import train_random_forest


@pytest.fixture(scope="module")
def client(synthetic_dir, tmp_root, small_rf_cfg):
    from api.inference import get_inference
    from api.main import app

    rf, _ = train_random_forest(synthetic_dir, read_manifest(synthetic_dir), tmp_root / "api_rf.joblib", small_rf_cfg)
    get_inference().set_predictor(ChangePredictor(rf=rf))
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def scene_id(client):
    r = client.post("/scenes", json={"name": "test-scene", "region": "nowhere"})
    assert r.status_code == 201
    sid = r.json()["id"]
    r = client.post(f"/scenes/{sid}/import", json={"manifest_dir": "synthetic", "limit": 20})
    assert r.status_code == 201 and r.json()["added"] == 20
    return sid


def test_health_and_classes(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/classes").json() == list(CHANGE_CLASSES)


def test_duplicate_scene_conflict(client, scene_id):
    assert client.post("/scenes", json={"name": "test-scene"}).status_code == 409


def test_import_rejects_path_escape(client, scene_id):
    r = client.post(f"/scenes/{scene_id}/import", json={"manifest_dir": "../../etc"})
    assert r.status_code == 400


def test_reimport_is_idempotent(client, scene_id):
    r = client.post(f"/scenes/{scene_id}/import", json={"manifest_dir": "synthetic", "limit": 20})
    assert r.json()["added"] == 0
    assert client.get(f"/scenes/{scene_id}").json()["tile_count"] == 20


def test_models_endpoint(client):
    assert client.get("/models").json()["available"] == ["random_forest"]


def test_predict_unknown_model(client, scene_id):
    tile_id = client.get("/tiles", params={"scene_id": scene_id}).json()[0]["id"]
    assert client.post(f"/tiles/{tile_id}/predict", json={"model": "resnet50"}).status_code == 400


def test_full_review_flow(client, scene_id):
    r = client.post(f"/scenes/{scene_id}/predict", json={"model": "auto"})
    assert r.status_code == 200 and r.json()["predicted"] == 20

    queue = client.get("/review/queue", params={"scene_id": scene_id, "limit": 5}).json()
    assert len(queue) == 5
    confidences = [t["latest_prediction"]["confidence"] for t in queue]
    assert confidences == sorted(confidences)

    tile = queue[0]
    pred = tile["latest_prediction"]
    wrong = next(c for c in CHANGE_CLASSES if c != pred["label"])

    r = client.post(f"/tiles/{tile['id']}/annotations", json={"analyst": "ana", "label": wrong, "notes": "clearly roads"})
    assert r.status_code == 201
    assert r.json()["action"] == "corrected" and r.json()["prediction_id"] == pred["id"]

    second = queue[1]
    r = client.post(
        f"/tiles/{second['id']}/annotations", json={"analyst": "ana", "label": second["latest_prediction"]["label"]}
    )
    assert r.json()["action"] == "confirmed"

    remaining = client.get("/review/queue", params={"scene_id": scene_id, "limit": 50}).json()
    assert tile["id"] not in {t["id"] for t in remaining}
    assert len(remaining) == 18

    stats = client.get("/stats", params={"scene_id": scene_id}).json()
    assert stats["tiles"] == 20 and stats["predicted"] == 20
    assert stats["reviewed"] == 2 and stats["corrected"] == 1
    assert stats["agreement_rate"] == pytest.approx(0.5)

    export = client.get("/annotations/export").json()
    assert {row["tile_id"] for row in export} == {tile["id"], second["id"]}

    detail = client.get(f"/tiles/{tile['id']}").json()
    assert detail["latest_annotation"]["label"] == wrong
    assert len(detail["predictions"]) == 1


def test_annotation_validation(client, scene_id):
    tile_id = client.get("/tiles", params={"scene_id": scene_id}).json()[0]["id"]
    assert client.post(f"/tiles/{tile_id}/annotations", json={"analyst": "a", "label": "lava"}).status_code == 422
    assert client.post(f"/tiles/{tile_id}/annotations", json={"analyst": "", "label": "no_change"}).status_code == 422


def test_previews(client, scene_id):
    tile_id = client.get("/tiles", params={"scene_id": scene_id}).json()[0]["id"]
    for which in ("before", "after", "ndvi_diff"):
        r = client.get(f"/tiles/{tile_id}/preview/{which}.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert client.get(f"/tiles/{tile_id}/preview/bogus.png").status_code == 400
    assert client.get("/tiles/999999").status_code == 404
