from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import crud, models as m, schemas as s
from ..database import get_db
from ..inference import InferenceService, get_inference
from .tiles import get_tile

router = APIRouter(tags=["predictions"])


def _check_model(svc: InferenceService, model: str) -> None:
    available = svc.available()
    if not available:
        raise HTTPException(503, "no trained models found; run scripts/train.py first")
    if model != "auto" and model not in available:
        raise HTTPException(400, f"model '{model}' not available; choose from {available}")


@router.get("/models", response_model=s.ModelInfo)
def models_info(db: Session = Depends(get_db), svc: InferenceService = Depends(get_inference)):
    return s.ModelInfo(available=svc.available(), versions=crud.list_model_versions(db))


@router.post("/models/reload", response_model=s.ModelInfo)
def reload_models(db: Session = Depends(get_db), svc: InferenceService = Depends(get_inference)):
    return s.ModelInfo(available=svc.reload(), versions=crud.list_model_versions(db))


@router.post("/tiles/{tile_id}/predict", response_model=s.PredictionOut)
def predict_tile(
    tile_id: int,
    req: s.PredictRequest = s.PredictRequest(),
    db: Session = Depends(get_db),
    svc: InferenceService = Depends(get_inference),
):
    _check_model(svc, req.model)
    tile = get_tile(db, tile_id)
    try:
        pred = svc.predict(tile.before_path, tile.after_path, req.model)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    return crud.prediction_out(crud.save_prediction(db, tile, pred, svc.artifact_dir))


@router.post("/scenes/{scene_id}/predict", response_model=s.BatchPredictResult)
def predict_scene(
    scene_id: int,
    req: s.PredictRequest = s.PredictRequest(),
    only_missing: bool = Query(True, description="Skip tiles that already have a prediction"),
    db: Session = Depends(get_db),
    svc: InferenceService = Depends(get_inference),
):
    _check_model(svc, req.model)
    if db.get(m.Scene, scene_id) is None:
        raise HTTPException(404, "scene not found")
    tiles = list(db.scalars(select(m.Tile).where(m.Tile.scene_id == scene_id)))
    n, model_used = 0, req.model
    for tile in tiles:
        if only_missing and tile.predictions:
            continue
        try:
            pred = svc.predict(tile.before_path, tile.after_path, req.model)
        except (FileNotFoundError, ValueError):
            continue
        crud.save_prediction(db, tile, pred, svc.artifact_dir, commit=False)
        model_used = pred.model
        n += 1
        if n % 100 == 0:
            db.commit()
    db.commit()
    return s.BatchPredictResult(scene_id=scene_id, predicted=n, model=model_used)


@router.get("/review/queue", response_model=list[s.TileOut])
def review_queue(
    scene_id: int | None = None,
    limit: int = Query(25, ge=1, le=200),
    strategy: str = Query("uncertainty", pattern="^(uncertainty|entropy|changes)$"),
    db: Session = Depends(get_db),
):
    return [crud.tile_out(t) for t in crud.review_queue(db, scene_id, limit, strategy)]
