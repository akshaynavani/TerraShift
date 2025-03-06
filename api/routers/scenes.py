from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .. import crud, models as m, schemas as s
from ..database import get_db
from ..inference import InferenceService, get_inference

router = APIRouter(prefix="/scenes", tags=["scenes"])


def _get_scene(db: Session, scene_id: int) -> m.Scene:
    scene = db.get(m.Scene, scene_id)
    if scene is None:
        raise HTTPException(404, "scene not found")
    return scene


@router.post("", response_model=s.SceneOut, status_code=201)
def create_scene(data: s.SceneCreate, db: Session = Depends(get_db)):
    try:
        scene = crud.create_scene(db, data)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"scene '{data.name}' already exists")
    return crud.scene_out(db, scene)


@router.get("", response_model=list[s.SceneOut])
def list_scenes(db: Session = Depends(get_db)):
    return [crud.scene_out(db, sc) for sc in crud.list_scenes(db)]


@router.get("/{scene_id}", response_model=s.SceneOut)
def get_scene(scene_id: int, db: Session = Depends(get_db)):
    return crud.scene_out(db, _get_scene(db, scene_id))


@router.post("/{scene_id}/tiles", status_code=201)
def add_tiles(scene_id: int, tiles: list[s.TileCreate], db: Session = Depends(get_db)):
    _get_scene(db, scene_id)
    return {"added": crud.add_tiles(db, scene_id, tiles)}


@router.post("/{scene_id}/import", status_code=201)
def import_manifest(
    scene_id: int,
    req: s.SceneImport,
    db: Session = Depends(get_db),
    svc: InferenceService = Depends(get_inference),
):
    _get_scene(db, scene_id)
    try:
        svc.resolve(req.manifest_dir)
        added = crud.import_manifest(db, scene_id, svc.imagery_root, req)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc))
    return {"added": added}
