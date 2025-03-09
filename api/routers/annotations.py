from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from .. import crud, schemas as s
from ..database import get_db
from .tiles import get_tile

router = APIRouter(tags=["annotations"])


@router.post("/tiles/{tile_id}/annotations", response_model=s.AnnotationOut, status_code=201)
def annotate(tile_id: int, data: s.AnnotationCreate, db: Session = Depends(get_db)):
    tile = get_tile(db, tile_id)
    try:
        return crud.create_annotation(db, tile, data)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/tiles/{tile_id}/annotations", response_model=list[s.AnnotationOut])
def list_annotations(tile_id: int, db: Session = Depends(get_db)):
    return get_tile(db, tile_id).annotations


@router.get("/annotations/export", response_model=list[s.ExportRow])
def export(db: Session = Depends(get_db)):
    """Latest non-skipped analyst label per tile, ready to feed into retraining."""
    return crud.export_labels(db)
