from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from terrashift.data.bands import CHANGE_CLASSES

from .. import crud, schemas as s
from ..database import get_db

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=s.StatsOut)
def get_stats(scene_id: int | None = None, db: Session = Depends(get_db)):
    return crud.stats(db, scene_id)


@router.get("/classes", response_model=list[str])
def classes():
    return list(CHANGE_CLASSES)
