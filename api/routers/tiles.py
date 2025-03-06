from __future__ import annotations

import io

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.orm import Session

from terrashift.data.indices import INDEX_FUNCS
from terrashift.data.io import to_rgb_uint8

from .. import crud, models as m, schemas as s
from ..database import get_db
from ..inference import InferenceService, get_inference

router = APIRouter(prefix="/tiles", tags=["tiles"])


def get_tile(db: Session, tile_id: int) -> m.Tile:
    tile = db.get(m.Tile, tile_id)
    if tile is None:
        raise HTTPException(404, "tile not found")
    return tile


@router.get("", response_model=list[s.TileOut])
def list_tiles(
    scene_id: int | None = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return [crud.tile_out(t) for t in crud.list_tiles(db, scene_id, offset, limit)]


@router.get("/{tile_id}", response_model=s.TileDetail)
def tile_detail(tile_id: int, db: Session = Depends(get_db)):
    return crud.tile_out(get_tile(db, tile_id), detail=True)


def _png(arr) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).resize((256, 256), Image.NEAREST).save(buf, format="PNG")
    return buf.getvalue()


@router.get("/{tile_id}/preview/{which}.png", response_class=Response)
def preview(
    tile_id: int,
    which: str,
    db: Session = Depends(get_db),
    svc: InferenceService = Depends(get_inference),
):
    """RGB preview of `before` / `after`, or an index-difference heatmap (`ndvi_diff`, `ndbi_diff`, ...)."""
    import numpy as np

    tile = get_tile(db, tile_id)
    try:
        before, after = svc.load_pair(tile.before_path, tile.after_path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(404, f"imagery unavailable: {exc}")

    if which == "before":
        arr = to_rgb_uint8(before)
    elif which == "after":
        arr = to_rgb_uint8(after)
    elif which.endswith("_diff") and which[:-5] in INDEX_FUNCS:
        fn = INDEX_FUNCS[which[:-5]]
        d = np.clip((fn(after) - fn(before)) / 1.0, -1, 1)
        # diverging red (decrease) / white / blue (increase)
        pos, neg = np.clip(d, 0, 1), np.clip(-d, 0, 1)
        arr = np.stack([1 - pos, 1 - pos - neg + neg, 1 - neg], axis=-1)
        arr = np.stack([1 - pos, 1 - np.maximum(pos, neg), 1 - neg], axis=-1)
        arr = (arr * 255).astype(np.uint8)
    else:
        raise HTTPException(400, "which must be before, after or <index>_diff")
    return Response(_png(arr), media_type="image/png", headers={"Cache-Control": "max-age=3600"})
