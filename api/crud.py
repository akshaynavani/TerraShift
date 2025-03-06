"""Database operations used by the routers."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from terrashift import __version__
from terrashift.models.predictor import Prediction as ModelPrediction

from . import models as m
from . import schemas as s


# ---- scenes -----------------------------------------------------------------
def create_scene(db: Session, data: s.SceneCreate) -> m.Scene:
    scene = m.Scene(**data.model_dump())
    db.add(scene)
    db.commit()
    db.refresh(scene)
    return scene


def scene_out(db: Session, scene: m.Scene) -> s.SceneOut:
    count = db.scalar(select(func.count(m.Tile.id)).where(m.Tile.scene_id == scene.id)) or 0
    return s.SceneOut.model_validate(scene).model_copy(update={"tile_count": count})


def list_scenes(db: Session) -> list[m.Scene]:
    return list(db.scalars(select(m.Scene).order_by(m.Scene.id.desc())))


def add_tiles(db: Session, scene_id: int, tiles: list[s.TileCreate]) -> int:
    existing = set(db.scalars(select(m.Tile.tile_key).where(m.Tile.scene_id == scene_id)))
    added = 0
    for t in tiles:
        if t.tile_key in existing:
            continue
        db.add(m.Tile(scene_id=scene_id, **t.model_dump()))
        existing.add(t.tile_key)
        added += 1
    db.commit()
    return added


def import_manifest(db: Session, scene_id: int, imagery_root: Path, req: s.SceneImport) -> int:
    base = Path(req.manifest_dir)
    with open(imagery_root / base / "manifest.json") as f:
        rows = json.load(f)
    if req.limit:
        rows = rows[: req.limit]
    tiles = [
        s.TileCreate(
            tile_key=r["tile_id"],
            before_path=str(base / r["before"]),
            after_path=str(base / r["after"]),
            lat=r.get("lat"),
            lon=r.get("lon"),
            ground_truth=r.get("label") if req.include_labels else None,
        )
        for r in rows
    ]
    return add_tiles(db, scene_id, tiles)


# ---- tiles ------------------------------------------------------------------
def prediction_out(p: m.Prediction) -> s.PredictionOut:
    return s.PredictionOut(
        id=p.id,
        tile_id=p.tile_id,
        label=p.label,
        confidence=p.confidence,
        entropy=p.entropy,
        probabilities=p.probabilities,
        model_name=p.model_version.name,
        created_at=p.created_at,
    )


def tile_out(tile: m.Tile, detail: bool = False) -> s.TileOut:
    latest_p = prediction_out(tile.predictions[0]) if tile.predictions else None
    latest_a = s.AnnotationOut.model_validate(tile.annotations[0]) if tile.annotations else None
    base = dict(
        id=tile.id,
        scene_id=tile.scene_id,
        tile_key=tile.tile_key,
        lat=tile.lat,
        lon=tile.lon,
        ground_truth=tile.ground_truth,
        latest_prediction=latest_p,
        latest_annotation=latest_a,
    )
    if not detail:
        return s.TileOut(**base)
    return s.TileDetail(
        **base,
        predictions=[prediction_out(p) for p in tile.predictions],
        annotations=[s.AnnotationOut.model_validate(a) for a in tile.annotations],
    )


def list_tiles(db: Session, scene_id: int | None, offset: int, limit: int) -> list[m.Tile]:
    q = select(m.Tile).order_by(m.Tile.id)
    if scene_id is not None:
        q = q.where(m.Tile.scene_id == scene_id)
    return list(db.scalars(q.offset(offset).limit(limit)))


# ---- model versions & predictions ---------------------------------------------
def get_or_create_model_version(db: Session, name: str, artifact_dir: Path) -> m.ModelVersion:
    version = __version__
    mv = db.scalar(select(m.ModelVersion).where(m.ModelVersion.name == name, m.ModelVersion.version == version))
    if mv is None:
        metrics: dict = {}
        metrics_file = artifact_dir / "train_metrics.json"
        if metrics_file.exists():
            all_metrics = json.loads(metrics_file.read_text())
            metrics = {k: v for k, v in all_metrics.get(name, {}).items() if k in ("accuracy", "macro_f1", "change_detection_f1")}
        mv = m.ModelVersion(name=name, version=version, artifact_path=str(artifact_dir), metrics=metrics)
        db.add(mv)
        db.flush()
    return mv


def save_prediction(db: Session, tile: m.Tile, pred: ModelPrediction, artifact_dir: Path, commit: bool = True) -> m.Prediction:
    mv = get_or_create_model_version(db, pred.model, artifact_dir)
    row = m.Prediction(
        tile_id=tile.id,
        model_version_id=mv.id,
        label=pred.label,
        confidence=pred.confidence,
        entropy=pred.entropy,
        probabilities=pred.probabilities,
    )
    row.model_version = mv
    db.add(row)
    if commit:
        db.commit()
        db.refresh(row)
    return row


def list_model_versions(db: Session) -> list[dict]:
    rows = db.scalars(select(m.ModelVersion).order_by(m.ModelVersion.id.desc()))
    return [{"id": r.id, "name": r.name, "version": r.version, "metrics": r.metrics, "is_active": r.is_active} for r in rows]


# ---- review queue -------------------------------------------------------------
def _latest_prediction_subq():
    return (
        select(m.Prediction.tile_id, func.max(m.Prediction.id).label("pid"))
        .group_by(m.Prediction.tile_id)
        .subquery()
    )


def review_queue(db: Session, scene_id: int | None, limit: int, strategy: str) -> list[m.Tile]:
    """Unreviewed tiles with predictions, ordered by an active-learning strategy.

    - ``uncertainty``: lowest confidence first
    - ``entropy``: highest predictive entropy first
    - ``changes``: predicted changes (non no_change) first, then lowest confidence
    """
    latest = _latest_prediction_subq()
    reviewed = select(m.Annotation.tile_id).where(m.Annotation.action != "skipped")
    q = (
        select(m.Tile)
        .join(latest, latest.c.tile_id == m.Tile.id)
        .join(m.Prediction, m.Prediction.id == latest.c.pid)
        .where(m.Tile.id.not_in(reviewed))
    )
    if scene_id is not None:
        q = q.where(m.Tile.scene_id == scene_id)
    if strategy == "entropy":
        q = q.order_by(m.Prediction.entropy.desc())
    elif strategy == "changes":
        q = q.order_by((m.Prediction.label == "no_change").asc(), m.Prediction.confidence.asc())
    else:
        q = q.order_by(m.Prediction.confidence.asc())
    return list(db.scalars(q.limit(limit)))


# ---- annotations --------------------------------------------------------------
def create_annotation(db: Session, tile: m.Tile, data: s.AnnotationCreate) -> m.Annotation:
    pred = None
    if data.prediction_id is not None:
        pred = db.get(m.Prediction, data.prediction_id)
        if pred is None or pred.tile_id != tile.id:
            raise ValueError("prediction_id does not belong to this tile")
    elif tile.predictions:
        pred = tile.predictions[0]
    action = data.action or ("confirmed" if pred is not None and pred.label == data.label else "corrected")
    ann = m.Annotation(
        tile_id=tile.id,
        prediction_id=pred.id if pred else None,
        analyst=data.analyst,
        label=data.label,
        action=action,
        notes=data.notes,
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)
    return ann


def latest_annotations(db: Session) -> list[m.Annotation]:
    sub = (
        select(m.Annotation.tile_id, func.max(m.Annotation.id).label("aid"))
        .where(m.Annotation.action != "skipped")
        .group_by(m.Annotation.tile_id)
        .subquery()
    )
    return list(db.scalars(select(m.Annotation).join(sub, sub.c.aid == m.Annotation.id)))


def export_labels(db: Session) -> list[s.ExportRow]:
    return [
        s.ExportRow(
            tile_id=a.tile_id,
            tile_key=a.tile.tile_key,
            before=a.tile.before_path,
            after=a.tile.after_path,
            label=a.label,
            analyst=a.analyst,
            action=a.action,
        )
        for a in latest_annotations(db)
    ]


# ---- stats ------------------------------------------------------------------
def stats(db: Session, scene_id: int | None = None) -> s.StatsOut:
    tile_q = select(m.Tile.id)
    if scene_id is not None:
        tile_q = tile_q.where(m.Tile.scene_id == scene_id)
    tile_ids = set(db.scalars(tile_q))

    latest = _latest_prediction_subq()
    preds = {
        p.tile_id: p
        for p in db.scalars(select(m.Prediction).join(latest, latest.c.pid == m.Prediction.id))
        if p.tile_id in tile_ids
    }
    anns = [a for a in latest_annotations(db) if a.tile_id in tile_ids]

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    agree = compared = 0
    for a in anns:
        if a.prediction_id is not None:
            p = db.get(m.Prediction, a.prediction_id)
            if p is not None:
                confusion[a.label][p.label] += 1
                compared += 1
                agree += int(p.label == a.label)

    return s.StatsOut(
        tiles=len(tile_ids),
        predicted=len(preds),
        reviewed=len(anns),
        corrected=sum(a.action == "corrected" for a in anns),
        agreement_rate=(agree / compared) if compared else None,
        predicted_label_counts=dict(Counter(p.label for p in preds.values())),
        reviewed_label_counts=dict(Counter(a.label for a in anns)),
        confusion={k: dict(v) for k, v in confusion.items()},
    )
