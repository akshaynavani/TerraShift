"""Pydantic request/response schemas."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from terrashift.data.bands import CHANGE_CLASSES

ModelChoice = Literal["auto", "random_forest", "resnet50", "ensemble"]


def _check_label(v: str) -> str:
    if v not in CHANGE_CLASSES:
        raise ValueError(f"label must be one of {CHANGE_CLASSES}")
    return v


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class SceneCreate(BaseModel):
    name: str
    sensor: str = "sentinel-2"
    region: str | None = None
    acquired_before: date | None = None
    acquired_after: date | None = None


class SceneOut(ORM):
    id: int
    name: str
    sensor: str
    region: str | None
    acquired_before: date | None
    acquired_after: date | None
    created_at: datetime
    tile_count: int = 0


class SceneImport(BaseModel):
    manifest_dir: str = Field(description="Directory (relative to imagery root) containing manifest.json")
    include_labels: bool = True
    limit: int | None = None


class TileCreate(BaseModel):
    tile_key: str
    before_path: str
    after_path: str
    lat: float | None = None
    lon: float | None = None
    ground_truth: str | None = None


class PredictionOut(ORM):
    id: int
    tile_id: int
    label: str
    confidence: float
    entropy: float
    probabilities: dict[str, float]
    model_name: str
    created_at: datetime


class AnnotationCreate(BaseModel):
    analyst: str = Field(min_length=1)
    label: str
    action: Literal["confirmed", "corrected", "skipped"] | None = None
    notes: str | None = None
    prediction_id: int | None = None

    _v = field_validator("label")(_check_label)


class AnnotationOut(ORM):
    id: int
    tile_id: int
    prediction_id: int | None
    analyst: str
    label: str
    action: str
    notes: str | None
    created_at: datetime


class TileOut(ORM):
    id: int
    scene_id: int
    tile_key: str
    lat: float | None
    lon: float | None
    ground_truth: str | None
    latest_prediction: PredictionOut | None = None
    latest_annotation: AnnotationOut | None = None


class TileDetail(TileOut):
    predictions: list[PredictionOut] = []
    annotations: list[AnnotationOut] = []


class PredictRequest(BaseModel):
    model: ModelChoice = "auto"


class BatchPredictResult(BaseModel):
    scene_id: int
    predicted: int
    model: str


class ModelInfo(BaseModel):
    available: list[str]
    versions: list[dict]


class StatsOut(BaseModel):
    tiles: int
    predicted: int
    reviewed: int
    corrected: int
    agreement_rate: float | None
    predicted_label_counts: dict[str, int]
    reviewed_label_counts: dict[str, int]
    confusion: dict[str, dict[str, int]]


class ExportRow(BaseModel):
    tile_id: int
    tile_key: str
    before: str
    after: str
    label: str
    analyst: str
    action: str
