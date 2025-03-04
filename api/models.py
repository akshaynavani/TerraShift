"""SQLAlchemy ORM models mirroring db/schema.sql."""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")


class Scene(Base):
    __tablename__ = "scenes"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String, unique=True)
    sensor: Mapped[str] = mapped_column(String, default="sentinel-2")
    region: Mapped[str | None] = mapped_column(String, nullable=True)
    acquired_before: Mapped[date | None] = mapped_column(Date, nullable=True)
    acquired_after: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tiles: Mapped[list["Tile"]] = relationship(back_populates="scene", cascade="all, delete-orphan")


class Tile(Base):
    __tablename__ = "tiles"
    __table_args__ = (UniqueConstraint("scene_id", "tile_key"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scene_id: Mapped[int] = mapped_column(ForeignKey("scenes.id", ondelete="CASCADE"), index=True)
    tile_key: Mapped[str] = mapped_column(String)
    before_path: Mapped[str] = mapped_column(Text)
    after_path: Mapped[str] = mapped_column(Text)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_truth: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scene: Mapped[Scene] = relationship(back_populates="tiles")
    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="tile", cascade="all, delete-orphan", order_by="Prediction.id.desc()"
    )
    annotations: Mapped[list["Annotation"]] = relationship(
        back_populates="tile", cascade="all, delete-orphan", order_by="Annotation.id.desc()"
    )


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("name", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    version: Mapped[str] = mapped_column(String)
    artifact_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("tiles.id", ondelete="CASCADE"), index=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    label: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(Float, index=True)
    entropy: Mapped[float] = mapped_column(Float)
    probabilities: Mapped[dict] = mapped_column(JSONType)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tile: Mapped[Tile] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship()


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True)
    tile_id: Mapped[int] = mapped_column(ForeignKey("tiles.id", ondelete="CASCADE"), index=True)
    prediction_id: Mapped[int | None] = mapped_column(ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True)
    analyst: Mapped[str] = mapped_column(String)
    label: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)  # confirmed | corrected | skipped
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    tile: Mapped[Tile] = relationship(back_populates="annotations")
