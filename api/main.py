"""TerraShift FastAPI application."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from terrashift import __version__

from .database import Base, engine
from .routers import annotations, predictions, scenes, stats, tiles
from .settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Idempotent; db/schema.sql is the canonical DDL for production Postgres.
    Base.metadata.create_all(bind=engine)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="TerraShift API", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_settings().cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for r in (scenes, tiles, predictions, annotations, stats):
        app.include_router(r.router)

    @app.get("/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
