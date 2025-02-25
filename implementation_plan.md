# TerraShift — Implementation Plan

Satellite imagery land-use change detection proof-of-concept.
Classifies land-use change between two co-registered multispectral acquisitions
(before / after) of the same tile, using:

- a **PyTorch ResNet-50** adapted to bi-temporal multispectral input (early fusion of both dates), and
- a **scikit-learn Random Forest** baseline over hand-crafted spectral-index features,

served through a **FastAPI** REST API backed by **PostgreSQL** (scenes, tiles, predictions,
analyst annotations), with a **React** dashboard for analysts to review predictions and submit
correction labels that feed back into retraining.

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Imagery format | Per-tile `.npy` arrays `(bands, H, W)` for each date, plus optional GeoTIFF read via `rasterio` | Keeps the PoC dependency-light; rasterio optional |
| Bands | Sentinel-2-like 6 bands: B, G, R, NIR, SWIR1, SWIR2 | Enough for NDVI / NDWI / NDBI indices |
| Change classes | `no_change`, `vegetation_loss`, `urban_expansion`, `water_change`, `agricultural_change` | Common land-use change taxonomy |
| Deep model | ResNet-50 with first conv widened to `2 * bands` channels (early fusion), ImageNet weights averaged into extra channels | Simple, strong baseline for bi-temporal change classification |
| Classical model | Random Forest over per-band stats + spectral-index deltas | Fast, interpretable, works on small data |
| Data for dev | Synthetic generator producing plausible multispectral pairs per class | Lets the whole pipeline run with no downloads |
| DB access | SQLAlchemy 2.0 ORM + raw SQL schema file | Explicit schema for review; ORM for the API |
| Dashboard | React + Vite + TypeScript, plain fetch client | Minimal toolchain |

## Phases

### Phase 1 — Package scaffold & data layer
Deliverables: Python package layout, config, band/class definitions, spectral indices,
tile I/O, synthetic bi-temporal data generator, PyTorch dataset with augmentation.
Files: `pyproject.toml`, `requirements.txt`, `terrashift/__init__.py`, `terrashift/config.py`,
`terrashift/data/{bands.py,indices.py,io.py,synthetic.py,dataset.py}`, `scripts/generate_synthetic.py`.

### Phase 2 — Models, training & evaluation
Deliverables: spectral feature extractor, Random Forest pipeline (train/save/load),
bi-temporal ResNet-50, training loop with early stopping, evaluation metrics (per-class
precision/recall/F1, confusion matrix), unified predictor interface, CLI scripts.
Files: `terrashift/features.py`, `terrashift/models/{random_forest.py,resnet.py,predictor.py}`,
`terrashift/training/{train_rf.py,train_cnn.py,metrics.py}`, `scripts/train.py`, `scripts/evaluate.py`.

### Phase 3 — PostgreSQL schema & FastAPI backend
Deliverables: SQL schema (scenes, tiles, model_versions, predictions, annotations),
SQLAlchemy models, Pydantic schemas, CRUD/service layer, REST endpoints for scenes, tiles,
running inference, review queue, annotations, stats, and export of corrected labels.
Files: `db/schema.sql`, `api/{main.py,database.py,models.py,schemas.py,crud.py,settings.py,inference.py}`,
`api/routers/{scenes.py,tiles.py,predictions.py,annotations.py,stats.py}`, `.env.example`.

### Phase 4 — React review dashboard
Deliverables: Vite + React + TS app with review queue (lowest-confidence first), tile detail
with before/after RGB previews and class probabilities, correction form, stats page.
Files: `dashboard/{package.json,vite.config.ts,tsconfig.json,index.html}`,
`dashboard/src/{main.tsx,App.tsx,api.ts,types.ts,styles.css}`,
`dashboard/src/components/{ReviewQueue.tsx,TileDetail.tsx,CorrectionForm.tsx,StatsPanel.tsx,ProbabilityBars.tsx}`.

### Phase 5 — Feedback loop, tests & deployment
Deliverables: retraining from analyst corrections, seed script, pytest suite (indices,
features, synthetic data, models, API with SQLite override), Dockerfiles and docker-compose
(Postgres + API + dashboard).
Files: `scripts/{retrain_from_annotations.py,seed_db.py}`, `tests/*`, `Dockerfile.api`,
`dashboard/Dockerfile`, `docker-compose.yml`, `Makefile`.

After phases: `.gitignore`, `README.md` (architecture diagram, run & test instructions).
