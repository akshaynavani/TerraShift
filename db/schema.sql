-- TerraShift PostgreSQL schema: scenes, tiles, model versions, predictions and analyst annotations.

CREATE TABLE IF NOT EXISTS scenes (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    sensor          TEXT NOT NULL DEFAULT 'sentinel-2',
    region          TEXT,
    acquired_before DATE,
    acquired_after  DATE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS tiles (
    id           SERIAL PRIMARY KEY,
    scene_id     INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    tile_key     TEXT NOT NULL,
    before_path  TEXT NOT NULL,
    after_path   TEXT NOT NULL,
    lat          DOUBLE PRECISION,
    lon          DOUBLE PRECISION,
    ground_truth TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (scene_id, tile_key)
);
CREATE INDEX IF NOT EXISTS idx_tiles_scene ON tiles(scene_id);

CREATE TABLE IF NOT EXISTS model_versions (
    id            SERIAL PRIMARY KEY,
    name          TEXT NOT NULL,             -- random_forest | resnet50 | ensemble
    version       TEXT NOT NULL,
    artifact_path TEXT,
    metrics       JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    tile_id          INTEGER NOT NULL REFERENCES tiles(id) ON DELETE CASCADE,
    model_version_id INTEGER NOT NULL REFERENCES model_versions(id),
    label            TEXT NOT NULL,
    confidence       REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    entropy          REAL NOT NULL,
    probabilities    JSONB NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_predictions_tile ON predictions(tile_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_confidence ON predictions(confidence);

CREATE TABLE IF NOT EXISTS annotations (
    id            SERIAL PRIMARY KEY,
    tile_id       INTEGER NOT NULL REFERENCES tiles(id) ON DELETE CASCADE,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE SET NULL,
    analyst       TEXT NOT NULL,
    label         TEXT NOT NULL,
    action        TEXT NOT NULL CHECK (action IN ('confirmed', 'corrected', 'skipped')),
    notes         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_annotations_tile ON annotations(tile_id, created_at DESC);

-- Latest analyst label per tile, used for exporting corrected training data.
CREATE OR REPLACE VIEW latest_annotations AS
SELECT DISTINCT ON (tile_id) *
FROM annotations
WHERE action <> 'skipped'
ORDER BY tile_id, created_at DESC;
