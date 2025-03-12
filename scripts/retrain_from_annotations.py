"""Retrain models using analyst corrections stored in PostgreSQL.

Analyst labels (latest non-skipped annotation per tile) override any reference label;
they are merged with an optional base training manifest so the model keeps its prior
knowledge while learning from corrections. Corrected tiles can be up-weighted by
duplication (``--correction-weight``).

After training, call ``POST /models/reload`` on the API to hot-swap the new models.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from api.crud import export_labels
from api.database import SessionLocal
from api.settings import get_settings
from terrashift.config import CNNTrainConfig
from terrashift.data.io import TileRecord, read_manifest
from terrashift.training.metrics import format_report
from terrashift.training.train_cnn import train_resnet
from terrashift.training.train_rf import train_random_forest


def main() -> None:
    settings = get_settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-manifest-dir", help="Training set dir relative to the imagery root, e.g. synthetic")
    p.add_argument("--out", default=str(settings.terrashift_artifact_dir))
    p.add_argument("--model", choices=["rf", "cnn", "both"], default="rf")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--correction-weight", type=int, default=3)
    p.add_argument("--min-annotations", type=int, default=20)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    root = settings.terrashift_imagery_root
    with SessionLocal() as db:
        rows = export_labels(db)
    if len(rows) < args.min_annotations:
        raise SystemExit(f"Only {len(rows)} annotations available (< {args.min_annotations}); not retraining.")

    records: dict[str, TileRecord] = {}
    if args.base_manifest_dir:
        base = Path(args.base_manifest_dir)
        for r in read_manifest(root / base):
            records[str(base / r.before)] = TileRecord(r.tile_id, str(base / r.before), str(base / r.after), r.label)

    annotated: list[TileRecord] = []
    for row in rows:
        rec = TileRecord(row.tile_key, row.before, row.after, row.label)
        records[row.before] = rec  # analyst label wins over the base label
        if row.action == "corrected":
            annotated += [rec] * (args.correction_weight - 1)

    train_set = list(records.values()) + annotated
    print(f"Training on {len(train_set)} samples ({len(rows)} analyst labels): {dict(Counter(r.label for r in train_set))}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    if args.model in ("rf", "both"):
        _, m = train_random_forest(root, train_set, out / "random_forest.joblib")
        print(format_report(m))
        summary["random_forest"] = m
    if args.model in ("cnn", "both"):
        _, m = train_resnet(root, train_set, out / "resnet50.pt", CNNTrainConfig(epochs=args.epochs))
        print(format_report(m))
        summary["resnet50"] = m
    (out / "train_metrics.json").write_text(json.dumps(summary, indent=2))
    print("Done. Call POST /models/reload to load the new models.")


if __name__ == "__main__":
    main()
