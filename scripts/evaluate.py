"""Evaluate trained models on a labelled dataset (e.g. a held-out test set)."""
from __future__ import annotations

import argparse
import json

import numpy as np
from pathlib import Path

from terrashift.config import ARTIFACT_DIR
from terrashift.data.bands import class_index
from terrashift.data.io import load_pair, read_manifest
from terrashift.models.predictor import ChangePredictor
from terrashift.training.metrics import evaluate, format_report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    p.add_argument("--json", help="Optional path to write metrics JSON")
    args = p.parse_args()

    predictor = ChangePredictor.from_artifacts(args.artifacts)
    records = [r for r in read_manifest(args.data) if r.label]
    pairs = [load_pair(Path(args.data), r) for r in records]
    y_true = np.array([class_index(r.label) for r in records])

    results = {}
    for model in predictor.available:
        y_pred = np.array([class_index(predictor.predict(b, a, model).label) for b, a in pairs])
        results[model] = evaluate(y_true, y_pred)
        print(f"== {model} ==\n{format_report(results[model])}\n")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
