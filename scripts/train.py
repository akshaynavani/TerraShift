"""Train the Random Forest and/or ResNet-50 change classifiers.

Example:
    python -m scripts.train --data data/synthetic --model both --epochs 15
"""
from __future__ import annotations

import argparse
import json
import logging

from terrashift.config import ARTIFACT_DIR, DATA_DIR, CNNTrainConfig, RFTrainConfig
from terrashift.data.io import read_manifest
from terrashift.training.metrics import format_report
from terrashift.training.train_cnn import train_resnet
from terrashift.training.train_rf import train_random_forest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(DATA_DIR / "synthetic"))
    p.add_argument("--out", default=str(ARTIFACT_DIR))
    p.add_argument("--model", choices=["rf", "cnn", "both"], default="both")
    p.add_argument("--epochs", type=int, default=CNNTrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=CNNTrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=CNNTrainConfig.lr)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--n-estimators", type=int, default=RFTrainConfig.n_estimators)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = read_manifest(args.data)
    summary = {}
    if args.model in ("rf", "both"):
        _, m = train_random_forest(args.data, records, f"{args.out}/random_forest.joblib", RFTrainConfig(n_estimators=args.n_estimators))
        print("== Random Forest ==\n" + format_report(m))
        summary["random_forest"] = m
    if args.model in ("cnn", "both"):
        cfg = CNNTrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, pretrained=not args.no_pretrained)
        _, m = train_resnet(args.data, records, f"{args.out}/resnet50.pt", cfg)
        print("== ResNet-50 ==\n" + format_report(m))
        summary["resnet50"] = m

    with open(f"{args.out}/train_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
