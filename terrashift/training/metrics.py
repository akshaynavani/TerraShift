"""Evaluation metrics for change classification."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from ..data.bands import CHANGE_CLASSES


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = list(range(len(CHANGE_CLASSES)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    # Binary "any change" view: everything except no_change counts as change.
    change_true, change_pred = y_true != 0, y_pred != 0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "change_detection_f1": float(f1_score(change_true, change_pred, zero_division=0)),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(CHANGE_CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_report(metrics: dict) -> str:
    lines = [
        f"accuracy={metrics['accuracy']:.4f}  macro_f1={metrics['macro_f1']:.4f}  "
        f"change_f1={metrics['change_detection_f1']:.4f}",
        f"{'class':<22}{'prec':>8}{'rec':>8}{'f1':>8}{'n':>6}",
    ]
    for c, m in metrics["per_class"].items():
        lines.append(f"{c:<22}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['support']:>6}")
    lines.append("confusion matrix (rows=true, cols=pred):")
    for row in metrics["confusion_matrix"]:
        lines.append("  " + " ".join(f"{v:>5}" for v in row))
    return "\n".join(lines)
