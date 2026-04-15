from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from app.utils.io import write_json


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int], labels: list[str]) -> dict:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    accuracy = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "labels": labels,
    }


def save_confusion_matrix(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[str],
    output_path: Path,
) -> None:
    cm = confusion_matrix(list(y_true), list(y_pred))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_evaluation_report(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    labels: list[str],
    json_path: Path,
    cm_path: Path,
) -> dict:
    metrics = classification_metrics(y_true, y_pred, labels)
    write_json(json_path, metrics)
    save_confusion_matrix(y_true, y_pred, labels, cm_path)
    return metrics
