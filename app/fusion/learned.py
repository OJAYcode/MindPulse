from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


def train_learned_fusion(
    face_probabilities: np.ndarray,
    voice_probabilities: np.ndarray,
    stress_labels: np.ndarray,
    output_path: Path,
) -> LogisticRegression:
    features = np.concatenate([face_probabilities, voice_probabilities], axis=1)
    model = LogisticRegression(max_iter=1000)
    model.fit(features, stress_labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(model, handle)
    return model


def load_learned_fusion(path: Path) -> LogisticRegression | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_learned_fusion(
    model: LogisticRegression,
    face_probabilities: np.ndarray,
    voice_probabilities: np.ndarray,
) -> tuple[str, float]:
    features = np.concatenate([face_probabilities, voice_probabilities], axis=1)
    probabilities = model.predict_proba(features)[0]
    label_index = int(np.argmax(probabilities))
    return str(model.classes_[label_index]), float(probabilities[label_index])
