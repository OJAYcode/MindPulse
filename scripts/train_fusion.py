from __future__ import annotations

import argparse
from collections import Counter
from itertools import product

import _bootstrap  # noqa: F401
import numpy as np

from app.fusion.learned import train_learned_fusion
from app.fusion.rules import late_fusion_rule
from app.utils.config import get_settings
from app.utils.io import read_json, write_json
from app.utils.logging import get_logger
from training.face.data import load_face_dataset
from training.face.model import load_face_model
from training.voice.data import load_voice_dataset
from training.voice.model import load_voice_model


LOGGER = get_logger(__name__)
STRESS_LABELS = ["low", "medium", "high"]


def _stress_index(label: str) -> int:
    return STRESS_LABELS.index(label)


def _build_fusion_training_set(
    face_probabilities: np.ndarray,
    face_labels: list[str],
    voice_probabilities: np.ndarray,
    voice_labels: list[str],
    max_pairs: int = 1500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    pairs = list(product(range(len(face_probabilities)), range(len(voice_probabilities))))
    if len(pairs) > max_pairs:
        sampled_indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[index] for index in sampled_indices]

    face_rows = []
    voice_rows = []
    stress_targets = []

    for face_index, voice_index in pairs:
        face_row = face_probabilities[face_index]
        voice_row = voice_probabilities[voice_index]
        fusion = late_fusion_rule(
            {face_labels[i]: float(face_row[i]) for i in range(len(face_labels))},
            {voice_labels[i]: float(voice_row[i]) for i in range(len(voice_labels))},
        )
        face_rows.append(face_row)
        voice_rows.append(voice_row)
        stress_targets.append(_stress_index(fusion.stress_level))

    return (
        np.asarray(face_rows, dtype="float32"),
        np.asarray(voice_rows, dtype="float32"),
        np.asarray(stress_targets, dtype="int32"),
    )


def train_fusion_model(
    demo_mode: bool = False,
    max_pairs: int = 1500,
    face_max_files_per_class: int | None = 750,
    voice_max_files_per_class: int | None = 192,
) -> dict:
    settings = get_settings()
    face_labels = read_json(settings.face_labels_path, default=[])
    voice_labels = read_json(settings.voice_labels_path, default=[])

    if not face_labels or not voice_labels:
        raise FileNotFoundError("Face and voice label files must exist before training fusion.")

    face_dataset = load_face_dataset(
        settings.face_data_dir,
        demo_mode=demo_mode,
        max_files_per_class=face_max_files_per_class,
    )
    voice_dataset = load_voice_dataset(
        settings.voice_data_dir,
        sample_rate=settings.audio_sample_rate,
        demo_mode=demo_mode,
        max_files_per_class=voice_max_files_per_class,
    )
    face_model = load_face_model(settings.face_model_path, face_dataset.input_shape, len(face_labels))
    voice_model = load_voice_model(settings.voice_model_path, input_shape=(64, 184, 1), num_classes=len(voice_labels))
    face_probabilities = face_model.predict(face_dataset.test_data, verbose=0)
    voice_probabilities = voice_model.predict(voice_dataset.x_test, verbose=0)

    face_rows, voice_rows, stress_targets = _build_fusion_training_set(
        face_probabilities,
        face_labels,
        voice_probabilities,
        voice_labels,
        max_pairs=max_pairs,
    )
    model = train_learned_fusion(face_rows, voice_rows, stress_targets, settings.fusion_model_path)
    target_distribution = Counter(STRESS_LABELS[target] for target in stress_targets.tolist())

    metadata = {
        "fusion_model_path": str(settings.fusion_model_path),
        "stress_labels": STRESS_LABELS,
        "training_pairs": int(len(stress_targets)),
        "demo_mode": demo_mode,
        "face_classes": face_labels,
        "voice_classes": voice_labels,
        "classes_seen_by_model": [str(item) for item in model.classes_],
        "training_target_note": "Targets are generated from the default rule-based late fusion mapping. This logistic model is an optional experimental add-on, not the primary presentation-ready fusion path.",
        "target_distribution": {label: int(target_distribution.get(label, 0)) for label in STRESS_LABELS},
        "face_max_files_per_class": face_max_files_per_class,
        "voice_max_files_per_class": voice_max_files_per_class,
    }
    write_json(settings.fusion_model_path.with_suffix(".json"), metadata)
    LOGGER.info("Saved learned fusion model to %s", settings.fusion_model_path)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train optional learned logistic-regression fusion model.")
    parser.add_argument("--demo", action="store_true", help="Use synthetic branch datasets.")
    parser.add_argument("--max-pairs", type=int, default=1500, help="Maximum paired branch predictions to use.")
    parser.add_argument("--face-max-files-per-class", type=int, default=750)
    parser.add_argument("--voice-max-files-per-class", type=int, default=192)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    face_cap = None if not args.face_max_files_per_class or args.face_max_files_per_class <= 0 else args.face_max_files_per_class
    voice_cap = None if not args.voice_max_files_per_class or args.voice_max_files_per_class <= 0 else args.voice_max_files_per_class
    train_fusion_model(
        demo_mode=args.demo,
        max_pairs=args.max_pairs,
        face_max_files_per_class=face_cap,
        voice_max_files_per_class=voice_cap,
    )
