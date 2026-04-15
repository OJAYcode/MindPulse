from __future__ import annotations

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401
from sklearn.metrics import accuracy_score, f1_score

from app.fusion.io import save_fusion_rule_config
from app.fusion.rules import late_fusion_rule
from app.inference.face_runtime import FaceRuntime
from app.inference.voice_runtime import VoiceRuntime
from app.utils.config import get_settings
from app.utils.io import write_json
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
VALID_STRESS_LABELS = {"low", "medium", "high"}


def _load_manifest(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Paired multimodal manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required_columns = {"face_image", "audio_file", "stress_label"}
    if not rows or not required_columns.issubset(rows[0].keys()):
        raise ValueError("Manifest must contain face_image, audio_file, and stress_label columns.")
    return rows


def _resolve_sample_path(base_dir: Path, relative_path: str) -> Path:
    return (base_dir / relative_path).resolve()


def tune_fusion_rules(manifest_path: Path) -> dict:
    settings = get_settings()
    rows = _load_manifest(manifest_path)
    base_dir = manifest_path.parent

    face_runtime = FaceRuntime(settings.face_model_path, settings.face_labels_path)
    voice_runtime = VoiceRuntime(
        settings.voice_model_path,
        settings.voice_labels_path,
        sample_rate=settings.audio_sample_rate,
        record_seconds=settings.audio_record_seconds,
    )

    samples = []
    for row in rows:
        stress_label = row["stress_label"].strip().lower()
        if stress_label not in VALID_STRESS_LABELS:
            raise ValueError(f"Unsupported stress label '{row['stress_label']}'. Expected low, medium, or high.")
        face_path = _resolve_sample_path(base_dir, row["face_image"])
        audio_path = _resolve_sample_path(base_dir, row["audio_file"])
        face_prediction = face_runtime.predict_from_image_file(face_path)
        voice_prediction = voice_runtime.predict_from_audio_file(audio_path)
        samples.append(
            {
                "true_label": stress_label,
                "face_probabilities": face_prediction.probabilities,
                "voice_probabilities": voice_prediction.probabilities,
            }
        )

    candidate_face_weights = [round(step / 10, 1) for step in range(0, 11)]
    candidate_medium_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    candidate_high_thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]

    best_result: dict | None = None
    for face_weight in candidate_face_weights:
        voice_weight = round(1.0 - face_weight, 1)
        for medium_threshold in candidate_medium_thresholds:
            for high_threshold in candidate_high_thresholds:
                if high_threshold <= medium_threshold:
                    continue
                predictions = []
                truths = []
                for sample in samples:
                    result = late_fusion_rule(
                        sample["face_probabilities"],
                        sample["voice_probabilities"],
                        thresholds={"low": 0.0, "medium": medium_threshold, "high": high_threshold},
                        weights={"face": face_weight, "voice": voice_weight},
                    )
                    predictions.append(result.stress_level)
                    truths.append(sample["true_label"])
                score = f1_score(truths, predictions, average="weighted")
                accuracy = accuracy_score(truths, predictions)
                candidate = {
                    "weights": {"face": face_weight, "voice": voice_weight},
                    "thresholds": {"low": 0.0, "medium": medium_threshold, "high": high_threshold},
                    "weighted_f1": float(score),
                    "accuracy": float(accuracy),
                    "samples_used": len(samples),
                }
                if best_result is None or candidate["weighted_f1"] > best_result["weighted_f1"] or (
                    candidate["weighted_f1"] == best_result["weighted_f1"]
                    and candidate["accuracy"] > best_result["accuracy"]
                ):
                    best_result = candidate

    if best_result is None:
        raise RuntimeError("Could not tune fusion rules from the provided manifest.")

    output_config = {
        "weights": best_result["weights"],
        "thresholds": best_result["thresholds"],
        "source": "paired_multimodal_tuning",
        "manifest_path": str(manifest_path.resolve()),
        "samples_used": best_result["samples_used"],
    }
    save_fusion_rule_config(settings.fusion_rules_path, output_config)
    write_json(settings.fusion_rules_path.with_suffix(".metrics.json"), best_result)
    LOGGER.info("Saved tuned fusion rule config to %s", settings.fusion_rules_path)
    return best_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune rule-based fusion weights and thresholds on paired multimodal data.")
    parser.add_argument(
        "--manifest",
        default="data/raw/multimodal/manifest.csv",
        help="CSV manifest with face_image,audio_file,stress_label columns.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tune_fusion_rules(Path(args.manifest).resolve())
