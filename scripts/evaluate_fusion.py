from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import _bootstrap  # noqa: F401
import numpy as np

from app.fusion.io import load_fusion_rule_config
from app.fusion.learned import load_learned_fusion, predict_learned_fusion
from app.fusion.rules import late_fusion_rule
from app.inference.face_runtime import FaceRuntime
from app.inference.voice_runtime import VoiceRuntime
from app.utils.config import get_settings
from app.utils.io import write_json
from app.utils.logging import get_logger
from app.utils.metrics import classification_metrics, save_confusion_matrix
from training.face.data import IMAGE_EXTENSIONS
from training.voice.data import AUDIO_EXTENSIONS


LOGGER = get_logger(__name__)
STRESS_LABELS = ["low", "medium", "high"]


@dataclass(slots=True)
class FusionEvalSample:
    sample_id: str
    face_path: Path
    voice_path: Path
    label: str


def _find_first_file(directory: Path, extensions: set[str]) -> Path | None:
    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            return file_path
    return None


def _load_label(sample_dir: Path) -> str | None:
    label_path = sample_dir / "label.txt"
    if not label_path.exists():
        return None
    label = label_path.read_text(encoding="utf-8").strip().lower()
    return label if label in STRESS_LABELS else None


def discover_fusion_eval_samples(data_dir: Path) -> list[FusionEvalSample]:
    if not data_dir.exists():
        return []

    samples: list[FusionEvalSample] = []
    for sample_dir in sorted(item for item in data_dir.iterdir() if item.is_dir()):
        label = _load_label(sample_dir)
        face_path = _find_first_file(sample_dir, IMAGE_EXTENSIONS)
        voice_path = _find_first_file(sample_dir, AUDIO_EXTENSIONS)
        if label is None or face_path is None or voice_path is None:
            LOGGER.warning(
                "Skipping %s. Expected one face image, one voice file, and label.txt containing low, medium, or high.",
                sample_dir,
            )
            continue
        samples.append(
            FusionEvalSample(
                sample_id=sample_dir.name,
                face_path=face_path,
                voice_path=voice_path,
                label=label,
            )
        )
    return samples


def evaluate_fusion(
    data_dir: Path,
    use_learned_fusion: bool = False,
    output_json: Path | None = None,
    output_confusion_matrix: Path | None = None,
) -> dict:
    settings = get_settings()
    samples = discover_fusion_eval_samples(data_dir)
    if not samples:
        raise RuntimeError(
            f"No valid paired fusion samples found in {data_dir}. "
            "Create sample folders with a face image, voice audio, and label.txt."
        )

    face_runtime = FaceRuntime(settings.face_model_path, settings.face_labels_path)
    voice_runtime = VoiceRuntime(
        settings.voice_model_path,
        settings.voice_labels_path,
        sample_rate=settings.audio_sample_rate,
        record_seconds=settings.audio_record_seconds,
    )
    rule_config = load_fusion_rule_config(settings.fusion_rules_path)
    learned_model = load_learned_fusion(settings.fusion_model_path) if use_learned_fusion else None

    y_true: list[int] = []
    y_pred: list[int] = []
    predictions: list[dict] = []
    skipped: list[dict] = []

    for sample in samples:
        try:
            face_prediction = face_runtime.predict_from_image_file(sample.face_path)
            voice_prediction = voice_runtime.predict_from_audio_file(sample.voice_path)

            if learned_model is not None:
                predicted_label, fusion_confidence = predict_learned_fusion(
                    learned_model,
                    face_prediction.probabilities,
                    voice_prediction.probabilities,
                    STRESS_LABELS,
                )
                fusion_score = float(fusion_confidence)
                fusion_method = "learned_logistic"
            else:
                fusion = late_fusion_rule(
                    face_prediction.probabilities,
                    voice_prediction.probabilities,
                    face_mapping=rule_config.get("face_mapping"),
                    voice_mapping=rule_config.get("voice_mapping"),
                    thresholds=rule_config.get("thresholds"),
                    weights=rule_config.get("weights"),
                )
                predicted_label = fusion.stress_level
                fusion_score = fusion.stress_score
                fusion_method = "rule_based"

            true_index = STRESS_LABELS.index(sample.label)
            predicted_index = STRESS_LABELS.index(predicted_label)
            y_true.append(true_index)
            y_pred.append(predicted_index)
            predictions.append(
                {
                    "sample_id": sample.sample_id,
                    "true_label": sample.label,
                    "predicted_label": predicted_label,
                    "correct": sample.label == predicted_label,
                    "fusion_method": fusion_method,
                    "fusion_score": fusion_score,
                    "face_emotion": face_prediction.label,
                    "face_confidence": face_prediction.confidence,
                    "voice_emotion": voice_prediction.label,
                    "voice_confidence": voice_prediction.confidence,
                    "face_path": str(sample.face_path),
                    "voice_path": str(sample.voice_path),
                }
            )
        except Exception as exc:
            skipped.append({"sample_id": sample.sample_id, "reason": str(exc)})
            LOGGER.warning("Skipping fusion eval sample %s: %s", sample.sample_id, exc)

    if not y_true:
        raise RuntimeError("All paired fusion samples were skipped. Check image/audio quality and labels.")

    metrics = classification_metrics(y_true, y_pred, STRESS_LABELS)
    output_json = output_json or (settings.face_model_path.parent / "fusion_evaluation.json")
    output_confusion_matrix = output_confusion_matrix or (
        settings.face_model_path.parent / "fusion_confusion_matrix.png"
    )
    save_confusion_matrix(y_true, y_pred, STRESS_LABELS, output_confusion_matrix)
    report = {
        **metrics,
        "sample_count": len(y_true),
        "discovered_samples": len(samples),
        "skipped_count": len(skipped),
        "skipped": skipped,
        "fusion_method": "learned_logistic" if use_learned_fusion else "rule_based",
        "data_dir": str(data_dir),
        "predictions": predictions,
    }
    write_json(output_json, report)
    LOGGER.info("Fusion evaluation report: %s", {key: report[key] for key in ["accuracy", "precision", "recall", "f1"]})
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate true multimodal fusion accuracy on paired samples.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/fusion_eval"))
    parser.add_argument("--use-learned-fusion", action="store_true", help="Evaluate optional learned fusion instead of rules.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-confusion-matrix", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        evaluate_fusion(
            data_dir=args.data_dir,
            use_learned_fusion=args.use_learned_fusion,
            output_json=args.output_json,
            output_confusion_matrix=args.output_confusion_matrix,
        )
    except RuntimeError as exc:
        print(f"Fusion evaluation could not run: {exc}", file=sys.stderr)
        sys.exit(1)
