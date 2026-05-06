from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from app.fusion.io import load_fusion_rule_config
from app.fusion.rules import late_fusion_rule
from app.inference.face_runtime import FaceRuntime
from app.inference.voice_runtime import VoiceRuntime
from app.utils.config import get_settings


@dataclass(slots=True)
class UploadedInferenceResult:
    timestamp: datetime
    face_emotion: str
    face_confidence: float
    voice_emotion: str
    voice_confidence: float
    stress_level: str
    source: str


_FACE_RUNTIME: FaceRuntime | None = None
_VOICE_RUNTIME: VoiceRuntime | None = None


def _get_face_runtime() -> FaceRuntime:
    global _FACE_RUNTIME
    if _FACE_RUNTIME is None:
        settings = get_settings()
        _FACE_RUNTIME = FaceRuntime(settings.face_model_path, settings.face_labels_path)
    return _FACE_RUNTIME


def _get_voice_runtime() -> VoiceRuntime:
    global _VOICE_RUNTIME
    if _VOICE_RUNTIME is None:
        settings = get_settings()
        _VOICE_RUNTIME = VoiceRuntime(
            settings.voice_model_path,
            settings.voice_labels_path,
            sample_rate=settings.audio_sample_rate,
            record_seconds=settings.audio_record_seconds,
        )
    return _VOICE_RUNTIME


def _decode_frame(image_bytes: bytes) -> np.ndarray:
    frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode uploaded webcam image.")
    return frame


def _predict_face_from_bytes(image_bytes: bytes):
    frame = _decode_frame(image_bytes)
    face_runtime = _get_face_runtime()
    face_crop = face_runtime.detect_and_crop(frame)
    if face_crop is None:
        raise ValueError("No clear face was detected in the uploaded webcam image.")
    return face_runtime.predict_from_face(face_crop)


def _average_face_predictions(image_samples: list[bytes]):
    face_runtime = _get_face_runtime()
    valid_predictions = []
    for image_bytes in image_samples:
        try:
            frame = _decode_frame(image_bytes)
        except ValueError:
            continue
        face_crop = face_runtime.detect_and_crop(frame)
        if face_crop is None:
            continue
        valid_predictions.append(face_runtime.predict_from_face(face_crop))

    if not valid_predictions:
        raise ValueError("No clear face was detected. Please face the camera with steady lighting and try again.")

    labels = face_runtime.labels
    averaged = {
        label: float(np.mean([prediction.probabilities.get(label, 0.0) for prediction in valid_predictions]))
        for label in labels
    }
    stabilized = _stabilize_face_probabilities(averaged)
    label = max(stabilized, key=stabilized.get)
    return type(valid_predictions[0])(
        label=label,
        confidence=float(stabilized[label]),
        probabilities=stabilized,
    )


def _stabilize_face_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Reduce single-class spikes from noisy webcam frames before fusion."""
    if not probabilities:
        return probabilities
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    top_label, top_confidence = sorted_probs[0]
    runner_up = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    margin = top_confidence - runner_up

    if top_label.lower() == "angry" and (top_confidence < 0.62 or margin < 0.18):
        probabilities = probabilities.copy()
        shift = min(probabilities[top_label] * 0.35, 0.18)
        probabilities[top_label] -= shift
        fallback_label = "neutral" if "neutral" in probabilities else sorted_probs[1][0]
        probabilities[fallback_label] += shift

    total = sum(probabilities.values()) or 1.0
    return {label: float(value / total) for label, value in probabilities.items()}


def _predict_voice_from_bytes(audio_bytes: bytes, suffix: str):
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(audio_bytes)
        temp_path = Path(handle.name)
    try:
        return _get_voice_runtime().predict_from_audio_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def analyze_uploaded_sample(
    image_bytes: bytes,
    audio_bytes: bytes,
    audio_filename: str,
    image_samples: list[bytes] | None = None,
) -> UploadedInferenceResult:
    settings = get_settings()
    face_prediction = _average_face_predictions(image_samples or [image_bytes])
    suffix = Path(audio_filename).suffix or ".webm"
    voice_prediction = _predict_voice_from_bytes(audio_bytes, suffix=suffix)
    rule_config = load_fusion_rule_config(settings.fusion_rules_path)
    fusion = late_fusion_rule(
        face_prediction.probabilities,
        voice_prediction.probabilities,
        face_mapping=rule_config.get("face_mapping"),
        voice_mapping=rule_config.get("voice_mapping"),
        thresholds=rule_config.get("thresholds"),
        weights=rule_config.get("weights"),
    )
    return UploadedInferenceResult(
        timestamp=datetime.now(timezone.utc),
        face_emotion=face_prediction.label,
        face_confidence=face_prediction.confidence,
        voice_emotion=voice_prediction.label,
        voice_confidence=voice_prediction.confidence,
        stress_level=fusion.stress_level,
        source="browser_capture",
    )
