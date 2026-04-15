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


def _predict_face_from_bytes(image_bytes: bytes):
    frame_array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode uploaded webcam image.")
    face_runtime = _get_face_runtime()
    face_crop = face_runtime.detect_and_crop(frame)
    if face_crop is None:
        face_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_crop = cv2.resize(face_crop, (224, 224)).astype("float32")
    return face_runtime.predict_from_face(face_crop)


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
) -> UploadedInferenceResult:
    settings = get_settings()
    face_prediction = _predict_face_from_bytes(image_bytes)
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
