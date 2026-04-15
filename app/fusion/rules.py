from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.fusion.config import (
    DEFAULT_FACE_STRESS_MAP,
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_STRESS_THRESHOLDS,
    DEFAULT_VOICE_STRESS_MAP,
)


@dataclass(slots=True)
class FusionResult:
    stress_level: str
    stress_score: float
    face_weighted_score: float
    voice_weighted_score: float


def _expected_stress(probabilities: dict[str, float], mapping: dict[str, float]) -> float:
    if not probabilities:
        return 0.5
    total = 0.0
    for label, probability in probabilities.items():
        total += float(probability) * float(mapping.get(label.lower(), 0.5))
    top_label, top_probability = max(probabilities.items(), key=lambda item: float(item[1]))
    dominant_score = float(mapping.get(top_label.lower(), 0.5))
    confidence_adjusted = total + 0.35 * float(top_probability) * (dominant_score - total)
    return float(np.clip(confidence_adjusted, 0.0, 1.0))


def _score_to_label(score: float, thresholds: dict[str, float]) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["medium"]:
        return "medium"
    return "low"


def late_fusion_rule(
    face_probabilities: dict[str, float],
    voice_probabilities: dict[str, float],
    face_mapping: dict[str, float] | None = None,
    voice_mapping: dict[str, float] | None = None,
    thresholds: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> FusionResult:
    face_mapping = face_mapping or DEFAULT_FACE_STRESS_MAP
    voice_mapping = voice_mapping or DEFAULT_VOICE_STRESS_MAP
    thresholds = thresholds or DEFAULT_STRESS_THRESHOLDS
    weights = weights or DEFAULT_FUSION_WEIGHTS

    face_score = _expected_stress(face_probabilities, face_mapping)
    voice_score = _expected_stress(voice_probabilities, voice_mapping)
    face_weight = float(weights.get("face", 0.5))
    voice_weight = float(weights.get("voice", 0.5))
    total_weight = face_weight + voice_weight
    if total_weight <= 0:
        face_weight = 0.5
        voice_weight = 0.5
        total_weight = 1.0
    normalized_face_weight = face_weight / total_weight
    normalized_voice_weight = voice_weight / total_weight
    combined_score = float(
        np.clip(
            normalized_face_weight * face_score + normalized_voice_weight * voice_score,
            0.0,
            1.0,
        )
    )

    return FusionResult(
        stress_level=_score_to_label(combined_score, thresholds),
        stress_score=combined_score,
        face_weighted_score=face_score,
        voice_weighted_score=voice_score,
    )
