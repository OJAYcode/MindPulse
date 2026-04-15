from __future__ import annotations


DEFAULT_FACE_STRESS_MAP = {
    "angry": 0.95,
    "fear": 0.9,
    "sad": 0.75,
    "neutral": 0.3,
    "surprise": 0.55,
    "happy": 0.05,
    "disgust": 0.9,
}

DEFAULT_VOICE_STRESS_MAP = {
    "stressed": 1.0,
    "angry": 0.9,
    "fearful": 0.9,
    "sad": 0.75,
    "neutral": 0.3,
    "calm": 0.05,
    "happy": 0.1,
}

DEFAULT_STRESS_THRESHOLDS = {
    "low": 0.0,
    "medium": 0.36,
    "high": 0.52,
}

DEFAULT_FUSION_WEIGHTS = {
    "face": 0.45,
    "voice": 0.55,
}
