from __future__ import annotations

from pathlib import Path

from app.fusion.config import (
    DEFAULT_FACE_STRESS_MAP,
    DEFAULT_FUSION_WEIGHTS,
    DEFAULT_STRESS_THRESHOLDS,
    DEFAULT_VOICE_STRESS_MAP,
)
from app.utils.io import read_json, write_json


def default_fusion_rule_config() -> dict:
    return {
        "weights": DEFAULT_FUSION_WEIGHTS,
        "thresholds": DEFAULT_STRESS_THRESHOLDS,
        "face_mapping": DEFAULT_FACE_STRESS_MAP,
        "voice_mapping": DEFAULT_VOICE_STRESS_MAP,
        "source": "default",
    }


def load_fusion_rule_config(path: Path) -> dict:
    config = read_json(path, default=None)
    if not config:
        return default_fusion_rule_config()
    merged = default_fusion_rule_config()
    merged.update({key: value for key, value in config.items() if key in merged})
    return merged


def save_fusion_rule_config(path: Path, config: dict) -> None:
    payload = default_fusion_rule_config()
    payload.update(config)
    write_json(path, payload)
