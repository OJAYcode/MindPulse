from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _resolve_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _csv_env(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


DEFAULT_FRONTEND_ORIGINS = ",".join(
    [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://mind-pulse-two.vercel.app",
    ]
)


@dataclass(slots=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    mongodb_uri: str = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017/mindpulse")
    mongodb_database: str | None = os.getenv("MONGODB_DATABASE")
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    face_data_dir: Path = _resolve_path(os.getenv("FACE_DATA_DIR", "./data/raw/face"))
    voice_data_dir: Path = _resolve_path(os.getenv("VOICE_DATA_DIR", "./data/raw/voice"))
    face_processed_dir: Path = _resolve_path(os.getenv("FACE_PROCESSED_DIR", "./data/processed/face"))
    voice_processed_dir: Path = _resolve_path(os.getenv("VOICE_PROCESSED_DIR", "./data/processed/voice"))

    face_model_path: Path = _resolve_path(os.getenv("FACE_MODEL_PATH", "./models/face_emotion_model.weights.h5"))
    face_labels_path: Path = _resolve_path(os.getenv("FACE_LABELS_PATH", "./models/face_labels.json"))
    voice_model_path: Path = _resolve_path(os.getenv("VOICE_MODEL_PATH", "./models/voice_emotion_model.weights.h5"))
    voice_labels_path: Path = _resolve_path(os.getenv("VOICE_LABELS_PATH", "./models/voice_labels.json"))
    voice_preprocessing_path: Path = _resolve_path(
        os.getenv("VOICE_PREPROCESSING_PATH", "./models/voice_preprocessing.json")
    )
    fusion_model_path: Path = _resolve_path(os.getenv("FUSION_MODEL_PATH", "./models/fusion_model.pkl"))
    fusion_rules_path: Path = _resolve_path(os.getenv("FUSION_RULES_PATH", "./models/fusion_rules.json"))

    live_capture_interval_seconds: int = int(os.getenv("LIVE_CAPTURE_INTERVAL_SECONDS", "5"))
    audio_record_seconds: int = int(os.getenv("AUDIO_RECORD_SECONDS", "3"))
    audio_sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    api_base_url: str = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    frontend_origins: list[str] = None

    def __post_init__(self) -> None:
        if self.frontend_origins is None:
            self.frontend_origins = _csv_env(os.getenv("FRONTEND_ORIGINS", DEFAULT_FRONTEND_ORIGINS))


def get_settings() -> Settings:
    settings = Settings()
    settings.face_processed_dir.mkdir(parents=True, exist_ok=True)
    settings.voice_processed_dir.mkdir(parents=True, exist_ok=True)
    settings.face_model_path.parent.mkdir(parents=True, exist_ok=True)
    settings.voice_model_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
