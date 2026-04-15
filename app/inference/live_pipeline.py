from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import requests

from app.fusion.io import load_fusion_rule_config
from app.fusion.learned import load_learned_fusion, predict_learned_fusion
from app.fusion.rules import late_fusion_rule
from app.inference.face_runtime import FaceRuntime
from app.inference.voice_runtime import VoiceRuntime
from app.utils.config import get_settings
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)


def _demo_predictions() -> tuple[dict[str, float], str, float, dict[str, float], str, float]:
    face_probs = {"happy": 0.15, "neutral": 0.25, "sad": 0.60}
    voice_probs = {"calm": 0.10, "neutral": 0.20, "stressed": 0.70}
    return face_probs, "sad", 0.60, voice_probs, "stressed", 0.70


def _fuse_prediction(
    learned_fusion_model,
    rule_config: dict,
    face_probabilities: dict[str, float],
    voice_probabilities: dict[str, float],
):
    if learned_fusion_model is not None:
        learned_label, learned_confidence = predict_learned_fusion(
            learned_fusion_model,
            np.asarray([list(face_probabilities.values())], dtype="float32"),
            np.asarray([list(voice_probabilities.values())], dtype="float32"),
        )
        return learned_label, learned_confidence

    rule_result = late_fusion_rule(
        face_probabilities,
        voice_probabilities,
        face_mapping=rule_config.get("face_mapping"),
        voice_mapping=rule_config.get("voice_mapping"),
        thresholds=rule_config.get("thresholds"),
        weights=rule_config.get("weights"),
    )
    return rule_result.stress_level, rule_result.stress_score


def run_live_inference(
    demo_mode: bool = False,
    use_learned_fusion: bool = False,
    max_cycles: int | None = None,
) -> None:
    settings = get_settings()
    rule_config = load_fusion_rule_config(settings.fusion_rules_path)
    learned_fusion_model = None
    if use_learned_fusion:
        learned_fusion_model = load_learned_fusion(settings.fusion_model_path)
        if learned_fusion_model is None:
            LOGGER.warning("Requested learned fusion, but no saved fusion model was found. Falling back to rules.")

    if demo_mode:
        cycle_count = 0
        while max_cycles is None or cycle_count < max_cycles:
            face_probs, face_label, face_conf, voice_probs, voice_label, voice_conf = _demo_predictions()
            stress_level, _fusion_confidence = _fuse_prediction(
                learned_fusion_model,
                rule_config,
                face_probs,
                voice_probs,
            )
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "face_emotion": face_label,
                "face_confidence": face_conf,
                "voice_emotion": voice_label,
                "voice_confidence": voice_conf,
                "stress_level": stress_level,
                "source": "demo_session",
            }
            try:
                requests.post(f"{settings.api_base_url}/inference-result", json=payload, timeout=10).raise_for_status()
                LOGGER.info("Demo inference sent: %s", payload)
            except Exception as exc:
                LOGGER.error("Failed to send demo inference result: %s", exc)
            cycle_count += 1
            if max_cycles is not None and cycle_count >= max_cycles:
                break
            time.sleep(settings.live_capture_interval_seconds)
        return

    face_runtime = FaceRuntime(settings.face_model_path, settings.face_labels_path)
    voice_runtime = VoiceRuntime(
        settings.voice_model_path,
        settings.voice_labels_path,
        sample_rate=settings.audio_sample_rate,
        record_seconds=settings.audio_record_seconds,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        cycle_count = 0
        while max_cycles is None or cycle_count < max_cycles:
            success, frame = cap.read()
            if not success:
                LOGGER.warning("Webcam frame capture failed.")
                time.sleep(1)
                continue

            face_crop = face_runtime.detect_and_crop(frame)
            if face_crop is None:
                LOGGER.warning("No face detected in current frame.")
                time.sleep(settings.live_capture_interval_seconds)
                continue

            face_prediction = face_runtime.predict_from_face(face_crop)
            audio_samples = voice_runtime.record_audio()
            voice_prediction = voice_runtime.predict_from_audio_samples(audio_samples)
            stress_level, _fusion_confidence = _fuse_prediction(
                learned_fusion_model,
                rule_config,
                face_prediction.probabilities,
                voice_prediction.probabilities,
            )

            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "face_emotion": face_prediction.label,
                "face_confidence": face_prediction.confidence,
                "voice_emotion": voice_prediction.label,
                "voice_confidence": voice_prediction.confidence,
                "stress_level": stress_level,
                "source": "laptop_webcam_mic",
            }

            try:
                response = requests.post(f"{settings.api_base_url}/inference-result", json=payload, timeout=10)
                response.raise_for_status()
                LOGGER.info("Inference sent successfully: %s", payload)
            except Exception as exc:
                LOGGER.error("Failed to send inference result: %s", exc)

            cycle_count += 1
            if max_cycles is not None and cycle_count >= max_cycles:
                break
            time.sleep(settings.live_capture_interval_seconds)
    finally:
        cap.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live multimodal stress inference.")
    parser.add_argument("--demo", action="store_true", help="Run without webcam, microphone, or trained models.")
    parser.add_argument(
        "--use-learned-fusion",
        action="store_true",
        help="Use the saved logistic fusion model if available. Otherwise fall back to rule-based fusion.",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional number of inference cycles to run before exiting. Useful for validation and demos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_live_inference(
        demo_mode=args.demo,
        use_learned_fusion=args.use_learned_fusion,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
