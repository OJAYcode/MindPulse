from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from app.utils.io import read_json
from training.voice.data import audio_to_mel_spectrogram
from training.voice.model import load_voice_model


@dataclass(slots=True)
class VoicePrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class VoiceRuntime:
    def __init__(self, model_path: Path, labels_path: Path, sample_rate: int, record_seconds: int) -> None:
        self.labels = read_json(labels_path, default=[])
        self.model = load_voice_model(model_path, (64, 184, 1), len(self.labels))
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds

    def record_audio(self) -> np.ndarray:
        try:
            import sounddevice as sd
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Local microphone recording requires PortAudio. "
                "Install the local development dependencies before running live laptop inference."
            ) from exc

        samples = int(self.sample_rate * self.record_seconds)
        recording = sd.rec(samples, samplerate=self.sample_rate, channels=1, dtype="float32")
        sd.wait()
        return np.squeeze(recording, axis=1)

    def predict_from_audio_samples(self, samples: np.ndarray) -> VoicePrediction:
        temp_path = Path("data/processed/voice/live_temp.wav").resolve()
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        import soundfile as sf

        sf.write(temp_path, samples, self.sample_rate)
        self._validate_audio_quality(temp_path)
        mel = audio_to_mel_spectrogram(temp_path, sample_rate=self.sample_rate, clip_duration=self.record_seconds)
        batch = np.expand_dims(mel, axis=0)
        probabilities = self.model.predict(batch, verbose=0)[0]
        label_index = int(np.argmax(probabilities))
        label = self.labels[label_index] if self.labels else str(label_index)
        return VoicePrediction(
            label=label,
            confidence=float(probabilities[label_index]),
            probabilities={self.labels[i]: float(probabilities[i]) for i in range(len(probabilities))},
        )

    def predict_from_audio_file(self, audio_path: Path) -> VoicePrediction:
        self._validate_audio_quality(audio_path)
        mel = audio_to_mel_spectrogram(audio_path, sample_rate=self.sample_rate, clip_duration=self.record_seconds)
        batch = np.expand_dims(mel, axis=0)
        probabilities = self.model.predict(batch, verbose=0)[0]
        label_index = int(np.argmax(probabilities))
        label = self.labels[label_index] if self.labels else str(label_index)
        return VoicePrediction(
            label=label,
            confidence=float(probabilities[label_index]),
            probabilities={self.labels[i]: float(probabilities[i]) for i in range(len(probabilities))},
        )

    def _validate_audio_quality(self, audio_path: Path) -> None:
        samples, _sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0:
            raise ValueError("The microphone sample is empty. Please try again.")
        rms = float(np.sqrt(np.mean(np.square(samples))))
        peak = float(np.max(np.abs(samples)))
        active_ratio = float(np.mean(np.abs(samples) > 0.01))
        if rms < 0.003 or peak < 0.02 or active_ratio < 0.04:
            raise ValueError("The microphone sample was too quiet. Speak clearly and try the scan again.")
