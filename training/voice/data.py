from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy import signal
from sklearn.model_selection import train_test_split

from app.utils.config import get_settings
from app.utils.io import write_json
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
VOICE_SAMPLE_SEED = 42
VOICE_CACHE_VERSION = "v2"
VOICE_STRESS_MAPPING = {
    "angry": "stressed",
    "calm": "calm",
    "fear": "stressed",
    "fearful": "stressed",
    "happy": "calm",
    "neutral": "neutral",
    "sad": "stressed",
    "stress": "stressed",
    "stressed": "stressed",
}
STRESS_LABEL_ORDER = ["calm", "neutral", "stressed"]


@dataclass(slots=True)
class VoiceDataset:
    x_train: np.ndarray
    x_val: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    labels: list[str]
    sample_rate: int
    synthetic: bool = False


def audio_to_mel_spectrogram(
    file_path: Path,
    sample_rate: int = 16000,
    clip_duration: float = 3.0,
    n_mels: int = 64,
    hop_length: int = 256,
    frame_length: int = 1024,
) -> np.ndarray:
    samples, original_sample_rate = sf.read(file_path, dtype="float32", always_2d=False)
    if getattr(samples, "ndim", 1) > 1:
        samples = samples.mean(axis=1)

    if int(original_sample_rate) != sample_rate:
        target_samples = int(round(len(samples) * sample_rate / float(original_sample_rate)))
        samples = signal.resample(samples, target_samples).astype("float32")

    target_length = int(sample_rate * clip_duration)
    if len(samples) < target_length:
        samples = np.pad(samples, (0, target_length - len(samples)))
    else:
        samples = samples[:target_length]

    stft = tf.signal.stft(
        samples,
        frame_length=frame_length,
        frame_step=hop_length,
        fft_length=frame_length,
    )
    spectrogram = tf.abs(stft) ** 2
    mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mels,
        num_spectrogram_bins=int(spectrogram.shape[-1]),
        sample_rate=sample_rate,
        lower_edge_hertz=20.0,
        upper_edge_hertz=sample_rate / 2.0,
    )
    mel = tf.matmul(spectrogram, mel_weight_matrix)
    mel = tf.math.log(mel + 1e-6)
    mel = tf.transpose(mel)
    mel = mel.numpy()
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
    return mel.astype("float32")[..., np.newaxis]


def _synthetic_voice_dataset(
    sample_rate: int = 16000,
    n_mels: int = 64,
    time_bins: int = 188,
    samples_per_class: int = 40,
) -> VoiceDataset:
    rng = np.random.default_rng(123)
    labels = ["calm", "neutral", "stressed"]
    items = []
    targets = []
    for index, _label in enumerate(labels):
        for _ in range(samples_per_class):
            base = rng.normal(loc=index * 0.35, scale=0.2, size=(n_mels, time_bins, 1))
            items.append(np.clip(base, 0.0, 1.0).astype("float32"))
            targets.append(index)
    x = np.array(items, dtype="float32")
    y = np.array(targets, dtype="int32")
    return _split_dataset(x, y, labels, sample_rate, synthetic=True)


def _split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    sample_rate: int,
    synthetic: bool = False,
) -> VoiceDataset:
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    return VoiceDataset(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        labels=labels,
        sample_rate=sample_rate,
        synthetic=synthetic,
    )


def load_voice_dataset(
    data_dir: Path,
    sample_rate: int = 16000,
    demo_mode: bool = False,
    max_files_per_class: int | None = None,
    label_mode: str = "emotion",
) -> VoiceDataset:
    settings = get_settings()
    cache_suffix = f"_{max_files_per_class}" if max_files_per_class is not None else ""
    normalized_label_mode = label_mode.lower().strip()
    if normalized_label_mode not in {"emotion", "stress"}:
        raise ValueError("label_mode must be either 'emotion' or 'stress'.")
    cache_path = settings.voice_processed_dir / (
        f"voice_dataset_cache_{VOICE_CACHE_VERSION}_{normalized_label_mode}{cache_suffix}.npz"
    )
    if demo_mode or not data_dir.exists() or not any(data_dir.iterdir()):
        LOGGER.warning("Voice dataset missing or demo mode enabled. Using synthetic voice dataset.")
        dataset = _synthetic_voice_dataset(sample_rate=sample_rate)
        write_json(
            settings.voice_preprocessing_path,
            {"sample_rate": sample_rate, "n_mels": 64, "synthetic_data": True},
        )
        return dataset

    source_labels = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    if normalized_label_mode == "stress":
        labels = [label for label in STRESS_LABEL_ORDER if any(VOICE_STRESS_MAPPING.get(source.lower()) == label for source in source_labels)]
    else:
        labels = source_labels
    label_to_index = {label: index for index, label in enumerate(labels)}

    if cache_path.exists():
        LOGGER.info("Loading cached voice dataset from %s", cache_path)
        cached = np.load(cache_path, allow_pickle=True)
        return _split_dataset(
            cached["x"],
            cached["y"],
            list(cached["labels"]),
            sample_rate,
            synthetic=False,
        )

    features: list[np.ndarray] = []
    targets: list[int] = []
    grouped_files: dict[str, list[Path]] = {label: [] for label in labels}
    for source_index, source_label in enumerate(source_labels):
        target_label = VOICE_STRESS_MAPPING.get(source_label.lower(), source_label) if normalized_label_mode == "stress" else source_label
        if target_label not in grouped_files:
            LOGGER.warning("Skipping voice folder '%s' because it does not map to a supported label.", source_label)
            continue
        label_files = [
            file_path
            for file_path in (data_dir / source_label).rglob("*")
            if file_path.suffix.lower() in AUDIO_EXTENSIONS
        ]
        grouped_files[target_label].extend(sorted(label_files))

    for label_index, label in enumerate(labels):
        label_files = sorted(grouped_files[label])
        if max_files_per_class is not None and len(label_files) > max_files_per_class:
            rng = np.random.default_rng(VOICE_SAMPLE_SEED + label_index)
            sampled_indices = np.sort(rng.choice(len(label_files), size=max_files_per_class, replace=False))
            label_files = [label_files[index] for index in sampled_indices]
        LOGGER.info("Processing %s audio files for label '%s'", len(label_files), label)
        for file_path in label_files:
            try:
                mel = audio_to_mel_spectrogram(file_path, sample_rate=sample_rate)
                features.append(mel)
                targets.append(label_to_index[label])
            except Exception as exc:
                LOGGER.warning("Skipping unreadable audio file %s: %s", file_path, exc)

    if not features:
        LOGGER.warning("No audio files could be processed. Falling back to synthetic voice dataset.")
        return _synthetic_voice_dataset(sample_rate=sample_rate)

    x = np.array(features, dtype="float32")
    y = np.array(targets, dtype="int32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving processed voice dataset cache to %s", cache_path)
    np.savez(cache_path, x=x, y=y, labels=np.array(labels))
    write_json(
        settings.voice_preprocessing_path,
        {
            "sample_rate": sample_rate,
            "n_mels": int(x.shape[1]),
            "time_bins": int(x.shape[2]),
            "synthetic_data": False,
        },
    )
    return _split_dataset(x, y, labels, sample_rate, synthetic=False)
