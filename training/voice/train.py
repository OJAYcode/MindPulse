from __future__ import annotations

from collections import Counter

import tensorflow as tf

from app.utils.config import get_settings
from app.utils.io import write_json
from app.utils.logging import get_logger
from training.voice.data import load_voice_dataset
from training.voice.model import build_voice_model


LOGGER = get_logger(__name__)


def _compute_class_weights(labels) -> dict[int, float] | None:
    label_counts = Counter(int(label) for label in labels)
    if not label_counts:
        return None
    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    raw_weights = {
        label: total_samples / (num_classes * count)
        for label, count in label_counts.items()
        if count > 0
    }
    return {label: float(min(max(weight, 0.65), 3.0)) for label, weight in raw_weights.items()}


def train_voice_model(
    epochs: int = 8,
    batch_size: int = 16,
    demo_mode: bool = False,
    max_files_per_class: int | None = None,
    label_mode: str = "emotion",
) -> dict:
    settings = get_settings()
    dataset = load_voice_dataset(
        settings.voice_data_dir,
        sample_rate=settings.audio_sample_rate,
        demo_mode=demo_mode,
        max_files_per_class=max_files_per_class,
        label_mode=label_mode,
    )
    model = build_voice_model(dataset.x_train.shape[1:], len(dataset.labels))
    class_weights = None if dataset.synthetic else _compute_class_weights(dataset.y_train)
    if class_weights:
        LOGGER.info("Using voice class weights: %s", class_weights)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(settings.voice_model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
        class_weight=class_weights,
    )

    write_json(settings.voice_labels_path, dataset.labels)
    history_payload = {
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "labels": dataset.labels,
        "synthetic_data": dataset.synthetic,
        "class_weights": class_weights,
        "label_mode": label_mode,
    }
    write_json(settings.voice_model_path.parent / "voice_training_history.json", history_payload)
    model.save_weights(settings.voice_model_path)
    LOGGER.info("Saved voice model to %s", settings.voice_model_path)
    return history_payload
