from __future__ import annotations

import tensorflow as tf

from app.utils.config import get_settings
from app.utils.io import write_json
from app.utils.logging import get_logger
from training.voice.data import load_voice_dataset
from training.voice.model import build_voice_model


LOGGER = get_logger(__name__)


def train_voice_model(
    epochs: int = 8,
    batch_size: int = 16,
    demo_mode: bool = False,
    max_files_per_class: int | None = None,
) -> dict:
    settings = get_settings()
    dataset = load_voice_dataset(
        settings.voice_data_dir,
        sample_rate=settings.audio_sample_rate,
        demo_mode=demo_mode,
        max_files_per_class=max_files_per_class,
    )
    model = build_voice_model(dataset.x_train.shape[1:], len(dataset.labels))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]
    history = model.fit(
        dataset.x_train,
        dataset.y_train,
        validation_data=(dataset.x_val, dataset.y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=callbacks,
    )

    write_json(settings.voice_labels_path, dataset.labels)
    history_payload = {
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "labels": dataset.labels,
        "synthetic_data": dataset.synthetic,
    }
    write_json(settings.voice_model_path.parent / "voice_training_history.json", history_payload)
    model.save_weights(settings.voice_model_path)
    LOGGER.info("Saved voice model to %s", settings.voice_model_path)
    return history_payload
