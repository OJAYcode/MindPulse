from __future__ import annotations

import json
from collections import Counter

import tensorflow as tf

from app.utils.config import get_settings
from app.utils.io import write_json
from app.utils.logging import get_logger
from training.face.data import load_face_dataset
from training.face.model import build_face_model, compile_face_model, unfreeze_face_backbone


LOGGER = get_logger(__name__)


def _compute_class_weights(dataset) -> dict[int, float] | None:
    if dataset.synthetic:
        return None
    train_labels = getattr(dataset.train_data, "labels", None)
    if train_labels is None:
        return None
    label_counts = Counter(int(label) for label in train_labels)
    if not label_counts:
        return None
    total_samples = sum(label_counts.values())
    num_classes = len(label_counts)
    raw_weights = {
        label: total_samples / (num_classes * count)
        for label, count in label_counts.items()
        if count > 0
    }
    # Keep weighting helpful but not extreme for a laptop-friendly baseline.
    return {label: float(min(max(weight, 0.75), 2.5)) for label, weight in raw_weights.items()}


def train_face_model(
    epochs: int = 5,
    batch_size: int = 32,
    demo_mode: bool = False,
    max_files_per_class: int | None = None,
    fine_tune_epochs: int = 4,
    fine_tune_layers: int = 60,
) -> dict:
    settings = get_settings()
    dataset = load_face_dataset(
        settings.face_data_dir,
        batch_size=batch_size,
        demo_mode=demo_mode,
        max_files_per_class=max_files_per_class,
    )
    write_json(settings.face_labels_path, dataset.labels)

    model = build_face_model(
        input_shape=dataset.input_shape,
        num_classes=len(dataset.labels),
    )
    class_weights = _compute_class_weights(dataset)
    if class_weights:
        LOGGER.info("Using face class weights: %s", class_weights)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(settings.face_model_path.parent / "face_training_log.csv")),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(settings.face_model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]

    if dataset.synthetic:
        x_train, y_train = dataset.train_data
        initial_history = model.fit(
            x_train,
            y_train,
            validation_data=dataset.val_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=callbacks,
            class_weight=class_weights,
        )
    else:
        initial_history = model.fit(
            dataset.train_data,
            validation_data=dataset.val_data,
            epochs=epochs,
            verbose=1,
            callbacks=callbacks,
            class_weight=class_weights,
        )

    fine_tune_history = None
    if not dataset.synthetic and fine_tune_epochs > 0:
        unfreeze_face_backbone(model, trainable_layers=fine_tune_layers)
        compile_face_model(model, learning_rate=1e-4)
        fine_tune_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            tf.keras.callbacks.CSVLogger(
                str(settings.face_model_path.parent / "face_training_log.csv"),
                append=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(settings.face_model_path),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
            ),
        ]
        fine_tune_history = model.fit(
            dataset.train_data,
            validation_data=dataset.val_data,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=epochs,
            verbose=1,
            callbacks=fine_tune_callbacks,
            class_weight=class_weights,
        )

    history = initial_history
    merged_history = {
        key: [float(value) for value in values]
        for key, values in initial_history.history.items()
    }
    if fine_tune_history is not None:
        for key, values in fine_tune_history.history.items():
            merged_history.setdefault(key, [])
            merged_history[key].extend(float(value) for value in values)
    history_payload = {
        "history": merged_history,
        "labels": dataset.labels,
        "synthetic_data": dataset.synthetic,
        "fine_tuned": bool(fine_tune_history is not None),
        "class_weights": class_weights,
    }
    write_json(settings.face_model_path.parent / "face_training_history.json", history_payload)
    (settings.face_model_path.parent / "face_training_history.raw.json").write_text(
        json.dumps(merged_history, default=float, indent=2),
        encoding="utf-8",
    )
    LOGGER.info("Saved face model weights to %s", settings.face_model_path)
    return history_payload
