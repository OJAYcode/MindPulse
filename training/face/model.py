from __future__ import annotations

from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models


def build_face_model(input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08, fill_mode="nearest")(x)
    x = layers.RandomZoom(0.12, fill_mode="nearest")(x)
    x = layers.RandomTranslation(0.08, 0.08, fill_mode="nearest")(x)
    x = layers.RandomBrightness(0.16)(x)
    x = layers.RandomContrast(0.18)(x)
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)(x)

    try:
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling=None,
        )
    except Exception:
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=None,
        )
    base_model._name = "face_backbone"
    base_model.trainable = False

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.45)(x)
    x = layers.Dense(96, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs, name="face_emotion_mobilenetv2_enhanced")
    compile_face_model(model, learning_rate=7e-4)
    return model


def load_face_model(weights_path: Path, input_shape: tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    model = build_face_model(input_shape=input_shape, num_classes=num_classes)
    model.load_weights(weights_path)
    return model


def compile_face_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def unfreeze_face_backbone(model: tf.keras.Model, trainable_layers: int = 40) -> None:
    backbone = next(
        (
            layer
            for layer in model.layers
            if isinstance(layer, tf.keras.Model)
            and (
                "face_backbone" in layer.name.lower()
                or "mobilenetv2" in layer.name.lower()
                or "efficientnet" in layer.name.lower()
            )
        ),
        None,
    )
    if backbone is None:
        raise ValueError("Could not locate the face backbone for fine-tuning.")
    backbone.trainable = True

    if trainable_layers <= 0:
        trainable_layers = len(backbone.layers)

    for layer in backbone.layers[:-trainable_layers]:
        layer.trainable = False

    for layer in backbone.layers[-trainable_layers:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
