from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FACE_SAMPLE_SEED = 42


def enhance_face_image(rgb_image: np.ndarray) -> np.ndarray:
    """Apply light normalization that is safe for both training and live inference."""
    image = np.clip(rgb_image, 0, 255).astype("uint8")
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    enhanced = cv2.merge((lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB).astype("float32")


class FaceDataSequence(tf.keras.utils.Sequence):
    def __init__(
        self,
        file_paths: list[Path],
        labels: np.ndarray,
        image_size: tuple[int, int],
        batch_size: int,
        shuffle: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.file_paths = file_paths
        self.labels = np.asarray(labels, dtype="int32")
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.file_paths))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.file_paths) / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        images = []
        labels = self.labels[batch_indices]

        for item_index in batch_indices:
            file_path = self.file_paths[item_index]
            image = cv2.imread(str(file_path))
            if image is None:
                LOGGER.warning("Replacing unreadable image with zeros: %s", file_path)
                image = np.zeros((*self.image_size, 3), dtype="uint8")
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.image_size)
                image = enhance_face_image(image)
            images.append(image.astype("float32"))

        return np.asarray(images, dtype="float32"), labels

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


@dataclass(slots=True)
class FaceDataset:
    train_data: tf.keras.utils.Sequence | np.ndarray
    val_data: tuple[np.ndarray, np.ndarray] | tf.keras.utils.Sequence
    test_data: tuple[np.ndarray, np.ndarray] | tf.keras.utils.Sequence
    y_test: np.ndarray
    labels: list[str]
    input_shape: tuple[int, int, int]
    synthetic: bool = False


def _collect_face_files(
    data_dir: Path,
    max_files_per_class: int | None = None,
) -> tuple[list[Path], np.ndarray, list[str]]:
    labels = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    file_paths: list[Path] = []
    targets: list[int] = []

    for label_index, label in enumerate(labels):
        label_files = [
            file_path
            for file_path in (data_dir / label).rglob("*")
            if file_path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        label_files = sorted(label_files)
        if max_files_per_class is not None:
            rng = np.random.default_rng(FACE_SAMPLE_SEED + label_index)
            if len(label_files) > max_files_per_class:
                sampled_indices = np.sort(rng.choice(len(label_files), size=max_files_per_class, replace=False))
                label_files = [label_files[index] for index in sampled_indices]
        LOGGER.info("Using %s face images for label '%s'", len(label_files), label)
        for file_path in label_files:
            file_paths.append(file_path)
            targets.append(label_index)

    return file_paths, np.asarray(targets, dtype="int32"), labels


def _synthetic_face_dataset(
    image_size: tuple[int, int] = (224, 224),
    samples_per_class: int = 32,
) -> FaceDataset:
    rng = np.random.default_rng(42)
    labels = ["angry", "happy", "neutral", "sad"]
    images = []
    targets = []
    for index, _label in enumerate(labels):
        base_value = (index + 1) / len(labels)
        for _ in range(samples_per_class):
            image = rng.normal(loc=base_value * 180, scale=35, size=(*image_size, 3))
            image = np.clip(image, 0, 255)
            images.append(image.astype("float32"))
            targets.append(index)

    x = np.asarray(images, dtype="float32")
    y = np.asarray(targets, dtype="int32")
    x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=0.25,
        random_state=42,
        stratify=y_temp,
    )
    return FaceDataset(
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        test_data=(x_test, y_test),
        y_test=y_test,
        labels=labels,
        input_shape=(image_size[0], image_size[1], 3),
        synthetic=True,
    )


def load_face_dataset(
    data_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 16,
    demo_mode: bool = False,
    max_files_per_class: int | None = None,
) -> FaceDataset:
    if demo_mode or not data_dir.exists() or not any(data_dir.iterdir()):
        LOGGER.warning("Face dataset missing or demo mode enabled. Using synthetic face dataset.")
        return _synthetic_face_dataset(image_size=image_size)

    file_paths, targets, labels = _collect_face_files(data_dir, max_files_per_class=max_files_per_class)
    if not file_paths:
        LOGGER.warning("No face images found. Falling back to synthetic face dataset.")
        return _synthetic_face_dataset(image_size=image_size)

    train_paths, test_paths, y_train, y_test = train_test_split(
        file_paths,
        targets,
        test_size=0.2,
        random_state=42,
        stratify=targets,
    )
    train_paths, val_paths, y_train, y_val = train_test_split(
        train_paths,
        y_train,
        test_size=0.25,
        random_state=42,
        stratify=y_train,
    )

    train_data = FaceDataSequence(train_paths, y_train, image_size=image_size, batch_size=batch_size, shuffle=True)
    val_data = FaceDataSequence(val_paths, y_val, image_size=image_size, batch_size=batch_size, shuffle=False)
    test_data = FaceDataSequence(test_paths, y_test, image_size=image_size, batch_size=batch_size, shuffle=False)

    return FaceDataset(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        y_test=np.asarray(y_test, dtype="int32"),
        labels=labels,
        input_shape=(image_size[0], image_size[1], 3),
        synthetic=False,
    )
