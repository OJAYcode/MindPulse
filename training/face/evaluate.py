from __future__ import annotations

from pathlib import Path

import numpy as np

from app.utils.config import get_settings
from app.utils.logging import get_logger
from app.utils.metrics import save_evaluation_report
from training.face.data import load_face_dataset
from training.face.model import load_face_model


LOGGER = get_logger(__name__)


def evaluate_face_model(
    demo_mode: bool = False,
    max_files_per_class: int | None = None,
    data_dir: Path | None = None,
) -> dict:
    settings = get_settings()
    dataset = load_face_dataset(
        data_dir or settings.face_data_dir,
        demo_mode=demo_mode,
        max_files_per_class=max_files_per_class,
    )
    model = load_face_model(settings.face_model_path, dataset.input_shape, len(dataset.labels))

    if dataset.synthetic:
        x_test, _y_test = dataset.test_data
        probabilities = model.predict(x_test, verbose=0)
    else:
        probabilities = model.predict(dataset.test_data, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = save_evaluation_report(
        dataset.y_test.tolist(),
        predictions,
        dataset.labels,
        settings.face_model_path.parent / "face_evaluation.json",
        settings.face_model_path.parent / "face_confusion_matrix.png",
    )
    LOGGER.info("Face evaluation report: %s", report)
    return report
