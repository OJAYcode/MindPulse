from __future__ import annotations

import numpy as np

from app.utils.config import get_settings
from app.utils.logging import get_logger
from app.utils.metrics import save_evaluation_report
from training.voice.data import load_voice_dataset
from training.voice.model import load_voice_model


LOGGER = get_logger(__name__)


def evaluate_voice_model(
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
    model = load_voice_model(settings.voice_model_path, dataset.x_train.shape[1:], len(dataset.labels))
    probabilities = model.predict(dataset.x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    report = save_evaluation_report(
        dataset.y_test,
        predictions,
        dataset.labels,
        settings.voice_model_path.parent / "voice_evaluation.json",
        settings.voice_model_path.parent / "voice_confusion_matrix.png",
    )
    LOGGER.info("Voice evaluation report: %s", report)
    return report
