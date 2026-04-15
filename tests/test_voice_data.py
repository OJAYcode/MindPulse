from app.utils.config import get_settings
from training.voice.data import load_voice_dataset


def test_voice_dataset_demo_mode():
    settings = get_settings()
    dataset = load_voice_dataset(settings.voice_data_dir, demo_mode=True)
    assert dataset.x_train.ndim == 4
    assert len(dataset.labels) > 0
