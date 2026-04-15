from app.utils.config import get_settings
from training.face.data import load_face_dataset


def test_face_dataset_demo_mode():
    settings = get_settings()
    dataset = load_face_dataset(settings.face_data_dir, demo_mode=True)
    x_train, _y_train = dataset.train_data
    assert x_train.ndim == 4
    assert len(dataset.labels) > 0
