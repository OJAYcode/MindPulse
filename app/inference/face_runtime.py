from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app.utils.io import read_json
from training.face.model import load_face_model


@dataclass(slots=True)
class FacePrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class FaceRuntime:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.labels = read_json(labels_path, default=[])
        self.model = load_face_model(model_path, (224, 224, 3), len(self.labels))
        self.detector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def detect_and_crop(self, frame: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb_frame)
        if not result.detections:
            return None

        bbox = result.detections[0].location_data.relative_bounding_box
        height, width, _ = frame.shape
        x1 = max(int(bbox.xmin * width), 0)
        y1 = max(int(bbox.ymin * height), 0)
        x2 = min(int((bbox.xmin + bbox.width) * width), width)
        y2 = min(int((bbox.ymin + bbox.height) * height), height)
        face = rgb_frame[y1:y2, x1:x2]
        if face.size == 0:
            return None
        face = cv2.resize(face, target_size).astype("float32")
        return face

    def predict_from_face(self, face_image: np.ndarray) -> FacePrediction:
        batch = np.expand_dims(face_image, axis=0)
        probabilities = self.model.predict(batch, verbose=0)[0]
        label_index = int(np.argmax(probabilities))
        label = self.labels[label_index] if self.labels else str(label_index)
        return FacePrediction(
            label=label,
            confidence=float(probabilities[label_index]),
            probabilities={self.labels[i]: float(probabilities[i]) for i in range(len(probabilities))},
        )

    def predict_from_image_file(self, image_path: Path, target_size: tuple[int, int] = (224, 224)) -> FacePrediction:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not read face image: {image_path}")
        face_crop = self.detect_and_crop(frame, target_size=target_size)
        if face_crop is None:
            face_crop = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = cv2.resize(face_crop, target_size).astype("float32")
        return self.predict_from_face(face_crop)
