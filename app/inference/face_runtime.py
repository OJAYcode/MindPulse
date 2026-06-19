from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from app.utils.io import read_json
from training.face.data import enhance_face_image
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
        self.detector = self._create_detector()
        self.haar_detector = self._create_haar_detector()

    def _create_detector(self):
        solutions = getattr(mp, "solutions", None)
        if solutions is None or not hasattr(solutions, "face_detection"):
            return None
        return solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    def _create_haar_detector(self):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        return detector if not detector.empty() else None

    def detect_and_crop(self, frame: np.ndarray, target_size: tuple[int, int] = (224, 224)) -> np.ndarray | None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.detector is None:
            return self._detect_and_crop_with_haar(frame, rgb_frame, target_size)

        result = self.detector.process(rgb_frame)
        if not result.detections:
            return self._detect_and_crop_with_haar(frame, rgb_frame, target_size)

        bbox = result.detections[0].location_data.relative_bounding_box
        height, width, _ = frame.shape
        x1 = max(int(bbox.xmin * width), 0)
        y1 = max(int(bbox.ymin * height), 0)
        x2 = min(int((bbox.xmin + bbox.width) * width), width)
        y2 = min(int((bbox.ymin + bbox.height) * height), height)
        face = rgb_frame[y1:y2, x1:x2]
        if face.size == 0 or not self._is_usable_face(face, frame.shape[:2]):
            return None
        face = cv2.resize(face, target_size)
        return enhance_face_image(face)

    def _detect_and_crop_with_haar(
        self,
        frame: np.ndarray,
        rgb_frame: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray | None:
        if self.haar_detector is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.haar_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(72, 72),
        )
        if len(detections) == 0:
            return None

        x, y, width, height = max(detections, key=lambda item: item[2] * item[3])
        padding = int(max(width, height) * 0.18)
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + width + padding, rgb_frame.shape[1])
        y2 = min(y + height + padding, rgb_frame.shape[0])
        face = rgb_frame[y1:y2, x1:x2]
        if face.size == 0 or not self._is_usable_face(face, frame.shape[:2]):
            return None
        face = cv2.resize(face, target_size)
        return enhance_face_image(face)

    def _is_usable_face(self, face: np.ndarray, frame_shape: tuple[int, int]) -> bool:
        frame_height, frame_width = frame_shape
        face_height, face_width = face.shape[:2]
        if face_height < 72 or face_width < 72:
            return False
        face_area_ratio = (face_height * face_width) / float(max(frame_height * frame_width, 1))
        if face_area_ratio < 0.025:
            return False

        gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        brightness = float(np.mean(gray))
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return 35.0 <= brightness <= 225.0 and blur_score >= 18.0

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
            face_crop = enhance_face_image(cv2.resize(face_crop, target_size))
        return self.predict_from_face(face_crop)
