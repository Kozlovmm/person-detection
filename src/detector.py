"""Wrapper for person detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Single detection box with class and confidence."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str


class Detector:
    """Wrapper around a YOLO model to find people in frames."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        imgsz: int | Tuple[int, int] | None = None,
        device: str = "auto",
    ) -> None:
        """Load weights and set inference parameters."""
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(model_path)
        self.person_class_id = self._resolve_person_class_id()

    def _resolve_person_class_id(self) -> int:
        """Return the numeric class id for 'person'."""
        names = self.model.model.names  
        for class_id, name in names.items():
            if name.lower() == "person":
                return int(class_id)
        raise ValueError("The loaded model does not contain a 'person' class.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a frame and return person detections.

        Args:
            frame: BGR frame as numpy array.

        Returns:
            List of Detection objects filtered to the person class.
        """
        predict_kwargs = {
            "source": frame,
            "conf": self.conf_threshold,
            "classes": [self.person_class_id],
            "verbose": False,
        }
        if self.imgsz is not None:
            predict_kwargs["imgsz"] = self.imgsz
        predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)

        detections: List[Detection] = []
        if not results:
            return detections

        first_result = results[0]
        boxes = getattr(first_result, "boxes", None)
        if boxes is None:
            return detections

        for box in boxes:
            if box.cls is None or box.conf is None or box.xyxy is None:
                continue
            coords = box.xyxy[0].tolist()
            bbox = tuple(float(c) for c in coords)
            confidence = float(box.conf[0])
            detections.append(
                Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_name="person",
                )
            )
        return detections
