"""Drawing helpers for visualizing detections on frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np

from .detector import Detection

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 1
BOX_THICKNESS = 2
BOX_COLOR = (46, 204, 113)
TEXT_COLOR = (20, 20, 20)


@dataclass
class DrawScale:
    """Scale settings for a single detection."""

    text_scale: float
    text_thickness: int
    box_thickness: int
    padding: int


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    """Draw thin boxes and labels on a frame."""
    annotated = frame.copy()
    for detection in detections:
        scale = _scales_for_bbox(detection.bbox)
        x1, y1, x2, y2 = _as_int_tuple(detection.bbox)
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            BOX_COLOR,
            scale.box_thickness,
            cv2.LINE_AA,
        )
        label = f"{detection.class_name} {detection.confidence:.2f}"
        _draw_label(
            annotated,
            x1,
            y1,
            label,
            text_scale=scale.text_scale,
            text_thickness=scale.text_thickness,
            padding=scale.padding,
        )
    return annotated


def _as_int_tuple(bbox: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    """Cast bbox values to integer pixel coordinates."""
    return tuple(int(coord) for coord in bbox) 


def _draw_label(
    canvas: np.ndarray,
    x1: int,
    y1: int,
    text: str,
    text_scale: float = TEXT_SCALE,
    text_thickness: int = TEXT_THICKNESS,
    padding: int = 4,
) -> None:
    """Draw a small filled label and text near the top-left corner."""
    (text_width, text_height), baseline = cv2.getTextSize(text, FONT, text_scale, text_thickness)
    y_top = max(y1 - text_height - baseline - 2 * padding, 0)
    x_end = x1 + text_width + 2 * padding
    y_end = y_top + text_height + baseline + 2 * padding

    cv2.rectangle(canvas, (x1, y_top), (x_end, y_end), BOX_COLOR, cv2.FILLED, cv2.LINE_AA)
    text_org = (x1 + padding, y_end - padding - baseline // 2)
    cv2.putText(canvas, text, text_org, FONT, text_scale, TEXT_COLOR, text_thickness, cv2.LINE_AA)


def _scales_for_bbox(bbox: Tuple[float, float, float, float]) -> DrawScale:
    """Scale thickness and text size using size buckets to avoid jitter per frame."""
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    ref = min(w, h)

    if ref < 60:
        return DrawScale(text_scale=0.40, text_thickness=1, box_thickness=1, padding=2)
    if ref < 150:
        return DrawScale(text_scale=0.55, text_thickness=1, box_thickness=2, padding=4)
    return DrawScale(text_scale=0.75, text_thickness=2, box_thickness=3, padding=6)
