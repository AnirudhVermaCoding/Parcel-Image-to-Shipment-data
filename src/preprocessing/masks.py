from __future__ import annotations

import cv2
import numpy as np

from src.config import AppConfig


def yellow_text_mask(bgr: np.ndarray, config: AppConfig) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(config.yellow_hsv_lower, dtype=np.uint8),
        np.array(config.yellow_hsv_upper, dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def suppress_region(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    output = image.copy()
    if bbox is not None:
        x, y, width, height = bbox
        output[y : y + height, x : x + width] = 0
    return output

