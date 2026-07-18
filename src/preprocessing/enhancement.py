from __future__ import annotations

import cv2
import numpy as np


def clahe_gray(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def adaptive_binary(image: np.ndarray) -> np.ndarray:
    gray = clahe_gray(image)
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        9,
    )


def resize_for_ocr(image: np.ndarray, min_height: int = 320) -> np.ndarray:
    if image.shape[0] >= min_height:
        return image
    scale = min_height / max(image.shape[0], 1)
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

