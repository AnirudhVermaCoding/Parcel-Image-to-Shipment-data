from __future__ import annotations

import cv2
import numpy as np


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    degrees %= 360
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

