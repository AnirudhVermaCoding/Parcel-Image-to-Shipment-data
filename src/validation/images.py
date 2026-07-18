from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.config import AppConfig
from src.schemas import QualityMetrics


@dataclass(slots=True)
class LoadedImage:
    bgr: np.ndarray
    width: int
    height: int
    orientation: str


def load_image(data: bytes, config: AppConfig) -> LoadedImage:
    with Image.open(io.BytesIO(data)) as image:
        corrected = ImageOps.exif_transpose(image).convert("RGB")
        if max(corrected.size) > config.processing_max_side:
            ratio = config.processing_max_side / max(corrected.size)
            corrected = corrected.resize(
                (max(1, int(corrected.width * ratio)), max(1, int(corrected.height * ratio))),
                Image.Resampling.LANCZOS,
            )
        rgb = np.asarray(corrected)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # Downstream processing only needs BGR. Releasing the RGB array here avoids
    # keeping a second full-resolution copy alive through OCR/detection, which
    # halves the peak per-image footprint under concurrent workers.
    del rgb
    height, width = bgr.shape[:2]
    orientation = "LANDSCAPE" if width > height else ("PORTRAIT" if height > width else "SQUARE")
    return LoadedImage(bgr=bgr, width=width, height=height, orientation=orientation)


def calculate_quality_metrics(bgr: np.ndarray) -> QualityMetrics:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    edges = cv2.Canny(gray, 80, 180)
    return QualityMetrics(
        blur_score=round(float(laplacian.var()), 3),
        brightness=round(float(gray.mean()), 3),
        contrast=round(float(gray.std()), 3),
        underexposure_pct=round(float(np.mean(gray < 25) * 100), 3),
        overexposure_pct=round(float(np.mean(gray > 230) * 100), 3),
        edge_density=round(float(np.mean(edges > 0)), 5),
        skew_angle=estimate_skew(gray),
    )


def estimate_skew(gray: np.ndarray) -> float | None:
    edges = cv2.Canny(gray, 80, 180)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(50, min(gray.shape) // 15),
        minLineLength=max(30, min(gray.shape) // 12),
        maxLineGap=20,
    )
    if lines is None:
        return None
    angles: list[float] = []
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if -45 <= angle <= 45:
            angles.append(angle)
    return round(float(np.median(angles)), 2) if angles else None


def is_low_quality(metrics: QualityMetrics, config: AppConfig) -> bool:
    return any(
        (
            metrics.blur_score < config.low_blur_threshold,
            metrics.contrast < config.low_contrast_threshold,
            metrics.underexposure_pct > config.underexposure_threshold_pct,
            metrics.overexposure_pct > config.overexposure_threshold_pct,
        )
    )
