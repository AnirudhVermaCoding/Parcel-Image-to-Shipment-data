from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AppConfig
from src.preprocessing.enhancement import clahe_gray
from src.schemas import BoundingBox


@dataclass(slots=True)
class LabelCandidate:
    bbox: BoundingBox
    score: float
    crop: np.ndarray
    rectangularity: float
    brightness: float
    edge_density: float
    barcode_pattern: float


def _ordered_quad(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    ordered = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _perspective_crop(image: np.ndarray, contour: np.ndarray) -> np.ndarray | None:
    rectangle = cv2.minAreaRect(contour)
    box = _ordered_quad(cv2.boxPoints(rectangle))
    top_left, top_right, bottom_right, bottom_left = box
    width = int(
        max(
            np.linalg.norm(bottom_right - bottom_left),
            np.linalg.norm(top_right - top_left),
        )
    )
    height = int(
        max(
            np.linalg.norm(top_right - bottom_right),
            np.linalg.norm(top_left - bottom_left),
        )
    )
    if width < 30 or height < 30:
        return None
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(box, destination)
    crop = cv2.warpPerspective(image, transform, (width, height))
    if crop.shape[0] > crop.shape[1] * 1.8:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def _iou(first: BoundingBox, second: BoundingBox) -> float:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def _candidate_score(gray: np.ndarray, contour: np.ndarray, image_area: float) -> tuple[float, dict[str, float]]:
    x, y, width, height = cv2.boundingRect(contour)
    if width < 30 or height < 30:
        return 0.0, {}
    area = float(cv2.contourArea(contour))
    box_area = float(width * height)
    relative_area = box_area / image_area
    if relative_area < 0.004 or relative_area > 0.55:
        return 0.0, {}
    aspect = width / max(height, 1)
    if aspect < 0.22 or aspect > 5.2:
        return 0.0, {}
    rectangularity = area / max(box_area, 1.0)
    region = gray[y : y + height, x : x + width]
    brightness = float(region.mean()) / 255
    edges = cv2.Canny(region, 60, 160)
    edge_density = float(np.mean(edges > 0))
    sobel_x = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    vertical_strength = float(np.mean(np.abs(sobel_x) > 60))
    area_score = min(1.0, relative_area / 0.08)
    aspect_score = max(0.0, 1.0 - abs(np.log(max(aspect, 0.01))) / 2.2)
    score = (
        rectangularity * 0.27
        + min(1.0, brightness * 1.8) * 0.20
        + min(1.0, edge_density * 8) * 0.18
        + min(1.0, vertical_strength * 10) * 0.20
        + area_score * 0.10
        + aspect_score * 0.05
    )
    return score, {
        "rectangularity": rectangularity,
        "brightness": brightness,
        "edge_density": edge_density,
        "barcode_pattern": vertical_strength,
    }


def detect_label_candidates(bgr: np.ndarray, config: AppConfig) -> list[LabelCandidate]:
    gray = clahe_gray(bgr)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        41,
        7,
    )
    adaptive_regions = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7)),
        iterations=2,
    )
    percentile = float(np.percentile(gray, 84))
    brightness_threshold = int(max(72, min(190, percentile + 20)))
    _, bright = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
    bright_regions = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 19)),
        iterations=2,
    )
    contours = []
    for mask in (adaptive_regions, bright_regions):
        found, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.extend(found)
    image_area = float(bgr.shape[0] * bgr.shape[1])
    candidates: list[LabelCandidate] = []
    for contour in contours:
        score, metrics = _candidate_score(gray, contour, image_area)
        if score < 0.34:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        padding = max(3, int(min(width, height) * 0.04))
        x1, y1 = max(0, x - padding), max(0, y - padding)
        x2 = min(bgr.shape[1], x + width + padding)
        y2 = min(bgr.shape[0], y + height + padding)
        bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        crop = _perspective_crop(bgr, contour)
        if crop is None or crop.size == 0:
            crop = bgr[y1:y2, x1:x2]
        candidates.append(
            LabelCandidate(
                bbox=bbox,
                score=round(score, 4),
                crop=crop,
                rectangularity=round(metrics["rectangularity"], 4),
                brightness=round(metrics["brightness"], 4),
                edge_density=round(metrics["edge_density"], 4),
                barcode_pattern=round(metrics["barcode_pattern"], 4),
            )
        )
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: list[LabelCandidate] = []
    for candidate in candidates:
        if all(_iou(candidate.bbox, existing.bbox) < 0.55 for existing in selected):
            selected.append(candidate)
        if len(selected) >= config.max_label_candidates:
            break
    return selected
