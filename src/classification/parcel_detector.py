from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import AppConfig
from src.preprocessing.masks import suppress_region
from src.schemas import BoundingBox, ParcelVisibility


@dataclass(slots=True)
class ParcelEvidence:
    count: int | None = None
    confidence: float = 0.0
    visibility: ParcelVisibility = ParcelVisibility.UNCERTAIN
    bboxes: list[BoundingBox] = field(default_factory=list)
    strong_no_parcel: bool = False


def _resize(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    if max(image.shape[:2]) <= max_side:
        return image, 1.0
    scale = max_side / max(image.shape[:2])
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale


def detect_parcels(
    bgr: np.ndarray,
    config: AppConfig,
    overlay_bbox: BoundingBox | None = None,
    has_external_evidence: bool = False,
) -> ParcelEvidence:
    overlay_tuple = None
    if overlay_bbox:
        overlay_tuple = (overlay_bbox.x, overlay_bbox.y, overlay_bbox.width, overlay_bbox.height)
    working = suppress_region(bgr, overlay_tuple)
    resized, scale = _resize(working, config.parcel_detection_max_side)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)),
        iterations=2,
    )
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(resized.shape[0] * resized.shape[1])
    boxes: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        relative_area = width * height / image_area
        contour_area = cv2.contourArea(contour) / image_area
        if relative_area < 0.025 or contour_area < 0.012:
            continue
        if relative_area > 0.92:
            continue
        boxes.append((x, y, width, height, relative_area))
    boxes.sort(key=lambda item: item[4], reverse=True)
    significant = [box for box in boxes if box[4] >= 0.055][:6]
    merged = True
    while merged:
        merged = False
        for first_index in range(len(significant)):
            if merged:
                break
            ax, ay, aw, ah, _ = significant[first_index]
            for second_index in range(first_index + 1, len(significant)):
                bx, by, bw, bh, _ = significant[second_index]
                overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
                overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
                gap_x = max(0, max(ax, bx) - min(ax + aw, bx + bw))
                gap_y = max(0, max(ay, by) - min(ay + ah, by + bh))
                horizontal_related = (
                    overlap_y / max(1, min(ah, bh)) >= 0.28
                    and gap_x <= resized.shape[1] * 0.045
                )
                vertical_related = (
                    overlap_x / max(1, min(aw, bw)) >= 0.28
                    and gap_y <= resized.shape[0] * 0.045
                )
                if not (horizontal_related or vertical_related):
                    continue
                x1, y1 = min(ax, bx), min(ay, by)
                x2, y2 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
                relative_area = (x2 - x1) * (y2 - y1) / image_area
                significant[first_index] = (x1, y1, x2 - x1, y2 - y1, relative_area)
                significant.pop(second_index)
                merged = True
                break
    significant.sort(key=lambda item: item[4], reverse=True)
    significant = significant[:4]
    converted = [
        BoundingBox(
            x=int(x / scale),
            y=int(y / scale),
            width=int(width / scale),
            height=int(height / scale),
        )
        for x, y, width, height, _ in significant
    ]
    if not significant:
        occupancy = float(np.mean(threshold > 0))
        strong_no_parcel = occupancy < 0.015 and not has_external_evidence
        return ParcelEvidence(
            count=0 if strong_no_parcel else None,
            confidence=0.88 if strong_no_parcel else 0.28,
            visibility=ParcelVisibility.NONE if strong_no_parcel else ParcelVisibility.UNCERTAIN,
            bboxes=[],
            strong_no_parcel=strong_no_parcel,
        )
    primary = significant[0]
    x, y, width, height, relative_area = primary
    margin = max(3, int(min(resized.shape[:2]) * 0.008))
    touches = (
        x <= margin
        or y <= margin
        or x + width >= resized.shape[1] - margin
        or y + height >= resized.shape[0] - margin
    )
    count = 1
    if len(significant) >= 2 and significant[1][4] >= max(0.07, primary[4] * 0.38):
        count = 2
    confidence = min(0.92, 0.55 + min(relative_area, 0.35))
    if count > 1:
        confidence = min(confidence, 0.78)
    return ParcelEvidence(
        count=count,
        confidence=round(confidence, 3),
        visibility=ParcelVisibility.PARTIAL if touches else ParcelVisibility.FULL,
        bboxes=converted,
        strong_no_parcel=False,
    )
