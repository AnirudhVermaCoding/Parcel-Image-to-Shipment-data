from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.schemas import BoundingBox, ImageResult


def _draw_box(
    image: np.ndarray,
    bbox: BoundingBox,
    color: tuple[int, int, int],
    label: str,
) -> None:
    x2 = bbox.x + bbox.width
    y2 = bbox.y + bbox.height
    cv2.rectangle(image, (bbox.x, bbox.y), (x2, y2), color, 3)
    cv2.putText(
        image,
        label,
        (bbox.x, max(24, bbox.y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def create_annotation(bgr: np.ndarray, result: ImageResult, path: Path) -> Path:
    annotated = bgr.copy()
    if result.overlay_bbox:
        _draw_box(annotated, result.overlay_bbox, (0, 220, 255), "MACHINE OVERLAY")
    for index, bbox in enumerate(result.parcel_bboxes):
        _draw_box(annotated, bbox, (255, 140, 0), f"PARCEL {index + 1}")
    for index, bbox in enumerate(result.label_bboxes):
        _draw_box(annotated, bbox, (0, 200, 0), f"LABEL {index + 1}")
    status = result.primary_status.value if hasattr(result.primary_status, "value") else str(result.primary_status)
    awb_source = result.awb_source.value if hasattr(result.awb_source, "value") else str(result.awb_source)
    cv2.rectangle(annotated, (0, 0), (min(annotated.shape[1], 900), 78), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        f"{status} | AWB source: {awb_source}",
        (18, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), annotated):
        raise OSError(f"Could not write annotation to {path}")
    return path

