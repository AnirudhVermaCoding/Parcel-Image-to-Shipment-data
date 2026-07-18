from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import zxingcpp

from src.preprocessing.enhancement import clahe_gray
from src.preprocessing.orientation import rotate_image
from src.schemas import BarcodeObservation, BoundingBox


def _position_bbox(position: object) -> BoundingBox | None:
    points = []
    for name in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(position, name, None)
        if point is not None:
            points.append((int(point.x), int(point.y)))
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return BoundingBox(
        x=min(xs),
        y=min(ys),
        width=max(xs) - min(xs),
        height=max(ys) - min(ys),
    )


def _decode_variant(image: np.ndarray, source_crop: str, rotation: int) -> list[BarcodeObservation]:
    observations: list[BarcodeObservation] = []
    try:
        results = zxingcpp.read_barcodes(image)
    except Exception:
        return observations
    for result in results:
        value = str(result.text or "").strip()
        if not value:
            continue
        observations.append(
            BarcodeObservation(
                value=value,
                barcode_type=str(getattr(result, "format", "UNKNOWN")).split(".")[-1],
                source_crop=source_crop,
                bounding_box=_position_bbox(getattr(result, "position", None)),
                rotation=rotation,
                evidence_level=0.82,
            )
        )
    return observations


def decode_barcodes(
    bgr: np.ndarray,
    label_crops: list[tuple[str, np.ndarray]] | None = None,
) -> list[BarcodeObservation]:
    sources: list[tuple[str, np.ndarray, tuple[int, ...]]] = [
        ("full_original", bgr, (0,)),
        ("full_gray", cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (0,)),
        ("full_clahe", clahe_gray(bgr), (0,)),
    ]
    for name, crop in label_crops or []:
        sources.append((name, crop, (0, 90, 180, 270)))

    all_observations: list[BarcodeObservation] = []
    counts: defaultdict[str, int] = defaultdict(int)
    for source_name, image, rotations in sources:
        for rotation in rotations:
            rotated = rotate_image(image, rotation)
            found = _decode_variant(rotated, source_name, rotation)
            for observation in found:
                counts[observation.value] += 1
                all_observations.append(observation)

    deduplicated: dict[tuple[str, str], BarcodeObservation] = {}
    for observation in all_observations:
        repetition_bonus = min(0.12, max(0, counts[observation.value] - 1) * 0.04)
        observation.evidence_level = min(0.95, observation.evidence_level + repetition_bonus)
        key = (observation.value, observation.source_crop)
        current = deduplicated.get(key)
        if current is None or observation.evidence_level > current.evidence_level:
            deduplicated[key] = observation
    return sorted(
        deduplicated.values(),
        key=lambda item: (item.evidence_level, item.value),
        reverse=True,
    )

