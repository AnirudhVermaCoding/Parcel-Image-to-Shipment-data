from __future__ import annotations

from src.extraction.label_detector import LabelCandidate
from src.extraction.label_ocr import LabelOcrExtraction
from src.schemas import BarcodeObservation, LabelStatus


def classify_label_status(
    label_candidates: list[LabelCandidate],
    label_ocr: LabelOcrExtraction,
    barcode_observations: list[BarcodeObservation],
) -> LabelStatus:
    crop_barcodes = list(barcode_observations)
    if label_ocr.readable_candidates > 1:
        return LabelStatus.MULTIPLE_LABEL_CANDIDATES
    if label_ocr.readable_candidates:
        return LabelStatus.LABEL_VISIBLE_READABLE
    if crop_barcodes:
        return LabelStatus.LABEL_VISIBLE_BARCODE_ONLY
    if label_candidates:
        best = label_candidates[0]
        if best.score >= 0.55:
            return LabelStatus.LABEL_VISIBLE_LOW_CONFIDENCE
        return LabelStatus.LABEL_STATUS_UNCERTAIN
    return LabelStatus.LABEL_NOT_VISIBLE
