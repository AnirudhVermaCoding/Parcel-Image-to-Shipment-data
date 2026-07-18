from __future__ import annotations

from src.config import AppConfig
from src.schemas import (
    ExtractionSource,
    ImageResult,
    LabelStatus,
    ParcelVisibility,
    PrimaryStatus,
    StatusFlag,
)


REVIEW_FLAGS = {
    StatusFlag.MULTIPLE_PARCELS,
    StatusFlag.PARCEL_PARTIALLY_VISIBLE,
    StatusFlag.PARCEL_DETECTION_UNCERTAIN,
    StatusFlag.LABEL_BLOCKED,
    StatusFlag.LABEL_UNREADABLE,
    StatusFlag.LABEL_PARTIALLY_VISIBLE,
    StatusFlag.MULTIPLE_LABEL_CANDIDATES,
    StatusFlag.AWB_CONFLICT,
    StatusFlag.LOW_OCR_CONFIDENCE,
}


REVIEW_REASON_BY_FLAG = {
    StatusFlag.MULTIPLE_PARCELS: "multiple_parcels",
    StatusFlag.PARCEL_PARTIALLY_VISIBLE: "parcel_partially_visible",
    StatusFlag.PARCEL_DETECTION_UNCERTAIN: "parcel_detection_uncertain",
    StatusFlag.LABEL_BLOCKED: "label_blocked",
    StatusFlag.LABEL_UNREADABLE: "label_unreadable",
    StatusFlag.LABEL_PARTIALLY_VISIBLE: "label_partially_visible",
    StatusFlag.MULTIPLE_LABEL_CANDIDATES: "multiple_label_candidates",
    StatusFlag.AWB_CONFLICT: "awb_conflict",
    StatusFlag.LOW_OCR_CONFIDENCE: "awb_below_clean_threshold",
}


def assign_status(result: ImageResult, config: AppConfig) -> ImageResult:
    flags = set(result.status_flags)
    if result.error_code:
        result.primary_status = PrimaryStatus.PROCESSING_ERROR
        result.requires_review = True
        result.review_reasons = ["processing_error"]
        return result
    if result.parcel_count == 0 and result.parcel_visibility == ParcelVisibility.NONE:
        flags.add(StatusFlag.NO_PARCEL)
        result.primary_status = PrimaryStatus.REJECTED
        result.requires_review = False
        result.review_reasons = []
        result.status_flags = sorted(flags, key=lambda item: item.value)
        return result
    if result.parcel_count is None:
        flags.add(StatusFlag.PARCEL_DETECTION_UNCERTAIN)
    elif result.parcel_count > 1:
        flags.add(StatusFlag.MULTIPLE_PARCELS)
    if result.parcel_visibility == ParcelVisibility.PARTIAL:
        flags.add(StatusFlag.PARCEL_PARTIALLY_VISIBLE)

    if result.label_status == LabelStatus.LABEL_NOT_VISIBLE:
        flags.add(StatusFlag.LABEL_NOT_VISIBLE)
    elif result.label_status == LabelStatus.LABEL_PARTIALLY_VISIBLE:
        flags.add(StatusFlag.LABEL_PARTIALLY_VISIBLE)
    elif result.label_status == LabelStatus.LABEL_BLOCKED_OR_OCCLUDED:
        flags.add(StatusFlag.LABEL_BLOCKED)
    elif result.label_status == LabelStatus.MULTIPLE_LABEL_CANDIDATES:
        flags.add(StatusFlag.MULTIPLE_LABEL_CANDIDATES)
    elif result.label_status in {
        LabelStatus.LABEL_VISIBLE_LOW_CONFIDENCE,
        LabelStatus.LABEL_STATUS_UNCERTAIN,
    }:
        flags.add(StatusFlag.LABEL_UNREADABLE)

    if not result.awb_number:
        flags.add(StatusFlag.AWB_NOT_FOUND)
    elif result.awb_confidence < config.awb_clean_threshold:
        flags.add(StatusFlag.LOW_OCR_CONFIDENCE)
    if result.weight_grams is None:
        flags.add(StatusFlag.WEIGHT_NOT_FOUND)
    if any(value is None for value in (result.length_cm, result.width_cm, result.height_cm)):
        flags.add(StatusFlag.DIMENSIONS_NOT_FOUND)

    risky = bool(flags & REVIEW_FLAGS)
    if StatusFlag.AWB_CONFLICT in flags or risky:
        result.primary_status = PrimaryStatus.REVIEW_REQUIRED
        result.requires_review = True
    elif (
        result.awb_number
        and result.awb_source == ExtractionSource.OVERLAY_OCR
        and result.label_status == LabelStatus.LABEL_NOT_VISIBLE
        and result.awb_confidence >= config.awb_clean_threshold
    ):
        result.primary_status = PrimaryStatus.SUCCESS_OVERLAY_ONLY
        result.requires_review = False
    elif result.awb_number and {
        ExtractionSource.OVERLAY_OCR,
        ExtractionSource.BARCODE,
    }.issubset({candidate.source for candidate in result.candidates if candidate.field_name == "awb_number"}):
        result.primary_status = PrimaryStatus.SUCCESS_OVERLAY_AND_LABEL
        result.requires_review = False
    elif result.awb_number and result.awb_source in {
        ExtractionSource.BARCODE,
        ExtractionSource.LABEL_OCR,
    }:
        result.primary_status = PrimaryStatus.SUCCESS_LABEL_VERIFIED
        result.requires_review = False
    elif result.awb_number and result.awb_source == ExtractionSource.OVERLAY_OCR:
        result.primary_status = PrimaryStatus.SUCCESS_OVERLAY_ONLY
        result.requires_review = False
    else:
        result.primary_status = PrimaryStatus.SUCCESS_PARTIAL_FIELDS
        result.requires_review = True
    reasons = {
        REVIEW_REASON_BY_FLAG[flag]
        for flag in flags & REVIEW_FLAGS
        if flag in REVIEW_REASON_BY_FLAG
    }
    if result.requires_review and not result.awb_number:
        reasons.add("awb_not_found")
    if result.requires_review and result.vision_attempted and not result.vision_succeeded:
        reasons.add("vision_unavailable_local_evidence_insufficient")
    result.review_reasons = sorted(reasons)
    result.status_flags = sorted(flags, key=lambda item: item.value)
    return result
