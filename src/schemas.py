from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExtractionSource(StrEnum):
    OVERLAY_OCR = "OVERLAY_OCR"
    BARCODE = "BARCODE"
    LABEL_OCR = "LABEL_OCR"
    VISION_FALLBACK = "VISION_FALLBACK"
    NOT_EXTRACTED = "NOT_EXTRACTED"


class ProcessingStatus(StrEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PrimaryStatus(StrEnum):
    SUCCESS_LABEL_VERIFIED = "SUCCESS_LABEL_VERIFIED"
    SUCCESS_OVERLAY_AND_LABEL = "SUCCESS_OVERLAY_AND_LABEL"
    SUCCESS_OVERLAY_ONLY = "SUCCESS_OVERLAY_ONLY"
    SUCCESS_PARTIAL_FIELDS = "SUCCESS_PARTIAL_FIELDS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    REJECTED = "REJECTED"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class StatusFlag(StrEnum):
    NO_PARCEL = "NO_PARCEL"
    MULTIPLE_PARCELS = "MULTIPLE_PARCELS"
    PARCEL_PARTIALLY_VISIBLE = "PARCEL_PARTIALLY_VISIBLE"
    PARCEL_DETECTION_UNCERTAIN = "PARCEL_DETECTION_UNCERTAIN"
    LABEL_NOT_VISIBLE = "LABEL_NOT_VISIBLE"
    LABEL_BLOCKED = "LABEL_BLOCKED"
    LABEL_UNREADABLE = "LABEL_UNREADABLE"
    LABEL_PARTIALLY_VISIBLE = "LABEL_PARTIALLY_VISIBLE"
    MULTIPLE_LABEL_CANDIDATES = "MULTIPLE_LABEL_CANDIDATES"
    MULTIPLE_BARCODES = "MULTIPLE_BARCODES"
    AWB_NOT_FOUND = "AWB_NOT_FOUND"
    AWB_CONFLICT = "AWB_CONFLICT"
    DIMENSIONS_NOT_FOUND = "DIMENSIONS_NOT_FOUND"
    WEIGHT_NOT_FOUND = "WEIGHT_NOT_FOUND"
    OVERLAY_NOT_FOUND = "OVERLAY_NOT_FOUND"
    OVERLAY_PARTIAL = "OVERLAY_PARTIAL"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    LOW_OCR_CONFIDENCE = "LOW_OCR_CONFIDENCE"
    OCR_UNAVAILABLE = "OCR_UNAVAILABLE"
    VISION_API_UNAVAILABLE = "VISION_API_UNAVAILABLE"
    DETECTOR_UNAVAILABLE = "DETECTOR_UNAVAILABLE"
    UNSUPPORTED_IMAGE = "UNSUPPORTED_IMAGE"
    DUPLICATE_IMAGE = "DUPLICATE_IMAGE"


class ParcelVisibility(StrEnum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    NONE = "NONE"
    UNCERTAIN = "UNCERTAIN"


class LabelStatus(StrEnum):
    LABEL_VISIBLE_READABLE = "LABEL_VISIBLE_READABLE"
    LABEL_VISIBLE_BARCODE_ONLY = "LABEL_VISIBLE_BARCODE_ONLY"
    LABEL_VISIBLE_LOW_CONFIDENCE = "LABEL_VISIBLE_LOW_CONFIDENCE"
    LABEL_PARTIALLY_VISIBLE = "LABEL_PARTIALLY_VISIBLE"
    LABEL_BLOCKED_OR_OCCLUDED = "LABEL_BLOCKED_OR_OCCLUDED"
    LABEL_NOT_VISIBLE = "LABEL_NOT_VISIBLE"
    MULTIPLE_LABEL_CANDIDATES = "MULTIPLE_LABEL_CANDIDATES"
    LABEL_STATUS_UNCERTAIN = "LABEL_STATUS_UNCERTAIN"


class JobStage(StrEnum):
    UPLOADED = "UPLOADED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    GENERATING_CSV = "GENERATING_CSV"
    COMPLETED = "COMPLETED"
    COMPLETED_WITH_WARNINGS = "COMPLETED_WITH_WARNINGS"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class QualityMetrics(BaseModel):
    blur_score: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    underexposure_pct: float = 0.0
    overexposure_pct: float = 0.0
    edge_density: float = 0.0
    skew_angle: float | None = None


class FieldCandidate(BaseModel):
    field_name: str
    value: str | float | int | None
    normalised_value: str | float | int | None = None
    source: ExtractionSource
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""
    bounding_box: BoundingBox | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BarcodeObservation(BaseModel):
    value: str
    barcode_type: str = "UNKNOWN"
    source_crop: str = "full_image"
    bounding_box: BoundingBox | None = None
    rotation: int = 0
    evidence_level: float = Field(default=0.75, ge=0.0, le=1.0)


class ImageResult(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    image_id: str
    original_filename: str
    sha256: str = ""
    image_width: int = 0
    image_height: int = 0
    orientation: str = "UNKNOWN"
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    primary_status: PrimaryStatus = PrimaryStatus.REVIEW_REQUIRED
    status_flags: list[StatusFlag] = Field(default_factory=list)
    parcel_count: int | None = None
    parcel_count_confidence: float = 0.0
    parcel_visibility: ParcelVisibility = ParcelVisibility.UNCERTAIN
    label_count: int = 0
    label_status: LabelStatus = LabelStatus.LABEL_STATUS_UNCERTAIN

    awb_number: str | None = None
    awb_source: ExtractionSource = ExtractionSource.NOT_EXTRACTED
    awb_confidence: float = 0.0

    weight_value: float | None = None
    weight_unit: str | None = None
    weight_grams: float | None = None
    weight_source: ExtractionSource = ExtractionSource.NOT_EXTRACTED
    weight_confidence: float = 0.0

    length_cm: float | None = None
    width_cm: float | None = None
    height_cm: float | None = None
    dimensions_text: str | None = None
    dimensions_source: ExtractionSource = ExtractionSource.NOT_EXTRACTED
    dimensions_confidence: float = 0.0

    recorded_volume: float | None = None
    recorded_volume_unit: str | None = None
    recorded_volume_source: ExtractionSource = ExtractionSource.NOT_EXTRACTED
    recorded_volume_confidence: float = 0.0

    location: str | None = None
    location_source: ExtractionSource = ExtractionSource.NOT_EXTRACTED
    location_confidence: float = 0.0
    capture_timestamp: datetime | None = None
    capture_timestamp_raw: str | None = None
    capture_timestamp_source: ExtractionSource = ExtractionSource.NOT_EXTRACTED
    capture_timestamp_confidence: float = 0.0

    overlay_detected: bool = False
    barcode_values: list[str] = Field(default_factory=list)
    barcode_observations: list[BarcodeObservation] = Field(default_factory=list)
    overall_confidence: float = 0.0
    requires_review: bool = True
    review_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_duration_ms: int = 0
    error_code: str | None = None
    error_message: str | None = None

    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    candidates: list[FieldCandidate] = Field(default_factory=list)
    overlay_ocr_text: str = ""
    overlay_ocr_confidence: float = 0.0
    overlay_variant: str | None = None
    label_ocr_texts: list[str] = Field(default_factory=list)
    overlay_bbox: BoundingBox | None = None
    label_bboxes: list[BoundingBox] = Field(default_factory=list)
    parcel_bboxes: list[BoundingBox] = Field(default_factory=list)
    duplicate_of_image_id: str | None = None
    annotated_image_path: str | None = None
    detector_version: str | None = None
    detector_confidence: float = 0.0
    detector_used: bool = False
    ocr_engines_used: list[str] = Field(default_factory=list)
    vision_attempted: bool = False
    vision_succeeded: bool = False
    vision_cache_hit: bool = False
    vision_error_category: str | None = None
    vision_request_id: str | None = None
    vision_attempt_count: int = 0
    vision_duration_ms: int = 0
    calibration_version: str = "conservative-v1"


class VisionObservation(BaseModel):
    parcel_count: int | None = Field(default=None, ge=0)
    parcel_count_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    parcel_visibility: ParcelVisibility = ParcelVisibility.UNCERTAIN
    label_status: str = "UNCERTAIN"
    awb_candidate: str | None = None
    weight_candidate: str | None = None
    dimensions_candidate: str | None = None
    reasoning_summary: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class VisionDiagnostic(BaseModel):
    success: bool = False
    category: str = "not_attempted"
    message: str = ""
    http_status: int | None = None
    request_id: str | None = None
    attempt_count: int = 0
    duration_ms: int = 0
    cache_hit: bool = False


class VisionAnalysis(BaseModel):
    observation: VisionObservation
    diagnostic: VisionDiagnostic = Field(default_factory=VisionDiagnostic)


class BatchOptions(BaseModel):
    local_only: bool = True
    enable_vision: bool = False
    generate_annotations: bool = False
    max_workers: int = 2
    awb_acceptance_threshold: float = 0.90


class JobRecord(BaseModel):
    job_id: str
    stage: JobStage
    total_images: int = 0
    processed_images: int = 0
    successful_images: int = 0
    review_required_images: int = 0
    rejected_images: int = 0
    failed_images: int = 0
    warning_count: int = 0
    progress_percentage: float = 0.0
    current_image: str | None = None
    started_at: datetime | None = None
    last_progress_at: datetime | None = None
    checkpoint_at: datetime | None = None
    completed_at: datetime | None = None
    report_path: str | None = None
    json_report_path: str | None = None
    error_message: str | None = None
    resumed_count: int = 0
