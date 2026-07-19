from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2

from src.classification.label_quality import classify_label_status
from src.classification.onnx_detector import detect_with_onnx
from src.classification.parcel_detector import detect_parcels
from src.classification.status_engine import assign_status
from src.config import AppConfig
from src.extraction.barcode import decode_barcodes
from src.extraction.courier_rules import matching_rules
from src.extraction.field_parser import normalise_awb
from src.extraction.label_detector import LabelCandidate, detect_label_candidates
from src.extraction.label_ocr import extract_label_text
from src.extraction.overlay import extract_overlay
from src.extraction.vision_fallback import VisionProvider, XaiVisionProvider
from src.preprocessing.masks import suppress_region
from src.reconciliation.candidates import ReconciledField, reconcile
from src.schemas import (
    ExtractionSource,
    FieldCandidate,
    ImageResult,
    LabelStatus,
    PrimaryStatus,
    ProcessingStatus,
    StatusFlag,
    VisionAnalysis,
    VisionObservation,
)
from src.validation.files import InputValidationError, validate_image_bytes
from src.validation.images import calculate_quality_metrics, is_low_quality, load_image


def _apply_reconciled(result: ImageResult, field: str, reconciled: ReconciledField) -> None:
    candidate = reconciled.candidate
    if candidate is None:
        return
    if field == "awb_number":
        result.awb_number = str(candidate.normalised_value)
        result.awb_source = candidate.source
        result.awb_confidence = reconciled.confidence
    elif field == "weight":
        result.weight_value = float(candidate.value) if candidate.value is not None else None
        result.weight_unit = str(candidate.metadata.get("original_unit") or "") or None
        result.weight_grams = float(candidate.normalised_value)
        result.weight_source = candidate.source
        result.weight_confidence = reconciled.confidence
    elif field in {"length_cm", "width_cm", "height_cm"}:
        setattr(result, field, float(candidate.normalised_value))
    elif field == "recorded_volume":
        result.recorded_volume = float(candidate.normalised_value)
        result.recorded_volume_unit = str(candidate.metadata.get("unit", "cm3"))
        result.recorded_volume_source = candidate.source
        result.recorded_volume_confidence = reconciled.confidence
    elif field == "location":
        result.location = str(candidate.normalised_value)
        result.location_source = candidate.source
        result.location_confidence = reconciled.confidence
    elif field == "capture_timestamp":
        result.capture_timestamp_raw = str(candidate.value)
        result.capture_timestamp = datetime.fromisoformat(str(candidate.normalised_value))
        result.capture_timestamp_source = candidate.source
        result.capture_timestamp_confidence = reconciled.confidence


def _barcode_candidates(
    observations,
    existing_candidates: list[FieldCandidate],
) -> list[FieldCandidate]:
    existing_awbs = {
        str(candidate.normalised_value)
        for candidate in existing_candidates
        if candidate.field_name == "awb_number" and candidate.normalised_value
    }
    unique_values = {observation.value for observation in observations}
    candidates: list[FieldCandidate] = []
    for observation in observations:
        awb = normalise_awb(observation.value)
        if not awb:
            continue
        agrees = awb in existing_awbs
        if not agrees and len(unique_values) > 1:
            confidence = min(0.73, observation.evidence_level)
            evidence = "Plausible barcode among multiple decoded values"
        elif agrees:
            confidence = min(0.97, observation.evidence_level + 0.08)
            evidence = "Barcode agrees with OCR AWB candidate"
        else:
            confidence = min(0.84, observation.evidence_level)
            evidence = "Single plausible decoded barcode"
        candidates.append(
            FieldCandidate(
                field_name="awb_number",
                value=observation.value,
                normalised_value=awb,
                source=ExtractionSource.BARCODE,
                confidence=confidence,
                evidence=evidence,
                bounding_box=observation.bounding_box,
                metadata={
                    "barcode_type": observation.barcode_type,
                    "source_crop": observation.source_crop,
                    "rotation": observation.rotation,
                    "agrees_with_ocr": agrees,
                },
            )
        )
    return candidates


def _dimension_summary(
    result: ImageResult,
    reconciled_dimensions: list[ReconciledField],
) -> None:
    present = [field for field in reconciled_dimensions if field.candidate is not None]
    if not present:
        return
    result.dimensions_confidence = round(min(field.confidence for field in present), 4)
    source_counts: dict[ExtractionSource, int] = {}
    for field in present:
        source = field.candidate.source
        source_counts[source] = source_counts.get(source, 0) + 1
    result.dimensions_source = max(
        source_counts,
        key=lambda source: (
            source_counts[source],
            {
                ExtractionSource.BARCODE: 4,
                ExtractionSource.OVERLAY_OCR: 3,
                ExtractionSource.LABEL_OCR: 2,
                ExtractionSource.VISION_FALLBACK: 1,
                ExtractionSource.NOT_EXTRACTED: 0,
            }[source],
        ),
    )
    if all(value is not None for value in (result.length_cm, result.width_cm, result.height_cm)):
        result.dimensions_text = (
            f"{result.length_cm:g} x {result.width_cm:g} x {result.height_cm:g} cm"
        )


def _vision_candidates(observation) -> list[FieldCandidate]:
    candidates: list[FieldCandidate] = []
    if observation.awb_candidate:
        awb = normalise_awb(observation.awb_candidate)
        if awb:
            candidates.append(
                FieldCandidate(
                    field_name="awb_number",
                    value=observation.awb_candidate,
                    normalised_value=awb,
                    source=ExtractionSource.VISION_FALLBACK,
                    confidence=min(0.74, observation.confidence),
                    evidence=observation.reasoning_summary,
                )
            )
    return candidates


def _detector_label_candidates(bgr, evidence) -> list[LabelCandidate]:
    candidates: list[LabelCandidate] = []
    for detection in evidence.detections:
        if not (
            detection.class_name.startswith("shipping_label_")
            or detection.class_name == "barcode_region"
        ):
            continue
        bbox = detection.bbox
        if detection.class_name == "barcode_region":
            padding_x = max(12, int(bbox.width * 0.55))
            padding_y = max(12, int(bbox.height * 2.0))
        else:
            padding_x = padding_y = max(4, int(min(bbox.width, bbox.height) * 0.06))
        x1, y1 = max(0, bbox.x - padding_x), max(0, bbox.y - padding_y)
        x2 = min(bgr.shape[1], bbox.x + bbox.width + padding_x)
        y2 = min(bgr.shape[0], bbox.y + bbox.height + padding_y)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        expanded_bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        candidates.append(
            LabelCandidate(
                bbox=expanded_bbox,
                score=detection.confidence,
                crop=crop,
                rectangularity=1.0,
                brightness=0.0,
                edge_density=0.0,
                barcode_pattern=0.0,
            )
        )
    return candidates


def process_image(
    image: str | Path | bytes,
    original_filename: str | None = None,
    config: AppConfig | None = None,
    vision_provider: VisionProvider | None = None,
) -> ImageResult:
    config = config or AppConfig.from_env()
    started = time.perf_counter()
    image_id = uuid.uuid4().hex
    if isinstance(image, bytes):
        data = image
        filename = original_filename or f"{image_id}.jpg"
    else:
        path = Path(image)
        data = path.read_bytes()
        filename = original_filename or path.name
    result = ImageResult(
        image_id=image_id,
        original_filename=filename,
        calibration_version=config.calibration_version,
        processing_status=ProcessingStatus.PROCESSING,
    )
    try:
        validate_image_bytes(filename, data, config)
        result.sha256 = hashlib.sha256(data).hexdigest()
        loaded = load_image(data, config)
        result.image_width = loaded.width
        result.image_height = loaded.height
        result.orientation = loaded.orientation
        result.quality_metrics = calculate_quality_metrics(loaded.bgr)
        low_quality = is_low_quality(result.quality_metrics, config)
        if low_quality:
            result.status_flags.append(StatusFlag.LOW_IMAGE_QUALITY)

        overlay = extract_overlay(loaded.bgr, config)
        result.overlay_detected = overlay.detected
        result.overlay_bbox = overlay.bbox
        result.overlay_ocr_text = overlay.raw_text
        result.overlay_ocr_confidence = overlay.confidence
        result.overlay_variant = overlay.variant
        result.warnings.extend(overlay.warnings)
        result.candidates.extend(overlay.candidates)
        if not overlay.detected:
            result.status_flags.append(StatusFlag.OVERLAY_NOT_FOUND)
        elif len(overlay.candidates) < 4:
            result.status_flags.append(StatusFlag.OVERLAY_PARTIAL)
        if not overlay.ocr_available:
            result.status_flags.append(StatusFlag.OCR_UNAVAILABLE)

        overlay_tuple = None
        if overlay.bbox:
            overlay_tuple = (
                overlay.bbox.x,
                overlay.bbox.y,
                overlay.bbox.width,
                overlay.bbox.height,
            )
        label_search_image = suppress_region(loaded.bgr, overlay_tuple)
        detector = detect_with_onnx(loaded.bgr, config)
        result.detector_used = detector.used
        result.detector_version = detector.version
        result.detector_confidence = detector.confidence
        if not detector.available:
            result.status_flags.append(StatusFlag.DETECTOR_UNAVAILABLE)
        detector_labels = _detector_label_candidates(label_search_image, detector)
        heuristic_labels = detect_label_candidates(label_search_image, config)
        label_candidates = detector_labels + [
            candidate
            for candidate in heuristic_labels
            if all(
                abs(candidate.bbox.x - existing.bbox.x) > max(8, existing.bbox.width * 0.1)
                or abs(candidate.bbox.y - existing.bbox.y) > max(8, existing.bbox.height * 0.1)
                for existing in detector_labels
            )
        ]
        label_candidates = label_candidates[: config.max_label_candidates]
        result.label_bboxes = [candidate.bbox for candidate in label_candidates]
        has_local_awb = any(
            candidate.field_name == "awb_number" for candidate in result.candidates
        )
        label_ocr = extract_label_text(
            label_candidates,
            config,
            secondary_fallback=not has_local_awb,
        )
        result.label_ocr_texts = label_ocr.texts
        result.ocr_engines_used = sorted(set(label_ocr.engines_used))
        result.candidates.extend(label_ocr.candidates)
        result.warnings.extend(label_ocr.warnings)
        if not label_ocr.ocr_available:
            result.status_flags.append(StatusFlag.OCR_UNAVAILABLE)

        label_crops = [
            (f"label_{index}", candidate.crop)
            for index, candidate in enumerate(label_candidates)
        ]
        barcode_observations = decode_barcodes(loaded.bgr, label_crops)
        result.barcode_observations = barcode_observations
        result.barcode_values = sorted({observation.value for observation in barcode_observations})
        if len(result.barcode_values) > 1:
            result.status_flags.append(StatusFlag.MULTIPLE_BARCODES)
        result.candidates.extend(_barcode_candidates(barcode_observations, result.candidates))
        for field_candidate in result.candidates:
            if field_candidate.field_name == "awb_number" and field_candidate.normalised_value:
                field_candidate.metadata["courier_rule_matches"] = matching_rules(
                    str(field_candidate.normalised_value),
                    config.courier_rules_path,
                    field_candidate.evidence,
                )
                if (
                    field_candidate.source == ExtractionSource.BARCODE
                    and not field_candidate.metadata["courier_rule_matches"]
                    and not field_candidate.metadata.get("agrees_with_ocr")
                ):
                    field_candidate.confidence = min(field_candidate.confidence, 0.69)
                    field_candidate.warnings.append(
                        "Barcode does not match a configured courier rule or independent OCR"
                    )

        result.label_count = (
            label_ocr.readable_candidates
            if label_ocr.readable_candidates
            else (1 if label_candidates else 0)
        )
        result.label_status = classify_label_status(
            label_candidates,
            label_ocr,
            barcode_observations,
        )

        if detector.used and detector.label_status == LabelStatus.LABEL_BLOCKED_OR_OCCLUDED:
            result.label_status = detector.label_status

        if detector.used and detector.parcel_count is not None:
            result.parcel_count = detector.parcel_count
            result.parcel_count_confidence = detector.confidence
            result.parcel_visibility = detector.parcel_visibility
            result.parcel_bboxes = detector.parcel_bboxes
        else:
            parcel = detect_parcels(
                loaded.bgr,
                config,
                overlay.bbox,
                has_external_evidence=bool(
                    overlay.detected or label_candidates or barcode_observations
                ),
            )
            result.parcel_count = parcel.count
            result.parcel_count_confidence = parcel.confidence
            result.parcel_visibility = parcel.visibility
            result.parcel_bboxes = parcel.bboxes

        parcel_ambiguous = (
            result.parcel_count is None
            or result.label_status
            in {
                LabelStatus.LABEL_STATUS_UNCERTAIN,
                LabelStatus.LABEL_VISIBLE_LOW_CONFIDENCE,
            }
        )
        field_ambiguous = not any(
            candidate.field_name == "awb_number" for candidate in result.candidates
        )
        ambiguous = parcel_ambiguous or field_ambiguous
        if (
            config.enable_xai_fallback
            and config.xai_api_key
            and ambiguous
        ):
            try:
                provider = vision_provider or XaiVisionProvider(config)
                result.vision_attempted = True
                label_crop_bytes = None
                if label_candidates:
                    encoded, buffer = cv2.imencode(".jpg", label_candidates[0].crop)
                    if encoded:
                        label_crop_bytes = buffer.tobytes()
                analysis = provider.analyze(
                    data,
                    label_crop_bytes=label_crop_bytes,
                    include_full_frame=parcel_ambiguous or label_crop_bytes is None,
                )
                if isinstance(analysis, VisionAnalysis):
                    vision = analysis.observation
                    diagnostic = analysis.diagnostic
                    result.vision_succeeded = diagnostic.success
                    result.vision_cache_hit = diagnostic.cache_hit
                    result.vision_error_category = None
                    result.vision_request_id = diagnostic.request_id
                    result.vision_attempt_count = diagnostic.attempt_count
                    result.vision_duration_ms = diagnostic.duration_ms
                else:
                    vision = VisionObservation.model_validate(analysis)
                    result.vision_succeeded = True
                result.candidates.extend(_vision_candidates(vision))
                if result.parcel_count is None and vision.parcel_count_confidence >= 0.75:
                    result.parcel_count = vision.parcel_count
                    result.parcel_count_confidence = vision.parcel_count_confidence
                    result.parcel_visibility = vision.parcel_visibility
            except Exception as exc:
                result.status_flags.append(StatusFlag.VISION_API_UNAVAILABLE)
                diagnostic = getattr(exc, "diagnostic", None)
                result.vision_attempted = True
                result.vision_succeeded = False
                if diagnostic is not None:
                    result.vision_error_category = diagnostic.category
                    result.vision_request_id = diagnostic.request_id
                    result.vision_attempt_count = diagnostic.attempt_count
                    result.vision_duration_ms = diagnostic.duration_ms
                else:
                    result.vision_error_category = "unexpected"
                result.warnings.append(
                    f"Optional vision provider unavailable: {result.vision_error_category}"
                )

        fields = {
            name: reconcile(name, result.candidates, low_quality=low_quality)
            for name in (
                "awb_number",
                "weight",
                "length_cm",
                "width_cm",
                "height_cm",
                "recorded_volume",
                "location",
                "capture_timestamp",
            )
        }
        if fields["awb_number"].conflict:
            result.status_flags.append(StatusFlag.AWB_CONFLICT)
        for field_name, reconciled in fields.items():
            _apply_reconciled(result, field_name, reconciled)
        _dimension_summary(
            result,
            [fields["length_cm"], fields["width_cm"], fields["height_cm"]],
        )
        if result.awb_number and result.awb_confidence < config.awb_review_threshold:
            result.awb_number = None
            result.awb_source = ExtractionSource.NOT_EXTRACTED

        confidences = [
            score
            for score in (
                result.awb_confidence if result.awb_number else 0,
                result.weight_confidence if result.weight_grams is not None else 0,
                result.dimensions_confidence if result.length_cm is not None else 0,
                result.location_confidence if result.location else 0,
                result.capture_timestamp_confidence if result.capture_timestamp else 0,
            )
            if score > 0
        ]
        result.overall_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        result.processing_status = ProcessingStatus.COMPLETED
        assign_status(result, config)
    except InputValidationError as exc:
        result.processing_status = ProcessingStatus.FAILED
        result.primary_status = PrimaryStatus.REJECTED
        result.status_flags.append(StatusFlag.UNSUPPORTED_IMAGE)
        result.error_code = exc.code
        result.error_message = str(exc)
        result.requires_review = False
    except Exception as exc:
        result.processing_status = ProcessingStatus.FAILED
        result.error_code = "PROCESSING_EXCEPTION"
        result.error_message = f"{type(exc).__name__}: {exc}"
        result.requires_review = True
        assign_status(result, config)
    result.processing_duration_ms = int((time.perf_counter() - started) * 1000)
    result.warnings = list(dict.fromkeys(result.warnings))
    result.status_flags = list(dict.fromkeys(result.status_flags))
    return result
