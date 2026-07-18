from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

from src.schemas import ImageResult


CSV_COLUMNS = [
    "image_id",
    "filename",
    "sha256",
    "processing_status",
    "primary_status",
    "status_flags",
    "requires_review",
    "review_reasons",
    "parcel_count",
    "parcel_count_confidence",
    "parcel_visibility",
    "label_count",
    "label_status",
    "awb_number",
    "awb_number_excel",
    "awb_source",
    "awb_confidence",
    "weight_original",
    "weight_value",
    "weight_unit",
    "weight_grams",
    "weight_source",
    "weight_confidence",
    "length_cm",
    "width_cm",
    "height_cm",
    "dimensions_text",
    "dimensions_source",
    "dimensions_confidence",
    "recorded_volume",
    "recorded_volume_unit",
    "recorded_volume_source",
    "recorded_volume_confidence",
    "location",
    "location_source",
    "location_confidence",
    "capture_timestamp",
    "capture_timestamp_raw",
    "capture_timestamp_source",
    "capture_timestamp_confidence",
    "overlay_detected",
    "barcode_values",
    "overall_confidence",
    "detector_version",
    "detector_confidence",
    "detector_used",
    "ocr_engines_used",
    "vision_attempted",
    "vision_succeeded",
    "vision_cache_hit",
    "vision_error_category",
    "vision_request_id",
    "vision_attempt_count",
    "vision_duration_ms",
    "calibration_version",
    "warnings",
    "processing_duration_ms",
    "error_code",
    "error_message",
]


def _enum(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


def _excel_identifier(value: str | None) -> str:
    if not value or not re.fullmatch(r"[A-Z0-9-]{1,40}", value):
        return ""
    return f'="{value}"'


def result_row(result: ImageResult) -> dict[str, object]:
    return {
        "image_id": result.image_id,
        "filename": result.original_filename,
        "sha256": result.sha256,
        "processing_status": _enum(result.processing_status),
        "primary_status": _enum(result.primary_status),
        "status_flags": ";".join(_enum(flag) for flag in result.status_flags),
        "requires_review": result.requires_review,
        "review_reasons": ";".join(result.review_reasons),
        "parcel_count": result.parcel_count,
        "parcel_count_confidence": result.parcel_count_confidence,
        "parcel_visibility": _enum(result.parcel_visibility),
        "label_count": result.label_count,
        "label_status": _enum(result.label_status),
        "awb_number": result.awb_number or "",
        "awb_number_excel": _excel_identifier(result.awb_number),
        "awb_source": _enum(result.awb_source),
        "awb_confidence": result.awb_confidence,
        "weight_original": (
            f"{result.weight_value:g} {result.weight_unit}"
            if result.weight_value is not None and result.weight_unit
            else ""
        ),
        "weight_value": result.weight_value,
        "weight_unit": result.weight_unit or "",
        "weight_grams": result.weight_grams,
        "weight_source": _enum(result.weight_source),
        "weight_confidence": result.weight_confidence,
        "length_cm": result.length_cm,
        "width_cm": result.width_cm,
        "height_cm": result.height_cm,
        "dimensions_text": result.dimensions_text or "",
        "dimensions_source": _enum(result.dimensions_source),
        "dimensions_confidence": result.dimensions_confidence,
        "recorded_volume": result.recorded_volume,
        "recorded_volume_unit": result.recorded_volume_unit or "",
        "recorded_volume_source": _enum(result.recorded_volume_source),
        "recorded_volume_confidence": result.recorded_volume_confidence,
        "location": result.location or "",
        "location_source": _enum(result.location_source),
        "location_confidence": result.location_confidence,
        "capture_timestamp": result.capture_timestamp.isoformat() if result.capture_timestamp else "",
        "capture_timestamp_raw": result.capture_timestamp_raw or "",
        "capture_timestamp_source": _enum(result.capture_timestamp_source),
        "capture_timestamp_confidence": result.capture_timestamp_confidence,
        "overlay_detected": result.overlay_detected,
        "barcode_values": ";".join(result.barcode_values),
        "overall_confidence": result.overall_confidence,
        "detector_version": result.detector_version or "",
        "detector_confidence": result.detector_confidence,
        "detector_used": result.detector_used,
        "ocr_engines_used": ";".join(result.ocr_engines_used),
        "vision_attempted": result.vision_attempted,
        "vision_succeeded": result.vision_succeeded,
        "vision_cache_hit": result.vision_cache_hit,
        "vision_error_category": result.vision_error_category or "",
        "vision_request_id": result.vision_request_id or "",
        "vision_attempt_count": result.vision_attempt_count,
        "vision_duration_ms": result.vision_duration_ms,
        "calibration_version": result.calibration_version,
        "warnings": ";".join(result.warnings),
        "processing_duration_ms": result.processing_duration_ms,
        "error_code": result.error_code or "",
        "error_message": result.error_message or "",
    }


def results_dataframe(results: list[ImageResult]) -> pd.DataFrame:
    dataframe = pd.DataFrame([result_row(result) for result in results], columns=CSV_COLUMNS)
    for column in ("awb_number", "awb_number_excel", "sha256", "image_id", "filename"):
        dataframe[column] = dataframe[column].astype("string")
    return dataframe


def write_csv(results: list[ImageResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    results_dataframe(results).to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
    )
    return path


def write_json(results: list[ImageResult], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for result in results:
        item = result.model_dump(mode="json")
        if result.annotated_image_path:
            item["annotated_image_path"] = (
                f"annotations/{Path(result.annotated_image_path).name}"
            )
        payload.append(item)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
