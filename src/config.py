from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(slots=True)
class AppConfig:
    runtime_dir: Path = Path("runtime")
    max_file_bytes: int = 25 * 1024 * 1024
    max_pixels: int = 50_000_000
    max_dimension: int = 12_000
    max_batch_images: int = 50
    max_zip_entries: int = 500
    max_zip_uncompressed_bytes: int = 250 * 1024 * 1024
    max_zip_ratio: float = 100.0
    default_workers: int = 2
    max_workers: int = 8
    hosted_mode: bool = False
    generate_annotations: bool = False
    job_retention_hours: int = 24
    job_stuck_seconds: int = 300
    ocr_timeout_seconds: int = 30

    yellow_hsv_lower: tuple[int, int, int] = (15, 80, 80)
    yellow_hsv_upper: tuple[int, int, int] = (42, 255, 255)
    min_yellow_pixels: int = 30
    overlay_expand_ratio: float = 0.012
    max_label_candidates: int = 5
    processing_max_side: int = 4096
    parcel_detection_max_side: int = 1800

    detector_model_path: str = "models/parcel_detector.onnx"
    detector_confidence_threshold: float = 0.35
    detector_iou_threshold: float = 0.45
    detector_input_size: int = 640

    awb_clean_threshold: float = 0.90
    awb_review_threshold: float = 0.75
    field_accept_threshold: float = 0.75
    field_review_threshold: float = 0.50
    low_blur_threshold: float = 60.0
    low_contrast_threshold: float = 18.0
    underexposure_threshold_pct: float = 82.0
    overexposure_threshold_pct: float = 35.0

    enable_xai_fallback: bool = False
    xai_api_key: str | None = None
    xai_model: str = "grok-4.5"
    xai_timeout_seconds: float = 45.0
    xai_max_retries: int = 2
    xai_max_images_per_job: int = 10
    xai_max_image_side: int = 1600
    xai_max_concurrency: int = 2
    xai_base_url: str = "https://api.x.ai/v1"
    vision_cache_enabled: bool = True
    vision_prompt_version: str = "parcel-observation-v2"

    ocr_engines: tuple[str, ...] = ("tesseract",)
    ocr_max_passes_per_label: int = 6
    ocr_max_label_candidates: int = 1
    calibration_version: str = "conservative-v1"
    courier_rules_path: str = "config/courier_rules.json"
    calibration_path: str = "config/calibration.json"

    tesseract_cmd: str | None = None
    extra: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "AppConfig":
        hosted = _bool_env("STREAMLIT_CLOUD", False)
        batch_default = 500
        worker_default = 2 if hosted else min(4, max(1, (os.cpu_count() or 2) // 2))
        requested_default_workers = max(1, _int_env("MAX_WORKERS", worker_default))
        requested_worker_limit = max(
            1,
            _int_env(
                "MAX_WORKER_LIMIT",
                8 if hosted else max(8, requested_default_workers),
            ),
        )
        worker_limit = (
            min(8, requested_worker_limit)
            if hosted
            else max(requested_default_workers, requested_worker_limit)
        )
        engines = tuple(
            value.strip().lower()
            for value in os.getenv("OCR_ENGINES", "tesseract").split(",")
            if value.strip()
        )
        calibration_path = os.getenv("CALIBRATION_PATH", "config/calibration.json")
        calibrated_threshold = 0.90
        calibrated_version = os.getenv("CALIBRATION_VERSION", "conservative-v1")
        try:
            payload = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
            if payload.get("frozen") is True:
                calibrated_threshold = float(payload["awb_clean_threshold"])
                calibrated_version = str(payload.get("version") or calibrated_version)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            pass
        return cls(
            runtime_dir=Path(os.getenv("RUNTIME_DIR", "runtime")),
            max_batch_images=_int_env("MAX_BATCH_IMAGES", batch_default),
            default_workers=min(requested_default_workers, worker_limit),
            max_workers=worker_limit,
            hosted_mode=hosted,
            generate_annotations=_bool_env("GENERATE_ANNOTATIONS", False),
            awb_clean_threshold=_float_env("AWB_CLEAN_THRESHOLD", calibrated_threshold),
            job_retention_hours=_int_env("JOB_RETENTION_HOURS", 24),
            job_stuck_seconds=max(60, _int_env("JOB_STUCK_SECONDS", 300)),
            ocr_timeout_seconds=max(5, _int_env("OCR_TIMEOUT_SECONDS", 30)),
            detector_model_path=os.getenv("DETECTOR_MODEL_PATH", "models/parcel_detector.onnx"),
            detector_confidence_threshold=_float_env("DETECTOR_CONFIDENCE_THRESHOLD", 0.35),
            detector_iou_threshold=_float_env("DETECTOR_IOU_THRESHOLD", 0.45),
            enable_xai_fallback=_bool_env("ENABLE_XAI_FALLBACK", False),
            xai_api_key=os.getenv("XAI_API_KEY") or None,
            xai_model=os.getenv("XAI_MODEL", "grok-4.5"),
            xai_timeout_seconds=_float_env("XAI_TIMEOUT_SECONDS", 45.0),
            xai_max_retries=_int_env("XAI_MAX_RETRIES", 2),
            xai_max_images_per_job=_int_env("XAI_MAX_IMAGES_PER_JOB", 50),
            xai_max_concurrency=max(1, _int_env("XAI_MAX_CONCURRENCY", 2)),
            xai_base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").rstrip("/"),
            vision_cache_enabled=_bool_env("VISION_CACHE_ENABLED", True),
            vision_prompt_version=os.getenv("VISION_PROMPT_VERSION", "parcel-observation-v2"),
            ocr_engines=engines or ("tesseract",),
            ocr_max_passes_per_label=max(1, _int_env("OCR_MAX_PASSES_PER_LABEL", 6)),
            ocr_max_label_candidates=max(1, _int_env("OCR_MAX_LABEL_CANDIDATES", 1)),
            calibration_version=calibrated_version,
            courier_rules_path=os.getenv("COURIER_RULES_PATH", "config/courier_rules.json"),
            calibration_path=calibration_path,
            tesseract_cmd=os.getenv("TESSERACT_CMD") or None,
        )

    def ensure_runtime_dirs(self) -> None:
        for child in ("jobs", "uploads", "reports", "annotations"):
            (self.runtime_dir / child).mkdir(parents=True, exist_ok=True)
