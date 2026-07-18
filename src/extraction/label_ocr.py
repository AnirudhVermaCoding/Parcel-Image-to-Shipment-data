from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec

import cv2
import numpy as np

from src.config import AppConfig
from src.extraction.field_parser import parse_ocr_fields
from src.extraction.label_detector import LabelCandidate
from src.extraction.overlay import _ocr, _rapid_ocr, configure_tesseract
from src.preprocessing.enhancement import adaptive_binary, clahe_gray, resize_for_ocr
from src.preprocessing.orientation import rotate_image
from src.schemas import ExtractionSource, FieldCandidate


@dataclass(slots=True)
class LabelOcrExtraction:
    texts: list[str] = field(default_factory=list)
    candidates: list[FieldCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    readable_candidates: int = 0
    ocr_available: bool = True
    engines_used: list[str] = field(default_factory=list)


def _variants(image: np.ndarray) -> dict[str, np.ndarray]:
    resized = resize_for_ocr(image, 420)
    gray = clahe_gray(resized)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    sharpened = cv2.filter2D(
        gray,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    return {
        "original": resized,
        "clahe": gray,
        "adaptive": adaptive_binary(resized),
        "otsu": otsu,
        "sharpened": sharpened,
    }


def _schedule() -> tuple[tuple[int, str, int], ...]:
    return (
        (0, "original", 6),
        (0, "clahe", 11),
        (0, "adaptive", 6),
        (0, "otsu", 12),
        (0, "sharpened", 7),
        (90, "clahe", 6),
        (180, "clahe", 6),
        (270, "clahe", 6),
        (90, "adaptive", 11),
        (270, "adaptive", 11),
    )


def _metadata_candidates(
    parsed: list[FieldCandidate],
    engine: str,
    variant: str,
    rotation: int,
    psm: int | None,
    crop_index: int,
) -> list[FieldCandidate]:
    return [
        item.model_copy(
            update={
                "metadata": {
                    **item.metadata,
                    "ocr_engine": engine,
                    "preprocessing_variant": variant,
                    "rotation": rotation,
                    "psm": psm,
                    "label_candidate": crop_index,
                }
            }
        )
        for item in parsed
    ]


def _strong_awb(candidates: list[FieldCandidate]) -> bool:
    return any(
        item.field_name == "awb_number" and item.confidence >= 0.86
        for item in candidates
    )


def extract_label_text(
    label_candidates: list[LabelCandidate],
    config: AppConfig,
    secondary_fallback: bool = True,
) -> LabelOcrExtraction:
    result = LabelOcrExtraction()
    if not label_candidates:
        return result

    use_tesseract = "tesseract" in config.ocr_engines and configure_tesseract(config)
    use_rapid = (
        "rapidocr" in config.ocr_engines
        and find_spec("rapidocr_onnxruntime") is not None
    )
    if not use_tesseract and not use_rapid:
        result.ocr_available = False
        result.warnings.append("No configured OCR engine is available for label OCR")
        return result

    if use_tesseract:
        result.engines_used.append("tesseract")
    if use_rapid:
        result.engines_used.append("rapidocr")

    # More candidates multiply OCR cost quickly. The detector/heuristic scores are already ranked.
    for candidate_index, candidate in enumerate(
        label_candidates[: config.ocr_max_label_candidates]
    ):
        crop_candidates: list[FieldCandidate] = []
        crop_texts: list[tuple[float, str]] = []
        variants_by_rotation: dict[int, dict[str, np.ndarray]] = {}
        if use_tesseract:
            pass_limit = config.ocr_max_passes_per_label if secondary_fallback else min(2, config.ocr_max_passes_per_label)
            for rotation, variant_name, psm in _schedule()[:pass_limit]:
                if rotation not in variants_by_rotation:
                    variants_by_rotation[rotation] = _variants(
                        rotate_image(candidate.crop, rotation)
                    )
                variants = variants_by_rotation[rotation]
                try:
                    text, confidence = _ocr(variants[variant_name], psm, config)
                except Exception as exc:
                    result.warnings.append(
                        f"Tesseract label OCR failed for candidate {candidate_index}: "
                        f"{type(exc).__name__}"
                    )
                    continue
                if text.strip():
                    crop_texts.append((confidence, text.strip()))
                parsed = parse_ocr_fields(
                    text,
                    ExtractionSource.LABEL_OCR,
                    confidence,
                    candidate.bbox,
                )
                crop_candidates.extend(
                    _metadata_candidates(
                        parsed.candidates,
                        "tesseract",
                        variant_name,
                        rotation,
                        psm,
                        candidate_index,
                    )
                )
                if _strong_awb(crop_candidates):
                    break
        # RapidOCR is a second opinion for unresolved crops, not an unconditional duplicate pass.
        if use_rapid and secondary_fallback and not _strong_awb(crop_candidates):
            for rotation in (0,):
                if rotation not in variants_by_rotation:
                    variants_by_rotation[rotation] = _variants(
                        rotate_image(candidate.crop, rotation)
                    )
                variants = variants_by_rotation[rotation]
                for variant_name in ("original",):
                    try:
                        text, confidence = _rapid_ocr(variants[variant_name])
                    except Exception as exc:
                        result.warnings.append(
                            f"RapidOCR failed for candidate {candidate_index}: {type(exc).__name__}"
                        )
                        continue
                    if text.strip():
                        crop_texts.append((confidence, text.strip()))
                    parsed = parse_ocr_fields(
                        text,
                        ExtractionSource.LABEL_OCR,
                        confidence,
                        candidate.bbox,
                    )
                    crop_candidates.extend(
                        _metadata_candidates(
                            parsed.candidates,
                            "rapidocr",
                            variant_name,
                            rotation,
                            None,
                            candidate_index,
                        )
                    )
                    if _strong_awb(crop_candidates):
                        break
                if _strong_awb(crop_candidates):
                    break

        # Retain all independent evidence while removing exact duplicate engine/variant emissions.
        seen: set[tuple] = set()
        for item in crop_candidates:
            key = (
                item.field_name,
                str(item.normalised_value),
                item.metadata.get("ocr_engine"),
                item.metadata.get("preprocessing_variant"),
                item.metadata.get("rotation"),
                item.metadata.get("psm"),
            )
            if key not in seen:
                seen.add(key)
                result.candidates.append(item)
        unique_texts: list[str] = []
        for _, text in sorted(crop_texts, reverse=True):
            if text not in unique_texts:
                unique_texts.append(text)
        result.texts.extend(unique_texts[:5])
        if crop_candidates:
            result.readable_candidates += 1
    return result
