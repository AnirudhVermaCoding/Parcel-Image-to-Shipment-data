from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from importlib.util import find_spec

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from src.config import AppConfig
from src.extraction.field_parser import ParsedFields, parse_ocr_fields
from src.preprocessing.enhancement import adaptive_binary, resize_for_ocr
from src.preprocessing.masks import yellow_text_mask
from src.schemas import BoundingBox, ExtractionSource, FieldCandidate

_RAPID_LOCAL = threading.local()
_ACTIVE_ENGINE = "NONE"


@dataclass(slots=True)
class OverlayExtraction:
    detected: bool = False
    bbox: BoundingBox | None = None
    raw_text: str = ""
    confidence: float = 0.0
    variant: str | None = None
    candidates: list[FieldCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ocr_available: bool = True


def ocr_engine_name(config: AppConfig) -> str:
    global _ACTIVE_ENGINE
    if config.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = config.tesseract_cmd
        _ACTIVE_ENGINE = "TESSERACT"
    elif shutil.which("tesseract") is not None:
        _ACTIVE_ENGINE = "TESSERACT"
    elif find_spec("rapidocr_onnxruntime") is not None:
        _ACTIVE_ENGINE = "RAPIDOCR"
    else:
        _ACTIVE_ENGINE = "NONE"
    return _ACTIVE_ENGINE


def configure_tesseract(config: AppConfig) -> bool:
    return ocr_engine_name(config) != "NONE"


def _overlay_bbox(mask: np.ndarray, config: AppConfig) -> BoundingBox | None:
    count = int(cv2.countNonZero(mask))
    if count < config.min_yellow_pixels:
        return None
    height, width = mask.shape
    horizontal = max(5, width // 120)
    vertical = max(3, height // 180)
    dilated = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal, vertical)),
        iterations=2,
    )
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    useful = []
    minimum_area = max(20, int(width * height * 0.00002))
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        yellow_count = int(cv2.countNonZero(mask[y : y + box_height, x : x + box_width]))
        if yellow_count >= minimum_area:
            useful.append((x, y, box_width, box_height, yellow_count))
    if not useful:
        points = cv2.findNonZero(mask)
        if points is None:
            return None
        x, y, box_width, box_height = cv2.boundingRect(points)
    else:
        useful.sort(key=lambda item: item[4], reverse=True)
        selected = useful[: min(12, len(useful))]
        x = min(item[0] for item in selected)
        y = min(item[1] for item in selected)
        x2 = max(item[0] + item[2] for item in selected)
        y2 = max(item[1] + item[3] for item in selected)
        box_width, box_height = x2 - x, y2 - y
    padding = max(6, int(max(width, height) * config.overlay_expand_ratio))
    x = max(0, x - padding)
    y = max(0, y - padding)
    x2 = min(width, x + box_width + padding * 2)
    y2 = min(height, y + box_height + padding * 2)
    return BoundingBox(x=x, y=y, width=x2 - x, height=y2 - y)


def _rapid_ocr(image: np.ndarray) -> tuple[str, float]:
    from rapidocr_onnxruntime import RapidOCR

    engine = getattr(_RAPID_LOCAL, "engine", None)
    if engine is None:
        engine = RapidOCR()
        _RAPID_LOCAL.engine = engine
    detections, _ = engine(image)
    if not detections:
        return "", 0.0
    items = []
    for detection in detections:
        box, token, confidence = detection
        top = min(point[1] for point in box)
        bottom = max(point[1] for point in box)
        left = min(point[0] for point in box)
        items.append(
            {
                "token": str(token).strip(),
                "confidence": float(confidence),
                "center_y": (top + bottom) / 2,
                "height": max(1.0, bottom - top),
                "left": left,
            }
        )
    items.sort(key=lambda item: (item["center_y"], item["left"]))
    lines: list[list[dict]] = []
    for item in items:
        if not item["token"]:
            continue
        if not lines:
            lines.append([item])
            continue
        current = lines[-1]
        current_center = sum(part["center_y"] for part in current) / len(current)
        tolerance = max(item["height"], max(part["height"] for part in current)) * 0.62
        if abs(item["center_y"] - current_center) <= tolerance:
            current.append(item)
        else:
            lines.append([item])
    text = "\n".join(
        " ".join(part["token"] for part in sorted(line, key=lambda part: part["left"]))
        for line in lines
    )
    confidences = [item["confidence"] for item in items]
    return text, sum(confidences) / len(confidences) if confidences else 0.0


def _ocr(image: np.ndarray, psm: int, config: AppConfig | None = None) -> tuple[str, float]:
    engine_name = ocr_engine_name(config or AppConfig.from_env())
    if engine_name == "RAPIDOCR":
        return _rapid_ocr(image)
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config=f"--oem 3 --psm {psm}",
        lang="eng",
        timeout=(config or AppConfig.from_env()).ocr_timeout_seconds,
    )
    tokens: list[str] = []
    confidences: list[float] = []
    line_parts: dict[tuple[int, int, int], list[tuple[int, str]]] = {}
    for index, token in enumerate(data["text"]):
        token = token.strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1
        if not token:
            continue
        key = (data["block_num"][index], data["par_num"][index], data["line_num"][index])
        line_parts.setdefault(key, []).append((data["left"][index], token))
        tokens.append(token)
        if confidence >= 0:
            confidences.append(confidence)
    lines = [
        " ".join(token for _, token in sorted(parts))
        for _, parts in sorted(line_parts.items())
    ]
    text = "\n".join(lines) if lines else " ".join(tokens)
    mean_confidence = sum(confidences) / len(confidences) / 100 if confidences else 0.0
    return text, max(0.0, min(1.0, mean_confidence))


def extract_overlay(bgr: np.ndarray, config: AppConfig) -> OverlayExtraction:
    mask = yellow_text_mask(bgr, config)
    bbox = _overlay_bbox(mask, config)
    if bbox is None:
        return OverlayExtraction(detected=False)
    result = OverlayExtraction(detected=True, bbox=bbox)
    engine_name = ocr_engine_name(config)
    if engine_name == "NONE":
        result.ocr_available = False
        result.warnings.append("Tesseract OCR is not available")
        return result
    if engine_name == "RAPIDOCR":
        result.warnings.append(
            "Tesseract binary unavailable; using the development RapidOCR fallback"
        )

    x, y, width, height = bbox.x, bbox.y, bbox.width, bbox.height
    crop = bgr[y : y + height, x : x + width]
    mask_crop = mask[y : y + height, x : x + width]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    yellow_on_black = mask_crop
    yellow_on_white = cv2.bitwise_not(mask_crop)
    variants = {
        "original": resize_for_ocr(crop),
        "yellow_mask": resize_for_ocr(yellow_on_black),
        "yellow_inverted": resize_for_ocr(yellow_on_white),
        "grayscale": resize_for_ocr(gray),
        "adaptive_binary": resize_for_ocr(adaptive_binary(crop)),
    }
    best_score = -1.0
    best_parsed = ParsedFields()
    if engine_name == "RAPIDOCR":
        variants = {"original": crop}
    psms = (6,) if engine_name == "RAPIDOCR" else (6, 11)
    for variant_name, variant in variants.items():
        for psm in psms:
            try:
                text, ocr_confidence = _ocr(variant, psm, config)
            except Exception as exc:
                result.warnings.append(f"Overlay OCR failed for {variant_name}/psm{psm}: {type(exc).__name__}")
                continue
            parsed = parse_ocr_fields(
                text,
                ExtractionSource.OVERLAY_OCR,
                ocr_confidence,
                bbox,
            )
            score = parsed.anchor_count * 20 + len(parsed.candidates) * 25 + ocr_confidence * 20
            if score > best_score:
                best_score = score
                result.raw_text = text
                result.confidence = ocr_confidence
                result.variant = f"{variant_name}:psm{psm}"
                best_parsed = parsed
    result.candidates = best_parsed.candidates
    result.warnings.extend(best_parsed.warnings)
    if best_parsed.anchor_count < 4:
        result.warnings.append("Overlay OCR found only partial anchor coverage")
    return result
