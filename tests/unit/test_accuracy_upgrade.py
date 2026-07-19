from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
import time

import cv2
import numpy as np
import pandas as pd
import httpx
from openai import APIConnectionError, APIStatusError, RateLimitError

from scripts.evaluate_accuracy import evaluate
from scripts.validate_yolo_annotations import validate
from src.classification.onnx_detector import (
    CLASS_NAMES,
    OnnxParcelDetector,
    _internal_class_name,
    _model_class_names,
    detect_with_onnx,
)
from src.classification.status_engine import assign_status
from src.extraction.courier_rules import matching_rules, validate_rule_file
from src.extraction.label_detector import LabelCandidate
from src.extraction.label_ocr import _schedule, _variants, extract_label_text
from src.extraction.vision_fallback import XaiVisionProvider, _category_for_exception
from src.jobs.manager import LimitedVisionProvider
from src.schemas import BoundingBox, ImageResult, LabelStatus, ParcelVisibility, VisionAnalysis, VisionDiagnostic, VisionObservation


def test_missing_detector_falls_back_explicitly(test_config) -> None:
    test_config.detector_model_path = "missing/model.onnx"
    evidence = detect_with_onnx(np.zeros((100, 100, 3), dtype=np.uint8), test_config)
    assert not evidence.available
    assert evidence.error == "model_not_found"


def test_yolo_output_orientation_is_normalised() -> None:
    rows = np.zeros((4 + len(CLASS_NAMES), 10), dtype=np.float32)
    assert OnnxParcelDetector._rows(rows).shape == (10, 4 + len(CLASS_NAMES))


def test_detector_reads_embedded_class_metadata() -> None:
    names = _model_class_names("{0: 'barcode', 1: 'cardboard box', 2: 'person'}")
    assert names == ("barcode", "cardboard box", "person")
    assert _internal_class_name(names[0]) == "barcode_region"
    assert _internal_class_name(names[1]) == "parcel_full"
    assert _internal_class_name(names[2]) is None


def test_generic_yolo_output_orientation_uses_model_class_count() -> None:
    rows = np.zeros((4 + 20, 8400), dtype=np.float32)
    assert OnnxParcelDetector._rows(rows, 20).shape == (8400, 24)


def test_courier_rules_validate_numeric_awb(test_config) -> None:
    assert validate_rule_file(test_config.courier_rules_path) == []
    assert "numeric_14" in matching_rules(
        "11112222333344", test_config.courier_rules_path, "Explicit AWB anchor"
    )


def test_review_reasons_are_explicit(test_config) -> None:
    result = ImageResult(
        image_id="one",
        original_filename="one.jpg",
        parcel_count=1,
        parcel_visibility=ParcelVisibility.PARTIAL,
        label_status=LabelStatus.LABEL_VISIBLE_READABLE,
    )
    assign_status(result, test_config)
    assert "parcel_partially_visible" in result.review_reasons
    assert "awb_not_found" in result.review_reasons


def test_multi_engine_label_ocr_records_provenance(monkeypatch, test_config) -> None:
    candidate = LabelCandidate(
        bbox=BoundingBox(x=0, y=0, width=120, height=80),
        score=0.9,
        crop=np.full((80, 120, 3), 255, dtype=np.uint8),
        rectangularity=1,
        brightness=1,
        edge_density=0,
        barcode_pattern=0,
    )
    monkeypatch.setattr("src.extraction.label_ocr.configure_tesseract", lambda config: True)
    monkeypatch.setattr(
        "src.extraction.label_ocr._ocr",
        lambda image, psm, config: ("AWB No: 11112222333344", 0.95),
    )
    test_config.ocr_engines = ("tesseract",)
    extracted = extract_label_text([candidate], test_config)
    assert extracted.candidates
    assert extracted.candidates[0].metadata["ocr_engine"] == "tesseract"
    assert extracted.candidates[0].metadata["preprocessing_variant"] == "original"


def test_ocr_schedule_covers_required_variants_and_psms() -> None:
    variants = _variants(np.full((50, 100, 3), 255, dtype=np.uint8))
    assert {"original", "clahe", "adaptive", "otsu", "sharpened"} <= set(variants)
    assert {6, 7, 11, 12} <= {psm for _, _, psm in _schedule()}


def test_vision_cache_avoids_second_api_call(tmp_path, test_config) -> None:
    test_config.runtime_dir = tmp_path
    test_config.xai_api_key = "test-key"
    test_config.vision_cache_enabled = True
    provider = XaiVisionProvider(test_config)
    observation = {
        "parcel_count": 1,
        "parcel_count_confidence": 0.9,
        "parcel_visibility": "FULL",
        "label_status": "VISIBLE",
        "confidence": 0.8,
    }
    calls = {"count": 0}

    def create(**kwargs):
        calls["count"] += 1
        return SimpleNamespace(id="resp_test", output_text=__import__("json").dumps(observation))

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    ok, encoded = cv2.imencode(".jpg", np.full((64, 64, 3), 255, dtype=np.uint8))
    assert ok
    first = provider.analyze(encoded.tobytes())
    second = provider.analyze(encoded.tobytes())
    assert first.diagnostic.success
    assert second.diagnostic.cache_hit
    assert calls["count"] == 1


def test_vision_error_categories_are_safe() -> None:
    request = httpx.Request("POST", "https://api.x.ai/v1/responses")
    response = httpx.Response(401, request=request)
    category, status, _, retryable = _category_for_exception(
        APIStatusError("secret provider body", response=response, body={"secret": "value"})
    )
    assert (category, status, retryable) == ("authentication", 401, False)
    response = httpx.Response(429, request=request)
    assert _category_for_exception(RateLimitError("limited", response=response, body=None))[0] == "rate_limit"
    assert _category_for_exception(APIConnectionError(request=request))[0] == "network"


def test_vision_semaphore_limits_concurrency(tmp_path, test_config) -> None:
    test_config.runtime_dir = tmp_path
    test_config.xai_api_key = "test-key"
    test_config.vision_cache_enabled = False
    test_config.xai_max_concurrency = 2
    provider = XaiVisionProvider(test_config)
    active = 0
    maximum = 0
    lock = threading.Lock()

    def create(**kwargs):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return SimpleNamespace(
            id="resp",
            output_text='{"parcel_count":null,"parcel_count_confidence":0,"parcel_visibility":"UNCERTAIN","label_status":"UNCERTAIN","confidence":0}',
        )

    provider.client = SimpleNamespace(responses=SimpleNamespace(create=create))
    images = []
    for value in range(4):
        ok, encoded = cv2.imencode(".jpg", np.full((32, 32, 3), value, dtype=np.uint8))
        assert ok
        images.append(encoded.tobytes())
    threads = [threading.Thread(target=provider.analyze, args=(item,)) for item in images]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert maximum == 2


def test_per_job_vision_cap(monkeypatch, test_config) -> None:
    class FakeProvider:
        def __init__(self, config):
            pass

        def cached_analysis(self, *args, **kwargs):
            return None

        def analyze(self, *args, **kwargs):
            return VisionAnalysis(
                observation=VisionObservation(),
                diagnostic=VisionDiagnostic(success=True, category="success"),
            )

    monkeypatch.setattr("src.jobs.manager.XaiVisionProvider", FakeProvider)
    test_config.xai_max_images_per_job = 1
    limited = LimitedVisionProvider(test_config)
    limited.analyze(b"one")
    try:
        limited.analyze(b"two")
    except Exception as exc:
        assert exc.diagnostic.category == "job_limit"
    else:
        raise AssertionError("Expected per-job limit")


def test_accuracy_metrics_and_threshold_sweep() -> None:
    truth = pd.DataFrame(
        [
            {"filename": "a.jpg", "expected_awb": "11111111", "annotated": "true", "split": "val"},
            {"filename": "b.jpg", "expected_awb": "22222222", "annotated": "true", "split": "val"},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"filename": "a.jpg", "awb_number": "11111111", "awb_confidence": 0.95, "requires_review": False, "processing_status": "COMPLETED"},
            {"filename": "b.jpg", "awb_number": "", "awb_confidence": 0.0, "requires_review": True, "processing_status": "COMPLETED"},
        ]
    )
    metrics, failures, sweep = evaluate(truth, predictions, "val")
    assert metrics["clean_awb_precision"] == 1.0
    assert metrics["review_rate"] == 0.5
    assert failures.empty
    assert not sweep.empty


def test_annotation_validator_catches_invalid_class(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    cv2.imwrite(str(images / "one.jpg"), np.zeros((20, 20, 3), dtype=np.uint8))
    (labels / "one.txt").write_text("99 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    assert any(error.startswith("invalid_class") for error in validate(tmp_path))
