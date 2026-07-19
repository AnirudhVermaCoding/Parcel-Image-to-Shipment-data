from __future__ import annotations

import pytest

from src.classification.status_engine import REVIEW_REASON_BY_FLAG
from src.reporting import presentation
from src.schemas import ImageResult, PrimaryStatus, StatusFlag


def _make(**kwargs) -> ImageResult:
    params = {"image_id": "x", "original_filename": "p.jpg"}
    params.update(kwargs)
    return ImageResult(**params)


@pytest.mark.parametrize("status", list(PrimaryStatus))
def test_every_primary_status_has_outcome(status):
    result = _make(primary_status=status, requires_review=False)
    descriptor = presentation.outcome(result)
    assert descriptor["label"]
    assert descriptor["tone"] in {"good", "warn", "bad"}
    assert descriptor["emoji"]


@pytest.mark.parametrize("flag", list(StatusFlag))
def test_every_status_flag_has_plain_text(flag):
    result = _make(primary_status=PrimaryStatus.REVIEW_REQUIRED, status_flags=[flag])
    reasons = presentation.plain_reasons(result)
    assert len(reasons) == 1
    assert reasons[0].strip()


def test_simple_dataframe_columns_and_rows():
    results = [_make(), _make(original_filename="q.jpg")]
    frame = presentation.simple_dataframe(results)
    assert list(frame.columns) == [
        "File",
        "Tracking number",
        "Weight",
        "Length (cm)",
        "Width (cm)",
        "Height (cm)",
        "Location",
        "Result",
        "Needs review",
        "Notes",
    ]
    assert len(frame) == 2


def test_review_required_is_warn_with_reasons():
    result = _make(
        primary_status=PrimaryStatus.REVIEW_REQUIRED,
        requires_review=True,
        review_reasons=["Please double-check the weight"],
    )
    assert presentation.outcome(result)["tone"] == "warn"
    reasons = presentation.plain_reasons(result)
    assert reasons == ["Please double-check the weight"]


@pytest.mark.parametrize(
    ("technical_reason", "expected"),
    [
        ("awb_not_found", "No tracking number was found"),
        ("parcel_partially_visible", "The parcel is only partly visible"),
        (
            "awb_not_label_verified",
            "The tracking number was read only from the machine overlay",
        ),
        ("future_reason_key", "Future reason key"),
    ],
)
def test_technical_review_reasons_are_plain_english(technical_reason, expected):
    result = _make(
        primary_status=PrimaryStatus.REVIEW_REQUIRED,
        requires_review=True,
        review_reasons=[technical_reason],
    )
    assert presentation.plain_reasons(result) == [expected]


def test_every_current_flag_and_review_reason_has_plain_language() -> None:
    assert {flag.value for flag in StatusFlag} <= set(presentation.FLAG_PLAIN)
    generated_reasons = set(REVIEW_REASON_BY_FLAG.values()) | {
        "processing_error",
        "awb_not_found",
        "vision_unavailable_local_evidence_insufficient",
    }
    assert generated_reasons <= set(presentation.REVIEW_REASON_PLAIN)


def test_requires_review_overrides_good_outcome():
    result = _make(
        primary_status=PrimaryStatus.SUCCESS_LABEL_VERIFIED,
        requires_review=True,
    )
    assert presentation.outcome(result)["tone"] == "warn"


def test_formatters_return_not_found_when_absent():
    result = _make()
    assert presentation.plain_awb(result) == "Not found"
    assert presentation.plain_weight(result) == "Not found"
    assert presentation.plain_size(result) == "Not found"
    assert presentation.plain_location(result) == "Not found"


def test_formatters_render_values():
    result = _make(
        awb_number="ABC123",
        weight_value=1.5,
        weight_unit="kg",
        length_cm=10.0,
        width_cm=20.0,
        height_cm=30.0,
        location="Mumbai",
    )
    assert presentation.plain_awb(result) == "ABC123"
    assert presentation.plain_weight(result) == "1.5 kg"
    assert presentation.plain_size(result) == "10 x 20 x 30 cm"
    assert presentation.plain_location(result) == "Mumbai"


def test_weight_grams_fallback():
    result = _make(weight_grams=500.0)
    assert presentation.plain_weight(result) == "500 g"


def test_csv_bytes_roundtrip():
    data = presentation.simple_csv_bytes([_make()])
    assert isinstance(data, bytes)
    assert data.decode("utf-8-sig").splitlines()[0].startswith("File,")
