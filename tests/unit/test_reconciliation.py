from __future__ import annotations

from src.reconciliation.candidates import reconcile
from src.schemas import ExtractionSource, FieldCandidate


def candidate(value: str, source: ExtractionSource, confidence: float) -> FieldCandidate:
    return FieldCandidate(
        field_name="awb_number",
        value=value,
        normalised_value=value,
        source=source,
        confidence=confidence,
    )


def test_independent_agreement_increases_confidence() -> None:
    result = reconcile(
        "awb_number",
        [
            candidate("12345678901", ExtractionSource.OVERLAY_OCR, 0.85),
            candidate("12345678901", ExtractionSource.BARCODE, 0.88),
        ],
    )
    assert result.candidate.normalised_value == "12345678901"
    assert result.confidence >= 0.95
    assert not result.conflict


def test_conflicting_high_confidence_awbs_require_conflict() -> None:
    result = reconcile(
        "awb_number",
        [
            candidate("12345678901", ExtractionSource.OVERLAY_OCR, 0.94),
            candidate("99876543210", ExtractionSource.BARCODE, 0.93),
        ],
    )
    assert result.conflict
    assert result.confidence < 0.90

