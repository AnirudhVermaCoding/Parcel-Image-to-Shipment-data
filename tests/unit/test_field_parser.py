from __future__ import annotations

from src.extraction.field_parser import normalise_awb, parse_ocr_fields, parse_timestamp
from src.schemas import ExtractionSource


def test_overlay_parser_normalises_units_and_timestamp() -> None:
    text = """
    Location: Bangalore_Hoskote GW (Karnataka)
    AW8 No.: 55556666777788
    Length: 570 mm
    Width: 13.307 in
    Height: 0.424 m
    R Vol: 80223.000 cm (cubic)
    W eight: 7.020 kg
    Time: 7/13/2025 5:53:26PM
    """
    parsed = parse_ocr_fields(text, ExtractionSource.OVERLAY_OCR, 0.93)
    values = {candidate.field_name: candidate.normalised_value for candidate in parsed.candidates}
    assert values["location"] == "Bangalore_Hoskote_GW (Karnataka)"
    assert values["awb_number"] == "55556666777788"
    assert values["length_cm"] == 57.0
    assert round(float(values["width_cm"]), 2) == 33.8
    assert values["height_cm"] == 42.4
    assert values["weight"] == 7020.0
    assert values["capture_timestamp"] == "2025-07-13T17:53:26"


def test_timestamp_accepts_missing_spaces() -> None:
    assert parse_timestamp("7/17/20256:11:38PM").isoformat() == "2025-07-17T18:11:38"


def test_awb_validation_is_conservative() -> None:
    assert normalise_awb("1111-2222-333344") == "11112222333344"
    assert normalise_awb("123") is None
    assert normalise_awb("TRACKING") is None
