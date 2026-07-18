from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from src.schemas import BoundingBox, ExtractionSource, FieldCandidate

NUMERIC_CORRECTIONS = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
    }
)


@dataclass(slots=True)
class ParsedFields:
    candidates: list[FieldCandidate] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    anchor_count: int = 0


ANCHOR_PATTERNS = {
    "location": re.compile(r"\bLocati[o0]n\b", re.IGNORECASE),
    "awb_number": re.compile(r"\bAW[8B]\s*(?:No|Number)?\.?\b", re.IGNORECASE),
    "length_cm": re.compile(r"\bLength\b", re.IGNORECASE),
    "width_cm": re.compile(r"\bW\s*idth\b", re.IGNORECASE),
    "height_cm": re.compile(r"\bHeight\b", re.IGNORECASE),
    "recorded_volume": re.compile(r"\bR\.?\s*Vol\.?\b", re.IGNORECASE),
    "weight": re.compile(r"\bW\s*eight\b", re.IGNORECASE),
    "capture_timestamp": re.compile(r"\bTime\b", re.IGNORECASE),
}


def _field_line(text: str, anchor: re.Pattern[str]) -> str | None:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        match = anchor.search(line)
        if match:
            value = line[match.end() :].lstrip(" :.-")
            if value:
                return value
            if index + 1 < len(lines):
                return lines[index + 1]
    flattened = re.sub(r"\s+", " ", text)
    match = anchor.search(flattened)
    if not match:
        return None
    tail = flattened[match.end() :].lstrip(" :.-")
    next_anchor_positions = [
        found.start()
        for pattern in ANCHOR_PATTERNS.values()
        if (found := pattern.search(tail)) is not None
    ]
    if next_anchor_positions:
        tail = tail[: min(next_anchor_positions)]
    return tail.strip() or None


def _correct_numeric(value: str, warnings: list[str]) -> str:
    corrected = value.translate(NUMERIC_CORRECTIONS)
    if corrected != value:
        warnings.append(f"Numeric OCR correction applied: {value!r} -> {corrected!r}")
    return corrected


def _number(value: str, warnings: list[str]) -> float | None:
    corrected = _correct_numeric(value, warnings)
    match = re.search(r"[-+]?\d+(?:[.,]\d+)?", corrected)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def normalise_awb(value: str) -> str | None:
    value = value.strip().upper()
    value = re.sub(r"[^A-Z0-9 -]", "", value)
    value = re.sub(r"[\s-]+", "", value)
    if not 8 <= len(value) <= 30:
        return None
    if sum(character.isdigit() for character in value) < 6:
        return None
    if not value.isalnum():
        return None
    return value


def parse_timestamp(value: str) -> datetime | None:
    cleaned = re.sub(r"\s+", " ", value.strip())
    cleaned = re.sub(r"(?<=\d{4})(?=\d{1,2}:\d{2}:\d{2})", " ", cleaned)
    cleaned = re.sub(r"(?<=\d)(?=[AP]M$)", " ", cleaned, flags=re.IGNORECASE)
    formats = (
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    )
    for pattern in formats:
        try:
            return datetime.strptime(cleaned, pattern)
        except ValueError:
            continue
    return None


def parse_ocr_fields(
    text: str,
    source: ExtractionSource,
    ocr_confidence: float,
    bbox: BoundingBox | None = None,
) -> ParsedFields:
    result = ParsedFields()
    result.anchor_count = sum(bool(pattern.search(text)) for pattern in ANCHOR_PATTERNS.values())

    location = _field_line(text, ANCHOR_PATTERNS["location"])
    if location:
        location = re.split(r"\s+(?=AW[8B]\b|Length\b|Width\b|Height\b|R\.?\s*Vol|W\s*eight\b|Time\b)", location, 1, flags=re.IGNORECASE)[0]
        location = location.strip(" :.-")
        compact_letters = re.sub(r"[\s_()]+", "", location)
        if "_" in location or compact_letters.isupper():
            location = re.sub(r"\s+(?!\()", "_", location)
        if 2 <= len(location) <= 100:
            result.candidates.append(
                FieldCandidate(
                    field_name="location",
                    value=location,
                    normalised_value=location,
                    source=source,
                    confidence=min(0.99, ocr_confidence + 0.05),
                    evidence="Explicit Location anchor",
                    bounding_box=bbox,
                )
            )

    awb_raw = _field_line(text, ANCHOR_PATTERNS["awb_number"])
    if awb_raw:
        token_match = re.search(r"[A-Za-z0-9][A-Za-z0-9 -]{6,34}", awb_raw)
        if token_match:
            warnings: list[str] = []
            raw_token = token_match.group(0).strip()
            corrected = _correct_numeric(raw_token, warnings) if not re.search(r"[A-HJ-NP-RT-Z]", raw_token.upper()) else raw_token
            awb = normalise_awb(corrected)
            if awb:
                result.candidates.append(
                    FieldCandidate(
                        field_name="awb_number",
                        value=raw_token,
                        normalised_value=awb,
                        source=source,
                        confidence=min(0.99, ocr_confidence + 0.05 - 0.15 * len(warnings)),
                        evidence="Explicit AWB anchor",
                        bounding_box=bbox,
                        warnings=warnings,
                    )
                )
                result.warnings.extend(warnings)

    for field_name in ("length_cm", "width_cm", "height_cm"):
        raw = _field_line(text, ANCHOR_PATTERNS[field_name])
        if not raw:
            continue
        warnings = []
        number = _number(raw, warnings)
        unit_match = re.search(r"\b(mm|cm|m|in|inch|inches)\b", raw, re.IGNORECASE)
        unit = unit_match.group(1).lower() if unit_match else "cm"
        if number is None:
            continue
        if unit == "mm":
            normalised = number / 10
        elif unit == "m":
            normalised = number * 100
        elif unit in {"in", "inch", "inches"}:
            normalised = number * 2.54
        else:
            normalised = number
        result.candidates.append(
            FieldCandidate(
                field_name=field_name,
                value=number,
                normalised_value=round(normalised, 4),
                source=source,
                confidence=min(0.99, ocr_confidence + 0.05 - 0.15 * len(warnings)),
                evidence=f"Explicit {field_name.removesuffix('_cm')} anchor",
                bounding_box=bbox,
                warnings=warnings,
                metadata={"original_unit": unit},
            )
        )
        result.warnings.extend(warnings)

    volume_raw = _field_line(text, ANCHOR_PATTERNS["recorded_volume"])
    if volume_raw:
        warnings = []
        volume = _number(volume_raw, warnings)
        if volume is not None:
            result.candidates.append(
                FieldCandidate(
                    field_name="recorded_volume",
                    value=volume,
                    normalised_value=volume,
                    source=source,
                    confidence=min(0.99, ocr_confidence + 0.05 - 0.15 * len(warnings)),
                    evidence="Explicit R.Vol anchor",
                    bounding_box=bbox,
                    warnings=warnings,
                    metadata={"unit": "cm3"},
                )
            )
            result.warnings.extend(warnings)

    weight_raw = _field_line(text, ANCHOR_PATTERNS["weight"])
    if weight_raw:
        warnings = []
        weight = _number(weight_raw, warnings)
        unit_match = re.search(r"\b(kg|kgs|g|gm|gms|gram|grams)\b", weight_raw, re.IGNORECASE)
        if weight is not None and unit_match:
            unit = unit_match.group(1).lower()
            grams = weight * 1000 if unit in {"kg", "kgs"} else weight
            normalised_unit = "kg" if unit in {"kg", "kgs"} else "g"
            result.candidates.append(
                FieldCandidate(
                    field_name="weight",
                    value=weight,
                    normalised_value=round(grams, 4),
                    source=source,
                    confidence=min(0.99, ocr_confidence + 0.05 - 0.15 * len(warnings)),
                    evidence="Explicit Weight anchor and unit",
                    bounding_box=bbox,
                    warnings=warnings,
                    metadata={"original_unit": normalised_unit, "grams": grams},
                )
            )
            result.warnings.extend(warnings)

    time_raw = _field_line(text, ANCHOR_PATTERNS["capture_timestamp"])
    if time_raw:
        timestamp_match = re.search(
            r"\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\s+\d{1,2}:\d{2}:\d{2}(?:\s*[AP]M)?",
            time_raw,
            re.IGNORECASE,
        )
        raw_timestamp = timestamp_match.group(0) if timestamp_match else time_raw
        timestamp = parse_timestamp(raw_timestamp)
        if timestamp:
            result.candidates.append(
                FieldCandidate(
                    field_name="capture_timestamp",
                    value=raw_timestamp,
                    normalised_value=timestamp.isoformat(),
                    source=source,
                    confidence=min(0.99, ocr_confidence + 0.05),
                    evidence="Explicit Time anchor and parsed timestamp",
                    bounding_box=bbox,
                )
            )
    return result
