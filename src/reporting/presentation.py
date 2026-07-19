from __future__ import annotations

import pandas as pd


def _enum(value) -> str:
    return value.value if hasattr(value, "value") else str(value)


_GOOD_STATUSES = {
    "SUCCESS_LABEL_VERIFIED",
    "SUCCESS_OVERLAY_AND_LABEL",
    "SUCCESS_OVERLAY_ONLY",
    "SUCCESS_PARTIAL_FIELDS",
}

_GOOD_DESCRIPTOR = {"label": "Read successfully", "emoji": "✅", "tone": "good"}
_REVIEW_DESCRIPTOR = {"label": "Needs a quick check", "emoji": "\U0001f50e", "tone": "warn"}
_REJECTED_DESCRIPTOR = {"label": "Not a usable photo", "emoji": "\U0001f6ab", "tone": "bad"}
_ERROR_DESCRIPTOR = {"label": "Could not process", "emoji": "⚠️", "tone": "bad"}


def outcome(result) -> dict:
    """Return a plain-English outcome descriptor for an ImageResult.

    The descriptor has keys "label", "emoji" and "tone" where tone is one of
    "good", "warn" or "bad".
    """
    status = _enum(result.primary_status)

    if status in _GOOD_STATUSES:
        descriptor = dict(_GOOD_DESCRIPTOR)
    elif status == "REVIEW_REQUIRED":
        descriptor = dict(_REVIEW_DESCRIPTOR)
    elif status == "REJECTED":
        descriptor = dict(_REJECTED_DESCRIPTOR)
    elif status == "PROCESSING_ERROR":
        descriptor = dict(_ERROR_DESCRIPTOR)
    else:
        # Unknown / future status: treat conservatively as needing a check.
        descriptor = dict(_REVIEW_DESCRIPTOR)

    # Needs-check wins: if the result flags itself for review but mapped to a
    # "good" outcome, surface the review descriptor instead.
    if getattr(result, "requires_review", False) and descriptor["tone"] == "good":
        descriptor = dict(_REVIEW_DESCRIPTOR)

    return descriptor


FLAG_PLAIN: dict[str, str] = {
    "NO_PARCEL": "No parcel was found in the photo",
    "MULTIPLE_PARCELS": "More than one parcel is in the photo",
    "PARCEL_PARTIALLY_VISIBLE": "The parcel is only partly visible",
    "PARCEL_DETECTION_UNCERTAIN": "The system was unsure about the parcel",
    "LABEL_NOT_VISIBLE": "No shipping label was visible",
    "LABEL_BLOCKED": "The label was covered or blocked",
    "LABEL_UNREADABLE": "The label could not be read",
    "LABEL_PARTIALLY_VISIBLE": "The label was only partly visible",
    "MULTIPLE_LABEL_CANDIDATES": "Several possible labels were seen",
    "MULTIPLE_BARCODES": "More than one barcode was found",
    "AWB_NOT_FOUND": "No tracking number was found",
    "AWB_CONFLICT": "Two different tracking numbers were read",
    "AWB_NOT_LABEL_VERIFIED": "The tracking number was read only from the machine overlay",
    "DIMENSIONS_NOT_FOUND": "The size was not found",
    "WEIGHT_NOT_FOUND": "The weight was not found",
    "OVERLAY_NOT_FOUND": "The scale/machine reading was not found",
    "OVERLAY_PARTIAL": "The scale/machine reading was incomplete",
    "LOW_IMAGE_QUALITY": "The photo was blurry or dark",
    "LOW_OCR_CONFIDENCE": "The text was hard to read",
    "OCR_UNAVAILABLE": "Text reading was unavailable",
    "VISION_API_UNAVAILABLE": "The optional AI helper was unavailable",
    "DETECTOR_UNAVAILABLE": "The parcel detector model was unavailable",
    "UNSUPPORTED_IMAGE": "The file was not a supported image",
    "DUPLICATE_IMAGE": "This photo is a duplicate of another",
}


REVIEW_REASON_PLAIN: dict[str, str] = {
    "processing_error": "The image could not be processed",
    "multiple_parcels": "More than one parcel is in the photo",
    "parcel_partially_visible": "The parcel is only partly visible",
    "parcel_detection_uncertain": "The system was unsure about the parcel",
    "label_not_visible": "No shipping label was visible",
    "label_blocked": "The label was covered or blocked",
    "label_unreadable": "The label could not be read",
    "label_partially_visible": "The label was only partly visible",
    "multiple_label_candidates": "Several possible labels were seen",
    "awb_not_found": "No tracking number was found",
    "awb_conflict": "Two different tracking numbers were read",
    "awb_not_label_verified": "The tracking number was read only from the machine overlay",
    "awb_below_clean_threshold": "The tracking number confidence was too low",
    "vision_unavailable_local_evidence_insufficient": (
        "The optional AI helper was unavailable and the local evidence was insufficient"
    ),
}


def _humanize(flag: str) -> str:
    text = flag.replace("_", " ").strip().lower()
    if not text:
        return text
    return text[0].upper() + text[1:]


def _flag_plain(flag) -> str:
    key = _enum(flag)
    if key in FLAG_PLAIN:
        return FLAG_PLAIN[key]
    return _humanize(key)


def plain_reasons(result) -> list[str]:
    """Return human-readable review reasons for an ImageResult.

    Prefers the explicit review_reasons; otherwise translates the status flags.
    """
    if result.review_reasons:
        return [
            REVIEW_REASON_PLAIN.get(str(reason), _humanize(str(reason)))
            for reason in result.review_reasons
        ]
    if result.status_flags:
        return [_flag_plain(flag) for flag in result.status_flags]
    return []


def plain_awb(result) -> str:
    return result.awb_number or "Not found"


def plain_weight(result) -> str:
    if result.weight_value is not None and result.weight_unit:
        return f"{result.weight_value:g} {result.weight_unit}"
    if result.weight_grams is not None:
        return f"{result.weight_grams:g} g"
    return "Not found"


def plain_size(result) -> str:
    if result.dimensions_text:
        return result.dimensions_text
    if (
        result.length_cm is not None
        and result.width_cm is not None
        and result.height_cm is not None
    ):
        return f"{result.length_cm:g} x {result.width_cm:g} x {result.height_cm:g} cm"
    return "Not found"


def plain_location(result) -> str:
    return result.location or "Not found"


SIMPLE_COLUMNS = [
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


def _simple_row(result) -> list:
    descriptor = outcome(result)
    return [
        result.original_filename,
        plain_awb(result),
        plain_weight(result),
        result.length_cm if result.length_cm is not None else "",
        result.width_cm if result.width_cm is not None else "",
        result.height_cm if result.height_cm is not None else "",
        plain_location(result),
        f'{descriptor["emoji"]} {descriptor["label"]}',
        bool(result.requires_review),
        "; ".join(plain_reasons(result)),
    ]


def simple_dataframe(results) -> pd.DataFrame:
    return pd.DataFrame(
        [_simple_row(result) for result in results],
        columns=SIMPLE_COLUMNS,
    )


def simple_csv_bytes(results) -> bytes:
    return simple_dataframe(results).to_csv(index=False).encode("utf-8-sig")
