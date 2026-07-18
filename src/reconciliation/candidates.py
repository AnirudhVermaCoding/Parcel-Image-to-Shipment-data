from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from src.reconciliation.confidence import adjusted_confidence
from src.schemas import ExtractionSource, FieldCandidate


@dataclass(slots=True)
class ReconciledField:
    candidate: FieldCandidate | None = None
    confidence: float = 0.0
    conflict: bool = False
    agreeing_sources: list[ExtractionSource] = field(default_factory=list)


def reconcile(
    field_name: str,
    candidates: list[FieldCandidate],
    low_quality: bool = False,
) -> ReconciledField:
    relevant = [
        candidate
        for candidate in candidates
        if candidate.field_name == field_name and candidate.normalised_value not in (None, "")
    ]
    if not relevant:
        return ReconciledField()
    grouped: defaultdict[str, list[FieldCandidate]] = defaultdict(list)
    for candidate in relevant:
        key = str(candidate.normalised_value).strip().upper()
        grouped[key].append(candidate)
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            len({candidate.source for candidate in item[1]}),
            max(candidate.confidence for candidate in item[1]),
            len(item[1]),
        ),
        reverse=True,
    )
    best_key, best_group = ranked[0]
    strong_alternatives = [
        group
        for key, group in ranked[1:]
        if key != best_key and max(candidate.confidence for candidate in group) >= 0.72
    ]
    conflict = bool(strong_alternatives)
    best = max(best_group, key=lambda candidate: candidate.confidence)
    sources = sorted({candidate.source for candidate in best_group}, key=lambda item: item.value)
    confidence = adjusted_confidence(
        best,
        source_agreement=len(sources) >= 2,
        repeated=len(best_group) >= 2,
        low_quality=low_quality,
        conflict=conflict,
    )
    source_priority = {
        ExtractionSource.BARCODE: 4,
        ExtractionSource.OVERLAY_OCR: 3,
        ExtractionSource.LABEL_OCR: 2,
        ExtractionSource.VISION_FALLBACK: 1,
        ExtractionSource.NOT_EXTRACTED: 0,
    }
    selected = max(
        best_group,
        key=lambda candidate: (source_priority[candidate.source], candidate.confidence),
    ).model_copy(update={"confidence": confidence})
    return ReconciledField(
        candidate=selected,
        confidence=confidence,
        conflict=conflict,
        agreeing_sources=sources,
    )
