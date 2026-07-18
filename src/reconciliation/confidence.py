from __future__ import annotations

from src.schemas import FieldCandidate


def adjusted_confidence(
    candidate: FieldCandidate,
    source_agreement: bool = False,
    repeated: bool = False,
    low_quality: bool = False,
    conflict: bool = False,
) -> float:
    score = candidate.confidence
    if source_agreement:
        score += 0.10
    if repeated:
        score += 0.05
    if low_quality:
        score -= 0.10
    if conflict:
        score -= 0.18
    return round(max(0.0, min(0.99, score)), 4)
