from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


FIELD_COLUMNS = {
    "awb": ("expected_awb", "awb_number"),
    "weight": ("expected_weight_grams", "weight_grams"),
    "length": ("expected_length_cm", "length_cm"),
    "width": ("expected_width_cm", "width_cm"),
    "height": ("expected_height_cm", "height_cm"),
}


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _non_empty(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().ne("")


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _macro_f1(expected: pd.Series, predicted: pd.Series) -> float:
    expected = expected.fillna("MISSING").astype(str).str.strip().str.upper()
    predicted = predicted.fillna("MISSING").astype(str).str.strip().str.upper()
    labels = sorted(set(expected) | set(predicted))
    scores = []
    for label in labels:
        true_positive = int(((expected == label) & (predicted == label)).sum())
        false_positive = int(((expected != label) & (predicted == label)).sum())
        false_negative = int(((expected == label) & (predicted != label)).sum())
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(sum(scores) / len(scores)) if scores else 0.0


def _labelled_merge(truth: pd.DataFrame, predictions: pd.DataFrame, split: str | None) -> pd.DataFrame:
    labelled = truth[_bool_series(truth["annotated"])] if "annotated" in truth else truth
    if split:
        labelled = labelled[labelled["split"] == split]
    return labelled.merge(predictions, on="filename", how="left", validate="one_to_one")


def confusion_tables(truth: pd.DataFrame, predictions: pd.DataFrame, split: str | None) -> dict[str, pd.DataFrame]:
    merged = _labelled_merge(truth, predictions, split)
    tables: dict[str, pd.DataFrame] = {}
    for expected_column, predicted_column in (
        ("expected_parcel_count", "parcel_count"),
        ("expected_parcel_visibility", "parcel_visibility"),
        ("expected_label_status", "label_status"),
        ("expected_review", "requires_review"),
    ):
        if expected_column not in merged or predicted_column not in merged:
            continue
        subset = merged[merged[expected_column].fillna("").astype(str).str.strip().ne("")]
        if subset.empty:
            continue
        tables[predicted_column] = pd.crosstab(
            subset[expected_column].astype(str),
            subset[predicted_column].fillna("MISSING").astype(str),
            rownames=["expected"],
            colnames=["predicted"],
            dropna=False,
        )
    return tables


def evaluate(truth: pd.DataFrame, predictions: pd.DataFrame, split: str | None) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    merged = _labelled_merge(truth, predictions, split)
    expected = merged["expected_awb"].fillna("").astype(str).str.strip()
    predicted = merged["awb_number"].fillna("").astype(str).str.strip()
    clean = ~_bool_series(merged["requires_review"].fillna(True))
    awb_labelled = expected.ne("")
    awb_extracted = awb_labelled & predicted.ne("")
    correct = awb_labelled & predicted.eq(expected) & predicted.ne("")
    clean_awb = clean & awb_extracted
    awb_labelled_count = int(awb_labelled.sum())
    awb_extracted_count = int(awb_extracted.sum())
    correct_count = int(correct.sum())
    clean_awb_count = int(clean_awb.sum())
    clean_correct_count = int((correct & clean_awb).sum())
    false_clean_awb_count = int((clean_awb & ~correct).sum())
    awb_precision = _ratio(correct_count, awb_extracted_count)
    clean_precision = _ratio(clean_correct_count, clean_awb_count)
    awb_recall = _ratio(correct_count, awb_labelled_count)
    review_rate = float((~clean).mean()) if len(merged) else 0.0
    failed = merged["processing_status"].fillna("MISSING").ne("COMPLETED")

    review_labelled = _non_empty(merged.get("expected_review", pd.Series(index=merged.index)))
    expected_review = _bool_series(
        merged.get("expected_review", pd.Series(False, index=merged.index))
    )
    predicted_review = _bool_series(merged["requires_review"].fillna(True))
    review_positives = review_labelled & expected_review
    review_true_positives = review_positives & predicted_review
    review_false_negatives = review_positives & ~predicted_review

    parcel_labelled = pd.Series(False, index=merged.index)
    parcel_correct = pd.Series(True, index=merged.index)
    if "expected_parcel_count" in merged and "parcel_count" in merged:
        count_labelled = _non_empty(merged["expected_parcel_count"])
        parcel_labelled |= count_labelled
        expected_count = pd.to_numeric(merged["expected_parcel_count"], errors="coerce")
        predicted_count = pd.to_numeric(merged["parcel_count"], errors="coerce")
        parcel_correct &= ~count_labelled | expected_count.eq(predicted_count)
    if "expected_parcel_visibility" in merged and "parcel_visibility" in merged:
        visibility_labelled = _non_empty(merged["expected_parcel_visibility"])
        parcel_labelled |= visibility_labelled
        expected_visibility = merged["expected_parcel_visibility"].astype(str).str.upper()
        predicted_visibility = merged["parcel_visibility"].fillna("MISSING").astype(str).str.upper()
        parcel_correct &= ~visibility_labelled | expected_visibility.eq(predicted_visibility)

    field_coverage: dict[str, dict[str, int | float | None]] = {}
    total_expected_fields = 0
    total_extracted_fields = 0
    for name, (expected_column, predicted_column) in FIELD_COLUMNS.items():
        if expected_column not in merged or predicted_column not in merged:
            continue
        expected_present = _non_empty(merged[expected_column])
        predicted_present = _non_empty(merged[predicted_column])
        expected_count = int(expected_present.sum())
        extracted_count = int((expected_present & predicted_present).sum())
        total_expected_fields += expected_count
        total_extracted_fields += extracted_count
        field_coverage[name] = {
            "labelled": expected_count,
            "extracted": extracted_count,
            "coverage": _ratio(extracted_count, expected_count),
        }

    metrics = {
        "labelled_images": len(merged),
        "awb_labelled_images": awb_labelled_count,
        "awb_extracted_images": awb_extracted_count,
        "awb_exact_matches": correct_count,
        "awb_exact_match_precision": awb_precision,
        "awb_exact_match_recall": awb_recall,
        "clean_awb_records": clean_awb_count,
        "clean_awb_precision": clean_precision,
        "false_clean_awb_count": false_clean_awb_count,
        "clean_missing_awb_count": int((clean & awb_labelled & ~awb_extracted).sum()),
        "review_labelled_images": int(review_labelled.sum()),
        "review_positive_images": int(review_positives.sum()),
        "review_recall": _ratio(
            int(review_true_positives.sum()), int(review_positives.sum())
        ),
        "review_false_negatives": int(review_false_negatives.sum()),
        "parcel_condition_labelled_images": int(parcel_labelled.sum()),
        "parcel_condition_accuracy": _ratio(
            int((parcel_labelled & parcel_correct).sum()), int(parcel_labelled.sum())
        ),
        "field_coverage": field_coverage,
        "overall_field_coverage": _ratio(total_extracted_fields, total_expected_fields),
        # Backward-compatible names retained for older report consumers.
        "clean_records": int(clean.sum()),
        "awb_extraction_recall": awb_recall,
        "review_rate": round(review_rate, 6),
        "failed_rate": round(float(failed.mean()) if len(merged) else 0.0, 6),
        "wrong_clean_awbs": false_clean_awb_count,
        "median_processing_ms": float(pd.to_numeric(merged.get("processing_duration_ms"), errors="coerce").median()) if "processing_duration_ms" in merged else None,
        "p95_processing_ms": float(pd.to_numeric(merged.get("processing_duration_ms"), errors="coerce").quantile(0.95)) if "processing_duration_ms" in merged else None,
        "flag_frequency": dict(
            Counter(
                flag
                for value in merged.get("status_flags", pd.Series(dtype=str)).fillna("")
                for flag in str(value).split(";")
                if flag
            )
        ),
        "acceptance": {
            "clean_awb_precision_gte_0_99": (
                clean_precision is not None and clean_precision >= 0.99
            ),
            "false_clean_awb_count_eq_0": false_clean_awb_count == 0,
            "review_rate_lte_0_35": review_rate <= 0.35,
            "failed_rate_lt_0_01": (float(failed.mean()) if len(merged) else 1.0) < 0.01,
        },
    }
    for expected_column, predicted_column in (
        ("expected_parcel_count", "parcel_count"),
        ("expected_parcel_visibility", "parcel_visibility"),
        ("expected_label_status", "label_status"),
        ("expected_review", "requires_review"),
    ):
        if expected_column not in merged or predicted_column not in merged:
            continue
        subset = merged[merged[expected_column].fillna("").astype(str).str.strip().ne("")]
        if not subset.empty:
            metrics[f"{predicted_column}_macro_f1"] = round(
                _macro_f1(
                    subset[expected_column].astype(str),
                    subset[predicted_column].fillna("MISSING").astype(str),
                ),
                6,
            )
    failures = merged[(clean_awb & ~correct) | failed].copy()
    sweep_rows = []
    confidence = pd.to_numeric(merged["awb_confidence"], errors="coerce").fillna(0.0)
    for threshold in [round(value / 100, 2) for value in range(75, 100)]:
        accepted = predicted.ne("") & confidence.ge(threshold)
        accepted_count = int(accepted.sum())
        sweep_rows.append(
            {
                "threshold": threshold,
                "accepted": accepted_count,
                "precision": float((accepted & correct).sum() / accepted_count) if accepted_count else 0.0,
                "review_rate": float((~accepted).mean()) if len(merged) else 1.0,
            }
        )
    return metrics, failures, pd.DataFrame(sweep_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate shipment extraction against human ground truth")
    parser.add_argument("--truth", type=Path, default=Path("evaluation/ground_truth.csv"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    args = parser.parse_args()
    truth = pd.read_csv(args.truth, dtype=str, keep_default_na=False)
    predictions = pd.read_csv(args.predictions, dtype={"awb_number": str})
    metrics, failures, sweep = evaluate(truth, predictions, args.split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    failures.to_csv(args.output_dir / "failures.csv", index=False)
    sweep.to_csv(args.output_dir / "threshold_sweep.csv", index=False)
    for name, table in confusion_tables(truth, predictions, args.split).items():
        table.to_csv(args.output_dir / f"confusion_{name}.csv")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
