from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def _bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _macro_f1(expected: pd.Series, predicted: pd.Series) -> float:
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
    correct = predicted.eq(expected) & predicted.ne("")
    clean_count = int(clean.sum())
    precision = float((correct & clean).sum() / clean_count) if clean_count else 0.0
    positives = int(expected.ne("").sum())
    recall = float(correct.sum() / positives) if positives else 0.0
    review_rate = float((~clean).mean()) if len(merged) else 0.0
    failed = merged["processing_status"].fillna("MISSING").ne("COMPLETED")
    metrics = {
        "labelled_images": len(merged),
        "clean_records": clean_count,
        "clean_awb_precision": round(precision, 6),
        "awb_extraction_recall": round(recall, 6),
        "review_rate": round(review_rate, 6),
        "failed_rate": round(float(failed.mean()) if len(merged) else 0.0, 6),
        "wrong_clean_awbs": int((clean & ~correct).sum()),
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
            "clean_awb_precision_gte_0_99": precision >= 0.99,
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
    failures = merged[(clean & ~correct) | failed].copy()
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
