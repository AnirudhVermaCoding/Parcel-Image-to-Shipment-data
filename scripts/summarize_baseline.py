from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from src.jobs.database import JobDatabase


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a non-sensitive baseline summary from a job")
    parser.add_argument("--database", type=Path, default=Path("runtime/jobs.sqlite3"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/baseline_summary.json"))
    args = parser.parse_args()
    results = JobDatabase(args.database).get_results(args.job_id)
    durations = [result.processing_duration_ms for result in results]
    flags = Counter(flag.value for result in results for flag in result.status_flags)
    summary = {
        "job_id": args.job_id,
        "processed": len(results),
        "clean": sum(not result.requires_review for result in results),
        "review": sum(result.requires_review for result in results),
        "failed": sum(result.processing_status.value == "FAILED" for result in results),
        "awb_extracted": sum(bool(result.awb_number) for result in results),
        "review_rate": round(sum(result.requires_review for result in results) / len(results), 6) if results else 0,
        "median_processing_ms": statistics.median(durations) if durations else 0,
        "p95_processing_ms": sorted(durations)[min(len(durations) - 1, int(len(durations) * 0.95))] if durations else 0,
        "flag_frequency": dict(flags.most_common()),
        "accuracy_metrics_pending_human_ground_truth": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
