from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the lowest-review AWB threshold meeting precision")
    parser.add_argument("--sweep", type=Path, default=Path("evaluation/results/threshold_sweep.csv"))
    parser.add_argument("--minimum-precision", type=float, default=0.99)
    parser.add_argument("--output", type=Path, default=Path("config/calibration.json"))
    parser.add_argument("--version", default="validation-v1")
    args = parser.parse_args()
    sweep = pd.read_csv(args.sweep)
    eligible = sweep[(sweep["accepted"] > 0) & (sweep["precision"] >= args.minimum_precision)]
    if eligible.empty:
        raise SystemExit("No threshold meets the requested precision; collect data or improve extraction")
    selected = eligible.sort_values(["review_rate", "threshold"]).iloc[0]
    payload = {
        "version": args.version,
        "awb_clean_threshold": float(selected["threshold"]),
        "validation_precision": float(selected["precision"]),
        "validation_review_rate": float(selected["review_rate"]),
        "frozen": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
