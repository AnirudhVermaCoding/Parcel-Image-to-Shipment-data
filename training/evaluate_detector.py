from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def _values(value) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        value = [value]
    return [round(float(item), 6) for item in value]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen detector split")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path, default=Path("training/detector_metrics.json"))
    args = parser.parse_args()
    metrics = YOLO(str(args.weights)).val(data=str(args.data), split=args.split, imgsz=640, device="cpu")
    precision = _values(getattr(metrics.box, "p", None))
    recall = _values(getattr(metrics.box, "r", None))
    f1 = [
        round(2 * p * r / (p + r), 6) if p + r else 0.0
        for p, r in zip(precision, recall)
    ]
    payload = {
        "split": args.split,
        "map50": round(float(metrics.box.map50), 6),
        "map50_95": round(float(metrics.box.map), 6),
        "per_class_map50_95": _values(metrics.box.maps),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "macro_f1": round(sum(f1) / len(f1), 6) if f1 else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
