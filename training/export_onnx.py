from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and document the deployment ONNX model")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/parcel_detector.onnx"))
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()
    exported = Path(YOLO(str(args.weights)).export(format="onnx", imgsz=640, opset=17, dynamic=False, simplify=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported, args.output)
    sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8")) if args.metrics and args.metrics.is_file() else {}
    card = args.output.with_suffix(".model-card.md")
    card.write_text(
        "\n".join(
            [
                "# Parcel Detector Model Card",
                "",
                f"- SHA-256: `{sha}`",
                "- Architecture: YOLO11n",
                "- Input: 640×640 RGB",
                "- ONNX opset: 17",
                "- Classes: parcel_full, parcel_partial, shipping_label_visible, shipping_label_blocked, hand_or_obstruction",
                f"- Metrics: `{json.dumps(metrics, sort_keys=True)}`",
                "- Intended use: conservative parcel and shipping-label evidence inside the Streamlit application.",
                "- Limitations: performance depends on warehouse, camera, courier, occlusion, and annotation coverage.",
                "- License note: verify Ultralytics licensing obligations before commercial deployment.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"model": str(args.output), "sha256": sha, "model_card": str(card)}))


if __name__ == "__main__":
    main()
