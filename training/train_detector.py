from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the parcel/label YOLO11n detector")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="training/runs")
    args = parser.parse_args()
    model = YOLO("yolo11n.pt")
    result = model.train(
        data=str(args.data),
        imgsz=640,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name="parcel_detector",
        seed=2026,
        deterministic=True,
        patience=20,
        degrees=10.0,
        translate=0.1,
        scale=0.35,
        perspective=0.0005,
        hsv_h=0.015,
        hsv_s=0.35,
        hsv_v=0.30,
        erasing=0.25,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=10,
    )
    print(json.dumps({"save_dir": str(result.save_dir)}))


if __name__ == "__main__":
    main()
