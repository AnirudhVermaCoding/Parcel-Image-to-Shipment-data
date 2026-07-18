from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.config import AppConfig
from src.extraction.overlay import extract_overlay
from src.validation.images import calculate_quality_metrics, load_image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("sample_data/supplied"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_data/dataset_observations.csv"),
    )
    args = parser.parse_args()
    config = AppConfig.from_env()
    rows = []
    for path in sorted(args.input.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        loaded = load_image(path.read_bytes(), config)
        quality = calculate_quality_metrics(loaded.bgr)
        overlay = extract_overlay(loaded.bgr, config)
        rows.append(
            {
                "filename": path.name,
                "width": loaded.width,
                "height": loaded.height,
                "orientation": loaded.orientation,
                "blur_score": quality.blur_score,
                "brightness": quality.brightness,
                "contrast": quality.contrast,
                "underexposure_pct": quality.underexposure_pct,
                "overexposure_pct": quality.overexposure_pct,
                "edge_density": quality.edge_density,
                "overlay_detected": overlay.detected,
                "overlay_bbox": (
                    f"{overlay.bbox.x},{overlay.bbox.y},{overlay.bbox.width},{overlay.bbox.height}"
                    if overlay.bbox
                    else ""
                ),
                "overlay_ocr_confidence": overlay.confidence,
                "parsed_field_count": len(overlay.candidates),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

