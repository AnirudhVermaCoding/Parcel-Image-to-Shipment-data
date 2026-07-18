from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

from PIL import Image


CLASSES = (
    "parcel_full",
    "parcel_partial",
    "shipping_label_visible",
    "shipping_label_blocked",
    "hand_or_obstruction",
)


def validate(dataset: Path, manifest: Path | None = None) -> list[str]:
    errors: list[str] = []
    images = {path.stem: path for path in (dataset / "images").rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    labels = {path.stem: path for path in (dataset / "labels").rglob("*.txt")}
    for stem in sorted(images.keys() - labels.keys()):
        errors.append(f"missing_label:{stem}")
    for stem in sorted(labels.keys() - images.keys()):
        errors.append(f"missing_image:{stem}")
    digests: defaultdict[str, list[str]] = defaultdict(list)
    for stem, image_path in images.items():
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        digests[digest].append(stem)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
        except OSError:
            errors.append(f"invalid_image:{stem}")
            continue
        label_path = labels.get(stem)
        if not label_path:
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"invalid_columns:{stem}:{line_number}")
                continue
            try:
                class_id = int(parts[0])
                x, y, box_width, box_height = map(float, parts[1:])
            except ValueError:
                errors.append(f"invalid_number:{stem}:{line_number}")
                continue
            if not 0 <= class_id < len(CLASSES):
                errors.append(f"invalid_class:{stem}:{line_number}:{class_id}")
            if not all(0 <= value <= 1 for value in (x, y, box_width, box_height)):
                errors.append(f"out_of_range:{stem}:{line_number}")
            if box_width * width < 4 or box_height * height < 4:
                errors.append(f"tiny_box:{stem}:{line_number}")
    for stems in digests.values():
        if len(stems) > 1:
            errors.append(f"duplicate_images:{','.join(sorted(stems))}")
    if manifest and manifest.is_file():
        by_hash: defaultdict[str, set[str]] = defaultdict(set)
        with manifest.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                by_hash[row["sha256"]].add(row["split"])
        for sha, splits in by_hash.items():
            if len(splits) > 1:
                errors.append(f"split_leakage:{sha}:{','.join(sorted(splits))}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CVAT YOLO annotations")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    errors = validate(args.dataset, args.manifest)
    for error in errors:
        print(error)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
