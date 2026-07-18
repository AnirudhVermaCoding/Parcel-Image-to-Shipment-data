from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def save(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to a local parcel image. Private source images are not committed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/generated"),
    )
    args = parser.parse_args()
    image = cv2.imread(str(args.source))
    if image is None:
        raise FileNotFoundError(args.source)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    save(output / "blurred.jpg", cv2.GaussianBlur(image, (31, 31), 0))
    save(output / "darkened.jpg", cv2.convertScaleAbs(image, alpha=0.35, beta=0))
    save(output / "overexposed.jpg", cv2.convertScaleAbs(image, alpha=2.2, beta=80))
    save(output / "rotated.jpg", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    save(output / "cropped_parcel.jpg", image[:, : int(image.shape[1] * 0.68)])

    occluded = image.copy()
    cv2.rectangle(
        occluded,
        (int(image.shape[1] * 0.40), int(image.shape[0] * 0.12)),
        (int(image.shape[1] * 0.67), int(image.shape[0] * 0.48)),
        (90, 90, 90),
        -1,
    )
    save(output / "label_occluded.jpg", occluded)

    label_blurred = image.copy()
    x1, y1 = int(image.shape[1] * 0.38), int(image.shape[0] * 0.08)
    x2, y2 = int(image.shape[1] * 0.68), int(image.shape[0] * 0.48)
    label_blurred[y1:y2, x1:x2] = cv2.GaussianBlur(
        label_blurred[y1:y2, x1:x2],
        (51, 51),
        0,
    )
    save(output / "label_unreadable.jpg", label_blurred)

    right = cv2.flip(image, 1)
    combined = np.hstack(
        [
            cv2.resize(image, (image.shape[1] // 2, image.shape[0])),
            cv2.resize(right, (image.shape[1] // 2, image.shape[0])),
        ]
    )
    save(output / "multiple_parcels.jpg", combined)
    save(output / "empty_background.jpg", np.full_like(image, 12))

    conflict = image.copy()
    cv2.rectangle(conflict, (30, 125), (1050, 230), (0, 0, 0), -1)
    cv2.putText(
        conflict,
        "AWB No: 99999999999999",
        (75, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        (0, 255, 255),
        5,
        cv2.LINE_AA,
    )
    save(output / "conflicting_overlay_awb.jpg", conflict)

    shutil.copy2(args.source, output / "duplicate_different_name.jpeg")
    (output / "corrupt.jpg").write_bytes(b"\xff\xd8\xffcorrupt-jpeg")
    print(f"Generated fixtures in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
