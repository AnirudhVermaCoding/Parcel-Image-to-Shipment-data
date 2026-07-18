from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image


FIELDS = [
    "filename",
    "sha256",
    "group_id",
    "split",
    "expected_awb",
    "expected_weight_grams",
    "expected_length_cm",
    "expected_width_cm",
    "expected_height_cm",
    "expected_parcel_count",
    "expected_parcel_visibility",
    "expected_label_status",
    "expected_review",
    "annotated",
]


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    values = np.asarray(gray)
    bits = values[:, 1:] > values[:, :-1]
    result = 0
    for bit in bits.reshape(-1):
        result = (result << 1) | int(bit)
    return result


def _split(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "val" if bucket < 85 else "test"


def _job_inputs(database: Path, job_id: str) -> list[tuple[str, Path, str]]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT original_filename, input_path, sha256 FROM results WHERE job_id = ? ORDER BY original_filename",
        (job_id,),
    ).fetchall()
    items = [(str(name), Path(path), str(sha or "")) for name, path, sha in rows]
    known = {path.resolve() for _, path, _ in items if path.exists()}
    uploads = database.parent / "jobs" / job_id / "uploads"
    for path in sorted(uploads.glob("*")):
        if path.is_file() and path.resolve() not in known:
            items.append((path.name, path, ""))
    return items


def build_rows(items: list[tuple[str, Path, str]]) -> list[dict[str, str]]:
    hashes: list[tuple[str, int]] = []
    rows: list[dict[str, str]] = []
    for filename, path, sha in items:
        if not path.is_file():
            continue
        sha = sha or hashlib.sha256(path.read_bytes()).hexdigest()
        visual_hash = _dhash(path)
        group_id = sha[:12]
        for existing_group, existing_hash in hashes:
            if (visual_hash ^ existing_hash).bit_count() <= 6:
                group_id = existing_group
                break
        else:
            hashes.append((group_id, visual_hash))
        row = {field: "" for field in FIELDS}
        row.update(
            filename=filename,
            sha256=sha,
            group_id=group_id,
            split=_split(group_id),
            annotated="false",
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a human-labelling ground-truth template")
    parser.add_argument("--database", type=Path, default=Path("runtime/jobs.sqlite3"))
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/ground_truth.csv"))
    args = parser.parse_args()
    rows = build_rows(_job_inputs(args.database, args.job_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print(json.dumps({"rows": len(rows), "splits": summary, "output": str(args.output)}))


if __name__ == "__main__":
    main()
