from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import AppConfig
from src.jobs.manager import JobManager
from src.schemas import BatchOptions
from src.validation.files import collect_path_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract shipment data from parcel images")
    parser.add_argument("--input", required=True, type=Path, help="Image, ZIP, or directory")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--annotations", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = AppConfig.from_env()
    if args.workers:
        config.max_workers = max(1, args.workers)
    files, warnings = collect_path_inputs(args.input, config)
    for warning in warnings:
        print(f"warning: {warning}")
    manager = JobManager(config)
    job_id = manager.submit(
        files,
        BatchOptions(
            local_only=True,
            enable_vision=False,
            generate_annotations=args.annotations,
            max_workers=config.max_workers,
            awb_acceptance_threshold=config.awb_clean_threshold,
        ),
        background=False,
    )
    job = manager.get_job(job_id)
    if job is None or not job.report_path:
        print(f"Job {job_id} failed: {job.error_message if job else 'unknown error'}")
        return 1
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(job.report_path, args.output / "shipment_results.csv")
    if job.json_report_path:
        shutil.copy2(job.json_report_path, args.output / "shipment_results.json")
    if args.annotations:
        annotations_dir = config.runtime_dir / "jobs" / job_id / "annotations"
        if annotations_dir.exists():
            destination = args.output / "annotations"
            destination.mkdir(parents=True, exist_ok=True)
            for annotation in annotations_dir.glob("*.jpg"):
                shutil.copy2(annotation, destination / annotation.name)
    print(f"Job ID: {job_id}")
    print(f"Processed: {job.processed_images}/{job.total_images}")
    print(f"CSV: {args.output / 'shipment_results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
