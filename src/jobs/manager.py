from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from src.config import AppConfig
from src.jobs.database import JobDatabase
from src.pipeline import process_image
from src.extraction.vision_fallback import VisionProviderError, XaiVisionProvider
from src.reporting.annotations import create_annotation
from src.reporting.csv_report import write_csv, write_json
from src.schemas import (
    BatchOptions,
    ImageResult,
    JobRecord,
    JobStage,
    PrimaryStatus,
    StatusFlag,
    VisionDiagnostic,
)
from src.validation.files import InputFile, detect_image_type, internal_filename
from src.validation.images import load_image


@dataclass(slots=True)
class StoredInput:
    original_filename: str
    path: Path
    sha256: str


class LimitedVisionProvider:
    def __init__(self, config: AppConfig):
        self.provider = XaiVisionProvider(config)
        self.limit = config.xai_max_images_per_job
        self.used = 0
        self.failures = 0
        self.verified = False
        self.lock = threading.Lock()

    def analyze(
        self,
        image_bytes: bytes,
        label_crop_bytes: bytes | None = None,
        include_full_frame: bool = True,
    ):
        cached = self.provider.cached_analysis(
            image_bytes, label_crop_bytes, include_full_frame
        )
        if cached is not None:
            with self.lock:
                self.verified = True
            return cached
        with self.lock:
            if not self.verified and self.failures >= 3:
                raise VisionProviderError(
                    "External vision circuit is open after repeated configuration failures",
                    VisionDiagnostic(
                        category="circuit_open",
                        message="Run the vision configuration diagnostic before retrying",
                    ),
                )
            if self.used >= self.limit:
                raise VisionProviderError(
                    "Per-job external vision limit reached",
                    VisionDiagnostic(
                        category="job_limit",
                        message="Per-job external vision limit reached",
                    ),
                )
            self.used += 1
        try:
            analysis = self.provider.analyze(
                image_bytes,
                label_crop_bytes=label_crop_bytes,
                include_full_frame=include_full_frame,
            )
            with self.lock:
                self.verified = True
            return analysis
        except VisionProviderError:
            with self.lock:
                self.failures += 1
            raise


class JobManager:
    # Bump this whenever the public manager API or its in-memory state changes.
    # Streamlit uses it to avoid reusing an incompatible cached instance after
    # a hot reload while keeping persisted SQLite checkpoints intact.
    CACHE_API_VERSION = 3

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig.from_env()
        self.config.ensure_runtime_dirs()
        self.database = JobDatabase(self.config.runtime_dir / "jobs.sqlite3")
        self.database.mark_interrupted_jobs()
        self._threads: dict[str, threading.Thread] = {}
        self._cancellations: set[str] = set()
        self._lock = threading.Lock()
        self._cleanup_expired_jobs()

    def _cleanup_expired_jobs(self) -> None:
        if self.config.job_retention_hours <= 0:
            return
        jobs_root = (self.config.runtime_dir / "jobs").resolve()
        if not jobs_root.exists():
            return
        cutoff = time.time() - self.config.job_retention_hours * 3600
        for directory in jobs_root.iterdir():
            if (
                not directory.is_dir()
                or len(directory.name) != 32
                or any(character not in "0123456789abcdef" for character in directory.name.lower())
            ):
                continue
            resolved = directory.resolve()
            if not resolved.is_relative_to(jobs_root):
                continue
            try:
                if resolved.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(resolved)
                self.database.delete_job(directory.name)
            except OSError:
                continue

    def _store_inputs(self, job_id: str, inputs: list[InputFile]) -> list[StoredInput]:
        upload_dir = self.config.runtime_dir / "jobs" / job_id / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        stored: list[StoredInput] = []
        for item in inputs:
            detected = detect_image_type(item.data)
            extension = detected or Path(item.original_filename).suffix.lower() or ".bin"
            path = upload_dir / internal_filename(extension)
            path.write_bytes(item.data)
            stored.append(
                StoredInput(
                    original_filename=item.original_filename,
                    path=path,
                    sha256=hashlib.sha256(item.data).hexdigest(),
                )
            )
        manifest_path = upload_dir.parent / "input_manifest.json"
        manifest_path.write_text(
            json.dumps(
                [
                    {
                        "original_filename": item.original_filename,
                        "path": str(item.path),
                        "sha256": item.sha256,
                    }
                    for item in stored
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        return stored

    def _job_dir(self, job_id: str) -> Path:
        return self.config.runtime_dir / "jobs" / job_id

    def _save_options(self, job_id: str, options: BatchOptions) -> None:
        (self._job_dir(job_id) / "options.json").write_text(
            options.model_dump_json(indent=2), encoding="utf-8"
        )

    def _load_options(self, job_id: str) -> BatchOptions:
        path = self._job_dir(job_id) / "options.json"
        if path.is_file():
            try:
                return BatchOptions.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                pass
        options = BatchOptions(
            # Legacy jobs have no persisted consent/options. Resume them locally by default.
            local_only=True,
            enable_vision=False,
            generate_annotations=self.config.generate_annotations,
            max_workers=self.config.default_workers,
            awb_acceptance_threshold=self.config.awb_clean_threshold,
        )
        self._save_options(job_id, options)
        return options

    def _load_stored_inputs(self, job_id: str) -> list[StoredInput]:
        job_dir = self._job_dir(job_id)
        manifest_path = job_dir / "input_manifest.json"
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                stored = [
                    StoredInput(
                        original_filename=str(item["original_filename"]),
                        path=Path(item["path"]),
                        sha256=str(item["sha256"]),
                    )
                    for item in payload
                ]
                if stored and all(item.path.is_file() for item in stored):
                    return stored
            except (OSError, ValueError, KeyError, TypeError):
                pass

        known_by_path = {
            str(Path(item["input_path"]).resolve()): item
            for item in self.database.get_result_input_records(job_id)
            if item["input_path"]
        }
        stored = []
        for path in sorted((job_dir / "uploads").glob("*")):
            if not path.is_file():
                continue
            known = known_by_path.get(str(path.resolve()))
            sha = (
                known["sha256"]
                if known and known["sha256"]
                else hashlib.sha256(path.read_bytes()).hexdigest()
            )
            stored.append(
                StoredInput(
                    original_filename=(known["original_filename"] if known else path.name),
                    path=path,
                    sha256=sha,
                )
            )
        if stored:
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "original_filename": item.original_filename,
                            "path": str(item.path),
                            "sha256": item.sha256,
                        }
                        for item in stored
                    ],
                    indent=2,
                ),
                encoding="utf-8",
            )
        return stored

    def submit(
        self,
        inputs: list[InputFile],
        options: BatchOptions | None = None,
        background: bool = True,
    ) -> str:
        if not inputs:
            raise ValueError("At least one image is required")
        if len(inputs) > self.config.max_batch_images:
            raise ValueError(f"Batch exceeds {self.config.max_batch_images} images")
        job_id = uuid.uuid4().hex
        stored = self._store_inputs(job_id, inputs)
        self.database.create_job(
            JobRecord(
                job_id=job_id,
                stage=JobStage.UPLOADED,
                total_images=len(stored),
                started_at=datetime.now(),
                last_progress_at=datetime.now(),
            )
        )
        options = options or BatchOptions(
            local_only=not self.config.enable_xai_fallback,
            enable_vision=self.config.enable_xai_fallback,
            generate_annotations=self.config.generate_annotations,
            max_workers=self.config.default_workers,
            awb_acceptance_threshold=self.config.awb_clean_threshold,
        )
        self._save_options(job_id, options)
        if background:
            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, stored, options),
                daemon=True,
                name=f"parcel-job-{job_id[:8]}",
            )
            with self._lock:
                self._threads[job_id] = thread
            thread.start()
        else:
            self._run_job(job_id, stored, options)
        return job_id

    def is_running(self, job_id: str) -> bool:
        with self._lock:
            thread = self._threads.get(job_id)
            return bool(thread and thread.is_alive())

    def _write_checkpoint_reports(
        self,
        job_id: str,
        results: list[ImageResult],
    ) -> tuple[Path, Path] | None:
        if not results:
            return None
        report_dir = self._job_dir(job_id) / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = report_dir / "shipment_results.partial.csv"
        json_path = report_dir / "shipment_results.partial.json"
        csv_temp = report_dir / "shipment_results.partial.csv.tmp"
        json_temp = report_dir / "shipment_results.partial.json.tmp"
        try:
            write_csv(results, csv_temp)
            write_json(results, json_temp)
            csv_temp.replace(csv_path)
            json_temp.replace(json_path)
            checkpoint_at = datetime.now()
            self.database.update_job(
                job_id,
                report_path=str(csv_path),
                json_report_path=str(json_path),
                checkpoint_at=checkpoint_at,
            )
            return csv_path, json_path
        except OSError:
            for temporary in (csv_temp, json_temp):
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            return None

    def resume(self, job_id: str, background: bool = True) -> int:
        job = self.database.get_job(job_id)
        if job is None:
            raise ValueError("Job not found")
        if self.is_running(job_id):
            raise RuntimeError("Job is still running in this application process")
        stored = self._load_stored_inputs(job_id)
        if not stored:
            raise RuntimeError("Saved uploads are unavailable; the job cannot be resumed")
        existing = self.database.get_results(job_id)
        completed_hashes = {result.sha256 for result in existing if result.sha256}
        pending = [item for item in stored if item.sha256 not in completed_hashes]
        self._cancellations.discard(job_id)
        self.database.update_job(
            job_id,
            stage=JobStage.QUEUED,
            total_images=len(stored),
            processed_images=len(existing),
            progress_percentage=round(len(existing) / max(len(stored), 1) * 100, 2),
            current_image=None,
            completed_at=None,
            error_message=None,
            last_progress_at=datetime.now(),
            resumed_count=job.resumed_count + 1,
        )
        options = self._load_options(job_id)
        if background:
            thread = threading.Thread(
                target=self._run_job,
                args=(job_id, pending, options, existing, len(stored)),
                daemon=True,
                name=f"parcel-resume-{job_id[:8]}",
            )
            with self._lock:
                self._threads[job_id] = thread
            thread.start()
        else:
            self._run_job(job_id, pending, options, existing, len(stored))
        return len(pending)

    def _annotate(self, job_id: str, stored: StoredInput, result: ImageResult) -> None:
        data = stored.path.read_bytes()
        loaded = load_image(data, self.config)
        annotation_path = (
            self.config.runtime_dir
            / "jobs"
            / job_id
            / "annotations"
            / f"{result.image_id}.jpg"
        )
        create_annotation(loaded.bgr, result, annotation_path)
        result.annotated_image_path = str(annotation_path)

    def _save_with_duplicates(
        self,
        job_id: str,
        canonical: StoredInput,
        duplicates: list[StoredInput],
        result: ImageResult,
        options: BatchOptions,
    ) -> list[ImageResult]:
        saved: list[ImageResult] = []
        if options.generate_annotations and result.processing_status.value == "COMPLETED":
            try:
                self._annotate(job_id, canonical, result)
            except Exception as exc:
                result.warnings.append(f"Annotation generation failed: {type(exc).__name__}")
        self.database.save_result(job_id, result, str(canonical.path))
        saved.append(result)
        for duplicate in duplicates:
            cloned = result.model_copy(deep=True)
            cloned.image_id = uuid.uuid4().hex
            cloned.original_filename = duplicate.original_filename
            cloned.duplicate_of_image_id = result.image_id
            cloned.annotated_image_path = result.annotated_image_path
            if StatusFlag.DUPLICATE_IMAGE not in cloned.status_flags:
                cloned.status_flags.append(StatusFlag.DUPLICATE_IMAGE)
            cloned.warnings.append(f"Duplicate content reused from {canonical.original_filename}")
            self.database.save_result(job_id, cloned, str(duplicate.path))
            saved.append(cloned)
        return saved

    def _run_job(
        self,
        job_id: str,
        stored: list[StoredInput],
        options: BatchOptions,
        existing_results: list[ImageResult] | None = None,
        total_images: int | None = None,
    ) -> None:
        try:
            self.database.update_job(job_id, stage=JobStage.VALIDATING)
            groups: dict[str, list[StoredInput]] = {}
            for item in stored:
                groups.setdefault(item.sha256, []).append(item)
            canonical_inputs = [group[0] for group in groups.values()]
            self.database.update_job(job_id, stage=JobStage.QUEUED)
            results: list[ImageResult] = list(existing_results or [])
            total = total_images if total_images is not None else len(stored)
            workers = max(
                1,
                min(options.max_workers, self.config.max_workers, max(1, len(canonical_inputs))),
            )
            job_config = replace(
                self.config,
                enable_xai_fallback=bool(options.enable_vision),
                awb_clean_threshold=options.awb_acceptance_threshold,
            )
            vision_provider = (
                LimitedVisionProvider(job_config)
                if options.enable_vision and job_config.xai_api_key
                else None
            )
            self.database.update_job(
                job_id,
                stage=JobStage.PROCESSING,
                last_progress_at=datetime.now(),
            )
            chunk_size = max(8, workers * 4)
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="parcel-worker") as executor:
                for offset in range(0, len(canonical_inputs), chunk_size):
                    chunk = canonical_inputs[offset : offset + chunk_size]
                    futures = {
                        executor.submit(
                            process_image,
                            item.path,
                            item.original_filename,
                            job_config,
                            vision_provider,
                        ): item
                        for item in chunk
                    }
                    for future in as_completed(futures):
                        if job_id in self._cancellations:
                            for pending in futures:
                                pending.cancel()
                            self.database.update_job(
                                job_id,
                                stage=JobStage.CANCELLED,
                                completed_at=datetime.now(),
                            )
                            return
                        stored_input = futures[future]
                        try:
                            result = future.result()
                        except Exception as exc:
                            result = ImageResult(
                                image_id=uuid.uuid4().hex,
                                original_filename=stored_input.original_filename,
                                processing_status="FAILED",
                                primary_status=PrimaryStatus.PROCESSING_ERROR,
                                error_code="WORKER_EXCEPTION",
                                error_message=f"{type(exc).__name__}: {exc}",
                                requires_review=True,
                            )
                        group = groups[stored_input.sha256]
                        saved = self._save_with_duplicates(
                            job_id,
                            stored_input,
                            group[1:],
                            result,
                            options,
                        )
                        results.extend(saved)
                        self._update_progress(job_id, results, total, stored_input.original_filename)
                        self._write_checkpoint_reports(job_id, results)

            self.database.update_job(job_id, stage=JobStage.GENERATING_CSV)
            report_dir = self.config.runtime_dir / "jobs" / job_id / "reports"
            csv_path = write_csv(results, report_dir / "shipment_results.csv")
            json_path = write_json(results, report_dir / "shipment_results.json")
            warning_count = sum(len(result.warnings) for result in results)
            stage = JobStage.COMPLETED_WITH_WARNINGS if warning_count else JobStage.COMPLETED
            self.database.update_job(
                job_id,
                stage=stage,
                processed_images=len(results),
                progress_percentage=100.0,
                current_image=None,
                completed_at=datetime.now(),
                report_path=str(csv_path),
                json_report_path=str(json_path),
                warning_count=warning_count,
                last_progress_at=datetime.now(),
                checkpoint_at=datetime.now(),
            )
        except Exception as exc:
            self.database.update_job(
                job_id,
                stage=JobStage.FAILED,
                error_message=f"{type(exc).__name__}: {exc}",
                completed_at=datetime.now(),
            )
        finally:
            with self._lock:
                self._threads.pop(job_id, None)

    def _update_progress(
        self,
        job_id: str,
        results: list[ImageResult],
        total: int,
        current_image: str,
    ) -> None:
        processed = len(results)
        successful = sum(
            result.primary_status
            in {
                PrimaryStatus.SUCCESS_LABEL_VERIFIED,
                PrimaryStatus.SUCCESS_OVERLAY_AND_LABEL,
                PrimaryStatus.SUCCESS_OVERLAY_ONLY,
                PrimaryStatus.SUCCESS_PARTIAL_FIELDS,
            }
            for result in results
        )
        reviews = sum(result.requires_review for result in results)
        rejected = sum(result.primary_status == PrimaryStatus.REJECTED for result in results)
        failed = sum(result.primary_status == PrimaryStatus.PROCESSING_ERROR for result in results)
        self.database.update_job(
            job_id,
            processed_images=processed,
            successful_images=successful,
            review_required_images=reviews,
            rejected_images=rejected,
            failed_images=failed,
            warning_count=sum(len(result.warnings) for result in results),
            progress_percentage=round(processed / max(total, 1) * 100, 2),
            current_image=current_image,
            last_progress_at=datetime.now(),
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        job = self.database.get_job(job_id)
        if job and job.processed_images and (
            not job.report_path or not Path(job.report_path).is_file()
        ):
            self._write_checkpoint_reports(job_id, self.database.get_results(job_id))
            job = self.database.get_job(job_id)
        return job

    def get_results(self, job_id: str) -> list[ImageResult]:
        return self.database.get_results(job_id)

    def cancel(self, job_id: str) -> None:
        self._cancellations.add(job_id)
