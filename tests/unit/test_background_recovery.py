from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

from src.jobs.database import JobDatabase
from src.jobs.manager import JobManager
from src.schemas import (
    BatchOptions,
    ImageResult,
    JobStage,
    PrimaryStatus,
    ProcessingStatus,
)
from src.validation.files import InputFile


def _result(path, original_filename, config, vision_provider) -> ImageResult:
    data = Path(path).read_bytes()
    return ImageResult(
        image_id=hashlib.sha256((original_filename + "result").encode()).hexdigest()[:32],
        original_filename=original_filename,
        sha256=hashlib.sha256(data).hexdigest(),
        processing_status=ProcessingStatus.COMPLETED,
        primary_status=PrimaryStatus.SUCCESS_PARTIAL_FIELDS,
        requires_review=False,
    )


def test_background_job_exposes_partial_checkpoint(monkeypatch, test_config) -> None:
    release_second = threading.Event()

    def controlled(path, original_filename, config, vision_provider):
        if original_filename == "second.jpg":
            assert release_second.wait(timeout=5)
        return _result(path, original_filename, config, vision_provider)

    monkeypatch.setattr("src.jobs.manager.process_image", controlled)
    manager = JobManager(test_config)
    job_id = manager.submit(
        [InputFile("first.jpg", b"first"), InputFile("second.jpg", b"second")],
        BatchOptions(max_workers=1),
        background=True,
    )
    deadline = time.time() + 5
    job = manager.get_job(job_id)
    while job.processed_images < 1 and time.time() < deadline:
        time.sleep(0.02)
        job = manager.get_job(job_id)
    try:
        assert job.processed_images == 1
        assert len(manager.get_results(job_id)) == 1
        assert job.report_path and Path(job.report_path).is_file()
        assert job.json_report_path and Path(job.json_report_path).is_file()
        assert job.last_progress_at is not None
    finally:
        release_second.set()
    deadline = time.time() + 5
    while manager.is_running(job_id) and time.time() < deadline:
        time.sleep(0.02)
    assert manager.get_job(job_id).stage in {
        JobStage.COMPLETED,
        JobStage.COMPLETED_WITH_WARNINGS,
    }


def test_resume_processes_only_missing_hashes(monkeypatch, test_config) -> None:
    calls: list[str] = []

    def tracked(path, original_filename, config, vision_provider):
        calls.append(original_filename)
        return _result(path, original_filename, config, vision_provider)

    monkeypatch.setattr("src.jobs.manager.process_image", tracked)
    manager = JobManager(test_config)
    job_id = manager.submit(
        [
            InputFile("one.jpg", b"one"),
            InputFile("two.jpg", b"two"),
            InputFile("three.jpg", b"three"),
        ],
        BatchOptions(max_workers=1),
        background=False,
    )
    results = manager.get_results(job_id)
    missing = next(result for result in results if result.original_filename == "two.jpg")
    with manager.database._connect() as connection:
        connection.execute(
            "DELETE FROM results WHERE job_id = ? AND image_id = ?",
            (job_id, missing.image_id),
        )
    manager.database.update_job(
        job_id,
        stage=JobStage.FAILED,
        processed_images=2,
        progress_percentage=66.67,
        error_message="simulated interruption",
    )
    pending = manager.resume(job_id, background=False)
    assert pending == 1
    assert calls.count("two.jpg") == 2
    assert calls.count("one.jpg") == 1
    assert calls.count("three.jpg") == 1
    resumed = manager.get_job(job_id)
    assert resumed.resumed_count == 1
    assert resumed.processed_images == resumed.total_images == 3
    assert len(manager.get_results(job_id)) == 3


def test_database_migrates_existing_jobs_table(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, stage TEXT NOT NULL,
            total_images INTEGER NOT NULL DEFAULT 0,
            processed_images INTEGER NOT NULL DEFAULT 0,
            successful_images INTEGER NOT NULL DEFAULT 0,
            review_required_images INTEGER NOT NULL DEFAULT 0,
            rejected_images INTEGER NOT NULL DEFAULT 0,
            failed_images INTEGER NOT NULL DEFAULT 0,
            warning_count INTEGER NOT NULL DEFAULT 0,
            progress_percentage REAL NOT NULL DEFAULT 0,
            current_image TEXT, started_at TEXT, completed_at TEXT,
            report_path TEXT, json_report_path TEXT, error_message TEXT
        )
        """
    )
    connection.commit()
    connection.close()
    database = JobDatabase(path)
    with database._connect() as migrated:
        columns = {row["name"] for row in migrated.execute("PRAGMA table_info(jobs)")}
    assert {"last_progress_at", "checkpoint_at", "resumed_count"} <= columns


def test_job_manager_exposes_versioned_recovery_api() -> None:
    assert JobManager.CACHE_API_VERSION >= 2
    assert all(
        callable(getattr(JobManager, method, None))
        for method in ("get_job", "get_results", "is_running", "resume")
    )
