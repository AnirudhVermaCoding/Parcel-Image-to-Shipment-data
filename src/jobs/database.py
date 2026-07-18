from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from src.schemas import ImageResult, JobRecord, JobStage


class JobDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    total_images INTEGER NOT NULL DEFAULT 0,
                    processed_images INTEGER NOT NULL DEFAULT 0,
                    successful_images INTEGER NOT NULL DEFAULT 0,
                    review_required_images INTEGER NOT NULL DEFAULT 0,
                    rejected_images INTEGER NOT NULL DEFAULT 0,
                    failed_images INTEGER NOT NULL DEFAULT 0,
                    warning_count INTEGER NOT NULL DEFAULT 0,
                    progress_percentage REAL NOT NULL DEFAULT 0,
                    current_image TEXT,
                    started_at TEXT,
                    last_progress_at TEXT,
                    checkpoint_at TEXT,
                    completed_at TEXT,
                    report_path TEXT,
                    json_report_path TEXT,
                    error_message TEXT,
                    resumed_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS results (
                    job_id TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    sha256 TEXT,
                    result_json TEXT NOT NULL,
                    input_path TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, image_id)
                );
                CREATE INDEX IF NOT EXISTS idx_results_job ON results(job_id);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
            }
            migrations = {
                "last_progress_at": "ALTER TABLE jobs ADD COLUMN last_progress_at TEXT",
                "checkpoint_at": "ALTER TABLE jobs ADD COLUMN checkpoint_at TEXT",
                "resumed_count": "ALTER TABLE jobs ADD COLUMN resumed_count INTEGER NOT NULL DEFAULT 0",
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)

    def create_job(self, record: JobRecord) -> None:
        payload = record.model_dump(mode="json")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    job_id, stage, total_images, processed_images,
                    successful_images, review_required_images, rejected_images,
                    failed_images, warning_count, progress_percentage,
                    current_image, started_at, completed_at, report_path,
                    json_report_path, error_message, last_progress_at,
                    checkpoint_at, resumed_count
                ) VALUES (
                    :job_id, :stage, :total_images, :processed_images,
                    :successful_images, :review_required_images, :rejected_images,
                    :failed_images, :warning_count, :progress_percentage,
                    :current_image, :started_at, :completed_at, :report_path,
                    :json_report_path, :error_message, :last_progress_at,
                    :checkpoint_at, :resumed_count
                )
                """,
                payload,
            )

    def update_job(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        normalised = {
            key: value.value if hasattr(value, "value") else (
                value.isoformat() if isinstance(value, datetime) else value
            )
            for key, value in values.items()
        }
        assignments = ", ".join(f"{key} = :{key}" for key in normalised)
        normalised["job_id"] = job_id
        with self._connect() as connection:
            connection.execute(
                f"UPDATE jobs SET {assignments} WHERE job_id = :job_id",
                normalised,
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return JobRecord.model_validate(dict(row)) if row else None

    def save_result(self, job_id: str, result: ImageResult, input_path: str | None = None) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO results (
                    job_id, image_id, original_filename, sha256,
                    result_json, input_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    result.image_id,
                    result.original_filename,
                    result.sha256,
                    result.model_dump_json(),
                    input_path,
                    datetime.now().isoformat(),
                ),
            )

    def get_results(self, job_id: str) -> list[ImageResult]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT result_json FROM results WHERE job_id = ? ORDER BY created_at, original_filename",
                (job_id,),
            ).fetchall()
        return [ImageResult.model_validate_json(row["result_json"]) for row in rows]

    def result_input_path(self, job_id: str, image_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT input_path FROM results WHERE job_id = ? AND image_id = ?",
                (job_id, image_id),
            ).fetchone()
        return row["input_path"] if row else None

    def get_result_input_records(self, job_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT original_filename, sha256, input_path
                FROM results WHERE job_id = ? ORDER BY created_at
                """,
                (job_id,),
            ).fetchall()
        return [
            {
                "original_filename": str(row["original_filename"]),
                "sha256": str(row["sha256"] or ""),
                "input_path": str(row["input_path"] or ""),
            }
            for row in rows
        ]

    def mark_interrupted_jobs(self) -> None:
        terminal = {
            JobStage.COMPLETED.value,
            JobStage.COMPLETED_WITH_WARNINGS.value,
            JobStage.FAILED.value,
            JobStage.CANCELLED.value,
        }
        placeholders = ",".join("?" for _ in terminal)
        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE jobs
                SET stage = ?, error_message = ?, completed_at = ?
                WHERE stage NOT IN ({placeholders})
                """,
                (
                    JobStage.FAILED.value,
                    "Application restarted before this job completed",
                    datetime.now().isoformat(),
                    *terminal,
                ),
            )

    def delete_job(self, job_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM results WHERE job_id = ?", (job_id,))
            connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
