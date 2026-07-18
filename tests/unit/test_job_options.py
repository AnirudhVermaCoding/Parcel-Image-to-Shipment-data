from __future__ import annotations

from src.jobs.manager import JobManager
from src.schemas import (
    BatchOptions,
    ImageResult,
    PrimaryStatus,
    ProcessingStatus,
)
from src.validation.files import InputFile


def test_local_only_job_overrides_globally_configured_vision(monkeypatch, test_config) -> None:
    observed = {}

    def fake_process(path, original_filename, config, vision_provider):
        observed["vision_enabled"] = config.enable_xai_fallback
        observed["provider"] = vision_provider
        return ImageResult(
            image_id="result-1",
            original_filename=original_filename,
            processing_status=ProcessingStatus.COMPLETED,
            primary_status=PrimaryStatus.SUCCESS_PARTIAL_FIELDS,
        )

    monkeypatch.setattr("src.jobs.manager.process_image", fake_process)
    test_config.enable_xai_fallback = True
    test_config.xai_api_key = "configured-but-not-consented"
    manager = JobManager(test_config)
    manager.submit(
        [InputFile("parcel.jpg", b"placeholder")],
        BatchOptions(local_only=True, enable_vision=False, max_workers=1),
        background=False,
    )
    assert observed == {"vision_enabled": False, "provider": None}

