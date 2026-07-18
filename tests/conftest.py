from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig


@pytest.fixture
def sample_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "sample_data" / "supplied"


@pytest.fixture
def test_config(tmp_path: Path) -> AppConfig:
    config = AppConfig.from_env()
    config.runtime_dir = tmp_path / "runtime"
    config.max_workers = 1
    config.max_batch_images = 500
    config.generate_annotations = False
    config.enable_xai_fallback = False
    config.xai_api_key = None
    return config

