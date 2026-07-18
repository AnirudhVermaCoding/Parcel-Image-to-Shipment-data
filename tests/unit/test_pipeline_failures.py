from __future__ import annotations

import io

from PIL import Image

from src.pipeline import process_image
from src.schemas import PrimaryStatus, StatusFlag


def test_corrupt_image_returns_structured_rejection(test_config) -> None:
    result = process_image(b"\xff\xd8\xffbroken", "corrupt.jpg", test_config)
    assert result.primary_status == PrimaryStatus.REJECTED
    assert result.error_code
    assert result.original_filename == "corrupt.jpg"


def test_vision_is_disabled_without_key(test_config) -> None:
    image = Image.new("RGB", (320, 240), color=(220, 220, 220))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    test_config.enable_xai_fallback = True
    test_config.xai_api_key = None
    result = process_image(
        buffer.getvalue(),
        "synthetic.jpg",
        config=test_config,
    )
    assert "VISION_API_UNAVAILABLE" not in {flag.value for flag in result.status_flags}


def test_vision_provider_failure_keeps_local_result(test_config) -> None:
    class FailingProvider:
        def analyze(self, image_bytes: bytes):
            raise TimeoutError("provider timed out")

    image = Image.new("RGB", (320, 240), color=(12, 12, 12))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    test_config.enable_xai_fallback = True
    test_config.xai_api_key = "test-only-key"
    result = process_image(
        buffer.getvalue(),
        "empty.jpg",
        test_config,
        vision_provider=FailingProvider(),
    )
    assert StatusFlag.VISION_API_UNAVAILABLE in result.status_flags
    assert result.error_code is None
