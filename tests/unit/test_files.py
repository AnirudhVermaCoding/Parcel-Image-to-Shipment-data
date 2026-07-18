from __future__ import annotations

import io
import zipfile

import pytest

from src.validation.files import InputValidationError, extract_zip_bytes, validate_image_bytes


def test_signature_validation_rejects_fake_jpeg(test_config) -> None:
    with pytest.raises(InputValidationError) as exc:
        validate_image_bytes("fake.jpg", b"not an image", test_config)
    assert exc.value.code == "BAD_SIGNATURE"


def test_zip_path_traversal_is_rejected(test_config) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.jpg", b"\xff\xd8\xffbroken")
    with pytest.raises(InputValidationError) as exc:
        extract_zip_bytes("unsafe.zip", buffer.getvalue(), test_config)
    assert exc.value.code == "ZIP_PATH_TRAVERSAL"


def test_empty_zip_is_rejected(test_config) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    with pytest.raises(InputValidationError) as exc:
        extract_zip_bytes("empty.zip", buffer.getvalue(), test_config)
    assert exc.value.code == "EMPTY_ZIP"

