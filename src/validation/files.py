from __future__ import annotations

import hashlib
import io
import re
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from PIL import Image

from src.config import AppConfig

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
JPEG_SIGNATURES = (b"\xff\xd8\xff",)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class InputValidationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class InputFile:
    original_filename: str
    data: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


def safe_display_name(name: str) -> str:
    name = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._() -]+", "_", name).strip(" .")
    return cleaned[:180] or "unnamed-image"


def detect_image_type(data: bytes) -> str | None:
    if any(data.startswith(signature) for signature in JPEG_SIGNATURES):
        return ".jpg"
    if data.startswith(PNG_SIGNATURE):
        return ".png"
    return None


def validate_image_bytes(name: str, data: bytes, config: AppConfig) -> str:
    if not data:
        raise InputValidationError("EMPTY_FILE", f"{name}: file is empty")
    if len(data) > config.max_file_bytes:
        raise InputValidationError("FILE_TOO_LARGE", f"{name}: exceeds file-size limit")
    extension = Path(name).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise InputValidationError("UNSUPPORTED_IMAGE", f"{name}: unsupported extension")
    detected = detect_image_type(data)
    if detected is None:
        raise InputValidationError("BAD_SIGNATURE", f"{name}: signature is not JPEG or PNG")
    if extension == ".png" and detected != ".png":
        raise InputValidationError("BAD_SIGNATURE", f"{name}: extension/signature mismatch")
    if extension in {".jpg", ".jpeg"} and detected != ".jpg":
        raise InputValidationError("BAD_SIGNATURE", f"{name}: extension/signature mismatch")
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image.load()
    except Exception as exc:
        raise InputValidationError("CORRUPT_IMAGE", f"{name}: cannot decode image") from exc
    if width <= 0 or height <= 0:
        raise InputValidationError("CORRUPT_IMAGE", f"{name}: invalid dimensions")
    if width > config.max_dimension or height > config.max_dimension:
        raise InputValidationError("IMAGE_TOO_LARGE", f"{name}: dimension limit exceeded")
    if width * height > config.max_pixels:
        raise InputValidationError("IMAGE_TOO_LARGE", f"{name}: pixel limit exceeded")
    return detected


def internal_filename(extension: str) -> str:
    return f"{uuid.uuid4().hex}{extension}"


def extract_zip_bytes(name: str, data: bytes, config: AppConfig) -> tuple[list[InputFile], list[str]]:
    files: list[InputFile] = []
    warnings: list[str] = []
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise InputValidationError("CORRUPT_ZIP", f"{name}: invalid ZIP file") from exc
    members = archive.infolist()
    if not members:
        raise InputValidationError("EMPTY_ZIP", f"{name}: ZIP is empty")
    if len(members) > config.max_zip_entries:
        raise InputValidationError("ZIP_TOO_MANY_ENTRIES", f"{name}: too many ZIP entries")
    total_uncompressed = 0
    for member in members:
        if member.is_dir():
            continue
        pure = PurePosixPath(member.filename.replace("\\", "/"))
        if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
            raise InputValidationError("ZIP_PATH_TRAVERSAL", f"{name}: unsafe ZIP path")
        if member.filename.lower().endswith((".zip", ".tar", ".gz", ".7z", ".rar")):
            warnings.append(f"Ignored nested archive: {member.filename}")
            continue
        total_uncompressed += member.file_size
        if total_uncompressed > config.max_zip_uncompressed_bytes:
            raise InputValidationError("ZIP_TOO_LARGE", f"{name}: uncompressed limit exceeded")
        compressed = max(member.compress_size, 1)
        if member.file_size / compressed > config.max_zip_ratio:
            raise InputValidationError("ZIP_SUSPICIOUS_RATIO", f"{name}: suspicious compression ratio")
        extension = Path(member.filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            warnings.append(f"Ignored unsupported ZIP entry: {member.filename}")
            continue
        member_data = archive.read(member)
        try:
            validate_image_bytes(member.filename, member_data, config)
        except InputValidationError as exc:
            warnings.append(str(exc))
            continue
        files.append(InputFile(safe_display_name(member.filename), member_data))
    if not files:
        raise InputValidationError("EMPTY_ZIP", f"{name}: no supported images found")
    return files, warnings


def collect_path_inputs(path: Path, config: AppConfig) -> tuple[list[InputFile], list[str]]:
    if path.is_dir():
        candidates = sorted(
            item for item in path.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(candidates) > config.max_batch_images:
            raise InputValidationError("BATCH_TOO_LARGE", "Batch image limit exceeded")
        files = [InputFile(item.name, item.read_bytes()) for item in candidates]
        for item in files:
            validate_image_bytes(item.original_filename, item.data, config)
        return files, []
    if path.suffix.lower() == ".zip":
        return extract_zip_bytes(path.name, path.read_bytes(), config)
    data = path.read_bytes()
    validate_image_bytes(path.name, data, config)
    return [InputFile(path.name, data)], []

