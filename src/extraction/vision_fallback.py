from __future__ import annotations

import base64
import hashlib
import json
import random
import sqlite3
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Protocol

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
from PIL import Image
from pydantic import ValidationError

from src.config import AppConfig
from src.schemas import VisionAnalysis, VisionDiagnostic, VisionObservation


class VisionProvider(Protocol):
    def analyze(
        self,
        image_bytes: bytes,
        label_crop_bytes: bytes | None = None,
        include_full_frame: bool = True,
    ) -> VisionAnalysis | VisionObservation: ...


class VisionProviderError(RuntimeError):
    def __init__(self, message: str, diagnostic: VisionDiagnostic | None = None):
        super().__init__(message)
        self.diagnostic = diagnostic or VisionDiagnostic(category="provider_error", message=message)


def _category_for_exception(exc: Exception) -> tuple[str, int | None, str | None, bool]:
    if isinstance(exc, RateLimitError):
        return "rate_limit", 429, getattr(exc, "request_id", None), True
    if isinstance(exc, APITimeoutError):
        return "timeout", None, getattr(exc, "request_id", None), True
    if isinstance(exc, APIConnectionError):
        return "network", None, getattr(exc, "request_id", None), True
    if isinstance(exc, APIStatusError):
        status = int(exc.status_code)
        request_id = getattr(exc, "request_id", None)
        if status in {401, 403}:
            return "authentication", status, request_id, False
        if status == 429:
            return "rate_limit", status, request_id, True
        if status in {400, 404, 409, 422}:
            return "invalid_request", status, request_id, False
        return "server_error" if status >= 500 else "api_error", status, request_id, status >= 500
    if isinstance(exc, (ValidationError, json.JSONDecodeError, ValueError, TypeError)):
        return "malformed_response", None, None, False
    return "unexpected", None, None, False


class VisionCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS vision_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def get(self, key: str) -> VisionObservation | None:
        with self.lock, self._connect() as connection:
            row = connection.execute(
                "SELECT result_json FROM vision_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        try:
            return VisionObservation.model_validate_json(row[0])
        except ValidationError:
            return None

    def put(self, key: str, value: VisionObservation, model: str, prompt_version: str) -> None:
        with self.lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO vision_cache
                (cache_key, result_json, model, prompt_version, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, value.model_dump_json(), model, prompt_version, int(time.time())),
            )


class XaiVisionProvider:
    def __init__(self, config: AppConfig):
        if not config.xai_api_key:
            raise VisionProviderError(
                "XAI_API_KEY is not configured",
                VisionDiagnostic(category="not_configured", message="API key is not configured"),
            )
        self.config = config
        self.client = OpenAI(
            api_key=config.xai_api_key,
            base_url=config.xai_base_url,
            timeout=config.xai_timeout_seconds,
            max_retries=0,
        )
        self.semaphore = threading.BoundedSemaphore(max(1, config.xai_max_concurrency))
        self.cache = (
            VisionCache(config.runtime_dir / "vision_cache.sqlite3")
            if config.vision_cache_enabled
            else None
        )

    def _prepare_image(self, image_bytes: bytes) -> str:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image.thumbnail(
                (self.config.xai_max_image_side, self.config.xai_max_image_side),
                Image.Resampling.LANCZOS,
            )
            output = BytesIO()
            image.save(output, format="JPEG", quality=86, optimize=True)
        return base64.b64encode(output.getvalue()).decode("ascii")

    def _cache_key(
        self,
        image_bytes: bytes,
        label_crop_bytes: bytes | None,
        include_full_frame: bool,
    ) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.xai_model.encode())
        digest.update(self.config.vision_prompt_version.encode())
        digest.update(VisionObservation.model_json_schema().__repr__().encode())
        digest.update(b"full" if include_full_frame else b"crop")
        if include_full_frame:
            digest.update(image_bytes)
        if label_crop_bytes:
            digest.update(label_crop_bytes)
        return digest.hexdigest()

    @staticmethod
    def _output_text(response) -> str:
        direct = getattr(response, "output_text", None)
        if direct:
            return str(direct)
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    return str(text)
        raise ValueError("xAI response did not contain structured output")

    def analyze(
        self,
        image_bytes: bytes,
        label_crop_bytes: bytes | None = None,
        include_full_frame: bool = True,
        *,
        bypass_cache: bool = False,
    ) -> VisionAnalysis:
        started = time.perf_counter()
        key = self._cache_key(image_bytes, label_crop_bytes, include_full_frame)
        if self.cache and not bypass_cache:
            cached = self.cache.get(key)
            if cached is not None:
                return VisionAnalysis(
                    observation=cached,
                    diagnostic=VisionDiagnostic(
                        success=True,
                        category="success",
                        message="Reused cached structured observation",
                        cache_hit=True,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                    ),
                )

        content: list[dict[str, str]] = []
        if include_full_frame:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{self._prepare_image(image_bytes)}",
                    "detail": "high",
                }
            )
        if label_crop_bytes:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{self._prepare_image(label_crop_bytes)}",
                    "detail": "high",
                }
            )
        content.append(
            {
                "type": "input_text",
                "text": (
                    "Inspect this parcel-scanner evidence conservatively. The first image is the "
                    "full frame when present and the second is the best label crop. Return only "
                    "visible evidence. Never guess an AWB, weight, or dimensions. Use null when "
                    "characters are not unambiguous."
                ),
            }
        )

        schema = VisionObservation.model_json_schema()
        attempts = max(1, self.config.xai_max_retries + 1)
        last_error: Exception | None = None
        last_category = "unexpected"
        last_status: int | None = None
        last_request_id: str | None = None
        for attempt in range(1, attempts + 1):
            try:
                with self.semaphore:
                    response = self.client.responses.create(
                        model=self.config.xai_model,
                        store=False,
                        input=[{"role": "user", "content": content}],
                        text={
                            "format": {
                                "type": "json_schema",
                                "name": "parcel_observation",
                                "schema": schema,
                                "strict": True,
                            }
                        },
                    )
                observation = VisionObservation.model_validate_json(self._output_text(response))
                request_id = getattr(response, "id", None)
                diagnostic = VisionDiagnostic(
                    success=True,
                    category="success",
                    message="Structured vision observation received",
                    request_id=request_id,
                    attempt_count=attempt,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                if self.cache:
                    self.cache.put(
                        key,
                        observation,
                        self.config.xai_model,
                        self.config.vision_prompt_version,
                    )
                return VisionAnalysis(observation=observation, diagnostic=diagnostic)
            except Exception as exc:  # mapped below; no raw response data is exposed
                last_error = exc
                last_category, last_status, last_request_id, retryable = _category_for_exception(exc)
                if not retryable or attempt >= attempts:
                    break
                delay = min(4.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0.0, 0.25)
                time.sleep(delay)

        diagnostic = VisionDiagnostic(
            success=False,
            category=last_category,
            message=f"Vision request failed: {last_category}",
            http_status=last_status,
            request_id=last_request_id,
            attempt_count=attempt,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        raise VisionProviderError(diagnostic.message, diagnostic) from last_error

    def cached_analysis(
        self,
        image_bytes: bytes,
        label_crop_bytes: bytes | None = None,
        include_full_frame: bool = True,
    ) -> VisionAnalysis | None:
        if not self.cache:
            return None
        key = self._cache_key(image_bytes, label_crop_bytes, include_full_frame)
        cached = self.cache.get(key)
        if cached is None:
            return None
        return VisionAnalysis(
            observation=cached,
            diagnostic=VisionDiagnostic(
                success=True,
                category="success",
                message="Reused cached structured observation",
                cache_hit=True,
            ),
        )

    def diagnose(self) -> VisionDiagnostic:
        image = Image.new("RGB", (64, 64), "white")
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        try:
            return self.analyze(buffer.getvalue(), bypass_cache=True).diagnostic
        except VisionProviderError as exc:
            return exc.diagnostic
