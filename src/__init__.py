"""Local parcel-image extraction package.

Keep package initialization lightweight so importing a leaf module such as
``src.config`` does not initialize the complete OCR pipeline. The public
``process_image`` convenience export remains available through lazy loading.
"""

from typing import Any

__all__ = ["process_image"]


def __getattr__(name: str) -> Any:
    if name == "process_image":
        from src.pipeline import process_image

        return process_image
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
