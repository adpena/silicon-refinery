"""
SiliconRefinery public API.

Root imports are provided lazily so CLI workflows can run even when the Apple
Foundation Models SDK is not installed in the current interpreter.
"""

from __future__ import annotations

from typing import Any

from .exceptions import AppleFMSetupError, raise_setup_error

__all__ = [
    "AppleFMSetupError",
    "Extract",
    "Sink",
    "Source",
    "enhanced_debug",
    "local_extract",
    "stream_extract",
]


def _raise_missing_sdk() -> None:
    raise_setup_error(
        "silicon_refinery", reason="ModuleNotFoundError: No module named 'apple_fm_sdk'"
    )


def __getattr__(name: str) -> Any:
    if name == "local_extract":
        try:
            from .decorators import local_extract
        except ModuleNotFoundError as exc:
            if exc.name == "apple_fm_sdk":
                _raise_missing_sdk()
            raise
        return local_extract

    if name == "stream_extract":
        try:
            from .async_generators import stream_extract
        except ModuleNotFoundError as exc:
            if exc.name == "apple_fm_sdk":
                _raise_missing_sdk()
            raise
        return stream_extract

    if name == "enhanced_debug":
        try:
            from .debugging import enhanced_debug
        except ModuleNotFoundError as exc:
            if exc.name == "apple_fm_sdk":
                _raise_missing_sdk()
            raise
        return enhanced_debug

    if name in {"Source", "Extract", "Sink"}:
        try:
            from .pipeline import Extract, Sink, Source
        except ModuleNotFoundError as exc:
            if exc.name == "apple_fm_sdk":
                _raise_missing_sdk()
            raise
        return {"Source": Source, "Extract": Extract, "Sink": Sink}[name]

    raise AttributeError(f"module 'silicon_refinery' has no attribute {name!r}")
