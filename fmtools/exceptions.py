"""
Custom exception utilities for Apple Foundation Models setup checks.

This module centralizes user-facing setup diagnostics so examples, APIs, and
CLI output consistent troubleshooting guidance when the SDK is unavailable.
"""

from __future__ import annotations

import importlib
from typing import Any, NoReturn

__all__ = [
    "AppleFMSetupError",
    "ensure_model_available",
    "raise_setup_error",
    "require_apple_fm",
    "troubleshooting_message",
]


class AppleFMSetupError(RuntimeError):
    """Raised when Apple FM SDK/model setup is missing or unavailable."""


def troubleshooting_message(context: str, reason: str | None = None) -> str:
    """Build a standard setup troubleshooting message."""
    label = context.strip() if context.strip() else "fmtools"
    lines = [f"[{label}] Apple Foundation Models setup check failed."]
    if reason:
        lines.append(f"Reason: {reason}")
    lines.extend(
        [
            "",
            "Troubleshooting checklist:",
            "1. Use macOS 26+ on Apple Silicon (M-series).",
            "2. Install dependencies: uv sync --all-groups",
            "3. Activate environment: source .venv/bin/activate",
            "4. Verify SDK import:",
            '   uv run python -c "import apple_fm_sdk as fm; print(fm.__name__)"',
            "5. Verify model availability:",
            '   uv run python -c "import apple_fm_sdk as fm; m=fm.SystemLanguageModel(); print(m.is_available())"',
            "6. Run diagnostics: uv run fmtools doctor",
        ]
    )
    return "\n".join(lines)


def raise_setup_error(
    context: str,
    *,
    reason: str | None = None,
    exc: BaseException | None = None,
) -> NoReturn:
    """Raise :class:`AppleFMSetupError` with standardized diagnostics."""
    computed_reason = reason
    if computed_reason is None and exc is not None:
        computed_reason = f"{type(exc).__name__}: {exc}"

    error = AppleFMSetupError(troubleshooting_message(context, reason=computed_reason))
    if exc is not None:
        raise error from exc
    raise error


def ensure_model_available(model: Any, *, context: str) -> None:
    """Validate that a model can be used for local inference."""
    try:
        available, reason = model.is_available()
    except Exception as exc:
        raise_setup_error(context, exc=exc)

    if not available:
        detail = f"Foundation Model is not available: {reason}"
        raise_setup_error(context, reason=detail)


def require_apple_fm(context: str) -> tuple[Any, Any]:
    """Import ``apple_fm_sdk`` and ensure a local model is available."""
    try:
        fm = importlib.import_module("apple_fm_sdk")
    except Exception as exc:  # pragma: no cover - import error path
        raise_setup_error(context, exc=exc)

    try:
        model = fm.SystemLanguageModel()
    except Exception as exc:
        raise_setup_error(context, exc=exc)

    ensure_model_available(model, context=context)
    return fm, model
