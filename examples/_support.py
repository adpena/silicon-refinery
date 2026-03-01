"""
Shared support helpers for runnable examples.

Provides a custom setup exception and consistent troubleshooting guidance for
Apple Foundation Models SDK availability.
"""

from __future__ import annotations

from typing import Any

from fmtools.exceptions import (
    AppleFMSetupError,
    raise_setup_error,
)
from fmtools.exceptions import (
    require_apple_fm as _require_apple_fm,
)

__all__ = ["AppleFMSetupError", "raise_sdk_setup_error", "require_apple_fm"]


def raise_sdk_setup_error(example_name: str, exc: BaseException) -> None:
    """Raise a user-friendly setup exception from a lower-level import/runtime error."""
    raise_setup_error(example_name, exc=exc)


def require_apple_fm(example_name: str) -> tuple[Any, Any]:
    """Ensure `apple_fm_sdk` imports and the local model reports available."""
    return _require_apple_fm(example_name)
