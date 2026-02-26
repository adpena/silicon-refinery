"""
Pluggable backend protocols for SiliconRefinery.

Defines structural subtyping (typing.Protocol) interfaces so the framework can
work with any backend that satisfies the contract, not just Apple FM SDK.

Usage:
    from silicon_refinery.protocols import set_backend, get_backend

    # Default: AppleFMBackend wrapping apple_fm_sdk
    backend = get_backend()

    # Swap in a custom backend for testing or alternative providers:
    set_backend(my_custom_backend)
"""

from __future__ import annotations

import importlib
import logging
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger("silicon_refinery")

T = TypeVar("T")

__all__ = [
    "AppleFMBackend",
    "AppleFMModel",
    "AppleFMSession",
    "ModelProtocol",
    "SessionFactory",
    "SessionProtocol",
    "create_model",
    "create_session",
    "get_backend",
    "set_backend",
]


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural interface for a language model availability check."""

    def is_available(self) -> tuple[bool, str | None]:
        """Return (available, reason_if_not)."""
        ...


@runtime_checkable
class SessionProtocol(Protocol):
    """Structural interface for a language model session."""

    def respond(self, prompt: str, generating: type[T] | None = None) -> Awaitable[T]:
        """Send a prompt and return a structured (or plain) response."""
        ...


@runtime_checkable
class SessionFactory(Protocol):
    """Callable that creates a SessionProtocol from a model + instructions."""

    def __call__(self, model: ModelProtocol, instructions: str) -> SessionProtocol: ...


# ---------------------------------------------------------------------------
# Apple FM concrete backend
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _import_apple_fm_sdk() -> Any:
    """Import ``apple_fm_sdk`` lazily so protocol import does not hard-require it."""
    return importlib.import_module("apple_fm_sdk")


class AppleFMModel:
    """Wraps ``apple_fm_sdk.SystemLanguageModel`` behind :class:`ModelProtocol`."""

    def __init__(self) -> None:
        fm_sdk = _import_apple_fm_sdk()
        self._model = fm_sdk.SystemLanguageModel()

    def is_available(self) -> tuple[bool, str | None]:
        available, reason = self._model.is_available()
        return available, None if reason is None else str(reason)

    @property
    def raw(self) -> Any:
        """Access the underlying SDK model object."""
        return self._model


class AppleFMSession:
    """Wraps ``apple_fm_sdk.LanguageModelSession`` behind :class:`SessionProtocol`."""

    def __init__(self, model: ModelProtocol, instructions: str) -> None:
        fm_sdk = _import_apple_fm_sdk()
        # Accept either our wrapper or the raw SDK model
        raw_model = cast("Any", getattr(model, "raw", model))
        self._session = fm_sdk.LanguageModelSession(model=raw_model, instructions=instructions)

    async def respond(self, prompt: str, generating: type[T] | None = None) -> Any:
        return await self._session.respond(prompt, generating=cast("Any", generating))


class AppleFMBackend:
    """
    Default backend that delegates to ``apple_fm_sdk``.

    Satisfies both :class:`ModelProtocol` (via ``create_model``) and
    :class:`SessionFactory` (via ``__call__``).
    """

    def create_model(self) -> AppleFMModel:
        """Create a new :class:`AppleFMModel`."""
        return AppleFMModel()

    def __call__(self, model: ModelProtocol, instructions: str) -> AppleFMSession:
        """Create a new :class:`AppleFMSession` (satisfies :class:`SessionFactory`)."""
        return AppleFMSession(model, instructions)


# ---------------------------------------------------------------------------
# Module-level backend registry
# ---------------------------------------------------------------------------

_backend: Any = AppleFMBackend()


def set_backend(backend: Any) -> None:
    """Replace the active backend (module-level singleton)."""
    global _backend
    _backend = backend
    logger.info("[SiliconRefinery] Backend set to %s", type(backend).__name__)


def get_backend() -> Any:
    """Return the currently active backend."""
    return _backend


def _create_model_from_backend(backend: Any) -> ModelProtocol:
    create_model_fn = getattr(backend, "create_model", None)
    if not callable(create_model_fn):
        raise TypeError(f"Active backend must provide create_model(); got {type(backend).__name__}")
    return cast("ModelProtocol", create_model_fn())


def create_model() -> ModelProtocol:
    """Create a model using the currently active backend."""
    return _create_model_from_backend(get_backend())


def create_session(
    instructions: str,
    model: ModelProtocol | None = None,
) -> SessionProtocol:
    """Create a session via the active backend.

    If *model* is omitted, a new model is created via :func:`create_model`.
    """
    backend = get_backend()
    resolved_model = model if model is not None else _create_model_from_backend(backend)

    if callable(backend):
        return cast("SessionProtocol", backend(resolved_model, instructions))

    create_session_fn = getattr(backend, "create_session", None)
    if callable(create_session_fn):
        return cast("SessionProtocol", create_session_fn(resolved_model, instructions))

    raise TypeError(
        "Active backend must be callable(model, instructions) or "
        f"provide create_session(); got {type(backend).__name__}"
    )
