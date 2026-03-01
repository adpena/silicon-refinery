"""
Per-task / per-coroutine session scoping via :mod:`contextvars`.

Different async tasks get isolated FM sessions without manual passing:

.. code-block:: python

    async with session_scope("Extract names.") as session:
        result = await session.respond("Alice is 30", generating=PersonSchema)

The three context variables (_current_model, _current_session, _current_instructions)
are automatically set on entry and reset on exit, so nested or concurrent scopes
never leak into each other.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from .protocols import create_model, create_session

logger = logging.getLogger("fmtools")

# ---------------------------------------------------------------------------
# Context variables
# ---------------------------------------------------------------------------

_current_model: contextvars.ContextVar[Any] = contextvars.ContextVar("_current_model")
_current_session: contextvars.ContextVar[Any] = contextvars.ContextVar("_current_session")
_current_instructions: contextvars.ContextVar[str] = contextvars.ContextVar("_current_instructions")


# ---------------------------------------------------------------------------
# Async context manager
# ---------------------------------------------------------------------------


@asynccontextmanager
async def session_scope(
    instructions: str,
    model: Any | None = None,
) -> AsyncIterator[Any]:
    """Open an isolated FM session scope for the current async task.

    Args:
        instructions: System instructions for the session.
        model: An existing backend model instance. If *None*, a model is
            created via :func:`fmtools.protocols.create_model`.

    Yields:
        The backend session bound to this scope.
    """
    if model is None:
        model = create_model()

    session = create_session(instructions=instructions, model=model)

    # Set context variables and capture reset tokens
    tok_model = _current_model.set(model)
    tok_session = _current_session.set(session)
    tok_instructions = _current_instructions.set(instructions)

    try:
        yield session
    finally:
        # Reset to previous values regardless of success/failure
        _current_model.reset(tok_model)
        _current_session.reset(tok_session)
        _current_instructions.reset(tok_instructions)


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------


def get_session() -> Any:
    """Return the current FM session from :mod:`contextvars`.

    Raises:
        RuntimeError: If called outside of a :func:`session_scope`.
    """
    try:
        return _current_session.get()
    except LookupError:
        raise RuntimeError(
            "No active session scope. Wrap your code in 'async with session_scope(...):'."
        ) from None


def get_model() -> Any:
    """Return the current FM model from :mod:`contextvars`.

    Raises:
        RuntimeError: If called outside of a :func:`session_scope`.
    """
    try:
        return _current_model.get()
    except LookupError:
        raise RuntimeError(
            "No active session scope. Wrap your code in 'async with session_scope(...):'."
        ) from None


def get_instructions() -> str:
    """Return the current instructions string from :mod:`contextvars`.

    Raises:
        RuntimeError: If called outside of a :func:`session_scope`.
    """
    try:
        return _current_instructions.get()
    except LookupError:
        raise RuntimeError(
            "No active session scope. Wrap your code in 'async with session_scope(...):'."
        ) from None


def copy_context() -> contextvars.Context:
    """Snapshot the current :mod:`contextvars` context.

    Useful when spawning sub-tasks that should inherit the current session:

    .. code-block:: python

        ctx = copy_context()
        loop.run_in_executor(None, ctx.run, sync_fn)
    """
    return contextvars.copy_context()
