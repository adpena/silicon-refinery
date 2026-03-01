"""
Tests for fmtools._context — contextvars session scoping.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from fmtools._context import (
    copy_context,
    get_instructions,
    get_model,
    get_session,
    session_scope,
)

# ---------------------------------------------------------------------------
# session_scope
# ---------------------------------------------------------------------------


async def test_session_scope_creates_session():
    """session_scope should create a model and session, then yield the session."""
    mock_model = MagicMock(name="model")
    mock_session = MagicMock(name="session")

    with (
        patch("fmtools._context.create_model", return_value=mock_model) as create_model,
        patch("fmtools._context.create_session", return_value=mock_session) as create_session,
    ):
        async with session_scope("Extract names.") as session:
            assert session is mock_session
            create_model.assert_called_once_with()
            create_session.assert_called_once_with(instructions="Extract names.", model=mock_model)


async def test_session_scope_uses_provided_model():
    """When a model is passed explicitly, session_scope must not create a new one."""
    provided_model = MagicMock(name="provided_model")
    mock_session = MagicMock(name="session")

    with (
        patch("fmtools._context.create_model") as create_model,
        patch("fmtools._context.create_session", return_value=mock_session) as create_session,
    ):
        async with session_scope("inst", model=provided_model) as session:
            create_model.assert_not_called()
            create_session.assert_called_once_with(instructions="inst", model=provided_model)
            assert session is mock_session


async def test_session_scope_resets_on_exit():
    """After exiting session_scope, context variables must be unset."""
    mock_model = MagicMock()
    mock_session = MagicMock()

    with (
        patch("fmtools._context.create_model", return_value=mock_model),
        patch("fmtools._context.create_session", return_value=mock_session),
    ):
        async with session_scope("inst"):
            # Inside scope — should be set
            assert get_session() is mock_session

        # Outside scope — should raise
        with pytest.raises(RuntimeError):
            get_session()


async def test_nested_session_scopes():
    """Nested scopes should shadow and then restore outer values."""
    outer_model = MagicMock(name="outer_model")
    outer_session = MagicMock(name="outer_session")
    inner_model = MagicMock(name="inner_model")
    inner_session = MagicMock(name="inner_session")

    with (
        patch(
            "fmtools._context.create_model",
            side_effect=[outer_model, inner_model],
        ),
        patch(
            "fmtools._context.create_session",
            side_effect=[outer_session, inner_session],
        ),
    ):
        async with session_scope("outer") as s_outer:
            assert s_outer is outer_session
            assert get_instructions() == "outer"

            async with session_scope("inner") as s_inner:
                assert s_inner is inner_session
                assert get_instructions() == "inner"

            # Back to outer
            assert get_session() is outer_session
            assert get_instructions() == "outer"


# ---------------------------------------------------------------------------
# Accessor errors
# ---------------------------------------------------------------------------


async def test_get_session_outside_scope_raises():
    with pytest.raises(RuntimeError, match="No active session scope"):
        get_session()


async def test_get_model_outside_scope_raises():
    with pytest.raises(RuntimeError, match="No active session scope"):
        get_model()


async def test_get_instructions_returns_correct_value():
    mock_model = MagicMock()
    mock_session = MagicMock()

    with (
        patch("fmtools._context.create_model", return_value=mock_model),
        patch("fmtools._context.create_session", return_value=mock_session),
    ):
        async with session_scope("my custom instructions"):
            assert get_instructions() == "my custom instructions"


# ---------------------------------------------------------------------------
# copy_context
# ---------------------------------------------------------------------------


async def test_copy_context_preserves_session():
    """A snapshot taken inside a scope should still expose the session."""
    mock_model = MagicMock()
    mock_session = MagicMock()

    with (
        patch("fmtools._context.create_model", return_value=mock_model),
        patch("fmtools._context.create_session", return_value=mock_session),
    ):
        async with session_scope("inst"):
            ctx = copy_context()

        # Run get_session *inside* the snapshot context — should succeed
        result = ctx.run(get_session)
        assert result is mock_session


# ---------------------------------------------------------------------------
# Concurrency isolation
# ---------------------------------------------------------------------------


async def test_concurrent_scopes_are_isolated():
    """Two concurrent async tasks with different scopes must not interfere."""
    model_a = MagicMock(name="model_a")
    session_a = MagicMock(name="session_a")
    model_b = MagicMock(name="model_b")
    session_b = MagicMock(name="session_b")

    barrier = asyncio.Event()
    results: dict[str, object] = {}

    async def task_a() -> None:
        with (
            patch(
                "fmtools._context.create_model",
                return_value=model_a,
            ),
            patch(
                "fmtools._context.create_session",
                return_value=session_a,
            ),
        ):
            async with session_scope("task_a"):
                results["a_session"] = get_session()
                results["a_instructions"] = get_instructions()
                barrier.set()  # let task_b proceed
                # Give task_b time to set its own scope
                await asyncio.sleep(0.05)
                # Should still see its own values
                results["a_session_after"] = get_session()

    async def task_b() -> None:
        await barrier.wait()
        with (
            patch(
                "fmtools._context.create_model",
                return_value=model_b,
            ),
            patch(
                "fmtools._context.create_session",
                return_value=session_b,
            ),
        ):
            async with session_scope("task_b"):
                results["b_session"] = get_session()
                results["b_instructions"] = get_instructions()

    await asyncio.gather(task_a(), task_b())

    assert results["a_session"] is session_a
    assert results["b_session"] is session_b
    assert results["a_instructions"] == "task_a"
    assert results["b_instructions"] == "task_b"
    assert results["a_session_after"] is session_a
