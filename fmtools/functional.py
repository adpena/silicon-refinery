"""
Functional Pipeline API for FMTools.

A functional/compositional pipeline API inspired by Elixir pipes and Haskell,
complementing the existing OOP Pipeline class. Pipelines are built by composing
async generator transforms that each accept an ``AsyncIterable`` and yield items.

Usage::

    pipeline = pipe(
        source(["Alice is 30", "Bob is 25"]),
        extract(PersonSchema, instructions="Extract person."),
        filter_fn(lambda p: p.age > 20),
        map_fn(lambda p: f"{p.name}: {p.age}"),
        collect(),
    )
    results = await pipeline()
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterable, AsyncIterator, Callable, Iterable
from typing import Any, TypeVar, Union

from .protocols import create_model, create_session

logger = logging.getLogger("fmtools")

T = TypeVar("T")
U = TypeVar("U")

__all__ = [
    "batch",
    "collect",
    "extract",
    "filter_fn",
    "flat_map_fn",
    "map_fn",
    "pipe",
    "reduce_fn",
    "skip",
    "source",
    "take",
    "tap",
]


# ---------------------------------------------------------------------------
# Step wrapper — enables ``step1 | step2`` composition via __or__
# ---------------------------------------------------------------------------


class Step:
    """Wraps an async-generator transform so it can be composed with ``|``."""

    def __init__(self, fn: Callable[..., Any], *, is_terminal: bool = False) -> None:
        self._fn = fn
        self.is_terminal = is_terminal

    def __call__(self, upstream: AsyncIterable[Any]) -> Any:
        return self._fn(upstream)

    def __or__(self, other: Step) -> Step:
        """Compose two steps: ``self | other`` feeds self's output into other."""
        left = self
        right = other
        if left.is_terminal:
            raise ValueError("Cannot compose additional steps after a terminal step")

        if right.is_terminal:
            # Terminal steps return a value, not an async iterable
            async def _composed(upstream: AsyncIterable[Any]) -> Any:
                return await right(left(upstream))

            return Step(_composed, is_terminal=True)

        async def _composed_gen(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
            async for item in right(left(upstream)):
                yield item

        return Step(_composed_gen, is_terminal=False)


# ---------------------------------------------------------------------------
# Helper: turn a sync/async iterable into an async iterable
# ---------------------------------------------------------------------------


async def _to_async(
    iterable: Union[Iterable[Any], AsyncIterable[Any]],
) -> AsyncIterator[Any]:
    """Normalise any iterable into an ``AsyncIterator``."""
    if isinstance(iterable, AsyncIterable):
        async for item in iterable:
            yield item
    else:
        for idx, item in enumerate(iterable):
            yield item
            if idx > 0 and idx % 100 == 0:
                await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Source step
# ---------------------------------------------------------------------------


def source(iterable: Union[Iterable[Any], AsyncIterable[Any]]) -> Step:
    """Step that yields items from an iterable (sync or async).

    This is typically the first step in a pipeline.
    """

    async def _source(_upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        async for item in _to_async(iterable):
            yield item

    return Step(_source)


# ---------------------------------------------------------------------------
# Extract step (uses apple_fm_sdk)
# ---------------------------------------------------------------------------


def extract(
    schema: type[Any],
    instructions: str = "Extract and structure this input.",
    on_error: str = "skip",
) -> Step:
    """Step that runs Foundation Model extraction on each item.

    Args:
        schema: A class decorated with ``@apple_fm_sdk.generable()``.
        instructions: System prompt for the FM session.
        on_error: Error handling strategy — ``"skip"``, ``"raise"``, or ``"yield_none"``.
    """

    valid_on_error = {"skip", "raise", "yield_none"}
    if on_error not in valid_on_error:
        raise ValueError(
            f"Invalid on_error={on_error!r}. Must be one of: {', '.join(sorted(valid_on_error))}"
        )

    async def _extract(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        model = create_model()
        async for item in upstream:
            session = create_session(instructions=instructions, model=model)
            try:
                result = await session.respond(str(item), generating=schema)
                yield result
            except Exception as exc:
                logger.warning("[functional.extract] Failed on item: %s", exc)
                if on_error == "raise":
                    raise
                elif on_error == "yield_none":
                    yield None
                # else: skip

    return Step(_extract)


# ---------------------------------------------------------------------------
# Transform steps
# ---------------------------------------------------------------------------


def filter_fn(predicate: Callable[[Any], Any]) -> Step:
    """Step that filters items by *predicate* (sync or async callable)."""

    async def _filter(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        async for item in upstream:
            keep = predicate(item)
            if inspect.isawaitable(keep):
                keep = await keep
            if keep:
                yield item

    return Step(_filter)


def map_fn(func: Callable[[Any], Any]) -> Step:
    """Step that transforms each item with *func* (sync or async callable)."""

    async def _map(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        async for item in upstream:
            result = func(item)
            if inspect.isawaitable(result):
                result = await result
            yield result

    return Step(_map)


def flat_map_fn(func: Callable[[Any], Any]) -> Step:
    """Step that maps each item with *func* and flattens the result.

    *func* should return an iterable (sync) or async iterable.
    """

    async def _flat_map(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        async for item in upstream:
            result = func(item)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, AsyncIterable):
                async for sub in result:
                    yield sub
            else:
                for sub in result:
                    yield sub

    return Step(_flat_map)


def batch(size: int) -> Step:
    """Step that groups items into batches (lists) of *size*.

    The last batch may be smaller than *size*.
    """
    if size <= 0:
        raise ValueError("size must be > 0")

    async def _batch(upstream: AsyncIterable[Any]) -> AsyncIterator[list[Any]]:
        buf: list[Any] = []
        async for item in upstream:
            buf.append(item)
            if len(buf) >= size:
                yield buf
                buf = []
        if buf:
            yield buf

    return Step(_batch)


def take(n: int) -> Step:
    """Step that yields at most the first *n* items then stops."""
    if n < 0:
        raise ValueError("n must be >= 0")

    async def _take(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        if n == 0:
            return

        count = 0
        iterator = upstream.__aiter__()
        while count < n:
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                break
            yield item
            count += 1

    return Step(_take)


def skip(n: int) -> Step:
    """Step that skips the first *n* items and yields the rest."""
    if n < 0:
        raise ValueError("n must be >= 0")

    async def _skip(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        count = 0
        async for item in upstream:
            if count < n:
                count += 1
                continue
            yield item

    return Step(_skip)


def tap(func: Callable[[Any], Any]) -> Step:
    """Step that calls *func* on each item for side effects but passes items through unchanged.

    Useful for logging, debugging, or collecting stats.
    """

    async def _tap(upstream: AsyncIterable[Any]) -> AsyncIterator[Any]:
        async for item in upstream:
            result = func(item)
            if inspect.isawaitable(result):
                await result
            yield item

    return Step(_tap)


# ---------------------------------------------------------------------------
# Terminal steps
# ---------------------------------------------------------------------------


def collect() -> Step:
    """Terminal step that gathers all results into a list."""

    async def _collect(upstream: AsyncIterable[Any]) -> list[Any]:
        return [item async for item in upstream]

    return Step(_collect, is_terminal=True)


def reduce_fn(func: Callable[[Any, Any], Any], initial: Any = None) -> Step:
    """Terminal step that reduces the stream to a single value.

    Args:
        func: A two-argument reducer ``(accumulator, item) -> accumulator``.
        initial: The initial accumulator value. If ``None``, the first item is used.
    """

    async def _reduce(upstream: AsyncIterable[Any]) -> Any:
        acc = initial
        first = True
        async for item in upstream:
            if first and acc is None:
                acc = item
                first = False
                continue
            first = False
            acc = func(acc, item)
            if inspect.isawaitable(acc):
                acc = await acc
        return acc

    return Step(_reduce, is_terminal=True)


# ---------------------------------------------------------------------------
# pipe() — the composer
# ---------------------------------------------------------------------------


def pipe(*steps: Step) -> Callable[[], Any]:
    """Create a pipeline from a sequence of steps.

    Returns an async callable that, when awaited, runs the pipeline and returns
    the final result (or the async iterable if no terminal step is used).

    Example::

        result = await pipe(source([1, 2, 3]), map_fn(str), collect())()
    """

    async def _run() -> Any:
        if not steps:
            return []
        if any(step.is_terminal for step in steps[:-1]):
            raise ValueError("Terminal steps must appear only at the end of a pipeline")

        # The first step receives an empty async iterable as upstream
        async def _empty() -> AsyncIterator[Any]:
            return
            yield  # pragma: no cover — make it an async generator

        current: Any = steps[0](_empty())

        for step in steps[1:]:
            current = step(current)

        # If the last step is terminal it returns a coroutine, not an async iterable
        if inspect.isawaitable(current):
            return await current
        else:
            # No terminal step — collect automatically
            return [item async for item in current]

    return _run
