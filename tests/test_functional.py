"""
Tests for silicon_refinery.functional — Functional Pipeline API.

All tests mock apple_fm_sdk via the conftest.py module-level mock.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_refinery.functional import (
    Step,
    batch,
    collect,
    extract,
    filter_fn,
    flat_map_fn,
    map_fn,
    pipe,
    reduce_fn,
    skip,
    source,
    take,
    tap,
)

from .conftest import MockSchema

# ---------------------------------------------------------------------------
# Basic pipe + collect
# ---------------------------------------------------------------------------


class TestPipeCollect:
    async def test_pipe_collect(self):
        result = await pipe(source([1, 2, 3]), collect())()
        assert result == [1, 2, 3]

    async def test_pipe_empty_input(self):
        result = await pipe(source([]), collect())()
        assert result == []

    async def test_pipe_no_terminal_auto_collects(self):
        """Without a terminal step the pipeline auto-collects into a list."""
        result = await pipe(source([1, 2, 3]))()
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# map_fn
# ---------------------------------------------------------------------------


class TestMapFn:
    async def test_pipe_simple_map(self):
        result = await pipe(
            source([1, 2, 3]),
            map_fn(lambda x: x * 2),
            collect(),
        )()
        assert result == [2, 4, 6]

    async def test_pipe_map_with_strings(self):
        result = await pipe(
            source(["hello", "world"]),
            map_fn(str.upper),
            collect(),
        )()
        assert result == ["HELLO", "WORLD"]

    async def test_pipe_async_map(self):
        async def double(x):
            return x * 2

        result = await pipe(source([1, 2, 3]), map_fn(double), collect())()
        assert result == [2, 4, 6]

    async def test_pipe_async_callable_object_map(self):
        class AsyncDoubler:
            async def __call__(self, x):
                return x * 2

        result = await pipe(source([1, 2, 3]), map_fn(AsyncDoubler()), collect())()
        assert result == [2, 4, 6]


# ---------------------------------------------------------------------------
# filter_fn
# ---------------------------------------------------------------------------


class TestFilterFn:
    async def test_pipe_filter(self):
        result = await pipe(
            source([1, 2, 3, 4, 5]),
            filter_fn(lambda x: x > 3),
            collect(),
        )()
        assert result == [4, 5]

    async def test_pipe_filter_none_pass(self):
        result = await pipe(
            source([1, 2, 3]),
            filter_fn(lambda x: x > 100),
            collect(),
        )()
        assert result == []

    async def test_pipe_async_predicate(self):
        async def is_even(x):
            return x % 2 == 0

        result = await pipe(
            source([1, 2, 3, 4, 5, 6]),
            filter_fn(is_even),
            collect(),
        )()
        assert result == [2, 4, 6]

    async def test_pipe_async_callable_object_predicate(self):
        class AsyncPredicate:
            async def __call__(self, x):
                return x > 2

        result = await pipe(source([1, 2, 3, 4]), filter_fn(AsyncPredicate()), collect())()
        assert result == [3, 4]


# ---------------------------------------------------------------------------
# flat_map_fn
# ---------------------------------------------------------------------------


class TestFlatMapFn:
    async def test_pipe_flat_map(self):
        result = await pipe(
            source(["hello world", "foo bar"]),
            flat_map_fn(lambda s: s.split()),
            collect(),
        )()
        assert result == ["hello", "world", "foo", "bar"]

    async def test_pipe_flat_map_empty_inner(self):
        result = await pipe(
            source(["a", ""]),
            flat_map_fn(lambda s: s.split()),
            collect(),
        )()
        assert result == ["a"]


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


class TestBatch:
    async def test_pipe_batch(self):
        result = await pipe(
            source([1, 2, 3, 4, 5]),
            batch(2),
            collect(),
        )()
        assert result == [[1, 2], [3, 4], [5]]

    async def test_pipe_batch_exact_multiple(self):
        result = await pipe(
            source([1, 2, 3, 4]),
            batch(2),
            collect(),
        )()
        assert result == [[1, 2], [3, 4]]

    async def test_pipe_batch_larger_than_input(self):
        result = await pipe(
            source([1]),
            batch(10),
            collect(),
        )()
        assert result == [[1]]


# ---------------------------------------------------------------------------
# take / skip
# ---------------------------------------------------------------------------


class TestTakeSkip:
    async def test_pipe_take(self):
        result = await pipe(
            source([1, 2, 3, 4, 5]),
            take(3),
            collect(),
        )()
        assert result == [1, 2, 3]

    async def test_pipe_take_more_than_available(self):
        result = await pipe(
            source([1, 2]),
            take(10),
            collect(),
        )()
        assert result == [1, 2]

    async def test_pipe_skip(self):
        result = await pipe(
            source([1, 2, 3, 4, 5]),
            skip(2),
            collect(),
        )()
        assert result == [3, 4, 5]

    async def test_pipe_skip_all(self):
        result = await pipe(
            source([1, 2, 3]),
            skip(10),
            collect(),
        )()
        assert result == []

    async def test_take_zero_does_not_consume_upstream(self):
        pulled = 0

        async def upstream():
            nonlocal pulled
            for item in [1, 2, 3]:
                pulled += 1
                yield item

        result = await pipe(source(upstream()), take(0), collect())()
        assert result == []
        assert pulled == 0


# ---------------------------------------------------------------------------
# reduce_fn
# ---------------------------------------------------------------------------


class TestReduceFn:
    async def test_pipe_reduce(self):
        result = await pipe(
            source([1, 2, 3, 4]),
            reduce_fn(lambda acc, x: acc + x, initial=0),
        )()
        assert result == 10

    async def test_pipe_reduce_no_initial(self):
        """When initial is None, the first item is used as the accumulator."""
        result = await pipe(
            source([10, 20, 30]),
            reduce_fn(lambda acc, x: acc + x),
        )()
        assert result == 60

    async def test_pipe_reduce_with_strings(self):
        result = await pipe(
            source(["a", "b", "c"]),
            reduce_fn(lambda acc, x: acc + x, initial=""),
        )()
        assert result == "abc"

    async def test_pipe_reduce_async_callable_object(self):
        class AsyncReducer:
            async def __call__(self, acc, x):
                return acc + x

        result = await pipe(
            source([1, 2, 3]),
            reduce_fn(AsyncReducer(), initial=0),
        )()
        assert result == 6


# ---------------------------------------------------------------------------
# tap
# ---------------------------------------------------------------------------


class TestTap:
    async def test_pipe_tap_side_effects(self):
        seen: list = []
        result = await pipe(
            source([1, 2, 3]),
            tap(lambda x: seen.append(x)),
            collect(),
        )()
        assert result == [1, 2, 3]
        assert seen == [1, 2, 3]

    async def test_pipe_tap_does_not_modify(self):
        """tap should not change items even if the function returns something."""
        result = await pipe(
            source(["a", "b"]),
            tap(lambda x: x.upper()),
            collect(),
        )()
        assert result == ["a", "b"]

    async def test_pipe_tap_async_callable_object(self):
        seen: list[int] = []

        class AsyncTapper:
            async def __call__(self, value: int) -> None:
                seen.append(value)

        result = await pipe(source([1, 2, 3]), tap(AsyncTapper()), collect())()
        assert result == [1, 2, 3]
        assert seen == [1, 2, 3]


# ---------------------------------------------------------------------------
# extract (with mock FM)
# ---------------------------------------------------------------------------


class TestExtractStep:
    async def test_pipe_extract_with_mock_fm(self):
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            side_effect=lambda text, generating: MockSchema(name=text.strip())
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            result = await pipe(
                source(["Alice", "Bob"]),
                extract(MockSchema, instructions="Extract person."),
                collect(),
            )()

        assert len(result) == 2
        assert result[0].name == "Alice"
        assert result[1].name == "Bob"

    async def test_pipe_extract_skip_on_error(self):
        mock_model = MagicMock()

        call_count = 0

        async def flaky_respond(text, generating):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("FM error")
            return MockSchema(name=text.strip())

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            result = await pipe(
                source(["A", "B", "C"]),
                extract(MockSchema, on_error="skip"),
                collect(),
            )()

        # Second item fails and is skipped
        assert len(result) == 2

    def test_extract_invalid_on_error_raises(self):
        with pytest.raises(ValueError, match="Invalid on_error"):
            extract(MockSchema, on_error="invalid")


# ---------------------------------------------------------------------------
# Composition with | operator
# ---------------------------------------------------------------------------


class TestComposition:
    async def test_pipe_composition_with_or_operator(self):
        step1 = source([1, 2, 3, 4, 5])
        step2 = filter_fn(lambda x: x % 2 == 0)
        step3 = map_fn(lambda x: x * 10)
        terminal = collect()

        composed = step1 | step2 | step3 | terminal

        # The composed step is terminal, so calling it returns an awaitable
        async def _empty():
            return
            yield  # pragma: no cover

        result = await composed(_empty())
        assert result == [20, 40]

    async def test_or_operator_non_terminal(self):
        """Composing non-terminal steps produces a non-terminal step."""
        step = map_fn(lambda x: x + 1) | filter_fn(lambda x: x > 2)
        assert isinstance(step, Step)
        assert step.is_terminal is False

    async def test_or_operator_with_terminal(self):
        """Composing with a terminal step produces a terminal step."""
        step = map_fn(lambda x: x + 1) | collect()
        assert isinstance(step, Step)
        assert step.is_terminal is True

    def test_or_operator_rejects_steps_after_terminal(self):
        with pytest.raises(
            ValueError, match="Cannot compose additional steps after a terminal step"
        ):
            _ = (map_fn(lambda x: x + 1) | collect()) | map_fn(lambda x: x * 2)


# ---------------------------------------------------------------------------
# Chained transforms
# ---------------------------------------------------------------------------


class TestChainedTransforms:
    async def test_map_filter_take(self):
        result = await pipe(
            source(range(20)),
            map_fn(lambda x: x * 3),
            filter_fn(lambda x: x % 2 == 0),
            take(5),
            collect(),
        )()
        assert result == [0, 6, 12, 18, 24]

    async def test_skip_then_batch(self):
        result = await pipe(
            source(range(10)),
            skip(3),
            batch(3),
            collect(),
        )()
        assert result == [[3, 4, 5], [6, 7, 8], [9]]


class TestValidation:
    def test_batch_requires_positive_size(self):
        with pytest.raises(ValueError, match="size must be > 0"):
            batch(0)

    def test_take_requires_non_negative(self):
        with pytest.raises(ValueError, match="n must be >= 0"):
            take(-1)

    def test_skip_requires_non_negative(self):
        with pytest.raises(ValueError, match="n must be >= 0"):
            skip(-1)

    async def test_pipe_rejects_terminal_step_in_the_middle(self):
        with pytest.raises(ValueError, match="Terminal steps must appear only at the end"):
            await pipe(source([1, 2]), collect(), map_fn(lambda x: x + 1))()
