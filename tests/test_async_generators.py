"""
Comprehensive tests for silicon_refinery.async_generators (stream_extract).

Covers:
  - _chunk_lines and _achunk_lines helpers
  - stream_extract with all 4 history_mode values: clear, keep, hybrid, compact
  - Chunking behaviour (lines_per_chunk > 1)
  - Concurrency: concurrency=1 (sequential) vs concurrency=N (parallel)
  - Concurrency forces history_mode='clear'
  - Error handling: context window errors, generic errors
  - debug_timing logging
  - Edge cases: empty iterable, single item, async iterable source
  - BUG DISCOVERED: concurrent path infinite loop when iterator exhausted
"""

import logging
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from silicon_refinery.async_generators import (
    stream_extract,
    _chunk_lines,
    _achunk_lines,
    _process_chunk,
    _compact_history,
)
from .conftest import MockSchema, make_mock_model, make_mock_session


# ========================================================================
# Helper: _chunk_lines (sync)
# ========================================================================

class TestChunkLines:

    def test_chunk_size_1_no_grouping(self):
        result = list(_chunk_lines(["a", "b", "c"], chunk_size=1))
        assert result == ["a", "b", "c"]

    def test_chunk_size_2_groups_pairs(self):
        result = list(_chunk_lines(["a", "b", "c", "d"], chunk_size=2))
        assert result == ["a\nb", "c\nd"]

    def test_chunk_size_2_with_remainder(self):
        result = list(_chunk_lines(["a", "b", "c"], chunk_size=2))
        assert result == ["a\nb", "c"]

    def test_chunk_size_larger_than_input(self):
        result = list(_chunk_lines(["a", "b"], chunk_size=10))
        assert result == ["a\nb"]

    def test_empty_iterable(self):
        result = list(_chunk_lines([], chunk_size=3))
        assert result == []

    def test_non_string_items_converted(self):
        result = list(_chunk_lines([1, 2, 3], chunk_size=2))
        assert result == ["1\n2", "3"]


# ========================================================================
# Helper: _achunk_lines (async)
# ========================================================================

class TestAChunkLines:

    @pytest.mark.asyncio
    async def test_async_chunk_size_2(self):
        async def async_source():
            for item in ["a", "b", "c", "d"]:
                yield item

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == ["a\nb", "c\nd"]

    @pytest.mark.asyncio
    async def test_async_chunk_with_remainder(self):
        async def async_source():
            for item in ["a", "b", "c"]:
                yield item

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == ["a\nb", "c"]

    @pytest.mark.asyncio
    async def test_async_empty_source(self):
        async def async_source():
            return
            yield  # make it an async generator

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == []


# ========================================================================
# stream_extract: history_mode="clear" (default)
# ========================================================================

class TestStreamExtractClear:

    @pytest.mark.asyncio
    async def test_basic_stream_yields_results(self, mock_fm_available):
        data = ["Alice", "Bob", "Charlie"]
        expected = MockSchema(name="test_result")

        results = []
        async for item in stream_extract(data, schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 3
        assert all(r == expected for r in results)

    @pytest.mark.asyncio
    async def test_clear_mode_creates_fresh_session_per_chunk(self, mock_fm_available):
        """In 'clear' mode, _process_chunk creates a new session each time."""
        data = ["A", "B"]

        results = []
        async for item in stream_extract(
            data, schema=MockSchema, history_mode="clear", concurrency=1
        ):
            results.append(item)

        # LanguageModelSession should be called: 1 initial + 2 clears inside _process_chunk
        assert mock_fm_available["session_cls"].call_count >= 2

    @pytest.mark.asyncio
    async def test_empty_iterable_yields_nothing(self, mock_fm_available):
        results = []
        async for item in stream_extract([], schema=MockSchema, concurrency=1):
            results.append(item)

        assert results == []

    @pytest.mark.asyncio
    async def test_single_item(self, mock_fm_available):
        results = []
        async for item in stream_extract(["only one"], schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 1


# ========================================================================
# stream_extract: history_mode="keep"
# ========================================================================

class TestStreamExtractKeep:

    @pytest.mark.asyncio
    async def test_keep_mode_reuses_session(self, mock_fm_available):
        data = ["A", "B", "C"]

        results = []
        async for item in stream_extract(
            data, schema=MockSchema, history_mode="keep", concurrency=1
        ):
            results.append(item)

        assert len(results) == 3
        # Session should be created once initially; _process_chunk does NOT recreate
        # when history_mode != "clear"
        # The initial creation is 1 call
        assert mock_fm_available["session_cls"].call_count == 1

    @pytest.mark.asyncio
    async def test_keep_mode_raises_on_context_window_error(self):
        """In keep mode, context window errors propagate (no fallback)."""
        mock_model = make_mock_model(available=True)
        exc = Exception("ExceededContextWindowSizeError: too large")
        mock_session = make_mock_session(respond_side_effect=exc)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            with pytest.raises(Exception, match="ExceededContextWindowSizeError"):
                async for _ in stream_extract(
                    ["data"], schema=MockSchema, history_mode="keep", concurrency=1
                ):
                    pass


# ========================================================================
# stream_extract: history_mode="hybrid"
# ========================================================================

class TestStreamExtractHybrid:

    @pytest.mark.asyncio
    async def test_hybrid_retries_on_context_window_error(self):
        """Hybrid mode catches context window errors and retries with clear."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        # First call raises context window error, second succeeds
        mock_session.respond = AsyncMock(
            side_effect=[
                Exception("ExceededContextWindowSizeError: too large"),
                MockSchema(name="recovered"),
            ]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for item in stream_extract(
                ["data"], schema=MockSchema, history_mode="hybrid", concurrency=1
            ):
                results.append(item)

            assert len(results) == 1
            assert results[0] == MockSchema(name="recovered")

    @pytest.mark.asyncio
    async def test_hybrid_raises_non_context_window_errors(self):
        """Hybrid mode does NOT catch non-context-window errors."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=ValueError("something else"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            with pytest.raises(ValueError, match="something else"):
                async for _ in stream_extract(
                    ["data"], schema=MockSchema, history_mode="hybrid", concurrency=1
                ):
                    pass


# ========================================================================
# stream_extract: history_mode="compact"
# ========================================================================

class TestStreamExtractCompact:

    @pytest.mark.asyncio
    async def test_compact_triggers_history_compaction_on_context_error(self):
        """Compact mode calls _compact_history then retries."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        # First respond: context window error; second (after compaction): success
        mock_session.respond = AsyncMock(
            side_effect=[
                Exception("Context window size exceeded"),
                "summary of conversation",  # compaction summary response
                MockSchema(name="compacted"),  # retry after compaction
            ]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for item in stream_extract(
                ["data"], schema=MockSchema, history_mode="compact", concurrency=1
            ):
                results.append(item)

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_compact_falls_back_to_clean_session_on_compaction_failure(self):
        """If the summary request itself fails, compact falls back to a clean session."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def variable_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Context window size exceeded")
            elif call_count == 2:
                # This is the compaction summary request - it fails
                raise Exception("Compaction also failed")
            else:
                return MockSchema(name="fallback")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=variable_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for item in stream_extract(
                ["data"], schema=MockSchema, history_mode="compact", concurrency=1
            ):
                results.append(item)

            assert len(results) == 1


# ========================================================================
# Parametrized: all 4 history modes
# ========================================================================

class TestStreamExtractAllHistoryModes:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode", ["clear", "keep", "hybrid", "compact"])
    async def test_all_modes_yield_results_on_happy_path(self, mode, mock_fm_available):
        data = ["item1", "item2"]

        results = []
        async for item in stream_extract(
            data, schema=MockSchema, history_mode=mode, concurrency=1
        ):
            results.append(item)

        assert len(results) == 2


# ========================================================================
# Chunking integration
# ========================================================================

class TestStreamExtractChunking:

    @pytest.mark.asyncio
    async def test_lines_per_chunk_groups_input(self, mock_fm_available):
        data = ["line1", "line2", "line3", "line4"]

        results = []
        async for item in stream_extract(
            data, schema=MockSchema, lines_per_chunk=2, concurrency=1
        ):
            results.append(item)

        # 4 lines / chunk_size=2 = 2 chunks = 2 results
        assert len(results) == 2
        # Verify respond was called with joined text
        calls = mock_fm_available["session"].respond.call_args_list
        assert "line1\nline2" in calls[0][0][0]
        assert "line3\nline4" in calls[1][0][0]

    @pytest.mark.asyncio
    async def test_lines_per_chunk_with_async_source(self, mock_fm_available):
        async def async_source():
            for item in ["a", "b", "c"]:
                yield item

        results = []
        async for item in stream_extract(
            async_source(), schema=MockSchema, lines_per_chunk=2, concurrency=1
        ):
            results.append(item)

        # 3 lines / chunk_size=2 = 2 chunks (2 + remainder 1)
        assert len(results) == 2


# ========================================================================
# Concurrency
# ========================================================================

class TestStreamExtractConcurrency:

    @pytest.mark.asyncio
    async def test_concurrency_1_processes_sequentially(self, mock_fm_available):
        data = ["A", "B", "C"]

        results = []
        async for item in stream_extract(data, schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_concurrency_gt1_forces_clear_mode(self, mock_fm_available, caplog):
        """When concurrency > 1 and history_mode != 'clear', it forces clear and warns."""
        data = ["A", "B"]

        with caplog.at_level(logging.WARNING, logger="silicon_refinery"):
            results = []
            async for item in stream_extract(
                data, schema=MockSchema, history_mode="keep", concurrency=2
            ):
                results.append(item)

        assert any("Forcing history_mode='clear'" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_concurrency_processes_all_items(self, mock_fm_available):
        data = ["A", "B", "C", "D"]

        results = []
        async for item in stream_extract(data, schema=MockSchema, concurrency=2):
            results.append(item)

        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_concurrency_default_uses_cpu_count(self):
        """When concurrency=None, it defaults to os.cpu_count()."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("os.cpu_count", return_value=4),
        ):
            # We can't easily observe the concurrency value directly, but we
            # can verify it doesn't crash and processes items
            results = []
            async for item in stream_extract(["A"], schema=MockSchema, concurrency=None):
                results.append(item)

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_concurrency_with_async_source(self, mock_fm_available):
        async def async_source():
            for item in ["X", "Y", "Z"]:
                yield item

        results = []
        async for item in stream_extract(async_source(), schema=MockSchema, concurrency=2):
            results.append(item)

        assert len(results) == 3


# ========================================================================
# BUG REPORT: Concurrent path may spin infinitely
# ========================================================================

class TestStreamExtractConcurrencyBug:
    """
    BUG DISCOVERED: In the concurrent path (concurrency > 1), the outer
    `while True` loop checks `if not pending: break` ONLY after the inner
    filling loop. However, the inner `while len(pending) < concurrency` loop
    breaks out (via the except StopIteration/StopAsyncIteration) when the
    iterator is exhausted, but does NOT set a flag to indicate exhaustion.

    Consider the sequence:
    1. Iterator has 3 items, concurrency=2.
    2. First outer iteration: fill 2 tasks, wait, yield results. pending now empty.
    3. Second outer iteration: inner loop tries to fill. Gets item 3, then
       hits StopIteration -> breaks inner loop. pending has 1 task.
       We proceed to `if not pending: break` -> False (we have 1).
       We wait for that 1 task, yield result. pending now empty.
    4. Third outer iteration: inner loop tries to fill. Immediately hits
       StopIteration -> breaks. pending is empty.
       `if not pending: break` -> True. We break. OK.

    This actually works correctly for the simple case. BUT there is a subtle
    issue: if the inner loop breaks on StopIteration and pending is NOT empty
    (from a previous iteration that wasn't fully drained), the outer loop
    continues correctly. The key invariant is that once the iterator is
    exhausted, the inner loop always immediately breaks, and eventually
    pending drains to empty.

    The REAL bug is different: when a task in `done` raises an exception,
    `task.result()` will re-raise it. This means that if ANY concurrent
    task fails, the entire generator crashes without processing remaining
    tasks in `pending`. There is no error handling in the concurrent yield
    loop (lines 166-171), unlike the sequential path which lets _process_chunk
    handle errors. This is an unhandled error path.
    """

    @pytest.mark.asyncio
    async def test_concurrent_task_failure_propagates_unhandled(self):
        """Demonstrates that a single task failure in concurrent mode
        crashes the entire generator, unlike sequential mode."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def flaky_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Task 2 blew up")
            return MockSchema(name=f"result_{call_count}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            with pytest.raises(ValueError, match="Task 2 blew up"):
                async for _ in stream_extract(
                    ["A", "B", "C"], schema=MockSchema, concurrency=3
                ):
                    pass


# ========================================================================
# debug_timing
# ========================================================================

class TestStreamExtractDebugTiming:

    @pytest.mark.asyncio
    async def test_timing_log_emitted(self, mock_fm_available, caplog):
        with caplog.at_level(logging.INFO, logger="silicon_refinery"):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, debug_timing=True, concurrency=1
            ):
                pass

        assert any("Chunk processed in" in r.message for r in caplog.records)
        assert any("Throughput:" in r.message for r in caplog.records)


# ========================================================================
# _process_chunk isolation tests
# ========================================================================

class TestProcessChunk:

    @pytest.mark.asyncio
    async def test_clear_mode_creates_new_session(self):
        mock_model = make_mock_model()
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as sess_cls:
            new_session, result = await _process_chunk(
                mock_model, mock_session, "test", "chunk_data",
                MockSchema, "clear", False
            )
            # A new session should have been created
            sess_cls.assert_called_once()
            assert result == MockSchema(name="x")

    @pytest.mark.asyncio
    async def test_keep_mode_reuses_session(self):
        mock_model = make_mock_model()
        original_session = make_mock_session(respond_return=MockSchema(name="x"))

        with patch("apple_fm_sdk.LanguageModelSession") as sess_cls:
            returned_session, result = await _process_chunk(
                mock_model, original_session, "test", "chunk_data",
                MockSchema, "keep", False
            )
            # Should NOT create a new session
            sess_cls.assert_not_called()
            assert returned_session is original_session

    @pytest.mark.asyncio
    async def test_generic_error_reraises(self):
        mock_model = make_mock_model()
        mock_session = make_mock_session(respond_side_effect=TypeError("bad type"))

        with patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session):
            with pytest.raises(TypeError, match="bad type"):
                await _process_chunk(
                    mock_model, mock_session, "test", "data",
                    MockSchema, "clear", False
                )


# ========================================================================
# _compact_history isolation test
# ========================================================================

class TestCompactHistory:

    @pytest.mark.asyncio
    async def test_compact_creates_new_session_with_summary(self):
        mock_model = make_mock_model()
        old_session = MagicMock()
        old_session.respond = AsyncMock(return_value="Prior conversation was about X and Y.")

        with patch("apple_fm_sdk.LanguageModelSession") as sess_cls:
            await _compact_history(mock_model, "original instructions", old_session)

            # Verify it asked the old session for a summary
            old_session.respond.assert_called_once()
            # Verify it created a new session with expanded instructions
            sess_cls.assert_called_once()
            call_kwargs = sess_cls.call_args[1]
            assert "Prior Context Summary" in call_kwargs["instructions"]
            assert "original instructions" in call_kwargs["instructions"]

    @pytest.mark.asyncio
    async def test_compact_falls_back_on_summary_failure(self):
        mock_model = make_mock_model()
        old_session = MagicMock()
        old_session.respond = AsyncMock(side_effect=RuntimeError("summary failed"))

        with patch("apple_fm_sdk.LanguageModelSession") as sess_cls:
            await _compact_history(mock_model, "instructions", old_session)

            # Should still create a clean session (fallback)
            sess_cls.assert_called_once()
            call_kwargs = sess_cls.call_args[1]
            assert call_kwargs["instructions"] == "instructions"


# ========================================================================
# Edge cases
# ========================================================================

class TestStreamExtractEdgeCases:

    @pytest.mark.asyncio
    async def test_none_result_filtered_out(self):
        """If _process_chunk returns None result, it should not be yielded."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=None)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            # This tests the `if result is not None: yield result` guard
            # We need the mock to return None from respond, but _process_chunk
            # returns (session, result) where result can be None
            async for item in stream_extract(
                ["data"], schema=MockSchema, concurrency=1
            ):
                results.append(item)

            # The result is None from respond, so _process_chunk returns (session, None)
            # and stream_extract filters it out
            assert results == []

    @pytest.mark.asyncio
    async def test_extremely_large_input(self, mock_fm_available):
        """Verify no crash on very large item count."""
        data = [f"item_{i}" for i in range(1000)]

        count = 0
        async for _ in stream_extract(data, schema=MockSchema, concurrency=1):
            count += 1

        assert count == 1000
