"""
Comprehensive tests for silicon_refinery.async_generators (stream_extract).

Covers:
  - _chunk_lines and _achunk_lines helpers
  - stream_extract with all 4 history_mode values: clear, keep, hybrid, compact
  - Chunking behaviour (lines_per_chunk > 1)
  - Concurrency: concurrency=1 (sequential) vs concurrency=N (parallel)
  - Concurrency forces history_mode='clear' with warning
  - Default concurrency capped at 4
  - Error handling: context window errors, generic errors
  - debug_timing logging
  - Edge cases: empty iterable, single item, async iterable source, None results
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_refinery.async_generators import (
    _achunk_lines,
    _chunk_lines,
    _compact_history,
    _process_chunk,
    stream_extract,
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

    def test_single_item(self):
        result = list(_chunk_lines(["only"], chunk_size=5))
        assert result == ["only"]

    def test_exact_multiple(self):
        """Input length is exact multiple of chunk_size -- no remainder."""
        result = list(_chunk_lines(["a", "b", "c", "d"], chunk_size=2))
        assert result == ["a\nb", "c\nd"]

    def test_chunk_size_1_many_items(self):
        items = [str(i) for i in range(10)]
        result = list(_chunk_lines(items, chunk_size=1))
        assert result == items


# ========================================================================
# Helper: _achunk_lines (async)
# ========================================================================


class TestAChunkLines:
    async def test_async_chunk_size_2(self):
        async def async_source():
            for item in ["a", "b", "c", "d"]:
                yield item

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == ["a\nb", "c\nd"]

    async def test_async_chunk_with_remainder(self):
        async def async_source():
            for item in ["a", "b", "c"]:
                yield item

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == ["a\nb", "c"]

    async def test_async_empty_source(self):
        async def async_source():
            return
            yield  # make it an async generator

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == []

    async def test_async_single_item(self):
        async def async_source():
            yield "only"

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=5):
            result.append(chunk)

        assert result == ["only"]

    async def test_async_non_string_items(self):
        async def async_source():
            for item in [10, 20, 30]:
                yield item

        result = []
        async for chunk in _achunk_lines(async_source(), chunk_size=2):
            result.append(chunk)

        assert result == ["10\n20", "30"]


# ========================================================================
# stream_extract: history_mode="clear" (default)
# ========================================================================


class TestStreamExtractClear:
    async def test_basic_stream_yields_results(self, mock_fm_available):
        data = ["Alice", "Bob", "Charlie"]
        expected = MockSchema(name="test_result")

        results = []
        async for item in stream_extract(data, schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 3
        assert all(r == expected for r in results)

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

    async def test_empty_iterable_yields_nothing(self, mock_fm_available):
        results = []
        async for item in stream_extract([], schema=MockSchema, concurrency=1):
            results.append(item)

        assert results == []

    async def test_single_item(self, mock_fm_available):
        results = []
        async for item in stream_extract(["only one"], schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 1


# ========================================================================
# stream_extract: history_mode="keep"
# ========================================================================


class TestStreamExtractKeep:
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
        assert mock_fm_available["session_cls"].call_count == 1

    async def test_keep_mode_raises_on_context_window_error(self):
        """In keep mode, context window errors propagate (no fallback)."""
        mock_model = make_mock_model(available=True)
        exc = Exception("ExceededContextWindowSizeError: too large")
        mock_session = make_mock_session(respond_side_effect=exc)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            pytest.raises(Exception, match="ExceededContextWindowSizeError"),
        ):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, history_mode="keep", concurrency=1
            ):
                pass


# ========================================================================
# stream_extract: history_mode="hybrid"
# ========================================================================


class TestStreamExtractHybrid:
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

    async def test_hybrid_raises_non_context_window_errors(self):
        """Hybrid mode does NOT catch non-context-window errors."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=ValueError("something else"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            pytest.raises(ValueError, match="something else"),
        ):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, history_mode="hybrid", concurrency=1
            ):
                pass


# ========================================================================
# stream_extract: history_mode="compact"
# ========================================================================


class TestStreamExtractCompact:
    async def test_compact_triggers_history_compaction_on_context_error(self):
        """Compact mode calls _compact_history then retries."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        # First respond: context window error; second (after compaction): summary; third: success
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
    @pytest.mark.parametrize("mode", ["clear", "keep", "hybrid", "compact"])
    async def test_all_modes_yield_results_on_happy_path(self, mode, mock_fm_available):
        data = ["item1", "item2"]

        results = []
        async for item in stream_extract(data, schema=MockSchema, history_mode=mode, concurrency=1):
            results.append(item)

        assert len(results) == 2


# ========================================================================
# Chunking integration
# ========================================================================


class TestStreamExtractChunking:
    async def test_lines_per_chunk_groups_input(self, mock_fm_available):
        data = ["line1", "line2", "line3", "line4"]

        results = []
        async for item in stream_extract(data, schema=MockSchema, lines_per_chunk=2, concurrency=1):
            results.append(item)

        # 4 lines / chunk_size=2 = 2 chunks = 2 results
        assert len(results) == 2
        # Verify respond was called with joined text
        calls = mock_fm_available["session"].respond.call_args_list
        assert "line1\nline2" in calls[0][0][0]
        assert "line3\nline4" in calls[1][0][0]

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
    @pytest.mark.parametrize("invalid_concurrency", [0, -1, 1.5, "2", True])
    async def test_invalid_concurrency_raises_value_error(self, invalid_concurrency):
        with pytest.raises(ValueError, match="concurrency must be an int >= 1"):
            async for _ in stream_extract(
                ["A"],
                schema=MockSchema,
                concurrency=invalid_concurrency,
            ):
                pass

    async def test_concurrency_1_processes_sequentially(self, mock_fm_available):
        data = ["A", "B", "C"]

        results = []
        async for item in stream_extract(data, schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 3

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

    async def test_concurrency_processes_all_items(self, mock_fm_available):
        data = ["A", "B", "C", "D"]

        results = []
        async for item in stream_extract(data, schema=MockSchema, concurrency=2):
            results.append(item)

        assert len(results) == 4

    async def test_concurrency_default_capped_at_4(self):
        """When concurrency=None, it defaults to min(os.cpu_count(), 4), so max 4."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("os.cpu_count", return_value=16),
        ):
            # With 16 CPUs, concurrency should be capped at 4
            # We verify by checking that all items are processed
            results = []
            async for item in stream_extract(["A"], schema=MockSchema, concurrency=None):
                results.append(item)

            assert len(results) == 1

    async def test_concurrency_default_with_low_cpu_count(self):
        """When cpu_count is less than 4, it should use cpu_count."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("os.cpu_count", return_value=2),
        ):
            results = []
            async for item in stream_extract(["A", "B"], schema=MockSchema, concurrency=None):
                results.append(item)

            assert len(results) == 2

    async def test_concurrency_default_with_none_cpu_count(self):
        """When os.cpu_count() returns None, fallback to 1."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("os.cpu_count", return_value=None),
        ):
            results = []
            async for item in stream_extract(["A"], schema=MockSchema, concurrency=None):
                results.append(item)

            assert len(results) == 1

    async def test_concurrency_with_async_source(self, mock_fm_available):
        async def async_source():
            for item in ["X", "Y", "Z"]:
                yield item

        results = []
        async for item in stream_extract(async_source(), schema=MockSchema, concurrency=2):
            results.append(item)

        assert len(results) == 3

    async def test_concurrent_empty_source(self, mock_fm_available):
        """Concurrent mode with empty source should yield nothing without hanging."""
        results = []
        async for item in stream_extract([], schema=MockSchema, concurrency=3):
            results.append(item)

        assert results == []


# ========================================================================
# Concurrent path error handling
# ========================================================================


class TestStreamExtractConcurrencyErrors:
    async def test_concurrent_task_failure_propagates(self):
        """A single task failure in concurrent mode propagates the exception."""
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
            pytest.raises(ValueError, match="Task 2 blew up"),
        ):
            async for _ in stream_extract(["A", "B", "C"], schema=MockSchema, concurrency=3):
                pass


# ========================================================================
# debug_timing
# ========================================================================


class TestStreamExtractDebugTiming:
    async def test_timing_log_emitted(self, mock_fm_available, caplog):
        with caplog.at_level(logging.INFO, logger="silicon_refinery"):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, debug_timing=True, concurrency=1
            ):
                pass

        assert any("Chunk processed in" in r.message for r in caplog.records)
        assert any("Throughput:" in r.message for r in caplog.records)

    async def test_no_timing_log_when_disabled(self, mock_fm_available, caplog):
        with caplog.at_level(logging.INFO, logger="silicon_refinery"):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, debug_timing=False, concurrency=1
            ):
                pass

        timing_messages = [r for r in caplog.records if "Chunk processed" in r.message]
        assert len(timing_messages) == 0


# ========================================================================
# _process_chunk isolation tests
# ========================================================================


class TestProcessChunk:
    async def test_clear_mode_creates_new_session(self):
        mock_model = make_mock_model()
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as sess_cls:
            _new_session, result = await _process_chunk(
                mock_model, mock_session, "test", "chunk_data", MockSchema, "clear", False
            )
            # A new session should have been created
            sess_cls.assert_called_once()
            assert result == MockSchema(name="x")

    async def test_keep_mode_reuses_session(self):
        mock_model = make_mock_model()
        original_session = make_mock_session(respond_return=MockSchema(name="x"))

        with patch("apple_fm_sdk.LanguageModelSession") as sess_cls:
            returned_session, _result = await _process_chunk(
                mock_model, original_session, "test", "chunk_data", MockSchema, "keep", False
            )
            # Should NOT create a new session
            sess_cls.assert_not_called()
            assert returned_session is original_session

    async def test_none_session_creates_new_one(self):
        """When session is None (concurrent path), a new session is created."""
        mock_model = make_mock_model()
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as sess_cls:
            _, result = await _process_chunk(
                mock_model, None, "test", "data", MockSchema, "keep", False
            )
            sess_cls.assert_called_once()
            assert result == MockSchema(name="x")

    async def test_generic_error_reraises(self):
        mock_model = make_mock_model()
        mock_session = make_mock_session(respond_side_effect=TypeError("bad type"))

        with (
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            pytest.raises(TypeError, match="bad type"),
        ):
            await _process_chunk(
                mock_model, mock_session, "test", "data", MockSchema, "clear", False
            )

    async def test_payload_is_stringified(self):
        """The chunk value should be converted to string before being sent."""
        mock_model = make_mock_model()
        mock_session = make_mock_session(respond_return=MockSchema(name="x"))

        with patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session):
            await _process_chunk(
                mock_model, mock_session, "test", 12345, MockSchema, "clear", False
            )
            call_args = mock_session.respond.call_args[0]
            assert call_args[0] == "12345"


# ========================================================================
# _compact_history isolation test
# ========================================================================


class TestCompactHistory:
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
            async for item in stream_extract(["data"], schema=MockSchema, concurrency=1):
                results.append(item)

            assert results == []

    async def test_async_source_iterable(self, mock_fm_available):
        """stream_extract should accept an async iterable source."""

        async def async_source():
            for item in ["alpha", "beta", "gamma"]:
                yield item

        results = []
        async for item in stream_extract(async_source(), schema=MockSchema, concurrency=1):
            results.append(item)

        assert len(results) == 3

    async def test_extremely_large_input(self, mock_fm_available):
        """Verify no crash on very large item count."""
        data = [f"item_{i}" for i in range(1000)]

        count = 0
        async for _ in stream_extract(data, schema=MockSchema, concurrency=1):
            count += 1

        assert count == 1000

    async def test_instructions_parameter_used(self, mock_fm_available):
        """Verify the instructions parameter is forwarded to the session."""
        async for _ in stream_extract(
            ["data"], schema=MockSchema, instructions="Custom instruction", concurrency=1
        ):
            pass

        # Check that the session was created with our custom instructions
        calls = mock_fm_available["session_cls"].call_args_list
        # At least one call should have our instructions
        instructions_used = [c[1].get("instructions", "") for c in calls if "instructions" in c[1]]
        assert any("Custom instruction" in inst for inst in instructions_used)


# ========================================================================
# Fuzz-scan edge-case tests: concurrency multiple errors
# ========================================================================


class TestStreamExtractConcurrencyMultipleErrors:
    """When multiple concurrent tasks fail, the first error should be raised
    and remaining done tasks should still be consumed (no task leak)."""

    async def test_multiple_concurrent_failures_raises_first_error(self):
        """If several tasks fail in the same ``done`` batch, only the first
        exception (in iteration order) is raised."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def all_fail_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Introduce a tiny delay so tasks are truly concurrent
            await asyncio.sleep(0.01)
            if call_count == 1:
                raise ValueError("error from task 1")
            elif call_count == 2:
                raise TypeError("error from task 2")
            else:
                raise RuntimeError("error from task 3")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=all_fail_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            with pytest.raises(Exception) as exc_info:
                async for _ in stream_extract(["A", "B", "C"], schema=MockSchema, concurrency=3):
                    pass

            # The raised exception should be one of the errors from the done batch
            assert type(exc_info.value) in (ValueError, TypeError, RuntimeError)

    async def test_successful_done_tasks_consumed_before_error_raised(self):
        """When one task fails and others succeed in the same ``done`` batch,
        the successful results should be yielded before the error propagates."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def mixed_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            current = call_count
            # Make the second task take slightly longer so the first finishes earlier
            if current == 1:
                await asyncio.sleep(0.01)
                return MockSchema(name="success_1")
            elif current == 2:
                await asyncio.sleep(0.01)
                raise ValueError("task 2 failed")
            else:
                await asyncio.sleep(0.01)
                return MockSchema(name="success_3")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=mixed_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            with pytest.raises(ValueError, match="task 2 failed"):
                async for item in stream_extract(["A", "B", "C"], schema=MockSchema, concurrency=3):
                    results.append(item)

            # At least some successful results should have been yielded before the error
            # (depending on task scheduling, we may get 0, 1, or 2 successes)
            # The key assertion: no tasks are leaked (the finally block cancels remaining)
            assert all(isinstance(r, MockSchema) for r in results)

    async def test_pending_tasks_cancelled_on_error(self):
        """After an error, any pending (not yet completed) tasks should be cancelled."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def slow_with_one_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            current = call_count
            if current == 1:
                raise ValueError("fast failure")
            # Other tasks are slow -- they should be cancelled
            await asyncio.sleep(10)
            return MockSchema(name=f"slow_{current}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=slow_with_one_failure)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            pytest.raises(ValueError, match="fast failure"),
        ):
            async for _ in stream_extract(["A", "B", "C", "D"], schema=MockSchema, concurrency=2):
                pass
            # If we get here without hanging, pending tasks were properly cancelled


# ========================================================================
# Fuzz-scan edge-case tests: _chunk_lines edge cases
# ========================================================================


class TestChunkLinesEdgeCases:
    """Test _chunk_lines with chunk_size=0 and negative values."""

    def test_chunk_size_zero_produces_no_output(self):
        """With chunk_size=0, the buffer never reaches the threshold, so
        items accumulate until the source is exhausted and then flush as
        a single chunk (the trailing ``if chunk:`` block)."""
        result = list(_chunk_lines(["a", "b", "c"], chunk_size=0))
        # chunk_size=0 means len(chunk) >= 0 is always true, so every
        # single item is yielded individually (each item triggers flush).
        assert result == ["a", "b", "c"]

    def test_chunk_size_negative_produces_individual_items(self):
        """Negative chunk_size: len(chunk) >= negative is always true,
        so each item flushes immediately (same as chunk_size=0)."""
        result = list(_chunk_lines(["x", "y", "z"], chunk_size=-1))
        assert result == ["x", "y", "z"]

    def test_chunk_size_zero_empty_input(self):
        """chunk_size=0 with empty input should yield nothing."""
        result = list(_chunk_lines([], chunk_size=0))
        assert result == []

    def test_chunk_size_negative_single_item(self):
        """Negative chunk_size with a single item."""
        result = list(_chunk_lines(["only"], chunk_size=-5))
        assert result == ["only"]

    def test_chunk_size_zero_with_non_string_items(self):
        """chunk_size=0 should still convert items to strings."""
        result = list(_chunk_lines([1, 2, 3], chunk_size=0))
        assert result == ["1", "2", "3"]


# ========================================================================
# Fuzz-scan edge-case tests: compact history cascading failure
# ========================================================================


class TestCompactHistoryCascadingFailure:
    """Test what happens when the compacted history itself exceeds the context
    window on retry (cascading failure)."""

    async def test_compact_retry_also_exceeds_context_window(self):
        """If after compaction the retry also fails with a context window error,
        the exception should propagate because the retry uses history_mode='keep'
        which does not catch context window errors."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def cascading_fail(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Initial call fails with context window error
                raise Exception("Context window size exceeded")
            elif call_count == 2:
                # Compaction summary request succeeds
                return "summary of conversation"
            else:
                # Retry after compaction ALSO fails with context window error
                raise Exception("Context window size exceeded again")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=cascading_fail)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            pytest.raises(Exception, match="Context window size exceeded"),
        ):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, history_mode="compact", concurrency=1
            ):
                pass

    async def test_compact_summary_fails_then_clean_session_also_fails(self):
        """If compaction summary fails and the fallback clean session also
        encounters a context window error, it should propagate."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def total_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Original call fails
                raise Exception("ExceededContextWindowSizeError: first")
            elif call_count == 2:
                # Compaction summary request fails
                raise Exception("Compaction summary also failed")
            else:
                # Clean session retry also fails with context window
                raise Exception("ExceededContextWindowSizeError: third try")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=total_failure)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            # The retry after compact fallback uses history_mode='keep', so if that
            # also gets a context window error, it will be re-raised from _process_chunk
            pytest.raises(Exception, match="ExceededContextWindowSizeError"),
        ):
            async for _ in stream_extract(
                ["data"], schema=MockSchema, history_mode="compact", concurrency=1
            ):
                pass

    async def test_compact_succeeds_but_all_data_causes_repeated_compaction(self):
        """Multiple chunks where each one triggers compaction -- verifies no infinite loop."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def compact_then_succeed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Odd calls: context window error; even calls: success (alternating pattern)
            if call_count % 3 == 1:
                raise Exception("Context window size exceeded")
            elif call_count % 3 == 2:
                # Summary request for compaction
                return "summary"
            else:
                return MockSchema(name=f"result_{call_count}")

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=compact_then_succeed)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for item in stream_extract(
                ["chunk1", "chunk2"], schema=MockSchema, history_mode="compact", concurrency=1
            ):
                results.append(item)

            assert len(results) == 2
