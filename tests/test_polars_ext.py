"""
Comprehensive tests for fmtools.polars_ext (LocalLLMExpr).

Covers:
  - Polars namespace registration
  - .local_llm.extract() produces a String column with JSON
  - Correct FM session.respond calls per row
  - None values in input series
  - Custom instructions forwarded to session
  - Edge cases: empty series, single row, all None, result without __dict__
"""

import concurrent.futures
import json
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

# We must import polars_ext to trigger the @register_expr_namespace side effect
import fmtools.polars_ext as polars_ext

from .conftest import MockSchema, make_mock_model

# ========================================================================
# Helper to create a mock schema result with __dict__
# ========================================================================


class MockExtractResult:
    """Mimics an fm.generable() result that has __dict__."""

    def __init__(self, name, age):
        self.name = name
        self.age = age


# ========================================================================
# Namespace registration
# ========================================================================


class TestLocalLLMNamespace:
    def test_namespace_registered(self):
        """Verify that pl.col("x").local_llm is accessible."""
        expr = pl.col("text").local_llm.extract(MockSchema)
        # If this doesn't raise AttributeError, the namespace is registered
        assert expr is not None


class TestLocalLLMInputValidation:
    @pytest.mark.parametrize("invalid_concurrency", [0, -1, 1.5, "2", True])
    def test_extract_rejects_invalid_concurrency(self, invalid_concurrency):
        with pytest.raises(ValueError, match="concurrency must be an integer >= 1"):
            pl.col("text").local_llm.extract(MockSchema, concurrency=invalid_concurrency)


class TestLocalLLMTimeoutHandling:
    def test_extract_timeout_cancels_future_and_raises_timeout_error(self):
        class TimeoutFuture:
            def __init__(self):
                self.cancel_called = False
                self.timeout_arg = None

            def result(self, timeout=None):
                self.timeout_arg = timeout
                raise concurrent.futures.TimeoutError("simulated timeout")

            def cancel(self):
                self.cancel_called = True
                return True

        future = TimeoutFuture()

        def fake_run_coroutine_threadsafe(coro, loop):
            coro.close()
            return future

        with (
            patch("fmtools.polars_ext.create_model", return_value=MagicMock()),
            patch(
                "fmtools.polars_ext.asyncio.run_coroutine_threadsafe",
                side_effect=fake_run_coroutine_threadsafe,
            ),
        ):
            df = pl.DataFrame({"text": ["row1"]})
            with pytest.raises(TimeoutError, match="timed out waiting for batch results"):
                df.select(pl.col("text").local_llm.extract(MockSchema))

        assert future.cancel_called is True
        assert future.timeout_arg == polars_ext._FUTURE_RESULT_TIMEOUT_SECONDS

    def test_extract_wraps_future_result_errors(self):
        class FailingFuture:
            def result(self, timeout=None):
                raise RuntimeError("boom")

        future = FailingFuture()

        def fake_run_coroutine_threadsafe(coro, loop):
            coro.close()
            return future

        with (
            patch("fmtools.polars_ext.create_model", return_value=MagicMock()),
            patch(
                "fmtools.polars_ext.asyncio.run_coroutine_threadsafe",
                side_effect=fake_run_coroutine_threadsafe,
            ),
        ):
            df = pl.DataFrame({"text": ["row1"]})
            with pytest.raises(
                RuntimeError,
                match=r"LocalLLMExpr.extract failed while waiting for batch results",
            ):
                df.select(pl.col("text").local_llm.extract(MockSchema))


# ========================================================================
# Happy path extraction
# ========================================================================


class TestLocalLLMExtract:
    def test_extract_produces_json_strings(self):
        mock_model = make_mock_model(available=True)

        results = [
            MockExtractResult("Alice", 30),
            MockExtractResult("Bob", 25),
        ]
        call_idx = 0

        async def mock_respond(*args, **kwargs):
            nonlocal call_idx
            result = results[call_idx]
            call_idx += 1
            return result

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=mock_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["Alice is 30", "Bob is 25"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (2, 1)
            assert result_df.dtypes[0] == pl.String

            # Verify JSON content
            row0 = json.loads(result_df["text"][0])
            assert row0["name"] == "Alice"
            assert row0["age"] == 30

    def test_extract_with_custom_instructions(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("X", 1))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as sess_cls,
        ):
            df = pl.DataFrame({"text": ["data"]})
            df.select(
                pl.col("text").local_llm.extract(MockSchema, instructions="Custom instruction here")
            )

            # At least one session creation should have our custom instructions
            found = False
            for call in sess_cls.call_args_list:
                if call[1].get("instructions") == "Custom instruction here":
                    found = True
                    break
            assert found, "Custom instructions were not passed to LanguageModelSession"

    def test_extract_calls_respond_per_row(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("X", 1))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["a", "b", "c"]})
            df.select(pl.col("text").local_llm.extract(MockSchema))

            assert mock_session.respond.call_count == 3


# ========================================================================
# None handling
# ========================================================================


class TestLocalLLMNoneHandling:
    def test_none_values_produce_null_output(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("X", 1))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["Alice", None, "Bob"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            # None input should produce None output
            assert result_df["text"][1] is None
            # Non-None inputs should have JSON
            assert result_df["text"][0] is not None
            assert result_df["text"][2] is not None

            # respond should only be called for non-None values
            assert mock_session.respond.call_count == 2

    def test_all_none_series(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("X", 1))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": [None, None]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert all(v is None for v in result_df["text"])
            mock_session.respond.assert_not_called()


# ========================================================================
# Edge cases
# ========================================================================


class TestLocalLLMEdgeCases:
    def test_empty_dataframe(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("X", 1))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": []}).cast({"text": pl.String})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (0, 1)
            mock_session.respond.assert_not_called()

    def test_single_row(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("Only", 1))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["single"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (1, 1)
            row = json.loads(result_df["text"][0])
            assert row["name"] == "Only"

    def test_result_without_dict_uses_str_fallback(self):
        """If vars(res) raises TypeError, the code falls back to str(res)."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        # Return a plain string -- vars() on a string raises TypeError
        mock_session.respond = AsyncMock(return_value="plain string result")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["data"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            # Should produce a result without crashing
            assert result_df.shape == (1, 1)
            # The fallback path: {"_raw": str(res)}
            row = json.loads(result_df["text"][0])
            assert "_raw" in row
            assert row["_raw"] == "plain string result"

    def test_json_output_is_valid_json(self):
        """Every non-null cell should contain valid JSON."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockExtractResult("Test", 99))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["a", "b", "c"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            for val in result_df["text"]:
                parsed = json.loads(val)
                assert isinstance(parsed, dict)


# ========================================================================
# Fuzz-scan edge-case tests: single row failure fault tolerance
# ========================================================================


class TestLocalLLMExtractErrorHandling:
    """Test that a single row failure doesn't crash the entire batch.
    The source code has a try/except in process_one that returns None for
    failed rows, so the batch should continue processing."""

    def test_single_row_failure_returns_none_for_that_row(self):
        """When one row's respond() raises, that row should get None
        while other rows succeed."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def flaky_respond(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("FM crashed on row 2")
            return MockExtractResult("Name", call_count)

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=flaky_respond)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["row1", "row2", "row3"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (3, 1)

            # Row 0 and 2 should have valid JSON
            assert result_df["text"][0] is not None
            row0 = json.loads(result_df["text"][0])
            assert row0["name"] == "Name"

            # Row 1 (second call) should be None due to the failure
            assert result_df["text"][1] is None

            # Row 2 should have valid JSON
            assert result_df["text"][2] is not None
            row2 = json.loads(result_df["text"][2])
            assert row2["name"] == "Name"

    def test_all_rows_fail_returns_all_none(self):
        """When every row fails, the result should be all None without crashing."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=RuntimeError("all fail"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["a", "b", "c"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (3, 1)
            assert all(v is None for v in result_df["text"])

    def test_first_row_fails_rest_succeed(self):
        """Failure of the first row should not prevent subsequent rows."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def first_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("first row error")
            return MockExtractResult("OK", call_count)

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=first_fails)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["bad", "good1", "good2"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (3, 1)
            # First row should be None
            assert result_df["text"][0] is None
            # Second and third rows should have valid JSON
            assert result_df["text"][1] is not None
            assert result_df["text"][2] is not None

    def test_last_row_fails_preceding_rows_succeed(self):
        """Failure of the last row should not affect preceding rows."""
        mock_model = make_mock_model(available=True)

        call_count = 0

        async def last_fails(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("last row error")
            return MockExtractResult("OK", call_count)

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=last_fails)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            df = pl.DataFrame({"text": ["good1", "good2", "bad"]})
            result_df = df.select(pl.col("text").local_llm.extract(MockSchema))

            assert result_df.shape == (3, 1)
            assert result_df["text"][0] is not None
            assert result_df["text"][1] is not None
            # Last row should be None
            assert result_df["text"][2] is None
