"""
Comprehensive tests for silicon_refinery.debugging (enhanced_debug).

Covers:
  - Decorator wrapping for sync functions (name preservation)
  - Decorator wrapping for async functions (name preservation)
  - Normal execution returns value (sync + async)
  - Exception re-raised after analysis (sync + async)
  - _handle_exception: traceback printing to stderr
  - _handle_exception: stdout vs log routing
  - prompt_file generation
  - Model unavailability graceful degradation
  - FM analysis failure graceful degradation
"""

import concurrent.futures
import logging
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import MockDebuggingAnalysis, make_mock_model

# ========================================================================
# Decorator wrapping
# ========================================================================


class TestEnhancedDebugWrapping:
    def test_sync_function_name_preserved(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            def my_func():
                """My doc."""
                pass

            assert my_func.__name__ == "my_func"

    def test_async_function_name_preserved(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            async def my_async_func():
                """My doc."""
                pass

            assert my_async_func.__name__ == "my_async_func"

    def test_sync_function_docstring_preserved(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            def my_func():
                """My important docstring."""
                pass

            assert my_func.__doc__ == "My important docstring."

    def test_async_function_is_still_coroutine(self):
        import inspect

        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            async def my_async_func():
                pass

            assert inspect.iscoroutinefunction(my_async_func)


# ========================================================================
# Normal execution (no exception)
# ========================================================================


class TestEnhancedDebugNormalExecution:
    def test_sync_function_returns_normally(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            def add(a, b):
                """Add two numbers."""
                return a + b

            assert add(2, 3) == 5

    async def test_async_function_returns_normally(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            async def add(a, b):
                """Add two numbers."""
                return a + b

            assert await add(2, 3) == 5


# ========================================================================
# Exception re-raising
# ========================================================================


class TestEnhancedDebugExceptionReraised:
    def test_sync_exception_is_reraised(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            def bad_func():
                raise ValueError("sync boom")

            with pytest.raises(ValueError, match="sync boom"):
                bad_func()

    async def test_async_exception_is_reraised(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            async def bad_func():
                raise TypeError("async boom")

            with pytest.raises(TypeError, match="async boom"):
                await bad_func()

    def test_sync_exception_type_preserved(self):
        """The original exception type should be preserved, not wrapped."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            def bad_func():
                raise ZeroDivisionError("divide by zero")

            with pytest.raises(ZeroDivisionError):
                bad_func()


# ========================================================================
# _handle_exception outputs
# ========================================================================


class TestHandleException:
    async def test_prints_traceback_to_stderr(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis(
                error_summary="Test error",
                possible_causes=["cause1"],
                certainty_level="HIGH",
                suggested_fix="fix it",
            )
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("test error for traceback")
            except RuntimeError as e:
                await _handle_exception(e, "test_func", "stdout", None)

            captured = capsys.readouterr()
            assert "Exception caught in 'test_func'" in captured.err
            assert "test error for traceback" in captured.err

    async def test_stderr_contains_end_of_traceback_marker(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "End of standard traceback" in captured.err

    async def test_stdout_route_prints_analysis(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis(
                error_summary="Division by zero",
                possible_causes=["denominator is 0"],
                certainty_level="HIGH",
                suggested_fix="Check denominator",
            )
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise ZeroDivisionError("division by zero")
            except ZeroDivisionError as e:
                await _handle_exception(e, "divide", "stdout", None)

            captured = capsys.readouterr()
            assert "Division by zero" in captured.out
            assert "denominator is 0" in captured.out
            assert "Check denominator" in captured.out
            assert "HIGH" in captured.out

    async def test_log_route_uses_logger(self, caplog):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            with caplog.at_level(logging.ERROR, logger="silicon_refinery.debug"):
                try:
                    raise RuntimeError("log test")
                except RuntimeError as e:
                    await _handle_exception(e, "test_func", "log", None)

            assert any("SiliconRefinery AI Debug Analysis" in r.message for r in caplog.records)

    async def test_analysis_output_contains_all_causes(self, capsys):
        """All possible causes should appear in the output."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis(
                possible_causes=["cause_alpha", "cause_beta", "cause_gamma"]
            )
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("multi cause")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "cause_alpha" in captured.out
            assert "cause_beta" in captured.out
            assert "cause_gamma" in captured.out


# ========================================================================
# Prompt file generation
# ========================================================================


class TestPromptFileGeneration:
    async def test_prompt_file_written(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis(
                error_summary="NPE",
                possible_causes=["null ref"],
                certainty_level="MEDIUM",
                suggested_fix="Add null check",
            )
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                prompt_path = f.name

            try:
                try:
                    raise RuntimeError("prompt file test")
                except RuntimeError as e:
                    await _handle_exception(e, "func", "stdout", prompt_path)

                assert os.path.exists(prompt_path)
                with open(prompt_path) as f:
                    content = f.read()

                assert "prompt file test" in content
                assert "NPE" in content
                assert "null ref" in content
                assert "Add null check" in content
                assert "expert developer" in content
            finally:
                os.unlink(prompt_path)

    async def test_prompt_file_message_printed(self, capsys):
        """A confirmation message should be printed when prompt_file is written."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                prompt_path = f.name

            try:
                try:
                    raise RuntimeError("test")
                except RuntimeError as e:
                    await _handle_exception(e, "func", "stdout", prompt_path)

                captured = capsys.readouterr()
                assert "Generated AI Agent Prompt written to" in captured.out
            finally:
                os.unlink(prompt_path)

    async def test_no_prompt_file_when_none(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("no file test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "Generated AI Agent Prompt written to" not in captured.out


# ========================================================================
# Model unavailability graceful degradation
# ========================================================================


class TestEnhancedDebugModelUnavailable:
    async def test_model_unavailable_skips_analysis(self, capsys):
        mock_model = make_mock_model(available=False, reason="not downloaded")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession") as sess_cls,
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("unavailable model test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            # Session should never be created if model unavailable
            sess_cls.assert_not_called()
            captured = capsys.readouterr()
            # Should still print the original traceback
            assert "unavailable model test" in captured.err

    async def test_model_unavailable_no_analysis_in_stdout(self, capsys):
        """When model is unavailable, no analysis should appear in stdout."""
        mock_model = make_mock_model(available=False, reason="not installed")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession"),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "SiliconRefinery AI Debug Analysis" not in captured.out


# ========================================================================
# FM analysis failure
# ========================================================================


class TestEnhancedDebugAnalysisFailure:
    async def test_analysis_failure_prints_warning(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=RuntimeError("FM crashed during analysis"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise RuntimeError("original error")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "AI analysis failed" in captured.err
            # Original traceback should still be present
            assert "original error" in captured.err

    async def test_analysis_failure_does_not_crash(self, capsys):
        """Even if FM analysis fails, the function should complete gracefully."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=Exception("unexpected"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import _handle_exception

            try:
                raise ValueError("test")
            except ValueError as e:
                # Should not raise
                await _handle_exception(e, "func", "stdout", None)


# ========================================================================
# route_to parameter
# ========================================================================


class TestEnhancedDebugRouting:
    def test_route_to_stdout_via_decorator(self, capsys):
        """Integration test: decorator with route_to='stdout' prints analysis."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis(
                error_summary="Decorated error",
                certainty_level="MEDIUM",
            )
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug(route_to="stdout")
            def failing_func():
                raise RuntimeError("decorated failure")

            with pytest.raises(RuntimeError, match="decorated failure"):
                failing_func()

            captured = capsys.readouterr()
            # Analysis should be in stdout
            assert "Decorated error" in captured.out

    def test_prompt_file_via_decorator(self, capsys):
        """Integration test: decorator with prompt_file writes the file."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                prompt_path = f.name

            try:

                @enhanced_debug(route_to="stdout", prompt_file=prompt_path)
                def failing_func():
                    raise RuntimeError("file test")

                with pytest.raises(RuntimeError):
                    failing_func()

                assert os.path.exists(prompt_path)
                with open(prompt_path) as f:
                    content = f.read()
                assert "file test" in content
            finally:
                if os.path.exists(prompt_path):
                    os.unlink(prompt_path)


# ========================================================================
# Fuzz-scan edge-case tests: timeout handling in ThreadPoolExecutor
# ========================================================================


class TestEnhancedDebugTimeoutHandling:
    """Test that when the ThreadPoolExecutor times out (or the analysis fails
    inside the pool), the original exception is still raised -- not replaced
    by TimeoutError or any other internal error."""

    def test_threadpool_timeout_still_raises_original_exception(self):
        """When asyncio.run() fails (RuntimeError from running loop) and the
        ThreadPoolExecutor future times out, the original exception must still
        be re-raised."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        # Make respond hang so the ThreadPoolExecutor times out
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug(route_to="stdout")
            def bad_func():
                raise ValueError("original error must survive")

            # Force the asyncio.run path to fail with RuntimeError (simulating running loop)
            # and then force the ThreadPoolExecutor path to also time out
            with (
                patch("asyncio.run", side_effect=RuntimeError("already running loop")),
                patch("concurrent.futures.ThreadPoolExecutor") as mock_pool_cls,
            ):
                mock_pool = MagicMock()
                mock_pool_cls.return_value = mock_pool
                mock_future = MagicMock()
                mock_future.result.side_effect = concurrent.futures.TimeoutError("timed out")
                mock_pool.submit.return_value = mock_future

                # The ORIGINAL ValueError should be raised, not TimeoutError
                with pytest.raises(ValueError, match="original error must survive"):
                    bad_func()

                mock_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_threadpool_analysis_exception_still_raises_original(self):
        """When the ThreadPoolExecutor's analysis raises a generic exception,
        the original exception is still re-raised, not the analysis failure."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug(route_to="stdout")
            def bad_func():
                raise TypeError("type error is original")

            with (
                patch("asyncio.run", side_effect=RuntimeError("already running")),
                patch("concurrent.futures.ThreadPoolExecutor") as mock_pool_cls,
            ):
                mock_pool = MagicMock()
                mock_pool_cls.return_value = mock_pool
                mock_future = MagicMock()
                mock_future.result.side_effect = Exception("analysis crashed")
                mock_pool.submit.return_value = mock_future

                # The ORIGINAL TypeError should be raised
                with pytest.raises(TypeError, match="type error is original"):
                    bad_func()

                mock_pool.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_sync_wrapper_original_exception_preserved_when_analysis_succeeds(self, capsys):
        """Even when the analysis completes normally via ThreadPoolExecutor,
        the original exception must still be re-raised."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis(
                error_summary="ZeroDivision",
                certainty_level="HIGH",
            )
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug(route_to="stdout")
            def divide():
                return 1 / 0

            with pytest.raises(ZeroDivisionError):
                divide()

    def test_analysis_failure_never_masks_original_sync_exception(self):
        with patch(
            "silicon_refinery.debugging._handle_exception", side_effect=RuntimeError("boom")
        ):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug(route_to="stdout")
            def bad_func():
                raise ValueError("original")

            with pytest.raises(ValueError, match="original"):
                bad_func()
