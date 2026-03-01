"""
Comprehensive tests for fmtools.debugging (enhanced_debug).

Covers:
  - Decorator wrapping for sync functions (name preservation)
  - Decorator wrapping for async functions (name preservation)
  - Normal execution returns value (sync + async)
  - Exception re-raised after analysis (sync + async)
  - _handle_exception: traceback printing to stderr
  - _handle_exception: stdout/stderr/None routing
  - prompt_to routing and file generation
  - silenced mode
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
            from fmtools.debugging import enhanced_debug

            @enhanced_debug()
            def my_func():
                """My doc."""
                pass

            assert my_func.__name__ == "my_func"

    def test_async_function_name_preserved(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from fmtools.debugging import enhanced_debug

            @enhanced_debug()
            async def my_async_func():
                """My doc."""
                pass

            assert my_async_func.__name__ == "my_async_func"

    def test_sync_function_docstring_preserved(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from fmtools.debugging import enhanced_debug

            @enhanced_debug()
            def my_func():
                """My important docstring."""
                pass

            assert my_func.__doc__ == "My important docstring."

    def test_async_function_is_still_coroutine(self):
        import inspect

        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from fmtools.debugging import enhanced_debug

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
            from fmtools.debugging import enhanced_debug

            @enhanced_debug()
            def add(a, b):
                """Add two numbers."""
                return a + b

            assert add(2, 3) == 5

    async def test_async_function_returns_normally(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from fmtools.debugging import enhanced_debug

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
            from fmtools.debugging import enhanced_debug

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
            from fmtools.debugging import enhanced_debug

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
            from fmtools.debugging import enhanced_debug

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
            from fmtools.debugging import _handle_exception

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
            from fmtools.debugging import _handle_exception

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
            from fmtools.debugging import _handle_exception

            try:
                raise ZeroDivisionError("division by zero")
            except ZeroDivisionError as e:
                await _handle_exception(e, "divide", "stdout", None)

            captured = capsys.readouterr()
            assert "Division by zero" in captured.out
            assert "denominator is 0" in captured.out
            assert "Check denominator" in captured.out
            assert "HIGH" in captured.out

    async def test_stderr_route_prints_analysis(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("stderr test")
            except RuntimeError as e:
                await _handle_exception(e, "test_func", "stderr", None)

            captured = capsys.readouterr()
            assert "FMTools AI Debug Analysis" in captured.err

    async def test_log_route_uses_selected_level(self, caplog):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            with caplog.at_level(logging.DEBUG, logger="fmtools.debug"):
                try:
                    raise RuntimeError("log level test")
                except RuntimeError as e:
                    await _handle_exception(
                        e,
                        "test_func",
                        summary_to="log",
                        prompt_to=None,
                        summary_log_level="warning",
                    )

            matching = [
                r
                for r in caplog.records
                if "FMTools AI Debug Analysis" in r.message and r.levelno == logging.WARNING
            ]
            assert matching

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
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("multi cause")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "cause_alpha" in captured.out
            assert "cause_beta" in captured.out
            assert "cause_gamma" in captured.out


# ========================================================================
# Prompt routing generation
# ========================================================================


class TestPromptFileGeneration:
    async def test_prompt_to_written(self, capsys):
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
            from fmtools.debugging import _handle_exception

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
                assert "Debugging objective for the next agent" in content
            finally:
                os.unlink(prompt_path)

    async def test_prompt_to_message_printed(self, capsys):
        """A confirmation message should be printed when prompt_to is written."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

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

    async def test_no_prompt_to_when_none(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("no file test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "Generated AI Agent Prompt written to" not in captured.out

    async def test_prompt_to_defaults_to_stdout(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("default stdout prompt")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout")

            captured = capsys.readouterr()
            assert "I encountered a crash in my Python application." in captured.out

    async def test_prompt_to_path_without_suffix_writes_txt(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            with tempfile.TemporaryDirectory() as tmpdir:
                prompt_path = os.path.join(tmpdir, "llm_crash_report")
                expected_path = f"{prompt_path}.txt"

                try:
                    raise RuntimeError("suffix test")
                except RuntimeError as e:
                    await _handle_exception(e, "func", "stdout", prompt_path)

                assert os.path.exists(expected_path)
                with open(expected_path) as f:
                    content = f.read()
                assert "suffix test" in content

    async def test_prompt_to_log_uses_selected_level(self, caplog):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            with caplog.at_level(logging.DEBUG, logger="fmtools.debug"):
                try:
                    raise RuntimeError("prompt log test")
                except RuntimeError as e:
                    await _handle_exception(
                        e,
                        "func",
                        summary_to="stderr",
                        prompt_to="log",
                        prompt_log_level="debug",
                    )

            matching = [
                r
                for r in caplog.records
                if "I encountered a crash in my Python application." in r.message
                and r.levelno == logging.DEBUG
            ]
            assert matching


# ========================================================================
# Context overflow retries / query shaping
# ========================================================================


class TestContextOverflowRetries:
    async def test_retries_with_smaller_payload_on_context_overflow(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            side_effect=[
                RuntimeError("ExceededContextWindowSizeError: context window exceeded"),
                MockDebuggingAnalysis(),
            ]
        )
        fake_tb = (
            ["Traceback (most recent call last):\n"]
            + [f'  File "/tmp/app.py", line {i}, in fn_{i}\n' for i in range(200)]
            + ["RuntimeError: context overflow test\n"]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("fmtools.debugging.traceback.format_exception", return_value=fake_tb),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("context overflow test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "Context window overflow detected" in captured.err
            assert mock_session.respond.await_count == 2
            first_query = mock_session.respond.await_args_list[0].args[0]
            second_query = mock_session.respond.await_args_list[1].args[0]
            assert len(second_query) < len(first_query)

    async def test_retries_on_sdk_context_overflow_exception_type(self, capsys):
        class FakeExceededContextWindowSizeError(Exception):
            pass

        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            side_effect=[
                FakeExceededContextWindowSizeError("Context window size exceeded"),
                MockDebuggingAnalysis(),
            ]
        )
        fake_tb = (
            ["Traceback (most recent call last):\n"]
            + [f'  File "/tmp/app.py", line {i}, in fn_{i}\n' for i in range(160)]
            + ["RuntimeError: overflow with typed exception\n"]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch(
                "fmtools.debugging.fm.ExceededContextWindowSizeError",
                FakeExceededContextWindowSizeError,
            ),
            patch("fmtools.debugging.traceback.format_exception", return_value=fake_tb),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("overflow with typed exception")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "Context window overflow detected" in captured.err
            assert mock_session.respond.await_count == 2

    async def test_query_payload_respects_char_budget(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())
        huge_tb = (
            ["Traceback (most recent call last):\n"]
            + [f'  File "/tmp/huge.py", line {i}, in fn_{i}\n' for i in range(5000)]
            + ["RuntimeError: huge traceback\n"]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("fmtools.debugging.traceback.format_exception", return_value=huge_tb),
        ):
            from fmtools.debugging import _DEBUG_QUERY_MAX_CHARS, _handle_exception

            try:
                raise RuntimeError("huge traceback")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            query = mock_session.respond.await_args_list[0].args[0]
            assert len(query) <= _DEBUG_QUERY_MAX_CHARS
            assert "Traceback truncated by character budget" in query

    async def test_small_traceback_does_not_retry_identical_payloads(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            side_effect=RuntimeError("ExceededContextWindowSizeError: tiny traceback")
        )
        small_tb = [
            "Traceback (most recent call last):\n",
            '  File "/tmp/app.py", line 1, in fn\n',
            "RuntimeError: tiny traceback\n",
        ]

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("fmtools.debugging.traceback.format_exception", return_value=small_tb),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("tiny traceback")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "AI analysis failed" in captured.err
            assert mock_session.respond.await_count == 1


# ========================================================================
# Two-stage pipeline (forensic triage + handoff)
# ========================================================================


class TestTwoStageDebugPipeline:
    async def test_prompt_mode_runs_second_handoff_query(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("two stage test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", "stdout")

            captured = capsys.readouterr()
            assert mock_session.respond.await_count == 2
            first_query = mock_session.respond.await_args_list[0].args[0]
            second_query = mock_session.respond.await_args_list[1].args[0]
            assert "Local source snippets around failing frames" in first_query
            assert "Expected output:" in second_query
            assert "candidate_edits" in second_query
            assert "I encountered a crash in my Python application." in captured.out

    async def test_handoff_retry_on_context_overflow(self, capsys):
        class FakeHandoff:
            def __init__(self):
                self.objective = "Fix root cause and validate."
                self.prioritized_steps = ["Inspect frame F1", "Patch type normalization"]
                self.candidate_edits = ["app.py:10-18 | convert value to int"]
                self.instrumentation = ["app.py:10 | log incoming payload type"]
                self.verification_commands = ["uv run pytest tests/test_debugging.py -q"]
                self.risks = ["May affect callers that rely on implicit coercion"]
                self.handoff_prompt = "Use evidence-backed edits only."

        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            side_effect=[
                MockDebuggingAnalysis(),
                RuntimeError("Context window size exceeded"),
                FakeHandoff(),
            ]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("handoff overflow")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", "stdout")

            captured = capsys.readouterr()
            assert "Context window overflow during handoff generation" in captured.err
            assert "Use evidence-backed edits only." in captured.out
            assert mock_session.respond.await_count == 3


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
            from fmtools.debugging import _handle_exception

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
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("test")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", None)

            captured = capsys.readouterr()
            assert "FMTools AI Debug Analysis" not in captured.out


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
            from fmtools.debugging import _handle_exception

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
            from fmtools.debugging import _handle_exception

            try:
                raise ValueError("test")
            except ValueError as e:
                # Should not raise
                await _handle_exception(e, "func", "stdout", None)


# ========================================================================
# summary_to parameter
# ========================================================================


class TestEnhancedDebugRouting:
    def test_summary_to_stdout_via_decorator(self, capsys):
        """Integration test: decorator with summary_to='stdout' prints analysis."""
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
            from fmtools.debugging import enhanced_debug

            @enhanced_debug(summary_to="stdout")
            def failing_func():
                raise RuntimeError("decorated failure")

            with pytest.raises(RuntimeError, match="decorated failure"):
                failing_func()

            captured = capsys.readouterr()
            # Analysis should be in stdout
            assert "Decorated error" in captured.out

    def test_prompt_to_via_decorator(self, capsys):
        """Integration test: decorator with prompt_to writes the file."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import enhanced_debug

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                prompt_path = f.name

            try:

                @enhanced_debug(summary_to="stdout", prompt_to=prompt_path)
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

    async def test_summary_to_none_silences_analysis(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=MockDebuggingAnalysis())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("route none test")
            except RuntimeError as e:
                await _handle_exception(e, "test_func", None, None)

            captured = capsys.readouterr()
            assert "FMTools AI Debug Analysis" not in captured.out


class TestEnhancedDebugSilenced:
    async def test_silenced_true_skips_all_output_and_analysis(self, capsys):
        mock_model = make_mock_model(available=True)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession") as sess_cls,
        ):
            from fmtools.debugging import _handle_exception

            try:
                raise RuntimeError("fully silenced")
            except RuntimeError as e:
                await _handle_exception(e, "func", "stdout", "stdout", True)

            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == ""
            sess_cls.assert_not_called()


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
            from fmtools.debugging import enhanced_debug

            @enhanced_debug(summary_to="stdout")
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
            from fmtools.debugging import enhanced_debug

            @enhanced_debug(summary_to="stdout")
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
            from fmtools.debugging import enhanced_debug

            @enhanced_debug(summary_to="stdout")
            def divide():
                return 1 / 0

            with pytest.raises(ZeroDivisionError):
                divide()

    def test_analysis_failure_never_masks_original_sync_exception(self):
        with patch("fmtools.debugging._handle_exception", side_effect=RuntimeError("boom")):
            from fmtools.debugging import enhanced_debug

            @enhanced_debug(summary_to="stdout")
            def bad_func():
                raise ValueError("original")

            with pytest.raises(ValueError, match="original"):
                bad_func()
