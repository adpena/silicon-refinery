"""
Comprehensive tests for silicon_refinery.debugging (enhanced_debug).

Covers:
  - Decorator wrapping for sync functions
  - Decorator wrapping for async functions
  - Exception re-raised after analysis
  - _handle_exception: traceback printing, FM analysis, stdout vs log routing
  - prompt_file generation
  - Model unavailability graceful degradation
  - FM analysis failure graceful degradation
  - Edge cases: no exception, nested exceptions
"""

import os
import tempfile
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

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

    def test_sync_function_returns_normally(self):
        mock_model = make_mock_model(available=True)
        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.debugging import enhanced_debug

            @enhanced_debug()
            def add(a, b):
                """Add two numbers."""
                return a + b

            assert add(2, 3) == 5

    @pytest.mark.asyncio
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
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis()
        )

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

    @pytest.mark.asyncio
    async def test_async_exception_is_reraised(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis()
        )

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


# ========================================================================
# _handle_exception outputs
# ========================================================================

class TestHandleException:

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_log_route_uses_logger(self, caplog):
        import logging

        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis()
        )

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


# ========================================================================
# Prompt file generation
# ========================================================================

class TestPromptFileGeneration:

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_no_prompt_file_when_none(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            return_value=MockDebuggingAnalysis()
        )

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

    @pytest.mark.asyncio
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


# ========================================================================
# FM analysis failure
# ========================================================================

class TestEnhancedDebugAnalysisFailure:

    @pytest.mark.asyncio
    async def test_analysis_failure_prints_warning(self, capsys):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(
            side_effect=RuntimeError("FM crashed during analysis")
        )

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
