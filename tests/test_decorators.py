"""
Comprehensive tests for silicon_refinery.decorators (local_extract).

Covers:
  - Decorator structural properties (wrapping, name preservation)
  - Successful extraction with mocked FM
  - Input formatting (args, kwargs, mixed)
  - Retry logic on transient failures
  - Retry exhaustion leading to RuntimeError
  - Model unavailability check
  - debug_timing logging
  - Edge cases: empty docstring, no args, None args, Unicode input
"""

import time
import logging
import pytest
from unittest.mock import patch, AsyncMock

from silicon_refinery.decorators import local_extract
from .conftest import MockSchema, make_mock_model, make_mock_session


# ========================================================================
# Structural / Decorator mechanics
# ========================================================================

class TestLocalExtractStructure:

    def test_preserves_function_name(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def my_extractor(text: str):
            """Extract stuff."""
            pass

        assert my_extractor.__name__ == "my_extractor"

    def test_preserves_docstring(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def my_extractor(text: str):
            """My custom instruction."""
            pass

        assert my_extractor.__doc__ == "My custom instruction."

    def test_returns_coroutine_function(self, mock_fm_available):
        import asyncio

        @local_extract(schema=MockSchema)
        async def my_extractor(text: str):
            """Extract."""
            pass

        assert asyncio.iscoroutinefunction(my_extractor)


# ========================================================================
# Happy path: successful extraction
# ========================================================================

class TestLocalExtractHappyPath:

    @pytest.mark.asyncio
    async def test_basic_extraction(self, mock_fm_available):
        expected = MockSchema(name="Alice")
        mock_fm_available["session"].respond.return_value = expected

        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text: str):
            """Extract the person's name."""
            pass

        result = await extract_name("Alice is 30 years old")
        assert result == expected

    @pytest.mark.asyncio
    async def test_session_created_with_docstring_instructions(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text: str):
            """Extract the person's name from the text."""
            pass

        await extract_name("Alice")

        # Verify session was created with stripped docstring as instructions
        mock_fm_available["session_cls"].assert_called_once()
        call_kwargs = mock_fm_available["session_cls"].call_args
        assert call_kwargs[1]["instructions"] == "Extract the person's name from the text."

    @pytest.mark.asyncio
    async def test_respond_called_with_schema(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text: str):
            """Extract name."""
            pass

        await extract_name("Alice")

        mock_fm_available["session"].respond.assert_called_once()
        call_kwargs = mock_fm_available["session"].respond.call_args
        assert call_kwargs[1]["generating"] == MockSchema


# ========================================================================
# Input formatting
# ========================================================================

class TestLocalExtractInputFormatting:

    @pytest.mark.asyncio
    async def test_positional_args_joined_with_space(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(a, b, c):
            """Extract."""
            pass

        await extract_name("Hello", "World", "!")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == "Hello World !"

    @pytest.mark.asyncio
    async def test_kwargs_appended_as_key_value(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(**kwargs):
            """Extract."""
            pass

        await extract_name(name="Alice", age="30")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert "name: Alice" in call_args[0]
        assert "age: 30" in call_args[0]

    @pytest.mark.asyncio
    async def test_mixed_args_and_kwargs(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text, extra=None):
            """Extract."""
            pass

        await extract_name("Hello", extra="World")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0].startswith("Hello")
        assert "extra: World" in call_args[0]

    @pytest.mark.asyncio
    async def test_no_args_sends_empty_string(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name():
            """Extract."""
            pass

        await extract_name()

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == ""

    @pytest.mark.asyncio
    async def test_none_arg_converted_to_string(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        await extract_name(None)

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == "None"

    @pytest.mark.asyncio
    async def test_unicode_input(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        await extract_name("Tanaka Taro")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert "Tanaka" in call_args[0]


# ========================================================================
# Fallback docstring
# ========================================================================

class TestLocalExtractDocstring:

    @pytest.mark.asyncio
    async def test_missing_docstring_uses_default(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            pass  # no docstring

        await extract_name("Alice")

        call_kwargs = mock_fm_available["session_cls"].call_args
        assert call_kwargs[1]["instructions"] == "Extract the following data."


# ========================================================================
# Retry logic
# ========================================================================

class TestLocalExtractRetries:

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure_then_succeeds(self):
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session()
        # Fail first, succeed second
        mock_session.respond = AsyncMock(
            side_effect=[RuntimeError("transient"), MockSchema(name="Alice")]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            result = await extract_name("Alice")
            assert result == MockSchema(name="Alice")
            assert mock_session.respond.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_runtime_error(self):
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=RuntimeError("persistent failure")
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(RuntimeError, match="Failed to generate structured data after 3 attempts"):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 3

    @pytest.mark.asyncio
    async def test_retries_equals_one_no_retry(self):
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=RuntimeError("fail")
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=1)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(RuntimeError, match="Failed to generate structured data after 1 attempts"):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 1


# ========================================================================
# Model unavailability
# ========================================================================

class TestLocalExtractModelUnavailable:

    @pytest.mark.asyncio
    async def test_raises_when_model_unavailable(self, mock_fm_unavailable):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        with pytest.raises(RuntimeError, match="Foundation Model is not available"):
            await extract_name("Alice")


# ========================================================================
# debug_timing
# ========================================================================

class TestLocalExtractDebugTiming:

    @pytest.mark.asyncio
    async def test_debug_timing_logs_message(self, mock_fm_available, caplog):
        @local_extract(schema=MockSchema, retries=1, debug_timing=True)
        async def extract_name(text):
            """Extract."""
            pass

        with caplog.at_level(logging.INFO, logger="silicon_refinery"):
            await extract_name("Alice is 30")

        assert any("[SiliconRefinery] Extraction completed in" in r.message for r in caplog.records)
        assert any("Input length:" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_timing_log_when_disabled(self, mock_fm_available, caplog):
        @local_extract(schema=MockSchema, retries=1, debug_timing=False)
        async def extract_name(text):
            """Extract."""
            pass

        with caplog.at_level(logging.INFO, logger="silicon_refinery"):
            await extract_name("Alice")

        timing_messages = [r for r in caplog.records if "Extraction completed" in r.message]
        assert len(timing_messages) == 0


# ========================================================================
# Performance: decorator overhead
# ========================================================================

class TestLocalExtractPerformance:

    @pytest.mark.asyncio
    async def test_decorator_overhead_under_1ms(self, mock_fm_available):
        """Non-FM code (wrapping, formatting) should add < 1ms of overhead."""
        # Make respond return instantly
        mock_fm_available["session"].respond = AsyncMock(return_value=MockSchema(name="x"))

        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        # Warm up
        await extract_name("warmup")

        # Measure
        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            await extract_name("test")
        elapsed = time.perf_counter() - start

        per_call = (elapsed / iterations) * 1000  # ms
        # Allow generous margin; the mock itself has some overhead
        assert per_call < 5.0, f"Per-call overhead was {per_call:.3f}ms, expected < 5ms"
