"""
Comprehensive tests for silicon_refinery.decorators (local_extract).

Covers:
  - Decorator structural properties (wrapping, name preservation)
  - Successful extraction with mocked FM
  - Input formatting (args, kwargs, mixed)
  - Retry logic on transient failures (TimeoutError, ConnectionError, OSError)
  - Non-transient errors fail immediately (TypeError, ValueError)
  - Retry exhaustion leading to RuntimeError
  - Model unavailability check
  - debug_timing logging
  - Edge cases: empty docstring, no args, None args, Unicode input
  - Model caching across calls
  - Performance: decorator overhead
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_refinery.decorators import local_extract
from silicon_refinery.exceptions import AppleFMSetupError

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
        import inspect

        @local_extract(schema=MockSchema)
        async def my_extractor(text: str):
            """Extract."""
            pass

        assert inspect.iscoroutinefunction(my_extractor)


# ========================================================================
# Happy path: successful extraction
# ========================================================================


class TestLocalExtractHappyPath:
    async def test_basic_extraction(self, mock_fm_available):
        expected = MockSchema(name="Alice")
        mock_fm_available["session"].respond.return_value = expected

        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text: str):
            """Extract the person's name."""
            pass

        result = await extract_name("Alice is 30 years old")
        assert result == expected

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
    async def test_positional_args_joined_with_space(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(a, b, c):
            """Extract."""
            pass

        await extract_name("Hello", "World", "!")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == "Hello World !"

    async def test_kwargs_appended_as_key_value(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(**kwargs):
            """Extract."""
            pass

        await extract_name(name="Alice", age="30")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert "name: Alice" in call_args[0]
        assert "age: 30" in call_args[0]

    async def test_mixed_args_and_kwargs(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text, extra=None):
            """Extract."""
            pass

        await extract_name("Hello", extra="World")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0].startswith("Hello")
        assert "extra: World" in call_args[0]

    async def test_no_args_sends_empty_string(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name():
            """Extract."""
            pass

        await extract_name()

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == ""

    async def test_none_arg_converted_to_string(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        await extract_name(None)

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == "None"

    async def test_unicode_input(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        await extract_name("Tanaka Taro")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert "Tanaka" in call_args[0]

    async def test_empty_string_input(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        await extract_name("")

        call_args = mock_fm_available["session"].respond.call_args[0]
        assert call_args[0] == ""


# ========================================================================
# Fallback docstring
# ========================================================================


class TestLocalExtractDocstring:
    async def test_missing_docstring_uses_default(self, mock_fm_available):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            pass  # no docstring

        await extract_name("Alice")

        call_kwargs = mock_fm_available["session_cls"].call_args
        assert call_kwargs[1]["instructions"] == "Extract the following data."


# ========================================================================
# Retry logic - transient errors
# ========================================================================


class TestLocalExtractRetries:
    async def test_retries_on_timeout_error_then_succeeds(self):
        """TimeoutError is transient and should be retried."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session()
        # Fail first with transient TimeoutError, succeed second
        mock_session.respond = AsyncMock(
            side_effect=[TimeoutError("timed out"), MockSchema(name="Alice")]
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

    async def test_retries_on_connection_error(self):
        """ConnectionError is transient and should be retried."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session()
        mock_session.respond = AsyncMock(
            side_effect=[ConnectionError("lost connection"), MockSchema(name="Bob")]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            result = await extract_name("Bob")
            assert result == MockSchema(name="Bob")
            assert mock_session.respond.call_count == 2

    async def test_retries_on_os_error(self):
        """OSError is transient and should be retried."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session()
        mock_session.respond = AsyncMock(
            side_effect=[OSError("device busy"), MockSchema(name="Charlie")]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            result = await extract_name("Charlie")
            assert result == MockSchema(name="Charlie")
            assert mock_session.respond.call_count == 2

    async def test_retry_exhaustion_raises_runtime_error(self):
        """All retries exhausted on transient error raises RuntimeError."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=TimeoutError("persistent timeout"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(
                RuntimeError, match="Failed to generate structured data after 3 attempts"
            ):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 3

    async def test_retries_equals_one_no_retry(self):
        """With retries=1, only one attempt is made."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=TimeoutError("fail"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=1)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(
                RuntimeError, match="Failed to generate structured data after 1 attempts"
            ):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 1


# ========================================================================
# Non-transient errors fail immediately
# ========================================================================


class TestLocalExtractNonTransientErrors:
    async def test_type_error_not_retried(self):
        """TypeError is non-transient and should fail immediately."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=TypeError("bad type"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=5)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(RuntimeError, match="Failed to generate structured data"):
                await extract_name("Alice")

            # Should only be called once -- no retries for non-transient errors
            assert mock_session.respond.call_count == 1

    async def test_value_error_not_retried(self):
        """ValueError is non-transient and should fail immediately."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=ValueError("bad value"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=5)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(RuntimeError, match="Failed to generate structured data"):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 1

    async def test_key_error_not_retried(self):
        """KeyError is non-transient and should fail immediately."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(respond_side_effect=KeyError("missing key"))

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=5)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(RuntimeError, match="Failed to generate structured data"):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 1

    async def test_setup_error_passthrough_not_wrapped(self):
        """AppleFMSetupError should propagate unchanged in non-transient branch."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=AppleFMSetupError("preserve setup diagnostics")
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            with pytest.raises(AppleFMSetupError, match="preserve setup diagnostics"):
                await extract_name("Alice")

            assert mock_session.respond.call_count == 1


# ========================================================================
# Model unavailability
# ========================================================================


class TestLocalExtractModelUnavailable:
    async def test_raises_when_model_unavailable(self, mock_fm_unavailable):
        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        with pytest.raises(AppleFMSetupError, match="Foundation Model is not available"):
            await extract_name("Alice")


# ========================================================================
# Model caching
# ========================================================================


class TestLocalExtractModelCaching:
    async def test_model_cached_across_calls(self, mock_fm_available):
        """The model should be created only once and reused on subsequent calls."""

        @local_extract(schema=MockSchema, retries=1)
        async def extract_name(text):
            """Extract."""
            pass

        await extract_name("first")
        await extract_name("second")
        await extract_name("third")

        # SystemLanguageModel should only be called once
        assert mock_fm_available["model_cls"].call_count == 1
        # But session should be created each time
        assert mock_fm_available["session_cls"].call_count == 3


# ========================================================================
# debug_timing
# ========================================================================


class TestLocalExtractDebugTiming:
    async def test_debug_timing_logs_message(self, mock_fm_available, caplog):
        @local_extract(schema=MockSchema, retries=1, debug_timing=True)
        async def extract_name(text):
            """Extract."""
            pass

        with caplog.at_level(logging.INFO, logger="silicon_refinery"):
            await extract_name("Alice is 30")

        assert any("[SiliconRefinery] Extraction completed in" in r.message for r in caplog.records)
        assert any("Input length:" in r.message for r in caplog.records)

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
# Exponential backoff timing
# ========================================================================


class TestLocalExtractBackoff:
    async def test_exponential_backoff_between_retries(self):
        """Verify that asyncio.sleep is called with exponential backoff delays."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=[
                TimeoutError("fail 1"),
                TimeoutError("fail 2"),
                MockSchema(name="success"),
            ]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            result = await extract_name("test")
            assert result == MockSchema(name="success")

            # Check exponential backoff: (2^0)*0.1 = 0.1, (2^1)*0.1 = 0.2
            assert mock_sleep.call_count == 2
            delays = [call.args[0] for call in mock_sleep.call_args_list]
            assert abs(delays[0] - 0.1) < 0.01
            assert abs(delays[1] - 0.2) < 0.01


# ========================================================================
# Performance: decorator overhead
# ========================================================================


class TestLocalExtractPerformance:
    async def test_decorator_overhead_under_5ms(self, mock_fm_available):
        """Non-FM code (wrapping, formatting) should add < 5ms of overhead."""
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
        assert per_call < 5.0, f"Per-call overhead was {per_call:.3f}ms, expected < 5ms"


# ========================================================================
# Fuzz-scan edge-case tests: retries=0
# ========================================================================


class TestLocalExtractRetriesZero:
    """Test that invalid retries values are rejected eagerly."""

    def test_retries_zero_rejected(self):
        with pytest.raises(ValueError, match="retries must be >= 1"):
            local_extract(schema=MockSchema, retries=0)

    def test_retries_negative_rejected(self):
        with pytest.raises(ValueError, match="retries must be >= 1"):
            local_extract(schema=MockSchema, retries=-1)


# ========================================================================
# Fuzz-scan edge-case tests: fresh session per retry
# ========================================================================


class TestLocalExtractFreshSessionPerRetry:
    """Verify that a NEW session is created for each retry attempt (not reused)."""

    async def test_new_session_created_per_retry_attempt(self):
        """Each retry should create a brand new LanguageModelSession."""
        mock_model = make_mock_model(available=True)
        mock_session = make_mock_session(
            respond_side_effect=[
                TimeoutError("fail 1"),
                TimeoutError("fail 2"),
                MockSchema(name="success"),
            ]
        )

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as sess_cls,
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            result = await extract_name("Alice")
            assert result == MockSchema(name="success")

            # LanguageModelSession should be called 3 times: once per attempt
            assert sess_cls.call_count == 3

    async def test_sessions_are_distinct_objects(self):
        """Each session should be a separate instance (not reused)."""
        mock_model = make_mock_model(available=True)

        sessions_created = []

        def session_factory(*args, **kwargs):
            session = MagicMock()
            session.respond = AsyncMock(
                side_effect=[TimeoutError("fail")] if len(sessions_created) < 2 else None,
                return_value=MockSchema(name="ok"),
            )
            sessions_created.append(session)
            return session

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", side_effect=session_factory),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):

            @local_extract(schema=MockSchema, retries=3)
            async def extract_name(text):
                """Extract."""
                pass

            await extract_name("test")

            # All sessions should be distinct objects
            assert len(sessions_created) >= 2
            for i in range(len(sessions_created)):
                for j in range(i + 1, len(sessions_created)):
                    assert sessions_created[i] is not sessions_created[j]
