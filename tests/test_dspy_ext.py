"""
Comprehensive tests for silicon_refinery.dspy_ext (AppleFMLM).

Covers:
  - __init__ sets up model, kwargs, provider, history (deque with maxlen)
  - basic_request: sync-to-async bridge, session.respond called, returns list
  - __call__ with prompt string
  - __call__ with messages list (DSPy v2.5+ format)
  - __call__ with neither prompt nor messages raises ValueError
  - __call__ prefers messages over prompt when both provided
  - History tracking (bounded deque)
  - Event loop handling: running loop vs no loop
"""

import collections
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .conftest import make_mock_model

# ========================================================================
# Construction
# ========================================================================


class TestAppleFMLMInit:
    def test_default_init(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            assert lm.provider == "apple_fm"
            assert lm.kwargs["temperature"] == 0.0
            assert lm.kwargs["max_tokens"] == 1024

    def test_custom_model_name(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM(model_name="custom_model")
            # The model attribute is set by dspy.LM parent
            assert lm.model == "custom_model"

    def test_history_is_bounded_deque(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            assert isinstance(lm.history, collections.deque)
            assert lm.history.maxlen == 1000

    def test_fm_model_created_at_init(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model) as model_cls:
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            model_cls.assert_called_once()
            assert lm.fm_model.raw is mock_model


# ========================================================================
# basic_request
# ========================================================================


class TestAppleFMLMBasicRequest:
    def test_basic_request_returns_list_of_strings(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="The answer is 42.")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            result = lm.basic_request("What is the answer?")

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0] == "The answer is 42."

    def test_basic_request_calls_session_respond(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="response")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm.basic_request("test prompt")

            mock_session.respond.assert_called_once_with("test prompt", generating=None)

    def test_basic_request_converts_response_to_string(self):
        """Whatever respond returns gets wrapped in str() and a list."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        # Return an object, not a string
        mock_session.respond = AsyncMock(return_value=42)

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            result = lm.basic_request("prompt")

            assert result == ["42"]


# ========================================================================
# __call__ with prompt
# ========================================================================


class TestAppleFMLMCallWithPrompt:
    def test_call_with_prompt_string(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="response text")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            result = lm(prompt="Hello world")

            assert result == ["response text"]

    def test_call_appends_to_history(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm(prompt="test prompt")

            assert len(lm.history) == 1
            assert lm.history[0]["prompt"] == "test prompt"
            assert lm.history[0]["response"] == ["resp"]

    def test_multiple_calls_accumulate_history(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm(prompt="first")
            lm(prompt="second")
            lm(prompt="third")

            assert len(lm.history) == 3
            prompts = [entry["prompt"] for entry in lm.history]
            assert prompts == ["first", "second", "third"]


# ========================================================================
# __call__ with messages (DSPy v2.5+ format)
# ========================================================================


class TestAppleFMLMCallWithMessages:
    def test_call_with_messages_flattens_content(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="response")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
            ]
            result = lm(messages=messages)

            assert result == ["response"]
            # Verify the prompt was flattened
            call_args = mock_session.respond.call_args[0]
            assert "You are helpful." in call_args[0]
            assert "What is 2+2?" in call_args[0]

    def test_call_with_messages_handles_empty_content(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="response")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            messages = [
                {"role": "system"},  # missing "content" key
                {"role": "user", "content": "Hello"},
            ]
            lm(messages=messages)

            # msg.get("content", "") should handle missing key
            call_args = mock_session.respond.call_args[0]
            assert "Hello" in call_args[0]

    def test_messages_joined_with_newlines(self):
        """Messages should be joined with newline separators."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="response")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            messages = [
                {"role": "user", "content": "line1"},
                {"role": "user", "content": "line2"},
            ]
            lm(messages=messages)

            call_args = mock_session.respond.call_args[0]
            assert call_args[0] == "line1\nline2"


# ========================================================================
# __call__ error handling
# ========================================================================


class TestAppleFMLMCallErrors:
    def test_call_with_neither_prompt_nor_messages_raises(self):
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            with pytest.raises(ValueError, match="Either prompt or messages must be provided"):
                lm()

    def test_call_prefers_messages_over_prompt(self):
        """When both are provided, messages takes precedence."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm(
                prompt="this should be ignored",
                messages=[{"role": "user", "content": "this should be used"}],
            )

            call_args = mock_session.respond.call_args[0]
            assert "this should be used" in call_args[0]
            # The prompt should NOT appear because messages != None
            assert "this should be ignored" not in call_args[0]


# ========================================================================
# History bounding (deque maxlen)
# ========================================================================


class TestAppleFMLMHistoryBounding:
    def test_history_does_not_exceed_maxlen(self):
        """History should be bounded by the deque maxlen (1000)."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            # Manually set a small maxlen for testing
            lm.history = collections.deque(maxlen=5)

            for i in range(10):
                lm(prompt=f"prompt_{i}")

            # Only the last 5 should remain
            assert len(lm.history) == 5
            prompts = [entry["prompt"] for entry in lm.history]
            assert prompts == [f"prompt_{i}" for i in range(5, 10)]


# ========================================================================
# Fuzz-scan edge-case tests: __call__ edge cases
# ========================================================================


class TestAppleFMLMCallEdgeCases:
    """Test edge cases for the __call__ method: empty messages list and
    extra kwargs passed to basic_request."""

    def test_call_with_empty_messages_sends_empty_prompt(self):
        """messages=[] should result in an empty prompt string being sent."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="empty response")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            result = lm(messages=[])

            assert result == ["empty response"]
            # The prompt should be empty since no messages to join
            call_args = mock_session.respond.call_args[0]
            assert call_args[0] == ""

    def test_call_with_empty_messages_records_history(self):
        """Even with messages=[], history should be recorded."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm(messages=[])

            assert len(lm.history) == 1
            assert lm.history[0]["prompt"] == ""
            assert lm.history[0]["response"] == ["resp"]

    def test_extra_kwargs_passed_to_basic_request_do_not_error(self):
        """kwargs like temperature, max_tokens passed to __call__ should be
        forwarded to basic_request without causing errors."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp with kwargs")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            # Pass extra kwargs that basic_request accepts via **kwargs
            result = lm(
                prompt="test",
                temperature=0.5,
                max_tokens=2048,
                custom_param="value",
            )

            assert result == ["resp with kwargs"]

    def test_call_with_single_empty_content_message(self):
        """A single message with empty content should produce an empty prompt."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm(messages=[{"role": "user", "content": ""}])

            call_args = mock_session.respond.call_args[0]
            assert call_args[0] == ""


# ========================================================================
# Fuzz-scan edge-case tests: executor caching
# ========================================================================


class TestAppleFMLMExecutorCaching:
    """Verify _executor is created at init and reused across calls
    (not created per-call)."""

    def test_executor_exists_at_init(self):
        """_executor should be set during __init__."""
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            assert hasattr(lm, "_executor")
            assert lm._executor is not None

    def test_executor_is_threadpool_executor(self):
        """_executor should be a ThreadPoolExecutor instance."""
        import concurrent.futures

        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            assert isinstance(lm._executor, concurrent.futures.ThreadPoolExecutor)

    def test_executor_reused_across_calls(self):
        """The same _executor instance should be used for every call."""
        mock_model = make_mock_model(available=True)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value="resp")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            executor_before = lm._executor

            lm(prompt="first call")
            assert lm._executor is executor_before

            lm(prompt="second call")
            assert lm._executor is executor_before

            lm(prompt="third call")
            assert lm._executor is executor_before

    def test_executor_max_workers_is_one(self):
        """The executor should have max_workers=1."""
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            assert lm._executor._max_workers == 1


# ========================================================================
# Executor lifecycle cleanup
# ========================================================================


class TestAppleFMLMExecutorLifecycle:
    def test_close_shuts_down_executor_once(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm._executor.shutdown = MagicMock()

            lm.close()
            lm.close()

            lm._executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)

    def test_context_manager_closes_executor(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            with AppleFMLM() as lm:
                lm._executor.shutdown = MagicMock()

            lm._executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
            with pytest.raises(RuntimeError, match="AppleFMLM is closed"):
                lm.basic_request("hello")

    def test_finalizer_safe_when_invoked_before_close(self):
        mock_model = make_mock_model(available=True)

        with patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model):
            from silicon_refinery.dspy_ext import AppleFMLM

            lm = AppleFMLM()
            lm._executor.shutdown = MagicMock()

            lm._executor_finalizer()
            lm.close()  # Should not raise or call shutdown twice.

            lm._executor.shutdown.assert_called_once_with(wait=False, cancel_futures=True)
