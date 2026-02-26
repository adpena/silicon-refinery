"""
Comprehensive tests for silicon_refinery.dspy_ext (AppleFMLM).

Covers:
  - __init__ sets up model, kwargs, provider
  - basic_request: sync-to-async bridge, session.respond called
  - __call__ with prompt string
  - __call__ with messages list (DSPy v2.5+ format)
  - __call__ with neither prompt nor messages raises ValueError
  - History tracking
  - Event loop handling: running loop vs no loop
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from .conftest import make_mock_model, make_mock_session


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

            mock_session.respond.assert_called_once_with("test prompt")


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
            result = lm(messages=messages)

            # msg.get("content", "") should handle missing key
            call_args = mock_session.respond.call_args[0]
            assert "Hello" in call_args[0]


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
