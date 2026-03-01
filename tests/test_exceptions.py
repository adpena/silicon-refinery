"""Tests for fmtools.exceptions helpers."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fmtools.exceptions import (
    AppleFMSetupError,
    ensure_model_available,
    require_apple_fm,
    troubleshooting_message,
)


def test_troubleshooting_message_includes_context_reason_and_steps():
    message = troubleshooting_message("example.py", reason="boom")
    assert "[example.py] Apple Foundation Models setup check failed." in message
    assert "Reason: boom" in message
    assert "uv sync --all-groups" in message
    assert "fmtools doctor" in message


def test_ensure_model_available_raises_custom_error_when_unavailable():
    model = MagicMock()
    model.is_available.return_value = (False, "model not downloaded")

    with pytest.raises(AppleFMSetupError, match="Foundation Model is not available"):
        ensure_model_available(model, context="unit_test")


def test_require_apple_fm_raises_custom_error_on_import_failure():
    with (
        patch(
            "fmtools.exceptions.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'apple_fm_sdk'"),
        ),
        pytest.raises(AppleFMSetupError, match="No module named 'apple_fm_sdk'"),
    ):
        require_apple_fm("unit_test")


def test_require_apple_fm_returns_module_and_model_on_success():
    mock_model = MagicMock()
    mock_model.is_available.return_value = (True, None)
    mock_fm = SimpleNamespace(SystemLanguageModel=MagicMock(return_value=mock_model))

    with patch("fmtools.exceptions.importlib.import_module", return_value=mock_fm):
        fm_module, model = require_apple_fm("unit_test")

    assert fm_module is mock_fm
    assert model is mock_model
    mock_fm.SystemLanguageModel.assert_called_once_with()
    mock_model.is_available.assert_called_once_with()
