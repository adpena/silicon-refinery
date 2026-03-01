"""
Shared fixtures and mock factories for the FMTools test suite.

CRITICAL: The apple_fm_sdk module-level mock MUST be installed before any
fmtools modules are imported, because:
  - debugging.py uses @fm.generable() and fm.guide() at class definition time
  - decorators.py, pipeline.py, etc. do `import apple_fm_sdk as fm` at module level

All tests mock apple_fm_sdk objects because the real SDK requires macOS 26+
with Apple Silicon hardware and a running Foundation Model service.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Install a fake apple_fm_sdk into sys.modules BEFORE any fmtools
# imports happen.  This is the single most important line in the test suite.
# ---------------------------------------------------------------------------
_mock_fm = MagicMock()
_mock_fm.generable.return_value = lambda cls: cls  # passthrough decorator
_mock_fm.guide.return_value = None  # field descriptor -> None
_mock_fm.SystemLanguageModel = MagicMock
_mock_fm.LanguageModelSession = MagicMock
sys.modules["apple_fm_sdk"] = _mock_fm

# Now it is safe to import pytest (which may trigger conftest collection)
import pytest  # noqa: E402

# ---------------------------------------------------------------------------
# Lightweight schema stand-in (replaces @fm.generable() decorated classes)
# ---------------------------------------------------------------------------


class MockSchema:
    """A plain class that mimics an @fm.generable() schema for testing."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"MockSchema({attrs})"

    def __eq__(self, other):
        if not isinstance(other, MockSchema):
            return NotImplemented
        return vars(self) == vars(other)


class MockDebuggingAnalysis:
    """Stand-in for debugging.DebuggingAnalysis generable schema."""

    def __init__(
        self,
        error_summary="Test error",
        possible_causes=None,
        certainty_level="HIGH",
        suggested_fix="Fix the test",
    ):
        self.error_summary = error_summary
        self.possible_causes = possible_causes or ["cause1", "cause2"]
        self.certainty_level = certainty_level
        self.suggested_fix = suggested_fix


# ---------------------------------------------------------------------------
# Mock factory functions
# ---------------------------------------------------------------------------


def make_mock_model(available=True, reason=None):
    """Create a mock SystemLanguageModel with configurable availability."""
    model = MagicMock()
    model.is_available.return_value = (available, reason)
    return model


def make_mock_session(respond_return=None, respond_side_effect=None):
    """
    Create a mock LanguageModelSession.

    Args:
        respond_return: The value that session.respond() will return.
        respond_side_effect: An exception or list to use as side_effect.
    """
    session = MagicMock()
    session.respond = AsyncMock()
    if respond_side_effect is not None:
        session.respond.side_effect = respond_side_effect
    elif respond_return is not None:
        session.respond.return_value = respond_return
    else:
        session.respond.return_value = MockSchema(name="default")
    return session


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fm_available():
    """
    Patches apple_fm_sdk so that SystemLanguageModel is available and
    LanguageModelSession.respond returns a MockSchema by default.

    Returns a dict with handles to the mock objects for assertion.
    """
    mock_model = make_mock_model(available=True)
    mock_session = make_mock_session(respond_return=MockSchema(name="test_result"))

    with (
        patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model) as model_cls,
        patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as session_cls,
    ):
        yield {
            "model_cls": model_cls,
            "model": mock_model,
            "session_cls": session_cls,
            "session": mock_session,
        }


@pytest.fixture
def mock_fm_unavailable():
    """Patches apple_fm_sdk so that the model is reported as unavailable."""
    mock_model = make_mock_model(available=False, reason="Model not downloaded")

    with (
        patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model) as model_cls,
        patch("apple_fm_sdk.LanguageModelSession") as session_cls,
    ):
        yield {
            "model_cls": model_cls,
            "model": mock_model,
            "session_cls": session_cls,
        }


@pytest.fixture
def mock_fm_failing():
    """
    Patches apple_fm_sdk so the model is available but session.respond
    always raises an exception.
    """
    mock_model = make_mock_model(available=True)
    mock_session = make_mock_session(respond_side_effect=RuntimeError("generation failed"))

    with (
        patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
        patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
    ):
        yield {
            "model": mock_model,
            "session": mock_session,
        }


@pytest.fixture
def mock_fm_context_window_error():
    """
    Patches apple_fm_sdk so session.respond raises ExceededContextWindowSizeError.
    """
    mock_model = make_mock_model(available=True)
    exc = Exception("ExceededContextWindowSizeError: context window exceeded")
    mock_session = make_mock_session(respond_side_effect=exc)

    with (
        patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
        patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
    ):
        yield {
            "model": mock_model,
            "session": mock_session,
        }


@pytest.fixture
def sample_schema():
    """A reusable schema class for tests."""
    return MockSchema


@pytest.fixture
def sample_data():
    """Sample string data for stream/pipeline tests."""
    return ["Alice is 30", "Bob is 25", "Charlie is 40"]


@pytest.fixture
def large_sample_data():
    """A larger dataset for throughput/chunking tests."""
    return [f"Person_{i} is {20 + i}" for i in range(100)]


@pytest.fixture
def unicode_sample_data():
    """Unicode data for edge case testing."""
    return [
        "Rene is from Zurich",
        "Tanaka Taro lives in Tokyo",
        "Ahmed from Cairo",
        "",  # empty string edge case
    ]
