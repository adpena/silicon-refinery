"""
Tests for silicon_refinery.protocols — pluggable backend protocols.

Covers:
  - Runtime-checkable protocol verification
  - AppleFMBackend satisfies ModelProtocol and SessionFactory
  - set_backend / get_backend round-trip
  - Default backend is AppleFMBackend
  - Custom backends can be plugged in
"""

from unittest.mock import AsyncMock, MagicMock, patch

from silicon_refinery.protocols import (
    AppleFMBackend,
    AppleFMModel,
    AppleFMSession,
    ModelProtocol,
    SessionFactory,
    SessionProtocol,
    create_model,
    create_session,
    get_backend,
    set_backend,
)

# ========================================================================
# Lazy SDK import behavior
# ========================================================================


class TestLazySDKImport:
    def test_apple_model_imports_sdk_lazily(self):
        import silicon_refinery.protocols as protocols_mod

        protocols_mod._import_apple_fm_sdk.cache_clear()
        try:
            with patch("silicon_refinery.protocols.importlib.import_module") as import_module:
                sdk = MagicMock()
                sdk.SystemLanguageModel.return_value = MagicMock(
                    is_available=MagicMock(return_value=(True, None))
                )
                import_module.return_value = sdk

                AppleFMModel()

                import_module.assert_called_once_with("apple_fm_sdk")
        finally:
            protocols_mod._import_apple_fm_sdk.cache_clear()


# ========================================================================
# Protocol structural checks
# ========================================================================


class TestProtocolsAreRuntimeCheckable:
    def test_model_protocol_is_runtime_checkable(self):
        """ModelProtocol should be decorated with @runtime_checkable."""
        obj = MagicMock()
        obj.is_available = MagicMock(return_value=(True, None))
        assert isinstance(obj, ModelProtocol)

    def test_session_protocol_is_runtime_checkable(self):
        """SessionProtocol should be decorated with @runtime_checkable."""
        obj = MagicMock()
        obj.respond = AsyncMock()
        assert isinstance(obj, SessionProtocol)

    def test_session_factory_is_runtime_checkable(self):
        """SessionFactory should be decorated with @runtime_checkable."""
        obj = MagicMock()
        # A callable that accepts (model, instructions) and returns a session
        assert isinstance(obj, SessionFactory)


# ========================================================================
# AppleFMBackend satisfies protocols
# ========================================================================


class TestAppleFMBackendSatisfiesModelProtocol:
    def test_apple_fm_backend_satisfies_model_protocol(self):
        """AppleFMModel exposes is_available() matching ModelProtocol."""
        model = AppleFMModel()
        assert isinstance(model, ModelProtocol)

    def test_is_available_returns_tuple(self):
        """is_available() should return a (bool, str|None) tuple."""
        with patch("apple_fm_sdk.SystemLanguageModel") as mock_slm:
            mock_instance = MagicMock()
            mock_instance.is_available.return_value = (True, None)
            mock_slm.return_value = mock_instance

            model = AppleFMModel()
            result = model.is_available()
            assert isinstance(result, tuple)
            assert result == (True, None)


class TestAppleFMBackendSatisfiesSessionProtocol:
    def test_apple_fm_backend_satisfies_session_protocol(self):
        """AppleFMSession exposes respond() matching SessionProtocol."""
        model = MagicMock()
        session = AppleFMSession(model, "test instructions")
        assert isinstance(session, SessionProtocol)

    async def test_session_respond_delegates_to_sdk(self):
        """AppleFMSession.respond() should call the underlying SDK session."""
        mock_sdk_session = MagicMock()
        mock_sdk_session.respond = AsyncMock(return_value="hello")

        with patch("apple_fm_sdk.LanguageModelSession", return_value=mock_sdk_session):
            model = MagicMock()
            session = AppleFMSession(model, "instructions")
            result = await session.respond("prompt")
            assert result == "hello"
            mock_sdk_session.respond.assert_called_once_with("prompt", generating=None)


class TestAppleFMBackendSatisfiesSessionFactory:
    def test_backend_is_callable(self):
        """AppleFMBackend should be callable (SessionFactory)."""
        backend = AppleFMBackend()
        assert callable(backend)

    def test_backend_call_returns_session(self):
        """Calling the backend should produce an AppleFMSession."""
        backend = AppleFMBackend()
        model = MagicMock()
        session = backend(model, "test")
        assert isinstance(session, AppleFMSession)

    def test_backend_satisfies_session_factory_protocol(self):
        """AppleFMBackend instance satisfies SessionFactory protocol."""
        backend = AppleFMBackend()
        assert isinstance(backend, SessionFactory)


# ========================================================================
# set_backend / get_backend
# ========================================================================


class TestSetGetBackend:
    def test_set_get_backend_roundtrip(self):
        """set_backend then get_backend should return the same object."""
        original = get_backend()
        try:
            sentinel = object()
            set_backend(sentinel)
            assert get_backend() is sentinel
        finally:
            set_backend(original)

    def test_default_backend_is_apple_fm(self):
        """The default backend should be an AppleFMBackend instance."""
        # Reset to default by importing a fresh module state
        import silicon_refinery.protocols as mod

        original = mod._backend
        try:
            mod._backend = AppleFMBackend()
            assert isinstance(get_backend(), AppleFMBackend)
        finally:
            mod._backend = original


class TestCreateHelpers:
    def test_create_model_uses_active_backend(self):
        original = get_backend()
        try:
            backend = MagicMock()
            model = MagicMock()
            backend.create_model.return_value = model
            set_backend(backend)

            assert create_model() is model
            backend.create_model.assert_called_once()
        finally:
            set_backend(original)

    def test_create_session_uses_active_backend(self):
        original = get_backend()
        try:
            backend = MagicMock()
            model = MagicMock()
            session = MagicMock()
            backend.create_model.return_value = model
            backend.return_value = session
            set_backend(backend)

            created = create_session("hello")
            assert created is session
            backend.assert_called_once_with(model, "hello")
        finally:
            set_backend(original)

    def test_create_session_uses_same_backend_for_model_and_session(self, monkeypatch):
        backend1 = MagicMock()
        backend1_model = MagicMock()
        backend1_session = MagicMock()
        backend1.create_model.return_value = backend1_model
        backend1.return_value = backend1_session

        backend2 = MagicMock()
        backend2.create_model.return_value = MagicMock()
        backend2.return_value = MagicMock()

        backends = iter([backend1, backend2])
        monkeypatch.setattr("silicon_refinery.protocols.get_backend", lambda: next(backends))

        created = create_session("hello")
        assert created is backend1_session
        backend1.create_model.assert_called_once()
        backend1.assert_called_once_with(backend1_model, "hello")
        backend2.create_model.assert_not_called()


# ========================================================================
# Custom backend
# ========================================================================


class TestCustomBackend:
    def test_custom_backend_can_be_used(self):
        """A plain object with the right methods can serve as a backend."""
        original = get_backend()
        try:

            class MyModel:
                def is_available(self):
                    return (True, None)

            class MySession:
                def respond(self, prompt, generating=None):
                    return "custom"

            class MyBackend:
                def create_model(self):
                    return MyModel()

                def __call__(self, model, instructions):
                    return MySession()

            backend = MyBackend()
            set_backend(backend)
            assert get_backend() is backend

            model = backend.create_model()
            assert isinstance(model, ModelProtocol)

            session = backend(model, "instr")
            assert isinstance(session, SessionProtocol)
        finally:
            set_backend(original)

    def test_custom_backend_model_is_available(self):
        """Custom model.is_available() should be callable."""
        original = get_backend()
        try:

            class StubModel:
                def is_available(self):
                    return (False, "not ready")

            model = StubModel()
            assert isinstance(model, ModelProtocol)
            available, reason = model.is_available()
            assert available is False
            assert reason == "not ready"
        finally:
            set_backend(original)
