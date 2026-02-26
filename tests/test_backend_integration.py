"""Integration tests for protocol backends wired into runtime modules."""

from __future__ import annotations

from silicon_refinery._context import get_session, session_scope
from silicon_refinery.async_generators import stream_extract
from silicon_refinery.decorators import local_extract
from silicon_refinery.pipeline import Extract, Pipeline, Sink, Source
from silicon_refinery.protocols import get_backend, set_backend

from .conftest import MockSchema


class _DummyModel:
    def is_available(self):
        return (True, None)


class _DummySession:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions
        self.calls: list[tuple[str, object]] = []

    async def respond(self, prompt: str, generating=None):
        self.calls.append((prompt, generating))
        if generating is None:
            return f"echo:{prompt}"
        return generating(name=prompt)


class _DummyBackend:
    def __init__(self) -> None:
        self.models_created = 0
        self.sessions: list[_DummySession] = []

    def create_model(self):
        self.models_created += 1
        return _DummyModel()

    def __call__(self, model, instructions: str):
        session = _DummySession(instructions=instructions)
        self.sessions.append(session)
        return session


class TestBackendRuntimeIntegration:
    async def test_local_extract_uses_active_backend(self):
        original = get_backend()
        backend = _DummyBackend()
        set_backend(backend)
        try:

            @local_extract(schema=MockSchema, retries=1)
            async def extract_name(text: str):
                """Extract name."""

            result = await extract_name("Alice")
            assert result == MockSchema(name="Alice")
            assert backend.models_created == 1
            assert backend.sessions[0].instructions == "Extract name."
            assert backend.sessions[0].calls[0][0] == "Alice"
        finally:
            set_backend(original)

    async def test_stream_extract_uses_active_backend(self):
        original = get_backend()
        backend = _DummyBackend()
        set_backend(backend)
        try:
            results = [
                item
                async for item in stream_extract(
                    ["Bob", "Carol"],
                    MockSchema,
                    instructions="Stream extract names.",
                    concurrency=1,
                    history_mode="clear",
                )
            ]

            assert [r.name for r in results] == ["Bob", "Carol"]
            assert backend.models_created == 1
            assert all(s.instructions == "Stream extract names." for s in backend.sessions)
        finally:
            set_backend(original)

    async def test_pipeline_extract_uses_active_backend(self):
        original = get_backend()
        backend = _DummyBackend()
        set_backend(backend)
        try:
            seen = []
            pipe = Pipeline(
                Source(["Dave"]),
                Extract(MockSchema, instructions="Pipeline extract names."),
                Sink(lambda item: seen.append(item)),
            )

            results = [item async for item in pipe.execute()]
            assert [r.name for r in results] == ["Dave"]
            assert [r.name for r in seen] == ["Dave"]
            assert backend.models_created == 1
            assert backend.sessions[0].instructions == "Pipeline extract names."
        finally:
            set_backend(original)

    async def test_context_scope_uses_active_backend(self):
        original = get_backend()
        backend = _DummyBackend()
        set_backend(backend)
        try:
            async with session_scope("Context scope instructions.") as session:
                assert get_session() is session
                assert backend.models_created == 1
                assert backend.sessions[0] is session
                assert session.instructions == "Context scope instructions."
        finally:
            set_backend(original)
