"""
Context scoping example for FMTools.

Demonstrates task-local session/model access via `session_scope(...)`.
"""

from __future__ import annotations

import asyncio

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from fmtools._context import get_instructions, get_model, get_session, session_scope
    from fmtools.protocols import get_backend, set_backend
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("context_scope.py", exc)


class DemoModel:
    def is_available(self) -> tuple[bool, str | None]:
        return (True, None)


class DemoSession:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions

    async def respond(self, prompt: str, generating=None):
        del generating
        return f"{self.instructions} | {prompt}"


class DemoBackend:
    def create_model(self) -> DemoModel:
        return DemoModel()

    def __call__(self, model: DemoModel, instructions: str) -> DemoSession:
        del model
        return DemoSession(instructions=instructions)


async def main() -> None:
    require_apple_fm("context_scope.py")

    original = get_backend()
    set_backend(DemoBackend())
    try:
        async with session_scope("Extract entities.") as session:
            assert get_session() is session
            print("Instructions:", get_instructions())
            print("Model type:", type(get_model()).__name__)
            print("Response:", await session.respond("Alice is 31"))
    finally:
        set_backend(original)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
