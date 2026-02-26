"""
Custom backend example for SiliconRefinery protocols.

Shows how ``set_backend()`` can redirect ``@local_extract`` away from Apple FM
for testing, simulation, or alternative providers.
"""

from __future__ import annotations

import asyncio

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from silicon_refinery.decorators import local_extract
    from silicon_refinery.protocols import get_backend, set_backend
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("custom_backend.py", exc)


class Person:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"Person(name={self.name!r})"


class DemoModel:
    def is_available(self) -> tuple[bool, str | None]:
        return (True, None)


class DemoSession:
    def __init__(self, instructions: str) -> None:
        self.instructions = instructions

    async def respond(self, prompt: str, generating=None):
        if generating is None:
            return f"echo: {prompt}"
        return generating(name=prompt.strip().title())


class DemoBackend:
    def create_model(self) -> DemoModel:
        return DemoModel()

    def __call__(self, model: DemoModel, instructions: str) -> DemoSession:
        return DemoSession(instructions=instructions)


@local_extract(schema=Person, retries=1)
async def extract_person_name(text: str) -> Person:
    """Extract a person's name from the input."""


async def main() -> None:
    require_apple_fm("custom_backend.py")

    original = get_backend()
    set_backend(DemoBackend())
    try:
        person = await extract_person_name("alice")
        print(person)
    finally:
        set_backend(original)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
