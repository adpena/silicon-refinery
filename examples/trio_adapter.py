"""
Trio adapter example for SiliconRefinery.

Demonstrates ``TrioAdapter`` against a trio-style receive channel interface.
"""

from __future__ import annotations

import asyncio

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from silicon_refinery.adapters import TrioAdapter
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("trio_adapter.py", exc)


class EndOfChannel(Exception):
    """Small local stand-in for trio.EndOfChannel used in this demo."""


class DemoReceiveChannel:
    def __init__(self, values: list[str]) -> None:
        self._values = list(values)

    async def receive(self) -> str:
        if not self._values:
            raise EndOfChannel
        return self._values.pop(0)


async def main() -> None:
    require_apple_fm("trio_adapter.py")

    channel = DemoReceiveChannel(["event-1", "event-2", "event-3"])
    adapter = TrioAdapter(channel)

    print("Reading from TrioAdapter:")
    async for item in adapter:
        print(f"  - {item}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
