"""
Arrow bridge example for FMTools.

Demonstrates Arrow IPC buffer/file round-trips and streaming writer usage.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from fmtools.arrow_bridge import (
        ArrowStreamWriter,
        from_arrow_ipc,
        from_arrow_ipc_buffer,
        to_arrow_ipc,
        to_arrow_ipc_buffer,
    )
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("arrow_bridge.py", exc)


async def main() -> None:
    require_apple_fm("arrow_bridge.py")

    records = [
        {"name": "Alice", "score": 98.5},
        {"name": "Bob", "score": 88.0},
    ]

    try:
        buf = to_arrow_ipc_buffer(records)
        restored = from_arrow_ipc_buffer(buf)
    except ImportError as exc:
        print(exc)
        return

    print("Buffer round-trip:", restored)

    with tempfile.TemporaryDirectory(prefix="sr_arrow_") as tmp:
        out = Path(tmp) / "records.arrow"
        to_arrow_ipc(records, out)
        print("File round-trip:", from_arrow_ipc(out))

        stream_path = Path(tmp) / "stream.arrow"
        async with ArrowStreamWriter(stream_path) as writer:
            for row in records:
                await writer.write(row)
        print("Stream writer file:", from_arrow_ipc(stream_path))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
