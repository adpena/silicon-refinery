"""
Hot-folder watcher example for SiliconRefinery.

Demonstrates event streaming with `HotFolder` using a temporary directory.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from silicon_refinery.watcher import HotFolder
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("hot_folder_watcher.py", exc)


async def main() -> None:
    require_apple_fm("hot_folder_watcher.py")

    with tempfile.TemporaryDirectory(prefix="sr_watch_") as tmp:
        watch_dir = Path(tmp)
        target = watch_dir / "incoming.txt"

        async def writer() -> None:
            await asyncio.sleep(0.10)
            target.write_text("first\n", encoding="utf-8")
            await asyncio.sleep(1.10)
            target.write_text("second\n", encoding="utf-8")

        task = asyncio.create_task(writer())
        events_seen = 0

        async with HotFolder(watch_dir, patterns=["*.txt"], poll_interval=0.05) as watcher:
            async for event in watcher.watch():
                print(f"Event: {event.event_type} -> {event.path.name}")
                events_seen += 1
                if events_seen >= 2:
                    watcher.stop()

        await task
        print("Done.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
