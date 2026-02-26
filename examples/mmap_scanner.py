"""
mmap scanner example for SiliconRefinery.

Demonstrates `MMapScanner` and `line_split_scanner` on a temporary file.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from silicon_refinery.scanner import MMapScanner, line_split_scanner
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("mmap_scanner.py", exc)


async def main() -> None:
    require_apple_fm("mmap_scanner.py")

    with tempfile.TemporaryDirectory(prefix="sr_scanner_") as tmp:
        sample = Path(tmp) / "sample.log"
        sample.write_text(
            "alpha line\nbeta line\n\ngamma line with unicode: cafe\n",
            encoding="utf-8",
        )

        print("Windowed chunks:")
        async for chunk in MMapScanner(sample, window_size=16, overlap=4):
            print(f"  {chunk!r}")

        print("\nLine split scanner:")
        async for line in line_split_scanner(sample, batch_size=2):
            print(f"  {line!r}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
