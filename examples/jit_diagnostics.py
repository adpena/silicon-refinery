"""
Runtime diagnostics example for FMTools.

Demonstrates `@diagnose` and the `diagnostics()` collector.
"""

from __future__ import annotations

import asyncio

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from fmtools._jit import diagnose, diagnostics
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("jit_diagnostics.py", exc)


@diagnose
async def classify_line(text: str) -> int:
    if "error" in text.lower():
        raise ValueError("simulated extraction failure")
    return len(text)


async def main() -> None:
    require_apple_fm("jit_diagnostics.py")

    collector = diagnostics()
    collector.reset()

    samples = ["hello world", "another line", "contains error"]
    for sample in samples:
        try:
            value = await classify_line(sample)
            print(f"OK: {sample!r} -> {value}")
        except ValueError as exc:
            print(f"ERR: {sample!r} -> {exc}")

    collector.record_memory()
    print("\n" + collector.report())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
