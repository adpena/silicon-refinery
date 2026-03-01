"""
Extraction cache example for FMTools.

Demonstrates how to memoize repeated extractions with a sqlite-backed cache.
"""

from __future__ import annotations

import asyncio
import time

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    import apple_fm_sdk as fm
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("extraction_cache.py", exc)

try:
    from fmtools.cache import ExtractionCache, cached_local_extract
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("extraction_cache.py", exc)


@fm.generable()
class SupportTicket:
    category: str = fm.guide(anyOf=["billing", "technical", "account", "other"])
    urgency: str = fm.guide(anyOf=["low", "medium", "high", "critical"])
    summary: str = fm.guide(description="One-sentence issue summary")


CACHE = ExtractionCache(ttl=3600)


@cached_local_extract(schema=SupportTicket, cache=CACHE, retries=2)
async def classify_ticket(email_text: str) -> SupportTicket:
    """Classify support email text into category, urgency, and concise summary."""


async def _timed_call(text: str) -> tuple[SupportTicket, float]:
    start = time.perf_counter()
    result = await classify_ticket(text)
    elapsed = time.perf_counter() - start
    return result, elapsed


async def main() -> None:
    require_apple_fm("extraction_cache.py")

    sample = "I was charged twice for my subscription this month and need a refund."

    print("First call (cache miss, model inference):")
    first, t1 = await _timed_call(sample)
    print(f"  result: {first}")
    print(f"  elapsed: {t1:.3f}s")

    print("\nSecond call (cache hit):")
    second, t2 = await _timed_call(sample)
    print(f"  result: {second}")
    print(f"  elapsed: {t2:.3f}s")

    print("\nCache stats:")
    print(f"  {CACHE.stats()}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
