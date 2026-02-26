"""
Free-threading utilities example for SiliconRefinery.

Demonstrates runtime GIL detection and thread-safe counters/containers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from examples._support import AppleFMSetupError, raise_sdk_setup_error, require_apple_fm

try:
    from silicon_refinery._threading import (
        AtomicCounter,
        ThreadSafeDict,
        get_gil_status,
        is_free_threaded,
        safe_model_cache,
    )
except Exception as exc:  # pragma: no cover - import error path
    raise_sdk_setup_error("free_threading.py", exc)


def main() -> None:
    require_apple_fm("free_threading.py")

    print("Free-threaded build:", is_free_threaded())
    print("GIL status:", get_gil_status().value)
    print("Model cache guard:", type(safe_model_cache()).__name__)

    counter = AtomicCounter()
    seen = ThreadSafeDict()

    def worker(worker_id: int) -> None:
        for idx in range(100):
            counter.increment()
            seen.set(f"{worker_id}:{idx}", idx)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for wid in range(4):
            pool.submit(worker, wid).result()

    print("Counter value:", counter.value)
    print("Stored keys:", len(seen))


if __name__ == "__main__":
    try:
        main()
    except AppleFMSetupError as exc:
        print(exc)
        raise SystemExit(2) from exc
