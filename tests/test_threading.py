"""
Tests for silicon_refinery._threading — free-threading support utilities.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import silicon_refinery._threading as threading_mod
from silicon_refinery._threading import (
    AtomicCounter,
    CriticalSection,
    GILStatus,
    ThreadSafeDict,
    _NoOpLock,
    get_gil_status,
    is_free_threaded,
    safe_model_cache,
)

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def test_is_free_threaded_returns_bool():
    result = is_free_threaded()
    assert isinstance(result, bool)


def test_gil_status_returns_enum():
    status = get_gil_status()
    assert isinstance(status, GILStatus)
    assert status in (GILStatus.ENABLED, GILStatus.DISABLED, GILStatus.UNKNOWN)


# ---------------------------------------------------------------------------
# CriticalSection
# ---------------------------------------------------------------------------


def test_critical_section_sync_usage():
    cs = CriticalSection()
    with cs:
        pass  # should not raise


async def test_critical_section_async_usage():
    cs = CriticalSection()
    async with cs:
        pass  # should not raise


def test_critical_section_noop_on_gil_build():
    """On a regular GIL build the lock should be a _NoOpLock (zero overhead)."""
    cs = CriticalSection()
    if not cs.is_real_lock:
        # We're on a GIL build — verify internal lock type
        assert isinstance(cs._lock, _NoOpLock)
    else:
        # Free-threaded build — lock should be a real Lock
        assert isinstance(cs._lock, type(threading.Lock()))


# ---------------------------------------------------------------------------
# AtomicCounter
# ---------------------------------------------------------------------------


def test_atomic_counter_increment_decrement():
    counter = AtomicCounter(initial=0)
    assert counter.value == 0

    assert counter.increment() == 1
    assert counter.increment(5) == 6
    assert counter.decrement() == 5
    assert counter.decrement(3) == 2
    assert counter.value == 2


def test_atomic_counter_concurrent_increments():
    """Many threads incrementing the same counter should produce a correct total."""
    counter = AtomicCounter(initial=0)
    num_threads = 8
    increments_per_thread = 1000

    def worker() -> None:
        for _ in range(increments_per_thread):
            counter.increment()

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.value == num_threads * increments_per_thread


# ---------------------------------------------------------------------------
# ThreadSafeDict
# ---------------------------------------------------------------------------


def test_thread_safe_dict_basic_operations():
    d = ThreadSafeDict()

    # set / get
    d.set("a", 1)
    assert d.get("a") == 1
    assert d.get("missing", 42) == 42

    # __contains__ / __len__
    assert "a" in d
    assert "z" not in d
    assert len(d) == 1

    # items
    d.set("b", 2)
    assert sorted(d.items()) == [("a", 1), ("b", 2)]

    # delete
    d.delete("a")
    assert "a" not in d
    assert len(d) == 1


def test_thread_safe_dict_concurrent_access():
    """Multiple threads reading and writing concurrently should not corrupt data."""
    d = ThreadSafeDict()
    num_threads = 8
    ops_per_thread = 500

    def writer(tid: int) -> None:
        for i in range(ops_per_thread):
            d.set(f"{tid}:{i}", i)

    def reader(tid: int) -> None:
        for i in range(ops_per_thread):
            d.get(f"{tid}:{i}")

    with ThreadPoolExecutor(max_workers=num_threads * 2) as pool:
        futs = []
        for tid in range(num_threads):
            futs.append(pool.submit(writer, tid))
            futs.append(pool.submit(reader, tid))
        for f in futs:
            f.result()  # propagate exceptions

    assert len(d) == num_threads * ops_per_thread


# ---------------------------------------------------------------------------
# safe_model_cache
# ---------------------------------------------------------------------------


def test_safe_model_cache_returns_critical_section():
    cs = safe_model_cache()
    assert isinstance(cs, CriticalSection)

    # Calling again should return the same instance
    assert safe_model_cache() is cs


def test_safe_model_cache_initializes_once_under_concurrency(monkeypatch):
    monkeypatch.setattr(threading_mod, "_model_cache_cs", None)

    created = 0

    class FakeCriticalSection:
        def __init__(self):
            nonlocal created
            time.sleep(0.01)
            created += 1

    monkeypatch.setattr(threading_mod, "CriticalSection", FakeCriticalSection)

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = [pool.submit(safe_model_cache).result() for _ in range(64)]

    assert created == 1
    first = results[0]
    assert all(item is first for item in results)
