"""
Free-threading (PEP 703 / nogil) support utilities.

Provides thread-safe primitives that **adapt** to the runtime:

* On Python 3.13+ free-threaded builds the primitives use real locks.
* On regular GIL builds every lock is a zero-cost no-op so there is no
  overhead for the common case.
"""

from __future__ import annotations

import asyncio
import enum
import sys
import threading
from typing import Any

# ---------------------------------------------------------------------------
# GIL detection
# ---------------------------------------------------------------------------


def is_free_threaded() -> bool:
    """Return *True* if the interpreter is a free-threaded (nogil) build.

    On Python 3.13+ this checks ``sys._is_gil_enabled()``.  On older
    versions the GIL is always present so the function returns *False*.
    """
    _is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if _is_gil_enabled is not None:
        return not _is_gil_enabled()
    return False


class GILStatus(enum.Enum):
    """Describes the current GIL state of the interpreter."""

    ENABLED = "enabled"
    DISABLED = "disabled"
    UNKNOWN = "unknown"


def get_gil_status() -> GILStatus:
    """Return the current :class:`GILStatus`."""
    _is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    if _is_gil_enabled is None:
        # Python < 3.13 -- the GIL is always enabled but the runtime
        # doesn't expose the helper.
        if sys.version_info >= (3, 13):
            return GILStatus.UNKNOWN
        return GILStatus.ENABLED
    return GILStatus.DISABLED if not _is_gil_enabled() else GILStatus.ENABLED


# ---------------------------------------------------------------------------
# CriticalSection
# ---------------------------------------------------------------------------


class _NoOpLock:
    """A lock-alike that does absolutely nothing (zero overhead on GIL builds)."""

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        return True

    def release(self) -> None:
        pass

    def __enter__(self) -> bool:
        return True

    def __exit__(self, *args: object) -> None:
        pass

    async def __aenter__(self) -> bool:
        return True

    async def __aexit__(self, *args: object) -> None:
        pass


class CriticalSection:
    """A context manager that is a real lock on nogil builds, a no-op otherwise.

    Supports both synchronous (``with``) and asynchronous (``async with``)
    usage.
    """

    def __init__(self) -> None:
        self._free_threaded = is_free_threaded()
        self._lock: threading.Lock | _NoOpLock = (
            threading.Lock() if self._free_threaded else _NoOpLock()
        )

    @property
    def is_real_lock(self) -> bool:
        """Return *True* when backed by a real :class:`threading.Lock`."""
        return self._free_threaded

    # Sync protocol
    def __enter__(self) -> CriticalSection:
        self._lock.acquire()
        return self

    def __exit__(self, *args: object) -> None:
        self._lock.release()

    # Async protocol
    async def __aenter__(self) -> CriticalSection:
        if self._free_threaded:
            await asyncio.to_thread(self._lock.acquire)
        else:
            self._lock.acquire()
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._free_threaded:
            await asyncio.to_thread(self._lock.release)
        else:
            self._lock.release()


# ---------------------------------------------------------------------------
# AtomicCounter
# ---------------------------------------------------------------------------


class AtomicCounter:
    """Thread-safe integer counter.

    Uses a real lock only on free-threaded builds; on GIL builds the
    counter is just a plain ``int`` with no synchronisation overhead.
    """

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._cs = CriticalSection()

    def increment(self, n: int = 1) -> int:
        """Add *n* to the counter and return the new value."""
        with self._cs:
            self._value += n
            return self._value

    def decrement(self, n: int = 1) -> int:
        """Subtract *n* from the counter and return the new value."""
        with self._cs:
            self._value -= n
            return self._value

    @property
    def value(self) -> int:
        """Current counter value."""
        with self._cs:
            return self._value


# ---------------------------------------------------------------------------
# ThreadSafeDict
# ---------------------------------------------------------------------------


class ThreadSafeDict:
    """A dict wrapper guarded by :class:`CriticalSection`.

    On GIL builds this simply delegates to a regular ``dict`` with zero
    lock overhead.
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._cs = CriticalSection()

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default*."""
        with self._cs:
            return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set *key* to *value*."""
        with self._cs:
            self._data[key] = value

    def delete(self, key: str) -> None:
        """Delete *key*.  Raises :class:`KeyError` if missing."""
        with self._cs:
            del self._data[key]

    def items(self) -> list[tuple[str, Any]]:
        """Return a snapshot of all items as a list of (key, value) tuples."""
        with self._cs:
            return list(self._data.items())

    def __len__(self) -> int:
        with self._cs:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        with self._cs:
            return key in self._data


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_model_cache_cs: CriticalSection | None = None
_model_cache_init_lock = threading.Lock()


def safe_model_cache() -> CriticalSection:
    """Return a module-level :class:`CriticalSection` for guarding FM model creation.

    The instance is created lazily on first call and reused thereafter.
    """
    global _model_cache_cs
    if _model_cache_cs is None:
        with _model_cache_init_lock:
            if _model_cache_cs is None:
                _model_cache_cs = CriticalSection()
    return _model_cache_cs
