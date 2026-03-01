"""
Runtime Diagnostics — performance counters and diagnostic reporting for
profiling FMTools extraction pipelines.

All facilities are stdlib-only (time, resource, threading, dataclasses, functools).
"""

from __future__ import annotations

import functools
import logging
import resource
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("fmtools")

F = TypeVar("F", bound="Callable[..., Any]")


# ----------------------------------------------------------------------
# Metrics dataclass
# ----------------------------------------------------------------------


@dataclass
class ExtractionMetrics:
    """Aggregated extraction performance counters."""

    total_extractions: int = 0
    total_time_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: int = 0

    @property
    def avg_time_per_extraction(self) -> float:
        """Average wall-clock time per extraction, or 0.0 if none recorded."""
        if self.total_extractions == 0:
            return 0.0
        return self.total_time_seconds / self.total_extractions

    @property
    def extractions_per_second(self) -> float:
        """Throughput in extractions per second, or 0.0 if no time elapsed."""
        if self.total_time_seconds == 0.0:
            return 0.0
        return self.total_extractions / self.total_time_seconds


# ----------------------------------------------------------------------
# DiagnosticCollector (singleton)
# ----------------------------------------------------------------------


class DiagnosticCollector:
    """
    Thread-safe, singleton performance collector.

    Use :func:`diagnostics` to obtain the global instance.
    """

    _instance: DiagnosticCollector | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> DiagnosticCollector:
        with cls._instance_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._init_state()
                cls._instance = inst
            return cls._instance

    # -- internal initialisation (called once) -------------------------

    def _init_state(self) -> None:
        self._lock = threading.Lock()
        self._total_extractions: int = 0
        self._total_time: float = 0.0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._errors: int = 0
        self._memory_peak_mb: float = 0.0

    # -- recording API -------------------------------------------------

    def record_extraction(self, elapsed: float, cached: bool, error: bool = False) -> None:
        """Record the outcome of a single extraction."""
        with self._lock:
            self._total_extractions += 1
            self._total_time += elapsed
            if cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            if error:
                self._errors += 1

    def record_memory(self) -> None:
        """Snapshot current peak memory via ``resource.getrusage``."""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in bytes on macOS, kilobytes on Linux
        if sys.platform == "darwin":
            peak_mb = usage.ru_maxrss / (1024 * 1024)
        else:
            peak_mb = usage.ru_maxrss / 1024
        with self._lock:
            if peak_mb > self._memory_peak_mb:
                self._memory_peak_mb = peak_mb

    # -- query API -----------------------------------------------------

    @property
    def metrics(self) -> ExtractionMetrics:
        """Return a snapshot of current metrics."""
        with self._lock:
            return ExtractionMetrics(
                total_extractions=self._total_extractions,
                total_time_seconds=self._total_time,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                errors=self._errors,
            )

    @property
    def memory_peak_mb(self) -> float:
        """Peak RSS in megabytes as last recorded by :meth:`record_memory`."""
        with self._lock:
            return self._memory_peak_mb

    # -- lifecycle -----------------------------------------------------

    def reset(self) -> None:
        """Clear all counters and memory tracking."""
        with self._lock:
            self._total_extractions = 0
            self._total_time = 0.0
            self._cache_hits = 0
            self._cache_misses = 0
            self._errors = 0
            self._memory_peak_mb = 0.0

    # -- reporting -----------------------------------------------------

    def report(self) -> str:
        """Return a human-readable diagnostic report string."""
        m = self.metrics
        lines = [
            "FMTools Diagnostic Report",
            "=" * 40,
            f"Total extractions : {m.total_extractions}",
            f"Total time        : {m.total_time_seconds:.3f}s",
            f"Avg time/extract  : {m.avg_time_per_extraction:.3f}s",
            f"Extractions/sec   : {m.extractions_per_second:.2f}",
            f"Cache hits        : {m.cache_hits}",
            f"Cache misses      : {m.cache_misses}",
            f"Errors            : {m.errors}",
            f"Peak memory       : {self.memory_peak_mb:.1f} MB",
        ]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Module-level convenience
# ----------------------------------------------------------------------


def diagnostics() -> DiagnosticCollector:
    """Return the global :class:`DiagnosticCollector` singleton."""
    return DiagnosticCollector()


# ----------------------------------------------------------------------
# @diagnose decorator
# ----------------------------------------------------------------------


def diagnose(func: F) -> F:
    """
    Decorator that wraps an *async* function to automatically record
    timing and error diagnostics in the global :class:`DiagnosticCollector`.

    Usage::

        @diagnose
        async def my_extraction(text):
            ...
    """
    collector = diagnostics()

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        error_occurred = False
        try:
            return await func(*args, **kwargs)
        except Exception:
            error_occurred = True
            raise
        finally:
            elapsed = time.perf_counter() - start
            collector.record_extraction(elapsed=elapsed, cached=False, error=error_occurred)

    return wrapper  # type: ignore[return-value]
