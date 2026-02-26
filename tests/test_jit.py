"""
Tests for silicon_refinery._jit — Runtime Diagnostics.

Covers:
  - Recording extraction metrics
  - Computed properties (avg_time, throughput)
  - Reset behaviour
  - Report formatting
  - Thread-safety of concurrent recordings
  - @diagnose decorator timing
  - @diagnose decorator error recording
  - diagnostics() singleton identity
  - Memory tracking via resource.getrusage
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from silicon_refinery._jit import (
    DiagnosticCollector,
    diagnose,
    diagnostics,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture(autouse=True)
def _reset_collector():
    """Ensure the singleton collector is clean before each test."""
    diagnostics().reset()
    yield
    diagnostics().reset()


# =====================================================================
# test_diagnostic_collector_records_extraction
# =====================================================================


class TestDiagnosticCollectorRecordsExtraction:
    def test_single_extraction_recorded(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=0.5, cached=False)

        m = dc.metrics
        assert m.total_extractions == 1
        assert m.total_time_seconds == pytest.approx(0.5)
        assert m.cache_misses == 1
        assert m.cache_hits == 0
        assert m.errors == 0

    def test_cached_extraction_recorded(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=0.01, cached=True)

        m = dc.metrics
        assert m.cache_hits == 1
        assert m.cache_misses == 0

    def test_error_extraction_recorded(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=0.1, cached=False, error=True)

        m = dc.metrics
        assert m.errors == 1
        assert m.total_extractions == 1


# =====================================================================
# test_diagnostic_collector_metrics_computation
# =====================================================================


class TestDiagnosticCollectorMetricsComputation:
    def test_avg_time_per_extraction(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=1.0, cached=False)
        dc.record_extraction(elapsed=3.0, cached=False)

        assert dc.metrics.avg_time_per_extraction == pytest.approx(2.0)

    def test_extractions_per_second(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=0.5, cached=False)
        dc.record_extraction(elapsed=0.5, cached=False)

        # 2 extractions in 1.0 s total -> 2.0 eps
        assert dc.metrics.extractions_per_second == pytest.approx(2.0)

    def test_zero_extractions_return_zero(self) -> None:
        m = diagnostics().metrics
        assert m.avg_time_per_extraction == 0.0
        assert m.extractions_per_second == 0.0


# =====================================================================
# test_diagnostic_collector_reset
# =====================================================================


class TestDiagnosticCollectorReset:
    def test_reset_clears_all_counters(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=1.0, cached=True, error=True)
        dc.record_memory()

        dc.reset()

        m = dc.metrics
        assert m.total_extractions == 0
        assert m.total_time_seconds == 0.0
        assert m.cache_hits == 0
        assert m.cache_misses == 0
        assert m.errors == 0
        assert dc.memory_peak_mb == 0.0


# =====================================================================
# test_diagnostic_collector_report_format
# =====================================================================


class TestDiagnosticCollectorReportFormat:
    def test_report_contains_expected_sections(self) -> None:
        dc = diagnostics()
        dc.record_extraction(elapsed=1.0, cached=False)
        dc.record_extraction(elapsed=2.0, cached=True, error=True)
        dc.record_memory()

        report = dc.report()

        assert "SiliconRefinery Diagnostic Report" in report
        assert "Total extractions : 2" in report
        assert "Cache hits        : 1" in report
        assert "Cache misses      : 1" in report
        assert "Errors            : 1" in report
        assert "Peak memory" in report
        assert "Avg time/extract" in report
        assert "Extractions/sec" in report


# =====================================================================
# test_diagnostic_collector_thread_safety
# =====================================================================


class TestDiagnosticCollectorThreadSafety:
    def test_concurrent_recordings_are_consistent(self) -> None:
        """Many threads recording simultaneously should not lose data."""
        dc = diagnostics()
        n_threads = 10
        recordings_per_thread = 100

        def _worker() -> None:
            for _ in range(recordings_per_thread):
                dc.record_extraction(elapsed=0.001, cached=False)

        threads = [threading.Thread(target=_worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        m = dc.metrics
        expected = n_threads * recordings_per_thread
        assert m.total_extractions == expected
        assert m.total_time_seconds == pytest.approx(expected * 0.001)


# =====================================================================
# test_diagnose_decorator_records_timing
# =====================================================================


class TestDiagnoseDecoratorRecordsTiming:
    async def test_successful_call_is_recorded(self) -> None:
        @diagnose
        async def dummy(x: int) -> int:
            return x * 2

        result = await dummy(5)

        assert result == 10
        m = diagnostics().metrics
        assert m.total_extractions == 1
        assert m.errors == 0
        assert m.total_time_seconds > 0.0

    async def test_preserves_function_metadata(self) -> None:
        @diagnose
        async def my_func():
            """Docstring."""

        assert my_func.__name__ == "my_func"
        assert my_func.__doc__ == "Docstring."


# =====================================================================
# test_diagnose_decorator_records_errors
# =====================================================================


class TestDiagnoseDecoratorRecordsErrors:
    async def test_exception_is_recorded_and_reraised(self) -> None:
        @diagnose
        async def failing():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await failing()

        m = diagnostics().metrics
        assert m.total_extractions == 1
        assert m.errors == 1
        assert m.total_time_seconds > 0.0

    async def test_cancelled_error_is_not_counted_as_error(self) -> None:
        @diagnose
        async def cancelled():
            raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await cancelled()

        m = diagnostics().metrics
        assert m.total_extractions == 1
        assert m.errors == 0


# =====================================================================
# test_diagnostics_singleton
# =====================================================================


class TestDiagnosticsSingleton:
    def test_returns_same_instance(self) -> None:
        a = diagnostics()
        b = diagnostics()
        assert a is b

    def test_constructor_returns_same_instance(self) -> None:
        a = DiagnosticCollector()
        b = DiagnosticCollector()
        assert a is b


# =====================================================================
# test_memory_tracking
# =====================================================================


class TestMemoryTracking:
    def test_record_memory_sets_peak(self) -> None:
        dc = diagnostics()
        dc.record_memory()
        # Any running Python process should use > 0 MB
        assert dc.memory_peak_mb > 0.0

    def test_memory_peak_survives_reset(self) -> None:
        """After reset, memory_peak_mb is zeroed and can be re-recorded."""
        dc = diagnostics()
        dc.record_memory()
        assert dc.memory_peak_mb > 0.0

        dc.reset()
        assert dc.memory_peak_mb == 0.0

        dc.record_memory()
        assert dc.memory_peak_mb > 0.0
