"""
Tests for fmtools.arrow_bridge — Arrow IPC Bridge.

Tests that require pyarrow are skipped when the package is not installed.
"""

from __future__ import annotations

import os
import sys

import pytest

from fmtools.arrow_bridge import (
    ArrowStreamWriter,
    from_arrow_ipc,
    from_arrow_ipc_buffer,
    from_polars,
    to_arrow_ipc,
    to_arrow_ipc_buffer,
    to_polars,
)

# ---------------------------------------------------------------------------
# Sample data used across tests
# ---------------------------------------------------------------------------

SAMPLE_DATA = [
    {"name": "Alice", "age": 30, "score": 95.5},
    {"name": "Bob", "age": 25, "score": 88.0},
    {"name": "Charlie", "age": 40, "score": 72.3},
]


# ---------------------------------------------------------------------------
# IPC file round-trip
# ---------------------------------------------------------------------------


class TestArrowIPCFile:
    def test_to_from_arrow_ipc_roundtrip(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "data.arrow"
        to_arrow_ipc(SAMPLE_DATA, path)
        result = from_arrow_ipc(path)

        assert len(result) == len(SAMPLE_DATA)
        for original, restored in zip(SAMPLE_DATA, result, strict=False):
            assert original["name"] == restored["name"]
            assert original["age"] == restored["age"]
            assert abs(original["score"] - restored["score"]) < 0.01

    def test_to_from_arrow_ipc_empty(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "empty.arrow"
        to_arrow_ipc([], path)
        result = from_arrow_ipc(path)
        assert result == []

    def test_to_arrow_ipc_uses_atomic_replace_in_same_directory(self, tmp_path, monkeypatch):
        pytest.importorskip("pyarrow")
        path = tmp_path / "atomic.arrow"
        replace_calls: list[tuple[str, str]] = []
        real_replace = os.replace

        def tracking_replace(src, dst):
            replace_calls.append((str(src), str(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr("fmtools.arrow_bridge.os.replace", tracking_replace)
        to_arrow_ipc([{"x": 1}], path)

        assert len(replace_calls) == 1
        src, dst = replace_calls[0]
        assert os.path.dirname(src) == str(path.parent)
        assert dst == str(path)

    def test_to_arrow_ipc_failure_does_not_clobber_existing_file(self, tmp_path, monkeypatch):
        pa = pytest.importorskip("pyarrow")
        path = tmp_path / "atomic_failure.arrow"
        original = [{"name": "before", "count": 1}]
        to_arrow_ipc(original, path)

        original_new_file = pa.ipc.new_file

        class FailingWriter:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def write_table(self, table):
                raise RuntimeError("simulated write failure")

            def close(self):
                self._wrapped.close()

        def failing_new_file(*args, **kwargs):
            return FailingWriter(original_new_file(*args, **kwargs))

        monkeypatch.setattr(pa.ipc, "new_file", failing_new_file)

        with pytest.raises(RuntimeError, match="simulated write failure"):
            to_arrow_ipc([{"name": "after", "count": 2}], path)

        assert from_arrow_ipc(path) == original
        assert list(tmp_path.glob(f".{path.name}.*")) == []


# ---------------------------------------------------------------------------
# IPC buffer round-trip
# ---------------------------------------------------------------------------


class TestArrowIPCBuffer:
    def test_to_from_arrow_ipc_buffer_roundtrip(self):
        pytest.importorskip("pyarrow")
        buf = to_arrow_ipc_buffer(SAMPLE_DATA)
        assert isinstance(buf, bytes)
        assert len(buf) > 0

        result = from_arrow_ipc_buffer(buf)
        assert len(result) == len(SAMPLE_DATA)
        for original, restored in zip(SAMPLE_DATA, result, strict=False):
            assert original["name"] == restored["name"]
            assert original["age"] == restored["age"]

    def test_buffer_roundtrip_string_only(self):
        pytest.importorskip("pyarrow")
        data = [{"key": "value1"}, {"key": "value2"}]
        buf = to_arrow_ipc_buffer(data)
        result = from_arrow_ipc_buffer(buf)
        assert result == data

    def test_buffer_roundtrip_mixed_types_falls_back_to_string(self):
        pytest.importorskip("pyarrow")
        data = [{"value": 1}, {"value": "two"}]
        buf = to_arrow_ipc_buffer(data)
        result = from_arrow_ipc_buffer(buf)
        assert result == [{"value": "1"}, {"value": "two"}]


# ---------------------------------------------------------------------------
# ArrowStreamWriter
# ---------------------------------------------------------------------------


class TestArrowStreamWriter:
    async def test_arrow_stream_writer(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        path = tmp_path / "stream.arrow"

        async with ArrowStreamWriter(path) as writer:
            await writer.write({"name": "Alice", "age": 30})
            await writer.write({"name": "Bob", "age": 25})

        # Read back via pyarrow directly to verify
        reader = pa.ipc.open_file(str(path))
        table = reader.read_all()
        assert table.num_rows == 2
        assert table.column("name").to_pylist() == ["Alice", "Bob"]
        assert table.column("age").to_pylist() == [30, 25]

    async def test_arrow_stream_writer_with_schema_hint(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        path = tmp_path / "hinted.arrow"

        async with ArrowStreamWriter(path, schema_hint={"name": "str", "value": "float"}) as w:
            await w.write({"name": "X", "value": 1.5})

        reader = pa.ipc.open_file(str(path))
        table = reader.read_all()
        assert table.num_rows == 1

    async def test_arrow_stream_writer_close_is_idempotent(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "idem.arrow"

        writer = ArrowStreamWriter(path)
        await writer.__aenter__()
        await writer.write({"x": 1})
        await writer.close()
        # Second close should not raise
        await writer.close()

    async def test_arrow_stream_writer_rejects_unexpected_keys_after_schema_inference(
        self, tmp_path
    ):
        pytest.importorskip("pyarrow")
        path = tmp_path / "unexpected_inferred.arrow"

        async with ArrowStreamWriter(path) as writer:
            await writer.write({"name": "Alice"})
            with pytest.raises(ValueError, match=r"unexpected keys.*age"):
                await writer.write({"name": "Bob", "age": 25})

    async def test_arrow_stream_writer_rejects_unexpected_keys_with_schema_hint(self, tmp_path):
        pytest.importorskip("pyarrow")
        path = tmp_path / "unexpected_hinted.arrow"

        async with ArrowStreamWriter(path, schema_hint={"name": "str"}) as writer:
            with pytest.raises(ValueError, match=r"unexpected keys.*age"):
                await writer.write({"name": "Alice", "age": 30})


# ---------------------------------------------------------------------------
# Polars convenience
# ---------------------------------------------------------------------------


class TestPolarsConvenience:
    def test_to_polars_convenience(self):
        import polars as pl

        df = to_polars(SAMPLE_DATA)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 3
        assert df.columns == ["name", "age", "score"]

    def test_from_polars_convenience(self):
        import polars as pl

        df = pl.DataFrame(SAMPLE_DATA)
        result = from_polars(df)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["name"] == "Alice"

    def test_to_from_polars_roundtrip(self):
        df = to_polars(SAMPLE_DATA)
        result = from_polars(df)
        for original, restored in zip(SAMPLE_DATA, result, strict=False):
            assert original["name"] == restored["name"]
            assert original["age"] == restored["age"]


# ---------------------------------------------------------------------------
# ImportError when pyarrow is missing
# ---------------------------------------------------------------------------


class TestImportErrorHandling:
    def test_import_error_without_pyarrow(self, monkeypatch):
        """Verify a clear ImportError when pyarrow is not available."""
        # Temporarily remove pyarrow from sys.modules and make import fail
        saved = sys.modules.get("pyarrow")
        monkeypatch.setitem(sys.modules, "pyarrow", None)

        # Re-import the helper function to trigger the lazy import path
        from fmtools.arrow_bridge import _require_pyarrow

        with pytest.raises(ImportError, match="Install pyarrow"):
            _require_pyarrow()

        # Restore
        if saved is not None:
            monkeypatch.setitem(sys.modules, "pyarrow", saved)
        else:
            monkeypatch.delitem(sys.modules, "pyarrow", raising=False)

    def test_to_arrow_ipc_raises_without_pyarrow(self, monkeypatch):
        """to_arrow_ipc should raise ImportError when pyarrow is absent."""
        monkeypatch.setitem(sys.modules, "pyarrow", None)

        with pytest.raises(ImportError, match="Install pyarrow"):
            to_arrow_ipc([{"a": 1}], "/tmp/test.arrow")

        monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
