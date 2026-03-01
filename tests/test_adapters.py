"""
Tests for fmtools.adapters — IO Protocol Wrappers.

Covers: FileAdapter, CSVAdapter, JSONLAdapter, IterableAdapter,
        TextChunkAdapter, StdinAdapter, and the DataAdapter protocol.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from fmtools.adapters import (
    CSVAdapter,
    DataAdapter,
    FileAdapter,
    IterableAdapter,
    JSONLAdapter,
    StdinAdapter,
    TextChunkAdapter,
    TrioAdapter,
)

# ========================================================================
# FileAdapter
# ========================================================================


class TestFileAdapter:
    async def test_file_adapter_reads_lines(self, tmp_path):
        """FileAdapter yields each line from the file."""
        p = tmp_path / "sample.txt"
        p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

        adapter = FileAdapter(p)
        lines = [line async for line in adapter]

        assert lines == ["alpha", "beta", "gamma"]

    async def test_file_adapter_custom_encoding(self, tmp_path):
        """FileAdapter respects the encoding parameter."""
        p = tmp_path / "latin.txt"
        p.write_bytes("caf\xe9\n".encode("latin-1"))

        adapter = FileAdapter(p, encoding="latin-1")
        lines = [line async for line in adapter]

        assert lines[0] == "caf\xe9"

    async def test_file_adapter_repr(self, tmp_path):
        p = tmp_path / "x.txt"
        adapter = FileAdapter(p, encoding="ascii")
        r = repr(adapter)
        assert "FileAdapter" in r
        assert "ascii" in r

    async def test_file_adapter_incremental_reads(self, monkeypatch, tmp_path):
        p = tmp_path / "many.txt"
        p.write_text("".join(f"line{i}\n" for i in range(100)), encoding="utf-8")

        adapter = FileAdapter(p, batch_size=10)
        original = adapter._read_batch
        calls = 0

        def counted_read_batch(fh):
            nonlocal calls
            calls += 1
            return original(fh)

        monkeypatch.setattr(adapter, "_read_batch", counted_read_batch)

        out = []
        async for line in adapter:
            out.append(line)
            if len(out) == 5:
                break

        assert out == ["line0", "line1", "line2", "line3", "line4"]
        assert calls == 1


# ========================================================================
# CSVAdapter
# ========================================================================


class TestCSVAdapter:
    async def test_csv_adapter_yields_json_rows(self, tmp_path):
        """Each CSV row is emitted as a JSON string."""
        p = tmp_path / "data.csv"
        p.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

        adapter = CSVAdapter(p)
        rows = [json.loads(r) async for r in adapter]

        assert rows == [
            {"name": "Alice", "age": "30"},
            {"name": "Bob", "age": "25"},
        ]

    async def test_csv_adapter_single_column(self, tmp_path):
        """When column is given, only that column's value is yielded."""
        p = tmp_path / "data.csv"
        p.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

        adapter = CSVAdapter(p, column="name")
        names = [n async for n in adapter]

        assert names == ["Alice", "Bob"]

    async def test_csv_adapter_repr(self, tmp_path):
        adapter = CSVAdapter(tmp_path / "x.csv", column="col")
        r = repr(adapter)
        assert "CSVAdapter" in r
        assert "col" in r

    def test_csv_adapter_batch_size_validation(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            CSVAdapter(tmp_path / "x.csv", batch_size=0)


# ========================================================================
# JSONLAdapter
# ========================================================================


class TestJSONLAdapter:
    async def test_jsonl_adapter_yields_parsed_lines(self, tmp_path):
        """JSONLAdapter parses each line and re-serializes it."""
        p = tmp_path / "data.jsonl"
        content = '{"k": "v1"}\n{"k": "v2"}\n'
        p.write_text(content, encoding="utf-8")

        adapter = JSONLAdapter(p)
        rows = [json.loads(r) async for r in adapter]

        assert rows == [{"k": "v1"}, {"k": "v2"}]

    async def test_jsonl_adapter_skips_blank_lines(self, tmp_path):
        p = tmp_path / "data.jsonl"
        content = '{"a": 1}\n\n{"b": 2}\n'
        p.write_text(content, encoding="utf-8")

        adapter = JSONLAdapter(p)
        rows = [r async for r in adapter]

        assert len(rows) == 2

    async def test_jsonl_adapter_repr(self, tmp_path):
        adapter = JSONLAdapter(tmp_path / "x.jsonl")
        assert "JSONLAdapter" in repr(adapter)

    def test_jsonl_adapter_batch_size_validation(self, tmp_path):
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            JSONLAdapter(tmp_path / "x.jsonl", batch_size=0)


# ========================================================================
# IterableAdapter
# ========================================================================


class TestIterableAdapter:
    async def test_iterable_adapter_wraps_sync_iterable(self):
        """IterableAdapter makes a sync iterable async."""
        adapter = IterableAdapter(["one", "two", "three"])
        items = [item async for item in adapter]
        assert items == ["one", "two", "three"]

    async def test_iterable_adapter_with_generator(self):
        def gen():
            yield "a"
            yield "b"

        adapter = IterableAdapter(gen())
        items = [item async for item in adapter]
        assert items == ["a", "b"]

    async def test_iterable_adapter_yields_control_periodically(self):
        adapter = IterableAdapter([str(i) for i in range(300)])
        real_sleep = asyncio.sleep

        with patch(
            "fmtools.adapters.asyncio.sleep",
            new=AsyncMock(side_effect=real_sleep),
        ) as sleep_mock:
            items = [item async for item in adapter]

        assert len(items) == 300
        sleep_mock.assert_any_await(0)

    async def test_iterable_adapter_repr(self):
        adapter = IterableAdapter(["x"])
        assert "IterableAdapter" in repr(adapter)


# ========================================================================
# TextChunkAdapter
# ========================================================================


class TestTextChunkAdapter:
    async def test_text_chunk_adapter_rechunks_with_overlap(self):
        """Windows overlap by the configured amount."""
        # 20 chars of text, window=10, overlap=3 => step=7
        # Window 0: [0:10]  Window 1: [7:17]  Window 2: [14:24] (clamped to 20)
        source = IterableAdapter(["abcdefghijklmnopqrst"])
        adapter = TextChunkAdapter(source, window_size=10, overlap=3)
        chunks = [c async for c in adapter]

        assert chunks[0] == "abcdefghij"
        assert chunks[1] == "hijklmnopq"
        assert chunks[2] == "opqrst"
        assert len(chunks) == 3

        # Verify overlap: last 3 of chunk0 == first 3 of chunk1
        assert chunks[0][-3:] == chunks[1][:3]

    async def test_text_chunk_adapter_small_input(self):
        """Input smaller than window_size yields a single chunk."""
        source = IterableAdapter(["hi"])
        adapter = TextChunkAdapter(source, window_size=4096, overlap=256)
        chunks = [c async for c in adapter]
        assert chunks == ["hi"]

    async def test_text_chunk_adapter_empty_source(self):
        """Empty source yields nothing."""
        source = IterableAdapter([])
        adapter = TextChunkAdapter(source, window_size=10, overlap=2)
        chunks = [c async for c in adapter]
        assert chunks == []

    async def test_text_chunk_adapter_repr(self):
        source = IterableAdapter(["x"])
        adapter = TextChunkAdapter(source, window_size=100, overlap=10)
        r = repr(adapter)
        assert "TextChunkAdapter" in r
        assert "100" in r
        assert "10" in r

    def test_text_chunk_adapter_validates_window_and_overlap(self):
        source = IterableAdapter(["x"])
        with pytest.raises(ValueError, match="window_size must be > 0"):
            TextChunkAdapter(source, window_size=0, overlap=0)
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            TextChunkAdapter(source, window_size=10, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be < window_size"):
            TextChunkAdapter(source, window_size=10, overlap=10)


# ========================================================================
# StdinAdapter
# ========================================================================


class TestStdinAdapter:
    def test_stdin_adapter_exists(self):
        """StdinAdapter can be instantiated without error."""
        adapter = StdinAdapter()
        assert adapter is not None

    def test_stdin_adapter_repr(self):
        adapter = StdinAdapter()
        assert repr(adapter) == "StdinAdapter()"

    def test_stdin_adapter_has_aiter(self):
        adapter = StdinAdapter()
        assert hasattr(adapter, "__aiter__")


# ========================================================================
# Protocol conformance
# ========================================================================


class TestProtocol:
    def test_adapters_implement_async_iteration(self, tmp_path):
        """All concrete adapters satisfy the DataAdapter protocol."""
        f = tmp_path / "dummy.txt"
        f.write_text("x\n")

        adapters = [
            FileAdapter(f),
            StdinAdapter(),
            CSVAdapter(f),
            JSONLAdapter(f),
            IterableAdapter(["x"]),
            TrioAdapter(IterableAdapter(["x"])),
            TextChunkAdapter(IterableAdapter(["x"])),
        ]
        for adapter in adapters:
            assert isinstance(adapter, DataAdapter), f"{type(adapter).__name__} not a DataAdapter"
            assert hasattr(adapter, "__aiter__")


class TestTrioAdapter:
    async def test_trio_adapter_from_receive_channel(self, monkeypatch):
        class EndOfChannel(Exception):
            pass

        class FakeChannel:
            def __init__(self):
                self._values = ["a", "b"]

            async def receive(self):
                if not self._values:
                    raise EndOfChannel
                return self._values.pop(0)

        monkeypatch.setitem(sys.modules, "trio", SimpleNamespace(EndOfChannel=EndOfChannel))

        adapter = TrioAdapter(FakeChannel())
        items = [item async for item in adapter]
        assert items == ["a", "b"]

    async def test_trio_adapter_from_async_iterable(self):
        async def _source():
            yield "x"
            yield "y"

        adapter = TrioAdapter(_source())
        items = [item async for item in adapter]
        assert items == ["x", "y"]

    async def test_trio_adapter_invalid_source_raises(self):
        adapter = TrioAdapter(object())
        with pytest.raises(TypeError, match="must be an async iterable or expose async receive"):
            _ = [item async for item in adapter]
