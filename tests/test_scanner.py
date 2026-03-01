"""
Tests for fmtools.scanner — mmap Sliding Window Scanner.

Covers: MMapScanner window iteration, overlap, empty/small files,
        UTF-8 boundary handling, context manager, and line_split_scanner.
"""

from __future__ import annotations

import pytest

import fmtools.scanner as scanner_mod
from fmtools.scanner import MMapScanner, line_split_scanner

# ========================================================================
# MMapScanner — basic reading
# ========================================================================


class TestMMapScannerBasic:
    async def test_mmap_scanner_reads_small_file(self, tmp_path):
        """A file smaller than window_size is yielded in a single chunk."""
        p = tmp_path / "small.txt"
        p.write_text("hello world", encoding="utf-8")

        scanner = MMapScanner(p, window_size=1_048_576)
        chunks = [c async for c in scanner]

        assert len(chunks) == 1
        assert chunks[0] == "hello world"

    async def test_mmap_scanner_reads_large_file_in_windows(self, tmp_path):
        """A file larger than window_size produces multiple chunks."""
        p = tmp_path / "big.txt"
        # 100 bytes of data, window=30, overlap=5 => step=25
        data = "A" * 100
        p.write_bytes(data.encode("utf-8"))

        scanner = MMapScanner(p, window_size=30, overlap=5)
        chunks = [c async for c in scanner]

        assert len(chunks) > 1
        # Reassemble (accounting for overlap) should recover the original
        # The first chunk + unique part of subsequent chunks = original
        reassembled = chunks[0]
        for chunk in chunks[1:]:
            reassembled += chunk[5:]  # skip the overlapping prefix
        assert reassembled == data

    async def test_mmap_scanner_overlap_preserves_boundary_data(self, tmp_path):
        """The overlap region appears in both adjacent windows."""
        p = tmp_path / "overlap.txt"
        data = "0123456789" * 3  # 30 bytes
        p.write_bytes(data.encode("utf-8"))

        scanner = MMapScanner(p, window_size=15, overlap=5)
        chunks = [c async for c in scanner]

        assert len(chunks) >= 2
        # Last 5 chars of first chunk == first 5 chars of second chunk
        assert chunks[0][-5:] == chunks[1][:5]

    async def test_mmap_scanner_empty_file(self, tmp_path):
        """An empty file yields nothing."""
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")

        scanner = MMapScanner(p, window_size=1024)
        chunks = [c async for c in scanner]

        assert chunks == []

    def test_mmap_scanner_validates_window_and_overlap(self, tmp_path):
        p = tmp_path / "sample.txt"
        p.write_text("abc", encoding="utf-8")

        with pytest.raises(ValueError, match="window_size must be > 0"):
            MMapScanner(p, window_size=0, overlap=0)
        with pytest.raises(ValueError, match="overlap must be >= 0"):
            MMapScanner(p, window_size=10, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be < window_size"):
            MMapScanner(p, window_size=10, overlap=10)

    async def test_mmap_scanner_incremental_batches(self, monkeypatch, tmp_path):
        p = tmp_path / "streaming.txt"
        p.write_text("A" * 10_000, encoding="utf-8")

        calls = 0
        original = scanner_mod._MMapScanState.read_batch

        def counted_read_batch(self, batch_size):
            nonlocal calls
            calls += 1
            return original(self, batch_size)

        monkeypatch.setattr(scanner_mod._MMapScanState, "read_batch", counted_read_batch)
        monkeypatch.setattr(scanner_mod, "_BATCH_CHUNKS", 1)

        scanner = MMapScanner(p, window_size=128, overlap=16)
        seen = []
        async for chunk in scanner:
            seen.append(chunk)
            if len(seen) == 2:
                break

        assert len(seen) == 2
        assert calls == 2


# ========================================================================
# MMapScanner — context manager
# ========================================================================


class TestMMapScannerContextManager:
    async def test_mmap_scanner_context_manager(self, tmp_path):
        """Scanner works as an async context manager."""
        p = tmp_path / "ctx.txt"
        p.write_text("context test", encoding="utf-8")

        async with MMapScanner(p) as scanner:
            chunks = [c async for c in scanner]

        assert len(chunks) == 1
        assert chunks[0] == "context test"

    async def test_mmap_scanner_context_manager_closes(self, tmp_path):
        """__aexit__ calls close()."""
        p = tmp_path / "ctx2.txt"
        p.write_text("x", encoding="utf-8")

        scanner = MMapScanner(p)
        async with scanner:
            pass
        # After exit, internal state should be cleaned
        assert scanner._mm is None
        assert scanner._fd is None


# ========================================================================
# MMapScanner — UTF-8 boundary handling
# ========================================================================


class TestMMapScannerUTF8:
    async def test_mmap_scanner_binary_boundary_handling(self, tmp_path):
        """Multi-byte UTF-8 characters at window boundaries decode cleanly."""
        p = tmp_path / "utf8.txt"
        # Each emoji is 4 bytes in UTF-8
        # Build a string of 4-byte chars so boundaries will split them
        emojis = "\U0001f600" * 20  # 80 bytes total
        p.write_bytes(emojis.encode("utf-8"))

        scanner = MMapScanner(p, window_size=30, overlap=8)
        chunks: list[str] = []
        async for chunk in scanner:
            chunks.append(chunk)

        assert len(chunks) > 1
        # Every chunk should be valid unicode (no decode errors)
        full = "".join(chunks)
        # The full text (with overlaps) should contain all emojis
        assert "\U0001f600" in full

    async def test_mmap_scanner_two_byte_boundary(self, tmp_path):
        """2-byte UTF-8 characters at boundary are handled correctly."""
        p = tmp_path / "utf8_2byte.txt"
        # \u00e9 = 'e-acute', 2 bytes in UTF-8
        text = "\u00e9" * 30  # 60 bytes
        p.write_bytes(text.encode("utf-8"))

        scanner = MMapScanner(p, window_size=15, overlap=4)
        chunks = [c async for c in scanner]

        assert len(chunks) > 1
        for chunk in chunks:
            # All chunks should be valid strings (no UnicodeDecodeError)
            assert isinstance(chunk, str)


class TestMMapScannerNonUTF8:
    async def test_mmap_scanner_non_utf8_does_not_drop_high_bytes(self, tmp_path):
        p = tmp_path / "latin1.bin"
        raw = bytes([0x41, 0x80, 0x81, 0xBF, 0x42, 0x80, 0x43, 0xBF, 0x44, 0x45])
        p.write_bytes(raw)

        scanner = MMapScanner(p, window_size=4, overlap=1, encoding="latin-1")
        chunks = [c async for c in scanner]

        reassembled = chunks[0]
        for chunk in chunks[1:]:
            reassembled += chunk[1:]

        assert reassembled == raw.decode("latin-1")

    def test_mmap_scan_state_closes_fd_if_mmap_init_fails(self, monkeypatch, tmp_path):
        p = tmp_path / "boom.txt"
        p.write_bytes(b"abc")

        opened_fds: list[int] = []
        closed_fds: list[int] = []

        def fake_open(path, flags):
            del path, flags
            fd = 12345
            opened_fds.append(fd)
            return fd

        def fake_close(fd):
            closed_fds.append(fd)

        def fail_mmap(fd, length, access):
            del fd, length, access
            raise OSError("boom")

        monkeypatch.setattr(scanner_mod.os, "open", fake_open)
        monkeypatch.setattr(scanner_mod.os, "close", fake_close)
        monkeypatch.setattr(scanner_mod.mmap, "mmap", fail_mmap)

        with pytest.raises(OSError, match="boom"):
            scanner_mod._MMapScanState(
                p,
                window_size=4,
                overlap=1,
                encoding="utf-8",
            )

        assert opened_fds == [12345]
        assert closed_fds == [12345]


# ========================================================================
# MMapScanner — repr
# ========================================================================


class TestMMapScannerRepr:
    def test_mmap_scanner_repr(self, tmp_path):
        p = tmp_path / "repr.txt"
        scanner = MMapScanner(p, window_size=512, overlap=64, encoding="latin-1")
        r = repr(scanner)
        assert "MMapScanner" in r
        assert "512" in r
        assert "64" in r
        assert "latin-1" in r


# ========================================================================
# line_split_scanner
# ========================================================================


class TestLineSplitScanner:
    async def test_line_split_scanner_yields_complete_lines(self, tmp_path):
        """line_split_scanner yields whole lines, never breaking mid-line."""
        p = tmp_path / "log.txt"
        p.write_text("line one\nline two\nline three\n", encoding="utf-8")

        scanner = line_split_scanner(p)
        lines = [line async for line in scanner]

        assert lines == ["line one", "line two", "line three"]

    async def test_line_split_scanner_empty_file(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_bytes(b"")

        scanner = line_split_scanner(p)
        lines = [line async for line in scanner]
        assert lines == []

    async def test_line_split_scanner_preserves_blank_lines(self, tmp_path):
        p = tmp_path / "blank.log"
        p.write_text("a\n\nb\n", encoding="utf-8")

        scanner = line_split_scanner(p)
        lines = [line async for line in scanner]
        assert lines == ["a", "", "b"]

    async def test_line_split_scanner_repr(self, tmp_path):
        p = tmp_path / "x.log"
        scanner = line_split_scanner(p)
        r = repr(scanner)
        assert "LineSplitScanner" in r

    async def test_line_split_scanner_reads_incrementally(self, monkeypatch, tmp_path):
        p = tmp_path / "many.log"
        p.write_text("".join(f"line-{i}\n" for i in range(200)), encoding="utf-8")

        calls = 0
        original = scanner_mod._LineSplitScanner._read_batch

        def counted_read_batch(self, fh):
            nonlocal calls
            calls += 1
            return original(self, fh)

        monkeypatch.setattr(scanner_mod._LineSplitScanner, "_read_batch", counted_read_batch)

        scanner = line_split_scanner(p, batch_size=5)
        seen = []
        async for line in scanner:
            seen.append(line)
            if len(seen) == 12:
                break

        assert seen[0] == "line-0"
        assert len(seen) == 12
        assert calls == 3

    def test_line_split_scanner_validates_batch_size(self, tmp_path):
        p = tmp_path / "x.log"
        p.write_text("line\n", encoding="utf-8")
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            line_split_scanner(p, batch_size=0)
