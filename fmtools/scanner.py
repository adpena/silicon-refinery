"""
mmap Sliding Window Scanner — memory-mapped file scanning for multi-GB files.

Uses ``mmap`` to slide a window across a file without loading it entirely into
memory.  Each window is decoded to text and yielded via ``__aiter__``.

The configurable *overlap* ensures that entities (or any patterns) straddling
a window boundary are never split.

Stdlib-only: mmap, os, pathlib, asyncio.
"""

from __future__ import annotations

import asyncio
import logging
import mmap
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType

logger = logging.getLogger("fmtools")

_BATCH_CHUNKS = 64


class _MMapScanState:
    """Stateful mmap scanner used to stream batches from an executor thread."""

    def __init__(
        self,
        path: Path,
        window_size: int,
        overlap: int,
        encoding: str,
    ) -> None:
        self.path = path
        self.window_size = window_size
        self.overlap = overlap
        self.encoding = encoding
        self.file_size = os.path.getsize(path)
        self.pos = 0
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None

        if self.file_size == 0:
            return

        self.fd = os.open(str(path), os.O_RDONLY)
        try:
            self.mm = mmap.mmap(self.fd, 0, access=mmap.ACCESS_READ)
        except Exception:
            os.close(self.fd)
            self.fd = None
            raise

    def read_batch(self, batch_size: int) -> list[tuple[int, int, bytes]]:
        if self.mm is None or self.file_size == 0 or self.pos >= self.file_size:
            return []

        step = self.window_size - self.overlap
        batch: list[tuple[int, int, bytes]] = []

        while len(batch) < batch_size and self.pos < self.file_size:
            start = self.pos
            end = min(start + self.window_size, self.file_size)
            batch.append((start, end, self.mm[start:end]))
            self.pos += step

        return batch

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


class MMapScanner:
    """Async iterator over mmap windows of a file.

    Parameters
    ----------
    path:
        Path to the file to scan.
    window_size:
        Size of each window in bytes (default 1 MB).
    overlap:
        Number of bytes to overlap between consecutive windows (default 256).
    encoding:
        Text encoding for decoding raw bytes (default ``utf-8``).
    """

    def __init__(
        self,
        path: Union[str, Path],
        window_size: int = 1_048_576,
        overlap: int = 256,
        encoding: str = "utf-8",
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= window_size:
            raise ValueError("overlap must be < window_size")

        self.path = Path(path)
        self.window_size = window_size
        self.overlap = overlap
        self.encoding = encoding
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None
        self._file_size: int = 0

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MMapScanner:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Release the mmap and file descriptor (if open)."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    # ------------------------------------------------------------------
    # Async iteration
    # ------------------------------------------------------------------

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        encoding_key = self.encoding.lower().replace("_", "-")
        apply_utf8_boundary_fixups = encoding_key == "utf-8"
        state = await loop.run_in_executor(
            None,
            _MMapScanState,
            self.path,
            self.window_size,
            self.overlap,
            self.encoding,
        )
        try:
            while True:
                raw_batch = await loop.run_in_executor(None, state.read_batch, _BATCH_CHUNKS)
                if not raw_batch:
                    break

                for start, end, raw in raw_batch:
                    if apply_utf8_boundary_fixups:
                        # Skip leading continuation bytes when we are not at the start of
                        # the file — these belong to a character completed in the previous
                        # (overlapping) window.
                        if start > 0:
                            raw = self._skip_leading_continuation_bytes(raw)
                        # Trim trailing incomplete multi-byte sequences so the next
                        # overlapping window can decode them fully.
                        if end < state.file_size:
                            raw = self._trim_incomplete_utf8(raw)
                    if raw:
                        yield raw.decode(self.encoding)
        finally:
            await loop.run_in_executor(None, state.close)

    # ------------------------------------------------------------------
    # Synchronous scanning (runs in executor)
    # ------------------------------------------------------------------

    def _scan(self) -> list[str]:
        """Read the file via mmap and return decoded text chunks."""
        file_size = os.path.getsize(self.path)
        if file_size == 0:
            return []

        fd = os.open(str(self.path), os.O_RDONLY)
        try:
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            try:
                return self._extract_chunks(mm, file_size)
            finally:
                mm.close()
        finally:
            os.close(fd)

    def _extract_chunks(self, mm: mmap.mmap, file_size: int) -> list[str]:
        chunks: list[str] = []
        pos = 0
        step = self.window_size - self.overlap
        encoding_key = self.encoding.lower().replace("_", "-")
        apply_utf8_boundary_fixups = encoding_key == "utf-8"

        while pos < file_size:
            end = min(pos + self.window_size, file_size)
            raw = mm[pos:end]
            if apply_utf8_boundary_fixups:
                # Skip leading continuation bytes when we are not at the start of
                # the file — these belong to a character completed in the previous
                # (overlapping) window.
                if pos > 0:
                    raw = self._skip_leading_continuation_bytes(raw)
                # Trim trailing incomplete multi-byte sequences so the next
                # overlapping window can decode them fully.
                if end < file_size:
                    raw = self._trim_incomplete_utf8(raw)
            if raw:
                text = raw.decode(self.encoding)
                chunks.append(text)
            pos += step

        return chunks

    @staticmethod
    def _skip_leading_continuation_bytes(data: bytes) -> bytes:
        """Skip any leading UTF-8 continuation bytes (0x80..0xBF).

        When a window starts in the middle of a multi-byte character, the
        first bytes will be continuation bytes that cannot be decoded on
        their own.  This method strips them so decoding succeeds.
        """
        idx = 0
        while idx < len(data) and 0x80 <= data[idx] < 0xC0:
            idx += 1
        return data[idx:]

    @staticmethod
    def _trim_incomplete_utf8(data: bytes) -> bytes:
        """Remove trailing bytes that form an incomplete UTF-8 sequence.

        This prevents ``UnicodeDecodeError`` when a multi-byte character
        straddles the window boundary.
        """
        if not data:
            return data
        # Walk backwards at most 3 bytes (max continuation bytes in UTF-8)
        for i in range(1, min(4, len(data)) + 1):
            byte = data[-i]
            if byte < 0x80:
                # ASCII — the sequence is complete
                return data
            if byte >= 0xC0:
                # Start byte found — check if enough continuation bytes follow
                if byte < 0xE0:
                    expected = 2
                elif byte < 0xF0:
                    expected = 3
                else:
                    expected = 4
                if i >= expected:
                    return data  # sequence is complete
                # Incomplete — trim the partial character
                return data[:-i]
        return data

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MMapScanner(path={self.path!r}, window_size={self.window_size}, "
            f"overlap={self.overlap}, encoding={self.encoding!r})"
        )


# ---------------------------------------------------------------------------
# line_split_scanner factory
# ---------------------------------------------------------------------------


class _LineSplitScanner:
    """Wraps :class:`MMapScanner` to yield only complete lines.

    Internally it reads in small batches from disk and yields one complete line
    at a time.
    """

    def __init__(
        self,
        path: Union[str, Path],
        encoding: str = "utf-8",
        batch_size: int = 512,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.path = Path(path)
        self.encoding = encoding
        self.batch_size = batch_size

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        with open(self.path, encoding=self.encoding) as fh:
            while True:
                lines = await loop.run_in_executor(None, self._read_batch, fh)
                if not lines:
                    break
                for line in lines:
                    yield line

    def _read_batch(self, fh: Any) -> list[str]:
        lines: list[str] = []
        for _ in range(self.batch_size):
            raw = fh.readline()
            if not raw:
                break
            lines.append(raw.rstrip("\n").rstrip("\r"))
        return lines

    def __repr__(self) -> str:
        return (
            f"LineSplitScanner(path={self.path!r}, encoding={self.encoding!r}, "
            f"batch_size={self.batch_size})"
        )


def line_split_scanner(
    path: Union[str, Path],
    encoding: str = "utf-8",
    batch_size: int = 512,
) -> _LineSplitScanner:
    """Factory that creates a scanner yielding complete lines from *path*.

    Unlike :class:`MMapScanner`, this never breaks mid-line, making it ideal
    for log files or any newline-delimited data.
    """
    return _LineSplitScanner(path=path, encoding=encoding, batch_size=batch_size)
