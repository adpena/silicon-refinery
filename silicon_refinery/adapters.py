"""
IO Protocol Wrappers — flexible input adapters for SiliconRefinery pipelines.

Each adapter implements the ``DataAdapter`` protocol (``__aiter__`` yielding strings)
so it can be plugged directly into ``Source()``, ``stream_extract()``, or any other
async-for consumer.

Includes optional support for trio-style receive channels via ``TrioAdapter``.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Union, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable

logger = logging.getLogger("silicon_refinery")


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class DataAdapter(Protocol):
    """Common protocol for all input adapters."""

    def __aiter__(self) -> AsyncIterator[str]: ...


# ---------------------------------------------------------------------------
# FileAdapter
# ---------------------------------------------------------------------------


class FileAdapter:
    """Reads a text file line-by-line as an async iterator.

    Each yielded string is a single line with the trailing newline stripped.
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
            line = fh.readline()
            if not line:
                break
            lines.append(line.rstrip("\n").rstrip("\r"))
        return lines

    def __repr__(self) -> str:
        return (
            f"FileAdapter(path={self.path!r}, encoding={self.encoding!r}, "
            f"batch_size={self.batch_size})"
        )


# ---------------------------------------------------------------------------
# StdinAdapter
# ---------------------------------------------------------------------------


class StdinAdapter:
    """Reads lines from ``sys.stdin`` as an async iterator.

    Uses ``run_in_executor`` so the event loop is not blocked by synchronous
    stdin reads.
    """

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        while True:
            line: str | None = await loop.run_in_executor(None, self._read_line)
            if line is None:
                break
            yield line

    @staticmethod
    def _read_line() -> str | None:
        line = sys.stdin.readline()
        if not line:
            return None
        return line.rstrip("\n").rstrip("\r")

    def __repr__(self) -> str:
        return "StdinAdapter()"


# ---------------------------------------------------------------------------
# CSVAdapter
# ---------------------------------------------------------------------------


class CSVAdapter:
    """Reads a CSV file, yielding each row as a JSON string.

    If *column* is specified only that column's value is yielded as a plain
    string instead of the full JSON row.
    """

    def __init__(
        self,
        path: Union[str, Path],
        column: Union[str, None] = None,
        batch_size: int = 512,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.path = Path(path)
        self.column = column
        self.batch_size = batch_size

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        with open(self.path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            while True:
                rows = await loop.run_in_executor(None, self._read_batch, reader)
                if not rows:
                    break
                for row in rows:
                    yield row

    def _read_batch(self, reader: csv.DictReader[str]) -> list[str]:
        results: list[str] = []
        for _ in range(self.batch_size):
            try:
                row = next(reader)
            except StopIteration:
                break
            if self.column is not None:
                results.append(row[self.column])
            else:
                results.append(json.dumps(row))
        return results

    def __repr__(self) -> str:
        return (
            f"CSVAdapter(path={self.path!r}, column={self.column!r}, batch_size={self.batch_size})"
        )


# ---------------------------------------------------------------------------
# JSONLAdapter
# ---------------------------------------------------------------------------


class JSONLAdapter:
    """Reads a JSON Lines file, yielding each parsed line as a string."""

    def __init__(
        self,
        path: Union[str, Path],
        batch_size: int = 512,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.path = Path(path)
        self.batch_size = batch_size

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        with open(self.path, newline="", encoding="utf-8") as fh:
            while True:
                rows = await loop.run_in_executor(None, self._read_batch, fh)
                if not rows:
                    break
                for row in rows:
                    yield row

    def _read_batch(self, fh: Any) -> list[str]:
        results: list[str] = []
        while len(results) < self.batch_size:
            raw = fh.readline()
            if not raw:
                break
            stripped = raw.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            results.append(json.dumps(parsed))
        return results

    def __repr__(self) -> str:
        return f"JSONLAdapter(path={self.path!r}, batch_size={self.batch_size})"


# ---------------------------------------------------------------------------
# IterableAdapter
# ---------------------------------------------------------------------------


class IterableAdapter:
    """Wraps any synchronous ``Iterable[str]`` into an async iterator."""

    def __init__(self, iterable: Iterable[str]) -> None:
        self.iterable = iterable

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        for index, item in enumerate(self.iterable, start=1):
            yield item
            if index % 256 == 0:
                await asyncio.sleep(0)

    def __repr__(self) -> str:
        return f"IterableAdapter(iterable={self.iterable!r})"


# ---------------------------------------------------------------------------
# TrioAdapter
# ---------------------------------------------------------------------------


class TrioAdapter:
    """Wrap trio-style receive channels or async iterables as ``DataAdapter``.

    Supported sources:
    - Any async iterable yielding values.
    - Any object exposing an async ``receive()`` method.
      End-of-stream is detected via ``trio.EndOfChannel`` (if trio is installed)
      or exception class names commonly used by trio channels.
    """

    def __init__(self, source: Any) -> None:
        self.source = source

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        if hasattr(self.source, "__aiter__"):
            async for item in self.source:
                yield str(item)
            return

        receive = getattr(self.source, "receive", None)
        if not callable(receive):
            raise TypeError(
                "TrioAdapter source must be an async iterable or expose async receive()"
            )

        trio_end_of_channel: type[BaseException] | None = None
        try:
            import trio

            trio_end_of_channel = getattr(trio, "EndOfChannel", None)
        except ImportError:
            pass

        while True:
            try:
                item = await receive()
            except Exception as exc:
                if trio_end_of_channel is not None and isinstance(exc, trio_end_of_channel):
                    break
                if type(exc).__name__ in {"EndOfChannel", "ClosedResourceError"}:
                    break
                raise
            yield str(item)

    def __repr__(self) -> str:
        return f"TrioAdapter(source={self.source!r})"


# ---------------------------------------------------------------------------
# TextChunkAdapter
# ---------------------------------------------------------------------------


class TextChunkAdapter:
    """Re-chunks output from any ``DataAdapter`` into fixed-size text windows.

    The adapter incrementally buffers streamed text from *source* and yields
    windows of *window_size* characters with *overlap* characters carried over
    into the next window, without pre-loading the full stream.
    """

    def __init__(
        self,
        source: DataAdapter,
        window_size: int = 4096,
        overlap: int = 256,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if overlap >= window_size:
            raise ValueError("overlap must be < window_size")

        self.source = source
        self.window_size = window_size
        self.overlap = overlap

    def __aiter__(self) -> AsyncIterator[str]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[str]:
        step = self.window_size - self.overlap
        buffer = ""
        async for chunk in self.source:
            buffer += chunk
            while len(buffer) >= self.window_size:
                yield buffer[: self.window_size]
                buffer = buffer[step:]

        if buffer:
            yield buffer

    def __repr__(self) -> str:
        return (
            f"TextChunkAdapter(source={self.source!r}, "
            f"window_size={self.window_size}, overlap={self.overlap})"
        )
