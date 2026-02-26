"""
Hot Folder Daemon — filesystem watcher for automatic extraction pipelines.

Monitors a directory for new/changed files using stdlib-only polling
(os.scandir + asyncio.sleep) and feeds them through the SiliconRefinery
extraction pipeline.
"""

from __future__ import annotations

import asyncio
import contextlib
import fnmatch
import hashlib
import inspect
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Union

from .protocols import create_model, create_session

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

logger = logging.getLogger("silicon_refinery")


@dataclass(frozen=True)
class FileEvent:
    """Represents a filesystem event detected by the watcher."""

    path: Path
    event_type: Literal["created", "modified"]
    timestamp: float


class HotFolder:
    """
    Watches a directory for new or modified files using polling.

    Uses os.scandir() + asyncio.sleep() to detect changes without any
    third-party dependencies.  Supports glob-pattern filtering, recursive
    scanning, and debouncing.

    Usage::

        async with HotFolder("/data/incoming", patterns=["*.txt"]) as watcher:
            async for event in watcher.watch():
                print(event)
    """

    def __init__(
        self,
        watch_dir: Union[str, Path],
        patterns: list[str] | None = None,
        poll_interval: float = 1.0,
        recursive: bool = False,
    ) -> None:
        self._watch_dir = Path(watch_dir)
        self._patterns: list[str] = patterns if patterns is not None else ["*"]
        self._poll_interval = poll_interval
        self._recursive = recursive

        # mtime tracking: path -> last known mtime
        self._known: dict[str, float] = {}
        # debounce tracking: path -> last emitted timestamp
        self._last_emitted: dict[str, float] = {}

        self._running = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> HotFolder:
        self._ensure_directory()
        self._running = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def watch(self) -> AsyncGenerator[FileEvent, None]:
        """Async generator that yields :class:`FileEvent` objects when files appear or change."""
        self._ensure_directory()
        self._running = True

        # Snapshot the current directory state so pre-existing files are not
        # treated as "created" on the first poll.
        self._snapshot()

        while self._running:
            events = await asyncio.to_thread(self._poll)
            for event in events:
                yield event
            await asyncio.sleep(self._poll_interval)

    async def start(self, callback: Callable[[FileEvent], Union[Awaitable[None], None]]) -> None:
        """Start watching and invoke *callback* for every detected event."""
        async for event in self.watch():
            result = callback(event)
            if inspect.isawaitable(result):
                await result

    def stop(self) -> None:
        """Signal the watcher to stop after the current poll cycle."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_directory(self) -> None:
        """Create the watch directory if it does not exist."""
        self._watch_dir.mkdir(parents=True, exist_ok=True)

    def _matches_pattern(self, name: str) -> bool:
        """Return ``True`` if *name* matches any of the configured glob patterns."""
        return any(fnmatch.fnmatch(name, pat) for pat in self._patterns)

    def _scan_files(self) -> dict[str, float]:
        """Return a mapping of ``{absolute_path_str: mtime}`` for matching files."""
        result: dict[str, float] = {}
        if self._recursive:
            self._scan_recursive(self._watch_dir, result)
        else:
            self._scan_flat(self._watch_dir, result)
        return result

    def _scan_flat(self, directory: Path, out: dict[str, float]) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and self._matches_pattern(entry.name):
                        with contextlib.suppress(OSError):
                            out[entry.path] = entry.stat().st_mtime
        except OSError:
            pass

    def _scan_recursive(self, directory: Path, out: dict[str, float]) -> None:
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_file() and self._matches_pattern(entry.name):
                        with contextlib.suppress(OSError):
                            out[entry.path] = entry.stat().st_mtime
                    elif entry.is_dir(follow_symlinks=False):
                        self._scan_recursive(Path(entry.path), out)
        except OSError:
            pass

    def _snapshot(self) -> None:
        """Populate ``_known`` with the current directory state (no events emitted)."""
        self._known = self._scan_files()

    def _poll(self) -> list[FileEvent]:
        """Scan the directory and return a list of new events since the last poll."""
        now = time.time()
        current = self._scan_files()
        events: list[FileEvent] = []

        for path_str, mtime in current.items():
            last_emitted = self._last_emitted.get(path_str, 0.0)

            if path_str not in self._known:
                # New file
                if now - last_emitted >= 1.0:
                    events.append(
                        FileEvent(
                            path=Path(path_str),
                            event_type="created",
                            timestamp=now,
                        )
                    )
                    self._last_emitted[path_str] = now
            elif mtime != self._known[path_str] and now - last_emitted >= 1.0:
                # Modified file (with debounce)
                events.append(
                    FileEvent(
                        path=Path(path_str),
                        event_type="modified",
                        timestamp=now,
                    )
                )
                self._last_emitted[path_str] = now

        self._known = current
        stale = [path for path in self._last_emitted if path not in current]
        for path in stale:
            del self._last_emitted[path]
        return events


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------


async def process_folder(
    watch_dir: Union[str, Path],
    schema: type,
    instructions: str = "Extract data.",
    patterns: list[str] | None = None,
    output_dir: Union[str, Path, None] = None,
) -> AsyncGenerator[tuple[Path, Any], None]:
    """
    Watch *watch_dir* for new files, extract structured data from each,
    and yield ``(path, result)`` tuples.

    If *output_dir* is provided, each result is also written as a JSON file
    with a ``.json`` suffix appended to the original filename.
    """
    if patterns is None:
        patterns = ["*.txt"]

    out_path: Path | None = Path(output_dir) if output_dir is not None else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    model = create_model()
    processed_hash: dict[Path, str] = {}

    async with HotFolder(watch_dir, patterns=patterns) as watcher:
        async for event in watcher.watch():
            if event.event_type not in {"created", "modified"}:
                continue

            file_path = event.path
            try:
                raw_bytes = await asyncio.to_thread(file_path.read_bytes)
            except OSError as exc:
                logger.warning("[process_folder] Could not read %s: %s", file_path, exc)
                continue

            content_hash = hashlib.sha256(raw_bytes).hexdigest()
            if processed_hash.get(file_path) == content_hash:
                continue

            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                logger.warning("[process_folder] Could not decode %s as UTF-8: %s", file_path, exc)
                continue

            session = create_session(instructions=instructions, model=model)
            try:
                result = await session.respond(text, generating=schema)
            except Exception as exc:
                logger.error("[process_folder] Extraction failed for %s: %s", file_path, exc)
                continue

            processed_hash[file_path] = content_hash

            if out_path is not None:
                json_file = out_path / (file_path.name + ".json")
                payload = json.dumps(result, default=str, indent=2)
                try:
                    await asyncio.to_thread(
                        json_file.write_text,
                        payload,
                        encoding="utf-8",
                    )
                except OSError as exc:
                    logger.warning("[process_folder] Could not write %s: %s", json_file, exc)

            yield file_path, result
