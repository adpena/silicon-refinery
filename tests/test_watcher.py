"""
Tests for silicon_refinery.watcher — Hot Folder Daemon.

Covers:
  - New file detection
  - Modified file detection
  - Glob pattern filtering
  - Auto-creation of missing directories
  - Clean stop behaviour
  - Context manager protocol
  - Debounce (same file within 1 s is not re-emitted)
  - Recursive directory watching
  - process_folder extraction convenience
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from silicon_refinery.watcher import FileEvent, HotFolder, process_folder

if TYPE_CHECKING:
    from pathlib import Path


# =====================================================================
# Helpers
# =====================================================================


async def _collect_events(
    watcher: HotFolder,
    *,
    max_events: int = 1,
    timeout: float = 5.0,
) -> list[FileEvent]:
    """Collect up to *max_events* from the watcher within *timeout* seconds."""
    events: list[FileEvent] = []
    try:
        async with asyncio.timeout(timeout):
            async for event in watcher.watch():
                events.append(event)
                if len(events) >= max_events:
                    watcher.stop()
    except TimeoutError:
        pass
    return events


# =====================================================================
# test_hot_folder_detects_new_file
# =====================================================================


class TestHotFolderDetectsNewFile:
    async def test_new_file_yields_created_event(self, tmp_path: Path) -> None:
        """Dropping a new file into the watch directory yields a 'created' event."""
        watcher = HotFolder(tmp_path, patterns=["*.txt"], poll_interval=0.05)

        async def _write_after_delay() -> None:
            await asyncio.sleep(0.1)
            (tmp_path / "hello.txt").write_text("content")

        task = asyncio.create_task(_write_after_delay())
        events = await _collect_events(watcher, max_events=1, timeout=3.0)
        await task

        assert len(events) == 1
        assert events[0].event_type == "created"
        assert events[0].path.name == "hello.txt"


# =====================================================================
# test_hot_folder_detects_modified_file
# =====================================================================


class TestHotFolderDetectsModifiedFile:
    async def test_modified_file_yields_modified_event(self, tmp_path: Path) -> None:
        """Modifying an existing file yields a 'modified' event."""
        target = tmp_path / "data.txt"
        target.write_text("original")

        watcher = HotFolder(tmp_path, patterns=["*.txt"], poll_interval=0.05)

        async def _modify_after_delay() -> None:
            await asyncio.sleep(0.15)
            # Ensure mtime actually differs
            target.write_text("updated")

        task = asyncio.create_task(_modify_after_delay())
        events = await _collect_events(watcher, max_events=1, timeout=3.0)
        await task

        assert len(events) >= 1
        assert events[0].event_type == "modified"
        assert events[0].path.name == "data.txt"


# =====================================================================
# test_hot_folder_pattern_filtering
# =====================================================================


class TestHotFolderPatternFiltering:
    async def test_non_matching_files_are_ignored(self, tmp_path: Path) -> None:
        """Files that don't match any pattern are silently ignored."""
        watcher = HotFolder(tmp_path, patterns=["*.csv"], poll_interval=0.05)

        async def _write_files() -> None:
            await asyncio.sleep(0.1)
            (tmp_path / "skip.txt").write_text("ignore me")
            await asyncio.sleep(0.15)
            (tmp_path / "take.csv").write_text("keep me")

        task = asyncio.create_task(_write_files())
        events = await _collect_events(watcher, max_events=1, timeout=3.0)
        await task

        assert len(events) == 1
        assert events[0].path.name == "take.csv"


# =====================================================================
# test_hot_folder_creates_missing_directory
# =====================================================================


class TestHotFolderCreatesMissingDirectory:
    async def test_directory_is_created_if_absent(self, tmp_path: Path) -> None:
        """HotFolder creates the watch directory if it does not exist."""
        target_dir = tmp_path / "sub" / "deep"
        assert not target_dir.exists()

        watcher = HotFolder(target_dir, poll_interval=0.05)

        async def _write_then_stop() -> None:
            await asyncio.sleep(0.1)
            (target_dir / "file.txt").write_text("hi")

        task = asyncio.create_task(_write_then_stop())
        events = await _collect_events(watcher, max_events=1, timeout=3.0)
        await task

        assert target_dir.exists()
        assert len(events) == 1


# =====================================================================
# test_hot_folder_stop_exits_cleanly
# =====================================================================


class TestHotFolderStopExitsCleanly:
    async def test_stop_terminates_watch_loop(self, tmp_path: Path) -> None:
        """Calling stop() causes the watch() generator to end."""
        watcher = HotFolder(tmp_path, poll_interval=0.05)

        async def _stop_soon() -> None:
            await asyncio.sleep(0.15)
            watcher.stop()

        task = asyncio.create_task(_stop_soon())
        events = await _collect_events(watcher, max_events=100, timeout=2.0)
        await task

        # We should exit cleanly with no events (nothing was written)
        assert isinstance(events, list)


# =====================================================================
# test_hot_folder_context_manager
# =====================================================================


class TestHotFolderContextManager:
    async def test_async_context_manager(self, tmp_path: Path) -> None:
        """HotFolder supports ``async with`` and stops on exit."""
        async with HotFolder(tmp_path, poll_interval=0.05) as watcher:
            assert watcher._running is True
        # After exiting the context, the watcher should have stopped
        assert watcher._running is False

    async def test_context_manager_creates_directory(self, tmp_path: Path) -> None:
        missing = tmp_path / "auto_created"
        async with HotFolder(missing, poll_interval=0.05):
            pass
        assert missing.exists()


# =====================================================================
# test_hot_folder_debounce
# =====================================================================


class TestHotFolderDebounce:
    async def test_rapid_writes_debounced(self, tmp_path: Path) -> None:
        """Modifying the same file in rapid succession emits only once."""
        target = tmp_path / "bounce.txt"
        target.write_text("v0")

        watcher = HotFolder(tmp_path, patterns=["*.txt"], poll_interval=0.05)

        events: list[FileEvent] = []

        async def _rapid_writes() -> None:
            await asyncio.sleep(0.1)
            # Write multiple times within the 1 s debounce window
            for i in range(5):
                target.write_text(f"v{i + 1}")
                await asyncio.sleep(0.06)
            # Let one more poll cycle complete then stop
            await asyncio.sleep(0.15)
            watcher.stop()

        task = asyncio.create_task(_rapid_writes())

        async for event in watcher.watch():
            events.append(event)

        await task

        # Debounce should collapse rapid modifications heavily (far fewer than writes).
        # Depending on scheduler timing, a second event can appear if the watch loop
        # crosses the 1-second debounce boundary.
        assert len(events) <= 2

    def test_poll_prunes_stale_last_emitted_entries(self, tmp_path: Path) -> None:
        watcher = HotFolder(tmp_path, patterns=["*.txt"], poll_interval=0.05)
        stale = str(tmp_path / "gone.txt")
        watcher._known = {stale: 1.0}
        watcher._last_emitted = {stale: 1.0}

        with patch.object(watcher, "_scan_files", return_value={}):
            events = watcher._poll()

        assert events == []
        assert stale not in watcher._last_emitted


# =====================================================================
# test_hot_folder_recursive_mode
# =====================================================================


class TestHotFolderRecursiveMode:
    async def test_recursive_detects_nested_file(self, tmp_path: Path) -> None:
        """With recursive=True, files in subdirectories are detected."""
        sub = tmp_path / "level1" / "level2"
        sub.mkdir(parents=True)

        watcher = HotFolder(tmp_path, patterns=["*.txt"], poll_interval=0.05, recursive=True)

        async def _write_nested() -> None:
            await asyncio.sleep(0.1)
            (sub / "nested.txt").write_text("deep content")

        task = asyncio.create_task(_write_nested())
        events = await _collect_events(watcher, max_events=1, timeout=3.0)
        await task

        assert len(events) == 1
        assert events[0].path.name == "nested.txt"
        assert events[0].event_type == "created"


# =====================================================================
# test_process_folder_extracts_and_yields
# =====================================================================


class TestProcessFolderExtractsAndYields:
    async def test_extraction_via_process_folder(self, tmp_path: Path) -> None:
        """process_folder reads new files, extracts via FM, and yields results."""
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        output_dir = tmp_path / "output"

        mock_model = MagicMock()
        mock_model.is_available.return_value = (True, None)

        extraction_result = {"name": "Alice", "age": 30}

        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=extraction_result)

        class DummySchema:
            pass

        results: list[tuple[Path, object]] = []

        async def _write_file() -> None:
            await asyncio.sleep(0.15)
            (watch_dir / "person.txt").write_text("Alice is 30")

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            task = asyncio.create_task(_write_file())

            try:
                async with asyncio.timeout(3.0):
                    async for path, result in process_folder(
                        watch_dir,
                        schema=DummySchema,
                        instructions="Extract person data.",
                        patterns=["*.txt"],
                        output_dir=output_dir,
                    ):
                        results.append((path, result))
                        break  # we only expect one
            except TimeoutError:
                pass

            await task

        assert len(results) == 1
        assert results[0][0].name == "person.txt"
        assert results[0][1] == extraction_result

        # Verify JSON output was written
        json_file = output_dir / "person.txt.json"
        assert json_file.exists()
        written = json.loads(json_file.read_text())
        assert written == extraction_result

    async def test_process_folder_handles_modified_events(self, tmp_path: Path) -> None:
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        target = watch_dir / "person.txt"
        target.write_text("Alice is 30")

        extraction_result = {"name": "Alice", "age": 30}
        mock_model = MagicMock()
        mock_model.is_available.return_value = (True, None)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=extraction_result)

        class DummySchema:
            pass

        class FakeHotFolder:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                del exc_type, exc, tb

            async def watch(self):
                yield FileEvent(path=target, event_type="modified", timestamp=0.0)

        with (
            patch("silicon_refinery.watcher.HotFolder", FakeHotFolder),
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for path, result in process_folder(
                watch_dir,
                schema=DummySchema,
                instructions="Extract person data.",
                patterns=["*.txt"],
            ):
                results.append((path, result))
                break

        assert len(results) == 1
        assert results[0][0].name == "person.txt"
        assert results[0][1] == extraction_result

    async def test_process_folder_dedupes_by_content_hash(self, tmp_path: Path) -> None:
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        target = watch_dir / "person.txt"
        target.write_text("Alice is 30", encoding="utf-8")
        original_mtime_ns = target.stat().st_mtime_ns

        extraction_results = [{"name": "Alice", "age": 30}, {"name": "Alice", "age": 31}]
        mock_model = MagicMock()
        mock_model.is_available.return_value = (True, None)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(side_effect=extraction_results)

        class DummySchema:
            pass

        class FakeHotFolder:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                del exc_type, exc, tb

            async def watch(self):
                yield FileEvent(path=target, event_type="modified", timestamp=0.0)
                target.write_text("Alice is 31", encoding="utf-8")
                os.utime(target, ns=(original_mtime_ns, original_mtime_ns))
                yield FileEvent(path=target, event_type="modified", timestamp=1.0)

        with (
            patch("silicon_refinery.watcher.HotFolder", FakeHotFolder),
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for path, result in process_folder(
                watch_dir,
                schema=DummySchema,
                instructions="Extract person data.",
                patterns=["*.txt"],
            ):
                results.append((path, result))

        assert [result for _, result in results] == extraction_results

    async def test_process_folder_continues_after_unicode_decode_error(
        self, tmp_path: Path
    ) -> None:
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        bad_file = watch_dir / "bad.txt"
        good_file = watch_dir / "good.txt"
        bad_file.write_bytes(b"\xff\xfe\xfd")
        good_file.write_text("good data", encoding="utf-8")

        extraction_result = {"name": "Good"}
        mock_model = MagicMock()
        mock_model.is_available.return_value = (True, None)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=extraction_result)

        class DummySchema:
            pass

        class FakeHotFolder:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                del exc_type, exc, tb

            async def watch(self):
                yield FileEvent(path=bad_file, event_type="created", timestamp=0.0)
                yield FileEvent(path=good_file, event_type="created", timestamp=1.0)

        with (
            patch("silicon_refinery.watcher.HotFolder", FakeHotFolder),
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            results = []
            async for path, result in process_folder(
                watch_dir,
                schema=DummySchema,
                instructions="Extract person data.",
                patterns=["*.txt"],
            ):
                results.append((path, result))

        assert len(results) == 1
        assert results[0][0] == good_file
        assert results[0][1] == extraction_result
        assert mock_session.respond.await_count == 1

    async def test_process_folder_continues_when_output_write_fails(self, tmp_path: Path) -> None:
        watch_dir = tmp_path / "inbox"
        watch_dir.mkdir()
        output_dir = tmp_path / "output"
        target = watch_dir / "person.txt"
        target.write_text("Alice is 30", encoding="utf-8")

        extraction_result = {"name": "Alice", "age": 30}
        mock_model = MagicMock()
        mock_model.is_available.return_value = (True, None)
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=extraction_result)

        class DummySchema:
            pass

        class FakeHotFolder:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                del exc_type, exc, tb

            async def watch(self):
                yield FileEvent(path=target, event_type="created", timestamp=0.0)

        with (
            patch("silicon_refinery.watcher.HotFolder", FakeHotFolder),
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
            patch("pathlib.Path.write_text", side_effect=OSError("disk full")),
        ):
            results = []
            async for path, result in process_folder(
                watch_dir,
                schema=DummySchema,
                instructions="Extract person data.",
                patterns=["*.txt"],
                output_dir=output_dir,
            ):
                results.append((path, result))

        assert len(results) == 1
        assert results[0][0] == target
        assert results[0][1] == extraction_result


class TestHotFolderStart:
    async def test_start_awaits_custom_awaitable_callback(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        watcher = HotFolder(tmp_path, poll_interval=0.01)

        async def fake_watch(self):
            yield FileEvent(path=tmp_path / "x.txt", event_type="created", timestamp=0.0)

        monkeypatch.setattr(HotFolder, "watch", fake_watch)

        state = {"awaited": False}

        class CustomAwaitable:
            def __await__(self):
                async def _run():
                    state["awaited"] = True

                return _run().__await__()

        def callback(_event):
            return CustomAwaitable()

        await watcher.start(callback)
        assert state["awaited"] is True
