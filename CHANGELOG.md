# Changelog

All notable changes to FMTools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions increment by patch (0.0.205 → 0.0.206 → 0.0.207 → 0.0.208 ...).

## [0.0.217] - 2026-02-26

Final Toga chat layout and transcript UX hardening for release.

### Changed
- Sidebar layout now uses a fixed-width, height-responsive frame that stays top-aligned with the status area and bottom-aligned with the transcript frame boundary across window resize events.
- Main pane layout now allocates extra width/height proportionally to non-sidebar content (status/transcript/compose region) while preserving a stable sidebar width.
- Transcript rendering now auto-scrolls to newest content with throttled/coalesced scheduling to remain responsive during high-frequency streaming updates.

### Validation
- `uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app ruff check src/fmchat/app.py`
- Startup smoke: `uv run fmtools chat --python` (process stayed alive for 10s, then cleanly terminated)
- Async autoscroll stress probe: 120 rapid transcript updates coalesced into 6 scroll calls

## [0.0.208] - 2026-02-26

Release and distribution hardening across GitHub, PyPI, and Homebrew.

### Added
- Automated standalone chat-repo publisher workflow and script:
  - `.github/workflows/publish-chat-repo.yml`
  - `scripts/publish_chat_repo.sh`
- Standalone `fmchat` sync template and documentation under `standalone/fmchat/`.

### Changed
- Standardized naming across parent and child app surfaces:
  - repo/package naming: `fmtools` / `fmchat`
  - app naming: `FMTools` / `FMChat`
- Toga chat app package path renamed to `fmchat`.
- Briefcase project metadata aligned with `FMChat`.
- `.gitignore` hardened to prevent local data leakage from SQLite/chat-history artifacts (`*.sqlite*`, `*.db*`) and local publish workspaces.

### Validation
- `uv run ruff check ...` on touched Python files
- `uv run ruff format --check ...` on touched Python files
- `uv run pytest tests/ -q` (482 passed)
- Standalone publisher idempotency verified against `adpena/fmchat`

## [0.0.207] - 2026-02-26

Release hardening focused on desktop-chat stability, no-gil fallback safety, and docs/packaging ergonomics.

### Added
- CLI `chat --standard-gil` flag to force stable standard-GIL runtime for GUI demos.
- Explicit runtime selection + fallback probing in `fmtools.cli` for free-threaded vs standard-GIL execution.

### Changed
- `fmtools chat` now prefers free-threaded CPython (`3.14t` then `3.13t`) and retries with an explicit standard-GIL interpreter (`3.14` then `3.13`) if no-gil launch fails.
- Toga desktop app UI polish:
  - responsive wrapped sidebar title
  - bottom-pinned action controls
  - unified transcript/compose typography and text insets
  - wrapped status messaging on narrow widths
  - status bar width alignment with transcript/compose columns on compact windows
- Disabled no-gil-unsafe Cocoa Enter-key delegate path in free-threaded mode to avoid Rubicon/ObjC callback segfaults.

### Validation
- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run ty check fmtools/`
- `uv run pytest` (482 passed)
- `uv run fmtools smoke` (passed)

## [0.0.206] - 2026-02-26

Release-candidate hardening pass for launch readiness.

### Added
- CLI `example` command to list and run standalone scripts under `examples/`
- CLI `smoke` command to run full examples smoke suite with SDK preflight and per-script timeouts
- CLI `notebook` command to launch `examples/examples_notebook.py` via marimo

### Changed
- `audit_directory(...)` now uses a bounded worker pool (`max_concurrency`) for better throughput under uneven per-file runtimes while preserving deterministic output ordering
- `examples/streaming_example.py` now prints token deltas correctly when SDK streaming returns cumulative text chunks
- README and examples docs synchronized with current CLI/API surface and examples workflow

### Validation
- Real Apple SDK smoke run completed across all top-level example scripts plus notebook startup
- Full quality gates: `ruff check`, `ruff format --check`, `ty check`, and pytest suite all green

## [0.0.205] - 2026-02-26

Initial public release. Consolidates all prior development into a clean baseline.

### Core Framework
- `@local_extract` decorator — transforms functions into on-device structured extractors
- `stream_extract` async generator — concurrent streaming extraction with 4 history modes (clear, keep, hybrid, compact)
- `Source >> Extract >> Sink` pipeline operators — composable async ETL pipelines
- `@enhanced_debug` decorator — AI-powered crash analysis via Neural Engine
- Polars `.local_llm.extract()` namespace extension
- DSPy `AppleFMLM` provider

### Phase 4 Feature Expansion
- `fmtools.cache` — sqlite3 content-addressable extraction cache + cached decorator helpers
- `fmtools.protocols` — typing.Protocol backend interfaces with swappable backend registry, wired through core extraction runtime
- `fmtools.adapters` — file/stdin/CSV/JSONL/iterable/trio adapters + chunking adapter
- `fmtools._context` — contextvars-based session scoping helpers
- `fmtools._threading` — free-threading detection and synchronization primitives
- `fmtools.scanner` — mmap sliding-window scanner for large files
- `fmtools.watcher` — hot-folder watcher daemon with extraction convenience API
- `fmtools._jit` — runtime diagnostics and performance counters
- `fmtools.arrow_bridge` — Arrow IPC file/buffer bridge + Polars conversion helpers
- `fmtools.functional` — functional pipeline composition API
- `fmtools.auditor` — on-device code auditor utilities

### Phase 4 Hardening Pass
- `_context.session_scope(...)` now uses the active backend registry (`create_model`/`create_session`) for full pluggable-backend parity
- `line_split_scanner(...)` now streams incrementally in batches instead of pre-buffering entire files
- `MMapScanner` now guards UTF-8 boundary fixups to UTF-8 mode only and closes file descriptors safely if `mmap` initialization fails
- `cached_stream_extract(...)` now supports both sync and async sources
- `safe_model_cache()` initialization is now race-safe under concurrent access
- `functional` pipeline steps now support async callable objects correctly and validate terminal-step ordering
- `audit_directory(...)` now continues on per-file failures and parser handling is hardened for malformed model payloads
- `watcher.start(...)` now awaits generic awaitables; `process_folder(...)` handles `modified` events with mtime dedupe
- `examples/transcript_processing.py` now runs against a bundled sample transcript dataset by default (`datasets/transcript_sample.json`)

### Toolchain
- `fmtools` CLI with `setup`, `doctor`, `lint`, `format`, `typecheck`, `test`, and `check` commands (Click-based)
- `scripts/setup.sh` — one-command development environment setup
- `scripts/doctor.sh` — system prerequisites verification (9 checks)
- `uv` for package management with `[tool.uv.sources]` git dependency for Apple FM SDK
- `ruff` linting and formatting with comprehensive rule set (zero violations)
- `ty` type checker (zero diagnostics)

### Testing
- 400+ mock-based tests (zero Apple Silicon required to run)
- Full coverage of core + Phase 4 modules via pytest + pytest-asyncio

### Documentation
- Comprehensive README with API reference, tutorials, benchmarks, and architecture
- 9 use case examples with sample datasets
- GEMINI.md for AI-assisted development context
