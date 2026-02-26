# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SiliconRefinery is a Python ETL framework for on-device structured data extraction using the Apple Foundation Models SDK (`apple_fm_sdk`). All inference runs locally on Apple Silicon via the Neural Engine — no cloud calls, no API keys. Requires macOS 26+ and Apple Silicon hardware.

## Build & Development Commands

Package manager: **uv** (with hatchling build backend).

```bash
# Setup
./scripts/setup.sh              # One-command bootstrap (installs uv, syncs deps)
uv sync --all-groups            # Install all dependencies (runtime + dev)

# Quality gates (via CLI or directly)
uv run silicon-refinery check   # Full CI gate: lint + format + typecheck + tests
uv run silicon-refinery lint    # Ruff linter
uv run silicon-refinery format  # Ruff formatter
uv run silicon-refinery typecheck  # ty type checker on silicon_refinery/

# Testing
uv run pytest tests/ -v                     # Full test suite
uv run pytest tests/test_decorators.py -v   # Single test file
uv run pytest tests/ -v -k "test_name"      # Single test by name

# Diagnostics
uv run silicon-refinery doctor  # Verify platform prerequisites
```

CI runs lint, format check, typecheck, and pytest on Python 3.13 and 3.14.

## Architecture

### Core Extraction Patterns

The framework offers three ways to extract structured data from text:

1. **`@local_extract` decorator** (`decorators.py`) — Transforms an async function into a structured extractor. The function's docstring becomes the LLM system prompt. Returns a Pydantic-like object matching the `@fm.generable()` schema.

2. **`stream_extract` async generator** (`async_generators.py`) — Processes streams of text chunks with configurable `history_mode` (clear/keep/hybrid/compact) and `concurrency` for parallel extraction.

3. **`Source >> Extract >> Sink` pipeline** (`pipeline.py`) — OOP composable pipeline using `__rshift__` operator chaining. `Pipeline.execute()` is an async generator; `Pipeline.collect()` materializes results.

There is also a **functional pipeline API** (`functional.py`) using `pipe(source(...), extract(...), map_fn(...), collect())` with `|` operator composition between Steps.

### Backend Abstraction

`protocols.py` defines `ModelProtocol` and `SessionProtocol` interfaces with a swappable backend registry (`set_backend`/`get_backend`). All core modules call `create_model()` and `create_session()` from protocols — never `apple_fm_sdk` directly. The default `AppleFMBackend` wraps `apple_fm_sdk.SystemLanguageModel` and `apple_fm_sdk.LanguageModelSession`.

### Lazy Imports

`__init__.py` uses `__getattr__` for lazy imports so the CLI and non-extraction code can run without `apple_fm_sdk` installed. The SDK is only imported when extraction features are actually accessed.

### Testing Strategy

All tests mock `apple_fm_sdk` — no Apple Silicon required to run the test suite. `tests/conftest.py` installs a fake `apple_fm_sdk` into `sys.modules` **before** any `silicon_refinery` imports. Key fixtures: `mock_fm_available`, `mock_fm_unavailable`, `mock_fm_failing`, `mock_fm_context_window_error`. The `asyncio_mode = "auto"` pytest setting means async test functions run automatically without `@pytest.mark.asyncio`.

## Key Conventions

- Python 3.13+ target, max line length 100 (Ruff enforced)
- Function docstrings influence LLM behavior — they become system prompts for extraction
- Schemas must use `@apple_fm_sdk.generable()` decorator with `fm.guide()` field descriptors
- Async-first: all extraction paths are `async`/`await`
- Sessions are recreated per extraction to avoid context window explosion (~32k char limit)
- Commit messages follow Conventional Commits: `feat:`, `fix:`, `chore:`, etc.
- Apple FM SDK is a git dependency via `[tool.uv.sources]` — not on PyPI

## Module Map

| Module | Purpose |
|--------|---------|
| `decorators.py` | `@local_extract` decorator with retry logic |
| `async_generators.py` | `stream_extract` with history modes and concurrency |
| `pipeline.py` | `Source >> Extract >> Sink` OOP pipeline |
| `functional.py` | Functional pipeline: `pipe()`, `source()`, `extract()`, `map_fn()`, etc. |
| `debugging.py` | `@enhanced_debug` AI crash analysis decorator |
| `protocols.py` | Backend protocols and registry (`create_model`/`create_session`) |
| `polars_ext.py` | Polars `.local_llm.extract()` namespace extension |
| `dspy_ext.py` | DSPy `AppleFMLM` provider |
| `cache.py` | sqlite3 content-addressable extraction cache |
| `adapters.py` | Input adapters: file/stdin/CSV/JSONL/trio |
| `scanner.py` | mmap sliding-window scanner for large files |
| `watcher.py` | Hot-folder polling daemon |
| `cli.py` | Click-based CLI (`silicon-refinery` command) |
| `exceptions.py` | `AppleFMSetupError` and setup diagnostics |
