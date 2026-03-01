# GEMINI.md - FMTools

## Project Overview

**FMTools** is an enterprise-grade Python ETL framework designed for **zero-trust, high-throughput data processing** exclusively on Apple Silicon. It leverages the [Apple Foundation Models SDK (`python-apple-fm-sdk`)](https://github.com/apple/python-apple-fm-sdk) to run on-device LLM inference, ensuring that sensitive data never leaves the local machine.

The framework abstracts complex LLM interactions into idiomatic Python patterns, making it easy to build structured data extraction pipelines, AI-powered debuggers, and interactive data analysis tools.

### Key Technologies
- **Inference Engine:** `python-apple-fm-sdk` (Apple Neural Engine accelerated)
- **Data Engineering:** `Polars` (Rust-backed DataFrames)
- **Async Orchestration:** Python `asyncio` & `AsyncGenerators`
- **Agentic Workflows:** `DSPy` integration
- **Web Integration:** `FastAPI` & `Uvicorn`
- **Packaging:** `hatchling`, `uv`

---

## Architecture & Core Modules

The project is structured into several core modules, each providing a specific interaction pattern with the local Foundation Models:

- **`fmtools.decorators`**:
  - `@local_extract`: Transforms standard functions into structured LLM extraction engines. Function docstrings act as system instructions.
  - `@enhanced_debug`: Catches exceptions and uses the local AI to provide a root-cause analysis and suggest fixes.
- **`fmtools.async_generators`**:
  - `stream_extract`: A high-concurrency asynchronous generator for processing massive text streams. Includes `history_mode` (clear, keep, hybrid, compact) for session memory management.
- **`fmtools.pipeline`**:
  - Provides a declarative, bitwise operator syntax (`Source >> Extract >> Sink`) for constructing ETL pipelines.
- **`fmtools.polars_ext`**:
  - Registers the `.local_llm` namespace in Polars, allowing `pl.col("...").local_llm.extract(Schema)` calls.
- **`fmtools.dspy_ext`**:
  - Implements `AppleFMLM`, a DSPy-compatible language model provider.
- **`fmtools.debugging`**:
  - Contains the structured AI analysis logic used by the `@enhanced_debug` decorator.
- **`fmtools.cache`**:
  - sqlite3 content-addressable extraction cache and cache-aware extraction helpers.
- **`fmtools.protocols`**:
  - Structural typing interfaces and swappable backend registry used by core extractors.
- **`fmtools.adapters`**:
  - Async adapters for file/stdin/CSV/JSONL/iterable/trio sources plus text chunking.
- **`fmtools._context`**:
  - `contextvars`-based per-task model/session scoping.
- **`fmtools._threading`**:
  - Free-threading detection and synchronization primitives.
- **`fmtools.scanner`**:
  - `mmap` sliding-window scanner for large file processing.
- **`fmtools.watcher`**:
  - Polling hot-folder daemon for auto-processing incoming files.
- **`fmtools._jit`**:
  - Runtime diagnostics, counters, and timing decorator.
- **`fmtools.arrow_bridge`**:
  - Arrow IPC file/buffer bridges and Polars conversion helpers.
- **`fmtools.functional`**:
  - Functional pipeline composition primitives.
- **`fmtools.auditor`**:
  - On-device code auditing and report formatting utilities.

---

## Building and Running

### Prerequisites
- **OS:** macOS 26.0+ (requires Apple Foundation Models support)
- **Hardware:** Apple Silicon (M1, M2, M3, M4 series)
- **Python:** CPython 3.13 or higher (3.14+ recommended for free-threading and JIT support)

### Setup & Installation
The project uses [`uv`](https://docs.astral.sh/uv/) for modern, fast dependency management.

```bash
# One-command setup (recommended)
./scripts/setup.sh

# Or manually:
uv sync --all-groups    # Creates venv, installs all deps (including Apple FM SDK from git)
source .venv/bin/activate
```

> The Apple FM SDK is not on PyPI. `uv sync` automatically clones and builds it from GitHub via `[tool.uv.sources]` in `pyproject.toml`.

### Key Commands

- **Run Tests:**
  ```bash
  uv run pytest tests/ -v
  ```
- **Linting & Formatting:**
  ```bash
  uv run ruff check .
  uv run ruff format .
  ```
- **Type Checking:**
  ```bash
  uv run ty check fmtools/
  ```
- **System Diagnostics:**
  ```bash
  ./scripts/doctor.sh
  ```
- **Run Examples:**
  ```bash
  uv run python use_cases/01_pipeline_operators/example.py
  ```
- **Stress Testing (Performance/Latency):**
  ```bash
  uv run python use_cases/07_stress_test_throughput/example.py
  ```

---

## Development Conventions

- **Type Safety:** Rigorous use of Python Type Annotations. Generics (`TypeVar`) are used extensively for schema-driven extraction.
- **Structured Generation:** All extractions MUST use a class decorated with `@apple_fm_sdk.generable()` to ensure valid JSON/Object outputs.
- **Async First:** The framework is built on `asyncio`. Use `await` and `async for` when interacting with the refinery.
- **Zero-Trust Principle:** Never include logic that sends data to external APIs. All processing must remain within the `apple_fm_sdk` local session.
- **Documentation:** Function docstrings are critical as they are often used as system prompts for the LLM.
- **Session Management:** During streaming, prioritize `history_mode='clear'` or `history_mode='compact'` to prevent `ExceededContextWindowSizeError` (approx 32k character limit).

---

## Use Cases & Examples

Detailed usage patterns are located in the `use_cases/` directory:
- `01_pipeline_operators`: Declarative ETL construction with `Source >> Extract >> Sink`.
- `02_decorators`: `@local_extract` decorator patterns for structured extraction.
- `03_async_generators`: `stream_extract` concurrent async streaming pipelines.
- `04_ecosystem_polars`: Massive parallel processing in Polars DataFrames.
- `05_dspy_optimization`: DSPy `AppleFMLM` provider for agentic workflows.
- `06_fastapi_integration`: Serving local AI as a REST microservice.
- `07_stress_test_throughput`: Performance profiling and throughput benchmarks.
- `08_context_limit_test`: Context window limit testing and payload escalation.
- `09_enhanced_debugging`: `@enhanced_debug` AI crash analysis decorator.
