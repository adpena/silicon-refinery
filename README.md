<div align="center">
  <h1>SiliconRefinery</h1>
  <p><b>A Zero-Trust, Zero-Latency, Zero-Cost ETL Framework for Apple Silicon</b></p>
  <p>
    <a href="https://pypi.org/project/silicon-refinery/"><img src="https://img.shields.io/pypi/v/silicon_refinery.svg" alt="PyPI Version"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.13%2B-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/macOS-26.0+-lightgrey.svg" alt="macOS Version">
    <img src="https://img.shields.io/badge/tests-400%2B%20passed-brightgreen.svg" alt="Tests">
  </p>
</div>

**SiliconRefinery** is an enterprise-grade Python framework built on the [Apple Foundation Models SDK (`python-apple-fm-sdk`)](https://github.com/apple/python-apple-fm-sdk). It transforms your Apple Silicon machine into a private, high-throughput data extraction node — processing unstructured text into guaranteed-schema objects without ever sending a single byte to the cloud.

```python
from silicon_refinery import local_extract

@local_extract(schema=SupportTicket, debug_timing=True)
async def classify(email: str) -> SupportTicket:
    """Classify a customer support email by category, urgency, and summary."""

ticket = await classify("I was charged twice and I need a refund!")
# -> SupportTicket(category="Billing", urgency="HIGH", summary="Duplicate charge refund request")
```

No API keys. No cloud calls. No token costs. Everything runs on the Neural Engine.

---

## Table of Contents

- [Why SiliconRefinery?](#why-siliconrefinery)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Public API Breakdown](#public-api-breakdown)
- [API Reference & Tutorials](#api-reference--tutorials)
  - [`@local_extract` — Structured Extraction Decorator](#1-local_extract--structured-extraction-decorator)
  - [`stream_extract` — Concurrent Async Streaming](#2-stream_extract--concurrent-async-streaming)
  - [`Source >> Extract >> Sink` — Composable Pipelines](#3-source--extract--sink--composable-pipelines)
  - [`@enhanced_debug` — AI Crash Analysis](#4-enhanced_debug--ai-crash-analysis)
  - [Polars `.local_llm` Extension](#5-polars-local_llm-extension)
  - [DSPy `AppleFMLM` Provider](#6-dspy-applefmlm-provider)
  - [FastAPI Integration](#7-fastapi-integration)
- [Use Cases & Examples](#use-cases--examples)
- [Sample Datasets](#sample-datasets)
- [Benchmarks & Empirical Results](#benchmarks--empirical-results)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [Gratitude to Apple](#gratitude-to-apple)
- [License](#license)

---

## Why SiliconRefinery?

Enterprises sit on petabytes of "dark data" — unstructured logs, CSV dumps, support emails, medical records. Processing this data intelligently has historically meant sending it to cloud LLM providers, incurring massive token costs and compliance nightmares (GDPR, HIPAA, SOC 2).

SiliconRefinery flips the paradigm:

| Problem | SiliconRefinery's Answer |
|---|---|
| **Data leaves your network** | All inference runs on the local Neural Engine. Zero egress. |
| **API costs scale with volume** | Zero token costs. Process millions of records for free. |
| **Schema validation is fragile** | Apple's `@generable()` protocol guarantees schema-valid outputs at the model level. |
| **Async complexity** | Idiomatic `asyncio` patterns — decorators, generators, pipelines — all async-native. |
| **Context window explosion** | Built-in session management: clear, keep, hybrid, and compact history modes. |
| **No DataFrame integration** | First-class Polars extension via `.local_llm.extract()`. |

**Measured throughput:** 250-350+ characters/sec (~60-90 tokens/sec) on Apple M1, purely on-device. See [Benchmarks](#benchmarks--empirical-results).

---

## Installation

### Requirements

- macOS 26.0+ (Tahoe or later)
- Apple Silicon (M1, M2, M3, M4 series)
- CPython 3.13+

### One-command setup (recommended)

```bash
git clone https://github.com/adpena/silicon-refinery && cd silicon-refinery
./scripts/setup.sh
```

This installs [uv](https://docs.astral.sh/uv/) if missing, creates a virtual environment, syncs dependency groups (including `apple-fm-sdk` via `tool.uv.sources`), and installs the `silicon-refinery` CLI globally in editable mode for local development. Run `./scripts/doctor.sh` afterwards to verify your system meets all requirements.

Useful setup flags:

```bash
./scripts/setup.sh --no-sdk          # Skip Apple SDK dependency group
./scripts/setup.sh --no-cli-install  # Skip global CLI install
```

> **Important behavior:** `uv sync`, `uv run`, and `uv pip install` do **not** execute `./scripts/setup.sh` automatically. `setup.sh` is a convenience bootstrap script that you run explicitly.

### Manual step-by-step

```bash
# 1. Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone SiliconRefinery
git clone https://github.com/adpena/silicon-refinery && cd silicon-refinery

# 3. Create venv and install all dependencies (including Apple FM SDK from git)
uv sync --all-groups
source .venv/bin/activate
```

> **Note:** The Apple FM SDK is not yet on PyPI. `uv sync` automatically clones and builds it from GitHub via the `[tool.uv.sources]` configuration in `pyproject.toml`.

### Command availability & PATH

Use whichever mode best fits your workflow:

| Mode | Command example | Best for |
|---|---|---|
| No activation required | `uv run silicon-refinery chat` | Most reliable local dev from repo root |
| Activated venv | `silicon-refinery chat` (after `source .venv/bin/activate`) | Traditional Python workflow |
| Global CLI install | `silicon-refinery chat` (after `uv tool install --editable --from . silicon-refinery`) | Seamless command usage across shells |

If `silicon-refinery` is not found after setup, open a new terminal (or run `source ~/.zshrc`) so your shell reloads PATH changes.

For chat UX/perf, `silicon-refinery chat` now prefers free-threaded CPython (`3.14t`, then `3.13t`) by default and gracefully falls back to standard-GIL only if a no-GIL runtime is unavailable.

Standalone desktop app repo: [adpena/silicon-refinery-chat](https://github.com/adpena/silicon-refinery-chat) (`SiliconRefineryChat`).

### Keep dependencies updated

```bash
# Refresh root lockfile to newest compatible versions
uv lock --upgrade
uv sync --all-groups

# Refresh Toga demo lockfile and env
uv lock --project examples/toga_local_chat_app --directory examples/toga_local_chat_app --upgrade
uv sync --project examples/toga_local_chat_app --directory examples/toga_local_chat_app
```

### PyPI and Homebrew install story

- **PyPI:** `silicon-refinery` publishes package metadata and the CLI entry point, but full on-device runtime support still depends on `apple-fm-sdk` (currently pulled from GitHub, not PyPI).
- **Recommended install path today:** clone this repo and run `./scripts/setup.sh` (or `uv sync --all-groups`) so the Apple SDK source dependency is resolved correctly.
- **Homebrew tap:** `adpena/homebrew-silicon-refinery`

```bash
# Tap once
brew tap adpena/silicon-refinery https://github.com/adpena/homebrew-silicon-refinery

# Install CLI formula
brew install --HEAD adpena/silicon-refinery/silicon-refinery

# Install standalone desktop app (.app via cask)
brew install --cask adpena/silicon-refinery/silicon-refinery-chat

# Verify
silicon-refinery --help
```

### Homebrew maintainer release flow

When shipping a new release and updating Homebrew support:

```bash
# 1) Ensure main branch has the new release commit/tag pushed
git push origin main --tags

# 2) Sync Formula + Cask into the tap repo and push
#    (tap repo: adpena/homebrew-silicon-refinery)
#    - Formula/silicon-refinery.rb
#    - Casks/silicon-refinery-chat.rb

# 3) Validate the installed command
silicon-refinery --help
silicon-refinery doctor
brew install --cask adpena/silicon-refinery/silicon-refinery-chat
```

Current CLI formula is HEAD-based (`Formula/silicon-refinery.rb`) while `apple-fm-sdk` remains git-sourced. The app installer is distributed as a cask (`Casks/silicon-refinery-chat.rb`) backed by release DMG assets from `adpena/silicon-refinery-chat`.

### Standalone Chat Repo release flow

The standalone macOS app repository (`silicon-refinery-chat`) is synced and packaged from this repo automatically.

```bash
# Local/manual sync (create-or-update repo, sync source/docs, build artifact, upload release asset)
./scripts/publish_chat_repo.sh --repo adpena/silicon-refinery-chat
```

GitHub Actions automation lives at [`.github/workflows/publish-chat-repo.yml`](.github/workflows/publish-chat-repo.yml) and runs on release publish. Configure secret `CHAT_REPO_GH_TOKEN` (a PAT with repo write access to `adpena/silicon-refinery-chat`) before enabling cross-repo publish.

### Optional dependencies

```bash
# For FastAPI/Uvicorn integration (use_cases/06_fastapi_integration)
uv pip install silicon-refinery[api]

# For trio-native adapters (TrioAdapter)
uv pip install silicon-refinery[adapters]

# For Arrow IPC examples/integration
uv pip install silicon-refinery[arrow]

# For the comprehensive marimo examples notebook
uv pip install marimo
```

> **Important:** The `api`, `adapters`, and `arrow` extras do not install `apple-fm-sdk`. Install base/dev dependencies first (for example, `uv sync --all-groups`) so `silicon_refinery` can import and run.

### CLI

With any command mode above (`uv run`, activated `.venv`, or global tool install), the `silicon-refinery` CLI supports:

If you have not activated `.venv` (or do not have global tool install), use `uv run` as a prefix, for example: `uv run silicon-refinery chat`.

```bash
silicon-refinery setup      # First-time dev environment setup
silicon-refinery install-homebrew  # Install/update via local Homebrew tap
silicon-refinery doctor     # Check all system prerequisites
silicon-refinery lint       # Run ruff linter
silicon-refinery format     # Run ruff formatter
silicon-refinery typecheck  # Run ty type checker
silicon-refinery test       # Run the full test suite (460+ tests)
silicon-refinery check      # Run lint + format + typecheck + tests (CI pipeline)
silicon-refinery example --list   # List standalone examples
silicon-refinery example simple_inference  # Run one example script
silicon-refinery smoke      # Run full examples smoke suite (+ notebook startup check)
silicon-refinery notebook   # Launch the comprehensive marimo notebook
silicon-refinery chat       # Run the Toga + Briefcase local chat demo
silicon-refinery chat --standard-gil  # Force standard-gil runtime for max GUI stability
```

### Verify your installation

```bash
# Quick system check
silicon-refinery doctor
```

```python
import silicon_refinery
import apple_fm_sdk as fm

model = fm.SystemLanguageModel()
available, reason = model.is_available()
print(f"Neural Engine available: {available}")  # -> True on macOS 26+ with Apple Silicon
```

---

## Quick Start

Three steps to go from unstructured text to a validated Python object:

**1. Define a schema** using the Apple FM SDK's `@generable()` decorator:

```python
import apple_fm_sdk as fm

@fm.generable()
class SupportTicket:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str  = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    summary: str  = fm.guide(description="One-sentence summary of the issue")
```

**2. Decorate a function** with `@local_extract`. The function's **docstring becomes the system prompt**:

```python
from silicon_refinery import local_extract

@local_extract(schema=SupportTicket, debug_timing=True)
async def classify_ticket(email_text: str) -> SupportTicket:
    """Classify a customer support email by category, urgency, and summary."""
```

**3. Call it:**

```python
import asyncio

async def main():
    ticket = await classify_ticket(
        "I was charged twice this month and I need a refund immediately!"
    )
    print(ticket.category)  # "Billing"
    print(ticket.urgency)   # "HIGH"
    print(ticket.summary)   # "Customer requests refund for duplicate charge"

asyncio.run(main())
```

That's it. The decorated function intercepts your arguments, sends them to the on-device model, enforces the schema via `@generable()`, and returns a fully-validated `SupportTicket` object.

---

## Public API Breakdown

SiliconRefinery follows a layered API design used by mature Python OSS projects:
- a small, stable root import surface for day-to-day usage
- module-level APIs for advanced workflows
- a CLI for reproducible local development workflows

### Stable root imports (`silicon_refinery`)

```python
from silicon_refinery import (
    AppleFMSetupError,
    Extract,
    Sink,
    Source,
    enhanced_debug,
    local_extract,
    stream_extract,
)
```

| Symbol | Kind | Signature | Primary use |
|---|---|---|---|
| `local_extract` | Decorator factory | `(schema, retries=3, debug_timing=False)` | Turn an async function into structured on-device extraction |
| `stream_extract` | Async generator | `(source_iterable, schema, ..., concurrency=None)` | High-throughput extraction over streams |
| `Source`, `Extract`, `Sink` | Pipeline nodes | `Source(iterable) >> Extract(schema) >> Sink(callback)` | Declarative ETL pipelines |
| `enhanced_debug` | Decorator factory | `(route_to=\"stdout\", prompt_file=None)` | AI-assisted crash analysis |
| `AppleFMSetupError` | Exception | setup/runtime exception type | Consistent SDK/model troubleshooting diagnostics |

### Module-level API index

| Module | Primary public API | Role |
|---|---|---|
| `silicon_refinery.cache` | `ExtractionCache`, `cache_extract`, `cached_stream_extract`, `cached_local_extract` | sqlite-backed extraction caching |
| `silicon_refinery.protocols` | `ModelProtocol`, `SessionProtocol`, `SessionFactory`, `AppleFMBackend`, `set_backend`, `get_backend`, `create_model`, `create_session` | Pluggable backend contracts wired into core extractors |
| `silicon_refinery.adapters` | `FileAdapter`, `CSVAdapter`, `JSONLAdapter`, `StdinAdapter`, `IterableAdapter`, `TrioAdapter`, `TextChunkAdapter` | Async input adapters (including trio-style channels) |
| `silicon_refinery._context` | `session_scope`, `get_session`, `get_model`, `get_instructions`, `copy_context` | Task-local session scoping |
| `silicon_refinery._threading` | `is_free_threaded`, `get_gil_status`, `CriticalSection`, `AtomicCounter`, `ThreadSafeDict` | Free-threading safety primitives |
| `silicon_refinery.scanner` | `MMapScanner`, `line_split_scanner` | Large-file scanning with overlap-safe windows |
| `silicon_refinery.watcher` | `HotFolder`, `FileEvent`, `process_folder` | Hot-folder ingestion daemon |
| `silicon_refinery._jit` | `diagnostics`, `diagnose`, `DiagnosticCollector`, `ExtractionMetrics` | Runtime diagnostics and metrics |
| `silicon_refinery.arrow_bridge` | `to_arrow_ipc`, `from_arrow_ipc`, `to_arrow_ipc_buffer`, `from_arrow_ipc_buffer`, `ArrowStreamWriter`, `to_polars`, `from_polars` | Arrow/Polars interoperability |
| `silicon_refinery.functional` | `pipe`, `source`, `extract`, `map_fn`, `filter_fn`, `flat_map_fn`, `batch`, `take`, `skip`, `tap`, `collect`, `reduce_fn` | Functional pipeline composition |
| `silicon_refinery.auditor` | `audit_file`, `audit_directory`, `audit_diff`, `format_audit_report` | On-device code auditing with bounded-concurrency directory scans |
| `silicon_refinery.exceptions` | `AppleFMSetupError`, `troubleshooting_message`, `require_apple_fm`, `ensure_model_available` | Shared setup validation and graceful diagnostics |

### CLI API (`silicon-refinery`)

Core commands:
- `setup`, `install-homebrew`, `doctor`
- `lint`, `format`, `typecheck`, `test`, `check`
- `run` (execute numbered use cases)
- `example` (list/run standalone scripts under `examples/`)
- `smoke` (run full examples smoke suite with SDK preflight + timeouts)
- `notebook` (launch `examples/examples_notebook.py` via marimo)
- `chat` (launch Toga + Briefcase desktop demo)
- `completions` (shell completion setup)

---

## API Reference & Tutorials

SiliconRefinery exposes seven foundational capabilities. Each one is documented below with its full function signature, parameter reference, behavior details, and a working example.

### 1. `@local_extract` — Structured Extraction Decorator

**Module:** `silicon_refinery.decorators`
**Import:** `from silicon_refinery import local_extract`

Transforms any Python function into an on-device LLM extraction engine. The function's docstring serves as the system prompt.

#### Signature

```python
@local_extract(schema: type[T], retries: int = 3, debug_timing: bool = False)
async def your_function(*args, **kwargs) -> T:
    """Your system prompt goes here as the docstring."""
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schema` | `type[T]` | *(required)* | A class decorated with `@fm.generable()`. The model's output is constrained to this shape. |
| `retries` | `int` | `3` | Number of retry attempts for transient errors (`TimeoutError`, `ConnectionError`, `OSError`). Non-transient errors (e.g., `TypeError`, `ValueError`) fail immediately. |
| `debug_timing` | `bool` | `False` | When `True`, logs extraction time and input length to the `silicon_refinery` logger. |

#### How it works

1. On the first call, the decorator lazily creates and caches a backend model via `create_model()`
2. A fresh backend session is created per call via `create_session(...)` with the docstring as `instructions`
3. All positional arguments are joined with spaces; keyword arguments are appended as `\nkey: value`
4. `session.respond(input_text, generating=schema)` invokes the Neural Engine
5. Transient errors trigger exponential backoff retries (`0.1s`, `0.2s`, `0.4s`, ...); non-transient errors propagate immediately

#### Full example — Medical triage extraction

This example reads the synthetic `medical_notes.csv` dataset and extracts structured triage data from each raw dictation note.

```python
import asyncio
import csv
import apple_fm_sdk as fm
from silicon_refinery import local_extract

@fm.generable()
class MedicalRecord:
    patient_symptoms: list[str] = fm.guide(description="List of isolated symptoms")
    suggested_triage: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    duration_days: int = fm.guide(
        description="How many days the symptoms have lasted. 0 if not mentioned."
    )

@local_extract(schema=MedicalRecord, debug_timing=True)
async def parse_doctor_notes(raw_text: str) -> MedicalRecord:
    """Extract structured medical data from raw dictated notes.
    Ensure triage urgency is inferred correctly based on symptom severity."""

async def main():
    with open("datasets/medical_notes.csv", newline="") as f:
        for row in csv.DictReader(f):
            record = await parse_doctor_notes(row["raw_note"])
            print(f"Triage: {record.suggested_triage} | Symptoms: {record.patient_symptoms}")

asyncio.run(main())
```

**Run it:** `python use_cases/02_decorators/example.py`

---

### 2. `stream_extract` — Concurrent Async Streaming

**Module:** `silicon_refinery.async_generators`
**Import:** `from silicon_refinery import stream_extract`

An asynchronous generator that processes massive data streams through the local model, yielding structured objects one at a time. Supports line-level chunking, four session history modes, and concurrent `imap_unordered`-style parallel extraction.

#### Signature

```python
async def stream_extract(
    source_iterable: Iterable | AsyncIterable,
    schema: type[T],
    instructions: str = "Extract data.",
    lines_per_chunk: int = 1,
    history_mode: Literal["clear", "keep", "hybrid", "compact"] = "clear",
    concurrency: int | None = None,
    debug_timing: bool = False,
) -> AsyncGenerator[T, None]:
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `source_iterable` | `Iterable` or `AsyncIterable` | *(required)* | The data source. Can be a list, generator, file reader, or any async iterable. |
| `schema` | `type[T]` | *(required)* | An `@fm.generable()` class for structured output. |
| `instructions` | `str` | `"Extract data."` | System prompt for the model session. |
| `lines_per_chunk` | `int` | `1` | Groups N input items into a single chunk before sending to the model. Useful for batch extraction. |
| `history_mode` | `str` | `"clear"` | Session history management strategy (see table below). |
| `concurrency` | `int \| None` | `min(cpu_count, 4)` | Number of parallel extraction tasks. If >1, forces `history_mode="clear"`. |
| `debug_timing` | `bool` | `False` | Logs processing time and throughput per chunk. |

#### History modes

| Mode | Behavior | Best for |
|---|---|---|
| `clear` | Fresh session per chunk. Zero memory accumulation. | Large/infinite streams, concurrent processing |
| `keep` | Retains session history across chunks. May throw `ExceededContextWindowSizeError`. | Short sequences where context matters |
| `hybrid` | Like `keep`, but automatically clears and retries on context overflow. | Medium streams with contextual benefit |
| `compact` | Like `keep`, but summarizes history when limits are approached. | Long streams where context is valuable |

#### How concurrency works

When `concurrency > 1`, `stream_extract` operates like Python's `multiprocessing.Pool.imap_unordered()`:

1. Up to `concurrency` tasks run simultaneously via `asyncio.create_task()`
2. Each task gets its own isolated `LanguageModelSession`
3. Results are yielded as they complete (out-of-order)
4. On error or generator close, all pending tasks are cancelled via `try/finally`

Default concurrency is `min(os.cpu_count(), 4)` — the Neural Engine doesn't scale linearly with CPU cores.

#### Full example — Product review sentiment analysis

```python
import asyncio
import csv
import apple_fm_sdk as fm
from silicon_refinery import stream_extract

@fm.generable()
class Feedback:
    sentiment: str = fm.guide(anyOf=["Positive", "Neutral", "Negative"])
    key_feature: str = fm.guide(description="Primary feature or aspect discussed")

def yield_reviews(filepath):
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            yield row["review_text"]

async def main():
    async for enriched in stream_extract(
        yield_reviews("datasets/product_reviews.csv"),
        schema=Feedback,
        instructions="Analyze user feedback and extract sentiment.",
        concurrency=4,
        debug_timing=True,
    ):
        print(f"[{enriched.sentiment:8}] Focus: {enriched.key_feature}")

asyncio.run(main())
```

**Run it:** `python use_cases/03_async_generators/example.py`

---

### 3. `Source >> Extract >> Sink` — Composable Pipelines

**Module:** `silicon_refinery.pipeline`
**Import:** `from silicon_refinery import Source, Extract, Sink`

A declarative ETL pipeline using Python's `>>` operator. Data flows from `Source` through `Extract` (LLM inference) into `Sink` (output callback). The pipeline streams results without buffering.

#### Classes

**`Source(iterable)`** — Wraps any iterable as the pipeline entry point.

**`Extract(schema, instructions="...", on_error="skip")`** — The LLM processing node.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schema` | `type[T]` | *(required)* | An `@fm.generable()` class. |
| `instructions` | `str` | `"Process and structure this input."` | System prompt. |
| `on_error` | `str` | `"skip"` | Error handling: `"skip"` silently drops failed items, `"raise"` propagates the error, `"yield_none"` yields `None`. |

**`Sink(callback)`** — Receives each processed item. Accepts both sync and async callables.

**`Pipeline`** — Created automatically by chaining nodes with `>>`.

| Method | Returns | Description |
|---|---|---|
| `pipeline.execute()` | `AsyncGenerator` | Streams results one at a time. Use `async for item in pipeline.execute()`. |
| `pipeline.collect()` | `list` | Materializes all results into a list. Convenience method. |

#### Full example — Server log parsing

```python
import asyncio
import csv
import apple_fm_sdk as fm
from silicon_refinery import Source, Extract, Sink

@fm.generable()
class LogEntry:
    level: str = fm.guide(anyOf=["INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG"])
    module: str = fm.guide(description="The service or module emitting the log")
    message: str = fm.guide(description="The core description of the event")

def read_logs(filepath):
    with open(filepath, newline="") as f:
        for row in csv.DictReader(f):
            yield row["log_message"]

async def main():
    pipeline = (
        Source(read_logs("datasets/server_logs.csv"))
        >> Extract(schema=LogEntry, instructions="Parse the raw server log string.")
        >> Sink(callback=lambda item: print(
            f"Level: {item.level:7} | Module: {item.module:15} | Message: {item.message}"
        ))
    )

    # Stream results one at a time (no buffering)
    async for _ in pipeline.execute():
        pass

    # Or collect all at once:
    # results = await pipeline.collect()

asyncio.run(main())
```

**Run it:** `python use_cases/01_pipeline_operators/example.py`

---

### 4. `@enhanced_debug` — AI Crash Analysis

**Module:** `silicon_refinery.debugging`
**Import:** `from silicon_refinery import enhanced_debug`

A decorator that catches exceptions, prints the standard Python traceback, and then invokes the Neural Engine to perform an automated root-cause analysis. Works with both sync and async functions.

#### Signature

```python
@enhanced_debug(route_to: str = "stdout", prompt_file: str | None = None)
def your_function(...):
    ...
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `route_to` | `str` | `"stdout"` | Where to print the AI analysis: `"stdout"` or `"log"` (via `logging.error`). |
| `prompt_file` | `str \| None` | `None` | If set, writes a detailed prompt payload to this file — ready to paste into Claude, Codex, or any coding assistant for a deeper fix. |

#### What happens when the function crashes

1. The standard Python traceback is printed to stderr
2. The traceback is sent to the on-device Foundation Model with expert analysis instructions
3. A structured `DebuggingAnalysis` is generated containing:
   - `error_summary` — Brief description of what went wrong
   - `possible_causes` — List of likely root causes
   - `certainty_level` — `"LOW"`, `"MEDIUM"`, or `"HIGH"`
   - `suggested_fix` — Actionable steps to resolve the issue
4. If `prompt_file` is set, a formatted prompt file is written for use with external coding agents
5. The original exception is re-raised (the decorator never swallows exceptions)

#### Sample AI analysis output

```
==================================================
SiliconRefinery AI Debug Analysis (Certainty: HIGH)
==================================================
Summary: The error occurs because data_payload contains a string, and attempting
to add an integer to it results in a TypeError.

Possible Causes:
  1. data_payload is not being correctly initialized or parsed as an integer.
  2. Data payload format is incorrect, leading to unexpected types.

Suggested Fix: Ensure data_payload is parsed correctly as an integer before
performing arithmetic operations.
==================================================
```

A [sample generated prompt file](use_cases/09_enhanced_debugging/sample_crash_report.txt) is included in the repository — ready to paste into Claude or any coding assistant.

#### Full example

```python
from silicon_refinery import enhanced_debug

@enhanced_debug(route_to="stdout", prompt_file="crash_report_for_llm.txt")
def process_data(data_payload):
    """A buggy function that will inevitably crash."""
    parsed_value = data_payload["value"] + 10  # TypeError!
    return parsed_value

try:
    process_data({"value": "100"})
except TypeError:
    print("Execution continued gracefully after AI analysis.")
```

**Run it:** `python use_cases/09_enhanced_debugging/example.py`

---

### 5. Polars `.local_llm` Extension

**Module:** `silicon_refinery.polars_ext`
**Import:** `from silicon_refinery.polars_ext import LocalLLMExpr`

Registers the `.local_llm` namespace directly onto Polars expressions, allowing you to run on-device LLM inference inside `df.select()` or `df.with_columns()` calls. Each row gets its own `LanguageModelSession` (preventing context window explosion), and rows within a batch are processed concurrently via `asyncio.Semaphore(4)`.

#### Usage

```python
import polars as pl
from silicon_refinery.polars_ext import LocalLLMExpr  # registers the namespace

enriched_df = df.with_columns(
    extracted_json=pl.col("text_column").local_llm.extract(
        schema=YourSchema,
        instructions="Your system prompt here."
    )
)
```

The result column contains JSON strings. Parse them with `pl.col("extracted_json").str.json_decode()` or similar.

#### Implementation details

- Uses a persistent background thread running its own `asyncio` event loop (avoids `asyncio.run()` conflicts with Polars' thread model)
- `asyncio.Semaphore(4)` limits concurrent Neural Engine calls within each batch
- `None` values in the input column produce `null` in the output
- Results that don't support `vars()` fall back to `{"_raw": str(result)}`

#### Full example — Support ticket classification in a DataFrame

```python
import polars as pl
import apple_fm_sdk as fm
from silicon_refinery.polars_ext import LocalLLMExpr

@fm.generable()
class Ticket:
    department: str = fm.guide(anyOf=["IT", "HR", "Sales", "Billing", "Other"])
    urgency: int = fm.guide(description="Scale 1 to 5, where 5 is critical")

df = pl.read_csv("datasets/support_tickets.csv")

enriched_df = df.with_columns(
    extracted_json=pl.col("email_body").local_llm.extract(schema=Ticket)
)
print(enriched_df.select(["ticket_id", "email_subject", "extracted_json"]))
```

**Run it:** `python use_cases/04_ecosystem_polars/example.py`

---

### 6. DSPy `AppleFMLM` Provider

**Module:** `silicon_refinery.dspy_ext`
**Import:** `from silicon_refinery.dspy_ext import AppleFMLM`

A custom `dspy.LM` subclass that routes all inference through the local Apple Foundation Model. This lets you use DSPy's full suite of prompt compilers, Chain-of-Thought reasoning, and agentic workflows — all running on free Apple hardware with zero cloud dependency.

#### Usage

```python
import dspy
from silicon_refinery.dspy_ext import AppleFMLM

dspy.settings.configure(lm=AppleFMLM())

# Now any DSPy module uses the local Neural Engine
classifier = dspy.ChainOfThought("customer_email -> summary, priority")
result = classifier(customer_email="I am locked out of my account!")
print(result.summary, result.priority)
```

#### Implementation details

- Supports both `prompt` (string) and `messages` (list of dicts) input formats for DSPy v2.5+ compatibility
- History is stored in a bounded `collections.deque(maxlen=1000)` to prevent unbounded memory growth
- Handles async-to-sync bridging: detects if an event loop is already running and uses `concurrent.futures.ThreadPoolExecutor` to avoid `asyncio.run()` conflicts

#### Full example — Chain-of-Thought support ticket analysis

```python
import csv
import dspy
from silicon_refinery.dspy_ext import AppleFMLM

class SupportClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought("customer_email -> summary, priority")

    def forward(self, email):
        return self.analyze(customer_email=email)

dspy.settings.configure(lm=AppleFMLM())
classifier = SupportClassifier()

with open("datasets/support_tickets.csv", newline="") as f:
    for row in csv.DictReader(f):
        result = classifier(row["email_body"])
        print(f"Ticket {row['ticket_id']}: {result.summary} [{result.priority}]")
```

**Run it:** `python use_cases/05_dspy_optimization/example.py`

---

### 7. FastAPI Integration

**Module:** Uses `silicon_refinery.decorators` with FastAPI
**Install:** `uv pip install silicon-refinery[api]`

Turn SiliconRefinery into a local REST API microservice. Because `@local_extract` returns an `async` function, it integrates natively with FastAPI's async request handling.

#### Full example — Document extraction API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import apple_fm_sdk as fm
from silicon_refinery import local_extract

app = FastAPI(title="SiliconRefinery Extraction API")

class ExtractionRequest(BaseModel):
    document_text: str

@fm.generable()
class ExtractedEntity:
    primary_topic: str = fm.guide(description="Main subject of the document")
    sentiment_score: int = fm.guide(range=(1, 10), description="1=negative, 10=positive")
    entities: list[str] = fm.guide(description="Key entities mentioned")

@local_extract(schema=ExtractedEntity, debug_timing=True)
async def process_document(raw_text: str) -> ExtractedEntity:
    """Analyze the text to extract the primary topic, sentiment score, and key entities."""

@app.post("/api/v1/extract")
async def extract_data(request: ExtractionRequest):
    result = await process_document(request.document_text)
    return {
        "primary_topic": result.primary_topic,
        "sentiment_score": result.sentiment_score,
        "entities": result.entities,
    }

# Run: uvicorn example:app --host 0.0.0.0 --port 8000
```

**Run it:** `python use_cases/06_fastapi_integration/example.py`

---

## Use Cases & Examples

The repository includes 9 use cases plus a broad standalone example catalog (14 scripts, a marimo notebook, and a desktop app). Each use case is self-contained with its own `example.py`.

| # | Directory | What it demonstrates | Key API |
|---|---|---|---|
| 01 | [`use_cases/01_pipeline_operators/`](use_cases/01_pipeline_operators/) | Declarative ETL with `>>` operator | `Source`, `Extract`, `Sink` |
| 02 | [`use_cases/02_decorators/`](use_cases/02_decorators/) | Medical triage from dictated notes | `@local_extract` |
| 03 | [`use_cases/03_async_generators/`](use_cases/03_async_generators/) | Streaming sentiment analysis | `stream_extract` |
| 04 | [`use_cases/04_ecosystem_polars/`](use_cases/04_ecosystem_polars/) | DataFrame-native LLM inference | `.local_llm.extract()` |
| 05 | [`use_cases/05_dspy_optimization/`](use_cases/05_dspy_optimization/) | Chain-of-Thought with DSPy | `AppleFMLM` |
| 06 | [`use_cases/06_fastapi_integration/`](use_cases/06_fastapi_integration/) | Local REST API microservice | `@local_extract` + FastAPI |
| 07 | [`use_cases/07_stress_test_throughput/`](use_cases/07_stress_test_throughput/) | Throughput profiling (1000 records) | `stream_extract` |
| 08 | [`use_cases/08_context_limit_test/`](use_cases/08_context_limit_test/) | Context window limit probing | `@local_extract` |
| 09 | [`use_cases/09_enhanced_debugging/`](use_cases/09_enhanced_debugging/) | AI crash analysis with sample output | `@enhanced_debug` |

### Standalone examples

| File | Description |
|---|---|
| [`examples/simple_inference.py`](examples/simple_inference.py) | Basic Apple FM SDK session and response |
| [`examples/streaming_example.py`](examples/streaming_example.py) | Streaming token-by-token response |
| [`examples/transcript_processing.py`](examples/transcript_processing.py) | Analyzing transcripts exported from Swift apps (defaults to bundled sample JSON) |
| [`examples/extraction_cache.py`](examples/extraction_cache.py) | sqlite-backed extraction cache + decorator usage |
| [`examples/functional_pipeline.py`](examples/functional_pipeline.py) | Functional pipeline composition with optional extraction |
| [`examples/custom_backend.py`](examples/custom_backend.py) | Runtime backend swapping via protocol registry |
| [`examples/trio_adapter.py`](examples/trio_adapter.py) | Trio-style receive channel adapter |
| [`examples/context_scope.py`](examples/context_scope.py) | Context-local model/session scoping |
| [`examples/free_threading.py`](examples/free_threading.py) | Free-threading helpers and safe shared state |
| [`examples/mmap_scanner.py`](examples/mmap_scanner.py) | Memory-mapped sliding-window and line scanners |
| [`examples/hot_folder_watcher.py`](examples/hot_folder_watcher.py) | Hot-folder polling watcher for ingestion workflows |
| [`examples/jit_diagnostics.py`](examples/jit_diagnostics.py) | Runtime diagnostics collector and `@diagnose` decorator |
| [`examples/arrow_bridge.py`](examples/arrow_bridge.py) | Arrow IPC file/buffer/stream round trips |
| [`examples/code_auditor.py`](examples/code_auditor.py) | On-device code auditing APIs |
| [`examples/examples_notebook.py`](examples/examples_notebook.py) | Comprehensive marimo notebook covering `examples/` and `use_cases/*/example.py` |
| [`examples/toga_local_chat_app/`](examples/toga_local_chat_app/) | `SiliconRefineryChat`: Toga + Briefcase local chat app with vertical chat tabs, steering interjection reruns, sqlite memory, and `/help` `/new` `/clear` `/export` |

All SDK-dependent examples and wrapper APIs use `AppleFMSetupError` for
consistent, actionable setup failures.

See [`examples/README.md`](examples/README.md) for prerequisites and run commands.

---

## Sample Datasets

All datasets are **synthetic** — generated for demonstration purposes with no real data. Located in [`datasets/`](datasets/).

| File | Schema | Used by | Description |
|---|---|---|---|
| `server_logs.csv` | `log_id, timestamp, log_message` | Use case 01 | Simulated server log entries with various severity levels |
| `medical_notes.csv` | `id, date, raw_note` | Use case 02 | Fictional doctor dictation notes for triage classification |
| `product_reviews.csv` | `review_id, product, review_text` | Use case 03 | Synthetic product reviews for sentiment analysis |
| `support_tickets.csv` | `ticket_id, email_subject, email_body` | Use cases 04, 05 | Simulated customer support emails |
| `transcript_sample.json` | Foundation Models transcript JSON | `examples/transcript_processing.py` | Synthetic transcript export sample for transcript analytics |

See [`datasets/README.md`](datasets/README.md) for details. Datasets were created with `scripts/generate_datasets.py`.

---

## Benchmarks & Empirical Results

We don't just build; we measure. Two dedicated use cases profile SiliconRefinery's performance characteristics.

### Test environment

| Component | Specification |
|---|---|
| **Hardware** | Apple M1 (MacBookPro17,1) — 8 Cores (4P + 4E), 8GB Unified Memory |
| **OS** | macOS 26.3 (Build 25D125) |
| **Python** | 3.14.3 |
| **SDK** | `python-apple-fm-sdk` 0.1.0 (Beta) |

### Throughput (use case 07)

Running 1,000 unstructured records through `stream_extract` with `lines_per_chunk=5`:

| Metric | Result |
|---|---|
| **Characters/sec** | 250–350+ |
| **Tokens/sec** (est.) | ~60–90 |
| **Records processed** | 1,000 |
| **Cloud cost** | $0.00 |

> Tokens/sec estimated using the standard ~4 characters/token approximation. Actual tokenization ratios vary by content.

**Reproduce it:** `python use_cases/07_stress_test_throughput/example.py`

### Context window limits (use case 08)

Payload escalation test — progressively larger input texts:

| Input size | Approx. tokens | Result | Time |
|---|---|---|---|
| 5,000 chars | ~1,250 | Success | 1.2s |
| 25,000 chars | ~6,250 | Success | 12.5s |
| 50,000 chars | ~12,500 | Success | ~25s |
| ~32,000+ chars | ~8,000+ | `ExceededContextWindowSizeError` | — |

The context window limit is approximately 32,000 characters. This is why `stream_extract` defaults to `history_mode="clear"` — recreating the session per chunk prevents context accumulation on infinite streams.

**Reproduce it:** `python use_cases/08_context_limit_test/example.py`

### Decorator overhead

Measured in the test suite (`test_decorators.py::TestLocalExtractPerformance`): the `@local_extract` wrapper adds **<5ms of overhead** per call, exclusive of model inference time.

---

## Architecture & Design Decisions

### Why docstrings as system prompts?

The `@local_extract` decorator uses the wrapped function's docstring as the system prompt. This is intentional:

1. **Documentation IS the prompt** — you read the docstring and understand exactly what the model is being asked to do
2. **IDE support** — hover over the function in any editor and see the prompt
3. **Version control** — prompt changes are tracked in git diffs, not hidden in config files
4. **Testing** — you can assert on `func.__doc__` in tests

### Why fresh sessions per item?

Both `@local_extract` and the `Extract` pipeline node create a new `LanguageModelSession` per invocation. The `stream_extract` generator defaults to `history_mode="clear"` (fresh session per chunk). This trades context continuity for safety:

- No risk of `ExceededContextWindowSizeError` on large streams
- Each extraction is independent and reproducible
- Concurrent processing is possible (each task gets its own session)

When context matters, use `history_mode="keep"`, `"hybrid"`, or `"compact"`.

### Why `asyncio` everywhere?

The Apple FM SDK's `session.respond()` is `async`. Rather than fighting this with `asyncio.run()` wrappers, SiliconRefinery embraces it:

- `@local_extract` returns an `async` function
- `stream_extract` is an `AsyncGenerator`
- `Pipeline.execute()` is an `AsyncGenerator`
- The Polars extension uses a persistent background event loop thread
- The DSPy extension bridges async-to-sync via `concurrent.futures.ThreadPoolExecutor`

### Error handling philosophy

- **Transient errors** (`TimeoutError`, `ConnectionError`, `OSError`) are retried with exponential backoff
- **Non-transient errors** (`TypeError`, `ValueError`, etc.) fail immediately — no point retrying a schema mismatch
- **Pipeline errors** are configurable via `on_error="skip"` (default), `"raise"`, or `"yield_none"`
- **Bare `raise`** is used everywhere (not `raise e`) to preserve original tracebacks

### Project structure

```
silicon_refinery/
    __init__.py          # Public API exports
    decorators.py        # @local_extract decorator
    async_generators.py  # stream_extract async generator
    pipeline.py          # Source >> Extract >> Sink
    debugging.py         # @enhanced_debug decorator
    cache.py             # sqlite3 content-addressable extraction cache
    protocols.py         # typing.Protocol backend interfaces
    adapters.py          # async IO adapters for files/CSV/JSONL/stdin/iterables
    _context.py          # contextvars session scoping
    _threading.py        # free-threading safety helpers
    scanner.py           # mmap sliding-window scanner
    watcher.py           # hot-folder daemon
    _jit.py              # runtime diagnostics and metrics
    arrow_bridge.py      # Arrow IPC + Polars conversion bridge
    functional.py        # functional pipeline composition API
    auditor.py           # on-device code auditor
    polars_ext.py        # Polars .local_llm namespace
    dspy_ext.py          # DSPy AppleFMLM provider
    py.typed             # PEP 561 type-checking marker

tests/                   # 400+ tests, 100% mock-based (no hardware required)
use_cases/               # 9 self-contained examples
datasets/                # 5 synthetic sample datasets
examples/                # 14 scripts + marimo notebook + desktop app
```

---

## Future Work

### Phase 4 Delivered

The following features are now implemented and covered by tests:
- `sqlite3` extraction cache (`silicon_refinery.cache`)
- Pluggable backend protocols (`silicon_refinery.protocols`)
- IO adapters including trio-style channels (`silicon_refinery.adapters`)
- Context-scoped sessions via `contextvars` (`silicon_refinery._context`)
- Free-threading helpers (`silicon_refinery._threading`)
- `mmap` sliding-window scanner (`silicon_refinery.scanner`)
- Hot-folder watcher daemon (`silicon_refinery.watcher`)
- Runtime diagnostics (`silicon_refinery._jit`)
- Arrow IPC + Polars bridge (`silicon_refinery.arrow_bridge`)
- Functional pipeline API (`silicon_refinery.functional`)
- On-device code auditor (`silicon_refinery.auditor`)

### Next Priorities

Build native adapters for:
- `io.BytesIO` / `io.StringIO` streams
- `asyncio.StreamReader` for network data
- `aiofiles` for async file I/O
- `websockets` for real-time data feeds

### SiliconRefineryChat Demo Roadmap

- TODO: Add support for attachments in `examples/toga_local_chat_app` with a nonblocking ingest pipeline and streaming-safe UI integration.
- TODO: Build conversation-query mode so users can start a chat that queries the sqlite database containing all prior conversations.
- TODO: Implement durable memory behavior for SiliconRefineryChat so relevant context can persist and be reused safely across sessions.
- TODO: Harden context compaction and add targeted tests for guardrail behavior and error handling paths.
- TODO: Validate API behavior on diverse real-world datasets and production-like workloads, then harden and flesh out the Python library based on findings.
- TODO: Continue exploring the frontier of modern Python + Apple FM SDK + OSS ecosystem capabilities (free-threading, async runtimes, Arrow/Polars, DSPy, and adjacent tooling).

### Free-threading & subinterpreters (PEP 703/684)

Python 3.13t/3.14t introduces experimental GIL-free execution. SiliconRefinery's `stream_extract(concurrency=N)` is architected to exploit this — if the Apple FM SDK's C extensions release the GIL, true thread-level parallelism becomes possible on separate M-series cores.

Next, we plan to auto-switch to `asyncio.to_thread()` parallelism on free-threaded builds.

### JIT compilation (PEP 744)

The copy-and-patch JIT in Python 3.13+ (enabled via `PYTHON_JIT=1`) can reduce Python-level overhead in hot loops — relevant for `stream_extract` managing thousands of concurrent tasks. Current benchmarks show <5% improvement (LLM inference latency dominates), but we include JIT-friendly code paths for forward-compatibility.

### Beyond current cache layer

Add pluggable cache backends (SQLite, in-memory LRU, Redis) behind a common cache protocol.

---

## Contributing

We welcome contributions. To get started:

```bash
git clone https://github.com/adpena/silicon-refinery
cd silicon-refinery
uv run silicon-refinery setup   # Or: ./scripts/setup.sh

# Development workflow
uv run silicon-refinery check   # Full CI pipeline (lint + format + typecheck + tests)

# Or run individually:
uv run silicon-refinery lint        # uv run ruff check .
uv run silicon-refinery format      # uv run ruff format .
uv run silicon-refinery typecheck   # uv run ty check silicon_refinery/
uv run silicon-refinery test        # uv run pytest tests/ -v (400+ tests, no Apple Silicon required)
```

**Toolchain:**
- **[uv](https://docs.astral.sh/uv/)** — Package management and virtual environments
- **[ruff](https://docs.astral.sh/ruff/)** — Linting and formatting (replaces flake8, isort, black)
- **[ty](https://docs.astral.sh/ty/)** — Type checking (configured in `pyproject.toml` under `[tool.ty]`)
- **[pytest](https://docs.pytest.org/)** — Test runner with `pytest-asyncio` for async test support

Function docstrings are especially important in this project — they serve as system prompts for the LLM. Write them carefully.

If you have questions, ideas, or feedback:
**Email:** `adpena@gmail.com`

---

## Gratitude to Apple

We extend our sincere thanks to Apple for releasing the [`python-apple-fm-sdk`](https://github.com/apple/python-apple-fm-sdk) — a Python bridge to the Foundation Models framework introduced alongside Apple Intelligence.

Special recognition to:
- The **Apple Intelligence** and **Foundation Models** teams for designing a structured generation protocol (`@generable()`) that guarantees schema-valid outputs — the foundation that makes SiliconRefinery's type-safe extraction possible
- The **`python-apple-fm-sdk`** contributors for providing first-class `asyncio` support, making it natural to build high-throughput streaming pipelines
- The macOS engineering teams for continuing to optimize the Neural Engine inference path, enabling the throughput numbers reported in our benchmarks

These are very early days for local AI. This SDK makes the possibilities truly boundless for the open-source community.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built on the `python-apple-fm-sdk`. See [CHANGELOG.md](CHANGELOG.md) for version history.*
