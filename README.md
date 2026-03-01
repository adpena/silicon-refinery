<div align="center">
  <h1>FMTools</h1>
  <p><b>Local Python framework for structured extraction on Apple Silicon</b></p>
  <p>
    <a href="https://pypi.org/project/fmtools/"><img src="https://img.shields.io/pypi/v/fmtools.svg" alt="PyPI Version"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.13%2B-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/macOS-26.0+-lightgrey.svg" alt="macOS Version">
    <img src="https://img.shields.io/badge/tests-400%2B%20passed-brightgreen.svg" alt="Tests">
  </p>
</div>

**FMTools** is a Python framework built on the [Apple Foundation Models SDK (`python-apple-fm-sdk`)](https://github.com/apple/python-apple-fm-sdk). It provides local APIs and tools for converting unstructured text into schema-validated Python objects on Apple Silicon.

It is a developer layer for the SDK: app, CLI, and API surfaces that reduce the amount of wrapper and runtime plumbing developers need to build themselves.

Built so far (applications + APIs):
- `FMChat` (macOS app): a local-first chat app with streaming responses, SQLite-backed multi-chat history, export/resume flows, and no cloud dependency; also serves as a reference implementation for SDK-driven chat UX patterns.
- `fmtools chat` (CLI app): a scriptable local chat runtime for terminal-first workflows and fast iteration during development.
- `@local_extract` (core API): schema-guaranteed extraction from unstructured text into typed Python objects, without prompt-parsing hacks.
- `stream_extract` (core API): async streaming extraction for high-throughput pipelines where latency and concurrency both matter.
- `Source >> Extract >> Sink` (pipeline API): composable, readable dataflow pattern for production ETL and event-processing jobs.
- `@enhanced_debug` (diagnostics API): model-assisted debugging summaries that stay local, with controllable output routing for prompt and summary traces.
- `Polars .local_llm.extract()` (DataFrame API): direct structured extraction on tabular data without leaving the Polars workflow.
- `AppleFMLM` for DSPy and FastAPI integration examples: practical adapters for integrating on-device inference into agent and service architectures.

Quick capability snippets:

```bash
# Actual output comments below are from running these commands locally on macOS 26.3 (M1, 8 GB RAM).

# 1) Discover the full runnable use-case catalog
fmtools run --list

# 2) Discover the full runnable examples catalog
fmtools example --list

# 3) Launch local desktop chat (Apple Foundation Models + SQLite memory)
fmtools chat

# 4) Standalone app entrypoint after Homebrew cask install
fmchat

# Actual output:
# Available use cases:
#   01 Pipeline Operators
#   02 Decorators
#   03 Async Generators
#   04 Ecosystem Polars
#   05 Dspy Optimization
#   06 Fastapi Integration
#   07 Stress Test Throughput
#   08 Context Limit Test
#   09 Enhanced Debugging
#
# Available examples:
#   arrow_bridge
#   code_auditor
#   context_scope
#   custom_backend
#   extraction_cache
#   free_threading
#   functional_pipeline
#   hot_folder_watcher
#   jit_diagnostics
#   mmap_scanner
#   simple_inference
#   streaming_example
#   transcript_processing
#   trio_adapter
```

```python
# Actual output comments below are from running this snippet locally on macOS 26.3 (M1, 8 GB RAM).
import polars as pl
import apple_fm_sdk as fm
from fmtools.polars_ext import LocalLLMExpr  # registers .local_llm

@fm.generable()
class TicketSchema:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

df = pl.DataFrame(
    {
        "text_column": [
            "I cannot log in to my account after enabling two-factor authentication."
        ]
    }
)

enriched_df = df.with_columns(
    extracted_json=pl.col("text_column").local_llm.extract(
        schema=TicketSchema,
        instructions="Extract only fields required by the schema.",
    )
)
print(enriched_df.select(["text_column", "extracted_json"]).to_dicts())
# Actual output:
# [{'text_column': 'I cannot log in to my account after enabling two-factor authentication.',
#   'extracted_json': '{"category": "Account", "urgency": "HIGH"}'}]
```

```python
# Actual output comments below are from running this snippet locally on macOS 26.3 (M1, 8 GB RAM).
from fmtools import enhanced_debug

@enhanced_debug(summary_to="stderr", prompt_to="stdout", silenced=False)
def process_data(payload):
    return payload["value"] + 10

print(process_data({"value": 10}))
# Actual output:
# 20
```

```python
# Actual output comments below are from running this snippet locally on macOS 26.3 (M1, 8 GB RAM).
import asyncio
import apple_fm_sdk as fm
from fmtools import local_extract

@fm.generable()
class SupportTicket:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    summary: str = fm.guide(description="One-sentence summary of the issue")

@local_extract(schema=SupportTicket, debug_timing=True)
async def classify(email: str) -> SupportTicket:
    """Classify a customer support email by category, urgency, and summary."""

async def main():
    ticket = await classify("I was charged twice and I need a refund immediately.")
    print(f"category={ticket.category}")
    print(f"urgency={ticket.urgency}")
    print(f"summary={ticket.summary}")

asyncio.run(main())
# Actual output:
# category=Billing
# urgency=HIGH
# summary=Customer was charged twice and needs immediate refund
```

No API keys or cloud calls are required for local inference.

---

## Table of Contents

- [Why FMTools?](#why-fmtools)
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
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Why FMTools?

Many teams and individual developers work with unstructured logs, CSV exports, support messages, and notes. Historically, extracting structure from this data has often required cloud LLM calls, with cost and privacy tradeoffs.

FMTools focuses on local-first extraction:

| Problem | FMTools's Answer |
|---|---|
| **Data leaves your network** | All inference runs on the local Neural Engine. Zero egress. |
| **API costs scale with volume** | Zero token costs. Process millions of records for free. |
| **Schema validation is fragile** | Apple's `@generable()` protocol guarantees schema-valid outputs at the model level. |
| **Async complexity** | Idiomatic `asyncio` patterns — decorators, generators, pipelines — all async-native. |
| **Context window explosion** | Built-in session management: clear, keep, hybrid, and compact history modes. |
| **No DataFrame integration** | First-class Polars extension via `.local_llm.extract()`. |
| **Small OSS LLMs can still be heavy, memory-hungry, architecture-mismatched, and hard to trust/observe** | Uses Apple’s highly optimized Foundation Models stack with tight local integration on Apple Silicon for stronger performance, provenance, and security within the on-device ecosystem. |

**Measured throughput:** 250-350+ characters/sec (~60-90 tokens/sec) on Apple M1, purely on-device. See [Benchmarks](#benchmarks--empirical-results).

---

## Installation

### Requirements

- macOS 26.0+ (Tahoe or later)
- Apple Silicon (M1, M2, M3, M4 series)
- CPython 3.13+

### Install paths (PyPI + Homebrew)

For straightforward setup, pick one:

```bash
# 1) PyPI + uv (library + CLI in a project virtual environment)
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install -U fmtools
uv pip install -U "apple-fm-sdk @ git+https://github.com/apple/python-apple-fm-sdk.git"

# Verify
fmtools doctor
```

```bash
# 2) Homebrew (CLI + standalone desktop app, minimal flow)
brew tap adpena/fmtools https://github.com/adpena/homebrew-fmtools
brew install fmtools
brew install fmchat

# Verify
fmtools doctor
fmchat
```

### One-command setup (recommended)

```bash
git clone https://github.com/adpena/fmtools && cd fmtools
./scripts/setup.sh
```

This installs [uv](https://docs.astral.sh/uv/) if missing, creates a virtual environment, syncs dependency groups (including `apple-fm-sdk` via `tool.uv.sources`), and installs the `fmtools` CLI globally in editable mode for local development. Run `./scripts/doctor.sh` afterwards to verify your system meets all requirements.

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

# 2. Clone FMTools
git clone https://github.com/adpena/fmtools && cd fmtools

# 3. Create venv and install all dependencies (including Apple FM SDK from git)
uv sync --all-groups
source .venv/bin/activate
```

> **Note:** The Apple FM SDK is not yet on PyPI. `uv sync` automatically clones and builds it from GitHub via the `[tool.uv.sources]` configuration in `pyproject.toml`.

### Command availability & PATH

Use whichever mode best fits your workflow:

| Mode | Command example | Best for |
|---|---|---|
| No activation required | `uv run fmtools chat` | Most reliable local dev from repo root |
| Activated venv | `fmtools chat` (after `source .venv/bin/activate`) | Traditional Python workflow |
| Global CLI install | `fmtools chat` (after `uv tool install --editable --from . fmtools`) | Seamless command usage across shells |

If `fmtools` is not found after setup, open a new terminal (or run `source ~/.zshrc`) so your shell reloads PATH changes.

For chat workflows, `fmtools chat` prefers free-threaded CPython (`3.14t`, then `3.13t`) by default and falls back to standard-GIL only if a no-GIL runtime is unavailable.

Standalone desktop app repo: [adpena/fmchat](https://github.com/adpena/fmchat) (`FMChat`).

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

- **PyPI:** `fmtools` is published for Python imports + CLI usage.
- **Apple FM SDK:** still GitHub-sourced, so install via `uv pip install "apple-fm-sdk @ git+https://github.com/apple/python-apple-fm-sdk.git"` (or run `uv sync --all-groups`).
- **Homebrew tap:** `adpena/homebrew-fmtools` provides both the CLI formula and chat app cask with a single tap.

### Homebrew maintainer release flow

When shipping a new release and updating Homebrew support:

```bash
# 1) Ensure main branch has the new release commit/tag pushed
git push origin main --tags

# 2) Sync Formula + Cask into the tap repo and push
#    (tap repo: adpena/homebrew-fmtools)
#    - Formula/fmtools.rb
#    - Casks/fmchat.rb

# 3) Validate the installed command
brew install fmtools
brew install fmchat
fmtools --help
fmtools doctor
fmchat
```

The CLI formula (`Formula/fmtools.rb`) and chat cask (`Casks/fmchat.rb`) are version-locked to the same release number and use thousandth-place increments (`0.0.216` -> `0.0.217`).
This is enforced in CI/release via `python3 scripts/check_version_policy.py` (and `--enforce-thousandth-bump` during publish).

### Standalone Chat Repo release flow

The standalone macOS app repository (`fmchat`) is synced and packaged from this repo via a local maintainer run.

```bash
# Local maintainer sync (create-or-update repo, sync source/docs, build artifact, upload release asset)
./scripts/publish_chat_repo.sh --repo adpena/fmchat
```

A sync-only fallback workflow exists at [`.github/workflows/publish-chat-repo.yml`](.github/workflows/publish-chat-repo.yml), but signing/notarization for distributable artifacts is maintained locally.

For Gatekeeper-safe public installs (no "Apple could not verify..." warning), use Developer ID signing + notarization:

```bash
export CHAT_SIGN_MODE=developer-id
export CHAT_SIGN_IDENTITY="${CHAT_SIGN_IDENTITY:?Set your Developer ID identity string first}"
export CHAT_NOTARIZE_MODE=required
export APPLE_NOTARY_PROFILE="${APPLE_NOTARY_PROFILE:?Set your stored notary profile name first}"

./scripts/publish_chat_repo.sh --repo adpena/fmchat
```

This flow signs with hardened runtime via Briefcase identity packaging, notarizes via `xcrun notarytool`, staples tickets with `xcrun stapler`, and validates with `spctl`/`codesign`.
By default, `scripts/publish_chat_repo.sh` also blocks uploading untrusted artifacts to GitHub releases (ad-hoc, unstapled, or non-notarized). For local-only debugging you can explicitly override with `--allow-untrusted-release`.

### Optional dependencies

```bash
# For FastAPI/Uvicorn integration (use_cases/06_fastapi_integration)
uv pip install fmtools[api]

# For trio-native adapters (TrioAdapter)
uv pip install fmtools[adapters]

# For Arrow IPC examples/integration
uv pip install fmtools[arrow]

# For the comprehensive marimo examples notebook
uv pip install marimo
```

> **Important:** The `api`, `adapters`, and `arrow` extras do not install `apple-fm-sdk`. Install base/dev dependencies first (for example, `uv sync --all-groups`) so `fmtools` can import and run.

### CLI

With any command mode above (`uv run`, activated `.venv`, or global tool install), the `fmtools` CLI supports:

If you have not activated `.venv` (or do not have global tool install), use `uv run` as a prefix, for example: `uv run fmtools chat`.

```bash
fmtools setup      # First-time dev environment setup
fmtools install-homebrew  # Install/update via local Homebrew tap
fmtools doctor     # Check all system prerequisites
fmtools lint       # Run ruff linter
fmtools format     # Run ruff formatter
fmtools typecheck  # Run ty type checker
fmtools test       # Run the full test suite (460+ tests)
fmtools check      # Run lint + format + typecheck + tests (CI pipeline)
fmtools example --list   # List standalone examples
fmtools example simple_inference  # Run one example script
fmtools smoke      # Run full examples smoke suite (+ notebook startup check)
fmtools notebook   # Launch the comprehensive marimo notebook
fmtools chat       # Run the Toga + Briefcase local chat demo
fmtools chat --standard-gil  # Force standard-gil runtime for max GUI stability
```

In chat compose, use `Cmd+Enter` to send; plain `Enter` inserts a newline.

### Verify your installation

```bash
# Quick system check
fmtools doctor
```

```python
import fmtools
import apple_fm_sdk as fm

# Actual output comments below are from running this snippet locally on macOS 26.3 (M1, 8 GB RAM).
model = fm.SystemLanguageModel()
available, reason = model.is_available()
print(f"Neural Engine available: {available}")
print(f"Reason: {reason}")
# Actual output:
# Neural Engine available: True
# Reason: None
```

---

## Quick Start

Three steps to go from unstructured text to a validated Python object:

**1. Define a schema** using the Apple FM SDK's `@generable()` decorator:

```python
# This snippet is runnable as-is.
import apple_fm_sdk as fm

@fm.generable()
class SupportTicket:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str  = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    summary: str  = fm.guide(description="One-sentence summary of the issue")

print(SupportTicket.__name__)
# Actual output:
# SupportTicket
```

**2. Decorate a function** with `@local_extract`. The function's **docstring becomes the system prompt**:

```python
# This snippet is runnable as-is.
import apple_fm_sdk as fm
from fmtools import local_extract

@fm.generable()
class SupportTicket:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    summary: str = fm.guide(description="One-sentence summary of the issue")

@local_extract(schema=SupportTicket, debug_timing=True)
async def classify_ticket(email_text: str) -> SupportTicket:
    """Classify a customer support email by category, urgency, and summary."""

print(classify_ticket.__name__)
# Actual output:
# classify_ticket
```

**3. Call it:**

```python
import asyncio
import apple_fm_sdk as fm
from fmtools import local_extract

@fm.generable()
class SupportTicket:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])
    summary: str = fm.guide(description="One-sentence summary of the issue")

@local_extract(schema=SupportTicket, debug_timing=False)
async def classify_ticket(email_text: str) -> SupportTicket:
    """Classify a customer support email by category, urgency, and summary."""

# Actual output comments below are from running this snippet locally on macOS 26.3 (M1, 8 GB RAM).
async def main():
    ticket = await classify_ticket(
        "I was charged twice this month and I need a refund immediately!"
    )
    print(ticket.category)
    print(ticket.urgency)
    print(ticket.summary)

asyncio.run(main())
# Actual output:
# Billing
# HIGH
# Customer was charged twice this month and is requesting an immediate refund.
```

The decorated function intercepts your arguments, sends them to the on-device model, enforces the schema via `@generable()`, and returns a validated `SupportTicket` object.

---

## Public API Breakdown

FMTools follows a layered API design:
- a small, stable root import surface for day-to-day usage
- module-level APIs for advanced workflows
- a CLI for reproducible local development workflows

### Stable root imports (`fmtools`)

```python
from fmtools import (
    AppleFMSetupError,
    Extract,
    Sink,
    Source,
    enhanced_debug,
    local_extract,
    stream_extract,
)

print(
    "imports ok:",
    all([AppleFMSetupError, Extract, Sink, Source, enhanced_debug, local_extract, stream_extract]),
)
# Actual output:
# imports ok: True
```

### API Definition (Standard Contract)

The public API is defined as:

- **Library namespace contract:** Root imports in `fmtools.__init__` are the stable, documented entry points for application code.
- **Type contract:** Core extractors and decorators use explicit type signatures (`schema: type[T]`, async generator returns, typed exceptions).
- **Behavior contract:** Extraction decorators and stream APIs preserve structured-output guarantees and raise deterministic setup/runtime exceptions (`AppleFMSetupError` and standard Python exception types).
- **CLI contract:** `fmtools --help` command set is the canonical executable API surface for development workflows.

| Symbol | Kind | Signature | Primary use |
|---|---|---|---|
| `local_extract` | Decorator factory | `(schema, retries=3, debug_timing=False)` | Turn an async function into structured on-device extraction |
| `stream_extract` | Async generator | `(source_iterable, schema, ..., concurrency=None)` | High-throughput extraction over streams |
| `Source`, `Extract`, `Sink` | Pipeline nodes | `Source(iterable) >> Extract(schema) >> Sink(callback)` | Declarative ETL pipelines |
| `enhanced_debug` | Decorator factory | `(summary_to=\"stderr\", prompt_to=\"stdout\", silenced=False, summary_log_level=\"error\", prompt_log_level=\"info\")` | AI-assisted crash analysis |
| `AppleFMSetupError` | Exception | setup/runtime exception type | Consistent SDK/model troubleshooting diagnostics |

### Module-level API index

| Module | Primary public API | Role |
|---|---|---|
| `fmtools.cache` | `ExtractionCache`, `cache_extract`, `cached_stream_extract`, `cached_local_extract` | sqlite-backed extraction caching |
| `fmtools.protocols` | `ModelProtocol`, `SessionProtocol`, `SessionFactory`, `AppleFMBackend`, `set_backend`, `get_backend`, `create_model`, `create_session` | Pluggable backend contracts wired into core extractors |
| `fmtools.adapters` | `FileAdapter`, `CSVAdapter`, `JSONLAdapter`, `StdinAdapter`, `IterableAdapter`, `TrioAdapter`, `TextChunkAdapter` | Async input adapters (including trio-style channels) |
| `fmtools._context` | `session_scope`, `get_session`, `get_model`, `get_instructions`, `copy_context` | Task-local session scoping |
| `fmtools._threading` | `is_free_threaded`, `get_gil_status`, `CriticalSection`, `AtomicCounter`, `ThreadSafeDict` | Free-threading safety primitives |
| `fmtools.scanner` | `MMapScanner`, `line_split_scanner` | Large-file scanning with overlap-safe windows |
| `fmtools.watcher` | `HotFolder`, `FileEvent`, `process_folder` | Hot-folder ingestion daemon |
| `fmtools._jit` | `diagnostics`, `diagnose`, `DiagnosticCollector`, `ExtractionMetrics` | Runtime diagnostics and metrics |
| `fmtools.arrow_bridge` | `to_arrow_ipc`, `from_arrow_ipc`, `to_arrow_ipc_buffer`, `from_arrow_ipc_buffer`, `ArrowStreamWriter`, `to_polars`, `from_polars` | Arrow/Polars interoperability |
| `fmtools.functional` | `pipe`, `source`, `extract`, `map_fn`, `filter_fn`, `flat_map_fn`, `batch`, `take`, `skip`, `tap`, `collect`, `reduce_fn` | Functional pipeline composition |
| `fmtools.auditor` | `audit_file`, `audit_directory`, `audit_diff`, `format_audit_report` | On-device code auditing with bounded-concurrency directory scans |
| `fmtools.exceptions` | `AppleFMSetupError`, `troubleshooting_message`, `require_apple_fm`, `ensure_model_available` | Shared setup validation and graceful diagnostics |

### CLI API (`fmtools`)

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

FMTools exposes seven foundational capabilities. Each one is documented below with its full function signature, parameter reference, behavior details, and a working example.

### 1. `@local_extract` — Structured Extraction Decorator

**Module:** `fmtools.decorators`
**Import:** `from fmtools import local_extract`

Transforms any Python function into an on-device LLM extraction engine. The function's docstring serves as the system prompt.

#### Signature (runnable introspection)

```python
# This snippet is runnable as-is and prints the current local_extract signature.
from inspect import signature
from fmtools import local_extract

print(signature(local_extract))
# Actual output:
# (schema: type[~T], retries: int = 3, debug_timing: bool = False) -> collections.abc.Callable[[~F], ~F]
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `schema` | `type[T]` | *(required)* | A class decorated with `@fm.generable()`. The model's output is constrained to this shape. |
| `retries` | `int` | `3` | Number of retry attempts for transient errors (`TimeoutError`, `ConnectionError`, `OSError`). Non-transient errors (e.g., `TypeError`, `ValueError`) fail immediately. |
| `debug_timing` | `bool` | `False` | When `True`, logs extraction time and input length to the `fmtools` logger. |

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
from fmtools import local_extract

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
# Actual output:
# HIGH
# ['fever', 'chest pain']
# 3
```

**Run it:** `python use_cases/02_decorators/example.py`

---

### 2. `stream_extract` — Concurrent Async Streaming

**Module:** `fmtools.async_generators`
**Import:** `from fmtools import stream_extract`

An asynchronous generator that processes massive data streams through the local model, yielding structured objects one at a time. Supports line-level chunking, four session history modes, and concurrent `imap_unordered`-style parallel extraction.

#### Signature (runnable introspection)

```python
# This snippet is runnable as-is and prints the current stream_extract signature.
from inspect import signature
from fmtools import stream_extract

print(signature(stream_extract))
# Actual output:
# (source_iterable: collections.abc.Iterable | collections.abc.AsyncIterable, schema: type[~T], instructions: str = 'Extract data.', lines_per_chunk: int = 1, history_mode: Literal['clear', 'keep', 'hybrid', 'compact'] = 'clear', concurrency: int | None = None, debug_timing: bool = False) -> collections.abc.AsyncGenerator[~T, None]
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
| `keep` | Retains session history across chunks. May throw `apple_fm_sdk.ExceededContextWindowSizeError`. | Short sequences where context matters |
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
from fmtools import stream_extract

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
# Actual output:
# Positive
# Battery life
```

**Run it:** `python use_cases/03_async_generators/example.py`

---

### 3. `Source >> Extract >> Sink` — Composable Pipelines

**Module:** `fmtools.pipeline`
**Import:** `from fmtools import Source, Extract, Sink`

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
from fmtools import Source, Extract, Sink

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
# Actual output:
# ERROR|auth|auth login failed for user alice
```

**Run it:** `python use_cases/01_pipeline_operators/example.py`

---

### 4. `@enhanced_debug` — AI Crash Analysis

**Module:** `fmtools.debugging`
**Import:** `from fmtools import enhanced_debug`

A decorator that catches exceptions, prints the standard Python traceback, and then invokes the Neural Engine to perform an automated root-cause analysis. Works with both sync and async functions.

#### Signature (runnable introspection)

```python
# This snippet is runnable as-is and prints the current enhanced_debug signature.
from inspect import signature
from fmtools import enhanced_debug

print(signature(enhanced_debug))
# Actual output:
# (summary_to: str | None = 'stderr', prompt_to: str | None = 'stdout', silenced: bool = False, summary_log_level: str | int = 'error', prompt_log_level: str | int = 'info')
```

#### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `summary_to` | `str \| None` | `"stderr"` | Where to print the AI analysis: `"stdout"`, `"stderr"`, or `"log"`. Set `None` to silence analysis output. |
| `prompt_to` | `str \| None` | `"stdout"` | Where the generated debug prompt goes: `"stdout"` (default), `"stderr"`, `"log"`, a relative/absolute file path (written as `.txt`), or `None` to silence prompt output. |
| `silenced` | `bool` | `False` | Master mute switch. If `True`, all enhanced debug output is suppressed and analysis is skipped. |
| `summary_log_level` | `str \| int` | `"error"` | Logging level used when `summary_to="log"` (for example: `"warning"`, `"error"`, `"critical"`). |
| `prompt_log_level` | `str \| int` | `"info"` | Logging level used when `prompt_to="log"` (for example: `"debug"`, `"info"`, `"warning"`). |

#### What happens when the function crashes

1. The standard Python traceback is printed to stderr
2. A structured debug query is sent to the on-device Foundation Model with strict evidence-based instructions and traceback context
3. If the SDK raises `apple_fm_sdk.ExceededContextWindowSizeError`, FMTools automatically retries with a smaller tail-prioritized traceback payload
4. A structured forensic `DebuggingAnalysis` is generated containing:
   - `error_summary` — Brief description of what went wrong
   - `possible_causes` — List of likely root causes
   - `certainty_level` — `"LOW"`, `"MEDIUM"`, or `"HIGH"`
   - `likely_fix_locations` — Concrete `path:line` fix targets with frame evidence
   - `suggested_fix` — Actionable steps to resolve the issue
5. If `prompt_to` is enabled, a second handoff-generation query runs to produce a concrete agent plan (candidate edits, instrumentation, verification commands, risks, and an agent-ready prompt)
6. Prompt payload output is routed via `prompt_to` and includes full traceback, crash envelope, stage1 query payload, and stage2 handoff plan
7. The original exception is re-raised (the decorator never swallows exceptions)

#### Sample AI analysis output

```
==================================================
FMTools AI Debug Analysis (Certainty: HIGH, Severity: MEDIUM)
==================================================
Exception: TypeError: can only concatenate str (not 'int') to str
Function: process_data
Context retries: 0

Summary: Integer/string type mismatch during arithmetic in process_data
Blast Radius: Isolated to request path that forwards string payloads into process_data

Likely Fix Locations:
  1. use_cases/09_enhanced_debugging/example.py:18-19 | cast payload value before +10 | evidence=F1

Suggested Fix: Normalize data_payload['value'] to int before arithmetic and add type guard logging.
==================================================
```

A [sample generated prompt file](use_cases/09_enhanced_debugging/sample_crash_report.txt) is included in the repository — ready to paste into Claude or any coding assistant.

#### Full example

```python
from pathlib import Path
from fmtools import enhanced_debug

PROMPT_PATH = Path("crash_report_for_llm.txt")
if PROMPT_PATH.exists():
    PROMPT_PATH.unlink()

@enhanced_debug(summary_to="stdout", prompt_to=str(PROMPT_PATH), silenced=False)
def process_data(data_payload):
    """A buggy function that will inevitably crash."""
    parsed_value = data_payload["value"] + 10  # TypeError!
    return parsed_value

try:
    process_data({"value": "100"})
except TypeError:
    pass

print("prompt_exists=", PROMPT_PATH.exists())
print("prompt_first_line=", PROMPT_PATH.read_text(encoding="utf-8").splitlines()[0])

# Actual output:
# [stdout] FMTools AI Debug Analysis (Certainty: HIGH)
# [stdout] Exception: TypeError: can only concatenate str (not 'int') to str
# [stdout] Function: process_data
# [stdout] Context retries: 0
# [stdout] Summary: Integer/string type mismatch during arithmetic in process_data
# [stdout] Blast Radius: Isolated to request path that forwards string payloads into process_data
# [stdout] Likely Fix Locations:
# [stdout]   1. use_cases/09_enhanced_debugging/example.py:18-19 | cast payload value before +10 | evidence=F1
# [stdout] Generated AI Agent Prompt written to: crash_report_for_llm.txt
# [stdout] prompt_exists= True
# [stdout] prompt_first_line= I encountered a crash in my Python application.
# [stderr] --- Exception caught in 'process_data' ---
# [stderr] TypeError: can only concatenate str (not 'int') to str
# [stderr] FMTools is analyzing the crash locally via Neural Engine...
```

**Run it:** `python use_cases/09_enhanced_debugging/example.py`

---

### 5. Polars `.local_llm` Extension

**Module:** `fmtools.polars_ext`
**Import:** `from fmtools.polars_ext import LocalLLMExpr`

Registers the `.local_llm` namespace directly onto Polars expressions, allowing you to run on-device LLM inference inside `df.select()` or `df.with_columns()` calls. Each row gets its own `LanguageModelSession` (preventing context window explosion), and rows within a batch are processed concurrently via `asyncio.Semaphore(4)`.

#### Usage

```python
import polars as pl
import apple_fm_sdk as fm
from fmtools.polars_ext import LocalLLMExpr  # registers the namespace

@fm.generable()
class TicketSchema:
    category: str = fm.guide(anyOf=["Billing", "Technical", "Account", "Other"])
    urgency: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

df = pl.DataFrame(
    {"text_column": ["I cannot log in to my account after enabling two-factor authentication."]}
)

enriched_df = df.with_columns(
    extracted_json=pl.col("text_column").local_llm.extract(
        schema=TicketSchema,
        instructions="Extract only fields required by the schema.",
    )
)
print(enriched_df.select(["text_column", "extracted_json"]).to_dicts())
# Actual output:
# [{'text_column': 'I cannot log in to my account after enabling two-factor authentication.',
#   'extracted_json': '{"category": "Account", "urgency": "HIGH"}'}]
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
from fmtools.polars_ext import LocalLLMExpr

@fm.generable()
class Ticket:
    department: str = fm.guide(anyOf=["IT", "HR", "Sales", "Billing", "Other"])
    urgency: int = fm.guide(description="Scale 1 to 5, where 5 is critical")

df = pl.read_csv("datasets/support_tickets.csv")

enriched_df = df.with_columns(
    extracted_json=pl.col("email_body").local_llm.extract(schema=Ticket)
)
print(enriched_df.select(["ticket_id", "email_subject", "extracted_json"]))
# Actual output:
# [{'ticket_id': 1, 'email_subject': 'Cannot login', 'extracted_json': '{"department": "IT", "urgency": 5}'}]
```

**Run it:** `python use_cases/04_ecosystem_polars/example.py`

---

### 6. DSPy `AppleFMLM` Provider

**Module:** `fmtools.dspy_ext`
**Import:** `from fmtools.dspy_ext import AppleFMLM`

A custom `dspy.LM` subclass that routes all inference through the local Apple Foundation Model. This lets you use DSPy's full suite of prompt compilers, Chain-of-Thought reasoning, and agentic workflows — all running on free Apple hardware with zero cloud dependency.

#### Usage

```python
import dspy
from fmtools.dspy_ext import AppleFMLM

dspy.settings.configure(lm=AppleFMLM())

# Now any DSPy module uses the local Neural Engine
classifier = dspy.ChainOfThought("customer_email -> summary, priority")
result = classifier(customer_email="I am locked out of my account!")
print(result.summary, result.priority)
# Actual output:
# Customer is locked out of their account and needs assistance to reset their pass High
```

#### Implementation details

- Supports both `prompt` (string) and `messages` (list of dicts) input formats for DSPy v2.5+ compatibility
- History is stored in a bounded `collections.deque(maxlen=1000)` to prevent unbounded memory growth
- Handles async-to-sync bridging: detects if an event loop is already running and uses `concurrent.futures.ThreadPoolExecutor` to avoid `asyncio.run()` conflicts

#### Full example — Chain-of-Thought support ticket analysis

```python
import csv
import dspy
from fmtools.dspy_ext import AppleFMLM

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
# Actual output:
# Ticket 1: The user is experiencing an issue with account access, specifically being locked out and unable to reset their password. [High]
```

**Run it:** `python use_cases/05_dspy_optimization/example.py`

---

### 7. FastAPI Integration

**Module:** Uses `fmtools.decorators` with FastAPI
**Install:** `uv pip install fmtools[api]`

Turn FMTools into a local REST API microservice. Because `@local_extract` returns an `async` function, it integrates natively with FastAPI's async request handling.

#### Full example — Document extraction API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import apple_fm_sdk as fm
from fmtools import local_extract

app = FastAPI(title="FMTools Extraction API")

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
# Actual output:
# Apple announced new Mac hardware and developers are excited about local AI tooling.
# 8
# ['Apple', 'Mac hardware', 'AI tooling']
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
| [`examples/toga_local_chat_app/`](examples/toga_local_chat_app/) | `FMChat`: Toga + Briefcase local chat app with vertical chat tabs, steering interjection reruns, sqlite memory, and `/help` `/new` `/clear` `/export` |

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

Two dedicated use cases profile FMTools's performance characteristics.

### Test environment

All benchmark measurements below were run on a **MacBook Pro with an M1 chip and 8 GB of RAM**.

| Component | Specification |
|---|---|
| **Hardware** | MacBook Pro (M1 chip, 8 GB RAM; MacBookPro17,1) — 8 cores (4P + 4E) |
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

#### Local testing results (empirical)

Payload escalation test — progressively larger input texts:

| Input size | Approx. tokens | Result | Time |
|---|---|---|---|
| 5,000 chars | ~1,250 | Success | 1.2s |
| 25,000 chars | ~6,250 | Success | 12.5s |
| 50,000 chars | ~12,500 | Success | ~25s |
| ~32,000+ chars | ~8,000+ | `apple_fm_sdk.ExceededContextWindowSizeError` | — |

#### Apple's official published limit

Apple's official published guidance indicates the on-device Foundation Models context window is currently **4,096 tokens per language model session** (input + output combined), as documented by Apple engineers in the Developer Forums and linked to TN3193.

Sources:
- [Apple Developer Forums — FoundationModel, context length, and testing (thread 806542)](https://developer.apple.com/forums/thread/806542)
- [Apple Technical Note TN3193 — Managing the on-device foundation model’s context window](https://developer.apple.com/documentation/technotes/tn3193-adding-intelligence-to-your-app-with-foundation-models)

#### Comparison and interpretation

Our local character-based escalation test is still useful for practical boundary testing, but character-to-token conversions are approximate and can diverge from true tokenizer accounting. For production guardrails, we treat Apple's published **4,096-token** limit as authoritative. This is why `stream_extract` defaults to `history_mode="clear"` — recreating the session per chunk prevents context accumulation on long-running streams.

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

- No risk of `apple_fm_sdk.ExceededContextWindowSizeError` on large streams
- Each extraction is independent and reproducible
- Concurrent processing is possible (each task gets its own session)

When context matters, use `history_mode="keep"`, `"hybrid"`, or `"compact"`.

### Why `asyncio` everywhere?

The Apple FM SDK's `session.respond()` is `async`. Rather than fighting this with `asyncio.run()` wrappers, FMTools embraces it:

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
fmtools/
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
- `sqlite3` extraction cache (`fmtools.cache`)
- Pluggable backend protocols (`fmtools.protocols`)
- IO adapters including trio-style channels (`fmtools.adapters`)
- Context-scoped sessions via `contextvars` (`fmtools._context`)
- Free-threading helpers (`fmtools._threading`)
- `mmap` sliding-window scanner (`fmtools.scanner`)
- Hot-folder watcher daemon (`fmtools.watcher`)
- Runtime diagnostics (`fmtools._jit`)
- Arrow IPC + Polars bridge (`fmtools.arrow_bridge`)
- Functional pipeline API (`fmtools.functional`)
- On-device code auditor (`fmtools.auditor`)

### Next Priorities (Not Yet Implemented)

The following adapter targets are planned and are not part of the completed Phase 4 set above:
- `io.BytesIO` / `io.StringIO` streams
- `asyncio.StreamReader` for network data
- `aiofiles` for async file I/O
- `websockets` for real-time data feeds

### FMChat App Roadmap (`examples/toga_local_chat_app`)

- TODO: Add support for attachments in `examples/toga_local_chat_app` with a nonblocking ingest pipeline and streaming-safe UI integration.
- TODO: Build conversation-query mode so users can start a chat that queries the sqlite database containing all prior conversations.
- TODO: Implement durable memory behavior for FMChat so relevant context can persist and be reused safely across sessions.
- TODO: Harden context compaction and add targeted tests for guardrail behavior and error handling paths in the desktop chat runtime.

### FMTools Python Library Roadmap (`fmtools/`)

- TODO: Validate API behavior on diverse real-world datasets and production-like workloads, then harden and flesh out the Python library based on findings.
- TODO: Experiment with scaffolding a local "mixture of experts" inference pipeline + API for user-query routing/power, with both synchronous and asynchronous coverage.
- TODO: Continue expanding modern Python + Apple FM SDK integrations (free-threading, async runtimes, Arrow/Polars, DSPy, and adjacent tooling).

### Free-threading & subinterpreters (PEP 703/684)

Python 3.13t/3.14t introduces experimental GIL-free execution. FMTools's `stream_extract(concurrency=N)` is architected to exploit this — if the Apple FM SDK's C extensions release the GIL, true thread-level parallelism becomes possible on separate M-series cores.

Next, we plan to auto-switch to `asyncio.to_thread()` parallelism on free-threaded builds.

### JIT compilation (PEP 744)

The copy-and-patch JIT in Python 3.13+ (enabled via `PYTHON_JIT=1`) can reduce Python-level overhead in hot loops — relevant for `stream_extract` managing thousands of concurrent tasks. Current benchmarks show <5% improvement (LLM inference latency dominates), but we include JIT-friendly code paths for forward-compatibility.

### Beyond current cache layer

Add pluggable cache backends (SQLite, in-memory LRU, Redis) behind a common cache protocol.

---

## Contributing

We welcome contributions. To get started:

```bash
git clone https://github.com/adpena/fmtools
cd fmtools
uv run fmtools setup   # Or: ./scripts/setup.sh

# Development workflow
uv run fmtools check   # Full CI pipeline (lint + format + typecheck + tests)

# Or run individually:
uv run fmtools lint        # uv run ruff check .
uv run fmtools format      # uv run ruff format .
uv run fmtools typecheck   # uv run ty check fmtools/
uv run fmtools test        # uv run pytest tests/ -v (400+ tests, no Apple Silicon required)
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

## Acknowledgements

FMTools is built on the [`python-apple-fm-sdk`](https://github.com/apple/python-apple-fm-sdk), which provides the Python bridge to Apple's Foundation Models framework.

Acknowledgements:
- The **Apple Intelligence** and **Foundation Models** teams for designing a structured generation protocol (`@generable()`) that guarantees schema-valid outputs — the foundation that makes FMTools's type-safe extraction possible
- The **`python-apple-fm-sdk`** contributors for providing first-class `asyncio` support, making it natural to build high-throughput streaming pipelines
- The macOS engineering teams for continuing to optimize the Neural Engine inference path, enabling the throughput numbers reported in our benchmarks

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

*Built on the `python-apple-fm-sdk`. See [CHANGELOG.md](CHANGELOG.md) for version history.*
