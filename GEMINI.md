# GEMINI.md - SiliconRefinery

## Project Overview

**SiliconRefinery** is an enterprise-grade Python ETL framework designed for **zero-trust, high-throughput data processing** exclusively on Apple Silicon. It leverages the [Apple Foundation Models SDK (`python-apple-fm-sdk`)](https://github.com/apple/python-apple-fm-sdk) to run on-device LLM inference, ensuring that sensitive data never leaves the local machine.

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

- **`silicon_refinery.decorators`**:
  - `@local_extract`: Transforms standard functions into structured LLM extraction engines. Function docstrings act as system instructions.
  - `@enhanced_debug`: Catches exceptions and uses the local AI to provide a root-cause analysis and suggest fixes.
- **`silicon_refinery.async_generators`**:
  - `stream_extract`: A high-concurrency asynchronous generator for processing massive text streams. Includes `history_mode` (clear, keep, hybrid, compact) for session memory management.
- **`silicon_refinery.pipeline`**:
  - Provides a declarative, bitwise operator syntax (`Source >> Extract >> Sink`) for constructing ETL pipelines.
- **`silicon_refinery.polars_ext`**:
  - Registers the `.local_llm` namespace in Polars, allowing `pl.col("...").local_llm.extract(Schema)` calls.
- **`silicon_refinery.dspy_ext`**:
  - Implements `AppleFMLM`, a DSPy-compatible language model provider.
- **`silicon_refinery.debugging`**:
  - Contains the structured AI analysis logic used by the `@enhanced_debug` decorator.

---

## Building and Running

### Prerequisites
- **OS:** macOS 26.0+ (requires Apple Foundation Models support)
- **Hardware:** Apple Silicon (M1, M2, M3, M4 series)
- **Python:** 3.10 or higher (3.13+ recommended for free-threading support)

### Setup & Installation
The project uses `uv` for modern, fast dependency management.

```bash
# Clone the Apple FM SDK (if not already installed)
git clone https://github.com/apple/python-apple-fm-sdk
cd python-apple-fm-sdk && uv pip install -e .

# Install SiliconRefinery in editable mode with dev dependencies
cd silicon-refinery
uv pip install -e .[dev]
```

### Key Commands

- **Run Tests:**
  ```bash
  pytest
  ```
- **Linting:**
  ```bash
  ruff check .
  ```
- **Run Examples:**
  ```bash
  python examples/simple_inference.py
  python examples/streaming_example.py
  ```
- **Stress Testing (Performance/Latency):**
  ```bash
  python use_cases/07_stress_test_throughput/example.py
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
- `01_pipeline_operators`: Declarative ETL construction.
- `04_ecosystem_polars`: Massive parallel processing in DataFrames.
- `06_fastapi_integration`: Serving local AI as a REST microservice.
- `09_enhanced_debugging`: Auto-analyzing crashes in production.
