<div align="center">
  <h1>🏭 SiliconRefinery</h1>
  <p><b>A Zero-Trust, Zero-Latency, Zero-Cost ETL Framework for Apple Silicon</b></p>
  <p>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT"></a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
    <img src="https://img.shields.io/badge/macOS-26.0+-lightgrey.svg" alt="macOS Version">
  </p>
</div>

Welcome to **SiliconRefinery**, an enterprise-grade Python framework built directly on top of the newly released [Apple Foundation Models SDK (`python-apple-fm-sdk`)](https://github.com/apple/python-apple-fm-sdk).

This repository is simultaneously a documentation hub, an engineering blog, and a research paper exploring the bleeding edge of on-device LLM inference.

## ✨ A Note of Gratitude

Before diving into the technical details, we want to extend a massive thank you to Apple's engineering and open-source teams. Releasing the `python-apple-fm-sdk` democratizes access to incredibly powerful, hardware-accelerated intelligence. These are very early days for local AI, and providing a pythonic bridge to the Foundation Models framework unlocks a world of exciting possibilities for the open-source community. The possibilities are truly boundless.

---

## 🚀 The Vision: A Billion-Dollar ETL Paradigm

Enterprises today sit on petabytes of "dark data" — unstructured logs, massive CSV dumps, internal support emails, and sensitive medical records. Historically, processing this data intelligently meant sending it to cloud providers like OpenAI or Anthropic, incurring millions of dollars in token costs and running head-first into strict GDPR, HIPAA, and InfoSec compliance nightmares.

**SiliconRefinery flips the paradigm.** It transforms your Apple Silicon machine (or a fleet of Mac Minis in a server rack) into a hyper-secure, high-throughput extraction node. 

By running massive unstructured datasets through on-device intelligence, `SiliconRefinery` allows you to extract, redact, and enrich data **without ever sending a single byte to the cloud.**

### Why use SiliconRefinery?

- 🛡️ **Absolute Zero-Trust (No Data Egress):** Process sensitive data safely.
- 💸 **Zero Token Costs:** Skip the millions of dollars in API costs.
- ⚡ **Insane Throughput:** Process 300+ characters per second entirely locally (see benchmarks below).
- 🐍 **Modern Pythonic Wizardry:** Uses native `asyncio`, type annotations, generic `TypeVar` decorators, and extensible generators.
- 🏗️ **Guaranteed Schemas:** Powered by the Apple FM SDK's strict `generating` protocol.
- 🧠 **Dynamic History Management:** Our streaming engines natively support session history compaction and sliding-window retries to prevent context explosion.

---

## 📦 Installation & Setup

`SiliconRefinery` requires macOS 26.0+ and Python 3.10+ running on a compatible Apple Silicon Mac. We strongly recommend using modern tooling like `uv` or `ruff`.

1. **Install the Apple FM SDK (Beta):**
   ```bash
   git clone https://github.com/apple/python-apple-fm-sdk
   cd python-apple-fm-sdk
   uv pip install -e .
   ```

2. **Install `SiliconRefinery` from PyPI:**
   ```bash
   uv pip install silicon_refinery
   ```

---

## 🛠️ The 7 Pillars of SiliconRefinery

`SiliconRefinery` abstract away the raw SDK interactions into elegant, battle-tested patterns.

### 1. The `@local_extract` Decorator
Turn any Python function into a strongly-typed LLM extraction engine. The docstring natively acts as the system prompt. Enable `debug_timing=True` to print latency metrics directly to your terminal.

```python
from silicon_refinery import local_extract
import apple_fm_sdk as fm

@fm.generable()
class MedicalRecord:
    patient_symptoms: list[str] = fm.guide(description="Isolated symptoms")
    suggested_triage: str = fm.guide(anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"])

@local_extract(schema=MedicalRecord, debug_timing=True)
async def parse_doctor_notes(raw_text: str) -> MedicalRecord:
    """Extract structured medical data and infer triage urgency."""
    pass

# Usage:
record = await parse_doctor_notes("Severe migraines and nausea for 4 days. ER immediately.")
print(record.suggested_triage) # -> "HIGH"
```

### 2. The `@enhanced_debug` Decorator (AI Crash Analysis)
Wrap any function to automatically catch exceptions, dump the traceback, and invoke the Neural Engine to perform a detailed root-cause analysis! It provides a certainty level, suggested fixes, and optionally dumps a formatted payload to file to easily feed into more powerful coding agents like Codex or Claude.

```python
from silicon_refinery import enhanced_debug

@enhanced_debug(route_to="stdout", prompt_file="crash_report_for_llm.txt")
def divide_data(a, b):
    return a / b

# Will catch the ZeroDivisionError, print the traceback, and use the local AI to explain why it failed!
divide_data(10, 0) 
```

### 3. Concurrent Async Streams (`stream_extract`)
Process gigabytes of data lazily using asynchronous generators. This function includes built-in line-level chunking (`lines_per_chunk`), advanced context window management (`history_mode`), and native **concurrency**.

```python
from silicon_refinery import stream_extract

# concurrency defaults to os.cpu_count(). It automatically spawns parallel tasks 
# and yields structured objects out-of-order, exactly like `imap_unordered`!
# 'compact' history_mode asks the LLM to summarize previous context to prevent overflow.
async for enriched in stream_extract(
    review_stream, 
    schema=Feedback, 
    instructions="Analyze sentiment.", 
    lines_per_chunk=5,
    history_mode='clear',
    debug_timing=True
):
    await db.insert(enriched)
```

### 4. FastAPI Integration (Local AI Microservices)
Spin up an instantaneous, secure local REST API powered by the Neural Engine.
*(See `use_cases/06_fastapi_integration/` for full example).*

```python
from fastapi import FastAPI
from silicon_refinery import local_extract

app = FastAPI()

@app.post("/api/v1/extract")
async def extract_data(text: str):
    # Invokes the local Neural Engine securely on your machine!
    return await parse_doctor_notes(text)
```

### 5. The Polars Ecosystem Extension
Run zero-cost local inference directly inside a Rust-backed Polars DataFrame using our `.local_llm` namespace.
*(See `use_cases/04_ecosystem_polars/` for full example).*

```python
import polars as pl
from silicon_refinery.polars_ext import LocalLLMExpr 

df = pl.read_csv("support_tickets.csv")

# Executes the local model across the dataframe column entirely on-device
enriched_df = df.with_columns(
    extracted_json=pl.col("email_body").local_llm.extract(schema=Ticket)
)
```

### 6. Composable Pipeline Operators (`>>`)
Use UNIX-style bitwise operators to construct declarative ETL pipelines.
*(See `use_cases/01_pipeline_operators/` for full example).*

```python
from silicon_refinery import Source, Extract, Sink
import csv

pipeline = (
    Source(csv.DictReader(open("server_logs.csv"))) 
    >> Extract(schema=LogEntry, instructions="Parse the raw server log.") 
    >> Sink(callback=lambda item: print(f"Level: {item.level}"))
)
await pipeline.execute()
```

### 7. DSPy Provider Integration (Agentic Swarms)
Initialize DSPy with our custom `AppleFMLM` provider to use robust agent swarms and prompt compilers on free Apple hardware.
*(See `use_cases/05_dspy_optimization/` for full example).*

```python
import dspy
from silicon_refinery.dspy_ext import AppleFMLM

dspy.settings.configure(lm=AppleFMLM())
classifier = dspy.ChainOfThought("customer_email -> summary, priority")
result = classifier(email="I am locked out of my account!")
```

---

## 📊 Empirical Benchmarks & Limitations

We don't just build; we measure. To prove `SiliconRefinery` is production-ready, we built aggressive profiling scripts (`use_cases/07_stress_test_throughput` and `use_cases/08_context_limit_test`).

### Test Environment
*   **Hardware:** Apple M1 (MacBookPro17,1) - 8 Cores (4P, 4E), 8GB Unified Memory
*   **OS:** macOS 26.3 (Build 25D125)
*   **Language:** Python 3.14.3
*   **SDK:** `python-apple-fm-sdk` version 0.1.0 (Beta)

### 1. Massive Throughput
Running a dataset of 1,000 unstructured logs bundled into chunks, `SiliconRefinery` consistently hit throughputs of **250 to 350+ characters per second** (roughly ~80-100 tokens/sec), running purely on-device without any cloud connectivity.

### 2. Context Window Limits
During our payload escalation test, the local foundation model flawlessly processed a ~25,000-character block (approx 6,250 tokens) in 12.5 seconds. However, injecting payloads larger than ~32,000 characters immediately triggered an `ExceededContextWindowSizeError`. 

*Note: This is why `SiliconRefinery` natively clears session history (`history_mode='clear'`) by default during infinite streaming to prevent context accumulation.*

---

## 🔮 Future Work: The Pythonic Frontier

While `SiliconRefinery` is built with robust typing, dataclasses, generic `TypeVar` extractors, and concurrent async generators, there is more magic to unlock as the Python ecosystem evolves:
- **IO & Protocol Wrappers:** We are exploring building native adapters for `BytesIO` streams, `multiprocessing` arrays, and real-time `websockets` to feed the Neural Engine from any source.
- **Free-Threading & Subinterpreters (PEP 703/684):** Python 3.13+ introduces experimental features that bypass the GIL. `SiliconRefinery`'s `stream_extract(concurrency=X)` implementation is architected to natively exploit the `python3.13t` free-threaded interpreter, potentially allowing parallel Foundation Model C-extensions to run truly concurrently on separate M-series cores without Python-level locking.
- **JIT Compilation (PEP 744):** The new copy-and-patch JIT in Python (enabled via `PYTHON_JIT=1`) can drastically reduce overhead during massive Polars/Pandas map-batch conversions or when managing thousands of asynchronous tasks.

---

## 🤝 Contributing

We welcome the world's foremost engineers to experiment, break things, and contribute. Please run `uv pip install -e .[dev]` and ensure your code passes `ruff check .` and `pytest` before submitting PRs. 

If you have questions, ideas, or feedback, please reach out to me directly:
**Email:** `adpena@gmail.com`

## 📄 License

This project is licensed under the permissive MIT License. See [LICENSE](LICENSE) for details.

---
*Built with ❤️ utilizing the `python-apple-fm-sdk`.*
