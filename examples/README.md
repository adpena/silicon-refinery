# Examples

This directory contains runnable, focused examples for FMTools APIs.

## Prerequisites

- CPython 3.13+

```bash
uv sync --all-groups
source .venv/bin/activate
```

These examples assume `apple_fm_sdk` is installed and the Foundation Model is
available on your machine. (`fmtools[api]` and
`fmtools[adapters]` extras do not install `apple_fm_sdk`.)

All example scripts use a shared custom exception:
- `fmtools.exceptions.AppleFMSetupError`

When setup checks fail, scripts print a standard troubleshooting checklist and
exit gracefully.

For real trio-native channels, install the optional extra (the included
`trio_adapter.py` demo also runs without trio):

```bash
uv pip install fmtools[adapters]
```

For Arrow IPC examples:

```bash
uv pip install fmtools[arrow]
```

For the marimo notebook:

```bash
uv pip install marimo
marimo edit examples/examples_notebook.py
```

CLI shortcuts:

```bash
fmtools example --list
fmtools example simple_inference
fmtools smoke
fmtools notebook
```

```text
Actual output excerpt from local run (`fmtools example --list`):

Available examples:
  arrow_bridge
  code_auditor
  context_scope
  custom_backend
  extraction_cache
  free_threading
  functional_pipeline
  hot_folder_watcher
  jit_diagnostics
  mmap_scanner
  simple_inference
  streaming_example
  transcript_processing
  trio_adapter
```

For the Toga + Briefcase desktop chat demo:

```bash
uv pip install briefcase toga-cocoa
# Apple FM SDK comes from this repo's uv source configuration:
uv sync --group apple
```

## Example Catalog

| Script | What it demonstrates | Run |
|---|---|---|
| `simple_inference.py` | Minimal direct usage of `apple_fm_sdk.LanguageModelSession` | `uv run python examples/simple_inference.py` |
| `streaming_example.py` | Token/segment streaming with `session.stream_response(...)` | `uv run python examples/streaming_example.py` |
| `transcript_processing.py` | Transcript loading, analytics, and JSONL export helpers (defaults to `datasets/transcript_sample.json`) | `uv run python examples/transcript_processing.py` |
| `extraction_cache.py` | `ExtractionCache` + `@cached_local_extract` cache-hit workflow | `uv run python examples/extraction_cache.py` |
| `functional_pipeline.py` | Functional pipeline composition (`source`, `map_fn`, `filter_fn`, `extract`, `collect`) | `uv run python examples/functional_pipeline.py` |
| `custom_backend.py` | Swapping runtime backend via `set_backend(...)` | `uv run python examples/custom_backend.py` |
| `trio_adapter.py` | `TrioAdapter` wrapping a trio-style receive channel | `uv run python examples/trio_adapter.py` |
| `context_scope.py` | Context-local session/model scoping with `session_scope(...)` | `uv run python examples/context_scope.py` |
| `free_threading.py` | Free-threading helpers (`CriticalSection`, `AtomicCounter`, `ThreadSafeDict`) | `uv run python examples/free_threading.py` |
| `mmap_scanner.py` | Windowed `MMapScanner` + line-based `line_split_scanner(...)` | `uv run python examples/mmap_scanner.py` |
| `hot_folder_watcher.py` | Polling watcher events via `HotFolder` | `uv run python examples/hot_folder_watcher.py` |
| `jit_diagnostics.py` | Runtime counters using `@diagnose` + `diagnostics().report()` | `uv run python examples/jit_diagnostics.py` |
| `arrow_bridge.py` | Arrow IPC file/buffer round-trips + `ArrowStreamWriter` | `uv run python examples/arrow_bridge.py` |
| `code_auditor.py` | Auditor APIs (`audit_file`, `audit_directory(max_concurrency=...)`, `audit_diff`) with a demo backend | `uv run python examples/code_auditor.py` |
| `examples_notebook.py` | Comprehensive marimo notebook covering `examples/` plus `use_cases/*/example.py` | `marimo edit examples/examples_notebook.py` |
| `toga_local_chat_app/` | Local-first desktop chat app (vertical chat tabs, steering interjections, streaming, sqlite memory, familiar slash commands) | `cd examples/toga_local_chat_app && briefcase dev` |

## Notes

- These scripts are examples, not tests; they are intended for interactive learning and smoke runs.
- Example setup failures raise `AppleFMSetupError` and include actionable troubleshooting steps.
- API docs for the underlying wrappers are in the root
  [`README.md`](../README.md#public-api-breakdown) and
  [`API Reference & Tutorials`](../README.md#api-reference--tutorials) sections.
