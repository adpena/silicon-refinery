import marimo

__generated_with = "0.11.0"
app = marimo.App(width="wide")


@app.cell
def _():
    import asyncio

    import marimo as mo

    from silicon_refinery.exceptions import AppleFMSetupError

    return AppleFMSetupError, asyncio, mo


@app.cell
def _(mo):
    mo.md(
        """
        # SiliconRefinery Examples Notebook

        This notebook mirrors the repository's example surface:

        - standalone scripts in `examples/`
        - numbered scenarios in `use_cases/*/example.py`
        - desktop app run command from `examples/toga_local_chat_app/`

        All SDK-dependent examples follow the same custom setup exception flow (`AppleFMSetupError`).

        ## Environment assumptions for live runs

        - Apple Silicon (M-series), macOS 26+
        - `uv sync --all-groups`
        - `apple_fm_sdk` available locally
        - Optional extras for some demos:
          - `uv pip install silicon-refinery[arrow]`
          - `uv pip install silicon-refinery[adapters]`
          - `uv pip install briefcase toga-cocoa` (desktop app)

        ## How to use

        - Read each markdown section for context.
        - Toggle `RUN_LIVE_DEMOS = True` to run `examples/*.py` demo code.
        - Toggle `RUN_USE_CASES = True` to execute `use_cases/*/example.py`.
        - If SDK/model setup is missing, cells print `AppleFMSetupError` with troubleshooting steps.
        """
    )
    return


@app.cell
def _():
    RUN_LIVE_DEMOS = False
    RUN_USE_CASES = False
    return RUN_LIVE_DEMOS, RUN_USE_CASES


@app.cell
def _(AppleFMSetupError, asyncio):
    def run_async_demo(coro_fn):
        try:
            return asyncio.run(coro_fn())
        except AppleFMSetupError as exc:
            print(exc)
            return None

    def run_sync_demo(fn):
        try:
            return fn()
        except AppleFMSetupError as exc:
            print(exc)
            return None

    return run_async_demo, run_sync_demo


@app.cell
def _():
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]

    def run_script(relative_path: str) -> int:
        script = repo_root / relative_path
        print(f"$ {sys.executable} {script.relative_to(repo_root)}")
        completed = subprocess.run([sys.executable, str(script)], cwd=repo_root)
        return completed.returncode

    return (run_script,)


@app.cell
def _(mo):
    mo.md("## 1) `simple_inference.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.simple_inference import main as simple_inference_main

    if RUN_LIVE_DEMOS:
        run_async_demo(simple_inference_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run simple_inference.py")
    return (simple_inference_main,)


@app.cell
def _(mo):
    mo.md("## 2) `streaming_example.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.streaming_example import main as streaming_example_main

    if RUN_LIVE_DEMOS:
        run_async_demo(streaming_example_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run streaming_example.py")
    return (streaming_example_main,)


@app.cell
def _(mo):
    mo.md("## 3) `transcript_processing.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_sync_demo):
    from examples.transcript_processing import main as transcript_processing_main

    if RUN_LIVE_DEMOS:
        run_sync_demo(transcript_processing_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run transcript_processing.py")
    return (transcript_processing_main,)


@app.cell
def _(mo):
    mo.md("## 4) `extraction_cache.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.extraction_cache import main as extraction_cache_main

    if RUN_LIVE_DEMOS:
        run_async_demo(extraction_cache_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run extraction_cache.py")
    return (extraction_cache_main,)


@app.cell
def _(mo):
    mo.md("## 5) `functional_pipeline.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.functional_pipeline import main as functional_pipeline_main

    if RUN_LIVE_DEMOS:
        run_async_demo(functional_pipeline_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run functional_pipeline.py")
    return (functional_pipeline_main,)


@app.cell
def _(mo):
    mo.md("## 6) `custom_backend.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.custom_backend import main as custom_backend_main

    if RUN_LIVE_DEMOS:
        run_async_demo(custom_backend_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run custom_backend.py")
    return (custom_backend_main,)


@app.cell
def _(mo):
    mo.md("## 7) `trio_adapter.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.trio_adapter import main as trio_adapter_main

    if RUN_LIVE_DEMOS:
        run_async_demo(trio_adapter_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run trio_adapter.py")
    return (trio_adapter_main,)


@app.cell
def _(mo):
    mo.md("## 8) `context_scope.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.context_scope import main as context_scope_main

    if RUN_LIVE_DEMOS:
        run_async_demo(context_scope_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run context_scope.py")
    return (context_scope_main,)


@app.cell
def _(mo):
    mo.md("## 9) `free_threading.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_sync_demo):
    from examples.free_threading import main as free_threading_main

    if RUN_LIVE_DEMOS:
        run_sync_demo(free_threading_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run free_threading.py")
    return (free_threading_main,)


@app.cell
def _(mo):
    mo.md("## 10) `mmap_scanner.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.mmap_scanner import main as mmap_scanner_main

    if RUN_LIVE_DEMOS:
        run_async_demo(mmap_scanner_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run mmap_scanner.py")
    return (mmap_scanner_main,)


@app.cell
def _(mo):
    mo.md("## 11) `hot_folder_watcher.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.hot_folder_watcher import main as hot_folder_watcher_main

    if RUN_LIVE_DEMOS:
        run_async_demo(hot_folder_watcher_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run hot_folder_watcher.py")
    return (hot_folder_watcher_main,)


@app.cell
def _(mo):
    mo.md("## 12) `jit_diagnostics.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.jit_diagnostics import main as jit_diagnostics_main

    if RUN_LIVE_DEMOS:
        run_async_demo(jit_diagnostics_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run jit_diagnostics.py")
    return (jit_diagnostics_main,)


@app.cell
def _(mo):
    mo.md("## 13) `arrow_bridge.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.arrow_bridge import main as arrow_bridge_main

    if RUN_LIVE_DEMOS:
        run_async_demo(arrow_bridge_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run arrow_bridge.py")
    return (arrow_bridge_main,)


@app.cell
def _(mo):
    mo.md("## 14) `code_auditor.py`")
    return


@app.cell
def _(RUN_LIVE_DEMOS, run_async_demo):
    from examples.code_auditor import main as code_auditor_main

    if RUN_LIVE_DEMOS:
        run_async_demo(code_auditor_main)
    else:
        print("Set RUN_LIVE_DEMOS = True to run code_auditor.py")
    return (code_auditor_main,)


@app.cell
def _(mo):
    mo.md("## 15) `use_cases/01_pipeline_operators/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/01_pipeline_operators/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 01.")
    return


@app.cell
def _(mo):
    mo.md("## 16) `use_cases/02_decorators/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/02_decorators/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 02.")
    return


@app.cell
def _(mo):
    mo.md("## 17) `use_cases/03_async_generators/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/03_async_generators/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 03.")
    return


@app.cell
def _(mo):
    mo.md("## 18) `use_cases/04_ecosystem_polars/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/04_ecosystem_polars/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 04.")
    return


@app.cell
def _(mo):
    mo.md("## 19) `use_cases/05_dspy_optimization/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/05_dspy_optimization/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 05.")
    return


@app.cell
def _(mo):
    mo.md("## 20) `use_cases/06_fastapi_integration/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/06_fastapi_integration/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 06.")
    return


@app.cell
def _(mo):
    mo.md("## 21) `use_cases/07_stress_test_throughput/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/07_stress_test_throughput/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 07.")
    return


@app.cell
def _(mo):
    mo.md("## 22) `use_cases/08_context_limit_test/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/08_context_limit_test/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 08.")
    return


@app.cell
def _(mo):
    mo.md("## 23) `use_cases/09_enhanced_debugging/example.py`")
    return


@app.cell
def _(RUN_USE_CASES, run_script):
    if RUN_USE_CASES:
        run_script("use_cases/09_enhanced_debugging/example.py")
    else:
        print("Set RUN_USE_CASES = True to run use case 09.")
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## 24) Desktop app example (`examples/toga_local_chat_app/`)

        This app is started from CLI or Briefcase rather than directly as a single script.
        Recommended command:

        ```bash
        uv run silicon-refinery chat
        ```
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Notebook tips

        - Keep `RUN_LIVE_DEMOS = False` and `RUN_USE_CASES = False` when editing.
        - Use per-script CLI runs for focused output:
          - `silicon-refinery example <name>`
          - `uv run python use_cases/<NN_name>/example.py`
        - For setup verification:
          - `uv run silicon-refinery doctor`
        - For full example validation:
          - `uv run silicon-refinery smoke`
        """
    )
    return


if __name__ == "__main__":
    app.run()
