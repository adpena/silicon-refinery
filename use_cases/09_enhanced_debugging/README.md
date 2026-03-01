# Use Case 09: Enhanced AI Debugging

The `@enhanced_debug` decorator catches exceptions, prints the standard Python traceback, and then invokes the Apple Neural Engine to perform an automated root-cause analysis. By default, summaries go to `stderr` and prompts go to `stdout`.

## How It Works

```python
from pathlib import Path
from fmtools import enhanced_debug

PROMPT_PATH = Path("crash_report_for_llm.txt")
if PROMPT_PATH.exists():
    PROMPT_PATH.unlink()

@enhanced_debug(summary_to="stdout", prompt_to=str(PROMPT_PATH), silenced=False)
def process_data(data_payload):
    """A buggy function that will inevitably crash."""
    parsed_value = data_payload["value"] + 10
    return parsed_value

try:
    process_data({"value": "100"})  # TypeError: can only concatenate str (not "int") to str
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

When the function crashes, FMTools will:

1. Print the standard Python traceback to stderr
2. Build an evidence-focused debug query (traceback + context + frame/logging clues) and send it to the on-device Foundation Model
3. Retry automatically with smaller tail-prioritized traceback payloads if `apple_fm_sdk.ExceededContextWindowSizeError` is raised
4. Print a structured forensic diagnosis including likely fix locations with path/line evidence
5. If `prompt_to` is enabled, run a second handoff-generation query that produces candidate edits, instrumentation, verification commands, and an agent-ready prompt
6. Route the detailed prompt via `prompt_to` (defaults to `stdout`; supports `stderr`, `log`, and file path `.txt` output)

## Sample Output

The file [`sample_crash_report.txt`](sample_crash_report.txt) shows what the generated prompt file looks like — ready to paste into Claude, Codex, or any coding assistant for a deeper fix.

## Run It

```bash
python use_cases/09_enhanced_debugging/example.py
```
