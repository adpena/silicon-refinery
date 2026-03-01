import asyncio
import concurrent.futures
import functools
import importlib
import inspect
import logging
import re
import sys
import traceback
from pathlib import Path

from .exceptions import AppleFMSetupError, ensure_model_available
from .protocols import create_model, create_session

fm = importlib.import_module("apple_fm_sdk")

logger = logging.getLogger("fmtools.debug")
_SYNC_ANALYSIS_TIMEOUT_SECONDS = 30
_DEBUG_QUERY_MAX_CHARS = 24_000
_HANDOFF_QUERY_MAX_CHARS = 20_000
_TRACEBACK_MIN_LINES = 24
_TRACEBACK_RETRY_RATIOS = (1.0, 0.75, 0.5, 0.35, 0.25)
_TEXT_RETRY_MIN_CHARS = 1_200
_FRAME_HINT_LIMIT = 10
_SOURCE_SNIPPET_LIMIT = 5
_SOURCE_CONTEXT_LINES = 8
_CONTEXT_OVERFLOW_HINTS = (
    "exceededcontextwindowsizeerror",
    "context window size exceeded",
    "exceeded context window size",
    "exceeded context window",
    "context length",
    "maximum context length",
    "too many tokens",
)
_LOG_SIGNAL_PATTERN = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")


@fm.generable()
class DebuggingAnalysis:
    error_summary: str = fm.guide(description="A brief summary of what went wrong.")
    possible_causes: list[str] = fm.guide(
        description="List of likely root causes based on the traceback."
    )
    certainty_level: str = fm.guide(
        anyOf=["LOW", "MEDIUM", "HIGH"], description="Confidence in the diagnosis."
    )
    severity: str = fm.guide(
        anyOf=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        description="Operational severity of the incident.",
    )
    blast_radius: str = fm.guide(
        description="Scope of impact (single call-site vs wider subsystem)."
    )
    suggested_fix: str = fm.guide(description="Actionable steps to fix the issue.")
    likely_fix_locations: list[str] = fm.guide(
        description=(
            "Prioritized list of likely fix locations. Include concrete file paths and line ranges "
            "formatted as 'path:line[-line] | reason | evidence=F#'."
        )
    )
    evidence: list[str] = fm.guide(
        description="Evidence-backed observations citing frame IDs (e.g. F1, F2)."
    )
    first_actions_30m: list[str] = fm.guide(
        description="First 30-minute debugging plan with concrete steps."
    )
    verification_steps: list[str] = fm.guide(
        description="How to validate the fix and avoid regressions."
    )
    unknowns: list[str] = fm.guide(
        description="Unknowns or ambiguities that require additional evidence."
    )


@fm.generable()
class DebuggingHandoff:
    objective: str = fm.guide(description="One-sentence objective for the next debugging agent.")
    prioritized_steps: list[str] = fm.guide(
        description="Ordered plan of attack, each step concrete and testable."
    )
    candidate_edits: list[str] = fm.guide(
        description="Candidate files/line ranges to inspect or edit."
    )
    instrumentation: list[str] = fm.guide(
        description="Logging/instrumentation suggestions with file/line hints."
    )
    verification_commands: list[str] = fm.guide(
        description="Commands/tests to verify correctness and prevent regressions."
    )
    risks: list[str] = fm.guide(description="Potential regression risks and mitigations.")
    handoff_prompt: str = fm.guide(
        description=(
            "A concise but detailed prompt for a coding/debugging agent, grounded only in provided evidence."
        )
    )


def _run_analysis_sync_once(
    exc: Exception,
    func_name: str,
    summary_to: str | None,
    prompt_to: str | None,
    silenced: bool,
    summary_log_level: str | int,
    prompt_log_level: str | int,
) -> None:
    coro = _handle_exception(
        exc,
        func_name,
        summary_to,
        prompt_to,
        silenced,
        summary_log_level,
        prompt_log_level,
    )
    try:
        asyncio.run(coro)
    finally:
        coro.close()


def _run_analysis_sync_best_effort(
    exc: Exception,
    func_name: str,
    summary_to: str | None,
    prompt_to: str | None,
    silenced: bool,
    summary_log_level: str | int,
    prompt_log_level: str | int,
) -> None:
    try:
        _run_analysis_sync_once(
            exc,
            func_name,
            summary_to,
            prompt_to,
            silenced,
            summary_log_level,
            prompt_log_level,
        )
        return
    except RuntimeError:
        # asyncio.run() can fail if the current thread already has a running loop.
        pass
    except Exception:
        logger.warning("[FMTools Debug] AI analysis failed.", exc_info=True)
        return

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(
            _run_analysis_sync_once,
            exc,
            func_name,
            summary_to,
            prompt_to,
            silenced,
            summary_log_level,
            prompt_log_level,
        )
        try:
            future.result(timeout=_SYNC_ANALYSIS_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.warning("[FMTools Debug] AI analysis timed out.")
        except Exception:
            logger.warning("[FMTools Debug] AI analysis failed.", exc_info=True)
    finally:
        # Do not block original exception propagation on background analysis shutdown.
        executor.shutdown(wait=False, cancel_futures=True)


def enhanced_debug(
    summary_to: str | None = "stderr",
    prompt_to: str | None = "stdout",
    silenced: bool = False,
    summary_log_level: str | int = "error",
    prompt_log_level: str | int = "info",
):
    """
    A decorator that catches exceptions, prints the traceback, and invokes the Apple Foundation Model
    to perform a detailed root-cause analysis.

    Args:
        summary_to: Where to output the analysis ("stdout", "stderr", "log", or None to silence).
        prompt_to: Where to route the detailed prompt payload. Use "stdout" (default), a
                   relative/absolute file path (written as .txt), "log", or None to silence prompt output.
        silenced: If True, silences all enhanced_debug outputs and skips analysis.
        summary_log_level: Logging level when summary_to="log" (e.g. "info", "warning", "error").
        prompt_log_level: Logging level when prompt_to="log" (e.g. "debug", "info", "warning").
    """

    def decorator(func):
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    try:
                        await _handle_exception(
                            e,
                            func.__name__,
                            summary_to,
                            prompt_to,
                            silenced,
                            summary_log_level,
                            prompt_log_level,
                        )
                    except Exception:
                        logger.warning("[FMTools Debug] AI analysis failed.", exc_info=True)
                    raise

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    _run_analysis_sync_best_effort(
                        e,
                        func.__name__,
                        summary_to,
                        prompt_to,
                        silenced,
                        summary_log_level,
                        prompt_log_level,
                    )
                    raise

            return sync_wrapper

    return decorator


def _resolve_log_level(level: str | int, fallback: int = logging.ERROR) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        if level.isdigit():
            return int(level)
        resolved = logging.getLevelName(level.upper())
        if isinstance(resolved, int):
            return resolved
    logger.warning(
        "[FMTools Debug] Unsupported log level '%s'; falling back to %s.",
        level,
        logging.getLevelName(fallback),
    )
    return fallback


def _log_text(message: str, level: str | int, fallback: int = logging.ERROR) -> None:
    logger.log(_resolve_log_level(level, fallback), message)


def _as_text(value, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_as_text(item).strip() for item in value if _as_text(item).strip()]
    text = _as_text(value).strip()
    return [text] if text else []


def _build_trace_frames(
    exc: Exception, limit: int = _FRAME_HINT_LIMIT
) -> list[dict[str, str | int]]:
    extracted = traceback.extract_tb(exc.__traceback__)
    if not extracted:
        return []

    selected = list(reversed(extracted[-limit:]))  # Most recent frame first.
    frames: list[dict[str, str | int]] = []
    for idx, frame in enumerate(selected, start=1):
        line_number = frame.lineno if frame.lineno is not None else 0
        frames.append(
            {
                "id": f"F{idx}",
                "path": frame.filename,
                "line": int(line_number),
                "function": frame.name,
                "code": (frame.line or "").strip(),
            }
        )
    return frames


def _render_frame_hints(frames: list[dict[str, str | int]]) -> str:
    if not frames:
        return "- none detected"
    rendered = []
    for frame in frames:
        rendered.append(
            f"- {frame['id']} | {frame['path']}:{frame['line']} | "
            f"{frame['function']} | code={_as_text(frame['code'], '<missing>')}"
        )
    return "\n".join(rendered)


def _build_source_snippet(path: str, line: int, context: int = _SOURCE_CONTEXT_LINES) -> str | None:
    try:
        source_path = Path(path).expanduser().resolve()
    except Exception:
        return None
    if not source_path.is_file():
        return None

    try:
        lines = source_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        try:
            lines = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return None

    if not lines:
        return None

    line_num = max(1, min(line, len(lines)))
    start = max(1, line_num - context)
    end = min(len(lines), line_num + context)

    snippet_lines = [f"# {source_path}:{line_num} (showing {start}-{end})"]
    for index in range(start, end + 1):
        marker = ">>" if index == line_num else "  "
        snippet_lines.append(f"{marker} {index:>5}: {lines[index - 1]}")
    return "\n".join(snippet_lines)


def _build_source_snippets(
    frames: list[dict[str, str | int]], limit: int = _SOURCE_SNIPPET_LIMIT
) -> str:
    snippets: list[str] = []
    seen: set[tuple[str, int]] = set()
    for frame in frames:
        path = _as_text(frame.get("path"))
        line = int(frame.get("line", 0) or 0)
        key = (path, line)
        if key in seen:
            continue
        seen.add(key)
        snippet = _build_source_snippet(path, line)
        if snippet:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break

    if not snippets:
        return "- no readable local source snippets found for traceback frames"
    return "\n\n".join(snippets)


def _trim_from_tail(text: str, max_chars: int, notice: str) -> str:
    if max_chars <= 0:
        return f"[{notice}; omitted]"
    if len(text) <= max_chars:
        return text
    keep_chars = max(256, max_chars - len(notice) - 12)
    return f"[{notice}; showing last {keep_chars} chars]\n{text[-keep_chars:]}"


def _build_text_retry_candidates(text: str) -> list[str]:
    if not text:
        return [""]
    candidates: list[str] = []
    seen: set[str] = set()
    total_chars = len(text)
    for ratio in _TRACEBACK_RETRY_RATIOS:
        keep_chars = max(_TEXT_RETRY_MIN_CHARS, int(total_chars * ratio))
        keep_chars = min(total_chars, keep_chars)
        candidate = text[-keep_chars:]
        if candidate in seen:
            continue
        seen.add(candidate)
        if keep_chars < total_chars:
            candidate = (
                f"[Context payload trimmed to {keep_chars}/{total_chars} chars for retry.]\n"
                f"{candidate}"
            )
        candidates.append(candidate)
    return candidates or [text]


def _build_crash_envelope(
    *,
    func_name: str,
    exc: Exception,
    frame_hints: str,
    logging_signals: str,
    source_snippets: str,
    full_traceback: str,
) -> str:
    return (
        "Crash Envelope\n"
        f"- function_wrapper: {func_name}\n"
        f"- exception: {_format_exception_summary(exc)}\n"
        f"- traceback_lines: {len(_traceback_lines(full_traceback))}\n\n"
        "Frame Hints:\n"
        f"{frame_hints}\n\n"
        "Logging Signals:\n"
        f"{logging_signals}\n\n"
        "Source Snippets:\n"
        f"{source_snippets}\n\n"
        "Traceback Tail:\n"
        "```python\n"
        f"{_traceback_tail(full_traceback, max_lines=120)}\n"
        "```"
    )


def _fallback_fix_locations(frames: list[dict[str, str | int]], limit: int = 4) -> list[str]:
    locations: list[str] = []
    for frame in frames[:limit]:
        locations.append(
            f"{frame['path']}:{frame['line']} | inspect function '{frame['function']}' "
            f"| evidence={frame['id']}"
        )
    return locations


def _traceback_lines(tb_str: str) -> list[str]:
    return tb_str.strip("\n").splitlines()


def _format_exception_summary(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def _extract_logging_signals(tb_str: str, limit: int = 8) -> list[str]:
    signals = []
    for line in _traceback_lines(tb_str):
        upper_line = line.upper()
        if _LOG_SIGNAL_PATTERN.search(upper_line):
            signals.append(line.strip())
    return signals[-limit:]


def _traceback_tail(tb_str: str, max_lines: int = 80) -> str:
    lines = _traceback_lines(tb_str)
    if len(lines) <= max_lines:
        return tb_str.strip()
    return "\n".join(lines[-max_lines:])


def _looks_like_context_overflow(exc: Exception) -> bool:
    context_overflow_type = getattr(fm, "ExceededContextWindowSizeError", None)
    if isinstance(context_overflow_type, type) and isinstance(exc, context_overflow_type):
        return True
    message = str(exc).lower()
    return any(hint in message for hint in _CONTEXT_OVERFLOW_HINTS)


def _build_traceback_payload(tb_str: str, ratio: float) -> tuple[str, int, int]:
    lines = _traceback_lines(tb_str)
    total_lines = len(lines)
    if total_lines == 0:
        return "", 0, 0

    keep_lines = max(_TRACEBACK_MIN_LINES, int(total_lines * ratio))
    keep_lines = min(total_lines, keep_lines)
    trimmed_lines = total_lines - keep_lines
    payload = "\n".join(lines[-keep_lines:])

    if trimmed_lines > 0:
        payload = (
            f"[Traceback truncated for model context; showing tail {keep_lines}/{total_lines} lines.]\n"
            f"{payload}"
        )
    return payload, trimmed_lines, total_lines


def _compose_debug_query(
    *,
    func_name: str,
    exc: Exception,
    traceback_payload: str,
    attempt: int,
    total_attempts: int,
    trimmed_lines: int,
    total_lines: int,
    frame_hints: str,
    logging_signals: str,
    source_snippets: str,
) -> str:
    lines_kept = total_lines - trimmed_lines if total_lines else 0
    context_section = f"""Relevant frame hints:
{frame_hints}

Logging signals:
{logging_signals}

Local source snippets around failing frames:
{source_snippets}
"""
    context_section = _trim_from_tail(
        context_section,
        max_chars=int(_DEBUG_QUERY_MAX_CHARS * 0.45),
        notice="Context section truncated",
    )

    prefix = f"""You are a senior debugging analyst. Provide a strict, evidence-based diagnosis of this Python failure.

Rules:
- Use only evidence from the provided traceback and signals.
- Do not invent files, line numbers, stack frames, logs, or runtime state.
- If evidence is insufficient, state uncertainty explicitly.
- Focus on root-cause debugging and concrete next validation steps.

Analysis objectives:
1) error_summary: precise failure synopsis tied to exception + likely failing boundary.
2) possible_causes: evidence-backed root-cause candidates.
3) certainty_level: LOW|MEDIUM|HIGH based on evidence quality.
4) severity + blast_radius: practical impact scope.
5) likely_fix_locations: include file paths + line ranges, plus reason and evidence frame IDs.
6) evidence: list concrete proof points citing frame IDs.
7) first_actions_30m + verification_steps + unknowns.
8) suggested_fix: concise initial remediation path.

Output policy:
- Every claim must be grounded in evidence.
- Every likely_fix_locations item must include at least one concrete path:line range.
- Keep output detailed, technical, and implementation-oriented.

Crash context:
- Function wrapper: {func_name}
- Exception: {_format_exception_summary(exc)}
- Traceback lines provided: {lines_kept}/{total_lines or lines_kept}
- Query attempt: {attempt}/{total_attempts}

{context_section}

Traceback payload:
```python
"""
    suffix = "\n```\n"
    available_tb_chars = _DEBUG_QUERY_MAX_CHARS - len(prefix) - len(suffix)
    tb_payload = traceback_payload
    if available_tb_chars > 0 and len(tb_payload) > available_tb_chars:
        keep_chars = max(512, available_tb_chars - 120)
        tb_payload = (
            "[Traceback truncated by character budget; retaining most recent tail section.]\n"
            + tb_payload[-keep_chars:]
        )

    return f"{prefix}{tb_payload}{suffix}"


def _format_triage_block(analysis: DebuggingAnalysis) -> str:
    return "\n".join(
        [
            f"- error_summary: {_as_text(getattr(analysis, 'error_summary', ''))}",
            f"- certainty_level: {_as_text(getattr(analysis, 'certainty_level', ''))}",
            f"- severity: {_as_text(getattr(analysis, 'severity', ''))}",
            f"- blast_radius: {_as_text(getattr(analysis, 'blast_radius', ''))}",
            f"- possible_causes: {_as_list(getattr(analysis, 'possible_causes', []))}",
            f"- likely_fix_locations: {_as_list(getattr(analysis, 'likely_fix_locations', []))}",
            f"- evidence: {_as_list(getattr(analysis, 'evidence', []))}",
            f"- first_actions_30m: {_as_list(getattr(analysis, 'first_actions_30m', []))}",
            f"- verification_steps: {_as_list(getattr(analysis, 'verification_steps', []))}",
            f"- unknowns: {_as_list(getattr(analysis, 'unknowns', []))}",
            f"- suggested_fix: {_as_text(getattr(analysis, 'suggested_fix', ''))}",
        ]
    )


def _compose_handoff_query(
    *,
    func_name: str,
    exc: Exception,
    crash_envelope: str,
    analysis: DebuggingAnalysis,
    attempt: int,
    total_attempts: int,
) -> str:
    triage_block = _trim_from_tail(
        _format_triage_block(analysis),
        max_chars=int(_HANDOFF_QUERY_MAX_CHARS * 0.35),
        notice="Triage block truncated",
    )
    envelope_block = _trim_from_tail(
        crash_envelope,
        max_chars=int(_HANDOFF_QUERY_MAX_CHARS * 0.45),
        notice="Crash envelope truncated",
    )
    query = f"""You are preparing a high-signal debugging handoff for another engineering agent.

Rules:
- Use only the provided triage + crash evidence.
- Do not fabricate files, lines, commands, or runtime details.
- Include concrete file paths and line ranges wherever possible.
- Keep response directly actionable.

Expected output:
- objective
- prioritized_steps
- candidate_edits (must include path:line or path:line-line)
- instrumentation (with where/why)
- verification_commands
- risks
- handoff_prompt (ready for a coding/debugging agent)

Context:
- Function wrapper: {func_name}
- Exception: {_format_exception_summary(exc)}
- Query attempt: {attempt}/{total_attempts}

Forensic triage JSON-like block:
{triage_block}

Crash envelope:
```text
{envelope_block}
```
"""
    if len(query) <= _HANDOFF_QUERY_MAX_CHARS:
        return query
    return _trim_from_tail(query, _HANDOFF_QUERY_MAX_CHARS, "Handoff query truncated")


def _build_handoff_fallback(analysis: DebuggingAnalysis) -> str:
    likely_fixes = _as_list(getattr(analysis, "likely_fix_locations", []))
    causes = _as_list(getattr(analysis, "possible_causes", []))
    evidence = _as_list(getattr(analysis, "evidence", []))
    actions = _as_list(getattr(analysis, "first_actions_30m", []))
    verify = _as_list(getattr(analysis, "verification_steps", []))

    sections = [
        "Objective: Complete root-cause debugging and implement a validated fix with regression coverage.",
        f"Triage summary: {_as_text(getattr(analysis, 'error_summary', ''))}",
        f"Suggested fix: {_as_text(getattr(analysis, 'suggested_fix', ''))}",
        "Likely fix locations:",
        *(f"- {item}" for item in (likely_fixes or ["No explicit file/line location returned."])),
        "Evidence:",
        *(f"- {item}" for item in (evidence or ["No explicit evidence entries returned."])),
        "Possible causes:",
        *(f"- {item}" for item in (causes or ["No explicit causes returned."])),
        "First actions (30m):",
        *(
            f"- {item}"
            for item in (
                actions or ["Reproduce locally and add targeted logging around failing frame."]
            )
        ),
        "Verification:",
        *(
            f"- {item}"
            for item in (
                verify
                or ["Run focused tests around the failing path and affected integration points."]
            )
        ),
    ]
    return "\n".join(sections)


def _build_prompt_content(
    *,
    func_name: str,
    exc: Exception,
    full_traceback: str,
    query_payload: str,
    crash_envelope: str,
    analysis: DebuggingAnalysis,
    handoff: DebuggingHandoff | None,
    context_retries: int,
    handoff_retries: int,
) -> str:
    safe_query_payload = query_payload.replace("```", "'''")
    safe_crash_envelope = crash_envelope.replace("```", "'''")
    triage_block = _format_triage_block(analysis)
    handoff_prompt = ""
    if handoff is not None:
        handoff_prompt = _as_text(getattr(handoff, "handoff_prompt", "")).strip()
    if not handoff_prompt:
        handoff_prompt = _build_handoff_fallback(analysis)

    handoff_steps = _as_list(getattr(handoff, "prioritized_steps", [])) if handoff else []
    handoff_edits = _as_list(getattr(handoff, "candidate_edits", [])) if handoff else []
    handoff_instrumentation = _as_list(getattr(handoff, "instrumentation", [])) if handoff else []
    handoff_verify = _as_list(getattr(handoff, "verification_commands", [])) if handoff else []
    handoff_risks = _as_list(getattr(handoff, "risks", [])) if handoff else []

    return f"""I encountered a crash in my Python application.

Debugging objective for the next agent:
- Perform deep, evidence-based debugging only.
- Avoid assumptions and hallucinations; explicitly call out unknowns.
- Include concrete line/frame references, exception details, and logging clues.
- Prioritize actionable validation steps and instrumentation suggestions.

Crash context:
- Function wrapper: {func_name}
- Exception: {_format_exception_summary(exc)}
- Stage1 context-window retries used: {context_retries}
- Stage2 context-window retries used: {handoff_retries}

Traceback (full, chronological):
```python
{full_traceback}
```

Crash envelope (frames + snippets + triage cues):
```text
{safe_crash_envelope}
```

Query payload sent to the local model (context-budgeted):
```text
{safe_query_payload}
```

Local AI forensic triage:
{triage_block}

Agent handoff plan:
- objective: {_as_text(getattr(handoff, "objective", "Fallback objective from triage"))}
- prioritized_steps: {handoff_steps}
- candidate_edits: {handoff_edits}
- instrumentation: {handoff_instrumentation}
- verification_commands: {handoff_verify}
- risks: {handoff_risks}

Agent handoff prompt:
{handoff_prompt}
"""


def _write_prompt_output(prompt_content: str, prompt_to: str, prompt_log_level: str | int) -> None:
    if prompt_to == "stdout":
        print(prompt_content)
        return
    if prompt_to == "stderr":
        print(prompt_content, file=sys.stderr)
        return
    if prompt_to == "log":
        _log_text(prompt_content, prompt_log_level, fallback=logging.INFO)
        return

    try:
        prompt_path = Path(prompt_to).expanduser()
        if prompt_path.name in {"", ".", ".."}:
            raise ValueError("prompt_to must point to a writable file path")
        prompt_path = prompt_path.with_suffix(".txt")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt_content, encoding="utf-8")
        print(f"Generated AI Agent Prompt written to: {prompt_path}\n")
    except (OSError, TypeError, ValueError):
        logger.warning(
            "[FMTools Debug] Could not write prompt_to path '%s'.",
            prompt_to,
            exc_info=True,
        )


async def _handle_exception(
    exc: Exception,
    func_name: str,
    summary_to: str | None = "stderr",
    prompt_to: str | None = "stdout",
    silenced: bool = False,
    summary_log_level: str | int = "error",
    prompt_log_level: str | int = "info",
):
    """Core logic to analyze the exception with the local FM."""
    if silenced:
        return

    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Print original traceback
    print(f"\n--- Exception caught in '{func_name}' ---", file=sys.stderr)
    print(tb_str, file=sys.stderr)
    print("--- End of standard traceback ---\n", file=sys.stderr)
    frames = _build_trace_frames(exc)
    frame_hints = _render_frame_hints(frames)
    logging_signals_list = _extract_logging_signals(tb_str)
    logging_signals = (
        "\n".join(f"- {item}" for item in logging_signals_list)
        if logging_signals_list
        else "- none detected"
    )
    source_snippets = _build_source_snippets(frames)
    crash_envelope = _build_crash_envelope(
        func_name=func_name,
        exc=exc,
        frame_hints=frame_hints,
        logging_signals=logging_signals,
        source_snippets=source_snippets,
        full_traceback=tb_str,
    )

    model = create_model()
    try:
        ensure_model_available(model, context="enhanced_debug")
    except AppleFMSetupError as setup_error:
        logger.error("%s", setup_error)
        return

    session = create_session(
        instructions=(
            "You are an expert Principal Software Engineer specializing in production debugging. "
            "Provide evidence-based diagnostics only. Do not fabricate files, line numbers, or logs. "
            "Call out uncertainty when evidence is missing. Prefer concrete path/line fix locations."
        ),
        model=model,
    )

    print("FMTools is analyzing the crash locally via Neural Engine...", file=sys.stderr)

    try:
        payload_candidates: list[tuple[str, int, int]] = []
        seen_payloads: set[str] = set()
        for ratio in _TRACEBACK_RETRY_RATIOS:
            candidate = _build_traceback_payload(tb_str, ratio)
            payload_text = candidate[0]
            if payload_text in seen_payloads:
                continue
            seen_payloads.add(payload_text)
            payload_candidates.append(candidate)
        if not payload_candidates:
            payload_candidates.append(_build_traceback_payload(tb_str, 1.0))

        query_payload = ""
        context_retries = 0
        analysis = None

        for attempt_index, (tb_payload, trimmed_lines, total_lines) in enumerate(
            payload_candidates, start=1
        ):
            query_payload = _compose_debug_query(
                func_name=func_name,
                exc=exc,
                traceback_payload=tb_payload,
                attempt=attempt_index,
                total_attempts=len(payload_candidates),
                trimmed_lines=trimmed_lines,
                total_lines=total_lines,
                frame_hints=frame_hints,
                logging_signals=logging_signals,
                source_snippets=source_snippets,
            )
            try:
                analysis = await session.respond(query_payload, generating=DebuggingAnalysis)
                break
            except Exception as respond_error:
                if _looks_like_context_overflow(respond_error) and attempt_index < len(
                    payload_candidates
                ):
                    context_retries += 1
                    print(
                        (
                            "[FMTools Debug] Context window overflow detected; "
                            "retrying with smaller traceback payload "
                            f"({attempt_index}/{len(payload_candidates) - 1})."
                        ),
                        file=sys.stderr,
                    )
                    continue
                raise

        if analysis is None:
            raise RuntimeError("No debugging analysis could be generated.")
        exception_summary = _format_exception_summary(exc)
        certainty_level = _as_text(getattr(analysis, "certainty_level", "MEDIUM")) or "MEDIUM"
        severity = _as_text(getattr(analysis, "severity", "MEDIUM")) or "MEDIUM"
        error_summary = _as_text(getattr(analysis, "error_summary", "No summary returned."))
        blast_radius = _as_text(getattr(analysis, "blast_radius", "Unknown scope."))
        possible_causes = _as_list(getattr(analysis, "possible_causes", []))
        evidence = _as_list(getattr(analysis, "evidence", []))
        likely_fix_locations = _as_list(getattr(analysis, "likely_fix_locations", []))
        if not likely_fix_locations:
            likely_fix_locations = _fallback_fix_locations(frames)
        first_actions = _as_list(getattr(analysis, "first_actions_30m", []))
        verification_steps = _as_list(getattr(analysis, "verification_steps", []))
        unknowns = _as_list(getattr(analysis, "unknowns", []))
        suggested_fix = _as_text(getattr(analysis, "suggested_fix", "No suggested fix returned."))

        output = [
            "\n" + "=" * 50,
            f"FMTools AI Debug Analysis (Certainty: {certainty_level}, Severity: {severity})",
            "=" * 50,
            f"Exception: {exception_summary}",
            f"Function: {func_name}",
            f"Context retries: {context_retries}\n",
            f"Summary: {error_summary}",
            f"Blast Radius: {blast_radius}\n",
            "Possible Causes:",
        ]
        for idx, cause in enumerate(possible_causes or ["No explicit causes returned."], 1):
            output.append(f"  {idx}. {cause}")
        output.append("\nLikely Fix Locations:")
        for idx, location in enumerate(likely_fix_locations or ["No fix locations returned."], 1):
            output.append(f"  {idx}. {location}")
        if evidence:
            output.append("\nEvidence:")
            for item in evidence:
                output.append(f"  - {item}")
        if first_actions:
            output.append("\nFirst Actions (30m):")
            for action in first_actions:
                output.append(f"  - {action}")
        if verification_steps:
            output.append("\nVerification Steps:")
            for step in verification_steps:
                output.append(f"  - {step}")
        if unknowns:
            output.append("\nUnknowns:")
            for item in unknowns:
                output.append(f"  - {item}")
        if logging_signals_list:
            output.append("\nLogging Signals:")
            for signal in logging_signals_list:
                output.append(f"  - {signal}")
        output.append(f"\nSuggested Fix: {suggested_fix}")
        output.append("=" * 50 + "\n")

        output_str = "\n".join(output)

        if summary_to == "stdout":
            print(output_str)
        elif summary_to == "stderr":
            print(output_str, file=sys.stderr)
        elif summary_to == "log":
            _log_text(output_str, summary_log_level, fallback=logging.ERROR)
        elif summary_to is not None:
            logger.warning(
                "[FMTools Debug] Unsupported summary_to value '%s'; expected 'stdout', 'stderr', 'log', or None.",
                summary_to,
            )

        if prompt_to is not None:
            handoff = None
            handoff_query = ""
            handoff_retries = 0
            handoff_candidates = _build_text_retry_candidates(crash_envelope)
            for attempt_index, envelope_payload in enumerate(handoff_candidates, start=1):
                handoff_query = _compose_handoff_query(
                    func_name=func_name,
                    exc=exc,
                    crash_envelope=envelope_payload,
                    analysis=analysis,
                    attempt=attempt_index,
                    total_attempts=len(handoff_candidates),
                )
                try:
                    handoff = await session.respond(handoff_query, generating=DebuggingHandoff)
                    break
                except Exception as handoff_error:
                    if _looks_like_context_overflow(handoff_error) and attempt_index < len(
                        handoff_candidates
                    ):
                        handoff_retries += 1
                        print(
                            (
                                "[FMTools Debug] Context window overflow during handoff generation; "
                                f"retrying with smaller payload ({attempt_index}/{len(handoff_candidates) - 1})."
                            ),
                            file=sys.stderr,
                        )
                        continue
                    logger.warning(
                        "[FMTools Debug] Handoff generation failed; using deterministic fallback.",
                        exc_info=True,
                    )
                    break

            prompt_content = _build_prompt_content(
                func_name=func_name,
                exc=exc,
                full_traceback=tb_str,
                query_payload=query_payload,
                crash_envelope=crash_envelope,
                analysis=analysis,
                handoff=handoff,
                context_retries=context_retries,
                handoff_retries=handoff_retries,
            )
            _write_prompt_output(prompt_content, prompt_to, prompt_log_level)

    except Exception as e:
        print(f"FMTools AI analysis failed: {e}", file=sys.stderr)
