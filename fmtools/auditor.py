"""
Code Auditor for FMTools.

An on-device code review tool that uses the Foundation Model to analyse Python
source files. It reads code, sends it to the FM with structured audit
instructions, and returns actionable results.
"""

from __future__ import annotations

import asyncio
import glob as glob_mod
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .protocols import create_model, create_session

logger = logging.getLogger("fmtools")

__all__ = [
    "AuditIssue",
    "AuditResult",
    "audit_diff",
    "audit_directory",
    "audit_file",
    "format_audit_report",
]

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

_DEFAULT_INSTRUCTIONS = (
    "You are an expert Python code reviewer. Analyse the provided code and return a JSON "
    "object with the following keys:\n"
    '  - "issues": a list of objects each with keys "line" (int or null), '
    '"severity" ("info"|"warning"|"error"), "category" (e.g. "security", "performance", '
    '"style", "correctness"), "message" (str), "suggestion" (str).\n'
    '  - "summary": a brief overall summary of the code quality.\n'
    '  - "score": an integer from 0 to 100 representing overall quality.\n'
    "Focus on: security vulnerabilities, performance issues, correctness bugs, "
    "and Pythonic style."
)
_MAX_PROMPT_CHARS = 120_000


@dataclass
class AuditIssue:
    """A single issue discovered during an audit."""

    line: int | None
    severity: Literal["info", "warning", "error"]
    category: str
    message: str
    suggestion: str


@dataclass
class AuditResult:
    """Full audit result for a single file or diff."""

    file_path: str
    issues: list[AuditIssue] = field(default_factory=list)
    summary: str = ""
    score: int = 0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_fm_response(response: Any, file_path: str) -> AuditResult:
    """Parse the Foundation Model response into an AuditResult.

    The FM may return a string (JSON) or an object with attributes.
    """
    raw: dict[str, Any] = {}

    if isinstance(response, str):
        try:
            raw = json.loads(response)
        except json.JSONDecodeError:
            return AuditResult(
                file_path=file_path,
                summary=response[:500],
                score=0,
            )
    elif isinstance(response, dict):
        raw = response
    else:
        # Attempt to read attributes (generable object)
        try:
            raw = vars(response)
        except TypeError:
            raw = {"summary": str(response)}

    issues_raw = raw.get("issues", [])
    if not isinstance(issues_raw, list):
        issues_raw = []

    valid_severity = {"info", "warning", "error"}
    issues: list[AuditIssue] = []
    for item in issues_raw:
        if isinstance(item, dict):
            sev = item.get("severity", "info")
            severity = sev if sev in valid_severity else "info"
            line = item.get("line")
            issues.append(
                AuditIssue(
                    line=line if isinstance(line, int) else None,
                    severity=severity,
                    category=item.get("category", "general"),
                    message=item.get("message", ""),
                    suggestion=item.get("suggestion", ""),
                )
            )
        elif hasattr(item, "severity"):
            raw_severity = getattr(item, "severity", "info")
            severity = raw_severity if raw_severity in valid_severity else "info"
            raw_line = getattr(item, "line", None)
            issues.append(
                AuditIssue(
                    line=raw_line if isinstance(raw_line, int) else None,
                    severity=severity,
                    category=getattr(item, "category", "general"),
                    message=getattr(item, "message", ""),
                    suggestion=getattr(item, "suggestion", ""),
                )
            )

    score_raw = raw.get("score", 0)
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        score = 0

    return AuditResult(
        file_path=file_path,
        issues=issues,
        summary=raw.get("summary", ""),
        score=score,
    )


def _truncate_for_prompt(text: str, *, max_chars: int = _MAX_PROMPT_CHARS) -> str:
    """Trim very large prompts to avoid context-window blowups."""
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    return f"{clipped}\n\n[truncated: showing first {max_chars} of {len(text)} characters]"


def _failed_audit_result(file_path: Path, exc: Exception) -> AuditResult:
    """Build a deterministic fallback result when per-file auditing fails."""
    return AuditResult(
        file_path=str(file_path),
        issues=[
            AuditIssue(
                line=None,
                severity="error",
                category="runtime",
                message=f"Audit failed: {exc}",
                suggestion="Inspect file and retry audit.",
            )
        ],
        summary=f"Audit failed for {file_path.name}",
        score=0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def audit_file(
    path: str | Path,
    instructions: str | None = None,
) -> AuditResult:
    """Audit a single source file using the Foundation Model.

    Args:
        path: Path to the Python file.
        instructions: Custom audit instructions. Defaults to a comprehensive
            code review prompt covering security, performance, correctness, and
            style.

    Returns:
        An ``AuditResult`` containing the issues, summary, and quality score.
    """
    file_path = Path(path)
    # Keep file I/O off the event loop so async callers remain responsive.
    content = await asyncio.to_thread(file_path.read_text, encoding="utf-8")
    prompt = f"File: {file_path.name}\n\n```python\n{_truncate_for_prompt(content)}\n```"

    model = create_model()
    session = create_session(
        instructions=instructions or _DEFAULT_INSTRUCTIONS,
        model=model,
    )
    response = await session.respond(prompt)
    return _parse_fm_response(response, str(file_path))


async def audit_directory(
    path: str | Path,
    patterns: list[str] | None = None,
    *,
    max_concurrency: int = 8,
) -> list[AuditResult]:
    """Audit all matching files in a directory.

    Args:
        path: Root directory to search.
        patterns: Glob patterns for files to audit (default: ``["*.py"]``).
        max_concurrency: Maximum number of files to audit concurrently.

    Returns:
        List of ``AuditResult`` objects, one per file.
    """
    if patterns is None:
        patterns = ["*.py"]
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    root = Path(path)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(Path(p) for p in glob_mod.glob(str(root / "**" / pattern), recursive=True))

    # Deduplicate and sort for deterministic output
    files = sorted(file_path for file_path in set(files) if file_path.is_file())
    if not files:
        return []

    async def _audit_file_safe(file_path: Path) -> AuditResult:
        try:
            return await audit_file(file_path)
        except Exception as exc:
            logger.warning("[audit_directory] Audit failed for %s: %s", file_path, exc)
            return _failed_audit_result(file_path, exc)

    # Bounded worker pool: preserves deterministic output order while keeping
    # workers busy even when per-file runtimes are uneven.
    queue: asyncio.Queue[tuple[int, Path] | None] = asyncio.Queue()
    for idx, file_path in enumerate(files):
        queue.put_nowait((idx, file_path))

    worker_count = min(max_concurrency, len(files))
    ordered_results: list[AuditResult | None] = [None] * len(files)

    async def _worker() -> None:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                return
            idx, file_path = item
            ordered_results[idx] = await _audit_file_safe(file_path)
            queue.task_done()

    workers = [asyncio.create_task(_worker()) for _ in range(worker_count)]
    await queue.join()
    for _ in workers:
        queue.put_nowait(None)
    await asyncio.gather(*workers)

    return [result for result in ordered_results if result is not None]


async def audit_diff(
    diff_text: str,
    instructions: str | None = None,
) -> AuditResult:
    """Audit a git diff string (for PR reviews).

    Args:
        diff_text: The raw ``git diff`` output.
        instructions: Custom audit instructions.

    Returns:
        An ``AuditResult`` for the diff.
    """
    diff_instructions = (instructions or _DEFAULT_INSTRUCTIONS) + (
        "\n\nYou are reviewing a git diff. Focus on the changed lines (prefixed with + or -)."
    )

    prompt = f"Git diff:\n\n```diff\n{_truncate_for_prompt(diff_text)}\n```"

    model = create_model()
    session = create_session(instructions=diff_instructions, model=model)
    response = await session.respond(prompt)
    return _parse_fm_response(response, "<diff>")


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def format_audit_report(results: list[AuditResult]) -> str:
    """Format audit results as a human-readable text report.

    Args:
        results: List of ``AuditResult`` objects.

    Returns:
        A formatted multi-line string report.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  FMTools Code Audit Report")
    lines.append("=" * 72)
    lines.append("")

    total_issues = sum(len(r.issues) for r in results)
    lines.append(f"Files audited: {len(results)}")
    lines.append(f"Total issues:  {total_issues}")

    if results:
        avg_score = sum(r.score for r in results) / len(results)
        lines.append(f"Average score: {avg_score:.0f}/100")
    lines.append("")

    for result in results:
        lines.append("-" * 72)
        lines.append(f"File:  {result.file_path}")
        lines.append(f"Score: {result.score}/100")
        lines.append(f"Summary: {result.summary}")
        lines.append("")

        if not result.issues:
            lines.append("  No issues found.")
        else:
            for issue in result.issues:
                loc = f"L{issue.line}" if issue.line is not None else "---"
                lines.append(
                    f"  [{issue.severity.upper():7s}] {loc:>5s} | {issue.category}: {issue.message}"
                )
                if issue.suggestion:
                    lines.append(f"           Suggestion: {issue.suggestion}")

        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
