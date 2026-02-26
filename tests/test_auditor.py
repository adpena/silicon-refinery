"""
Tests for silicon_refinery.auditor — Code Auditor.

All tests mock apple_fm_sdk via the conftest.py module-level mock.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from silicon_refinery.auditor import (
    AuditIssue,
    AuditResult,
    audit_diff,
    audit_directory,
    audit_file,
    format_audit_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fm_response(
    issues: list[dict] | None = None,
    summary: str = "Looks good.",
    score: int = 85,
) -> str:
    """Build a JSON string that mimics the FM structured response."""
    payload = {
        "issues": issues or [],
        "summary": summary,
        "score": score,
    }
    return json.dumps(payload)


def _patch_fm(response_text: str):
    """Context manager that patches apple_fm_sdk so session.respond returns *response_text*."""
    mock_model = MagicMock()
    mock_session = MagicMock()
    mock_session.respond = AsyncMock(return_value=response_text)
    return (
        patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
        patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
    )


# ---------------------------------------------------------------------------
# AuditResult / AuditIssue dataclass fields
# ---------------------------------------------------------------------------


class TestDataclassFields:
    def test_audit_result_dataclass_fields(self):
        result = AuditResult(file_path="test.py", issues=[], summary="ok", score=100)
        assert result.file_path == "test.py"
        assert result.issues == []
        assert result.summary == "ok"
        assert result.score == 100

    def test_audit_result_defaults(self):
        result = AuditResult(file_path="x.py")
        assert result.issues == []
        assert result.summary == ""
        assert result.score == 0

    def test_audit_issue_severities(self):
        for sev in ("info", "warning", "error"):
            issue = AuditIssue(
                line=10, severity=sev, category="style", message="msg", suggestion="fix"
            )
            assert issue.severity == sev
            assert issue.line == 10
            assert issue.category == "style"
            assert issue.message == "msg"
            assert issue.suggestion == "fix"

    def test_audit_issue_line_can_be_none(self):
        issue = AuditIssue(
            line=None, severity="info", category="general", message="no line", suggestion=""
        )
        assert issue.line is None


# ---------------------------------------------------------------------------
# audit_file
# ---------------------------------------------------------------------------


class TestAuditFile:
    async def test_audit_file_returns_audit_result(self, tmp_path):
        py_file = tmp_path / "sample.py"
        py_file.write_text("print('hello')\n", encoding="utf-8")

        response = _make_fm_response(
            issues=[
                {
                    "line": 1,
                    "severity": "info",
                    "category": "style",
                    "message": "Consider using logging.",
                    "suggestion": "Use logging.info instead of print.",
                }
            ],
            summary="Simple script.",
            score=90,
        )

        p1, p2 = _patch_fm(response)
        with p1, p2:
            result = await audit_file(py_file)

        assert isinstance(result, AuditResult)
        assert result.score == 90
        assert result.summary == "Simple script."
        assert len(result.issues) == 1
        assert result.issues[0].severity == "info"
        assert result.issues[0].line == 1

    async def test_audit_file_reads_file_content(self, tmp_path):
        """Verify that the file's content is actually sent to the FM."""
        py_file = tmp_path / "code.py"
        py_file.write_text("x = 42\n", encoding="utf-8")

        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=_make_fm_response())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            await audit_file(py_file)

        # The prompt sent to respond should contain the file content
        call_args = mock_session.respond.call_args
        prompt = call_args[0][0]
        assert "x = 42" in prompt

    async def test_audit_file_custom_instructions(self, tmp_path):
        """Verify custom instructions are passed to the FM session."""
        py_file = tmp_path / "code.py"
        py_file.write_text("pass\n", encoding="utf-8")

        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=_make_fm_response())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session) as session_cls,
        ):
            await audit_file(py_file, instructions="Custom review prompt.")

        # LanguageModelSession should have been called with our custom instructions
        call_kwargs = session_cls.call_args
        assert call_kwargs[1]["instructions"] == "Custom review prompt."

    async def test_audit_file_handles_invalid_issues_and_score(self, tmp_path):
        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n", encoding="utf-8")

        response = json.dumps(
            {
                "issues": None,
                "summary": "Malformed payload",
                "score": "not-a-number",
            }
        )

        p1, p2 = _patch_fm(response)
        with p1, p2:
            result = await audit_file(py_file)

        assert result.summary == "Malformed payload"
        assert result.score == 0
        assert result.issues == []

    async def test_audit_file_reads_with_to_thread(self, tmp_path):
        py_file = tmp_path / "code.py"
        py_file.write_text("x = 1\n", encoding="utf-8")

        response = _make_fm_response()
        p1, p2 = _patch_fm(response)
        with (
            p1,
            p2,
            patch(
                "silicon_refinery.auditor.asyncio.to_thread",
                AsyncMock(return_value="x = 1\n"),
            ) as to_thread,
        ):
            await audit_file(py_file)

        assert to_thread.await_count == 1
        args, kwargs = to_thread.call_args
        assert callable(args[0])
        assert getattr(args[0], "__name__", "") == "read_text"
        assert kwargs["encoding"] == "utf-8"


# ---------------------------------------------------------------------------
# audit_directory
# ---------------------------------------------------------------------------


class TestAuditDirectory:
    async def test_audit_directory_finds_python_files(self, tmp_path):
        # Create a small directory structure
        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("b = 2\n", encoding="utf-8")
        (tmp_path / "readme.txt").write_text("not python\n", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.py").write_text("c = 3\n", encoding="utf-8")

        response = _make_fm_response(summary="ok", score=80)
        p1, p2 = _patch_fm(response)
        with p1, p2:
            results = await audit_directory(tmp_path)

        # Should find 3 .py files, not the .txt
        assert len(results) == 3
        paths = {r.file_path for r in results}
        assert any("a.py" in p for p in paths)
        assert any("b.py" in p for p in paths)
        assert any("c.py" in p for p in paths)

    async def test_audit_directory_custom_patterns(self, tmp_path):
        (tmp_path / "a.py").write_text("a = 1\n", encoding="utf-8")
        (tmp_path / "b.txt").write_text("text\n", encoding="utf-8")

        response = _make_fm_response()
        p1, p2 = _patch_fm(response)
        with p1, p2:
            results = await audit_directory(tmp_path, patterns=["*.txt"])

        assert len(results) == 1
        assert "b.txt" in results[0].file_path

    async def test_audit_directory_ignores_non_files(self, tmp_path):
        (tmp_path / "real.py").write_text("print('ok')\n", encoding="utf-8")
        (tmp_path / "not_a_file.py").mkdir()

        response = _make_fm_response()
        p1, p2 = _patch_fm(response)
        with p1, p2:
            results = await audit_directory(tmp_path)

        assert len(results) == 1
        assert results[0].file_path.endswith("real.py")

    async def test_audit_directory_continues_when_file_audit_fails(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.write_text("a = 1\n", encoding="utf-8")
        file_b.write_text("b = 2\n", encoding="utf-8")

        good_result = AuditResult(file_path=str(file_b), summary="ok", score=90, issues=[])
        patched = AsyncMock(side_effect=[RuntimeError("boom"), good_result])

        with patch("silicon_refinery.auditor.audit_file", patched):
            results = await audit_directory(tmp_path)

        assert len(results) == 2
        failed = next(r for r in results if r.file_path.endswith("a.py"))
        succeeded = next(r for r in results if r.file_path.endswith("b.py"))
        assert failed.score == 0
        assert "Audit failed" in failed.summary
        assert succeeded.score == 90

    async def test_audit_directory_validates_max_concurrency(self, tmp_path):
        with pytest.raises(ValueError, match=r"max_concurrency must be >= 1"):
            await audit_directory(tmp_path, max_concurrency=0)

    async def test_audit_directory_respects_max_concurrency(self, tmp_path):
        for idx in range(6):
            (tmp_path / f"f{idx}.py").write_text(f"x = {idx}\n", encoding="utf-8")

        active = 0
        peak = 0
        lock = asyncio.Lock()

        async def fake_audit(file_path):
            nonlocal active, peak
            async with lock:
                active += 1
                peak = max(peak, active)
            await asyncio.sleep(0.01)
            async with lock:
                active -= 1
            return AuditResult(file_path=str(file_path), issues=[], summary="ok", score=100)

        with patch("silicon_refinery.auditor.audit_file", side_effect=fake_audit):
            results = await audit_directory(tmp_path, max_concurrency=2)

        assert len(results) == 6
        assert peak == 2

    async def test_audit_directory_preserves_sorted_output_order(self, tmp_path):
        paths = [tmp_path / "c.py", tmp_path / "a.py", tmp_path / "b.py"]
        for idx, p in enumerate(paths):
            p.write_text(f"x = {idx}\n", encoding="utf-8")

        delays = {"a.py": 0.03, "b.py": 0.01, "c.py": 0.02}

        async def fake_audit(file_path):
            await asyncio.sleep(delays[file_path.name])
            return AuditResult(
                file_path=str(file_path), issues=[], summary=file_path.name, score=100
            )

        with patch("silicon_refinery.auditor.audit_file", side_effect=fake_audit):
            results = await audit_directory(tmp_path, max_concurrency=3)

        assert [Path(r.file_path).name for r in results] == ["a.py", "b.py", "c.py"]


# ---------------------------------------------------------------------------
# audit_diff
# ---------------------------------------------------------------------------


class TestAuditDiff:
    async def test_audit_diff_processes_diff_text(self):
        diff = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 import os
+import subprocess
 def run():
-    os.system("ls")
+    subprocess.run(["ls"])
"""
        response = _make_fm_response(
            issues=[
                {
                    "line": 2,
                    "severity": "warning",
                    "category": "security",
                    "message": "subprocess usage needs input validation.",
                    "suggestion": "Validate inputs before passing to subprocess.run.",
                }
            ],
            summary="Improved from os.system to subprocess.",
            score=75,
        )
        p1, p2 = _patch_fm(response)
        with p1, p2:
            result = await audit_diff(diff)

        assert isinstance(result, AuditResult)
        assert result.file_path == "<diff>"
        assert result.score == 75
        assert len(result.issues) == 1
        assert result.issues[0].category == "security"

    async def test_audit_diff_sends_diff_to_fm(self):
        """The diff text should appear in the prompt sent to the FM."""
        diff = "+x = 1"
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=_make_fm_response())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            await audit_diff(diff)

        prompt = mock_session.respond.call_args[0][0]
        assert "+x = 1" in prompt

    async def test_audit_diff_truncates_very_large_inputs(self):
        diff = "+" + ("a" * 130_000)
        mock_model = MagicMock()
        mock_session = MagicMock()
        mock_session.respond = AsyncMock(return_value=_make_fm_response())

        with (
            patch("apple_fm_sdk.SystemLanguageModel", return_value=mock_model),
            patch("apple_fm_sdk.LanguageModelSession", return_value=mock_session),
        ):
            await audit_diff(diff)

        prompt = mock_session.respond.call_args[0][0]
        assert "[truncated:" in prompt


# ---------------------------------------------------------------------------
# format_audit_report
# ---------------------------------------------------------------------------


class TestFormatAuditReport:
    def test_format_audit_report_output(self):
        results = [
            AuditResult(
                file_path="app.py",
                issues=[
                    AuditIssue(
                        line=10,
                        severity="error",
                        category="security",
                        message="SQL injection risk.",
                        suggestion="Use parameterized queries.",
                    ),
                    AuditIssue(
                        line=None,
                        severity="info",
                        category="style",
                        message="Missing docstrings.",
                        suggestion="Add module docstring.",
                    ),
                ],
                summary="Needs security fixes.",
                score=55,
            ),
            AuditResult(
                file_path="utils.py",
                issues=[],
                summary="Clean code.",
                score=95,
            ),
        ]

        report = format_audit_report(results)

        assert "SiliconRefinery Code Audit Report" in report
        assert "Files audited: 2" in report
        assert "Total issues:  2" in report
        assert "app.py" in report
        assert "utils.py" in report
        assert "SQL injection risk." in report
        assert "No issues found." in report
        assert "55/100" in report
        assert "95/100" in report
        # Average score
        assert "75/100" in report

    def test_format_audit_report_empty(self):
        report = format_audit_report([])
        assert "Files audited: 0" in report
        assert "Total issues:  0" in report

    def test_format_audit_report_severity_formatting(self):
        results = [
            AuditResult(
                file_path="test.py",
                issues=[
                    AuditIssue(
                        line=5,
                        severity="warning",
                        category="performance",
                        message="Slow loop.",
                        suggestion="Use list comprehension.",
                    ),
                ],
                summary="Perf issue.",
                score=70,
            ),
        ]
        report = format_audit_report(results)
        assert "WARNING" in report
        assert "L5" in report
        assert "performance" in report
        assert "Suggestion:" in report
