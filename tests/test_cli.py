"""Tests for silicon_refinery.cli helper functions."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from silicon_refinery.cli import _discover_example_scripts, _run, _run_with_timeout, cli, cli_entry
from silicon_refinery.exceptions import AppleFMSetupError


def test_run_returns_subprocess_exit_code():
    proc = MagicMock(returncode=5)
    with patch("silicon_refinery.cli.subprocess.run", return_value=proc) as run:
        rc = _run(["uv", "--version"], cwd="/tmp")

    assert rc == 5
    run.assert_called_once_with(["uv", "--version"], cwd="/tmp", env=None)


def test_run_handles_missing_binary(capfd):
    with patch("silicon_refinery.cli.subprocess.run", side_effect=FileNotFoundError()):
        rc = _run(["missing-binary"])

    assert rc == 127
    captured = capfd.readouterr()
    assert "command not found: missing-binary" in captured.err


def test_cli_entry_handles_setup_error(capfd):
    with (
        patch("silicon_refinery.cli.cli", side_effect=AppleFMSetupError("setup failed")),
        pytest.raises(SystemExit) as exc_info,
    ):
        cli_entry()

    assert exc_info.value.code == 2
    captured = capfd.readouterr()
    assert "setup failed" in captured.err


def test_discover_example_scripts_filters_support_and_notebook(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    (examples / "_support.py").write_text("", encoding="utf-8")
    (examples / "examples_notebook.py").write_text("", encoding="utf-8")
    (examples / "a_demo.py").write_text("", encoding="utf-8")
    (examples / "b_demo.py").write_text("", encoding="utf-8")

    monkeypatch.setattr("silicon_refinery.cli._repo_root", lambda: str(tmp_path))
    discovered = _discover_example_scripts()

    assert [name for name, _ in discovered] == ["a_demo", "b_demo"]


def test_run_with_timeout_times_out(monkeypatch):
    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout", 1))

    monkeypatch.setattr("silicon_refinery.cli.subprocess.run", _raise_timeout)
    rc, timed_out, elapsed = _run_with_timeout(["python", "-V"], cwd="/tmp", timeout_s=0.01)

    assert rc == 124
    assert timed_out is True
    assert elapsed >= 0


def test_example_command_lists_examples(monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(
        "silicon_refinery.cli._discover_example_scripts",
        lambda: [("simple_inference", "/tmp/examples/simple_inference.py")],
    )

    result = runner.invoke(cli, ["example", "--list"])
    assert result.exit_code == 0
    assert "simple_inference" in result.output
