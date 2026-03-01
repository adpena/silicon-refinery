"""
FMTools CLI — unified development and diagnostic commands.

Registered as `fmtools` console script via pyproject.toml.
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

import click

from .exceptions import AppleFMSetupError, require_apple_fm


def _repo_root() -> str:
    """Walk up from this file to find the repository root (contains pyproject.toml)."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


def _run(args: list[str], cwd: str | None = None, env: dict[str, str] | None = None) -> int:
    """Run a command, return exit code."""
    try:
        return subprocess.run(args, cwd=cwd, env=env).returncode
    except FileNotFoundError:
        click.secho(f"Error: command not found: {args[0]}", fg="red", err=True)
        return 127


def _run_quiet(
    args: list[str], cwd: str | None = None, env: dict[str, str] | None = None
) -> tuple[int, str]:
    """Run command quietly for probes and return ``(rc, diagnostics)``."""
    try:
        result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True)
    except FileNotFoundError:
        return 127, f"command not found: {args[0]}"
    diagnostics = (result.stderr or result.stdout or "").strip()
    return result.returncode, diagnostics


def _run_or_exit(args: list[str], cwd: str | None = None, label: str = "") -> None:
    """Run a command. On failure, print an error and exit."""
    rc = _run(args, cwd=cwd)
    if rc != 0:
        if label:
            click.secho(f"\n{label} failed (exit code {rc})", fg="red", err=True)
        raise SystemExit(rc)


def _run_with_timeout(args: list[str], cwd: str, timeout_s: float) -> tuple[int, bool, float]:
    """Run a command with a timeout.

    Returns ``(exit_code, timed_out, elapsed_seconds)``.
    """
    start = time.perf_counter()
    try:
        rc = subprocess.run(args, cwd=cwd, timeout=timeout_s).returncode
        return rc, False, time.perf_counter() - start
    except subprocess.TimeoutExpired:
        return 124, True, time.perf_counter() - start


def _missing_python_modules(
    modules: list[str], *, project_dir: str | None = None, uv_args: list[str] | None = None
) -> tuple[list[str], str]:
    """Return missing importable modules in a uv environment and diagnostic stderr."""
    if not modules:
        return [], ""

    base = ["uv", "run"]
    if project_dir:
        base.extend(["--project", project_dir, "--directory", project_dir])
    if uv_args:
        base.extend(uv_args)
    probe = [
        *base,
        "python",
        "-c",
        (
            "import importlib.util, json, sys; "
            "mods=sys.argv[1:]; "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            "print(json.dumps(missing))"
        ),
        *modules,
    ]
    try:
        result = subprocess.run(probe, cwd=_repo_root(), capture_output=True, text=True)
    except FileNotFoundError:
        return modules, "uv not found"

    if result.returncode != 0:
        return modules, (result.stderr or result.stdout).strip()

    try:
        parsed = json.loads((result.stdout or "").strip() or "[]")
    except json.JSONDecodeError:
        return modules, (result.stderr or result.stdout).strip()
    if not isinstance(parsed, list):
        return modules, (result.stderr or result.stdout).strip()
    missing = [str(item) for item in parsed]
    return missing, (result.stderr or "").strip()


def _select_chat_uv_runtime(project_dir: str, root: str) -> tuple[list[str], list[str], str]:
    """Prefer free-threaded CPython for chat; fall back to standard-gil when unavailable."""
    base = ["uv", "run", "--project", project_dir, "--directory", project_dir]
    probe_code = (
        "import sys; "
        "probe=getattr(sys, '_is_gil_enabled', None); "
        "raise SystemExit(0 if callable(probe) and not probe() else 1)"
    )
    for candidate in ("3.14t", "3.13t"):
        uv_args = ["--python", candidate]
        rc, _ = _run_quiet([*base, *uv_args, "python", "-c", probe_code], cwd=root)
        if rc == 0:
            return [*base, *uv_args], uv_args, f"free-threaded ({candidate})"
    return base, [], "standard-gil"


def _select_standard_gil_uv_args(project_dir: str, root: str) -> tuple[list[str], str]:
    """Pick an explicit standard-gil CPython runtime for reliable fallback."""
    base = ["uv", "run", "--project", project_dir, "--directory", project_dir]
    probe_code = (
        "import sys; "
        "probe=getattr(sys, '_is_gil_enabled', None); "
        "raise SystemExit(0 if (not callable(probe)) or probe() else 1)"
    )
    for candidate in ("3.14", "3.13"):
        uv_args = ["--python", candidate]
        rc, _ = _run_quiet([*base, *uv_args, "python", "-c", probe_code], cwd=root)
        if rc == 0:
            return uv_args, f"standard-gil ({candidate})"
    return [], "standard-gil (project default)"


def _fail_missing_dependencies(
    *,
    command_name: str,
    missing: list[str],
    install_steps: list[str],
    details: str = "",
) -> None:
    """Exit with detailed, actionable dependency guidance."""
    if not missing:
        return
    click.secho(
        f"{command_name} requires optional dependencies that are missing:",
        fg="red",
        err=True,
        bold=True,
    )
    for module in missing:
        click.echo(f"  - {module}", err=True)
    if details:
        click.echo("", err=True)
        click.secho("Diagnostics:", fg="yellow", err=True)
        click.echo(f"  {details}", err=True)
    click.echo("", err=True)
    click.secho("Install with:", fg="cyan", err=True)
    for step in install_steps:
        click.echo(f"  {step}", err=True)
    raise SystemExit(2)


def _discover_example_scripts() -> list[tuple[str, str]]:
    """Discover runnable top-level example scripts under ``examples/``."""
    root = Path(_repo_root())
    examples_dir = root / "examples"
    if not examples_dir.is_dir():
        return []

    scripts: list[tuple[str, str]] = []
    for path in sorted(examples_dir.glob("*.py")):
        if path.name in {"_support.py", "examples_notebook.py"}:
            continue
        scripts.append((path.stem, str(path)))
    return scripts


def _print_example_table(examples: list[tuple[str, str]]) -> None:
    """Pretty-print available example scripts."""
    if not examples:
        click.secho("No examples found in examples/ directory.", fg="yellow", err=True)
        return

    click.secho("\nAvailable examples:\n", fg="cyan", bold=True)
    click.secho(f"  {'Name':<28}{'Command'}", fg="cyan")
    click.secho(f"  {'─' * 27} {'─' * 42}", fg="cyan")
    for name, _path in examples:
        click.echo(f"  {name:<28}fmtools example {name}")
    click.echo()
    click.echo("  Notebook:")
    click.echo("    fmtools notebook")
    click.echo()
    click.echo("  Full smoke run:")
    click.echo("    fmtools smoke")
    click.echo()


def _run_script(name: str, extra_args: tuple[str, ...] = ()) -> None:
    """Run a shell script from the scripts/ directory."""
    root = _repo_root()
    script = os.path.join(root, "scripts", name)
    if not os.path.isfile(script):
        click.secho(f"Error: script not found: {script}", fg="red", err=True)
        raise SystemExit(1)
    rc = _run(["bash", script, *extra_args], cwd=root)
    raise SystemExit(rc)


# ── Main group ────────────────────────────────────────────────────────────────


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="fmtools")
def cli() -> None:
    """FMTools — development & diagnostic CLI for Apple Silicon ETL."""


# ── Setup & Doctor ────────────────────────────────────────────────────────────


@cli.command()
@click.option("--no-sdk", is_flag=True, help="Skip Apple FM SDK clone/install.")
@click.argument("extra_args", nargs=-1, type=click.UNPROCESSED)
def setup(no_sdk: bool, extra_args: tuple[str, ...]) -> None:
    """Run first-time development environment setup."""
    args = list(extra_args)
    if no_sdk:
        args.append("--no-sdk")
    _run_script("setup.sh", tuple(args))


@cli.command(name="install-homebrew")
@click.argument("tap_name", required=False)
def install_homebrew(tap_name: str | None) -> None:
    """Install/update FMTools via a local Homebrew tap."""
    args: tuple[str, ...] = (tap_name,) if tap_name else ()
    _run_script("install_homebrew.sh", args)


@cli.command()
def doctor() -> None:
    """Check all system prerequisites (macOS, Python, uv, ruff, ty, Neural Engine)."""
    _run_script("doctor.sh")


# ── Linting & Formatting ─────────────────────────────────────────────────────


@cli.command()
@click.option("--fix", is_flag=True, help="Auto-fix lint violations where possible.")
@click.argument("paths", nargs=-1)
def lint(fix: bool, paths: tuple[str, ...]) -> None:
    """Run ruff linter on the project."""
    root = _repo_root()
    targets = list(paths) if paths else ["."]
    cmd = ["uv", "run", "ruff", "check"]
    if fix:
        cmd.append("--fix")
    cmd.extend(targets)
    rc = _run(cmd, cwd=root)
    raise SystemExit(rc)


@cli.command(name="format")
@click.option(
    "--check", "check_only", is_flag=True, help="Check formatting without modifying files."
)
@click.argument("paths", nargs=-1)
def format_cmd(check_only: bool, paths: tuple[str, ...]) -> None:
    """Run ruff formatter on the project."""
    root = _repo_root()
    targets = list(paths) if paths else ["."]
    cmd = ["uv", "run", "ruff", "format"]
    if check_only:
        cmd.append("--check")
    cmd.extend(targets)
    env = dict(os.environ)
    # Prevent nested `uv run` warnings when outer shells export a different VIRTUAL_ENV.
    env.pop("VIRTUAL_ENV", None)
    rc = _run(cmd, cwd=root, env=env)
    raise SystemExit(rc)


# ── Type Checking ─────────────────────────────────────────────────────────────


@cli.command()
@click.argument("paths", nargs=-1)
def typecheck(paths: tuple[str, ...]) -> None:
    """Run ty type checker (defaults to fmtools/)."""
    root = _repo_root()
    targets = list(paths) if paths else ["fmtools/"]
    rc = _run(["uv", "run", "ty", "check", *targets], cwd=root)
    raise SystemExit(rc)


# ── Testing ───────────────────────────────────────────────────────────────────


@cli.command()
@click.option(
    "-k", "expression", default=None, help="Only run tests matching the given expression."
)
@click.option("--no-verbose", is_flag=True, help="Suppress verbose output.")
@click.argument("paths", nargs=-1)
def test(expression: str | None, no_verbose: bool, paths: tuple[str, ...]) -> None:
    """Run the full test suite via pytest."""
    root = _repo_root()
    cmd = ["uv", "run", "pytest"]
    cmd.extend(list(paths) if paths else ["tests/"])
    if not no_verbose:
        cmd.append("-v")
    if expression:
        cmd.extend(["-k", expression])
    rc = _run(cmd, cwd=root)
    raise SystemExit(rc)


# ── Full CI Pipeline ─────────────────────────────────────────────────────────


@cli.command()
def check() -> None:
    """Run the full CI pipeline: lint + format check + typecheck + tests."""
    root = _repo_root()
    steps = [
        ("Lint", ["uv", "run", "ruff", "check", "."]),
        ("Format check", ["uv", "run", "ruff", "format", "--check", "."]),
        ("Type check", ["uv", "run", "ty", "check", "fmtools/"]),
        ("Tests", ["uv", "run", "pytest", "tests/", "-v"]),
    ]
    for label, cmd in steps:
        click.echo()
        click.secho(f"{'=' * 50}", fg="cyan")
        click.secho(f"  {label}", fg="cyan", bold=True)
        click.secho(f"{'=' * 50}", fg="cyan")
        click.echo()
        _run_or_exit(cmd, cwd=root, label=label)

    click.echo()
    click.secho(f"{'=' * 50}", fg="green")
    click.secho("  All checks passed!", fg="green", bold=True)
    click.secho(f"{'=' * 50}", fg="green")
    click.echo()


# ── Run Use Cases ─────────────────────────────────────────────────────────────


def _discover_use_cases() -> list[tuple[str, str, str]]:
    """Find all numbered use-case directories under ``use_cases/``.

    Returns a sorted list of ``(number, dir_name, full_path)`` tuples for
    directories matching the pattern ``NN_*`` that contain an ``example.py``.
    """
    root = _repo_root()
    use_cases_dir = os.path.join(root, "use_cases")
    if not os.path.isdir(use_cases_dir):
        return []

    results: list[tuple[str, str, str]] = []
    for entry in sorted(os.listdir(use_cases_dir)):
        match = re.match(r"^(\d{2})_", entry)
        if not match:
            continue
        full_path = os.path.join(use_cases_dir, entry)
        if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "example.py")):
            results.append((match.group(1), entry, full_path))
    return results


def _print_use_case_table(use_cases: list[tuple[str, str, str]]) -> None:
    """Pretty-print the available use cases as a table."""
    if not use_cases:
        click.secho("No use cases found in use_cases/ directory.", fg="yellow", err=True)
        return

    click.secho("\nAvailable use cases:\n", fg="cyan", bold=True)
    click.secho(f"  {'#':<6}{'Name'}", fg="cyan")
    click.secho(f"  {'─' * 5} {'─' * 45}", fg="cyan")
    for number, dir_name, _ in use_cases:
        # Strip the number prefix to show a cleaner name
        label = dir_name[3:].replace("_", " ").title()
        click.echo(f"  {number:<6}{label}")
    click.echo()
    click.echo("  Run with: fmtools run <number>")
    click.echo()


@cli.command()
@click.option("--list", "list_all", is_flag=True, help="List all available use cases.")
@click.argument("number", required=False)
def run(list_all: bool, number: str | None) -> None:
    """Run a numbered use-case example.

    \b
    Examples:
        fmtools run 01        # Pipeline operators
        fmtools run 07        # Stress test throughput
        fmtools run --list    # Show all available use cases
    """
    use_cases = _discover_use_cases()

    if list_all or number is None:
        _print_use_case_table(use_cases)
        raise SystemExit(0)

    # Normalise to zero-padded two-digit string
    number = number.zfill(2)

    # Find the matching use case
    matched = [uc for uc in use_cases if uc[0] == number]
    if not matched:
        click.secho(f"Error: no use case found for number '{number}'.", fg="red", err=True)
        _print_use_case_table(use_cases)
        raise SystemExit(1)

    _, dir_name, full_path = matched[0]
    example_script = os.path.join(full_path, "example.py")

    click.secho(f"\nRunning use case {number}: {dir_name}\n", fg="cyan", bold=True)
    rc = _run(["uv", "run", "python", example_script], cwd=_repo_root())
    raise SystemExit(rc)


# ── Example Catalog ────────────────────────────────────────────────────────────


@cli.command(name="example")
@click.option("--list", "list_all", is_flag=True, help="List all available example scripts.")
@click.argument("name", required=False)
def example_cmd(list_all: bool, name: str | None) -> None:
    """Run a standalone script from ``examples/``.

    \b
    Examples:
        fmtools example --list
        fmtools example simple_inference
        fmtools example code_auditor
    """
    examples = _discover_example_scripts()
    if list_all or name is None:
        _print_example_table(examples)
        raise SystemExit(0)

    # Normalise to stem form (accept either "foo" or "foo.py")
    normalized = name[:-3] if name.endswith(".py") else name
    matched = [item for item in examples if item[0] == normalized]
    if not matched:
        click.secho(f"Error: unknown example '{name}'.", fg="red", err=True)
        _print_example_table(examples)
        raise SystemExit(1)

    require_apple_fm(f"fmtools example {normalized}")
    _, script_path = matched[0]
    click.secho(f"\nRunning example: {normalized}\n", fg="cyan", bold=True)
    rc = _run(["uv", "run", "python", script_path], cwd=_repo_root())
    raise SystemExit(rc)


@cli.command()
@click.option(
    "--timeout",
    "script_timeout",
    default=180.0,
    show_default=True,
    type=float,
    help="Per-script timeout in seconds for each examples/*.py run.",
)
@click.option(
    "--include-notebook/--no-include-notebook",
    default=True,
    show_default=True,
    help="Also startup-check the marimo notebook in headless mode.",
)
@click.option(
    "--notebook-timeout",
    default=20.0,
    show_default=True,
    type=float,
    help="Notebook startup timeout in seconds.",
)
def smoke(script_timeout: float, include_notebook: bool, notebook_timeout: float) -> None:
    """Smoke-test the example surface on local Apple SDK + model runtime."""
    if script_timeout <= 0:
        raise click.BadParameter("--timeout must be > 0")
    if notebook_timeout <= 0:
        raise click.BadParameter("--notebook-timeout must be > 0")

    require_apple_fm("fmtools smoke")
    root = _repo_root()
    examples = _discover_example_scripts()
    if not examples:
        click.secho("No example scripts were discovered.", fg="yellow", err=True)
        raise SystemExit(1)

    click.secho(
        f"\nRunning smoke suite ({len(examples)} scripts, timeout={script_timeout:.0f}s)...\n",
        fg="cyan",
        bold=True,
    )
    failures = 0
    for name, script_path in examples:
        click.echo(f"[script] {name} ...")
        rc, timed_out, elapsed = _run_with_timeout(
            ["uv", "run", "python", script_path],
            cwd=root,
            timeout_s=script_timeout,
        )
        if timed_out:
            click.secho(f"  ✗ TIMEOUT after {elapsed:.1f}s", fg="red", err=True)
            failures += 1
            continue
        if rc != 0:
            click.secho(f"  ✗ FAIL (exit code {rc}, {elapsed:.1f}s)", fg="red", err=True)
            failures += 1
            continue
        click.secho(f"  ✓ PASS ({elapsed:.1f}s)", fg="green")

    if include_notebook:
        missing, details = _missing_python_modules(["marimo"])
        _fail_missing_dependencies(
            command_name="fmtools smoke --include-notebook",
            missing=missing,
            details=details,
            install_steps=[
                "uv pip install marimo",
                "uv sync --all-groups",
            ],
        )
        notebook = os.path.join(root, "examples", "examples_notebook.py")
        click.echo("[notebook] examples_notebook.py startup ...")
        rc, timed_out, elapsed = _run_with_timeout(
            ["uv", "run", "marimo", "run", notebook, "--headless", "--no-token"],
            cwd=root,
            timeout_s=notebook_timeout,
        )
        if timed_out:
            click.secho(f"  ✓ STARTED ({elapsed:.1f}s, timeout reached as expected)", fg="green")
        elif rc == 0:
            click.secho(f"  ✓ PASS (exited cleanly in {elapsed:.1f}s)", fg="green")
        else:
            click.secho(f"  ✗ FAIL (exit code {rc}, {elapsed:.1f}s)", fg="red", err=True)
            failures += 1

    click.echo()
    if failures:
        click.secho(f"Smoke suite completed with {failures} failure(s).", fg="red", bold=True)
        raise SystemExit(1)
    click.secho("Smoke suite passed.", fg="green", bold=True)


@cli.command()
@click.option("--headless", is_flag=True, help="Run marimo notebook without opening a browser.")
@click.option("--token/--no-token", default=False, show_default=True, help="Require auth token.")
def notebook(headless: bool, token: bool) -> None:
    """Launch the comprehensive marimo notebook in ``examples/examples_notebook.py``."""
    root = _repo_root()
    missing, details = _missing_python_modules(["marimo"])
    _fail_missing_dependencies(
        command_name="fmtools notebook",
        missing=missing,
        details=details,
        install_steps=[
            "uv pip install marimo",
            "uv sync --all-groups",
        ],
    )
    notebook_path = os.path.join(root, "examples", "examples_notebook.py")
    if not os.path.isfile(notebook_path):
        click.secho(f"Error: notebook not found at {notebook_path}", fg="red", err=True)
        raise SystemExit(1)

    cmd = ["uv", "run", "marimo", "run", notebook_path]
    if headless:
        cmd.append("--headless")
    cmd.append("--token" if token else "--no-token")
    rc = _run(cmd, cwd=root)
    raise SystemExit(rc)


# ── Desktop Demo ──────────────────────────────────────────────────────────────


@cli.command(name="chat")
@click.option(
    "--python",
    "run_python",
    is_flag=True,
    help="Run the demo as a Python module instead of Briefcase.",
)
@click.option(
    "--no-run",
    is_flag=True,
    help="For Briefcase mode: install/update app environment without launching UI.",
)
@click.option(
    "-r",
    "--update-requirements",
    is_flag=True,
    help="For Briefcase mode: force-refresh app requirements.",
)
@click.option(
    "--standard-gil",
    is_flag=True,
    help="Force standard-gil CPython runtime (recommended for maximum GUI stability).",
)
@click.argument("app_args", nargs=-1, type=click.UNPROCESSED)
def chat(
    run_python: bool,
    no_run: bool,
    update_requirements: bool,
    standard_gil: bool,
    app_args: tuple[str, ...],
) -> None:
    """Run the Toga + Briefcase local chat demo from the repository root.

    \b
    Examples:
        fmtools chat
        fmtools chat --no-run
        fmtools chat -r
        fmtools chat --python
        fmtools chat -- --test
    """
    root = _repo_root()
    project_dir = os.path.join(root, "examples", "toga_local_chat_app")
    project_toml = os.path.join(project_dir, "pyproject.toml")

    if not os.path.isfile(project_toml):
        click.secho(
            f"Error: demo project not found at {project_toml}",
            fg="red",
            err=True,
        )
        raise SystemExit(1)

    if standard_gil:
        standard_uv_args, standard_label = _select_standard_gil_uv_args(
            project_dir=project_dir, root=root
        )
        base = [
            "uv",
            "run",
            "--project",
            project_dir,
            "--directory",
            project_dir,
            *standard_uv_args,
        ]
        runtime_uv_args = standard_uv_args
        runtime_label = standard_label
        fallback_uv_args = standard_uv_args
        fallback_label = standard_label
        click.secho(f"Chat runtime: {runtime_label} (forced).", fg="green")
    else:
        base, runtime_uv_args, runtime_label = _select_chat_uv_runtime(
            project_dir=project_dir, root=root
        )
        fallback_uv_args, fallback_label = _select_standard_gil_uv_args(
            project_dir=project_dir, root=root
        )
        if runtime_uv_args:
            click.secho(f"Chat runtime: {runtime_label} (preferred no-gil).", fg="cyan")
        else:
            click.secho(
                "Chat runtime: standard-gil (free-threaded CPython unavailable, fallback active).",
                fg="yellow",
            )

    if run_python:
        missing, details = _missing_python_modules(
            ["fmtools_chat", "apple_fm_sdk", "toga"],
            project_dir=project_dir,
            uv_args=runtime_uv_args,
        )
        _fail_missing_dependencies(
            command_name="fmtools chat --python",
            missing=missing,
            details=details,
            install_steps=[
                "uv python install 3.14t",
                "uv python install 3.13t",
                "uv sync --project examples/toga_local_chat_app --directory examples/toga_local_chat_app",
                "uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app briefcase dev --no-run -r",
            ],
        )
        command_tail = ["python", "-m", "fmtools_chat", *app_args]
    else:
        missing, details = _missing_python_modules(
            ["briefcase"],
            project_dir=project_dir,
            uv_args=runtime_uv_args,
        )
        _fail_missing_dependencies(
            command_name="fmtools chat",
            missing=missing,
            details=details,
            install_steps=[
                "uv python install 3.14t",
                "uv python install 3.13t",
                "uv sync --project examples/toga_local_chat_app --directory examples/toga_local_chat_app",
                "uv run --project examples/toga_local_chat_app --directory examples/toga_local_chat_app briefcase dev --no-run -r",
            ],
        )
        command_tail = ["briefcase", "dev"]
        if update_requirements:
            command_tail.append("-r")
        if no_run:
            command_tail.append("--no-run")
        if app_args:
            command_tail.extend(["--", *app_args])

    cmd = [*base, *command_tail]
    rc = _run(cmd, cwd=root)
    if rc != 0 and runtime_uv_args and not standard_gil:
        click.secho(
            f"Free-threaded launch failed. Retrying with {fallback_label} runtime...",
            fg="yellow",
            err=True,
        )
        fallback_base = [
            "uv",
            "run",
            "--project",
            project_dir,
            "--directory",
            project_dir,
            *fallback_uv_args,
        ]
        rc = _run([*fallback_base, *command_tail], cwd=root)
    raise SystemExit(rc)


# ── Shell Completions ─────────────────────────────────────────────────────────

_COMPLETION_INSTRUCTIONS: dict[str, str] = {
    "bash": """\
# Add to ~/.bashrc:
eval "$(_FMTOOLS_COMPLETE=bash_source fmtools)"

# Or, generate a static file (faster startup):
_FMTOOLS_COMPLETE=bash_source fmtools > ~/.fmtools-complete.bash
echo '. ~/.fmtools-complete.bash' >> ~/.bashrc""",
    "zsh": """\
# Add to ~/.zshrc:
eval "$(_FMTOOLS_COMPLETE=zsh_source fmtools)"

# Or, generate a static file (faster startup):
_FMTOOLS_COMPLETE=zsh_source fmtools > ~/.fmtools-complete.zsh
echo '. ~/.fmtools-complete.zsh' >> ~/.zshrc""",
    "fish": """\
# Add to ~/.config/fish/completions/fmtools.fish:
_FMTOOLS_COMPLETE=fish_source fmtools > ~/.config/fish/completions/fmtools.fish""",
}


@cli.command()
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completions(shell: str) -> None:
    """Output shell completion setup instructions.

    \b
    Supported shells: bash, zsh, fish

    \b
    Examples:
        fmtools completions bash
        fmtools completions zsh
        fmtools completions fish
    """
    env_var = "_FMTOOLS_COMPLETE"
    source_type = f"{shell}_source"

    click.secho(f"\nShell completion for {shell}\n", fg="cyan", bold=True)
    click.secho("Setup instructions:", bold=True)
    click.echo()
    click.echo(_COMPLETION_INSTRUCTIONS[shell])
    click.echo()

    # Also emit the actual completion script to stdout for easy piping
    click.secho("Completion script:", bold=True)
    click.echo()
    try:
        result = subprocess.run(
            ["fmtools"],
            capture_output=True,
            text=True,
            env={**os.environ, env_var: source_type},
        )
        if result.stdout:
            click.echo(result.stdout)
        else:
            click.echo(f"  (Run: {env_var}={source_type} fmtools to generate the script directly)")
    except FileNotFoundError:
        click.echo(f"  (Run: {env_var}={source_type} fmtools to generate the script directly)")


# ── Entry point ───────────────────────────────────────────────────────────────


def cli_entry() -> None:
    """Entry point for the console_scripts."""
    try:
        cli()
    except AppleFMSetupError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    cli_entry()
