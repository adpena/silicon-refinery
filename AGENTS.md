# Repository Guidelines

## Project Structure & Module Organization
- Core package: `fmtools/` (public API, CLI, adapters, pipeline, debugging, integrations).
- Tests: `tests/` with `test_*.py` modules and shared fixtures in `tests/conftest.py`.
- Examples and runnable demos: `examples/` and `use_cases/`.
- Utility scripts: `scripts/` (`setup.sh`, `doctor.sh`, Homebrew/install helpers).
- Sample data: `datasets/`.
- Packaging and tool config: `pyproject.toml`; release notes in `CHANGELOG.md`.

## Build, Test, and Development Commands
- `./scripts/setup.sh`: bootstrap local dev environment (installs `uv`, syncs deps, optional CLI install).
- `uv sync --all-groups`: install runtime + dev dependencies into `.venv`.
- `uv run fmtools doctor`: validate platform and dependency prerequisites.
- `uv run fmtools lint`: run Ruff checks.
- `uv run fmtools format`: run Ruff formatter.
- `uv run fmtools typecheck`: run `ty` type checks for `fmtools/`.
- `uv run fmtools test`: run pytest suite.
- `uv run fmtools check`: full CI-equivalent gate (lint + format + typecheck + tests).

## Coding Style & Naming Conventions
- Python 3.13+ target, 4-space indentation, max line length 100 (Ruff config).
- Prefer explicit type hints and clean async patterns (`async`/`await`) where applicable.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep function docstrings precise: in this project they can influence LLM-facing behavior.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` (`asyncio_mode = auto`).
- Place tests in `tests/test_<feature>.py` and mirror source module intent.
- Add regression tests for bug fixes and edge cases (sync + async paths when relevant).
- Run `uv run fmtools test -v` locally before opening a PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history, e.g.:
  - `feat: add context compaction mode`
  - `chore: clean build artifacts`
- Keep commits focused and self-contained.
- PRs should include: clear summary, motivation, linked issue (if any), and test evidence (command(s) + result).
- Update docs/examples/changelog when behavior or public APIs change.

## Security & Configuration Tips
- Never commit secrets or local machine artifacts; keep `.gitignore` clean.
- Apple SDK support is sourced via `tool.uv.sources`; verify dependency changes carefully.
