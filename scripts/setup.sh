#!/usr/bin/env bash
# scripts/setup.sh — One-command setup for SiliconRefinery development.
#
# Usage:
#   ./scripts/setup.sh            # Full setup (install all dependency groups)
#   ./scripts/setup.sh --no-sdk   # Skip Apple FM SDK dependency group
#   ./scripts/setup.sh --no-cli-install  # Skip global CLI installation
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHELL_NAME="$(basename "${SHELL:-}")"
RC_FILE=""
case "$SHELL_NAME" in
    zsh) RC_FILE="$HOME/.zshrc" ;;
    bash) RC_FILE="$HOME/.bashrc" ;;
    fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
esac

echo "=== SiliconRefinery Setup ==="
echo ""

# --- Parse flags ---
SKIP_SDK=false
INSTALL_CLI=true
for arg in "$@"; do
    case "$arg" in
        --no-sdk) SKIP_SDK=true ;;
        --no-cli-install) INSTALL_CLI=false ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# --- Check prerequisites ---
echo "Checking prerequisites..."
if ! command -v uv &>/dev/null; then
    echo "uv not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "  uv $(uv --version | head -1)"

# Different filesystem roots can trigger uv hardlink warnings; copy mode is stable.
if [ -z "${UV_LINK_MODE:-}" ]; then
    export UV_LINK_MODE=copy
fi

# --- Apple FM SDK ---
echo ""
if [ "$SKIP_SDK" = false ]; then
    echo "Apple FM SDK will be resolved from pyproject via uv source."
else
    echo "Skipping Apple FM SDK dependency group (--no-sdk)."
fi

# --- Install SiliconRefinery ---
echo ""
echo "Installing SiliconRefinery in editable mode with dev dependencies..."
cd "$REPO_ROOT"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    uv venv
fi

# Sync dependency groups
if [ "$SKIP_SDK" = true ]; then
    uv sync --all-groups --no-group apple
else
    uv sync --all-groups
fi

if [ "$INSTALL_CLI" = true ]; then
    echo ""
    echo "Installing global silicon-refinery CLI (editable)..."
    uv tool install --editable --force --from "$REPO_ROOT" silicon-refinery

    echo "Ensuring uv tool bin directory is on PATH..."
    uv tool update-shell || true
fi

echo ""
echo "Verifying installation..."
uv run python -c "import silicon_refinery; print('  silicon_refinery imported successfully')"
uv run ruff --version | sed 's/^/  /'
uv run ty --version | sed 's/^/  /'
if command -v silicon-refinery &>/dev/null; then
    echo "  silicon-refinery available at: $(command -v silicon-refinery)"
else
    echo "  silicon-refinery not yet available in this shell PATH."
    if [ -n "$RC_FILE" ]; then
        echo "  Open a new terminal session (or run: source $RC_FILE) and try again."
    else
        echo "  Open a new terminal session and try again."
    fi
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate your environment:  source .venv/bin/activate"
echo "Run tests:                  uv run pytest"
echo "CLI (no activation needed): uv run silicon-refinery doctor"
echo "Global CLI:                 silicon-refinery doctor"
echo "Lint:                       uv run ruff check ."
echo "Type check:                 uv run ty check silicon_refinery/"
echo "Check prerequisites:        ./scripts/doctor.sh"
