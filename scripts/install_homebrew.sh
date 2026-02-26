#!/usr/bin/env bash
# scripts/install_homebrew.sh — install SiliconRefinery via a local Homebrew tap.
#
# Usage:
#   ./scripts/install_homebrew.sh
#   ./scripts/install_homebrew.sh <tap-name>
#
# Defaults:
#   tap-name = adpena/silicon-refinery
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORMULA_SRC="${REPO_ROOT}/Formula/silicon-refinery.rb"
TAP_NAME="${1:-adpena/silicon-refinery}"
TAP_USER="${TAP_NAME%%/*}"
TAP_REPO="${TAP_NAME##*/}"
TAP_FULL_REPO="${TAP_USER}/homebrew-${TAP_REPO}"
FORMULA_FQN="${TAP_NAME}/silicon-refinery"
SHELL_NAME="$(basename "${SHELL:-}")"
RC_FILE=""
case "$SHELL_NAME" in
    zsh) RC_FILE="$HOME/.zshrc" ;;
    bash) RC_FILE="$HOME/.bashrc" ;;
    fish) RC_FILE="$HOME/.config/fish/config.fish" ;;
esac

if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew is not installed."
    echo "Install Homebrew first: https://brew.sh"
    exit 1
fi

if [ ! -f "$FORMULA_SRC" ]; then
    echo "Error: formula file not found at $FORMULA_SRC"
    exit 1
fi

echo "=== SiliconRefinery Homebrew Installer ==="
echo "Tap:      $TAP_NAME"
echo "Formula:  $FORMULA_FQN"
echo ""

if ! brew tap | grep -qx "$TAP_NAME"; then
    echo "Creating tap $TAP_NAME..."
    brew tap-new "$TAP_NAME"
else
    echo "Tap $TAP_NAME already exists."
fi

TAP_PATH="$(brew --repository "$TAP_FULL_REPO")"
mkdir -p "$TAP_PATH/Formula"
cp "$FORMULA_SRC" "$TAP_PATH/Formula/silicon-refinery.rb"
echo "Synced formula to: $TAP_PATH/Formula/silicon-refinery.rb"

if brew list --formula "$FORMULA_FQN" &>/dev/null; then
    echo "Reinstalling existing formula from HEAD..."
    brew reinstall --HEAD "$FORMULA_FQN"
else
    echo "Installing formula from HEAD..."
    brew install --HEAD "$FORMULA_FQN"
fi

echo ""
echo "Installed. Verifying CLI..."
if command -v silicon-refinery &>/dev/null; then
    echo "  silicon-refinery -> $(command -v silicon-refinery)"
    silicon-refinery --help >/dev/null
    echo "  CLI check passed."
else
    echo "  CLI not on PATH in this shell yet."
    if [ -n "$RC_FILE" ]; then
        echo "  Restart terminal or run: source $RC_FILE"
    else
        echo "  Restart terminal to refresh PATH."
    fi
fi
