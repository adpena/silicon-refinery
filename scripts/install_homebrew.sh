#!/usr/bin/env bash
# scripts/install_homebrew.sh — install SiliconRefinery via published Homebrew tap.
#
# Usage:
#   ./scripts/install_homebrew.sh
#   ./scripts/install_homebrew.sh <tap-name>
#   ./scripts/install_homebrew.sh --no-chat
#
# Defaults:
#   tap-name = adpena/silicon-refinery
#
set -euo pipefail

TAP_NAME="adpena/silicon-refinery"
INSTALL_CHAT=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-chat)
      INSTALL_CHAT=0
      shift
      ;;
    -h|--help)
      cat <<USAGE
Usage: scripts/install_homebrew.sh [tap-name] [--no-chat]

Examples:
  ./scripts/install_homebrew.sh
  ./scripts/install_homebrew.sh adpena/silicon-refinery
  ./scripts/install_homebrew.sh --no-chat
USAGE
      exit 0
      ;;
    *)
      TAP_NAME="$1"
      shift
      ;;
  esac
done

TAP_USER="${TAP_NAME%%/*}"
TAP_REPO="${TAP_NAME##*/}"
TAP_FULL_REPO="${TAP_USER}/homebrew-${TAP_REPO}"
TAP_URL="https://github.com/${TAP_FULL_REPO}"
FORMULA_FQN="${TAP_NAME}/silicon-refinery"
CASK_FQN="${TAP_NAME}/silicon-refinery-chat"

if ! command -v brew >/dev/null 2>&1; then
  echo "Error: Homebrew is not installed. Install first: https://brew.sh" >&2
  exit 1
fi

echo "=== SiliconRefinery Homebrew Installer ==="
echo "Tap:      $TAP_NAME"
echo "Tap URL:  $TAP_URL"
echo "Formula:  $FORMULA_FQN"
if [[ "$INSTALL_CHAT" == "1" ]]; then
  echo "Cask:     $CASK_FQN"
fi
echo ""

if ! brew tap | grep -qx "$TAP_NAME"; then
  echo "Tapping $TAP_NAME..."
  brew tap "$TAP_NAME" "$TAP_URL"
else
  echo "Tap $TAP_NAME already configured."
fi

if brew list --formula "$FORMULA_FQN" >/dev/null 2>&1; then
  echo "Reinstalling CLI formula from HEAD..."
  brew reinstall --HEAD "$FORMULA_FQN"
else
  echo "Installing CLI formula from HEAD..."
  brew install --HEAD "$FORMULA_FQN"
fi

if [[ "$INSTALL_CHAT" == "1" ]]; then
  if brew list --cask "$CASK_FQN" >/dev/null 2>&1; then
    echo "Reinstalling chat app cask..."
    brew reinstall --cask "$CASK_FQN"
  else
    echo "Installing chat app cask..."
    brew install --cask "$CASK_FQN"
  fi
fi

echo ""
echo "Install complete."
if command -v silicon-refinery >/dev/null 2>&1; then
  echo "  silicon-refinery -> $(command -v silicon-refinery)"
  silicon-refinery --help >/dev/null
  echo "  CLI check passed."
else
  echo "  CLI not on PATH in this shell yet. Restart terminal/session."
fi
