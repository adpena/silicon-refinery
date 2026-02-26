#!/usr/bin/env bash
# scripts/doctor.sh — Verify that all prerequisites for SiliconRefinery are met.
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ok=0
warn=0
fail=0

pass()  { ok=$((ok+1));   printf "${GREEN}[PASS]${NC}  %s\n" "$1"; }
skip()  { warn=$((warn+1)); printf "${YELLOW}[WARN]${NC}  %s\n" "$1"; }
fail()  { fail=$((fail+1)); printf "${RED}[FAIL]${NC}  %s\n" "$1"; }

echo "=== SiliconRefinery Doctor ==="
echo ""

# --- macOS version ---
macos_ver=$(sw_vers -productVersion 2>/dev/null || echo "unknown")
major_ver=$(echo "$macos_ver" | cut -d. -f1)
if [ "$major_ver" -ge 26 ] 2>/dev/null; then
    pass "macOS $macos_ver (>= 26.0 required)"
else
    fail "macOS $macos_ver — version 26.0+ (Tahoe) required"
fi

# --- Apple Silicon ---
arch=$(uname -m)
if [ "$arch" = "arm64" ]; then
    pass "Apple Silicon ($arch)"
else
    fail "Architecture is $arch — Apple Silicon (arm64) required"
fi

# --- Xcode ---
if xcode-select -p &>/dev/null; then
    xcode_path=$(xcode-select -p)
    pass "Xcode command-line tools ($xcode_path)"
else
    fail "Xcode command-line tools not installed — run: xcode-select --install"
fi

# --- Python ---
if command -v python3 &>/dev/null; then
    py_ver=$(python3 --version 2>&1 | awk '{print $2}')
    py_major=$(echo "$py_ver" | cut -d. -f1)
    py_minor=$(echo "$py_ver" | cut -d. -f2)
    if [ "$py_major" -gt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -ge 13 ]; }; then
        pass "Python $py_ver (>= 3.13 required)"
    else
        fail "Python $py_ver — version 3.13+ required"
    fi
else
    fail "python3 not found"
fi

# --- uv ---
if command -v uv &>/dev/null; then
    uv_ver=$(uv --version 2>&1 | head -1)
    pass "$uv_ver"
else
    fail "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

# --- ruff ---
if command -v ruff &>/dev/null; then
    ruff_ver=$(ruff --version 2>&1)
    pass "ruff $ruff_ver"
else
    skip "ruff not found (installed via uv sync)"
fi

# --- ty ---
if command -v ty &>/dev/null; then
    ty_ver=$(ty --version 2>&1)
    pass "ty $ty_ver"
else
    skip "ty not found (installed via uv sync)"
fi

# --- Apple FM SDK ---
if python3 -c "import apple_fm_sdk" &>/dev/null; then
    pass "apple_fm_sdk importable"
else
    skip "apple_fm_sdk not importable — see installation instructions"
fi

# --- Neural Engine availability ---
if python3 -c "
import apple_fm_sdk as fm
model = fm.SystemLanguageModel()
ok, reason = model.is_available()
if not ok:
    raise RuntimeError(reason)
" &>/dev/null; then
    pass "Foundation Model is available and ready"
else
    skip "Foundation Model not available (may require Apple Intelligence to be enabled)"
fi

echo ""
echo "=== Results: $ok passed, $warn warnings, $fail failed ==="
if [ "$fail" -gt 0 ]; then
    echo ""
    echo "Fix the failures above before proceeding."
    exit 1
fi
