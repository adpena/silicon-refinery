#!/usr/bin/env bash
# Generate a macOS .icns file from a 1024x1024 source PNG.
#
# Usage:
#   ./scripts/generate_icns.sh path/to/source_1024x1024.png
#
# Output:
#   icons/fmchat.icns
#   src/fmchat/resources/fmchat.icns  (dev-mode copy)

set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:?Usage: $0 <source_1024x1024.png>}"
ICONSET="icons/fmchat.iconset"
ICNS="icons/fmchat.icns"
DEV_ICNS="src/fmchat/resources/fmchat.icns"

mkdir -p "$ICONSET"

# Generate all required sizes
sips -z   16   16 "$SRC" --out "$ICONSET/icon_16x16.png"      >/dev/null
sips -z   32   32 "$SRC" --out "$ICONSET/icon_16x16@2x.png"   >/dev/null
sips -z   32   32 "$SRC" --out "$ICONSET/icon_32x32.png"      >/dev/null
sips -z   64   64 "$SRC" --out "$ICONSET/icon_32x32@2x.png"   >/dev/null
sips -z  128  128 "$SRC" --out "$ICONSET/icon_128x128.png"    >/dev/null
sips -z  256  256 "$SRC" --out "$ICONSET/icon_128x128@2x.png" >/dev/null
sips -z  256  256 "$SRC" --out "$ICONSET/icon_256x256.png"    >/dev/null
sips -z  512  512 "$SRC" --out "$ICONSET/icon_256x256@2x.png" >/dev/null
sips -z  512  512 "$SRC" --out "$ICONSET/icon_512x512.png"    >/dev/null
sips -z 1024 1024 "$SRC" --out "$ICONSET/icon_512x512@2x.png" >/dev/null

# Convert to .icns
iconutil -c icns "$ICONSET" -o "$ICNS"

# Copy for dev mode
mkdir -p "$(dirname "$DEV_ICNS")"
cp "$ICNS" "$DEV_ICNS"

# Clean up .iconset
rm -rf "$ICONSET"

echo "Created: $ICNS"
echo "Copied:  $DEV_ICNS"
