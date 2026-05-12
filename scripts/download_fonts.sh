#!/usr/bin/env bash
# =============================================================================
# scripts/download_fonts.sh
#
# Fetch Montserrat-Regular.ttf into assets/fonts/.  Used as the global figure
# font (see examples/utils.py::setup_plot_style).
#
# Source: https://github.com/JulietaUla/Montserrat  (SIL Open Font License 1.1)
#
# Usage (from any working directory):
#   bash scripts/download_fonts.sh           # download if missing
#   bash scripts/download_fonts.sh --force   # re-download even if present
#
# Exit codes:
#   0 — font in place (already present or downloaded successfully)
#   1 — download failed / file looks corrupt
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FONT_DIR="$REPO_ROOT/assets/fonts"
FONT_PATH="$FONT_DIR/Montserrat-Regular.ttf"
URL="https://raw.githubusercontent.com/JulietaUla/Montserrat/master/fonts/ttf/Montserrat-Regular.ttf"
MIN_BYTES=100000   # plausibility check — a real TTF is ~200 KB

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

mkdir -p "$FONT_DIR"

if [[ -f "$FONT_PATH" && $FORCE -eq 0 ]]; then
    echo "✓ Font already present: $FONT_PATH"
    echo "  (run with --force to re-download)"
    exit 0
fi

echo "→ Downloading Montserrat-Regular.ttf"
echo "  from $URL"
echo "  to   $FONT_PATH"

if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error \
        --output "$FONT_PATH" "$URL"
elif command -v wget >/dev/null 2>&1; then
    wget --quiet --output-document="$FONT_PATH" "$URL"
else
    echo "✗ Neither curl nor wget is available." >&2
    exit 1
fi

# Cross-platform `stat -c` (Linux) vs `stat -f` (macOS / BSD).
size=$(stat -f%z "$FONT_PATH" 2>/dev/null || stat -c%s "$FONT_PATH")

if (( size < MIN_BYTES )); then
    echo "✗ Downloaded file is suspiciously small ($size bytes < ${MIN_BYTES})." >&2
    echo "  Removing $FONT_PATH" >&2
    rm -f "$FONT_PATH"
    exit 1
fi

echo "✓ Downloaded. Size: $size bytes"
