#!/usr/bin/env bash
# =============================================================================
# scripts/build_cost_matrix.sh
#
# Build the 1000×1000 ImageNet semantic cost matrix from FastText word
# embeddings. Output: cost_matrix.pt (~4 MB) — host this somewhere publicly
# readable and point COST_MATRIX_URL at it (see config/README.md Step 6).
#
# The ImageNet synset → label mapping ships with the repo at
# `assets/imagenet/LOC_synset_mapping.txt` so this script has no third-party
# download dependency for class names. The only external input is the
# FastText `cc.en.300.bin` model (~7 GB) — auto-downloaded if you don't have
# it (with --download), or pass --fasttext to point at a copy you already have.
#
# Usage
# -----
#   # Simplest: auto-locate FastText (searches a few common paths)
#   bash scripts/build_cost_matrix.sh
#
#   # Explicit FastText path
#   bash scripts/build_cost_matrix.sh --fasttext /path/to/cc.en.300.bin
#
#   # Download FastText to ~/.cache/cacis/ first if missing (~4.5 GB gz; needs 7 GB free)
#   bash scripts/build_cost_matrix.sh --download
#
#   # Custom output
#   bash scripts/build_cost_matrix.sh --out /tmp/imagenet_cost_matrix.pt
#
# Once produced, upload the .pt file to a publicly readable URL and export:
#   export COST_MATRIX_URL=https://<your-host>/cost_matrix.pt
# =============================================================================
set -euo pipefail

# --- Defaults ---------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNSET_MAPPING="${REPO_ROOT}/assets/imagenet/LOC_synset_mapping.txt"
DATA_ROOT_DIR="${REPO_ROOT}/assets/imagenet"           # contains LOC_synset_mapping.txt
DEFAULT_OUT="${REPO_ROOT}/cost_matrix.pt"
DEFAULT_FASTTEXT_CACHE="${HOME}/.cache/cacis/cc.en.300.bin"
FASTTEXT_URL="https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"

FASTTEXT=""
OUT="${DEFAULT_OUT}"
DO_DOWNLOAD=0

# --- Arg parse --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --fasttext) FASTTEXT="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        --download) DO_DOWNLOAD=1; shift ;;
        -h|--help)
            sed -n '2,/^# ===/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)  echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# --- Synset mapping sanity --------------------------------------------------
if [[ ! -s "${SYNSET_MAPPING}" ]]; then
    echo "✗ Missing ${SYNSET_MAPPING}" >&2
    echo "  This file ships with the repo. Did you delete it?" >&2
    exit 1
fi
N_LINES=$(wc -l < "${SYNSET_MAPPING}" | tr -d ' ')
if [[ "${N_LINES}" -ne 1000 ]]; then
    echo "✗ ${SYNSET_MAPPING} has ${N_LINES} lines, expected 1000" >&2
    exit 1
fi
echo "✓ Synset mapping OK (${N_LINES} classes at ${SYNSET_MAPPING})"

# --- Locate FastText --------------------------------------------------------
locate_fasttext() {
    # Try a list of well-known paths.
    local candidates=(
        "${HOME}/.cache/cacis/cc.en.300.bin"
        "${HOME}/Downloads/cc.en.300.bin"
        "${HOME}/web/deraison/ai/cc.en.300.bin"
        "${REPO_ROOT}/cc.en.300.bin"
    )
    for p in "${candidates[@]}"; do
        if [[ -s "$p" ]]; then
            echo "$p"
            return 0
        fi
    done
    return 1
}

if [[ -z "${FASTTEXT}" ]]; then
    if FASTTEXT=$(locate_fasttext); then
        echo "✓ Found FastText model: ${FASTTEXT}"
    elif [[ ${DO_DOWNLOAD} -eq 1 ]]; then
        mkdir -p "$(dirname "${DEFAULT_FASTTEXT_CACHE}")"
        echo "→ Downloading FastText cc.en.300.bin.gz (~4.5 GB) to ${DEFAULT_FASTTEXT_CACHE}.gz"
        echo "  This takes 10–30 min on a typical home connection."
        curl --fail --location --progress-bar \
             --output "${DEFAULT_FASTTEXT_CACHE}.gz" "${FASTTEXT_URL}"
        echo "→ Decompressing (writes ~7 GB) ..."
        gunzip "${DEFAULT_FASTTEXT_CACHE}.gz"
        FASTTEXT="${DEFAULT_FASTTEXT_CACHE}"
    else
        cat >&2 <<EOF
✗ Could not find a FastText model. Either:
    1. Pass --fasttext /path/to/cc.en.300.bin     (preferred if you already have it)
    2. Pass --download                            (auto-fetch to ${DEFAULT_FASTTEXT_CACHE})
    3. Drop the .bin file at one of these paths:
${HOME}/.cache/cacis/cc.en.300.bin
${HOME}/Downloads/cc.en.300.bin
${HOME}/web/deraison/ai/cc.en.300.bin
${REPO_ROOT}/cc.en.300.bin
EOF
        exit 1
    fi
fi

# --- Ensure the python deps are present ------------------------------------
if ! python3 -c "import fasttext, torch, numpy" 2>/dev/null; then
    echo "→ Installing python deps (fasttext-wheel, torch, numpy) ..."
    pip install --quiet fasttext-wheel torch numpy
fi

# --- Build cost matrix -----------------------------------------------------
echo
echo "→ Building cost matrix"
echo "  data-root : ${DATA_ROOT_DIR}"
echo "  fasttext  : ${FASTTEXT}"
echo "  out       : ${OUT}"
echo

cd "${REPO_ROOT}"
python3 -m examples.imagenet.cost_matrix \
    --data-root "${DATA_ROOT_DIR}" \
    --fasttext  "${FASTTEXT}" \
    --out       "${OUT}"

# --- Done -------------------------------------------------------------------
size_kb=$(du -k "${OUT}" | awk '{print $1}')
echo
echo "============================================================"
echo "✓ Built ${OUT}  (${size_kb} KB)"
echo "============================================================"
echo
echo "Next: host it publicly and export COST_MATRIX_URL. Options:"
echo
echo "  # Backblaze B2 (uses your existing rclone config)"
echo "  rclone copy ${OUT} b2:cacis-imagenet/"
echo "  # then mark cost_matrix.pt public in the B2 web UI and grab the URL:"
echo "  export COST_MATRIX_URL=https://f000.backblazeb2.com/file/cacis-imagenet/cost_matrix.pt"
echo
echo "  # …or any static host you control (S3, GCS, Cloudflare R2, GitHub Release, etc.)"
