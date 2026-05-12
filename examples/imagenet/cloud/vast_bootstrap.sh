#!/usr/bin/env bash
# =============================================================================
# examples/imagenet/cloud/vast_bootstrap.sh
#
# Bootstrap script executed inside a Vast.ai container at instance start.
# Baked into the cacis-imagenet Docker image (see examples/imagenet/Dockerfile).
#
# Reads its configuration from environment variables set by the launcher:
#
#   KAGGLE_USERNAME, KAGGLE_KEY        — Kaggle API credentials
#   B2_KEY_ID, B2_APP_KEY              — Backblaze B2 credentials
#   RCLONE_DEST                        — e.g. b2:my-bucket/cacis-imagenet
#   COST_MATRIX_URL                    — public URL of cost_matrix.pt
#   LOSSES                             — space-separated list, e.g.
#                                        "cross_entropy sinkhorn_envelope ..."
#   RUN_PREFIX                         — output directory prefix
#   ARCH NUM_CLASSES BATCH_SIZE EPOCHS WARMUP_EPOCHS LR WEIGHT_DECAY
#   LABEL_SMOOTHING SINKHORN_MAX_ITER EPSILON_MODE EPSILON_SCALE NUM_WORKERS
#   AUTO_DESTROY                       — "true" to self-destruct when done
#   VAST_API_KEY                       — needed iff AUTO_DESTROY=true
#
# Vast.ai injects:
#   VAST_CONTAINERLABEL                — instance id used for self-destruct
# =============================================================================
set -uo pipefail

LOG=/workspace/cacis-bootstrap.log
exec > >(tee -a "$LOG") 2>&1

echo "=== CACIS Vast.ai bootstrap @ $(date -u) ==="
echo "GPU(s):"
nvidia-smi -L || true
echo

# -----------------------------------------------------------------------------
# 1. Extras the base image doesn't ship
# -----------------------------------------------------------------------------
apt-get update -q && apt-get install -y -q --no-install-recommends \
    unzip curl rclone
pip install --quiet --no-cache-dir kaggle

# -----------------------------------------------------------------------------
# 2. Credentials
# -----------------------------------------------------------------------------
mkdir -p "$HOME/.kaggle"
cat > "$HOME/.kaggle/kaggle.json" <<EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF
chmod 600 "$HOME/.kaggle/kaggle.json"

mkdir -p "$HOME/.config/rclone"
cat > "$HOME/.config/rclone/rclone.conf" <<EOF
[b2]
type = b2
account = ${B2_KEY_ID}
key = ${B2_APP_KEY}
EOF
chmod 600 "$HOME/.config/rclone/rclone.conf"

# -----------------------------------------------------------------------------
# 3. Download ImageNet from Kaggle (idempotent)
# -----------------------------------------------------------------------------
mkdir -p /data/imagenet && cd /data/imagenet
if [[ ! -d "ILSVRC/Data/CLS-LOC/train" ]]; then
    echo "→ Downloading ImageNet from Kaggle (~150 GB) ..."
    kaggle competitions download -c imagenet-object-localization-challenge
    echo "→ Extracting ..."
    unzip -q imagenet-object-localization-challenge.zip
    rm -f imagenet-object-localization-challenge.zip
else
    echo "→ ImageNet already present at /data/imagenet/ILSVRC — skipping download."
fi

# -----------------------------------------------------------------------------
# 4. Reorganize val/ into synset subfolders (Kaggle ships it flat)
# -----------------------------------------------------------------------------
python3 - <<'PY'
import csv, shutil
from pathlib import Path
val_dir = Path("/data/imagenet/ILSVRC/Data/CLS-LOC/val")
sol = Path("/data/imagenet/LOC_val_solution.csv")
if not (val_dir.exists() and sol.exists()):
    raise SystemExit(f"val structure not found at {val_dir}")
moved = 0
with open(sol) as f:
    for row in csv.DictReader(f):
        img_id = row["ImageId"]
        synset = row["PredictionString"].split()[0]
        src = val_dir / f"{img_id}.JPEG"
        if src.exists():
            dst = val_dir / synset / f"{img_id}.JPEG"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            moved += 1
print(f"→ moved {moved} val images into synset folders")
PY

# -----------------------------------------------------------------------------
# 5. Stitch an ImageFolder-compatible data_root
# -----------------------------------------------------------------------------
mkdir -p /data/imagefolder
ln -sfn /data/imagenet/ILSVRC/Data/CLS-LOC/train /data/imagefolder/train
ln -sfn /data/imagenet/ILSVRC/Data/CLS-LOC/val   /data/imagefolder/val
ln -sfn /data/imagenet/LOC_synset_mapping.txt    /data/imagefolder/LOC_synset_mapping.txt

# -----------------------------------------------------------------------------
# 6. Cost matrix
# -----------------------------------------------------------------------------
curl --fail --location --silent --show-error \
    --output /workspace/cost_matrix.pt "${COST_MATRIX_URL}"

# -----------------------------------------------------------------------------
# 7. Train each loss in sequence
# -----------------------------------------------------------------------------
mkdir -p /workspace/imagenet_output

for LOSS in ${LOSSES}; do
    RUN_ID="${RUN_PREFIX}-${LOSS}"
    echo
    echo "============================================================"
    echo "=== Training ${LOSS} → ${RUN_ID} @ $(date -u)"
    echo "============================================================"

    torchrun --standalone --nproc-per-node=1 \
        -m examples.imagenet.train \
            --data-root /data/imagefolder \
            --cost-matrix /workspace/cost_matrix.pt \
            --loss "${LOSS}" \
            --arch "${ARCH}" \
            --num-classes "${NUM_CLASSES}" \
            --batch-size "${BATCH_SIZE}" \
            --epochs "${EPOCHS}" \
            --warmup-epochs "${WARMUP_EPOCHS}" \
            --lr "${LR}" \
            --weight-decay "${WEIGHT_DECAY}" \
            --label-smoothing "${LABEL_SMOOTHING}" \
            --sinkhorn-max-iter "${SINKHORN_MAX_ITER}" \
            --epsilon-mode "${EPSILON_MODE}" \
            --epsilon-scale "${EPSILON_SCALE}" \
            --num-workers "${NUM_WORKERS}" \
            --output-dir /workspace/imagenet_output \
            --run-id "${RUN_ID}" \
        || echo "✗ ${LOSS} training failed — continuing with next loss"

    # Incremental upload per loss so a later instance death doesn't lose work
    echo "→ rclone copy /workspace/imagenet_output/${RUN_ID} → ${RCLONE_DEST}/${RUN_ID}"
    rclone copy "/workspace/imagenet_output/${RUN_ID}" "${RCLONE_DEST}/${RUN_ID}" \
        --transfers 4 --checkers 8 || echo "✗ rclone copy failed for ${RUN_ID}"
done

# -----------------------------------------------------------------------------
# 8. Final sync (catches anything missed) + bootstrap log
# -----------------------------------------------------------------------------
rclone copy /workspace/imagenet_output "${RCLONE_DEST}" \
    --transfers 4 --checkers 8 || true
rclone copy "${LOG}" "${RCLONE_DEST}/cacis-bootstrap.log" || true

echo
echo "=== Done @ $(date -u) ==="

# -----------------------------------------------------------------------------
# 9. Self-destroy (cost discipline)
# -----------------------------------------------------------------------------
if [[ "${AUTO_DESTROY:-false}" == "true" && -n "${VAST_CONTAINERLABEL:-}" ]]; then
    echo "→ self-destroying instance ${VAST_CONTAINERLABEL}"
    curl -X DELETE --silent --show-error \
        "https://console.vast.ai/api/v0/instances/${VAST_CONTAINERLABEL}/?api_key=${VAST_API_KEY}" \
        || echo "✗ self-destroy request failed (instance will continue billing)"
fi
