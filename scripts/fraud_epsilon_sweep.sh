#!/usr/bin/env bash
# =============================================================================
# scripts/fraud_epsilon_sweep.sh
#
# Search for an ε that lets a Sinkhorn-based loss beat cross-entropy + the
# naive ``approve-all'' baseline on IEEE-CIS / Vesta.
#
# Sweeps a small grid over (loss, epsilon_mode, epsilon_scale) and writes
# every run under fraud_output/eps_sweep/<tag>/. Aggregates with
# scripts/eps_sweep_table.py at the end.
#
# Usage
# -----
#   bash scripts/fraud_epsilon_sweep.sh                  # default 5 epochs/run
#   EPOCHS=10 bash scripts/fraud_epsilon_sweep.sh        # more thorough
#   RUN_PREFIX=eps_v2 bash scripts/fraud_epsilon_sweep.sh # parallel namespace
#
# Cost
# ----
# Eight runs × ~15-30 min each = ~2-4 h on a MacBook. CPU only; no cloud.
# =============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EPOCHS=${EPOCHS:-5}
OUT_ROOT=${OUT_ROOT:-fraud_output}
RUN_PREFIX=${RUN_PREFIX:-eps_sweep}
DEVICE=${DEVICE:-auto}

mkdir -p "${OUT_ROOT}/${RUN_PREFIX}"
TS=$(date -u +%Y-%m-%dT%H-%M-%SZ)
LOG="${OUT_ROOT}/${RUN_PREFIX}/sweep_${TS}.log"

echo "Logging to ${LOG}"
echo "==== ε-sweep started ${TS} (epochs=${EPOCHS}) ====" | tee -a "${LOG}"

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
run_one() {
    local tag=$1 loss=$2 mode=$3 scale=$4
    local rid="${RUN_PREFIX}/${tag}"
    local start=$(date +%s)
    echo
    echo "─── ${tag}  loss=${loss}  ε-mode=${mode}  ε-scale=${scale} ───" | tee -a "${LOG}"

    if [[ "${loss}" == "cross_entropy" ]]; then
        python -m examples.fraud_detection \
            --loss "${loss}" \
            --epochs "${EPOCHS}" \
            --device "${DEVICE}" \
            --run-id "${rid}" \
            2>&1 | tee -a "${LOG}" | tail -5
    else
        python -m examples.fraud_detection \
            --loss "${loss}" \
            --epsilon-mode "${mode}" \
            --epsilon-scale "${scale}" \
            --epochs "${EPOCHS}" \
            --device "${DEVICE}" \
            --run-id "${rid}" \
            2>&1 | tee -a "${LOG}" | tail -5
    fi
    local rc=${PIPESTATUS[0]}
    local elapsed=$(( $(date +%s) - start ))
    if [[ ${rc} -eq 0 ]]; then
        echo "    ✓ ${tag} done in ${elapsed}s" | tee -a "${LOG}"
    else
        echo "    ✗ ${tag} FAILED (exit ${rc}) after ${elapsed}s" | tee -a "${LOG}"
    fi
}

# -----------------------------------------------------------------------------
# The grid — every diagnostic the previous run suggested
# -----------------------------------------------------------------------------
#
#  • CE baseline = upper bar to beat (val_realized_regret ≈ $5.30 historically)
#  • POT × offdiag_median × {0.05, 0.1, 0.5, 2.0}  ←  bracket the regret minimum
#  • POT × offdiag_mean   × 0.1                    ←  mode-comparison anchor
#  • FY  × offdiag_median × {2.0, 5.0}             ←  FY was least-collapsed
#
run_one  ce                 cross_entropy           none            0
run_one  pot_med_005        sinkhorn_pot            offdiag_median  0.05
run_one  pot_med_010        sinkhorn_pot            offdiag_median  0.10
run_one  pot_med_050        sinkhorn_pot            offdiag_median  0.50
run_one  pot_med_200        sinkhorn_pot            offdiag_median  2.00
run_one  pot_mean_010       sinkhorn_pot            offdiag_mean    0.10
run_one  fy_med_200         sinkhorn_fenchel_young  offdiag_median  2.00
run_one  fy_med_500         sinkhorn_fenchel_young  offdiag_median  5.00

# -----------------------------------------------------------------------------
# Aggregate
# -----------------------------------------------------------------------------
echo
echo "==== Building comparison table ====" | tee -a "${LOG}"
python scripts/eps_sweep_table.py \
    --root "${OUT_ROOT}/${RUN_PREFIX}" \
    --out  "${OUT_ROOT}/${RUN_PREFIX}/sweep_comparison.csv" 2>&1 | tee -a "${LOG}"

echo
echo "==== Done. Full log: ${LOG} ===="
echo "Inspect:   open ${OUT_ROOT}/${RUN_PREFIX}/<tag>/<loss>/val_realized_regret.png"
echo "Re-render: python -m examples.fraud_detection ...  (any single tag)"
