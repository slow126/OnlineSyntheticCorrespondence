#!/usr/bin/env bash
# Lean density sweep — fast diagnostic, no bootstrap.
#
# 6 runs sequential (each uses full cores, ~10-15min):
#   results_lean_canon_mixed                    canonical at default pairwise CSV
#   results_lean_dL{1..5}_mixed                 5 matched (flow_N, dino_N) levels
#
# Settings per run:
#   L_MODE=mixed
#   TARGETS=peak_pck only
#   SKIP_GBM=1, SKIP_BOOTSTRAP=1, SKIP_FIGURES=1, USE_RANKNET=0 (default)
#
# Output: 6 dirs each with summary.csv (point estimates, no CIs).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DENSITY_DIR="analysis_v3/density_invariance_pair_sharded/combined"
LEVELS_FLOW=(50000   200000   1000000  4000000  8000000)
LEVELS_DINO=(25000   100000   500000   2000000  4000000)

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT=/tmp/v4_lean_sweep_${STAMP}
mkdir -p "$LOG_ROOT"
echo "Lean sweep started at $(date)"
echo "Logs: $LOG_ROOT"

run_lean() {
    local out_dir="$1"
    local logfile="$2"
    local extra_env="${3:-}"
    echo "  -> $out_dir @ $(date +%H:%M:%S)"
    env $extra_env \
        OUT_DIR="$out_dir" \
        L_MODE=mixed \
        TARGETS=peak_pck \
        N_BOOT=0 \
        SKIP_BOOTSTRAP=1 \
        SKIP_FIGURES=1 \
        SKIP_GBM=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh > "$logfile" 2>&1
}

T_START=$(date +%s)

# 0. Canonical reference (default pairwise CSV)
echo "[1/6] canonical-mixed (default density)"
run_lean "scripts/transfer_analysis_v4/results_lean_canon_mixed" "$LOG_ROOT/canon_mixed.log"

# 1-5. Density levels
for i in 0 1 2 3 4; do
    IDX=$((i+1))
    FLOW_N="${LEVELS_FLOW[$i]}"
    DINO_N="${LEVELS_DINO[$i]}"
    DIST_CSV="$DENSITY_DIR/pairwise_self_combined_flow${FLOW_N}_dino${DINO_N}.csv"
    if [ ! -f "$DIST_CSV" ]; then
        echo "MISSING: $DIST_CSV"
        exit 1
    fi
    echo "[$((IDX+1))/6] density level $IDX (flow=$FLOW_N, dino=$DINO_N)"
    run_lean \
        "scripts/transfer_analysis_v4/results_lean_dL${IDX}_mixed" \
        "$LOG_ROOT/dL${IDX}_mixed.log" \
        "DIST=$DIST_CSV"
done

T_END=$(date +%s)
echo
echo "Lean sweep done. Wall: $(( (T_END-T_START)/60 ))m $(((T_END-T_START)%60))s"
echo

# Compile combined ABLATION for the lean sweep.
DIRS=(results_lean_canon_mixed)
for IDX in 1 2 3 4 5; do
    DIRS+=("results_lean_dL${IDX}_mixed")
done
python scripts/transfer_analysis_v4/compile_ablation_summary.py \
    --dirs "${DIRS[@]}" \
    --out scripts/transfer_analysis_v4/ABLATION_lean_density.md 2>&1 | tail -10

echo
echo "Report: scripts/transfer_analysis_v4/ABLATION_lean_density.md"
