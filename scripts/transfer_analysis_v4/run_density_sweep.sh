#!/usr/bin/env bash
# Density-variation sweep:
#   Context 1 (canonical, default pairwise CSV): full 17-mode ablation
#     6 L-modes + 11 feature subsets, both targets, ranknet on.
#   Contexts 2-6 (5 matched density levels along the diagonal):
#     3 headline L-modes only (mixed, eb_shrunk, density_idw).
#
# Concurrency: experiments.py uses ~17 BLAS cores per process; on a 32-core
# box we run modes 2-at-a-time to avoid thrashing. Sequential within a
# context's "batch" of 2 until all modes done; contexts run sequentially.
#
# Naming:
#   results_<mode>            -> canonical (default density)
#   results_dL<idx>_<mode>    -> density level idx (1..5 along diagonal)
#
# Env knobs:
#   N_BOOT (default 500)
#   PARALLEL (default 2 — modes running simultaneously)
#   SKIP_CANON=1 to skip canonical
#   SKIP_DENSITY=1 to skip density levels
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

N_BOOT="${N_BOOT:-500}"
PARALLEL="${PARALLEL:-2}"
SKIP_CANON="${SKIP_CANON:-0}"
SKIP_DENSITY="${SKIP_DENSITY:-0}"

# Canonical context: full 17-mode sweep.
CANON_LMODES=(mixed symmetric_informed symmetric_uninformed targeted_informed eb_shrunk density_idw)
CANON_FSUBS=(mean_nn mean_nn_sym mean_nn_asym coverage eps_1px eps_4px eps_16px kl kl_k5 kl_k20 asym_only)

# Density contexts: only the 3 headline L-modes.
DENS_LMODES=(mixed eb_shrunk density_idw)

# Density diagonal: 5 matched (flow_N, dino_N) levels.
DENSITY_DIR="analysis_v3/density_invariance_pair_sharded/combined"
LEVELS_FLOW=(50000   200000   1000000  4000000  8000000)
LEVELS_DINO=(25000   100000   500000   2000000  4000000)

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_ROOT=/tmp/v4_density_sweep_${STAMP}
mkdir -p "$LOG_ROOT"
echo "Sweep started at $(date)"
echo "Logs: $LOG_ROOT"
echo "Parallel modes per batch: $PARALLEL"
echo "N_BOOT: $N_BOOT"
echo

# Launch a single mode in the background; return its PID via stdout.
launch_mode() {
    local out_dir="$1"; shift
    local logfile="$1"; shift
    # remaining args: env-var=value pairs followed by `-- bash run_v4.sh`
    (
        env "$@" OUT_DIR="$out_dir" bash scripts/transfer_analysis_v4/run_v4.sh
    ) > "$logfile" 2>&1 &
    echo $!
}

# Wait until the number of background jobs drops below PARALLEL.
wait_for_slot() {
    while true; do
        local running
        running=$(jobs -rp | wc -l)
        if [ "$running" -lt "$PARALLEL" ]; then
            return
        fi
        sleep 5
    done
}

run_canonical() {
    echo "================================================================"
    echo "  CONTEXT 1: canonical (default pairwise_self_distances.csv)"
    echo "  Full 17-mode sweep, both targets, ranknet on"
    echo "================================================================"
    local t0=$(date +%s)
    for mode in "${CANON_LMODES[@]}"; do
        wait_for_slot
        echo "  [canon] L_MODE=$mode @ $(date +%H:%M:%S)"
        launch_mode \
            "scripts/transfer_analysis_v4/results_${mode}" \
            "$LOG_ROOT/canon_${mode}.log" \
            "L_MODE=$mode" "N_BOOT=$N_BOOT" "USE_RANKNET=1" \
            > /dev/null
    done
    for sub in "${CANON_FSUBS[@]}"; do
        wait_for_slot
        echo "  [canon] FEATURE_SUBSET=$sub @ $(date +%H:%M:%S)"
        launch_mode \
            "scripts/transfer_analysis_v4/results_fsub_${sub}" \
            "$LOG_ROOT/canon_fsub_${sub}.log" \
            "FEATURE_SUBSET=$sub" "N_BOOT=$N_BOOT" "USE_RANKNET=1" \
            > /dev/null
    done
    wait
    local t1=$(date +%s)
    echo "  CONTEXT 1 done. Wall: $(( (t1-t0)/60 ))m $(((t1-t0)%60))s"
}

run_density_level() {
    local idx="$1"
    local flow_n="$2"
    local dino_n="$3"
    local dist="$4"
    echo
    echo "================================================================"
    echo "  CONTEXT $((idx+1)): density level $idx (flow=$flow_n, dino=$dino_n)"
    echo "  Headline 3 L-modes (mixed, eb_shrunk, density_idw)"
    echo "  dist: $dist"
    echo "================================================================"
    local t0=$(date +%s)
    for mode in "${DENS_LMODES[@]}"; do
        wait_for_slot
        echo "  [dL${idx}] L_MODE=$mode @ $(date +%H:%M:%S)"
        launch_mode \
            "scripts/transfer_analysis_v4/results_dL${idx}_${mode}" \
            "$LOG_ROOT/dL${idx}_${mode}.log" \
            "L_MODE=$mode" "N_BOOT=$N_BOOT" "USE_RANKNET=1" "DIST=$dist" \
            > /dev/null
    done
    wait
    local t1=$(date +%s)
    echo "  CONTEXT $((idx+1)) done. Wall: $(( (t1-t0)/60 ))m $(((t1-t0)%60))s"
}

T_START=$(date +%s)

if [ "$SKIP_CANON" != "1" ]; then
    run_canonical
fi

if [ "$SKIP_DENSITY" != "1" ]; then
    for i in 0 1 2 3 4; do
        IDX=$((i+1))
        FLOW_N="${LEVELS_FLOW[$i]}"
        DINO_N="${LEVELS_DINO[$i]}"
        DIST_CSV="$DENSITY_DIR/pairwise_self_combined_flow${FLOW_N}_dino${DINO_N}.csv"
        if [ ! -f "$DIST_CSV" ]; then
            echo "MISSING: $DIST_CSV"
            exit 1
        fi
        run_density_level "$IDX" "$FLOW_N" "$DINO_N" "$DIST_CSV"
    done
fi

T_END=$(date +%s)
echo
echo "================================================================"
echo "SWEEP COMPLETE."
echo "  Total wall: $(( (T_END-T_START)/60 )) min ( $(( (T_END-T_START)/3600 )) h $(( ((T_END-T_START)%3600)/60 )) m )"
echo "  Logs: $LOG_ROOT"
echo "================================================================"

# Compile ABLATION summaries.
echo
echo "Compiling canonical ABLATION.md..."
CANON_DIRS=()
for mode in "${CANON_LMODES[@]}"; do CANON_DIRS+=("results_${mode}"); done
for sub in "${CANON_FSUBS[@]}"; do CANON_DIRS+=("results_fsub_${sub}"); done
python scripts/transfer_analysis_v4/compile_ablation_summary.py \
    --dirs "${CANON_DIRS[@]}" \
    --out scripts/transfer_analysis_v4/ABLATION.md 2>&1 | tail -3 || true

if [ "$SKIP_DENSITY" != "1" ]; then
    for IDX in 1 2 3 4 5; do
        DENS_DIRS=()
        for mode in "${DENS_LMODES[@]}"; do DENS_DIRS+=("results_dL${IDX}_${mode}"); done
        python scripts/transfer_analysis_v4/compile_ablation_summary.py \
            --dirs "${DENS_DIRS[@]}" \
            --out "scripts/transfer_analysis_v4/ABLATION_dL${IDX}.md" 2>&1 | tail -3 || true
    done
fi

echo
echo "Reports:"
echo "  scripts/transfer_analysis_v4/ABLATION.md         (canonical, 17 modes)"
echo "  scripts/transfer_analysis_v4/ABLATION_dL{1..5}.md (per density level, 3 headline modes)"
