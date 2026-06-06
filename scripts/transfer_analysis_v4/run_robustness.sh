#!/usr/bin/env bash
# Robustness suite for the transfer predictor — does the headline survive when we
# stop pretending the 11 sources are 11 independent draws?
#
# The 11 "pure" sources collapse to ~5 GENERATOR FAMILIES (see FAMILY_MAP in
# experiments.py): sdf3d (synthetic + 3 zoom/flip variants), warp2d
# (synthetic_2d_warp + imagenet2dwarp), kubric (movi_f), realflow (flyingthings,
# pointodyssey, sintel), semantic (spair). Two tests:
#
#   Tier 2  leave-one-generator-family-out: refit the WHOLE pipeline with each
#           family removed. If motion ctx_rho and the motion>appearance gap hold
#           under every drop, no single family is driving the result.
#   Tier 3  cluster bootstrap: resample at the family level (~5 clusters) instead
#           of the source level (11). Wider, honest CIs under within-family
#           correlation.
#
# Run from project root:
#   bash scripts/transfer_analysis_v4/run_robustness.sh
#
# Env overrides:
#   TARGETS="peak_pck"     restrict targets (default: peak_pck — the headline)
#   N_BOOT=1000            bootstrap iterations
#   FAMILIES="sdf3d warp2d kubric realflow semantic"   which families to drop
#   SKIP_FIGURES=1         (passed through) skip per-run figures to save time
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TARGETS="${TARGETS:-peak_pck}"
N_BOOT="${N_BOOT:-1000}"
FAMILIES="${FAMILIES:-sdf3d warp2d kubric realflow semantic}"
BASE_OUT="${BASE_OUT:-scripts/transfer_analysis_v4/results_robust}"
export TARGETS N_BOOT
# Per-run figures and the compile_v4 report are not needed here — compile_robustness
# reads the raw summary CSVs directly. compile_v4 also only knows summary.csv (not
# summary_cluster.csv), so it MUST be skipped for the cluster step or run_v4.sh's
# `set -e` aborts the whole suite. Skip both stages everywhere.
export SKIP_FIGURES="${SKIP_FIGURES:-1}"
export SKIP_COMPILE="${SKIP_COMPILE:-1}"
# compile_robustness only consumes motion + appearance (+ their paired gap), so the
# per-cell bootstrap doesn't need the other ~18 feature families. Restricting cuts
# each drop from ~2h to minutes. Override BOOT_FAMILIES="" to bootstrap everything.
export BOOT_FAMILIES="${BOOT_FAMILIES-motion appearance}"

log() { echo "[$(date '+%H:%M:%S')] [robust] $*"; }

# 0. Full-data reference (cluster bootstrap on the standard results dir) --------
# Set SKIP_CLUSTER=1 to reuse an existing results/summary_cluster.csv (e.g. to
# resume the suite after the family-drop loop failed downstream).
if [ "${SKIP_CLUSTER:-0}" = "1" ]; then
    log "Tier 3: SKIPPED (SKIP_CLUSTER=1, reusing existing results/summary_cluster.csv)"
else
    log "Tier 3: cluster bootstrap on the full-data fit (family-level resampling)"
    CLUSTER=1 N_BOOT="$N_BOOT" \
        bash scripts/transfer_analysis_v4/run_v4.sh \
        || { log "cluster bootstrap failed"; exit 1; }
fi
# (writes results/summary_cluster.csv + results/bootstrap_gap_cluster.csv)

# 1. Tier 2: leave-one-generator-family-out -----------------------------------
# Idempotent: a drop whose summary.csv already exists is skipped, so the suite can
# be re-run to resume after an interruption without redoing finished drops.
for fam in $FAMILIES; do
    out="${BASE_OUT}_drop_${fam}"
    if [ -f "$out/summary.csv" ]; then
        log "Tier 2: drop '$fam' already done ($out/summary.csv) — skipping"
        continue
    fi
    log "Tier 2: dropping family '$fam' -> $out"
    DROP_FAMILY="$fam" OUT_DIR="$out" \
        bash scripts/transfer_analysis_v4/run_v4.sh \
        || { log "drop '$fam' failed"; exit 1; }
done

# 2. Compile the comparison table ---------------------------------------------
log "Compiling robustness comparison -> ${BASE_OUT}/ROBUSTNESS_SUMMARY.csv"
python scripts/transfer_analysis_v4/compile_robustness.py \
    --base-out "$BASE_OUT" \
    --families "$FAMILIES" \
    --targets "$TARGETS"

log "Done. See ${BASE_OUT}/ROBUSTNESS_SUMMARY.csv and ROBUSTNESS_SUMMARY.md"
