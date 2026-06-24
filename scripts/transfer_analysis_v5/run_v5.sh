#!/usr/bin/env bash
# Transfer Analysis v5 — Regime-Direction Law: one clean run, paper-ready numbers.
# Run from project root:  bash scripts/transfer_analysis_v5/run_v5.sh
#
# Regenerates every Outline-v2 table from raw inputs. Bootstraps are kept
# moderate (N_BOOT=200 experiments CIs, 500 consensus ratio, 10k cheap
# benchmark-bootstraps inside the fit-free scripts) — same-evening turnaround.
#
# Env:
#   SKIP_EXPERIMENTS=1   reuse existing results_rule_v5core (skip ~6 min fit + bootstrap)
#   N_BOOT=200           experiments bootstrap iterations
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
V4=scripts/transfer_analysis_v4
V5=scripts/transfer_analysis_v5
RES=$V5/results
CORE=$V4/results_rule_v5core
N_BOOT="${N_BOOT:-200}"
mkdir -p "$RES"
log() { echo "[$(date '+%H:%M:%S')] v5: $*"; }

log "Stage 1 — Regime-Direction verification (flow + dino, fit-free, 10k boot)"
python $V4/verify_regime_direction.py \
    --out $V4/regime_direction_verification/REPORT.md
python $V4/verify_regime_direction.py --space dino \
    --out $V4/regime_direction_verification/REPORT_dino.md

log "Stage 2 — rule hold-out checks (LOVO direction selection + continuum)"
python $V5/rule_holdout_checks.py --out $RES/rule_holdout_checks.csv

log "Stage 3 — asymmetric vs symmetric (mean_nn dirs vs sym/FID/W2/MMD)"
python $V5/asym_vs_sym_table.py --out $RES/asym_vs_sym.csv

if [ "${SKIP_EXPERIMENTS:-0}" != "1" ]; then
    log "Stage 4 — rule family under full L+g fold machinery (+ CIs, N=$N_BOOT)"
    python $V4/experiments.py \
        --targets peak_pck --pure-only --no-gbm \
        --families motion_rule motion_precision motion_recall motion_sym motion appearance \
        --out $CORE
    python $V4/bootstrap.py --results $CORE --n-boot "$N_BOOT"
else
    log "Stage 4 — SKIPPED (reusing $CORE)"
fi

log "Stage 5 — cross-architecture consensus with the rule predictor"
TMP=$(mktemp -d)
for s in LOTO LOBO JOINT; do
    cp $CORE/predictions/peak_pck/rows_${s}_motion_rule.csv "$TMP/rows_${s}_motion.csv"
done
( cd $V4 && python regenerate_consensus_csv.py \
    --rows-dir "$TMP" --min-src 4 --n-boot 500 --seed 0 \
    --out "$ROOT/$CORE/CONSENSUS_RULE.csv" )
rm -rf "$TMP"

log "Stage 6 — decision metrics (rule, symmetric, regime-aware policy)"
python $V5/make_policy_rows.py --rows-dir $CORE/predictions/peak_pck
python $V4/pairwise_gap_analysis.py \
    --rows-dir $CORE/predictions/peak_pck \
    --families motion_policy motion_rule motion_meannn_sym appearance \
    --out $RES/pairwise_gap_rule.csv
python $V4/selection_regret.py \
    --rows-dir $CORE/predictions/peak_pck \
    --families motion_policy motion_rule motion_meannn_sym appearance \
    --out $RES/selection_regret_rule.csv

log "Stage 7 — pre-registered out-of-sample test (kubric intervention grid)"
python $V5/intervention_oos_test.py --out $RES/intervention_oos.csv

log "Done. Artifacts:"
log "  Table 1 (law + CIs):     $V4/regime_direction_verification/REPORT.md (+ REPORT_dino.md)"
log "  Table 2/4 (rule, folds): $CORE/summary.csv  $CORE/bootstrap_gap.csv"
log "  Table 3 (consensus):     $CORE/CONSENSUS_RULE.csv"
log "  Table 5 (regret/gap):    $RES/selection_regret_rule.csv  $RES/pairwise_gap_rule.csv"
log "  Table 6 (asym vs sym):   $RES/asym_vs_sym.csv"
log "  §7.3 (out-of-sample):    $RES/intervention_oos.csv"
log "  hold-out checks:         $RES/rule_holdout_checks.csv"
