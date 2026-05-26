#!/usr/bin/env bash
# Transfer Analysis v4 — focused, claim-driven sweep.
# Run from project root: bash scripts/transfer_analysis_v4/run_v4.sh
#
# Assumes v3 has already produced its feature outputs:
#   scripts/transfer_analysis_v3/transfer_table.csv
#   analysis_v3/pairwise_self_distances.csv
#
# Env overrides:
#   TARGET=peak_pck            (default: auc_normalized)
#   N_BOOT=2000                bootstrap iterations
#   FAMILY_MATCHED=1           run appearance LOBO with DINO-IDW (ablation)
#   SKIP_BOOTSTRAP=1           skip bootstrap (use point estimates only)
#   SKIP_FIGURES=1             skip figures

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TARGETS="${TARGETS:-auc_normalized peak_pck}"
N_BOOT="${N_BOOT:-1000}"     # 1000 ≈ 8–10 min; 500 ≈ 4 min; 200 ≈ 1.5 min
FAMILY_MATCHED="${FAMILY_MATCHED:-0}"
SKIP_BOOTSTRAP="${SKIP_BOOTSTRAP:-0}"
SKIP_FIGURES="${SKIP_FIGURES:-0}"

OUT_DIR="${OUT_DIR:-scripts/transfer_analysis_v4/results}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$LOG_DIR" "$OUT_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Sanity-check inputs.
for required in \
    scripts/transfer_analysis_v3/transfer_table.csv \
    analysis_v3/pairwise_self_distances.csv
do
    if [ ! -f "$required" ]; then
        log "ERROR: required v3 input missing: $required"
        log "Run the v3 pipeline first (scripts/transfer_analysis_v3/run_pipeline.sh)"
        exit 1
    fi
done

EXTRA_ARGS=""
if [ "$FAMILY_MATCHED" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --family-matched-prior"
fi
if [ "${SKIP_GBM:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --no-gbm"
    log "Mode: skipping GBM head"
fi
if [ "${FLAT_FLOW_PRIOR:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --flat-flow-prior"
    log "Mode: legacy flat flow-IDW prior for all families"
fi
if [ -n "${L_MODE:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --l-mode $L_MODE"
    log "L mode: $L_MODE"
fi
if [ -n "${FEATURE_SUBSET:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --feature-subset $FEATURE_SUBSET"
    log "Feature subset (global): $FEATURE_SUBSET"
fi
if [ -n "${TARGETED_SUBSET:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --targeted-subset $TARGETED_SUBSET"
    log "Targeted-IDW subset: $TARGETED_SUBSET"
fi
if [ "${USE_RANKNET:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-ranknet"
    log "RankNet head: enabled"
fi
if [ "${PURE_ONLY:-1}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --pure-only"
    log "Source filter: 11 pure training datasets (set PURE_ONLY=0 to include mixed variants)"
fi

log "Step 1: Running experiments (targets=$TARGETS)..."
# shellcheck disable=SC2086
python scripts/transfer_analysis_v4/experiments.py \
    --targets $TARGETS \
    --out "$OUT_DIR" \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_DIR/experiments.log"

if [ "$SKIP_BOOTSTRAP" = "1" ]; then
    log "Step 2: SKIPPED (SKIP_BOOTSTRAP=1)"
    # Still need a summary.csv for compile; promote point estimates with NaN CIs.
    python -c "
import pandas as pd, numpy as np
df = pd.read_csv('$OUT_DIR/summary_points.csv')
# Promote per-head metrics into long-form rows so the rest of the pipeline
# (figures, compile) can use the same shape as bootstrap output.
rows = []
HEAD_COLS = {
    'g':        ('ctx_rho_g', 'cent_rho_g', 'ctx_rho_L', 'ctx_rho_Lg', 'abs_r_Lg'),
    'g_zridge': ('ctx_rho_g_zridge', 'cent_rho_g_zridge', 'ctx_rho_L', 'ctx_rho_Lg', 'abs_r_Lg_zridge'),
    'g_rank':   ('ctx_rho_g_rank', 'cent_rho_g_rank', 'ctx_rho_L', 'ctx_rho_Lg', 'abs_r_Lg'),
    'g_gbm':    ('ctx_rho_g_gbm', 'cent_rho_g_gbm', 'ctx_rho_L', 'ctx_rho_Lg', 'abs_r_Lg_gbm'),
}
canon = ('ctx_rho_g','cent_rho_g','ctx_rho_L','ctx_rho_Lg','abs_r_Lg')
for _, r in df.iterrows():
    for head, cols in HEAD_COLS.items():
        rec = {c: r.get(cols[i]) for i, c in enumerate(canon)}
        for c in canon:
            rec[f'{c}_lo'] = np.nan
            rec[f'{c}_hi'] = np.nan
        rec.update(split=r['split'], family=r['family'], label=r['label'],
                   target=r['target'], head=head, n_rows=r.get('n_rows', np.nan))
        rows.append(rec)
pd.DataFrame(rows).to_csv('$OUT_DIR/summary.csv', index=False)
# Empty gap file so compile_v4 doesn't choke.
pd.DataFrame(columns=['target','split','head','ctx_rho_g_gap','ctx_rho_g_gap_lo',
                      'ctx_rho_g_gap_hi','ctx_rho_g_gap_p_gt_0']) \
    .to_csv('$OUT_DIR/bootstrap_gap.csv', index=False)
print('wrote summary.csv from point estimates (no CIs)')
"
else
    log "Step 2: Bootstrap CIs (n_boot=$N_BOOT)..."
    python scripts/transfer_analysis_v4/bootstrap.py \
        --results "$OUT_DIR" \
        --n-boot "$N_BOOT" \
        2>&1 | tee "$LOG_DIR/bootstrap.log"
fi

if [ "$SKIP_FIGURES" = "1" ]; then
    log "Step 3: SKIPPED (SKIP_FIGURES=1)"
else
    log "Step 3: Figures..."
    python scripts/transfer_analysis_v4/figures.py \
        --results "$OUT_DIR" \
        2>&1 | tee "$LOG_DIR/figures.log"
fi

log "Step 4: Compiling results.md..."
python scripts/transfer_analysis_v4/compile_v4.py \
    --results "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/compile.log"

log "Done."
log "  report : $OUT_DIR/results.md"
log "  figures: $OUT_DIR/figures/"
log "  data   : $OUT_DIR/summary.csv  $OUT_DIR/bootstrap_gap.csv"
