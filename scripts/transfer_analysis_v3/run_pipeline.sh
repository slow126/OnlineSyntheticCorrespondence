#!/usr/bin/env bash
# Transfer Analysis v3 — end-to-end pipeline
# Run from project root: bash scripts/transfer_analysis_v3/run_pipeline.sh
# Each step is skipped if its output already exists (safe to restart).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/scripts/transfer_analysis_v3/logs"
mkdir -p "$LOG_DIR" analysis_v3

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Run mode: set MINIMAL=1 to skip mixed training sets (10 train × 9 eval = 90
# pairs instead of 20 × 9 = 180). Useful for the first pass when mixed-dataset
# DINO vectors may not be needed / disk space is tight.
#
#   MINIMAL=1 bash scripts/transfer_analysis_v3/run_pipeline.sh   # minimal
#   bash scripts/transfer_analysis_v3/run_pipeline.sh              # full
# ---------------------------------------------------------------------------
MINIMAL="${MINIMAL:-0}"

PURE_TRAIN_DATASETS="flyingthings imagenet2dwarp pointodyssey sintel spair synthetic synthetic_2d_warp synthetic_large_zoom synthetic_random_flipping synthetic_small_zoom"

if [ "$MINIMAL" = "1" ]; then
    DINO_CONFIG="src/configs/coverage_configs/coverage_faiss_dino_v3_minimal.yaml"
    DINO_EXPECTED=90
    BUILD_TABLE_ARGS="--train-datasets ${PURE_TRAIN_DATASETS}"
    log "Mode: MINIMAL (10 pure training datasets, ${DINO_EXPECTED} DINO pairs expected)"
else
    DINO_CONFIG="src/configs/coverage_configs/coverage_faiss_dino_v3.yaml"
    DINO_EXPECTED=180
    BUILD_TABLE_ARGS=""
    log "Mode: FULL (including mixed training sets, ${DINO_EXPECTED} DINO pairs expected)"
fi

# ---------------------------------------------------------------------------
# Step 0a: DINO pairwise coverage (GPU FAISS, nprobe=4)
# Resumable — already-computed pairs are skipped by the coverage script.
# ---------------------------------------------------------------------------
DINO_CSV="$ROOT/analysis_v3/coverage_dino_full.csv"
if python3 -c "
import pandas as pd, sys
df = pd.read_csv('$DINO_CSV')
ok = 'mean_nn_eval_to_train_k1' in df.columns and len(df) >= $DINO_EXPECTED
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    log "Step 0a: DINO coverage already complete — skipping"
else
    log "Step 0a: Running DINO pairwise coverage (GPU IVF4096, ~1-2 hrs)..."
    python scripts/calculate_coverage_faiss_v2.py \
        --config "$DINO_CONFIG" \
        2>&1 | tee "$LOG_DIR/step0a_dino_coverage.log"
    log "Step 0a: Done"
fi

# ---------------------------------------------------------------------------
# Step 0b: DINO null-calibrated cosine coverage
# Requires DINO vectors and coverage_dino_full.csv from Step 0a. ~5-15 min GPU.
# ---------------------------------------------------------------------------
DINO_NULL_CSV="$ROOT/analysis_v3/dino_null_coverage.csv"
DINO_NULL_EXPECTED=$DINO_EXPECTED
if python3 -c "
import pandas as pd, sys
df = pd.read_csv('$DINO_NULL_CSV')
ok = 'eval_covered_by_train_null90' in df.columns and len(df) >= $DINO_NULL_EXPECTED
sys.exit(0 if ok else 1)
" 2>/dev/null; then
    log "Step 0b: DINO null coverage already complete — skipping"
else
    log "Step 0b: Computing DINO null-calibrated coverage..."
    python scripts/transfer_analysis_v3/compute_dino_null_coverage.py \
        --coverage-csv analysis_v3/coverage_dino_full.csv \
        --vec-dir /mnt/nvme_1tb_b/coverage_vectors \
        --output analysis_v3/dino_null_coverage.csv \
        --null-percentiles 80 90 95 99 \
        --gpu \
        2>&1 | tee "$LOG_DIR/step0b_dino_null_coverage.log"
    log "Step 0b: Done"
fi

# ---------------------------------------------------------------------------
# Step 0c: Symmetric distribution distances (FID + sliced Wasserstein)
# Resumable — already-computed pairs are skipped automatically.
# ---------------------------------------------------------------------------
SYM_CSV="$ROOT/analysis_v3/symmetric_distances.csv"
N_PAIRS=180
N_EXISTING=0
if [ -f "$SYM_CSV" ]; then
    N_EXISTING=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$SYM_CSV')))" 2>/dev/null || echo 0)
fi
if [ "$N_EXISTING" -ge "$N_PAIRS" ]; then
    log "Step 0c: Symmetric distances already complete ($N_EXISTING rows) — skipping"
else
    log "Step 0c: Computing FID + sliced Wasserstein (CPU, ~5-10 min)..."
    python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
        --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
        --vec-dir /mnt/nvme_1tb_b/coverage_vectors \
        --output "$SYM_CSV" \
        --n-proj 200 \
        --sw-samples 100000 \
        --fid-samples 200000 \
        2>&1 | tee "$LOG_DIR/step0c_symmetric_distances.log"
    log "Step 0c: Done"
fi

# ---------------------------------------------------------------------------
# Step 1: Assemble transfer table
# Always re-runs (fast, ~5s) so it picks up any new feature files.
# ---------------------------------------------------------------------------
log "Step 1: Assembling transfer table..."
# shellcheck disable=SC2086
python scripts/transfer_analysis_v3/build_table.py \
    $BUILD_TABLE_ARGS \
    2>&1 | tee "$LOG_DIR/step1_build_table.log"
log "Step 1: Done"

# ---------------------------------------------------------------------------
# Step 2: Run experiments (all splits × models × feature groups)
# Results are written incrementally; re-running is safe.
# ---------------------------------------------------------------------------
log "Step 2: Running experiments (overnight run — all splits × models × feature groups)..."
python scripts/transfer_analysis_v3/run_experiments.py \
    --splits loto loto_grouped lobo loco lomo \
    --models ridge bradley_terry plackett_luce kernel_ridge random global_prior \
    --feature-groups motion motion_km appearance density \
                     symmetric_mmd symmetric_ot symmetric_all \
                     motion_appearance all \
    --output-dir scripts/transfer_analysis_v3/results \
    2>&1 | tee "$LOG_DIR/step2_experiments.log"
log "Step 2: Done"

# ---------------------------------------------------------------------------
# Step 2b: Subsampling stability analysis (~30-60 min GPU)
# ---------------------------------------------------------------------------
STAB_CSV="$ROOT/scripts/transfer_analysis_v3/results/subsampling_stability/stability_table.csv"
if [ -f "$STAB_CSV" ]; then
    log "Step 2b: Subsampling stability already complete — skipping"
else
    log "Step 2b: Running subsampling stability analysis (~30-60 min, GPU)..."
    python scripts/transfer_analysis_v3/run_subsampling_stability.py \
        --coverage-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
        --vec-dir /mnt/nvme_1tb_b/coverage_vectors \
        --output-dir scripts/transfer_analysis_v3/results/subsampling_stability \
        --gpu \
        --caps 50000 200000 500000 2000000 -1 \
        2>&1 | tee "$LOG_DIR/step2b_subsampling_stability.log"
    log "Step 2b: Done"
fi

# ---------------------------------------------------------------------------
# Step 2c: Few-shot mixed-effects analysis (~1-2 min)
# ---------------------------------------------------------------------------
FEW_SHOT_CURVE="$ROOT/scripts/transfer_analysis_v3/results/few_shot/few_shot_learning_curve.csv"
if [ -f "$FEW_SHOT_CURVE" ]; then
    log "Step 2c: Few-shot analysis already complete — skipping"
else
    log "Step 2c: Running few-shot mixed-effects analysis..."
    python scripts/transfer_analysis_v3/run_few_shot_analysis.py \
        --table scripts/transfer_analysis_v3/transfer_table.csv \
        --output-dir scripts/transfer_analysis_v3/results/few_shot \
        --k-values 0 1 2 5 \
        --feature-groups motion motion_appearance all \
        2>&1 | tee "$LOG_DIR/step2c_few_shot.log"
    log "Step 2c: Done"
fi

# ---------------------------------------------------------------------------
# Step 3: Compile results → results.md
# ---------------------------------------------------------------------------
log "Step 3: Compiling results report..."
python scripts/transfer_analysis_v3/compile_results.py \
    --results-dir scripts/transfer_analysis_v3/results \
    --output scripts/transfer_analysis_v3/results/results.md \
    2>&1 | tee "$LOG_DIR/step3_compile_results.log"
log "Step 3: Done"

log "Pipeline complete."
log "  Summary CSV    : scripts/transfer_analysis_v3/results/summary_table.csv"
log "  Stability CSV  : scripts/transfer_analysis_v3/results/subsampling_stability/stability_table.csv"
log "  Few-shot CSV   : scripts/transfer_analysis_v3/results/few_shot/few_shot_learning_curve.csv"
log "  Results MD     : scripts/transfer_analysis_v3/results/results.md"
