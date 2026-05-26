#!/usr/bin/env bash
# Full from-scratch flow feature refresh into *_new.csv files.
# Old CSVs are never touched — once you're happy, copy _new over manually.
#
#   FLOW_KMEANS_CUDA_VISIBLE_DEVICES=1 \
#   FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES=1 \
#   FLOW_MMD_CUDA_VISIBLE_DEVICES=1 \
#   bash scripts/transfer_analysis_v3/run_fresh_features.sh

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LOG_DIR="$ROOT/scripts/transfer_analysis_v3/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

FLOW_KMEANS_CUDA_VISIBLE_DEVICES="${FLOW_KMEANS_CUDA_VISIBLE_DEVICES:-}"
FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES="${FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES:-}"
FLOW_MMD_CUDA_VISIBLE_DEVICES="${FLOW_MMD_CUDA_VISIBLE_DEVICES:-}"
VEC_DIR="${VEC_DIR:-/mnt/nvme_1tb_b/coverage_vectors}"

RAW_CSV="analysis/coverage_v2_flow_only_raw_joint_full_new.csv"
KMEANS_CSV="analysis/coverage_v2_flow_only_raw_joint_kmeans_full_new.csv"
KMEANS_CURVES_CSV="analysis/coverage_v2_flow_only_raw_joint_kmeans_curves_full_new.csv"
SYM_CSV="analysis_v3/symmetric_distances_new.csv"
MMD_CSV="flow_mmd_results_fast_new.csv"

# ---------------------------------------------------------------------------
# Step 1: materialize raw coverage
# ---------------------------------------------------------------------------
log "Step 1: Materializing raw flow coverage -> $RAW_CSV"
python scripts/transfer_analysis_v3/materialize_flow_raw_coverage_from_pairwise.py \
    --pairwise-self analysis_v3/pairwise_self_distances.csv \
    --output "$RAW_CSV" \
    2>&1 | tee "$LOG_DIR/fresh_flow_coverage.log"

# ---------------------------------------------------------------------------
# Step 2: k-means coverage (needs a temp config with new output paths)
# ---------------------------------------------------------------------------
log "Step 2: Refreshing k-means coverage -> $KMEANS_CSV"
KMEANS_CFG_ORIG="src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml"
KMEANS_CFG_TMP="$(mktemp /tmp/kmeans_cfg_new_XXXXXX.yaml)"
sed \
    -e "s|kmeans_results_file:.*|kmeans_results_file: $KMEANS_CSV|" \
    -e "s|kmeans_curves_file:.*|kmeans_curves_file: $KMEANS_CURVES_CSV|" \
    "$KMEANS_CFG_ORIG" > "$KMEANS_CFG_TMP"

if [ -n "$FLOW_KMEANS_CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="$FLOW_KMEANS_CUDA_VISIBLE_DEVICES" \
    python scripts/calculate_coverage_faiss_flow_kmeans.py \
        --config "$KMEANS_CFG_TMP" \
        2>&1 | tee "$LOG_DIR/fresh_flow_kmeans.log"
else
    python scripts/calculate_coverage_faiss_flow_kmeans.py \
        --config "$KMEANS_CFG_TMP" \
        2>&1 | tee "$LOG_DIR/fresh_flow_kmeans.log"
fi
rm -f "$KMEANS_CFG_TMP"

# ---------------------------------------------------------------------------
# Step 3: FID / SW2 symmetric distances
# ---------------------------------------------------------------------------
log "Step 3: Refreshing FID/SW2 symmetric distances -> $SYM_CSV"
if [ -n "$FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="$FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES" \
    python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
        --flow-csv "$RAW_CSV" \
        --vec-dir "$VEC_DIR" \
        --output "$SYM_CSV" \
        --n-proj 200 \
        --sw-samples 100000 \
        --fid-samples 200000 \
        --skip-dino \
        2>&1 | tee "$LOG_DIR/fresh_symmetric_distances.log"
else
    python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
        --flow-csv "$RAW_CSV" \
        --vec-dir "$VEC_DIR" \
        --output "$SYM_CSV" \
        --n-proj 200 \
        --sw-samples 100000 \
        --fid-samples 200000 \
        --skip-dino \
        2>&1 | tee "$LOG_DIR/fresh_symmetric_distances.log"
fi

# ---------------------------------------------------------------------------
# Step 4: MMD (needs a temp config with new output path)
# ---------------------------------------------------------------------------
log "Step 4: Refreshing flow MMD -> $MMD_CSV"
MMD_CFG_ORIG="src/configs/mmd_configs/flow_mmd_config_full.yaml"
MMD_CFG_TMP="$(mktemp /tmp/mmd_cfg_new_XXXXXX.yaml)"
sed \
    -e "s|results_file:.*|results_file: $MMD_CSV|" \
    -e "s|required_pairs_file:.*|required_pairs_file: $RAW_CSV|" \
    "$MMD_CFG_ORIG" > "$MMD_CFG_TMP"

if [ -n "$FLOW_MMD_CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="$FLOW_MMD_CUDA_VISIBLE_DEVICES" \
    python scripts/calculate_flow_mmd.py \
        --config "$MMD_CFG_TMP" \
        2>&1 | tee "$LOG_DIR/fresh_flow_mmd.log"
else
    python scripts/calculate_flow_mmd.py \
        --config "$MMD_CFG_TMP" \
        2>&1 | tee "$LOG_DIR/fresh_flow_mmd.log"
fi
rm -f "$MMD_CFG_TMP"

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
log "Auditing new feature coverage..."
python scripts/transfer_analysis_v3/audit_flow_feature_coverage.py \
    --flow-raw      "$RAW_CSV" \
    --flow-kmeans   "$KMEANS_CSV" \
    --symmetric     "$SYM_CSV" \
    --flow-mmd      "$MMD_CSV" \
    2>&1 | tee "$LOG_DIR/fresh_feature_audit.log" || true

log "Done. New CSVs:"
log "  $RAW_CSV"
log "  $KMEANS_CSV"
log "  $KMEANS_CURVES_CSV"
log "  $SYM_CSV"
log "  $MMD_CSV"
log ""
log "To swap them in, run:"
log "  for f in $RAW_CSV $KMEANS_CSV $KMEANS_CURVES_CSV $SYM_CSV $MMD_CSV; do"
log "    cp \"\$f\" \"\${f/_new/}\"; done"
