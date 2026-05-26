#!/usr/bin/env bash
# Full from-scratch flow feature rerun into an isolated timestamped directory.
# Nothing existing is touched. When you're satisfied, run the generated do_swap.sh.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/transfer_analysis_v3/run_clean_flow_rerun.sh
#
# Override defaults:
#   VEC_DIR=/path/to/vectors  CUDA_VISIBLE_DEVICES=1  ...

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
VEC_DIR="${VEC_DIR:-/mnt/nvme_1tb_b/coverage_vectors}"
EXISTING_PAIRWISE="${EXISTING_PAIRWISE:-analysis_v3/pairwise_self_distances.csv}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$ROOT/runs/flow_${TIMESTAMP}"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$RUN_DIR" "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "=========================================================="
log "Clean flow rerun: $RUN_DIR"
log "GPU: $GPU  |  VEC_DIR: $VEC_DIR"
log "=========================================================="

# ---------------------------------------------------------------------------
# Output paths — all inside RUN_DIR
# ---------------------------------------------------------------------------
PAIRWISE_FLOW_ONLY="$RUN_DIR/pairwise_flow_only.csv"
PAIRWISE_CSV="$RUN_DIR/pairwise_self_distances.csv"
RAW_CSV="$RUN_DIR/coverage_v2_flow_only_raw_joint_full.csv"
KMEANS_CSV="$RUN_DIR/coverage_v2_flow_only_raw_joint_kmeans_full.csv"
KMEANS_CURVES_CSV="$RUN_DIR/coverage_v2_flow_only_raw_joint_kmeans_curves_full.csv"
SYM_CSV="$RUN_DIR/symmetric_distances.csv"
MMD_CSV="$RUN_DIR/flow_mmd_results_fast.csv"

# ---------------------------------------------------------------------------
# Step 0: Pairwise NN distances (flow only, fresh — no seed CSV)
# ---------------------------------------------------------------------------
log "Step 0: Computing pairwise NN + KL distances (flow space only)..."
CUDA_VISIBLE_DEVICES="$GPU" \
python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
    --vec-dir "$VEC_DIR" \
    --output  "$PAIRWISE_FLOW_ONLY" \
    --spaces  flow \
    --gpu \
    2>&1 | tee "$LOG_DIR/pairwise.log"

# Merge with existing DINO rows so the pairwise CSV is complete.
# --seed-csv is loaded first, so new flow rows take precedence over old.
log "Step 0b: Merging fresh flow rows with existing DINO rows..."
python scripts/transfer_analysis_v3/merge_pairwise_distances.py \
    --seed-csv "$PAIRWISE_FLOW_ONLY" \
    --inputs   "$EXISTING_PAIRWISE" \
    --output   "$PAIRWISE_CSV" \
    2>&1 | tee "$LOG_DIR/pairwise_merge.log"

# ---------------------------------------------------------------------------
# Step 1: Materialize raw flow coverage
# ---------------------------------------------------------------------------
log "Step 1: Materializing raw flow coverage..."
python scripts/transfer_analysis_v3/materialize_flow_raw_coverage_from_pairwise.py \
    --pairwise-self "$PAIRWISE_CSV" \
    --output        "$RAW_CSV" \
    2>&1 | tee "$LOG_DIR/raw_coverage.log"

# ---------------------------------------------------------------------------
# Step 2: K-means coverage (codebook reused from cache — vectors unchanged)
# ---------------------------------------------------------------------------
log "Step 2: Computing k-means coverage..."
KMEANS_CFG_TMP="$(mktemp /tmp/kmeans_cfg_clean_XXXXXX.yaml)"
sed \
    -e "s|kmeans_results_file:.*|kmeans_results_file: $KMEANS_CSV|" \
    -e "s|kmeans_curves_file:.*|kmeans_curves_file: $KMEANS_CURVES_CSV|" \
    src/configs/coverage_configs/coverage_faiss_flow_only_raw_joint_full.yaml \
    > "$KMEANS_CFG_TMP"
CUDA_VISIBLE_DEVICES="$GPU" \
python scripts/calculate_coverage_faiss_flow_kmeans.py \
    --config "$KMEANS_CFG_TMP" \
    2>&1 | tee "$LOG_DIR/kmeans.log"
rm -f "$KMEANS_CFG_TMP"

# ---------------------------------------------------------------------------
# Step 3: FID + SW2 symmetric distances (flow only — DINO already fresh)
# ---------------------------------------------------------------------------
log "Step 3: Computing FID and SW2 symmetric distances (flow only)..."
CUDA_VISIBLE_DEVICES="$GPU" \
python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
    --flow-csv    "$RAW_CSV" \
    --vec-dir     "$VEC_DIR" \
    --output      "$SYM_CSV" \
    --n-proj      200 \
    --sw-samples  100000 \
    --fid-samples 200000 \
    --skip-dino \
    2>&1 | tee "$LOG_DIR/symmetric.log"

# ---------------------------------------------------------------------------
# Step 4: Flow MMD
# ---------------------------------------------------------------------------
log "Step 4: Computing flow MMD..."
MMD_CFG_TMP="$(mktemp /tmp/mmd_cfg_clean_XXXXXX.yaml)"
sed \
    -e "s|results_file:.*|results_file: $MMD_CSV|" \
    -e "s|required_pairs_file:.*|required_pairs_file: $RAW_CSV|" \
    src/configs/mmd_configs/flow_mmd_config_full.yaml \
    > "$MMD_CFG_TMP"
CUDA_VISIBLE_DEVICES="$GPU" \
python scripts/calculate_flow_mmd.py \
    --config "$MMD_CFG_TMP" \
    2>&1 | tee "$LOG_DIR/mmd.log"
rm -f "$MMD_CFG_TMP"

# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------
log "Auditing feature coverage..."
python scripts/transfer_analysis_v3/audit_flow_feature_coverage.py \
    --flow-raw      "$RAW_CSV" \
    --flow-kmeans   "$KMEANS_CSV" \
    --symmetric     "$SYM_CSV" \
    --flow-mmd      "$MMD_CSV" \
    --pairwise-self "$PAIRWISE_CSV" \
    2>&1 | tee "$LOG_DIR/audit.log" || true

# ---------------------------------------------------------------------------
# Generate do_swap.sh in the run directory
# ---------------------------------------------------------------------------
SWAP_SCRIPT="$RUN_DIR/do_swap.sh"
cat > "$SWAP_SCRIPT" << EOF
#!/usr/bin/env bash
# Swap the clean rerun into production.
# Run from the repo root after you're satisfied with the audit output.
set -euo pipefail
cd "$(dirname "$SWAP_SCRIPT")/../.."

TS="\$(date +%Y%m%d_%H%M%S)"
BACKUP="backups/pre_flow_rerun_\${TS}"
mkdir -p "\$BACKUP/analysis" "\$BACKUP/analysis_v3"

echo "==> Backing up existing production files to \$BACKUP ..."
for f in \\
    analysis_v3/pairwise_self_distances.csv \\
    analysis/coverage_v2_flow_only_raw_joint_full.csv \\
    analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv \\
    analysis/coverage_v2_flow_only_raw_joint_kmeans_curves_full.csv \\
    analysis_v3/symmetric_distances.csv \\
    flow_mmd_results_fast.csv; do
    [ -f "\$f" ] && cp -v "\$f" "\$BACKUP/\$(dirname \$f)/"
done

echo "==> Swapping in new files from $RUN_DIR ..."
cp -v $PAIRWISE_CSV        analysis_v3/pairwise_self_distances.csv
cp -v $RAW_CSV             analysis/coverage_v2_flow_only_raw_joint_full.csv
cp -v $KMEANS_CSV          analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv
cp -v $KMEANS_CURVES_CSV   analysis/coverage_v2_flow_only_raw_joint_kmeans_curves_full.csv
cp -v $SYM_CSV             analysis_v3/symmetric_distances.csv
cp -v $MMD_CSV             flow_mmd_results_fast.csv

echo "==> Rebuilding transfer table ..."
python scripts/transfer_analysis_v3/build_table.py

echo "Done. Run compile_results.py / run_experiments.py as needed."
EOF
chmod +x "$SWAP_SCRIPT"

log "=========================================================="
log "All outputs in: $RUN_DIR"
log "Audit log:      $LOG_DIR/audit.log"
log ""
log "When satisfied, swap into production with:"
log "  bash $SWAP_SCRIPT"
log "=========================================================="
