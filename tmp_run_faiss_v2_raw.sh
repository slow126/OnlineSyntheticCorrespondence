#!/usr/bin/env bash
# Coverage Pipeline v2.2 Runner (Raw Flow Epsilon Curves + K-Means + ResNet + DINO)

set -u

LOG_DIR="analysis/full_faiss_coverage_logs_v2_raw"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "Coverage Pipeline v2.2 (Raw Flow Epsilon Curves + K-Means)"
echo "========================================="
echo "Features:"
echo "  - Flow: RAW (dx, dy) with epsilon curves (no self-radius)"
echo "  - ResNet/DINO: PCA + L2 normalization (v2.0 pipeline)"
echo "  - Squared L2 distances (no sqrt)"
echo "========================================="
echo

run_step() {
  local name="$1"
  shift
  local log_file="$1"
  shift
  echo "$name"
  "$@" |& tee "$LOG_DIR/$log_file"
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "  ⚠️  Step failed (exit=$status), continuing..."
  fi
  return 0
}

# Flow representation (raw flow epsilon curves)
run_step "[1/5] Running RAW FLOW epsilon coverage analysis..." \
  coverage_flow_raw_full.log \
  python -u scripts/calculate_coverage_faiss_flow_eps.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_full.yaml

echo
echo "[1/5] RAW FLOW epsilon complete ✓"
echo

# Flow representation (k-means codebooks)
run_step "[2/5] Running RAW FLOW k-means coverage analysis..." \
  coverage_flow_raw_kmeans_full.log \
  python -u scripts/calculate_coverage_faiss_flow_kmeans.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_full.yaml

echo
echo "[2/5] RAW FLOW k-means complete ✓"
echo

# Flow representation (k-means manifold distances)
run_step "[3/5] Running RAW FLOW k-means manifold analysis..." \
  coverage_flow_raw_kmeans_manifold_full.log \
  python -u scripts/calculate_coverage_faiss_flow_kmeans_manifold.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_full.yaml

echo
echo "[3/5] RAW FLOW k-means manifold complete ✓"
echo

# ResNet representation (single feature space)
run_step "[4/5] Running RESNET coverage analysis..." \
  coverage_resnet_full_v2.log \
  python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2.yaml

echo
echo "[4/5] RESNET complete ✓"
echo

# DINO representation (single feature space)
run_step "[5/5] Running DINO coverage analysis..." \
  coverage_dino_full_v2.log \
  python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml

echo
echo "[5/5] DINO complete ✓"
echo

echo "========================================="
echo "All coverage analyses complete!"
echo "========================================="
echo "Output files:"
echo "  - analysis/coverage_v2_flow_only_raw_full.csv"
echo "  - analysis/coverage_v2_flow_only_raw_curves_full.csv"
echo "  - analysis/coverage_v2_flow_only_raw_kmeans_full.csv"
echo "  - analysis/coverage_v2_flow_only_raw_kmeans_curves_full.csv"
echo "  - analysis/coverage_v2_flow_only_raw_kmeans_manifold_full.csv"
echo "  - analysis/coverage_v2_resnet_full.csv"
echo "  - analysis/coverage_v2_dino_full.csv"
echo
echo "Logs:"
echo "  - $LOG_DIR/"
echo "========================================="
