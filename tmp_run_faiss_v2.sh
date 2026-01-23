#!/usr/bin/env bash
# Coverage Pipeline v2.0 Runner
# Runs all three representations with dual normalization and coverage curves

set -euo pipefail

LOG_DIR="analysis/full_faiss_coverage_logs_v2"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "Coverage Pipeline v2.0"
echo "========================================="
echo "Features:"
echo "  - Squared L2 distances (no sqrt)"
echo "  - Flow: multi-space (xy/flow/joint) + alpha calibration"
echo "  - Dino/ResNet: PCA + L2 normalization"
echo "  - Dual normalization (qnorm + rnorm)"
echo "  - Coverage curves over quantiles"
echo "  - 3090 24GB optimized batch sizes"
echo "========================================="
echo

# Flow representation (multi-space decomposition)
echo "[1/3] Running FLOW coverage analysis..."
python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full_v2.yaml \
  |& tee "$LOG_DIR/coverage_flow_full_v2.log"

echo
echo "[1/3] FLOW complete ✓"
echo

# ResNet representation (single feature space)
echo "[2/3] Running RESNET coverage analysis..."
python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2.yaml \
  |& tee "$LOG_DIR/coverage_resnet_full_v2.log"

echo
echo "[2/3] RESNET complete ✓"
echo

# DINO representation (single feature space)
echo "[3/3] Running DINO coverage analysis..."
python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml \
  |& tee "$LOG_DIR/coverage_dino_full_v2.log"

echo
echo "[3/3] DINO complete ✓"
echo

echo "========================================="
echo "All coverage analyses complete!"
echo "========================================="
echo "Output files:"
echo "  - analysis/coverage_v2_flow_full.csv"
echo "  - analysis/coverage_v2_resnet_full.csv"
echo "  - analysis/coverage_v2_dino_full.csv"
echo
echo "Coverage curves:"
echo "  - analysis/coverage_v2_flow_curves.csv"
echo "  - analysis/coverage_v2_resnet_curves.csv"
echo "  - analysis/coverage_v2_dino_curves.csv"
echo
echo "Logs:"
echo "  - $LOG_DIR/"
echo "========================================="
