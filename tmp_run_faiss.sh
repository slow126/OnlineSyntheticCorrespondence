#!/usr/bin/env bash
set -euo pipefail

# Coverage Pipeline v2.0 - Full Run
# Includes all optional metrics: dual normalization, coverage curves, k=[1,5]

LOG_DIR="analysis/coverage_v2_logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "Coverage Pipeline v2.0 - Starting"
echo "========================================="
echo ""

# Run Flow (with multi-space decomposition: xy, flow, joint)
echo "Running Flow coverage..."
python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full_v2.yaml \
  |& tee "$LOG_DIR/coverage_flow_v2.log"

# Run ResNet (single feature space with PCA + L2 norm)
echo ""
echo "Running ResNet coverage..."
python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2.yaml \
  |& tee "$LOG_DIR/coverage_resnet_v2.log"

# Run DINO (single feature space with PCA + L2 norm)
echo ""
echo "Running DINO coverage..."
python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml \
  |& tee "$LOG_DIR/coverage_dino_v2.log"

echo ""
echo "========================================="
echo "All runs complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Flow:   analysis/coverage_v2_flow_full.csv"
echo "  - ResNet: analysis/coverage_v2_resnet_full.csv"
echo "  - DINO:   analysis/coverage_v2_dino_full.csv"
echo ""
echo "Logs:"
echo "  - $LOG_DIR/"
echo ""
