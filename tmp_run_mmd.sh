#!/usr/bin/env bash
set -euo pipefail

# MMD Pipeline v2.0 - Full Run
# Aligned with coverage pipeline v2.0

LOG_DIR="analysis/mmd_v2_logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1

echo "========================================="
echo "MMD Pipeline v2.0 - Starting"
echo "========================================="
echo ""

# Run Flow MMD (raw flow space, normalized)
echo "Running Flow MMD (flow space, normalized)..."
python -u scripts/calculate_mmd_v2.py \
  --config src/configs/mmd_configs/mmd_flow_v2.yaml \
  |& tee "$LOG_DIR/mmd_flow_v2.log"

# # Run ResNet MMD (with PCA + L2 norm)
# echo ""
# echo "Running ResNet MMD..."
# python -u scripts/calculate_mmd_v2.py \
#   --config src/configs/mmd_configs/mmd_resnet_v2.yaml \
#   |& tee "$LOG_DIR/mmd_resnet_v2.log"

# Run DINO MMD (with PCA + L2 norm)
echo ""
echo "Running DINO MMD..."
python -u scripts/calculate_mmd_v2.py \
  --config src/configs/mmd_configs/mmd_dino_v2.yaml \
  |& tee "$LOG_DIR/mmd_dino_v2.log"
echo ""
echo "========================================="
echo "All MMD runs complete!"
echo "========================================="
echo ""
echo "Results:"
echo "  - Flow:   analysis/mmd_v2_flow_joint.csv"
echo "  - ResNet: analysis/mmd_v2_resnet.csv"
echo "  - DINO:   analysis/mmd_v2_dino.csv"
echo ""
echo "Logs:"
echo "  - $LOG_DIR/"
echo ""
