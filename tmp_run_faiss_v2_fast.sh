#!/bin/bash
# Coverage Pipeline v2.0 - FAST TEST
# Quick test with 2 training sets and 7 eval sets
# Tests all three representations (flow, dino, resnet)

set -e  # Exit on error

echo "========================================"
echo "Coverage Pipeline v2.0 - FAST TEST"
echo "========================================"
echo ""
echo "Training sets: flyingthings, imagenet2dwarp, sintel"
echo "Eval sets: flyingthings, kitti2015, kitti2012, spair, pfpascal, pfwillow, tss"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"

# Flow representation (4D vectors, 3 spaces: xy, flow, joint)
echo ""
echo "========================================"
echo "1/3: FLOW REPRESENTATION"
echo "========================================"
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full_v2_fast.yaml

# DINO representation (256D after PCA, 1 space: features)
echo ""
echo "========================================"
echo "2/3: DINO REPRESENTATION"
echo "========================================"
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2_fast.yaml

# ResNet representation (256D after PCA, 1 space: features)
echo ""
echo "========================================"
echo "3/3: RESNET REPRESENTATION"
echo "========================================"
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2_fast.yaml

echo ""
echo "========================================"
echo "FAST TEST COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - analysis/coverage_v2_flow_fast.csv"
echo "  - analysis/coverage_v2_dino_fast.csv"
echo "  - analysis/coverage_v2_resnet_fast.csv"
echo ""
echo "Coverage breakdown:"
echo "  - Flow: 3 spaces × 3 train × 7 eval = 63 train-eval pairs"
echo "  - DINO: 1 space × 3 train × 7 eval = 21 train-eval pairs"
echo "  - ResNet: 1 space × 3 train × 7 eval = 21 train-eval pairs"
echo "  - Total: 105 train-eval pairs"
echo ""
