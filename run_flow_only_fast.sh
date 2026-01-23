#!/bin/bash
# Coverage Pipeline v2.0 - Flow-Only (2D) + ResNet + DINO FAST TEST
# Runs flow-only (dx, dy), ResNet, and DINO on subset of datasets
# Quick test to verify flow-only approach

set -e  # Exit on error

echo "========================================"
echo "Coverage Pipeline v2.0 - Flow-Only + ResNet + DINO FAST TEST"
echo "========================================"
echo ""
echo "Analysis type: Flow-Only (2D motion vectors) + Deep Features"
echo "Training sets: flyingthings, imagenet2dwarp, sintel"
echo "Eval sets: spair, tss, flyingthings, kitti2012, kitti2015, pfpascal, pfwillow"
echo ""
echo "Key differences:"
echo "  - Flow: ONLY (dx, dy) - no xy, no joint space"
echo "  - No alpha calibration needed"
echo "  - Faster: 2D instead of 4D"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"

# Flow-only representation (2D vectors, 1 space: flow)
echo ""
echo "========================================"
echo "1/3: FLOW-ONLY REPRESENTATION (2D)"
echo "========================================"
echo "Space: flow (dx, dy only)"
echo "Expected: ~15-20 min (uses cached flow vectors)"
echo ""
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_only_fast.yaml

# ResNet representation (256D after PCA, 1 space: features)
echo ""
echo "========================================"
echo "2/3: RESNET REPRESENTATION"
echo "========================================"
echo "Space: features (256D after PCA)"
echo "Expected: ~20-30 min"
echo ""
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2_fast.yaml

# DINO representation (256D after PCA, 1 space: features)
echo ""
echo "========================================"
echo "3/3: DINO REPRESENTATION"
echo "========================================"
echo "Space: features (256D after PCA)"
echo "Expected: ~20-30 min"
echo ""
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2_fast.yaml

echo ""
echo "========================================"
echo "FAST TEST COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - analysis/coverage_v2_flow_only_fast.csv"
echo "  - analysis/coverage_v2_resnet_fast.csv"
echo "  - analysis/coverage_v2_dino_fast.csv"
echo ""
echo "Coverage breakdown:"
echo "  - Flow-only: 1 space × 3 train × 7 eval = 21 train-eval pairs"
echo "  - ResNet: 1 space × 3 train × 7 eval = 21 train-eval pairs"
echo "  - DINO: 1 space × 3 train × 7 eval = 21 train-eval pairs"
echo "  - Total: 63 train-eval pairs"
echo ""
echo "Key insight: Flow-only space should avoid joint distribution collapse"
echo "Next step: If results look good, run ./run_flow_only_full.sh for complete analysis"
echo ""
