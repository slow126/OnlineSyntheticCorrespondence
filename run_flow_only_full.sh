#!/bin/bash
# Coverage Pipeline v2.0 - Flow-Only (2D) + ResNet + DINO Full Analysis
# Runs flow-only (dx, dy), ResNet, and DINO on ALL datasets
# No spatial coordinates = no joint space problem = faster

set -e  # Exit on error

echo "========================================"
echo "Coverage Pipeline v2.0 - Flow-Only + ResNet + DINO FULL"
echo "========================================"
echo ""
echo "Analysis type: Flow-Only (2D motion vectors) + Deep Features"
echo "Training sets: flyingthings, imagenet2dwarp, sintel, synthetic, spair, pointodyssey + 10 mixed"
echo "Eval sets: flyingthings, kitti2015, kitti2012, spair, pfpascal, pfwillow, tss, pointodyssey, middlebury"
echo ""
echo "Changes from original:"
echo "  - Using ONLY flow space (dx, dy) - no xy, no joint"
echo "  - No alpha calibration needed"
echo "  - Faster: 2D instead of 4D"
echo "  - Same cache for vectors, separate radii cache for flow-only space"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"

# Flow-only representation (2D vectors, 1 space: flow)
echo ""
echo "========================================"
echo "1/3: FLOW-ONLY REPRESENTATION (2D)"
echo "========================================"
echo "Space: flow (dx, dy only)"
echo "No alpha calibration"
echo "Expected: ~30-40 min (uses cached flow vectors, recomputes radii for 2D)"
echo ""
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_only_full.yaml

# ResNet representation (256D after PCA, 1 space: features)
echo ""
echo "========================================"
echo "2/3: RESNET REPRESENTATION"
echo "========================================"
echo "Space: features (256D after PCA)"
echo "Expected: ~45-60 min (extraction + PCA + metrics)"
echo ""
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2.yaml

# DINO representation (256D after PCA, 1 space: features)
echo ""
echo "========================================"
echo "3/3: DINO REPRESENTATION"
echo "========================================"
echo "Space: features (256D after PCA)"
echo "Expected: ~45-60 min (extraction + PCA + metrics)"
echo ""
python scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml

echo ""
echo "========================================"
echo "FULL ANALYSIS COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - analysis/coverage_v2_flow_only_full.csv"
echo "  - analysis/coverage_v2_resnet_full.csv"
echo "  - analysis/coverage_v2_dino_full.csv"
echo ""
echo "Coverage breakdown:"
echo "  - Flow-only: 1 space × 16 train × 8 eval = 128 train-eval pairs"
echo "  - ResNet: 1 space × 16 train × 9 eval = 144 train-eval pairs"
echo "  - DINO: 1 space × 16 train × 9 eval = 144 train-eval pairs"
echo "  - Total: 416 train-eval pairs"
echo ""
echo "Key insight: Flow-only space avoids joint distribution collapse"
echo ""
