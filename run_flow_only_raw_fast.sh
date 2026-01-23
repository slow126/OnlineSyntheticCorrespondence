#!/bin/bash
# Coverage Pipeline v2.1 - Flow-Only RAW EPSILON FAST TEST
# Runs raw flow epsilon curves on subset of datasets

set -e

echo "========================================"
echo "Coverage Pipeline v2.1 - Flow-Only RAW EPSILON FAST TEST"
echo "========================================"
echo ""
echo "Analysis type: Raw flow (dx, dy) epsilon curves"
echo "Training sets: flyingthings, imagenet2dwarp, sintel"
echo "Eval sets: spair, tss, flyingthings, kitti2012, kitti2015, pfpascal, pfwillow"
echo ""
echo "Key differences:"
echo "  - Flow: RAW (dx, dy) in pixel space"
echo "  - No self-radius normalization"
echo "  - Epsilon curves: 1, 2, 4, 8, 16, 32, 64 px"
echo ""

# Set PYTHONPATH
export PYTHONPATH="$PWD/scripts:$PYTHONPATH"

echo ""
echo "========================================"
echo "1/1: RAW FLOW EPSILON CURVES"
echo "========================================"
echo ""
python scripts/calculate_coverage_faiss_flow_eps.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_only_raw_fast.yaml

echo ""
echo "========================================"
echo "RAW FLOW FAST TEST COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - analysis/coverage_v2_flow_only_raw_fast.csv"
echo "  - analysis/coverage_v2_flow_only_raw_curves_fast.csv"
echo ""
