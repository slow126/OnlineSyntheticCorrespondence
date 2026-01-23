#!/bin/bash
#
# Run alternative flow visualizations (more interpretable than Gaussian splats)
#

INPUT_DIR="/mnt/nvme_1tb_b/coverage_vectors"
OUT_DIR="./alternative_vis"
PATTERN="*_flow.npy"

SUBSAMPLE=50000           # For histograms and density plots
QUIVER_SUBSAMPLE=2000     # For vector field (keep low for readability)
DPI=150
SEED=42

echo "Running alternative flow visualizations..."
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUT_DIR}"
echo "Pattern: ${PATTERN}"
echo ""

python3 visualize_flow_alternatives.py \
  --input_dir "${INPUT_DIR}" \
  --pattern "${PATTERN}" \
  --out_dir "${OUT_DIR}" \
  --subsample ${SUBSAMPLE} \
  --quiver_subsample ${QUIVER_SUBSAMPLE} \
  --dpi ${DPI} \
  --seed ${SEED}

echo ""
echo "Done! Check ${OUT_DIR}/ for visualizations"
