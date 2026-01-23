#!/bin/bash
#
# Generate a single aggregated visualization across all datasets
#

INPUT_DIR="/mnt/nvme_1tb_b/coverage_vectors"
OUT_PATH="./aggregated_flow_splat.png"
PATTERN="*_flow.npy"  # Base flow files only

# Visualization parameters
K=1500                        # More clusters for aggregated view
SUBSAMPLE_PER_DATASET=500000  # 500K per dataset
SUBSAMPLE_FINAL=5000000       # 5M total max
MAX_RADIUS_PX=64
FLOW_BINS=512
DPI=250
SEED=42

echo "Running aggregated Gaussian splat visualization..."
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUT_PATH}"
echo "Pattern: ${PATTERN}"
echo ""
echo "K=${K}, Per-dataset subsample=${SUBSAMPLE_PER_DATASET}, Final subsample=${SUBSAMPLE_FINAL}"
echo ""

python3 visualize_aggregated_splats.py \
  --input_dir "${INPUT_DIR}" \
  --pattern "${PATTERN}" \
  --out_path "${OUT_PATH}" \
  --K ${K} \
  --subsample_per_dataset ${SUBSAMPLE_PER_DATASET} \
  --subsample_final ${SUBSAMPLE_FINAL} \
  --max_radius_px ${MAX_RADIUS_PX} \
  --flow_bins ${FLOW_BINS} \
  --dpi ${DPI} \
  --seed ${SEED}

echo ""
echo "Done! Output saved to: ${OUT_PATH}"
