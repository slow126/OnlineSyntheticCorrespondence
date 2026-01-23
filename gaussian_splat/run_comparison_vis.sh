#!/bin/bash
#
# Compare flow distributions across multiple datasets side-by-side
#

INPUT_DIR="/mnt/nvme_1tb_b/coverage_vectors"
OUT_PATH="./flow_comparison.png"

# Select datasets to compare (comma-separated, no spaces)
DATASETS="flyingthings_train_flow.npy,kitti2015_val_flow.npy,spair_test_flow.npy,pointodyssey_test_flow.npy"

SUBSAMPLE=50000
DPI=150
SEED=42

echo "Running flow comparison visualization..."
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUT_PATH}"
echo "Datasets: ${DATASETS}"
echo ""

python3 visualize_flow_comparison.py \
  --input_dir "${INPUT_DIR}" \
  --datasets "${DATASETS}" \
  --out_path "${OUT_PATH}" \
  --subsample ${SUBSAMPLE} \
  --dpi ${DPI} \
  --seed ${SEED}

echo ""
echo "Done! Comparison saved to ${OUT_PATH}"
