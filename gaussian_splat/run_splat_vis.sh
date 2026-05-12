#!/bin/bash
#
# Convenience script to run Gaussian splat visualization on cached flow vectors
# Modify parameters below as needed
#

INPUT_DIR="/mnt/nvme_1tb_b/coverage_vectors"
OUT_DIR="./output_splats"
PATTERN="*_flow.npy"  # Only visualize the base flow files (not radius variants)

# Visualization parameters
K=800                 # Number of Gaussian clusters
SUBSAMPLE=2000000     # Max vectors per dataset (2M)
MAX_RADIUS_PX=64      # Max splat radius
FLOW_BINS=512         # Flow space histogram bins
DPI=200               # Output figure DPI
SEED=42               # Random seed
SHOW_ENDPOINT=0       # 1=show endpoint footprint panel, 0=hide it
LEGEND_SIDE="left"    # inside, left, right

# Optional: specify image dimensions (leave commented to auto-infer)
# HEIGHT=384
# WIDTH=1248

echo "Running Gaussian splat visualization..."
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUT_DIR}"
echo "Pattern: ${PATTERN}"
echo ""

# Build command
CMD="python3 visualize_flow_splats.py \
  --input_dir ${INPUT_DIR} \
  --pattern \"${PATTERN}\" \
  --out_dir ${OUT_DIR} \
  --K ${K} \
  --subsample ${SUBSAMPLE} \
  --max_radius_px ${MAX_RADIUS_PX} \
  --flow_bins ${FLOW_BINS} \
  --dpi ${DPI} \
  --seed ${SEED} \
  --legend-side ${LEGEND_SIDE}"

if [ "${SHOW_ENDPOINT}" -eq 0 ]; then
  CMD="${CMD} --no-endpoint"
fi

# Add optional dimensions if set
if [ -n "${HEIGHT}" ]; then
  CMD="${CMD} --height ${HEIGHT}"
fi
if [ -n "${WIDTH}" ]; then
  CMD="${CMD} --width ${WIDTH}"
fi

# Run
eval $CMD
