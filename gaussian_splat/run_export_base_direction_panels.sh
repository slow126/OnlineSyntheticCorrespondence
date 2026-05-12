#!/usr/bin/env bash
set -euo pipefail

# Export clean per-dataset directional BFV splats (no pairwise support bins).
# Uses cached global flow vectors (e.g., /mnt/nvme_1tb_b/coverage_vectors/*_flow.npy).
#
# Usage examples:
#   bash gaussian_splat/run_export_base_direction_panels.sh
#   SCOPE=train bash gaussian_splat/run_export_base_direction_panels.sh
#   OUT_DIR=gaussian_splat/output_final_base_panels bash gaussian_splat/run_export_base_direction_panels.sh
#
# Environment variables:
#   FLOW_DIR  : directory containing *_flow.npy caches
#   OUT_DIR   : output directory for clean directional panels
#   SCOPE     : train | all
#   DIR_MODE  : grid | cluster | joint | fan (default: cluster)
#   MAX_RADIUS_PX : max splat radius; 0 disables clipping (default: 0)
#   SUPPORT_SIGMA : gaussian support radius in sigmas (default: 5.0)
#   SOFT_EDGE     : edge feathering amount (default: 0.30)
#   DIR_BASE_SIGMA: base directional sigma (default: 3.0)
#   DIR_MAX_SIGMA : max directional sigma cap (default: 48.0)
#   CANVAS_PAD_PX : padding before rendering; -1 auto (default: 0)
#   EDGE_FADE_FRAC: soft edge fade fraction of image size (default: 0.0)
#   FAN_MIN_COUNT : minimum vectors per spatial cluster for fan mode (default: 48)
#   FAN_MAG_Q_LOW : low magnitude quantile for fan band (default: 0.15)
#   FAN_MAG_Q_HIGH: high magnitude quantile for fan band (default: 0.90)
#   FAN_ANGLE_Q   : angular spread quantile for fan mode (default: 0.80)
#   FAN_RADIUS_SCALE: radial scale for fan mode (default: 1.0)
#   FAN_MAX_RADIUS: max fan radius in px (default: 96)

FLOW_DIR="${FLOW_DIR:-/mnt/nvme_1tb_b/coverage_vectors}"
OUT_DIR="${OUT_DIR:-ECCV26___Beyond_Realism__Aligning_Flow_Statistics_for_Dense_Correspondence_Pre_training/figures/eccv26/section3_direction_panels_base}"
SCOPE="${SCOPE:-all}"
DIR_MODE="${DIR_MODE:-cluster}"
MAX_RADIUS_PX="${MAX_RADIUS_PX:-0}"
SUPPORT_SIGMA="${SUPPORT_SIGMA:-5.0}"
SOFT_EDGE="${SOFT_EDGE:-0.30}"
DIR_BASE_SIGMA="${DIR_BASE_SIGMA:-3.0}"
DIR_MAX_SIGMA="${DIR_MAX_SIGMA:-48.0}"
CANVAS_PAD_PX="${CANVAS_PAD_PX:-0}"
EDGE_FADE_FRAC="${EDGE_FADE_FRAC:-0.0}"
FAN_MIN_COUNT="${FAN_MIN_COUNT:-48}"
FAN_MAG_Q_LOW="${FAN_MAG_Q_LOW:-0.15}"
FAN_MAG_Q_HIGH="${FAN_MAG_Q_HIGH:-0.90}"
FAN_ANGLE_Q="${FAN_ANGLE_Q:-0.80}"
FAN_RADIUS_SCALE="${FAN_RADIUS_SCALE:-1.0}"
FAN_MAX_RADIUS="${FAN_MAX_RADIUS:-96}"

TRAIN_DATASETS=(
  synthetic_train
  sintel_train
  pointodyssey_train
  flyingthings_train
  imagenet2dwarp_train
  spair_train
)

BENCHMARK_DATASETS=(
  flyingthings_test
  kitti2012_val
  kitti2015_val
  middlebury_val
  pfpascal_test
  pfwillow_test
  pointodyssey_test
  spair_test
  tss_val
)

DATASETS=()
if [[ "${SCOPE}" == "train" ]]; then
  DATASETS=("${TRAIN_DATASETS[@]}")
elif [[ "${SCOPE}" == "all" ]]; then
  DATASETS=("${TRAIN_DATASETS[@]}" "${BENCHMARK_DATASETS[@]}")
else
  echo "[ERROR] Unsupported SCOPE='${SCOPE}'. Use: train | all" >&2
  exit 1
fi

echo "[INFO] Exporting base directional panels"
echo "[INFO] flow dir : ${FLOW_DIR}"
echo "[INFO] out dir  : ${OUT_DIR}"
echo "[INFO] scope    : ${SCOPE}"
echo "[INFO] dir_mode : ${DIR_MODE}"
echo "[INFO] max_radius_px : ${MAX_RADIUS_PX}"
echo "[INFO] support_sigma : ${SUPPORT_SIGMA}"
echo "[INFO] soft_edge     : ${SOFT_EDGE}"
echo "[INFO] dir_base_sigma: ${DIR_BASE_SIGMA}"
echo "[INFO] dir_max_sigma : ${DIR_MAX_SIGMA}"
echo "[INFO] canvas_pad_px : ${CANVAS_PAD_PX}"
echo "[INFO] edge_fade_frac: ${EDGE_FADE_FRAC}"
if [[ "${DIR_MODE}" == "fan" ]]; then
  echo "[INFO] fan_min_count : ${FAN_MIN_COUNT}"
  echo "[INFO] fan_mag_q_low : ${FAN_MAG_Q_LOW}"
  echo "[INFO] fan_mag_q_high: ${FAN_MAG_Q_HIGH}"
  echo "[INFO] fan_angle_q   : ${FAN_ANGLE_Q}"
  echo "[INFO] fan_radius_scale: ${FAN_RADIUS_SCALE}"
  echo "[INFO] fan_max_radius  : ${FAN_MAX_RADIUS}"
fi
echo "[INFO] datasets : ${#DATASETS[@]}"

python3 gaussian_splat/export_final_direction_panels.py \
  --flow-dir "${FLOW_DIR}" \
  --out-dir "${OUT_DIR}" \
  --dir-mode "${DIR_MODE}" \
  --max-radius-px "${MAX_RADIUS_PX}" \
  --support-sigma "${SUPPORT_SIGMA}" \
  --soft-edge "${SOFT_EDGE}" \
  --dir-base-sigma "${DIR_BASE_SIGMA}" \
  --dir-max-sigma "${DIR_MAX_SIGMA}" \
  --canvas-pad-px "${CANVAS_PAD_PX}" \
  --edge-fade-frac "${EDGE_FADE_FRAC}" \
  --fan-min-count "${FAN_MIN_COUNT}" \
  --fan-mag-q-low "${FAN_MAG_Q_LOW}" \
  --fan-mag-q-high "${FAN_MAG_Q_HIGH}" \
  --fan-angle-q "${FAN_ANGLE_Q}" \
  --fan-radius-scale "${FAN_RADIUS_SCALE}" \
  --fan-max-radius "${FAN_MAX_RADIUS}" \
  --datasets "${DATASETS[@]}"

echo "[DONE] Base directional panel export complete."
