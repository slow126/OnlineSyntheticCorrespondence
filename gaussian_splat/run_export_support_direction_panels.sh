#!/usr/bin/env bash
set -euo pipefail

# Export direction-only support-bin splats from cached NPZ support vectors.
# No NN recomputation. No color wheel. No endpoint/magnitude panels.

CACHE_ROOT="${CACHE_ROOT:-gaussian_splat/joint_flow_support_vectors}"
RUN_LABEL="${RUN_LABEL:-space_joint__a1p0__t2e1p0__e2t1p5}"
OUT_DIR="${OUT_DIR:-ECCV26___Beyond_Realism__Aligning_Flow_Statistics_for_Dense_Correspondence_Pre_training/figures/eccv26/section4_support_direction_only_v1}"

# Optional space-separated list. Example:
# PAIRS="flyingthings_train__kitti2015_val spair_train__tss_val"
PAIRS="${PAIRS:-}"

# Optional subset of bins. Default = all four bins.
# BINS="train_in_eval train_out_eval eval_in_train eval_out_train"
BINS="${BINS:-}"

HEIGHT="${HEIGHT:-560}"
WIDTH="${WIDTH:-560}"
DIR_MODE="${DIR_MODE:-joint}"  # grid | cluster | joint
K="${K:-800}"
SUBSAMPLE="${SUBSAMPLE:-2000000}"
MAX_RADIUS_PX="${MAX_RADIUS_PX:-64}"
SOFT_EDGE="${SOFT_EDGE:-0.15}"
SUPPORT_SIGMA="${SUPPORT_SIGMA:-3.0}"

CMD=(
  python3 gaussian_splat/export_direction_panels_from_support_cache.py
  --cache-root "${CACHE_ROOT}"
  --run-label "${RUN_LABEL}"
  --out-dir "${OUT_DIR}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --dir-mode "${DIR_MODE}"
  --K "${K}"
  --subsample "${SUBSAMPLE}"
  --max-radius-px "${MAX_RADIUS_PX}"
  --soft-edge "${SOFT_EDGE}"
  --support-sigma "${SUPPORT_SIGMA}"
  --no-colorwheel
)

if [[ -n "${PAIRS}" ]]; then
  # shellcheck disable=SC2206
  arr=(${PAIRS})
  CMD+=(--pairs "${arr[@]}")
fi

if [[ -n "${BINS}" ]]; then
  # shellcheck disable=SC2206
  barr=(${BINS})
  CMD+=(--bins "${barr[@]}")
fi

echo "Exporting direction-only support splats"
echo "  cache: ${CACHE_ROOT}/${RUN_LABEL}"
echo "  out:   ${OUT_DIR}"
echo "  mode:  ${DIR_MODE}"
if [[ -n "${PAIRS}" ]]; then
  echo "  pairs: ${PAIRS}"
else
  echo "  pairs: ALL"
fi
if [[ -n "${BINS}" ]]; then
  echo "  bins:  ${BINS}"
else
  echo "  bins:  ALL"
fi

"${CMD[@]}"
