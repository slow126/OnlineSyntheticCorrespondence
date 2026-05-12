#!/usr/bin/env bash
set -euo pipefail

# Batch runner for joint-flow support splats.
# Discovers train/eval flow files and renders the four support bins per pair.

VECTORS_DIR="${VECTORS_DIR:-/mnt/nvme_1tb_b/coverage_vectors}"

# Threshold semantics matching your columns:
# - flow_train_to_eval_eps1px
# - flow_eval_to_train_eps1p5px
TRAIN_TO_EVAL_EPS_PX="${TRAIN_TO_EVAL_EPS_PX:-1.0}"
EVAL_TO_TRAIN_EPS_PX="${EVAL_TO_TRAIN_EPS_PX:-1.5}"

# Space for directed NN distance comparisons.
SPACE="${SPACE:-joint}"           # joint | flow | xy
JOINT_ALPHA="${JOINT_ALPHA:-1.0}" # only used when SPACE=joint

_fmt_num() {
  echo "$1" | sed 's/\./p/g'
}

RUN_LABEL_DEFAULT="space_${SPACE}__a$(_fmt_num "${JOINT_ALPHA}")__t2e$(_fmt_num "${TRAIN_TO_EVAL_EPS_PX}")__e2t$(_fmt_num "${EVAL_TO_TRAIN_EPS_PX}")"
RUN_LABEL="${RUN_LABEL:-${RUN_LABEL_DEFAULT}}"

OUT_DIR="${OUT_DIR:-gaussian_splat/output_joint_flow_splats/${RUN_LABEL}}"
FLOWS_OUT_DIR="${FLOWS_OUT_DIR:-gaussian_splat/joint_flow_support_vectors/${RUN_LABEL}}"

# Rendering / compute knobs.
K="${K:-800}"
SUBSAMPLE="${SUBSAMPLE:-2000000}"
FLOW_BINS="${FLOW_BINS:-512}"
MAX_RADIUS_PX="${MAX_RADIUS_PX:-64}"
DIR_MODE="${DIR_MODE:-joint}"
LEGEND_SIDE="${LEGEND_SIDE:-inside}"

# Pair discovery.
TRAIN_REGEX="${TRAIN_REGEX:-.*_train_flow\\.npy$}"
EVAL_REGEX="${EVAL_REGEX:-.*_(test|val)_flow\\.npy$}"
PAIR_REGEX="${PAIR_REGEX:-}"      # optional regex on pair name
MAX_PAIRS="${MAX_PAIRS:-}"        # optional integer cap
DRY_RUN="${DRY_RUN:-0}"           # 1 to preview discovered pairs
OVERWRITE="${OVERWRITE:-0}"       # 1 to rerun already-computed pairs
USE_CPU="${USE_CPU:-0}"           # 1 to force CPU FAISS
CLEAR_OUTPUT="${CLEAR_OUTPUT:-0}" # 1 to delete current OUT_DIR/FLOWS_OUT_DIR before running
NO_MONTAGE="${NO_MONTAGE:-0}"     # 1 to disable per-pair montage image
ALLOW_CROSS_DUPLICATES="${ALLOW_CROSS_DUPLICATES:-0}" # 1 to allow zero-distance cross matches

# FAISS search settings
INDEX_FACTORY="${INDEX_FACTORY:-IVF1024,Flat}"  # Flat | IVF...,Flat
NPROBE="${NPROBE:-64}"                        # only used for IVF
BATCH_SIZE="${BATCH_SIZE:-}"            # optional search batch size

# Optional caps for faster iteration/debug
MAX_TRAIN_VECTORS="${MAX_TRAIN_VECTORS:-}"
MAX_EVAL_VECTORS="${MAX_EVAL_VECTORS:-}"

CMD=(
  python3 gaussian_splat/visualize_joint_flow_support_splats.py
  --vectors-dir "${VECTORS_DIR}"
  --out-dir "${OUT_DIR}"
  --flows-out-dir "${FLOWS_OUT_DIR}"
  --space "${SPACE}"
  --joint-alpha "${JOINT_ALPHA}"
  --train-to-eval-eps-px "${TRAIN_TO_EVAL_EPS_PX}"
  --eval-to-train-eps-px "${EVAL_TO_TRAIN_EPS_PX}"
  --train-regex "${TRAIN_REGEX}"
  --eval-regex "${EVAL_REGEX}"
  --index-factory "${INDEX_FACTORY}"
  --K "${K}"
  --subsample "${SUBSAMPLE}"
  --flow_bins "${FLOW_BINS}"
  --max_radius_px "${MAX_RADIUS_PX}"
  --dir_mode "${DIR_MODE}"
  --legend-side "${LEGEND_SIDE}"
  --run-label "${RUN_LABEL}"
)

if [[ -n "${PAIR_REGEX}" ]]; then
  CMD+=(--pair-regex "${PAIR_REGEX}")
fi
if [[ -n "${MAX_PAIRS}" ]]; then
  CMD+=(--max-pairs "${MAX_PAIRS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  CMD+=(--overwrite)
fi
if [[ "${USE_CPU}" == "1" ]]; then
  CMD+=(--cpu)
fi
if [[ -n "${NPROBE}" ]]; then
  CMD+=(--nprobe "${NPROBE}")
fi
if [[ -n "${BATCH_SIZE}" ]]; then
  CMD+=(--batch-size "${BATCH_SIZE}")
fi
if [[ -n "${MAX_TRAIN_VECTORS}" ]]; then
  CMD+=(--max-train-vectors "${MAX_TRAIN_VECTORS}")
fi
if [[ -n "${MAX_EVAL_VECTORS}" ]]; then
  CMD+=(--max-eval-vectors "${MAX_EVAL_VECTORS}")
fi
if [[ "${NO_MONTAGE}" == "1" ]]; then
  CMD+=(--no-montage)
fi
if [[ "${ALLOW_CROSS_DUPLICATES}" == "1" ]]; then
  CMD+=(--allow-cross-duplicates)
else
  CMD+=(--exclude-cross-duplicates)
fi

echo "Running joint-flow support splats"
echo "  vectors: ${VECTORS_DIR}"
echo "  run-label: ${RUN_LABEL}"
echo "  out:     ${OUT_DIR}"
echo "  vectors-out: ${FLOWS_OUT_DIR}"
echo "  eps:     train->eval=${TRAIN_TO_EVAL_EPS_PX}, eval->train=${EVAL_TO_TRAIN_EPS_PX}"
echo "  space:   ${SPACE} (alpha=${JOINT_ALPHA})"
echo "  faiss:   ${INDEX_FACTORY} ${NPROBE:+(nprobe=${NPROBE})}"
echo "  device:  $([[ \"${USE_CPU}\" == \"1\" ]] && echo CPU || echo GPU)"
echo "  cross-duplicates: $([[ \"${ALLOW_CROSS_DUPLICATES}\" == \"1\" ]] && echo allowed || echo filtered)"
echo "  clear-output: ${CLEAR_OUTPUT}"
echo "  montage: $([[ \"${NO_MONTAGE}\" == \"1\" ]] && echo off || echo on)"
echo

if [[ "${CLEAR_OUTPUT}" == "1" ]]; then
  echo "Clearing output directories..."
  rm -rf "${OUT_DIR}" "${FLOWS_OUT_DIR}"
fi

"${CMD[@]}"
