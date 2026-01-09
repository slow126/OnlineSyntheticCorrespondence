#!/usr/bin/env bash
set -euo pipefail

# In-sample mixed-effects sweep for plot3d.py (DINO-only predictors).
# Override any of these via environment variables when running:
#   SNAPSHOT_DIRS="snapshots,snapshots_mixed,snapshots_raft"
#   FLOW_COVERAGE="coverage_faiss_flow_results_fast.csv"
#   DINO_COVERAGE="coverage_faiss_dino_results_fast.csv"
#   FLOW_MMD="flow_mmd_results_fast.csv"
#   DINO_MMD="dino_mmd_results_fast.csv"
#   OUTPUT_ROOT="insample_analysis/sweeps"

SNAPSHOT_DIRS="${SNAPSHOT_DIRS:-snapshots,snapshots_mixed,snapshots_raft}"
FLOW_COVERAGE="${FLOW_COVERAGE:-coverage_faiss_flow_results_fast.csv}"
DINO_COVERAGE="${DINO_COVERAGE:-coverage_faiss_dino_results_fast.csv}"
FLOW_MMD="${FLOW_MMD:-flow_mmd_results_fast.csv}"
DINO_MMD="${DINO_MMD:-dino_mmd_results_fast.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-insample_analysis/sweeps}"

STANDARDIZE_MODES=(global none benchmark encoder model_family)
PREDICTOR_SETS=(all trimmed)
EFFECTS_MODES=(none effects interactions both)

mkdir -p "${OUTPUT_ROOT}"

for std_mode in "${STANDARDIZE_MODES[@]}"; do
  for pred_set in "${PREDICTOR_SETS[@]}"; do
    for effects_mode in "${EFFECTS_MODES[@]}"; do
      suffix="_${std_mode}_${pred_set}_${effects_mode}"
      output_dir="${OUTPUT_ROOT}/${std_mode}/${pred_set}/${effects_mode}"
      mkdir -p "${output_dir}"

      extra_flags=()
      case "${effects_mode}" in
        effects)
          extra_flags+=(--encoder-offsets --model-family-offsets)
          ;;
        interactions)
          extra_flags+=(--encoder-interactions --model-family-interactions)
          ;;
        both)
          extra_flags+=(
            --encoder-offsets --model-family-offsets
            --encoder-interactions --model-family-interactions
          )
          ;;
      esac

      echo "Running: std=${std_mode} predictors=${pred_set} effects=${effects_mode}"
      python plot3d.py \
        --snapshots-dirs "${SNAPSHOT_DIRS}" \
        --coverage-csv "${FLOW_COVERAGE}" \
        --coverage-dino-csv "${DINO_COVERAGE}" \
        --flow-mmd-csv "${FLOW_MMD}" \
        --dino-mmd-csv "${DINO_MMD}" \
        --standardize-mode "${std_mode}" \
        --predictor-set "${pred_set}" \
        --analysis-suffix "${suffix}" \
        --output-dir "${output_dir}" \
        "${extra_flags[@]}"
    done
  done
done
