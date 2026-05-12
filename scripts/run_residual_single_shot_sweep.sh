#!/usr/bin/env bash
set -euo pipefail

# Residual single-shot transfer sweep:
# y' = theta^T s + beta^T d + delta^T (d ⊙ s)
#
# This script reuses build_leakage_free_eval.py for all CV/report logic and
# runs multiple predictor sets under one output root.
#
# Example:
# bash scripts/run_residual_single_shot_sweep.sh \
#   --input-root analysis \
#   --output-root analysis_comprehensive_runs/residual_single_shot_k1 \
#   --linear-model ridge \
#   --prediction-model ridge \
#   --ridge-alpha 10 \
#   --ranking-group train_dataset \
#   --ranking-context-cols model_family_encoder \
#   --pairwise-group-cols benchmark,model_family_encoder \
#   --cv-residualize-target-by-context \
#   --cv-residual-context-cols benchmark,model_family_encoder \
#   --cv-residual-target-transform zscore \
#   --cv-residual-eval-space residual \
#   --cv-fewshot-context-calibration \
#   --cv-fewshot-context-calibration-cols benchmark,model_family_encoder \
#   --cv-fewshot-context-calibration-k 1 \
#   --cv-fewshot-context-calibration-min-group-size 2 \
#   --cv-fewshot-context-calibration-backoff \
#   --cv-repeat-aggregation median \
#   --fit-sample-weighting inverse_task \
#   --fit-balance-real-synth \
#   --overall-aggregation macro_fold \
#   --joint-ood-holdout \
#   --no-loto-single-predictor-baselines \
#   --no-jointood-single-predictor-baselines \
#   --no-per-encoder \
#   --no-mixedlm

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
FLOW_STATS_DIR="${FLOW_STATS_DIR:-/mnt/nvme_1tb_b/coverage_vectors/stats}"
FLOW_COVERAGE_CSV="${FLOW_COVERAGE_CSV:-}"
DINO_COVERAGE_CSV="${DINO_COVERAGE_CSV:-}"
HOF_COVERAGE_CSV="${HOF_COVERAGE_CSV:-}"
SENSITIVITY_COLS="${SENSITIVITY_COLS:-log_n_samples_eval,log_avg_flows_eval}"
INTERACTION_SPECS="${INTERACTION_SPECS:-flow_train_to_eval_eps1px*log_n_samples_eval,flow_train_to_eval_eps1px*log_avg_flows_eval,dino_eval_to_train_mean_dist*log_n_samples_eval}"
DRY_RUN=0
NO_MIXEDLM=0
NO_DENSITY_CONTROLS=0

FORWARD_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-root)
      INPUT_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --flow-stats-dir)
      FLOW_STATS_DIR="$2"
      shift 2
      ;;
    --flow-coverage-csv)
      FLOW_COVERAGE_CSV="$2"
      shift 2
      ;;
    --dino-coverage-csv)
      DINO_COVERAGE_CSV="$2"
      shift 2
      ;;
    --hof-coverage-csv)
      HOF_COVERAGE_CSV="$2"
      shift 2
      ;;
    --sensitivity-cols)
      SENSITIVITY_COLS="$2"
      shift 2
      ;;
    --interaction-specs)
      INTERACTION_SPECS="$2"
      shift 2
      ;;
    --no-family-effects)
      # Compatibility with run_comprehensive_sweep_latest.sh.
      shift 1
      ;;
    --no-density-controls)
      NO_DENSITY_CONTROLS=1
      shift 1
      ;;
    --density-controls)
      NO_DENSITY_CONTROLS=0
      shift 1
      ;;
    --no-mixedlm)
      NO_MIXEDLM=1
      shift 1
      ;;
    --mixedlm)
      NO_MIXEDLM=0
      shift 1
      ;;
    --pairwise-all|--no-pairwise-all)
      # Pairwise reruns are not managed in this script.
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    *)
      FORWARD_ARGS+=("$1")
      shift 1
      ;;
  esac
done

trim_ws() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "${v}"
}

csv_to_array() {
  local csv="$1"
  local -n out_ref="$2"
  out_ref=()
  IFS=',' read -r -a _tmp <<< "${csv}"
  local t
  for t in "${_tmp[@]}"; do
    t="$(trim_ws "${t}")"
    if [[ -n "${t}" ]]; then
      out_ref+=("${t}")
    fi
  done
}

join_csv() {
  local -n arr_ref="$1"
  local IFS=','
  echo "${arr_ref[*]}"
}

if [[ -z "${OUTPUT_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="analysis_comprehensive_runs/residual_single_shot_${TS}"
fi

if [[ -z "${FLOW_COVERAGE_CSV}" ]]; then
  FLOW_COVERAGE_CSV="${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
fi
if [[ -z "${DINO_COVERAGE_CSV}" ]]; then
  DINO_COVERAGE_CSV="${INPUT_ROOT}/coverage_v2_dino_full_fast.csv"
fi
if [[ -z "${HOF_COVERAGE_CSV}" ]]; then
  HOF_COVERAGE_CSV="${INPUT_ROOT}/coverage_v2_hof_full.csv"
fi
if [[ ! -f "${HOF_COVERAGE_CSV}" ]]; then
  HOF_FALLBACK="${INPUT_ROOT}/coverage_v2_hof_full_occ.csv"
  if [[ -f "${HOF_FALLBACK}" ]]; then
    HOF_COVERAGE_CSV="${HOF_FALLBACK}"
  fi
fi

if [[ ! -f "${FLOW_COVERAGE_CSV}" ]]; then
  echo "Missing flow coverage CSV: ${FLOW_COVERAGE_CSV}" >&2
  exit 1
fi

DENSITY_OUT="${OUTPUT_ROOT}/density_joint"
mkdir -p "${DENSITY_OUT}"

SNAPSHOT_DIRS=(
  /mnt/nvme_1tb_b/snapshots_ptody_fix
  /mnt/nvme_1tb_b/snapshots_synth_2d
  /mnt/nvme_1tb_b/snapshots_synthetic_long
  ./snapshots_2d_warps
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_mixed
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft_2d_mix
  /home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_spair_only
)

COMMON_ARGS=(
  --snapshots-dir "${SNAPSHOT_DIRS[@]}"
  --output-dir "${DENSITY_OUT}/__placeholder__"
  --coverage-csv "${FLOW_COVERAGE_CSV}"
  --target auc_normalized_observed
  --flow-stats-dir "${FLOW_STATS_DIR}"
  --cv-standardize-mode local
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --no-allow-unsplit-flow-stats
  --no-logit-coverage
  --no-flow-eps-predictors
  --no-flow-eps-weighted-predictors
  --no-encoder-main-effects
  --no-encoder-interactions
  --no-model-family-main-effects
  --no-model-family-interactions
  --no-collapse-cv-cells
)

if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
  COMMON_ARGS+=(--coverage-dino-csv "${DINO_COVERAGE_CSV}")
else
  echo "Warning: missing dino coverage CSV (${DINO_COVERAGE_CSV}); dino modes will be skipped."
fi
if [[ -f "${HOF_COVERAGE_CSV}" ]]; then
  COMMON_ARGS+=(--coverage-hof-csv "${HOF_COVERAGE_CSV}")
else
  echo "Warning: missing hof coverage CSV (${HOF_COVERAGE_CSV}); hof modes will be skipped."
fi

COMMON_ARGS+=("${FORWARD_ARGS[@]}")
if [[ "${NO_MIXEDLM}" -eq 1 ]]; then
  COMMON_ARGS+=(--no-prediction-mixedlm --no-regression-mixedlm)
fi

declare -a SENS_COLS
csv_to_array "${SENSITIVITY_COLS}" SENS_COLS

declare -a INTER_SPECS
csv_to_array "${INTERACTION_SPECS}" INTER_SPECS

if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
  echo "Warning: --no-density-controls is ignored for this script; residual single-shot requires sensitivity terms."
fi

run_mode() {
  local mode_name="$1"
  local distance_csv="$2"

  declare -a DIST_COLS
  csv_to_array "${distance_csv}" DIST_COLS

  declare -A present=()
  local col
  for col in "${DIST_COLS[@]}"; do
    present["${col}"]=1
  done
  for col in "${SENS_COLS[@]}"; do
    present["${col}"]=1
  done

  declare -a applicable_specs=()
  declare -a interaction_cols=()
  local spec spec_core left right
  for spec in "${INTER_SPECS[@]}"; do
    spec_core="${spec%%@*}"
    if [[ "${spec_core}" == *"*"* ]]; then
      left="${spec_core%%\**}"
      right="${spec_core#*\*}"
    elif [[ "${spec_core}" == *":"* ]]; then
      left="${spec_core%%:*}"
      right="${spec_core#*:}"
    else
      continue
    fi
    left="$(trim_ws "${left}")"
    right="$(trim_ws "${right}")"
    if [[ -n "${present[${left}]+x}" && -n "${present[${right}]+x}" ]]; then
      applicable_specs+=("${spec}")
      interaction_cols+=("${left}_x_${right}")
    fi
  done

  declare -a preds=()
  declare -A seen=()
  local p
  for p in "${DIST_COLS[@]}" "${SENS_COLS[@]}" "${interaction_cols[@]}"; do
    if [[ -z "${p}" ]]; then
      continue
    fi
    if [[ -z "${seen[${p}]+x}" ]]; then
      seen["${p}"]=1
      preds+=("${p}")
    fi
  done

  local predictors_csv
  predictors_csv="$(join_csv preds)"
  local custom_interactions_csv
  custom_interactions_csv="$(join_csv applicable_specs)"
  local out_dir="${DENSITY_OUT}/leakage_free_residual_single_shot__${mode_name}"

  local cmd=(
    python scripts/build_leakage_free_eval.py
    "${COMMON_ARGS[@]}"
    --output-dir "${out_dir}"
    --predictors "${predictors_csv}"
  )
  if [[ -n "${custom_interactions_csv}" ]]; then
    cmd+=(--custom-interactions "${custom_interactions_csv}")
  fi

  echo "Running mode: ${mode_name}"
  echo "  output: ${out_dir}"
  echo "  predictors: ${predictors_csv}"
  if [[ -n "${custom_interactions_csv}" ]]; then
    echo "  interactions: ${custom_interactions_csv}"
  fi
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    printf '  cmd:'
    printf ' %q' "${cmd[@]}"
    echo
    return 0
  fi
  "${cmd[@]}"
}

run_mode "flow_eps_raw_single" \
  "flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px"

if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
  run_mode "dino_eval_dist" \
    "dino_eval_to_train_mean_dist"
fi

if [[ -f "${HOF_COVERAGE_CSV}" ]]; then
  run_mode "hof_train_dist" \
    "hof_train_to_eval_mean_dist"
fi

if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
  run_mode "flow_plus_dino" \
    "flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px,dino_eval_to_train_mean_dist"
fi

if [[ -f "${HOF_COVERAGE_CSV}" ]]; then
  run_mode "flow_plus_hof" \
    "flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px,hof_train_to_eval_mean_dist"
fi

if [[ -f "${DINO_COVERAGE_CSV}" && -f "${HOF_COVERAGE_CSV}" ]]; then
  run_mode "flow_plus_dino_plus_hof" \
    "flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px,dino_eval_to_train_mean_dist,hof_train_to_eval_mean_dist"
fi

echo "Residual single-shot sweep complete: ${DENSITY_OUT}"
