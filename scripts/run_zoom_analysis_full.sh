#!/usr/bin/env bash
set -euo pipefail

# Full zoom intervention runner.
# Purpose:
# - Run leakage-free eval with a matrix of zoom-focused predictor sets
# - Keep matched predictor-count tiers across modalities (k1/k2/k3/k6)
# - Filter to synthetic zoom train datasets for zoom-only analysis
# - Run analyze_zoom_variants.py per experiment

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
REUSE_ROOT="${REUSE_ROOT:-}"

FLOW_COVERAGE_CSV="${FLOW_COVERAGE_CSV:-${INPUT_ROOT}/coverage_v2_flow_only_raw_full.csv}"
DINO_COVERAGE_CSV="${DINO_COVERAGE_CSV:-${INPUT_ROOT}/coverage_v2_dino_full_fast.csv}"
HOF_COVERAGE_CSV="${HOF_COVERAGE_CSV:-${INPUT_ROOT}/coverage_v2_hof_full_occ.csv}"
HOF_KL_CSV="${HOF_KL_CSV:-${INPUT_ROOT}/kl_v2_hof_full_occ.csv}"

FLOW_MMD_V2="${FLOW_MMD_V2:-${INPUT_ROOT}/mmd_v2_flow_joint.csv}"
DINO_MMD_V2="${DINO_MMD_V2:-${INPUT_ROOT}/mmd_v2_dino.csv}"

FLOW_STATS_DIR="${FLOW_STATS_DIR:-/mnt/nvme_1tb_b/coverage_vectors/stats}"
TARGET_METRIC="${TARGET_METRIC:-auc_normalized_observed}"
PERF_METRIC="${PERF_METRIC:-auc_normalized_observed}"
RANKING_GROUP="${RANKING_GROUP:-train_dataset}"
ANALYSIS_MODE="${ANALYSIS_MODE:-intervention}"

USE_DENSITY_CONTROLS=1
PAIRWISE_ALL=0
DISABLE_MIXEDLM=1
SKIP_EVAL=0
SKIP_ANALYZE=0
EXPERIMENT_FILTER=""
EXPERIMENT_PRESET="${EXPERIMENT_PRESET:-intervention}"
REUSE_METADATA_FILES=""

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
    --reuse-root)
      REUSE_ROOT="$2"
      shift 2
      ;;
    --flow-coverage)
      FLOW_COVERAGE_CSV="$2"
      shift 2
      ;;
    --dino-coverage)
      DINO_COVERAGE_CSV="$2"
      shift 2
      ;;
    --hof-coverage)
      HOF_COVERAGE_CSV="$2"
      shift 2
      ;;
    --hof-kl)
      HOF_KL_CSV="$2"
      shift 2
      ;;
    --flow-mmd-v2)
      FLOW_MMD_V2="$2"
      shift 2
      ;;
    --dino-mmd-v2)
      DINO_MMD_V2="$2"
      shift 2
      ;;
    --flow-stats-dir)
      FLOW_STATS_DIR="$2"
      shift 2
      ;;
    --target-metric)
      TARGET_METRIC="$2"
      shift 2
      ;;
    --perf-metric)
      PERF_METRIC="$2"
      shift 2
      ;;
    --ranking-group)
      RANKING_GROUP="$2"
      shift 2
      ;;
    --analysis-mode)
      ANALYSIS_MODE="$2"
      shift 2
      ;;
    --experiments)
      EXPERIMENT_FILTER="$2"
      shift 2
      ;;
    --preset)
      EXPERIMENT_PRESET="$2"
      shift 2
      ;;
    --skip-eval)
      SKIP_EVAL=1
      shift 1
      ;;
    --skip-analyze)
      SKIP_ANALYZE=1
      shift 1
      ;;
    --pairwise-all)
      PAIRWISE_ALL=1
      shift 1
      ;;
    --no-pairwise-all)
      PAIRWISE_ALL=0
      shift 1
      ;;
    --mixedlm)
      DISABLE_MIXEDLM=0
      shift 1
      ;;
    --no-mixedlm)
      DISABLE_MIXEDLM=1
      shift 1
      ;;
    --use-density-controls)
      USE_DENSITY_CONTROLS=1
      shift 1
      ;;
    --no-density-controls)
      USE_DENSITY_CONTROLS=0
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

case "${EXPERIMENT_PRESET}" in
  all|intervention|negative_controls)
    ;;
  *)
    echo "Unknown --preset value: ${EXPERIMENT_PRESET} (expected: all|intervention|negative_controls)" >&2
    exit 1
    ;;
esac

case "${ANALYSIS_MODE}" in
  full|intervention)
    ;;
  *)
    echo "Unknown --analysis-mode value: ${ANALYSIS_MODE} (expected: full|intervention)" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="analysis_zoom_runs/${TS}"
fi
mkdir -p "${OUTPUT_ROOT}"

if [[ -n "${REUSE_ROOT}" ]]; then
  if [[ ! -d "${REUSE_ROOT}" ]]; then
    echo "Reuse root does not exist: ${REUSE_ROOT}" >&2
    exit 1
  fi
  REUSE_METADATA_FILES="$(rg --files "${REUSE_ROOT}" | rg 'run_metadata.json$' || true)"
  if [[ -z "${REUSE_METADATA_FILES}" ]]; then
    echo "Warning: no run_metadata.json found under reuse root: ${REUSE_ROOT}" >&2
  else
    n_meta="$(echo "${REUSE_METADATA_FILES}" | sed '/^$/d' | wc -l | tr -d ' ')"
    echo "Reuse mode: indexed ${n_meta} metadata files under ${REUSE_ROOT}"
  fi
fi

MMD_OUT="${OUTPUT_ROOT}/mmd"
mkdir -p "${MMD_OUT}"
FLOW_MMD_V1="${MMD_OUT}/flow_mmd_v1.csv"
DINO_MMD_V1="${MMD_OUT}/dino_mmd_v1.csv"

ZOOM_VARIANTS="synthetic synthetic_large_zoom synthetic_small_zoom synthetic_random_flipping"
ZOOM_VARIANTS_CSV="synthetic,synthetic_large_zoom,synthetic_small_zoom,synthetic_random_flipping"

convert_mmd_to_v1() {
  local in_path="$1"
  local out_path="$2"
  if [[ ! -f "${in_path}" ]]; then
    echo "Missing MMD input: ${in_path}" >&2
    return 1
  fi
  python scripts/convert_mmd_v2_to_v1.py --input "${in_path}" --output "${out_path}"
}

canonical_csv() {
  local raw="$1"
  echo "${raw}" \
    | tr ',' '\n' \
    | sed '/^[[:space:]]*$/d' \
    | sed 's/^ *//; s/ *$//' \
    | sort \
    | paste -sd, -
}

metadata_base_key() {
  local meta="$1"
  jq -r '
    [
      .predictors[]
      | select(. != "log_n_samples_eval")
      | select(. != "log_avg_flows_eval")
      | select(. != "log_n_samples_train")
      | select(. != "log_avg_flows_train")
      | select(startswith("enc_") | not)
      | select(startswith("mf_") | not)
    ]
    | sort
    | join(",")
  ' "${meta}" 2>/dev/null || true
}

find_reuse_dir_for_predictors() {
  local predictors="$1"
  local target_key
  local meta
  local dir
  local key
  local fallback_pairwise=""
  target_key="$(canonical_csv "${predictors}")"
  [[ -n "${REUSE_METADATA_FILES}" ]] || return 1
  while IFS= read -r meta; do
    [[ -f "${meta}" ]] || continue
    dir="$(dirname "${meta}")"
    [[ -f "${dir}/auc_with_features.csv" ]] || continue
    key="$(metadata_base_key "${meta}")"
    [[ "${key}" == "${target_key}" ]] || continue
    if [[ "${dir}" != *_pairwise ]]; then
      echo "${dir}"
      return 0
    fi
    if [[ -z "${fallback_pairwise}" ]]; then
      fallback_pairwise="${dir}"
    fi
  done <<< "${REUSE_METADATA_FILES}"
  if [[ -n "${fallback_pairwise}" ]]; then
    echo "${fallback_pairwise}"
    return 0
  fi
  return 1
}

filter_zoom_perf() {
  local in_csv="$1"
  local out_csv="$2"
  if [[ ! -f "${in_csv}" ]]; then
    echo "Missing perf CSV: ${in_csv}" >&2
    return 1
  fi
  PERF_IN="${in_csv}" PERF_OUT="${out_csv}" python - <<'PY'
import os
import pandas as pd

in_csv = os.environ["PERF_IN"]
out_csv = os.environ["PERF_OUT"]
keep = {
    "synthetic",
    "synthetic_large_zoom",
    "synthetic_small_zoom",
    "synthetic_random_flipping",
}
df = pd.read_csv(in_csv)
if "train_dataset" not in df.columns:
    raise SystemExit(f"Missing train_dataset column in {in_csv}")
df["train_dataset"] = df["train_dataset"].astype(str).str.lower()
out = df[df["train_dataset"].isin(keep)].copy()
out.to_csv(out_csv, index=False)
print(f"Wrote {out_csv} ({len(out)} rows)")
PY
}

resolve_hof_predictor_mode() {
  local preds="$1"
  if [[ "${preds}" == *"hof_density_"* && "${preds}" == *"hof_train_to_eval_mean_dist"* ]]; then
    echo "combined"
    return
  fi
  if [[ "${preds}" == *"hof_density_"* ]]; then
    echo "density"
    return
  fi
  if [[ "${preds}" == *"hof_train_to_eval_mean_dist"* || "${preds}" == *"hof_eval_to_train_mean_dist"* ]]; then
    echo "motion"
    return
  fi
  echo "combined"
}

run_eval() {
  local args=("$@")
  local out_dir=""
  local i
  local has_linear=0
  local has_pred=0

  for ((i=0; i<${#args[@]}; i++)); do
    if [[ "${args[i]}" == "--output-dir" ]]; then
      out_dir="${args[i+1]:-}"
    fi
    if [[ "${args[i]}" == "--linear-model" ]]; then
      has_linear=1
    fi
    if [[ "${args[i]}" == "--prediction-model" ]]; then
      has_pred=1
    fi
  done

  if [[ "${DISABLE_MIXEDLM}" -eq 1 ]]; then
    args+=(--no-prediction-mixedlm --no-regression-mixedlm)
  fi

  python scripts/build_leakage_free_eval.py "${args[@]}"

  if [[ "${PAIRWISE_ALL}" -ne 1 || -z "${out_dir}" ]]; then
    return 0
  fi

  local pair_args=("${args[@]}")
  for ((i=0; i<${#pair_args[@]}; i++)); do
    if [[ "${pair_args[i]}" == "--output-dir" ]]; then
      pair_args[i+1]="${out_dir}_pairwise"
      break
    fi
  done
  if [[ "${has_linear}" -eq 0 ]]; then
    pair_args+=(--linear-model pairwise_rank)
  fi
  if [[ "${has_pred}" -eq 0 ]]; then
    pair_args+=(--prediction-model pairwise_rank)
  fi
  if [[ "${DISABLE_MIXEDLM}" -eq 1 ]]; then
    pair_args+=(--no-prediction-mixedlm --no-regression-mixedlm)
  fi
  python scripts/build_leakage_free_eval.py "${pair_args[@]}"
}

should_run_experiment() {
  local name="$1"
  if [[ -z "${EXPERIMENT_FILTER}" ]]; then
    return 0
  fi
  IFS=',' read -ra wanted <<< "${EXPERIMENT_FILTER}"
  for w in "${wanted[@]}"; do
    if [[ "${name}" == "${w}" ]]; then
      return 0
    fi
  done
  return 1
}

needs_missing_source() {
  local preds="$1"
  if [[ "${preds}" == *"flow_"* && ! -f "${FLOW_COVERAGE_CSV}" ]]; then
    echo "flow coverage (${FLOW_COVERAGE_CSV})"
    return 0
  fi
  if [[ "${preds}" == *"dino_"* && ! -f "${DINO_COVERAGE_CSV}" ]]; then
    echo "dino coverage (${DINO_COVERAGE_CSV})"
    return 0
  fi
  if [[ "${preds}" == *"hof_"* && ! -f "${HOF_COVERAGE_CSV}" ]]; then
    echo "hof coverage (${HOF_COVERAGE_CSV})"
    return 0
  fi
  return 1
}

if [[ "${SKIP_EVAL}" -eq 0 || "${SKIP_ANALYZE}" -eq 0 ]]; then
  convert_mmd_to_v1 "${FLOW_MMD_V2}" "${FLOW_MMD_V1}"
  convert_mmd_to_v1 "${DINO_MMD_V2}" "${DINO_MMD_V1}"
fi

COMMON_ARGS=(
  --snapshots-dir "${SNAPSHOT_DIRS[@]}"
  --coverage-csv "${FLOW_COVERAGE_CSV}"
  --coverage-dino-csv "${DINO_COVERAGE_CSV}"
  --coverage-hof-csv "${HOF_COVERAGE_CSV}"
  --flow-mmd-csv "${FLOW_MMD_V1}"
  --dino-mmd-csv "${DINO_MMD_V1}"
  --target "${TARGET_METRIC}"
  --no-flow-eps-predictors
  --no-flow-eps-weighted-predictors
  --no-logit-coverage
  --no-include-kl
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --cv-standardize-mode local
  --ranking-group "${RANKING_GROUP}"
)

if [[ "${USE_DENSITY_CONTROLS}" -eq 1 ]]; then
  if [[ -d "${FLOW_STATS_DIR}" ]]; then
    COMMON_ARGS+=(--flow-stats-dir "${FLOW_STATS_DIR}" --use-flow-density-predictors --no-flow-density-interactions)
  else
    echo "Flow stats dir missing, disabling density controls: ${FLOW_STATS_DIR}" >&2
  fi
fi

EXPERIMENT_SPECS_ALL=(
  "flow_k1|flow_train_to_eval_eps1px|k1"
  "dino_k1|dino_train_to_eval_mean_dist|k1"
  "hof_k1|hof_train_to_eval_mean_dist|k1"
  "flow_k2|flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px|k2"
  "dino_k2|dino_train_to_eval_mean_dist,dino_eval_to_train_mean_dist|k2"
  "hof_k2|hof_train_to_eval_mean_dist,hof_eval_to_train_mean_dist|k2"
  "hof_density_k1|hof_density_l2|k1"
  "flow_dino_hof_k3|flow_train_to_eval_eps1px,dino_train_to_eval_mean_dist,hof_train_to_eval_mean_dist|k3"
  "flow_dino_hof_k6|flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px,dino_train_to_eval_mean_dist,dino_eval_to_train_mean_dist,hof_train_to_eval_mean_dist,hof_eval_to_train_mean_dist|k6"
  "hof_motion_density_k3|hof_train_to_eval_mean_dist,hof_eval_to_train_mean_dist,hof_density_l2|k3"
)

EXPERIMENT_SPECS=()
case "${EXPERIMENT_PRESET}" in
  intervention)
    EXPERIMENT_SPECS=(
      # Canonical intervention run: one reused eval source is enough because
      # intervention analysis consumes observed performance + coverage proxies.
      "hof_motion_density_p3|hof_train_to_eval_mean_dist,hof_eval_to_train_mean_dist,hof_density_l2|p3"
    )
    ;;
  negative_controls)
    EXPERIMENT_SPECS=(
      "dino_k1|dino_train_to_eval_mean_dist|k1"
      "dino_k2|dino_train_to_eval_mean_dist,dino_eval_to_train_mean_dist|k2"
      "hof_density_k1|hof_density_l2|k1"
    )
    ;;
  all)
    EXPERIMENT_SPECS=("${EXPERIMENT_SPECS_ALL[@]}")
    ;;
esac

MANIFEST_CSV="${OUTPUT_ROOT}/experiment_manifest.csv"
echo "experiment,predictors,matched_param_tier,n_predictors,status,notes" > "${MANIFEST_CSV}"
echo "Preset: ${EXPERIMENT_PRESET} (tier is predictor-count, not NN k)"
echo "Analysis mode: ${ANALYSIS_MODE}"

for spec in "${EXPERIMENT_SPECS[@]}"; do
  IFS='|' read -r exp_name predictors tier <<< "${spec}"

  if ! should_run_experiment "${exp_name}"; then
    continue
  fi

  if missing="$(needs_missing_source "${predictors}")"; then
    echo "Skipping ${exp_name}: missing ${missing}" >&2
    n_predictors="$(awk -F',' '{print NF}' <<< "${predictors}")"
    echo "${exp_name},\"${predictors}\",${tier},${n_predictors},skipped,\"missing ${missing}\"" >> "${MANIFEST_CSV}"
    continue
  fi

  exp_root="${OUTPUT_ROOT}/${exp_name}"
  eval_out="${exp_root}/leakage"
  if [[ "${ANALYSIS_MODE}" == "intervention" ]]; then
    zoom_out="${exp_root}/zoom_intervention"
  else
    zoom_out="${exp_root}/zoom_variants"
  fi
  mkdir -p "${exp_root}" "${zoom_out}"

  n_predictors="$(awk -F',' '{print NF}' <<< "${predictors}")"
  source_eval_out="${eval_out}"
  reused_from=""

  if [[ -n "${REUSE_ROOT}" ]]; then
    if reused_from="$(find_reuse_dir_for_predictors "${predictors}")"; then
      source_eval_out="${reused_from}"
      echo "Reusing ${exp_name} from ${reused_from}"
    fi
  fi

  echo "Running ${exp_name} (tier=${tier}, n=${n_predictors})"
  if [[ -z "${reused_from}" && "${SKIP_EVAL}" -eq 0 ]]; then
    run_eval \
      --output-dir "${eval_out}" \
      --predictors "${predictors}" \
      "${COMMON_ARGS[@]}"
    source_eval_out="${eval_out}"
  fi

  perf_csv="${source_eval_out}/auc_with_features.csv"
  zoom_perf_csv="${zoom_out}/auc_with_features_zoom_only.csv"

  if [[ ! -f "${perf_csv}" ]]; then
    echo "Missing ${perf_csv}; cannot run zoom analysis for ${exp_name}" >&2
    if [[ "${SKIP_EVAL}" -eq 1 ]]; then
      echo "${exp_name},\"${predictors}\",${tier},${n_predictors},skipped,\"missing auc_with_features.csv (skip-eval active; likely no reuse match)\"" >> "${MANIFEST_CSV}"
    else
      echo "${exp_name},\"${predictors}\",${tier},${n_predictors},failed,\"missing auc_with_features.csv\"" >> "${MANIFEST_CSV}"
    fi
    continue
  fi

  filter_zoom_perf "${perf_csv}" "${zoom_perf_csv}"
  if [[ -n "${reused_from}" ]]; then
    printf "%s\n" "${reused_from}" > "${zoom_out}/reused_from.txt"
  fi

  if [[ "${SKIP_ANALYZE}" -eq 0 ]]; then
    hof_predictor_mode="$(resolve_hof_predictor_mode "${predictors}")"
    analyze_args=(
      --perf-csv "${zoom_perf_csv}"
      --flow-coverage-csv "${FLOW_COVERAGE_CSV}"
      --dino-coverage-csv "${DINO_COVERAGE_CSV}"
      --flow-mmd-csv "${FLOW_MMD_V1}"
      --dino-mmd-csv "${DINO_MMD_V1}"
      --output-dir "${zoom_out}"
      --perf-metric "${PERF_METRIC}"
      --variants ${ZOOM_VARIANTS}
      --baseline synthetic
      --predictors "${hof_predictor_mode}"
    )
    if [[ "${ANALYSIS_MODE}" == "intervention" ]]; then
      analyze_args+=(--intervention-only)
    fi

    if [[ -f "${HOF_COVERAGE_CSV}" ]]; then
      analyze_args+=(--hof-coverage-csv "${HOF_COVERAGE_CSV}")
    fi
    if [[ -f "${HOF_KL_CSV}" ]]; then
      analyze_args+=(--hof-kl-csv "${HOF_KL_CSV}")
    fi

    python scripts/analyze_zoom_variants.py "${analyze_args[@]}"
  fi

  if [[ -n "${reused_from}" ]]; then
    echo "${exp_name},\"${predictors}\",${tier},${n_predictors},ok,\"reused_from=${reused_from}; zoom_variants=${ZOOM_VARIANTS_CSV}\"" >> "${MANIFEST_CSV}"
  else
    echo "${exp_name},\"${predictors}\",${tier},${n_predictors},ok,\"zoom_variants=${ZOOM_VARIANTS_CSV}\"" >> "${MANIFEST_CSV}"
  fi
done

echo "Done. Outputs: ${OUTPUT_ROOT}"
echo "Manifest: ${MANIFEST_CSV}"
