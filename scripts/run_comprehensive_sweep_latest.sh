#!/usr/bin/env bash
set -euo pipefail

# Comprehensive sweep runner:
# - Runs joint density analysis into an isolated output root
# - Converts MMD v2 outputs to v1 format for leakage_free_eval
# - Runs MMD-only and asym+MMD analyses
# - Builds a method summary table

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
DENSITY_ROOT_OVERRIDE=""
MMD_FLOW_V2="${MMD_FLOW_V2:-analysis/mmd_v2_flow_joint.csv}"
MMD_DINO_V2="${MMD_DINO_V2:-analysis/mmd_v2_dino.csv}"
DINO_COVERAGE_CSV="${DINO_COVERAGE_CSV:-${INPUT_ROOT}/coverage_v2_dino_full_fast.csv}"
DINO_COVERAGE_RNORM=""

SKIP_DENSITY=0
SKIP_MMD=0
SKIP_SUMMARY=0
SKIP_COMBOS=0

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
    --density-root)
      DENSITY_ROOT_OVERRIDE="$2"
      shift 2
      ;;
    --mmd-flow-v2)
      MMD_FLOW_V2="$2"
      shift 2
      ;;
    --mmd-dino-v2)
      MMD_DINO_V2="$2"
      shift 2
      ;;
    --dino-coverage)
      DINO_COVERAGE_CSV="$2"
      shift 2
      ;;
    --skip-density)
      SKIP_DENSITY=1
      shift 1
      ;;
    --skip-mmd)
      SKIP_MMD=1
      shift 1
      ;;
    --skip-summary)
      SKIP_SUMMARY=1
      shift 1
      ;;
    --skip-combos)
      SKIP_COMBOS=1
      shift 1
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${OUTPUT_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="analysis_comprehensive_runs/${TS}"
fi

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

DENSITY_ARGS_BASE=(
  --flow-stats-dir /mnt/nvme_1tb_b/coverage_vectors/stats
  --use-flow-density-predictors
  --flow-density-interactions
  --model-family-main-effects
)
DENSITY_ARGS=(
  "${DENSITY_ARGS_BASE[@]}"
  --target auc_normalized_observed
)

FLOW_EPS_VALUES="0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64"

if [[ -n "${DENSITY_ROOT_OVERRIDE}" ]]; then
  DENSITY_OUT="${DENSITY_ROOT_OVERRIDE}"
  SKIP_DENSITY=1
else
  DENSITY_OUT="${OUTPUT_ROOT}/density_joint"
fi
MMD_OUT="${OUTPUT_ROOT}/mmd"
MMD_FLOW_V1="${MMD_OUT}/mmd_v2_flow_joint_v1.csv"
MMD_DINO_V1="${MMD_OUT}/mmd_v2_dino_v1.csv"

build_eps_pred_list() {
  local prefix="$1"
  local suffix="$2"
  local out=""
  IFS=',' read -ra vals <<< "${FLOW_EPS_VALUES}"
  for v in "${vals[@]}"; do
    local v_clean="${v}"
    v_clean="${v_clean%px}"
    local key
    key="$(echo "${v_clean}" | sed 's/\./p/g')"
    out+="${prefix}${key}${suffix},"
  done
  echo "${out%,}"
}

ensure_dino_rnorm_csv() {
  if [[ -z "${DINO_COVERAGE_CSV}" || ! -f "${DINO_COVERAGE_CSV}" ]]; then
    return 0
  fi
  DINO_COVERAGE_RNORM="${OUTPUT_ROOT}/dino_coverage_rnorm_k5.csv"
  if [[ -f "${DINO_COVERAGE_RNORM}" ]]; then
    return 0
  fi
  DINO_COVERAGE_CSV="${DINO_COVERAGE_CSV}" \
  DINO_COVERAGE_RNORM="${DINO_COVERAGE_RNORM}" \
  python - <<'PY'
import pandas as pd
import numpy as np
import os

in_path = os.environ.get("DINO_COVERAGE_CSV")
out_path = os.environ.get("DINO_COVERAGE_RNORM")
if not in_path or not out_path:
    raise SystemExit("Missing DINO_COVERAGE_CSV or DINO_COVERAGE_RNORM env var.")

df = pd.read_csv(in_path)
required = [
    "train_dataset","train_split","eval_dataset","eval_split",
    "train_radius","eval_radius",
    "eval_covered_by_train_rnorm_k5","train_covered_by_eval_rnorm_k5","train_outside_eval_rnorm_k5",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns in DINO coverage CSV: {missing}")

out = pd.DataFrame({
    "space": df.get("space", "features"),
    "train_dataset": df["train_dataset"],
    "train_split": df["train_split"],
    "eval_dataset": df["eval_dataset"],
    "eval_split": df["eval_split"],
    "train_n_vectors": df.get("train_n_vectors", np.nan),
    "eval_n_vectors": df.get("eval_n_vectors", np.nan),
    "train_radius": df["train_radius"],
    "eval_radius": df["eval_radius"],
    "train_to_eval_coverage": df["eval_covered_by_train_rnorm_k5"],
    "eval_to_train_coverage": df["train_covered_by_eval_rnorm_k5"],
    "outside": df["train_outside_eval_rnorm_k5"],
    "k": 5,
    "radius_quantile": np.nan,
})
out.to_csv(out_path, index=False)
print(f"Wrote {out_path} ({len(out)} rows)")
PY
}

mkdir -p "${OUTPUT_ROOT}"

if [[ "${SKIP_DENSITY}" -eq 0 ]]; then
  bash scripts/run_density_analysis_joint_latest.sh \
    --input-root "${INPUT_ROOT}" \
    --output-root "${DENSITY_OUT}"

  RAW_TRAIN_PREDS="$(build_eps_pred_list "flow_train_to_eval_eps" "px")"
  RAW_EVAL_PREDS="$(build_eps_pred_list "flow_eval_to_train_eps" "px")"
  WEIGHTED_TRAIN_PREDS="$(build_eps_pred_list "flow_train_to_eval_eps" "px_weighted")"
  WEIGHTED_EVAL_PREDS="$(build_eps_pred_list "flow_eval_to_train_eps" "px_weighted")"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_train_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${RAW_TRAIN_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eval_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${RAW_EVAL_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_single" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_train_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${WEIGHTED_TRAIN_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_eval_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --no-flow-eps-predictors \
    --predictors "${WEIGHTED_EVAL_PREDS}" \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_train_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_eps_at50 \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_eval_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv" \
    --no-flow-eps-predictors \
    --predictors flow_eval_to_train_eps_at50 \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_train_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_auc \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_eval_only" \
    --coverage-csv "${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv" \
    --no-flow-eps-predictors \
    --predictors flow_eval_to_train_auc \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_train_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_mean_dist_over_radius_eval \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_eval_only" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_eval_to_train_mean_dist_over_radius_train \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}"

  if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
    ensure_dino_rnorm_csv
  fi
  if [[ -f "${DINO_COVERAGE_RNORM}" ]]; then
    DINO_ONLY_ARGS=(
      --snapshots-dir "${SNAPSHOT_DIRS[@]}"
      --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
      --coverage-dino-csv "${DINO_COVERAGE_RNORM}"
      --no-logit-coverage
      --no-flow-eps-predictors
      --no-flow-eps-weighted-predictors
      "${DENSITY_ARGS[@]}"
    )

    python scripts/build_leakage_free_eval.py \
      "${DINO_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_dino_rnorm_k5" \
      --predictors dino_eval_to_train_coverage,dino_train_to_eval_coverage

    python scripts/build_leakage_free_eval.py \
      "${DINO_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_eval_only" \
      --predictors dino_eval_to_train_coverage

    python scripts/build_leakage_free_eval.py \
      "${DINO_ONLY_ARGS[@]}" \
      --output-dir "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_train_only" \
      --predictors dino_train_to_eval_coverage

    python scripts/build_leakage_free_eval.py \
      --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
      --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
      --coverage-dino-csv "${DINO_COVERAGE_RNORM}" \
      --output-dir "${DENSITY_OUT}/leakage_free_flow_eval_only_plus_dino_eval_only" \
      --no-logit-coverage \
      --no-flow-eps-predictors \
      --predictors "${RAW_EVAL_PREDS},dino_eval_to_train_coverage" \
      "${DENSITY_ARGS[@]}"
  else
    echo "Skipping dino analyses (missing: ${DINO_COVERAGE_RNORM})"
  fi

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_target_fixed" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --use-flow-eps-weighted-predictors \
    --no-flow-eps-predictors \
    --no-logit-coverage \
    --target auc_normalized \
    --flow-eps-values "${FLOW_EPS_VALUES}" \
    "${DENSITY_ARGS_BASE[@]}"
fi

if [[ "${SKIP_MMD}" -eq 0 ]]; then
  mkdir -p "${MMD_OUT}"
  if [[ -f "${MMD_FLOW_V2}" ]]; then
    python scripts/convert_mmd_v2_to_v1.py \
      --input "${MMD_FLOW_V2}" \
      --output "${MMD_FLOW_V1}"
  else
    echo "Skipping flow MMD conversion (missing: ${MMD_FLOW_V2})"
  fi
  if [[ -f "${MMD_DINO_V2}" ]]; then
    python scripts/convert_mmd_v2_to_v1.py \
      --input "${MMD_DINO_V2}" \
      --output "${MMD_DINO_V1}"
  else
    echo "Skipping dino MMD conversion (missing: ${MMD_DINO_V2})"
  fi

  BASE_ARGS=(
    --snapshots-dir "${SNAPSHOT_DIRS[@]}"
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
    --no-logit-coverage
    --flow-eps-values "${FLOW_EPS_VALUES}"
    "${DENSITY_ARGS[@]}"
  )
  MMD_ONLY_ARGS=(
    --snapshots-dir "${SNAPSHOT_DIRS[@]}"
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
    --no-logit-coverage
    --target auc_normalized_observed
    --flow-stats-dir /mnt/nvme_1tb_b/coverage_vectors/stats
    --use-flow-density-predictors
    --flow-density-interactions
    --model-family-main-effects
    --no-flow-eps-predictors
    --no-flow-eps-weighted-predictors
  )
  if [[ -f "${MMD_FLOW_V1}" ]]; then
    BASE_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
    MMD_ONLY_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
  fi
  if [[ -f "${MMD_DINO_V1}" ]]; then
    BASE_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
    MMD_ONLY_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
  fi

  if [[ -f "${MMD_FLOW_V1}" || -f "${MMD_DINO_V1}" ]]; then
    python scripts/build_leakage_free_eval.py \
      "${MMD_ONLY_ARGS[@]}" \
      --output-dir "${OUTPUT_ROOT}/mmd_only" \
      --predictors flow_mmd,dino_mmd

    python scripts/build_leakage_free_eval.py \
      "${BASE_ARGS[@]}" \
      --output-dir "${OUTPUT_ROOT}/asym_and_mmd" \
      --use-flow-eps-predictors \
      --predictors flow_mmd,dino_mmd
  else
    echo "Skipping MMD-based analyses (no converted MMD CSVs available)."
  fi
fi

COMBO_METHODS_FILE="${OUTPUT_ROOT}/combo_methods.txt"
: > "${COMBO_METHODS_FILE}"

if [[ "${SKIP_COMBOS}" -eq 0 ]]; then
  mkdir -p "${DENSITY_OUT}"
  if [[ -f "${DINO_COVERAGE_CSV}" ]]; then
    ensure_dino_rnorm_csv
  fi

  DINO_AVAILABLE=0
  if [[ -f "${DINO_COVERAGE_RNORM}" ]]; then
    DINO_AVAILABLE=1
  fi

  MMD_AVAILABLE=0
  if [[ -f "${MMD_FLOW_V1}" || -f "${MMD_DINO_V1}" ]]; then
    MMD_AVAILABLE=1
  fi

  COMBO_BASE_ARGS=(
    --snapshots-dir "${SNAPSHOT_DIRS[@]}"
    --no-logit-coverage
    --flow-eps-values "${FLOW_EPS_VALUES}"
    "${DENSITY_ARGS[@]}"
  )
  if [[ -f "${MMD_FLOW_V1}" ]]; then
    COMBO_BASE_ARGS+=(--flow-mmd-csv "${MMD_FLOW_V1}")
  fi
  if [[ -f "${MMD_DINO_V1}" ]]; then
    COMBO_BASE_ARGS+=(--dino-mmd-csv "${MMD_DINO_V1}")
  fi
  if [[ "${DINO_AVAILABLE}" -eq 1 ]]; then
    COMBO_BASE_ARGS+=(--coverage-dino-csv "${DINO_COVERAGE_RNORM}")
  fi

  join_preds() {
    local out=""
    for part in "$@"; do
      if [[ -n "${part}" ]]; then
        if [[ -n "${out}" ]]; then
          out="${out},${part}"
        else
          out="${part}"
        fi
      fi
    done
    echo "${out}"
  }

  run_combo() {
    local name="$1"
    local coverage_csv="$2"
    local flow_args="$3"
    local predictors="$4"
    local family="$5"
    local symmetry="$6"
    local notes="$7"
    local out_dir="${DENSITY_OUT}/leakage_free_${name}"

    if [[ ! -f "${coverage_csv}" ]]; then
      echo "Skipping combo ${name} (missing coverage csv: ${coverage_csv})"
      return 0
    fi

    if [[ -d "${out_dir}" ]]; then
      echo "${name}|${out_dir}|${family}|${symmetry}|${notes}" >> "${COMBO_METHODS_FILE}"
      return 0
    fi

    local flow_arg_list=()
    if [[ -n "${flow_args}" ]]; then
      read -r -a flow_arg_list <<< "${flow_args}"
    fi

    local args=(
      "${COMBO_BASE_ARGS[@]}"
      --coverage-csv "${coverage_csv}"
      "${flow_arg_list[@]}"
      --output-dir "${out_dir}"
    )
    if [[ -n "${predictors}" ]]; then
      args+=(--predictors "${predictors}")
    fi

    python scripts/build_leakage_free_eval.py "${args[@]}"
    echo "${name}|${out_dir}|${family}|${symmetry}|${notes}" >> "${COMBO_METHODS_FILE}"
  }

  FLOW_VARIANTS=(
    "flow_eps_raw|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv|--use-flow-eps-predictors|"
    "flow_eps_raw_single|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv|--no-flow-eps-predictors|flow_train_to_eval_eps1px,flow_eval_to_train_eps1p5px"
    "flow_eps_raw_eps_at50|${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv|--no-flow-eps-predictors|flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50"
    "flow_eps_raw_auc_at95|${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv|--no-flow-eps-predictors|flow_train_to_eval_auc,flow_eval_to_train_auc"
    "flow_kmeans_weighted|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv|--use-flow-eps-weighted-predictors --no-flow-eps-predictors|"
    "flow_kmeans_manifold|${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv|--no-flow-eps-predictors|flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train"
  )

  DINO_VARIANTS=(
    "none||"
    "dino_rnorm_k5|dino_eval_to_train_coverage,dino_train_to_eval_coverage|"
    "dino_rnorm_k5_eval_only|dino_eval_to_train_coverage|dir=eval_only"
    "dino_rnorm_k5_train_only|dino_train_to_eval_coverage|dir=train_only"
  )
  if [[ "${DINO_AVAILABLE}" -eq 0 ]]; then
    DINO_VARIANTS=("none||")
  fi

  MMD_VARIANTS=(
    "none||"
    "mmd|flow_mmd,dino_mmd|"
  )
  if [[ "${MMD_AVAILABLE}" -eq 0 ]]; then
    MMD_VARIANTS=("none||")
  fi

  for flow_item in "${FLOW_VARIANTS[@]}"; do
    IFS='|' read -r flow_name flow_csv flow_args flow_preds <<< "${flow_item}"
    for dino_item in "${DINO_VARIANTS[@]}"; do
      IFS='|' read -r dino_name dino_preds dino_note <<< "${dino_item}"
      for mmd_item in "${MMD_VARIANTS[@]}"; do
        IFS='|' read -r mmd_name mmd_preds _ <<< "${mmd_item}"

        if [[ "${dino_name}" == "none" && "${mmd_name}" == "none" ]]; then
          continue
        fi

        combo_preds="$(join_preds "${flow_preds}" "${dino_preds}" "${mmd_preds}")"
        if [[ -z "${combo_preds}" ]]; then
          continue
        fi

        local_name="combo_${flow_name}"
        local_notes="combo:flow=${flow_name}"
        local_family="flow"
        local_symmetry="asym"

        if [[ "${dino_name}" != "none" ]]; then
          local_name+="__${dino_name}"
          local_notes+=",dino=${dino_name}"
          local_family="mixed"
        fi
        if [[ "${mmd_name}" != "none" ]]; then
          local_name+="__${mmd_name}"
          local_notes+=",mmd=on"
          local_family="mixed"
          local_symmetry="mixed"
        fi
        if [[ -n "${dino_note}" ]]; then
          local_notes+=",${dino_note}"
        fi

        run_combo "${local_name}" "${flow_csv}" "${flow_args}" "${combo_preds}" "${local_family}" "${local_symmetry}" "${local_notes}"
      done
    done
  done
fi

MANIFEST_PATH="${OUTPUT_ROOT}/method_summary_manifest.yaml"
{
  echo "methods:"
  add_method() {
    local name="$1"
    local path="$2"
    local family="$3"
    local symmetry="$4"
    local notes="${5:-}"
    if [[ -d "${path}" ]]; then
      echo "  - name: ${name}"
      echo "    path: ${path}"
      echo "    family: ${family}"
      echo "    symmetry: ${symmetry}"
      if [[ -n "${notes}" ]]; then
        echo "    notes: ${notes}"
      fi
    else
      echo "Skipping manifest entry (missing dir): ${path}" >&2
    fi
  }

  add_method "flow_eps_raw_joint" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint" "flow" "asym"
  add_method "flow_eps_raw_single" "${DENSITY_OUT}/leakage_free_flow_eps_raw_single" "flow" "asym" "single_eps"
  add_method "flow_eps_raw_joint_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_eps_raw_joint_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_eps_raw_joint_pairwise" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_pairwise" "flow" "asym"
  add_method "flow_eps_raw_joint_auc_at95" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95" "flow" "asym"
  add_method "flow_eps_raw_joint_auc_at95_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_eps_raw_joint_auc_at95_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_eps_raw_joint_auc_at95_pairwise" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_auc_at95_pairwise" "flow" "asym"
  add_method "flow_eps_raw_joint_eps_at50" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50" "flow" "asym"
  add_method "flow_eps_raw_joint_eps_at50_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_eps_raw_joint_eps_at50_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_eps_raw_joint_eps_at50_pairwise" "${DENSITY_OUT}/leakage_free_flow_eps_raw_joint_eps_at50_pairwise" "flow" "asym"
  add_method "flow_kmeans_weighted_all" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all" "flow" "asym"
  add_method "flow_kmeans_weighted_all_train_only" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_kmeans_weighted_all_eval_only" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_kmeans_weighted_all_target_fixed" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_target_fixed" "flow" "asym" "target=auc_normalized"
  add_method "flow_kmeans_weighted_all_pairwise" "${DENSITY_OUT}/leakage_free_flow_eps_joint_kmeans_weighted_all_pairwise" "flow" "asym"
  add_method "flow_kmeans_manifold" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold" "flow" "asym"
  add_method "flow_kmeans_manifold_train_only" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_train_only" "flow" "asym" "dir=train_only"
  add_method "flow_kmeans_manifold_eval_only" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_eval_only" "flow" "asym" "dir=eval_only"
  add_method "flow_kmeans_manifold_pairwise" "${DENSITY_OUT}/leakage_free_flow_joint_kmeans_manifold_pairwise" "flow" "asym"
  add_method "dino_rnorm_k5" "${DENSITY_OUT}/leakage_free_dino_rnorm_k5" "appearance" "asym"
  add_method "dino_rnorm_k5_train_only" "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_train_only" "appearance" "asym" "dir=train_only"
  add_method "dino_rnorm_k5_eval_only" "${DENSITY_OUT}/leakage_free_dino_rnorm_k5_eval_only" "appearance" "asym" "dir=eval_only"
  add_method "flow_eval_only_plus_dino_eval_only" "${DENSITY_OUT}/leakage_free_flow_eval_only_plus_dino_eval_only" "mixed" "asym" "dir=eval_only"
  add_method "mmd_only" "${OUTPUT_ROOT}/mmd_only" "mmd" "sym"
  add_method "asym_and_mmd" "${OUTPUT_ROOT}/asym_and_mmd" "mixed" "mixed"
  if [[ -s "${COMBO_METHODS_FILE}" ]]; then
    while IFS='|' read -r name path family symmetry notes; do
      add_method "${name}" "${path}" "${family}" "${symmetry}" "${notes}"
    done < "${COMBO_METHODS_FILE}"
  fi
} > "${MANIFEST_PATH}"

if [[ "${SKIP_SUMMARY}" -eq 0 ]]; then
  python scripts/compile_method_summary.py \
    --manifest "${MANIFEST_PATH}" \
    --output "${OUTPUT_ROOT}/method_summary.csv" \
    --output-md "${OUTPUT_ROOT}/method_summary.md"
fi

echo ""
echo "Done."
echo "Output root: ${OUTPUT_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
