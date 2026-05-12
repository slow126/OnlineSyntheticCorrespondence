#!/usr/bin/env bash
set -euo pipefail

# Run 1-flow + 1-dino predictor combo experiments (OLS/Ridge, Joint-OOD) into an
# existing comprehensive analysis root.
#
# Output dirs are created under:
#   ${ROOT}/density_joint/leakage_free_combo_1f1d__*
#
# By default this includes:
# - Flow predictor types: eps_raw_single, eps_at50, auc_at95, kmeans_manifold
#   (and flow_kl_k5 if its CSV exists)
# - Directions: train/eval for both flow and dino predictors
# - Dino predictor types: rnorm (and kl if dino KL columns exist)
#
# Usage:
#   bash scripts/run_single_flow_single_dino_combos.sh
#   bash scripts/run_single_flow_single_dino_combos.sh --dry-run
#   bash scripts/run_single_flow_single_dino_combos.sh --root <analysis_root>
#   bash scripts/run_single_flow_single_dino_combos.sh --root <analysis_root> --force-rerun

ROOT="${ROOT:-analysis_comprehensive_runs/hof_motion_v3_density_jointood_full_ridge_a100_v3}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-analysis}"
FLOW_STATS_DIR="${FLOW_STATS_DIR:-/mnt/nvme_1tb_b/coverage_vectors/stats}"
LINEAR_MODEL="${LINEAR_MODEL:-ridge}"
PREDICTION_MODEL="${PREDICTION_MODEL:-ridge}"
RIDGE_ALPHA="${RIDGE_ALPHA:-100}"
RANKING_GROUP="${RANKING_GROUP:-train_dataset}"
PAIRWISE_GROUP_COLS="${PAIRWISE_GROUP_COLS:-}"
RANKING_CONTEXT_COLS="${RANKING_CONTEXT_COLS:-}"
FLOW_MMD_CSV="${FLOW_MMD_CSV:-}"
FEATURE_MMD_CSV="${FEATURE_MMD_CSV:-}"
DINO_MMD_CSV="${DINO_MMD_CSV:-}"
COLLAPSE_CV_CELLS=1
NO_DENSITY_CONTROLS=0
NO_FAMILY_EFFECTS=0
PAIRWISE_ALL=0
FORCE_RERUN=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT="$2"
      shift 2
      ;;
    --analysis-root)
      ANALYSIS_ROOT="$2"
      shift 2
      ;;
    --flow-stats-dir)
      FLOW_STATS_DIR="$2"
      shift 2
      ;;
    --linear-model)
      LINEAR_MODEL="$2"
      shift 2
      ;;
    --prediction-model)
      PREDICTION_MODEL="$2"
      shift 2
      ;;
    --ridge-alpha)
      RIDGE_ALPHA="$2"
      shift 2
      ;;
    --ranking-group)
      RANKING_GROUP="$2"
      shift 2
      ;;
    --pairwise-group-cols)
      PAIRWISE_GROUP_COLS="$2"
      shift 2
      ;;
    --ranking-context-cols)
      RANKING_CONTEXT_COLS="$2"
      shift 2
      ;;
    --flow-mmd-csv)
      FLOW_MMD_CSV="$2"
      shift 2
      ;;
    --feature-mmd-csv)
      FEATURE_MMD_CSV="$2"
      shift 2
      ;;
    --dino-mmd-csv)
      DINO_MMD_CSV="$2"
      shift 2
      ;;
    --collapse-cv-cells)
      COLLAPSE_CV_CELLS=1
      shift 1
      ;;
    --no-collapse-cv-cells)
      COLLAPSE_CV_CELLS=0
      shift 1
      ;;
    --no-density-controls)
      NO_DENSITY_CONTROLS=1
      shift 1
      ;;
    --no-family-effects)
      NO_FAMILY_EFFECTS=1
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
    --force-rerun)
      FORCE_RERUN=1
      shift 1
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

ROOT="$(realpath "${ROOT}")"
DENSITY_OUT="${ROOT}/density_joint"
if [[ -z "${FLOW_MMD_CSV}" ]]; then
  FLOW_MMD_CSV="${ROOT}/mmd/mmd_v2_flow_joint_v1.csv"
fi
if [[ -z "${DINO_MMD_CSV}" ]]; then
  DINO_MMD_CSV="${ROOT}/mmd/mmd_v2_dino_v1.csv"
fi
if [[ -z "${FEATURE_MMD_CSV}" ]]; then
  FEATURE_MMD_CSV="${ROOT}/mmd/mmd_v2_feature_v1.csv"
fi
if [[ ! -f "${FEATURE_MMD_CSV}" && -f "${DINO_MMD_CSV}" ]]; then
  FEATURE_MMD_CSV="${DINO_MMD_CSV}"
fi
DINO_CSV="${ROOT}/dino_coverage_rnorm_k5_with_kl_k5.csv"
if [[ ! -f "${DINO_CSV}" ]]; then
  DINO_CSV="${ROOT}/dino_coverage_rnorm_k5.csv"
fi
if [[ ! -f "${DINO_CSV}" ]]; then
  echo "Missing dino coverage CSV under ${ROOT} (expected dino_coverage_rnorm_k5*.csv)." >&2
  exit 1
fi

FLOW_KL_CSV="${ROOT}/kl/kl_flow_k5.csv"

FLOW_CSV_EPS_RAW_SINGLE="${ANALYSIS_ROOT}/coverage_v2_flow_only_raw_joint_full.csv"
FLOW_CSV_EPS_AT50="${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv"
FLOW_CSV_AUC_AT95="${DENSITY_OUT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv"
FLOW_CSV_KMEANS_MANIFOLD="${ANALYSIS_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv"

for req in \
  "${FLOW_CSV_EPS_RAW_SINGLE}" \
  "${FLOW_CSV_EPS_AT50}" \
  "${FLOW_CSV_AUC_AT95}" \
  "${FLOW_CSV_KMEANS_MANIFOLD}"; do
  if [[ ! -f "${req}" ]]; then
    echo "Missing required flow CSV: ${req}" >&2
    exit 1
  fi
done

SNAPSHOTS=(
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
  --snapshots-dir "${SNAPSHOTS[@]}"
  --coverage-dino-csv "${DINO_CSV}"
  --flow-stats-dir "${FLOW_STATS_DIR}"
  --target auc_normalized_observed
  --joint-ood-holdout
  --linear-model "${LINEAR_MODEL}"
  --prediction-model "${PREDICTION_MODEL}"
  --ranking-group "${RANKING_GROUP}"
  --cv-standardize-mode local
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --no-allow-unsplit-flow-stats
  --no-per-encoder
  --no-logit-coverage
  --no-flow-eps-predictors
  --no-flow-eps-weighted-predictors
)
if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
  COMMON_ARGS+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
fi
if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
  COMMON_ARGS+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
fi
if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
  COMMON_ARGS+=(--collapse-cv-cells)
else
  COMMON_ARGS+=(--no-collapse-cv-cells)
fi
if [[ -f "${FLOW_MMD_CSV}" ]]; then
  COMMON_ARGS+=(--flow-mmd-csv "${FLOW_MMD_CSV}")
fi
if [[ -f "${FEATURE_MMD_CSV}" ]]; then
  COMMON_ARGS+=(--feature-mmd-csv "${FEATURE_MMD_CSV}")
fi
if [[ -f "${DINO_MMD_CSV}" ]]; then
  COMMON_ARGS+=(--dino-mmd-csv "${DINO_MMD_CSV}")
fi
if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
  COMMON_ARGS+=(
    --no-encoder-main-effects
    --no-encoder-interactions
    --no-model-family-main-effects
    --no-model-family-interactions
  )
else
  COMMON_ARGS+=(--model-family-main-effects)
fi
if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
  COMMON_ARGS+=(--no-use-flow-density-predictors --no-flow-density-interactions)
else
  COMMON_ARGS+=(--use-flow-density-predictors --flow-density-interactions)
fi
if [[ "${LINEAR_MODEL}" == "ridge" || "${PREDICTION_MODEL}" == "ridge" ]]; then
  COMMON_ARGS+=(--ridge-alpha "${RIDGE_ALPHA}")
fi

has_col() {
  local csv_path="$1"
  local col="$2"
  python - "$csv_path" "$col" <<'PY'
import csv, sys
path, col = sys.argv[1], sys.argv[2]
with open(path, "r", newline="") as f:
    r = csv.reader(f)
    header = next(r, [])
print("1" if col in header else "0")
PY
}

run_count=0
skip_count=0

run_one() {
  local out_dir="$1"
  local flow_csv="$2"
  local predictors="$3"

  if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 0 ]]; then
    echo "SKIP existing: ${out_dir}"
    skip_count=$((skip_count + 1))
  else
    if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 1 ]]; then
      echo "RERUN existing: ${out_dir}"
    fi
    local cmd=(
      python scripts/build_leakage_free_eval.py
      "${COMMON_ARGS[@]}"
      --coverage-csv "${flow_csv}"
      --output-dir "${out_dir}"
      --predictors "${predictors}"
    )

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf 'DRY-RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      run_count=$((run_count + 1))
    else
      printf 'RUN:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
      "${cmd[@]}"
      run_count=$((run_count + 1))
    fi
  fi

  if [[ "${PAIRWISE_ALL}" -eq 1 ]]; then
    local pair_out="${out_dir}_pairwise"
    local pair_cmd=(
      python scripts/build_leakage_free_eval.py
      --snapshots-dir "${SNAPSHOTS[@]}"
      --coverage-dino-csv "${DINO_CSV}"
      --flow-stats-dir "${FLOW_STATS_DIR}"
      --target auc_normalized_observed
      --joint-ood-holdout
      --linear-model pairwise_rank
      --prediction-model pairwise_rank
      --ranking-group "${RANKING_GROUP}"
      --cv-standardize-mode local
      --strict-dataset-match
      --no-allow-unsplit-coverage
      --no-allow-unsplit-mmd
      --no-allow-unsplit-flow-stats
      --no-per-encoder
      --no-logit-coverage
      --no-flow-eps-predictors
      --no-flow-eps-weighted-predictors
      --no-flow-density-interactions
      --coverage-csv "${flow_csv}"
      --output-dir "${pair_out}"
      --predictors "${predictors}"
    )
    if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
      pair_cmd+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
    fi
    if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
      pair_cmd+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
    fi
    if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
      pair_cmd+=(--collapse-cv-cells)
    else
      pair_cmd+=(--no-collapse-cv-cells)
    fi
    if [[ -f "${FLOW_MMD_CSV}" ]]; then
      pair_cmd+=(--flow-mmd-csv "${FLOW_MMD_CSV}")
    fi
    if [[ -f "${FEATURE_MMD_CSV}" ]]; then
      pair_cmd+=(--feature-mmd-csv "${FEATURE_MMD_CSV}")
    fi
    if [[ -f "${DINO_MMD_CSV}" ]]; then
      pair_cmd+=(--dino-mmd-csv "${DINO_MMD_CSV}")
    fi
    if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
      pair_cmd+=(
        --no-encoder-main-effects
        --no-encoder-interactions
        --no-model-family-main-effects
        --no-model-family-interactions
      )
    else
      pair_cmd+=(--model-family-main-effects)
    fi
    if [[ "${NO_DENSITY_CONTROLS}" -eq 1 ]]; then
      pair_cmd+=(--no-use-flow-density-predictors)
    else
      pair_cmd+=(--use-flow-density-predictors)
    fi
    if [[ -d "${pair_out}" && "${FORCE_RERUN}" -eq 0 ]]; then
      echo "SKIP existing: ${pair_out}"
      skip_count=$((skip_count + 1))
      return 0
    fi
    if [[ -d "${pair_out}" && "${FORCE_RERUN}" -eq 1 ]]; then
      echo "RERUN existing: ${pair_out}"
    fi
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      printf 'DRY-RUN:'
      printf ' %q' "${pair_cmd[@]}"
      printf '\n'
      run_count=$((run_count + 1))
      return 0
    fi
    printf 'RUN:'
    printf ' %q' "${pair_cmd[@]}"
    printf '\n'
    "${pair_cmd[@]}"
    run_count=$((run_count + 1))
  fi
}

# flow_name|flow_csv|flow_train_pred|flow_eval_pred
FLOW_VARIANTS=(
  "eps_raw_single|${FLOW_CSV_EPS_RAW_SINGLE}|flow_train_to_eval_eps1px|flow_eval_to_train_eps1p5px"
  "eps_at50|${FLOW_CSV_EPS_AT50}|flow_train_to_eval_eps_at50|flow_eval_to_train_eps_at50"
  "auc_at95|${FLOW_CSV_AUC_AT95}|flow_train_to_eval_auc|flow_eval_to_train_auc"
  "kmeans_manifold|${FLOW_CSV_KMEANS_MANIFOLD}|flow_train_to_eval_mean_dist_over_radius_eval|flow_eval_to_train_mean_dist_over_radius_train"
)
if [[ -f "${FLOW_KL_CSV}" ]]; then
  FLOW_VARIANTS+=("kl_k5|${FLOW_KL_CSV}|flow_train_to_eval_kl_div|flow_eval_to_train_kl_div")
fi

# dino_name|dino_train_pred|dino_eval_pred
DINO_VARIANTS=(
  "rnorm|dino_train_to_eval_mean_dist|dino_eval_to_train_mean_dist"
)
if [[ "$(has_col "${DINO_CSV}" "kl_train_to_eval")" == "1" ]] && \
   [[ "$(has_col "${DINO_CSV}" "kl_eval_to_train")" == "1" ]]; then
  DINO_VARIANTS+=("kl_k5|dino_train_to_eval_kl_div|dino_eval_to_train_kl_div")
fi

for fv in "${FLOW_VARIANTS[@]}"; do
  IFS='|' read -r flow_name flow_csv flow_train flow_eval <<< "${fv}"
  for dv in "${DINO_VARIANTS[@]}"; do
    IFS='|' read -r dino_name dino_train dino_eval <<< "${dv}"

    # 4 directional combinations per predictor type pair.
    run_one \
      "${DENSITY_OUT}/leakage_free_combo_1f1d__flow_${flow_name}_train__dino_${dino_name}_train" \
      "${flow_csv}" \
      "${flow_train},${dino_train}"
    run_one \
      "${DENSITY_OUT}/leakage_free_combo_1f1d__flow_${flow_name}_train__dino_${dino_name}_eval" \
      "${flow_csv}" \
      "${flow_train},${dino_eval}"
    run_one \
      "${DENSITY_OUT}/leakage_free_combo_1f1d__flow_${flow_name}_eval__dino_${dino_name}_train" \
      "${flow_csv}" \
      "${flow_eval},${dino_train}"
    run_one \
      "${DENSITY_OUT}/leakage_free_combo_1f1d__flow_${flow_name}_eval__dino_${dino_name}_eval" \
      "${flow_csv}" \
      "${flow_eval},${dino_eval}"
  done
done

echo "Done. launched=${run_count}, skipped_existing=${skip_count}, dry_run=${DRY_RUN}"
