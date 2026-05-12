#!/usr/bin/env bash
set -euo pipefail

# Quick ring/delta EPS-ladder sweep:
# - Mirrors zeroshot-v3 no-family/no-density/no-interaction ridge config
# - Runs only EPS-ladder flow variants and appearance combos

BASELINE_ROOT="${BASELINE_ROOT:-analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3}"
INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_comprehensive_runs/ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3_eps_rings_v1}"
FLOW_EPS_VALUES="${FLOW_EPS_VALUES:-0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64}"
FORCE_RERUN="${FORCE_RERUN:-0}"

DENSITY_OUT="${OUTPUT_ROOT}/density_joint"
DINO_COVERAGE="${DINO_COVERAGE:-${BASELINE_ROOT}/dino_coverage_rnorm_k5_with_kl_k5.csv}"
FLOW_RAW_CSV="${FLOW_RAW_CSV:-${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv}"
FLOW_WEIGHTED_CSV="${FLOW_WEIGHTED_CSV:-${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv}"

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

if [[ ! -f "${FLOW_RAW_CSV}" ]]; then
  echo "Missing flow raw coverage CSV: ${FLOW_RAW_CSV}" >&2
  exit 1
fi
if [[ ! -f "${FLOW_WEIGHTED_CSV}" ]]; then
  echo "Missing flow weighted coverage CSV: ${FLOW_WEIGHTED_CSV}" >&2
  exit 1
fi
if [[ ! -f "${DINO_COVERAGE}" ]]; then
  echo "Missing dino coverage CSV: ${DINO_COVERAGE}" >&2
  exit 1
fi

mkdir -p "${DENSITY_OUT}"

build_eps_pred_list() {
  local prefix="$1"
  local suffix="$2"
  local out=""
  IFS=',' read -ra vals <<< "${FLOW_EPS_VALUES}"
  for v in "${vals[@]}"; do
    local v_clean="${v%px}"
    local key
    key="$(echo "${v_clean}" | sed 's/\./p/g')"
    out+="${prefix}${key}${suffix},"
  done
  echo "${out%,}"
}

RAW_PREDS="$(build_eps_pred_list "flow_train_to_eval_eps" "px"),$(build_eps_pred_list "flow_eval_to_train_eps" "px")"
WEIGHTED_PREDS="$(build_eps_pred_list "flow_train_to_eval_eps" "px_weighted"),$(build_eps_pred_list "flow_eval_to_train_eps" "px_weighted")"

COMMON_ARGS=(
  --snapshots-dir "${SNAPSHOT_DIRS[@]}"
  --flow-stats-dir /mnt/nvme_1tb_b/coverage_vectors/stats
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --no-allow-unsplit-flow-stats
  --target auc_normalized_observed
  --linear-model ridge
  --prediction-model ridge
  --ridge-alpha 10
  --standardize
  --fit-sample-weighting inverse_task
  --fit-balance-real-synth
  --overall-aggregation macro_fold
  --cv-standardize-mode local
  --cv-residualize-target-by-context
  --cv-residual-context-cols benchmark,model_family_encoder
  --cv-residual-eval-space residual
  --cv-residual-target-transform zscore
  --cv-residual-target-std-eps 1e-9
  --cv-repeat-aggregation median
  --ranking-group train_dataset
  --ranking-context-cols model_family_encoder
  --pairwise-group-cols benchmark,model_family_encoder
  --joint-ood-holdout
  --no-per-encoder
  --no-logit-coverage
  --no-use-flow-density-predictors
  --no-flow-density-interactions
  --no-encoder-main-effects
  --no-encoder-interactions
  --no-model-family-main-effects
  --no-model-family-interactions
  --no-spair-indicator-interactions
  --flow-eps-values "${FLOW_EPS_VALUES}"
  --flow-eps-rings
  --no-loto-single-predictor-baselines
  --no-jointood-single-predictor-baselines
)

run_eval() {
  local out_dir="$1"
  shift
  if [[ -d "${out_dir}" && "${FORCE_RERUN}" -eq 0 ]]; then
    echo "Skipping existing run: ${out_dir}"
    return 0
  fi
  python scripts/build_leakage_free_eval.py "${COMMON_ARGS[@]}" --output-dir "${out_dir}" "$@"
}

run_eval \
  "${DENSITY_OUT}/leakage_free_flow_eps_raw_ring" \
  --coverage-csv "${FLOW_RAW_CSV}" \
  --no-flow-eps-predictors \
  --predictors "${RAW_PREDS}"

run_eval \
  "${DENSITY_OUT}/leakage_free_combo_flow_eps_raw_ring__dino_rnorm_k5" \
  --coverage-csv "${FLOW_RAW_CSV}" \
  --coverage-dino-csv "${DINO_COVERAGE}" \
  --no-flow-eps-predictors \
  --predictors "${RAW_PREDS},dino_eval_to_train_mean_dist,dino_train_to_eval_mean_dist"

run_eval \
  "${DENSITY_OUT}/leakage_free_combo_flow_eps_raw_ring__dino_kl_k5" \
  --coverage-csv "${FLOW_RAW_CSV}" \
  --coverage-dino-csv "${DINO_COVERAGE}" \
  --no-flow-eps-predictors \
  --predictors "${RAW_PREDS},dino_eval_to_train_kl_div,dino_train_to_eval_kl_div"

run_eval \
  "${DENSITY_OUT}/leakage_free_flow_kmeans_weighted_ring" \
  --coverage-csv "${FLOW_WEIGHTED_CSV}" \
  --no-flow-eps-predictors \
  --predictors "${WEIGHTED_PREDS}"

run_eval \
  "${DENSITY_OUT}/leakage_free_combo_flow_kmeans_weighted_ring__dino_rnorm_k5" \
  --coverage-csv "${FLOW_WEIGHTED_CSV}" \
  --coverage-dino-csv "${DINO_COVERAGE}" \
  --no-flow-eps-predictors \
  --predictors "${WEIGHTED_PREDS},dino_eval_to_train_mean_dist,dino_train_to_eval_mean_dist"

run_eval \
  "${DENSITY_OUT}/leakage_free_combo_flow_kmeans_weighted_ring__dino_kl_k5" \
  --coverage-csv "${FLOW_WEIGHTED_CSV}" \
  --coverage-dino-csv "${DINO_COVERAGE}" \
  --no-flow-eps-predictors \
  --predictors "${WEIGHTED_PREDS},dino_eval_to_train_kl_div,dino_train_to_eval_kl_div"

echo "Ring EPS sweep complete."
echo "Output root: ${OUTPUT_ROOT}"
