#!/usr/bin/env bash
set -euo pipefail

# Run the full joint-density analysis with configurable input/output roots.
# Inputs are read from INPUT_ROOT (default: analysis).
# Outputs are written to OUTPUT_ROOT (default: analysis_density_joint_runs/<timestamp>).

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
NO_FAMILY_EFFECTS=0
NO_DENSITY_CONTROLS=0
MODEL_FAMILY_INTERACTIONS=""
SPAIR_INDICATOR_INTERACTIONS=""
RANKING_GROUP="train_dataset"
PAIRWISE_GROUP_COLS=""
RANKING_CONTEXT_COLS=""
JOINT_OOD_HOLDOUT=0
LINEAR_MODEL=""
PREDICTION_MODEL=""
RIDGE_ALPHA=""
PER_ENCODER=1
PREDICTION_CLIP_SET=0
PREDICTION_CLIP=0
DISABLE_MIXEDLM=0
COLLAPSE_CV_CELLS=0
ALLOW_MIXED_CONTEXT_COLLAPSE=0
FIT_SAMPLE_WEIGHTING="none"
FIT_BALANCE_REAL_SYNTH=0
OVERALL_AGGREGATION="micro"
CV_RESIDUALIZE_TARGET_BY_CONTEXT=0
CV_RESIDUAL_CONTEXT_COLS=""
CV_RESIDUAL_EVAL_SPACE="residual"
CV_RESIDUAL_TARGET_TRANSFORM="residual"
CV_RESIDUAL_TARGET_STD_EPS="1e-9"
CV_FEWSHOT_CONTEXT_CALIBRATION=0
CV_FEWSHOT_CONTEXT_CALIBRATION_COLS=""
CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="1e-9"
CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE="2"
CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1
CV_FEWSHOT_CONTEXT_CALIBRATION_K="0"
CV_FEWSHOT_CONTEXT_CALIBRATION_SEED="0"
CV_REPEAT_AGGREGATION="none"
LOTO_SINGLE_PREDICTOR_BASELINES=1
JOINTOOD_SINGLE_PREDICTOR_BASELINES=1
FLOW_MMD_CSV=""
FEATURE_MMD_CSV=""
DINO_MMD_CSV=""
PAIRWISE_ALL=0

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
    --no-family-effects)
      NO_FAMILY_EFFECTS=1
      shift 1
      ;;
    --model-family-interactions)
      MODEL_FAMILY_INTERACTIONS="on"
      shift 1
      ;;
    --no-model-family-interactions)
      MODEL_FAMILY_INTERACTIONS="off"
      shift 1
      ;;
    --spair-indicator-interactions)
      SPAIR_INDICATOR_INTERACTIONS="on"
      shift 1
      ;;
    --no-spair-indicator-interactions)
      SPAIR_INDICATOR_INTERACTIONS="off"
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
    --joint-ood-holdout)
      JOINT_OOD_HOLDOUT=1
      shift 1
      ;;
    --no-joint-ood-holdout)
      JOINT_OOD_HOLDOUT=0
      shift 1
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
    --per-encoder)
      PER_ENCODER=1
      shift 1
      ;;
    --no-per-encoder)
      PER_ENCODER=0
      shift 1
      ;;
    --prediction-clip)
      PREDICTION_CLIP_SET=1
      PREDICTION_CLIP=1
      shift 1
      ;;
    --no-prediction-clip)
      PREDICTION_CLIP_SET=1
      PREDICTION_CLIP=0
      shift 1
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
    --allow-mixed-context-collapse)
      ALLOW_MIXED_CONTEXT_COLLAPSE=1
      shift 1
      ;;
    --fit-sample-weighting)
      FIT_SAMPLE_WEIGHTING="$2"
      shift 2
      ;;
    --fit-balance-real-synth)
      FIT_BALANCE_REAL_SYNTH=1
      shift 1
      ;;
    --no-fit-balance-real-synth)
      FIT_BALANCE_REAL_SYNTH=0
      shift 1
      ;;
    --overall-aggregation)
      OVERALL_AGGREGATION="$2"
      shift 2
      ;;
    --cv-residualize-target-by-context)
      CV_RESIDUALIZE_TARGET_BY_CONTEXT=1
      shift 1
      ;;
    --no-cv-residualize-target-by-context)
      CV_RESIDUALIZE_TARGET_BY_CONTEXT=0
      shift 1
      ;;
    --cv-residual-context-cols)
      CV_RESIDUAL_CONTEXT_COLS="$2"
      shift 2
      ;;
    --cv-residual-eval-space)
      CV_RESIDUAL_EVAL_SPACE="$2"
      shift 2
      ;;
    --cv-residual-target-transform)
      CV_RESIDUAL_TARGET_TRANSFORM="$2"
      shift 2
      ;;
    --cv-residual-target-std-eps)
      CV_RESIDUAL_TARGET_STD_EPS="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration)
      CV_FEWSHOT_CONTEXT_CALIBRATION=1
      shift 1
      ;;
    --no-cv-fewshot-context-calibration)
      CV_FEWSHOT_CONTEXT_CALIBRATION=0
      shift 1
      ;;
    --cv-fewshot-context-calibration-cols)
      CV_FEWSHOT_CONTEXT_CALIBRATION_COLS="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-std-eps)
      CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-min-group-size)
      CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-backoff)
      CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=1
      shift 1
      ;;
    --no-cv-fewshot-context-calibration-backoff)
      CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF=0
      shift 1
      ;;
    --cv-fewshot-context-calibration-k)
      CV_FEWSHOT_CONTEXT_CALIBRATION_K="$2"
      shift 2
      ;;
    --cv-fewshot-context-calibration-seed)
      CV_FEWSHOT_CONTEXT_CALIBRATION_SEED="$2"
      shift 2
      ;;
    --cv-repeat-aggregation)
      CV_REPEAT_AGGREGATION="$2"
      shift 2
      ;;
    --loto-single-predictor-baselines)
      LOTO_SINGLE_PREDICTOR_BASELINES=1
      shift 1
      ;;
    --no-loto-single-predictor-baselines)
      LOTO_SINGLE_PREDICTOR_BASELINES=0
      shift 1
      ;;
    --jointood-single-predictor-baselines)
      JOINTOOD_SINGLE_PREDICTOR_BASELINES=1
      shift 1
      ;;
    --no-jointood-single-predictor-baselines)
      JOINTOOD_SINGLE_PREDICTOR_BASELINES=0
      shift 1
      ;;
    --no-mixedlm)
      DISABLE_MIXEDLM=1
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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

trim_ws() {
  local v="$1"
  v="${v#"${v%%[![:space:]]*}"}"
  v="${v%"${v##*[![:space:]]}"}"
  printf '%s' "${v}"
}

if [[ "${COLLAPSE_CV_CELLS}" -eq 1 && "${ALLOW_MIXED_CONTEXT_COLLAPSE}" -ne 1 ]]; then
  if [[ -n "$(trim_ws "${RANKING_CONTEXT_COLS}")" ]]; then
    echo "Refusing --collapse-cv-cells with non-empty --ranking-context-cols (${RANKING_CONTEXT_COLS})." >&2
    echo "This can mix model contexts and contaminate LOBO/LOTO/Joint ranking metrics." >&2
    echo "Use --no-collapse-cv-cells (recommended), or add --allow-mixed-context-collapse to override." >&2
    exit 1
  fi
fi
case "${CV_RESIDUAL_EVAL_SPACE}" in
  absolute|residual)
    ;;
  *)
    echo "Invalid --cv-residual-eval-space: ${CV_RESIDUAL_EVAL_SPACE}" >&2
    exit 1
    ;;
esac
case "${CV_RESIDUAL_TARGET_TRANSFORM}" in
  residual|zscore)
    ;;
  *)
    echo "Invalid --cv-residual-target-transform: ${CV_RESIDUAL_TARGET_TRANSFORM}" >&2
    exit 1
    ;;
esac
case "${CV_REPEAT_AGGREGATION}" in
  none|mean|median)
    ;;
  *)
    echo "Invalid --cv-repeat-aggregation: ${CV_REPEAT_AGGREGATION}" >&2
    exit 1
    ;;
esac

if [[ -z "${OUTPUT_ROOT}" ]]; then
  TS="$(date +%Y%m%d_%H%M%S)"
  OUTPUT_ROOT="analysis_density_joint_runs/${TS}"
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

mkdir -p "${OUTPUT_ROOT}"

CURVE_SUMMARY_Q90_95="${OUTPUT_ROOT}/coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv"
CURVE_SUMMARY_Q50="${OUTPUT_ROOT}/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv"
KMEANS_CURVE_SUMMARY_Q90_95="${OUTPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_curve_summary_q90_95.csv"

DENSITY_ARGS=(
  --flow-stats-dir /mnt/nvme_1tb_b/coverage_vectors/stats
  --model-family-main-effects
  --target auc_normalized_observed
  --cv-standardize-mode local
  --strict-dataset-match
  --no-allow-unsplit-coverage
  --no-allow-unsplit-mmd
  --no-allow-unsplit-flow-stats
)
if [[ "${NO_DENSITY_CONTROLS}" -eq 0 ]]; then
  DENSITY_ARGS+=(
    --use-flow-density-predictors
    --flow-density-interactions
  )
else
  DENSITY_ARGS+=(
    --no-use-flow-density-predictors
    --no-flow-density-interactions
  )
fi
if [[ "${MODEL_FAMILY_INTERACTIONS}" == "on" ]]; then
  DENSITY_ARGS+=(--model-family-interactions)
elif [[ "${MODEL_FAMILY_INTERACTIONS}" == "off" ]]; then
  DENSITY_ARGS+=(--no-model-family-interactions)
fi
if [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "on" ]]; then
  DENSITY_ARGS+=(--spair-indicator-interactions)
elif [[ "${SPAIR_INDICATOR_INTERACTIONS}" == "off" ]]; then
  DENSITY_ARGS+=(--no-spair-indicator-interactions)
fi
DENSITY_ARGS+=(--ranking-group "${RANKING_GROUP}")
if [[ -n "${PAIRWISE_GROUP_COLS}" ]]; then
  DENSITY_ARGS+=(--pairwise-group-cols "${PAIRWISE_GROUP_COLS}")
fi
if [[ -n "${RANKING_CONTEXT_COLS}" ]]; then
  DENSITY_ARGS+=(--ranking-context-cols "${RANKING_CONTEXT_COLS}")
fi
if [[ "${JOINT_OOD_HOLDOUT}" -eq 1 ]]; then
  DENSITY_ARGS+=(--joint-ood-holdout)
fi
if [[ -n "${LINEAR_MODEL}" ]]; then
  DENSITY_ARGS+=(--linear-model "${LINEAR_MODEL}")
fi
if [[ -n "${PREDICTION_MODEL}" ]]; then
  DENSITY_ARGS+=(--prediction-model "${PREDICTION_MODEL}")
fi
if [[ -n "${RIDGE_ALPHA}" ]]; then
  DENSITY_ARGS+=(--ridge-alpha "${RIDGE_ALPHA}")
fi
if [[ "${PER_ENCODER}" -eq 0 ]]; then
  DENSITY_ARGS+=(--no-per-encoder)
fi
if [[ "${PREDICTION_CLIP_SET}" -eq 1 ]]; then
  if [[ "${PREDICTION_CLIP}" -eq 1 ]]; then
    DENSITY_ARGS+=(--prediction-clip)
  else
    DENSITY_ARGS+=(--no-prediction-clip)
  fi
fi
if [[ "${COLLAPSE_CV_CELLS}" -eq 1 ]]; then
  DENSITY_ARGS+=(--collapse-cv-cells)
else
  DENSITY_ARGS+=(--no-collapse-cv-cells)
fi
DENSITY_ARGS+=(--fit-sample-weighting "${FIT_SAMPLE_WEIGHTING}")
if [[ "${FIT_BALANCE_REAL_SYNTH}" -eq 1 ]]; then
  DENSITY_ARGS+=(--fit-balance-real-synth)
else
  DENSITY_ARGS+=(--no-fit-balance-real-synth)
fi
DENSITY_ARGS+=(--overall-aggregation "${OVERALL_AGGREGATION}")
if [[ "${CV_RESIDUALIZE_TARGET_BY_CONTEXT}" -eq 1 ]]; then
  DENSITY_ARGS+=(--cv-residualize-target-by-context)
fi
if [[ -n "${CV_RESIDUAL_CONTEXT_COLS}" ]]; then
  DENSITY_ARGS+=(--cv-residual-context-cols "${CV_RESIDUAL_CONTEXT_COLS}")
fi
DENSITY_ARGS+=(--cv-residual-eval-space "${CV_RESIDUAL_EVAL_SPACE}")
DENSITY_ARGS+=(--cv-residual-target-transform "${CV_RESIDUAL_TARGET_TRANSFORM}")
DENSITY_ARGS+=(--cv-residual-target-std-eps "${CV_RESIDUAL_TARGET_STD_EPS}")
if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION}" -eq 1 ]]; then
  DENSITY_ARGS+=(--cv-fewshot-context-calibration)
fi
if [[ -n "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}" ]]; then
  DENSITY_ARGS+=(--cv-fewshot-context-calibration-cols "${CV_FEWSHOT_CONTEXT_CALIBRATION_COLS}")
fi
DENSITY_ARGS+=(--cv-fewshot-context-calibration-std-eps "${CV_FEWSHOT_CONTEXT_CALIBRATION_STD_EPS}")
DENSITY_ARGS+=(--cv-fewshot-context-calibration-min-group-size "${CV_FEWSHOT_CONTEXT_CALIBRATION_MIN_GROUP_SIZE}")
if [[ "${CV_FEWSHOT_CONTEXT_CALIBRATION_BACKOFF}" -eq 1 ]]; then
  DENSITY_ARGS+=(--cv-fewshot-context-calibration-backoff)
else
  DENSITY_ARGS+=(--no-cv-fewshot-context-calibration-backoff)
fi
DENSITY_ARGS+=(--cv-fewshot-context-calibration-k "${CV_FEWSHOT_CONTEXT_CALIBRATION_K}")
DENSITY_ARGS+=(--cv-fewshot-context-calibration-seed "${CV_FEWSHOT_CONTEXT_CALIBRATION_SEED}")
DENSITY_ARGS+=(--cv-repeat-aggregation "${CV_REPEAT_AGGREGATION}")
if [[ "${LOTO_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
  DENSITY_ARGS+=(--loto-single-predictor-baselines)
else
  DENSITY_ARGS+=(--no-loto-single-predictor-baselines)
fi
if [[ "${JOINTOOD_SINGLE_PREDICTOR_BASELINES}" -eq 1 ]]; then
  DENSITY_ARGS+=(--jointood-single-predictor-baselines)
else
  DENSITY_ARGS+=(--no-jointood-single-predictor-baselines)
fi
if [[ -n "${FLOW_MMD_CSV}" ]]; then
  DENSITY_ARGS+=(--flow-mmd-csv "${FLOW_MMD_CSV}")
fi
if [[ -n "${FEATURE_MMD_CSV}" ]]; then
  DENSITY_ARGS+=(--feature-mmd-csv "${FEATURE_MMD_CSV}")
fi
if [[ -n "${DINO_MMD_CSV}" ]]; then
  DENSITY_ARGS+=(--dino-mmd-csv "${DINO_MMD_CSV}")
fi
if [[ "${DISABLE_MIXEDLM}" -eq 1 ]]; then
  DENSITY_ARGS+=(--no-prediction-mixedlm --no-regression-mixedlm)
fi
if [[ "${NO_FAMILY_EFFECTS}" -eq 1 ]]; then
  DENSITY_ARGS+=(
    --no-encoder-main-effects
    --no-encoder-interactions
    --no-model-family-main-effects
    --no-model-family-interactions
  )
fi

python scripts/summarize_flow_eps_curves.py \
  --input-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_curves_full.csv" \
  --output-csv "${CURVE_SUMMARY_Q90_95}" \
  --coverage-thresholds 0.9,0.95

python scripts/summarize_flow_eps_curves.py \
  --input-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_curves_full.csv" \
  --output-csv "${KMEANS_CURVE_SUMMARY_Q90_95}" \
  --coverage-thresholds 0.9,0.95 \
  --weighted

python scripts/summarize_flow_eps_curves.py \
  --input-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_curves_full.csv" \
  --output-csv "${CURVE_SUMMARY_Q50}" \
  --coverage-thresholds 0.5

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint" \
  --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
  --flow-eps-values 0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64 \
  --use-flow-eps-predictors \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_auc_at95" \
  --coverage-csv "${CURVE_SUMMARY_Q90_95}" \
  --no-flow-eps-predictors \
  --predictors flow_train_to_eval_auc,flow_eval_to_train_auc \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_eps_at50" \
  --coverage-csv "${CURVE_SUMMARY_Q50}" \
  --no-flow-eps-predictors \
  --predictors flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50 \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_joint_kmeans_weighted_all" \
  --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
  --use-flow-eps-weighted-predictors \
  --no-flow-eps-predictors \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_joint_kmeans_manifold" \
  --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
  --no-flow-eps-predictors \
  --predictors flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

if [[ "${PAIRWISE_ALL}" -eq 1 ]]; then
  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_pairwise" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
    --flow-eps-values 0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64 \
    --use-flow-eps-predictors \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}" \
    --linear-model pairwise_rank \
    --prediction-model pairwise_rank

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_auc_at95_pairwise" \
    --coverage-csv "${CURVE_SUMMARY_Q90_95}" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_auc,flow_eval_to_train_auc \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}" \
    --linear-model pairwise_rank \
    --prediction-model pairwise_rank

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_eps_at50_pairwise" \
    --coverage-csv "${CURVE_SUMMARY_Q50}" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50 \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}" \
    --linear-model pairwise_rank \
    --prediction-model pairwise_rank

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_joint_kmeans_weighted_all_pairwise" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
    --use-flow-eps-weighted-predictors \
    --no-flow-eps-predictors \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}" \
    --linear-model pairwise_rank \
    --prediction-model pairwise_rank

  python scripts/build_leakage_free_eval.py \
    --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
    --output-dir "${OUTPUT_ROOT}/leakage_free_flow_joint_kmeans_manifold_pairwise" \
    --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
    --no-flow-eps-predictors \
    --predictors flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train \
    --no-logit-coverage \
    "${DENSITY_ARGS[@]}" \
    --linear-model pairwise_rank \
    --prediction-model pairwise_rank
fi

echo ""
echo "Done."
echo "Output root: ${OUTPUT_ROOT}"
