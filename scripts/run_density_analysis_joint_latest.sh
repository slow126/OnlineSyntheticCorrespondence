#!/usr/bin/env bash
set -euo pipefail

# Run the full joint-density analysis with configurable input/output roots.
# Inputs are read from INPUT_ROOT (default: analysis).
# Outputs are written to OUTPUT_ROOT (default: analysis_density_joint_runs/<timestamp>).

INPUT_ROOT="${INPUT_ROOT:-analysis}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"

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
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

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
  --use-flow-density-predictors
  --flow-density-interactions
  --model-family-main-effects
  --target auc_normalized_observed
)

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

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_pairwise" \
  --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_full.csv" \
  --flow-eps-values 0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64 \
  --use-flow-eps-predictors \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_auc_at95_pairwise" \
  --coverage-csv "${CURVE_SUMMARY_Q90_95}" \
  --no-flow-eps-predictors \
  --predictors flow_train_to_eval_auc,flow_eval_to_train_auc \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_joint_eps_at50_pairwise" \
  --coverage-csv "${CURVE_SUMMARY_Q50}" \
  --no-flow-eps-predictors \
  --predictors flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50 \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_joint_kmeans_weighted_all_pairwise" \
  --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_full.csv" \
  --use-flow-eps-weighted-predictors \
  --no-flow-eps-predictors \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_joint_kmeans_manifold_pairwise" \
  --coverage-csv "${INPUT_ROOT}/coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv" \
  --no-flow-eps-predictors \
  --predictors flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

echo ""
echo "Done."
echo "Output root: ${OUTPUT_ROOT}"
