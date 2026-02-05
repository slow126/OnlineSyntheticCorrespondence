#!/usr/bin/env bash
set -euo pipefail

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

OUTPUT_ROOT="analysis_density"

DENSITY_ARGS=(
  --flow-stats-dir /mnt/nvme_1tb_b/coverage_vectors/stats
  --use-flow-density-predictors
  --flow-density-interactions
  --model-family-main-effects
)

python scripts/summarize_flow_eps_curves.py \
  --input-csv analysis/coverage_v2_flow_only_raw_curves_full.csv \
  --output-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --coverage-thresholds 0.9,0.95

python scripts/summarize_flow_eps_curves.py \
  --input-csv analysis/coverage_v2_flow_only_raw_kmeans_curves_full.csv \
  --output-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_kmeans_curve_summary.csv" \
  --coverage-thresholds 0.9,0.95 \
  --weighted

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw" \
  --coverage-csv analysis/coverage_v2_flow_only_raw_full.csv \
  --flow-eps-values 0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64 \
  --use-flow-eps-predictors \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_auc_at95" \
  --coverage-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --predictors flow_train_to_eval_auc,flow_eval_to_train_auc \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/summarize_flow_eps_curves.py \
  --input-csv analysis/coverage_v2_flow_only_raw_curves_full.csv \
  --output-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --coverage-thresholds 0.5

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_eps_at50" \
  --coverage-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --predictors flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50 \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_kmeans_weighted_all" \
  --coverage-csv analysis/coverage_v2_flow_only_raw_kmeans_full.csv \
  --use-flow-eps-weighted-predictors \
  --no-flow-eps-predictors \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_kmeans_manifold" \
  --coverage-csv analysis/coverage_v2_flow_only_raw_kmeans_manifold_full.csv \
  --predictors flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train \
  --no-logit-coverage \
  "${DENSITY_ARGS[@]}"

python scripts/summarize_flow_eps_curves.py \
  --input-csv analysis/coverage_v2_flow_only_raw_curves_full.csv \
  --output-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --coverage-thresholds 0.9,0.95

python scripts/summarize_flow_eps_curves.py \
  --input-csv analysis/coverage_v2_flow_only_raw_kmeans_curves_full.csv \
  --output-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_kmeans_curve_summary.csv" \
  --coverage-thresholds 0.9,0.95 \
  --weighted

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_pairwise" \
  --coverage-csv analysis/coverage_v2_flow_only_raw_full.csv \
  --flow-eps-values 0.5,0.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64 \
  --use-flow-eps-predictors \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_auc_at95_pairwise" \
  --coverage-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --predictors flow_train_to_eval_auc,flow_eval_to_train_auc \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/summarize_flow_eps_curves.py \
  --input-csv analysis/coverage_v2_flow_only_raw_curves_full.csv \
  --output-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --coverage-thresholds 0.5

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_raw_eps_at50_pairwise" \
  --coverage-csv "${OUTPUT_ROOT}/coverage_v2_flow_only_raw_curve_summary.csv" \
  --predictors flow_train_to_eval_eps_at50,flow_eval_to_train_eps_at50 \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_eps_kmeans_weighted_all_pairwise" \
  --coverage-csv analysis/coverage_v2_flow_only_raw_kmeans_full.csv \
  --use-flow-eps-weighted-predictors \
  --no-flow-eps-predictors \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"

python scripts/build_leakage_free_eval.py \
  --snapshots-dir "${SNAPSHOT_DIRS[@]}" \
  --output-dir "${OUTPUT_ROOT}/leakage_free_flow_kmeans_manifold_pairwise" \
  --coverage-csv analysis/coverage_v2_flow_only_raw_kmeans_manifold_full.csv \
  --predictors flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train \
  --no-logit-coverage \
  --prediction-model pairwise_rank \
  "${DENSITY_ARGS[@]}"
