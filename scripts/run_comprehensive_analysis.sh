#!/bin/bash
# Comprehensive stability and univariate analysis with filtered predictor set
# This runs both stability selection and univariate comparison on all relevant metrics

set -e

# Define comprehensive but filtered predictor set
# Includes: coverage, mean distance (not median/p90), all KL variants, MMD, normalized distances
PREDICTORS="flow_train_to_eval_over_eval_recall_logit,flow_eval_to_train_over_train_precision_logit,flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train,flow_eval_to_train_mean_dist,flow_train_to_eval_mean_dist,flow_eval_to_train_kl_div,flow_train_to_eval_kl_div,flow_eval_to_train_kl_div_hist,flow_train_to_eval_kl_div_hist,flow_eval_to_train_kl_div_hist_log1p_linear,flow_train_to_eval_kl_div_hist_log1p_linear,flow_mmd,dino_train_to_eval_over_eval_recall_logit,dino_eval_to_train_over_train_precision_logit,dino_train_to_eval_mean_dist_over_radius_eval,dino_eval_to_train_mean_dist_over_radius_train,dino_eval_to_train_mean_dist,dino_train_to_eval_mean_dist,dino_eval_to_train_kl_div,dino_train_to_eval_kl_div,dino_eval_to_train_kl_div_hist,dino_train_to_eval_kl_div_hist,dino_eval_to_train_kl_div_hist_log1p_linear,dino_train_to_eval_kl_div_hist_log1p_linear,dino_mmd"

echo "===== Running Comprehensive Stability Selection ====="
python scripts/build_leakage_free_eval.py \
  --snapshots-dir snapshots snapshots_mixed snapshots_raft \
  --mode auc \
  --metric pck \
  --auc-steps 5000 \
  --auc-pad \
  --coverage-csv coverage_faiss_flow_results_fast.csv \
  --coverage-dino-csv coverage_faiss_dino_results_fast.csv \
  --flow-mmd-csv flow_mmd_results_fast.csv \
  --dino-mmd-csv dino_mmd_results_fast.csv \
  --relative-target-baseline spair \
  --rank-target \
  --rank-target-source auc_delta \
  --rank-target-col auc_delta_rank \
  --rank-target-group benchmark \
  --rank-target-with-encoder \
  --logit-coverage \
  --rename-coverage \
  --radius-transform log \
  --linear-model ridge \
  --ridge-alpha 0.5 \
  --standardize \
  --cv-standardize-mode global \
  --cv-demean-target-by-encoder \
  --cv-demean-target-by-benchmark \
  --no-encoder-main-effects \
  --no-encoder-interactions \
  --per-encoder \
  --run-summary \
  --output-dir analysis/comprehensive/stability_all_predictors \
  --predictors "$PREDICTORS" \
  --prediction-target auc_delta_rank \
  --run-stability-selection \
  --stability-n-bootstrap 200 \
  --stability-threshold 0.7 \
  2>&1 | tee analysis/comprehensive/stability_all_predictors.log

echo ""
echo "===== Running Comprehensive Univariate Comparison ====="
python scripts/build_leakage_free_eval.py \
  --snapshots-dir snapshots snapshots_mixed snapshots_raft \
  --mode auc \
  --metric pck \
  --auc-steps 5000 \
  --auc-pad \
  --coverage-csv coverage_faiss_flow_results_fast.csv \
  --coverage-dino-csv coverage_faiss_dino_results_fast.csv \
  --flow-mmd-csv flow_mmd_results_fast.csv \
  --dino-mmd-csv dino_mmd_results_fast.csv \
  --relative-target-baseline spair \
  --rank-target \
  --rank-target-source auc_delta \
  --rank-target-col auc_delta_rank \
  --rank-target-group benchmark \
  --rank-target-with-encoder \
  --logit-coverage \
  --rename-coverage \
  --radius-transform log \
  --linear-model ridge \
  --ridge-alpha 0.5 \
  --standardize \
  --cv-standardize-mode global \
  --cv-demean-target-by-encoder \
  --cv-demean-target-by-benchmark \
  --no-encoder-main-effects \
  --no-encoder-interactions \
  --per-encoder \
  --run-summary \
  --output-dir analysis/comprehensive/univariate_all_predictors \
  --predictors "$PREDICTORS" \
  --prediction-target auc_delta_rank \
  --run-univariate-comparison \
  2>&1 | tee analysis/comprehensive/univariate_all_predictors.log

echo ""
echo "===== Analysis Complete ====="
echo "Results saved to:"
echo "  - analysis/comprehensive/stability_all_predictors/"
echo "  - analysis/comprehensive/univariate_all_predictors/"
