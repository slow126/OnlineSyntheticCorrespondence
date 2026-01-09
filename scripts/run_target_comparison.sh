#!/usr/bin/env bash
set -euo pipefail

OUT_BASE="analysis/leakage_free_local_fast_dino_faiss/target_compare"
PREDICTORS="flow_train_to_eval_coverage_logit,flow_eval_to_train_coverage_logit,flow_train_to_eval_mean_dist_over_radius_eval,flow_eval_to_train_mean_dist_over_radius_train,flow_mean_dist_asymmetry,dino_train_to_eval_coverage_logit,dino_eval_to_train_coverage_logit,dino_train_to_eval_mean_dist_over_radius_eval,dino_eval_to_train_mean_dist_over_radius_train,dino_mean_dist_asymmetry,flow_mmd,dino_mmd"

COMMON_ARGS=(
  --snapshots-dir snapshots snapshots_mixed
  --mode auc
  --metric pck
  --auc-steps 5000
  --auc-pad
  --coverage-csv coverage_faiss_flow_results_fast.csv
  --coverage-resnet-csv coverage_resnet_results_fast.csv
  --coverage-dino-csv coverage_faiss_dino_results_fast.csv
  --flow-mmd-csv flow_mmd_results_fast.csv
  --feature-mmd-csv feature_mmd_results_fast.csv
  --dino-mmd-csv dino_mmd_results_fast.csv
  --predictors "${PREDICTORS}"
  --include-kl
  --linear-model ridge
  --ridge-alpha 1.0
  --standardize
  --distance-radius-norm none
  --radius-transform log
  --radius-eps 1e-6
  --distance-radius-floor 0.01
  --exclude-encoder-configs FT
  --no-encoder-interactions
  --encoder-main-effects
  --per-encoder
  --run-summary
  --sanity-permutation
  --sanity-permute-group benchmark
  --sanity-permute-seed 17
  --no-rank-target
)

mkdir -p "${OUT_BASE}"

echo "=== Mode A: auc_delta (baseline-relative, continuous) ==="
python3 scripts/build_leakage_free_eval.py \
  --output-dir "${OUT_BASE}/auc_delta" \
  --relative-target-baseline spair \
  "${COMMON_ARGS[@]}"

echo "=== Mode B: peak_pck (benchmark-centered + encoder-demeaned) ==="
python3 scripts/build_leakage_free_eval.py \
  --output-dir "${OUT_BASE}/peak_pck_centered" \
  --target peak_pck \
  --prediction-target peak_pck \
  --benchmark-normalize-target center \
  --benchmark-normalize-scope all \
  --cv-demean-target-by-encoder \
  "${COMMON_ARGS[@]}"

echo "=== Summarizing ==="
python3 scripts/summarize_target_comparison.py \
  --out "${OUT_BASE}/summary_comparison.csv" \
  --label auc_delta "${OUT_BASE}/auc_delta/summary_report.txt" \
  --label peak_pck_centered "${OUT_BASE}/peak_pck_centered/summary_report.txt"
