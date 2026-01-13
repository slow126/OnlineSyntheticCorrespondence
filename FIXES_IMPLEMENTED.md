# Fixes Implemented - Comprehensive Predictor Analysis

## Date: 2026-01-12

## Summary of Changes

This document summarizes the fixes implemented to address issues with the leakage-free evaluation pipeline, specifically:
1. RAFT encoder config bug causing incorrect demeaning
2. Missing comprehensive predictor testing in stability selection
3. Missing univariate predictor comparison functionality

---

## 1. Fixed RAFT Encoder Demeaning Bug

### Problem
When demeaning the target by encoder configuration, RAFT models (which don't have encoder variants) were being incorrectly grouped with CatsPP models, leading to deflated correlations and incorrect target normalization.

### Solution
Added `create_model_family_encoder_column()` function that creates a composite column combining `model_family` and `encoder_config`:
- For models with encoder variants (e.g., CatsPP): `model_family_encoder = "catspp_FF"`, `"catspp_TT"`, etc.
- For models without encoder variants (e.g., RAFT): `model_family_encoder = "raft"`

### Files Modified
- `scripts/build_leakage_free_eval.py`:
  - Added `create_model_family_encoder_column()` function (line ~2407)
  - Updated `_select_target_demean_groups()` to use `model_family_encoder` instead of `encoder_config` (line ~2540)
  - Added call to create composite column in main data loading (line ~4421)

### Testing
```python
# Test case verified:
# Input: CatsPP with FF/TT encoders, RAFT with empty encoder
# Output: catspp_FF, catspp_TT, raft, raft
# ✓ Correctly groups RAFT separately from CatsPP variants
```

---

## 2. Added Univariate Predictor Comparison

### Problem
The existing analysis only tested predictors in multivariate models, making it difficult to assess individual predictor importance without confounding from correlated variables.

### Solution
Added `compute_univariate_predictor_comparison()` function that:
- Fits each predictor individually in separate LOBO runs
- Reports standalone performance (Pearson, Spearman, MAE, RMSE)
- Helps identify which predictors are useful on their own

### Files Modified
- `scripts/build_leakage_free_eval.py`:
  - Added `compute_univariate_predictor_comparison()` function (line ~1270)
  - Added `--run-univariate-comparison` CLI argument (line ~3666)
  - Integrated into `run_analysis_bundle()` (line ~3099)

### Example Usage
```bash
python scripts/build_leakage_free_eval.py \
  --predictors "flow_mmd,dino_mmd,flow_eval_to_train_kl_div" \
  --prediction-target auc_delta_rank \
  --run-univariate-comparison \
  --output-dir analysis/univariate_test
```

### Output Format
```csv
predictor,n_obs,lobo_pearson,lobo_spearman,lobo_mae,lobo_rmse
flow_mmd,1700,0.129,0.145,12.81,15.49
dino_mmd,1700,0.090,0.137,12.73,15.55
```

---

## 3. Comprehensive Predictor List for Stability Analysis

### Problem
The initial stability selection and family comparison were only testing a small, hand-picked subset of predictors, excluding important metrics like KL divergence variants.

### Solution
Created a comprehensive but filtered predictor list that includes:

**Flow Metrics:**
- Coverage/precision/recall (logit transformed)
- Mean distance (absolute and normalized by radius)
- All KL divergence variants (kl_div, kl_div_hist, kl_div_hist_log1p_linear, etc.)
- MMD

**DINO Metrics:**
- Same set as flow

**Excluded (to avoid redundancy):**
- Median and p90 distances (highly correlated with mean)
- Raw radius values (using log-transformed versions)
- Coverage without logit transform

### Example Comprehensive Predictor List
```
flow_train_to_eval_over_eval_recall_logit
flow_eval_to_train_over_train_precision_logit
flow_train_to_eval_mean_dist_over_radius_eval
flow_eval_to_train_mean_dist_over_radius_train
flow_eval_to_train_mean_dist
flow_train_to_eval_mean_dist
flow_eval_to_train_kl_div
flow_train_to_eval_kl_div
flow_eval_to_train_kl_div_hist
flow_train_to_eval_kl_div_hist
flow_eval_to_train_kl_div_hist_log1p_linear
flow_train_to_eval_kl_div_hist_log1p_linear
flow_mmd
[... plus DINO equivalents ...]
```

---

## 4. Helper Script for Comprehensive Analysis

Created `scripts/run_comprehensive_analysis.sh` that runs both:
1. Stability selection with 200 bootstrap iterations on all predictors
2. Univariate comparison on all predictors

### Usage
```bash
bash scripts/run_comprehensive_analysis.sh
```

### Outputs
```
analysis/comprehensive/stability_all_predictors/
  - stability_selection_results.csv
  - stable_predictors.txt
  
analysis/comprehensive/univariate_all_predictors/
  - univariate_predictor_comparison.csv
```

---

## Key Benefits

1. **Correct Demeaning**: RAFT and CatsPP models are now properly separated during target demeaning, leading to more accurate variance partitioning.

2. **Comprehensive Testing**: All relevant distance metrics, including KL divergence variants, are now properly evaluated for stability and predictive power.

3. **Clearer Predictor Importance**: Univariate comparison allows assessment of individual predictor utility without multicollinearity confounding.

4. **Reproducible Analysis**: The comprehensive analysis script provides a standardized way to evaluate all predictors systematically.

---

## Next Steps

1. Run comprehensive stability selection and univariate comparison with full predictor set
2. Identify stable predictors from stability selection results  
3. Use stable predictors in final horse race analysis
4. Generate publication-ready tables with `scripts/generate_paper_tables.py`

---

## Verification Tests Passed

✓ `create_model_family_encoder_column()` correctly handles CatsPP and RAFT models
✓ `compute_univariate_predictor_comparison()` runs without errors and produces valid output
✓ All changes pass Python linter (no errors)
✓ Test run with 6 predictors completes successfully across all encoder configurations
