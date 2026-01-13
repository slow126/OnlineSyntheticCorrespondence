# Usage Guide - Comprehensive Predictor Analysis

## Quick Start

### 1. Test the New Univariate Comparison

Test with a small subset of predictors:

```bash
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
  --skip-prediction \
  --output-dir analysis/test/univariate_quick \
  --predictors "flow_mmd,dino_mmd,flow_eval_to_train_kl_div,dino_eval_to_train_kl_div" \
  --prediction-target auc_delta_rank \
  --run-univariate-comparison
```

**Output:** `analysis/test/univariate_quick/univariate_predictor_comparison.csv`

Results show each predictor's standalone LOBO performance:
- `lobo_pearson`: Pearson correlation
- `lobo_spearman`: Spearman rank correlation (preferred for rank targets)
- `lobo_mae`: Mean absolute error
- `lobo_rmse`: Root mean squared error

---

### 2. Run Comprehensive Stability Selection

Test stability of all predictors with 200 bootstrap iterations:

```bash
bash scripts/run_comprehensive_analysis.sh
```

This will run both:
1. **Stability Selection**: Identifies predictors consistently selected across 200 bootstrap samples
2. **Univariate Comparison**: Tests each predictor individually

**Outputs:**
```
analysis/comprehensive/stability_all_predictors/
  - stability_selection_results.csv    # Selection frequency for each predictor
  - stable_predictors.txt               # Comma-separated list of stable predictors (>70% selection rate)
  
analysis/comprehensive/univariate_all_predictors/
  - univariate_predictor_comparison.csv # Univariate LOBO performance
```

---

### 3. Interpret Results

#### Stability Selection Results

```csv
predictor,selection_frequency,stable
flow_mmd,0.89,True
flow_eval_to_train_kl_div,0.75,True
flow_train_to_eval_mean_dist,0.45,False
```

- **selection_frequency**: Fraction of bootstrap samples where predictor was selected by Lasso
- **stable**: True if frequency > threshold (default 0.7)

#### Univariate Comparison Results

```csv
predictor,n_obs,lobo_pearson,lobo_spearman,lobo_mae,lobo_rmse
flow_mmd,1700,0.129,0.145,12.81,15.49
```

- **n_obs**: Number of complete observations
- **lobo_spearman**: Primary metric for ranking (higher is better)
- Sort by `lobo_spearman` to identify best standalone predictors

---

### 4. Use Results in Downstream Analysis

After identifying stable predictors:

```bash
# 1. Extract stable predictors
STABLE_PREDS=$(cat analysis/comprehensive/stability_all_predictors/stable_predictors.txt)

# 2. Run horse race with only stable predictors
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
  --output-dir analysis/final/stable_predictors_only \
  --predictors "$STABLE_PREDS" \
  --prediction-target auc_delta_rank
```

---

## Key CLI Arguments

### New Arguments

- `--run-univariate-comparison`: Fit each predictor individually in separate LOBO runs
- `--run-stability-selection`: Run stability selection with Lasso on bootstrap samples
- `--stability-n-bootstrap`: Number of bootstrap iterations (default: 100, recommended: 200)
- `--stability-threshold`: Selection frequency threshold for stability (default: 0.7)

### Existing Critical Arguments

- `--cv-demean-target-by-encoder`: Demean target by `model_family_encoder` (fixes RAFT bug)
- `--cv-demean-target-by-benchmark`: Demean target by benchmark difficulty
- `--cv-standardize-mode`: `global` (standardize once on full training set) or `local` (per fold)
- `--no-encoder-main-effects`: Exclude encoder dummy variables from predictors
- `--no-encoder-interactions`: Exclude encoder × predictor interactions

---

## Predictor Naming Convention

### Flow Metrics
- `flow_train_to_eval_over_eval_recall_logit`: Coverage (train→eval) with logit transform
- `flow_eval_to_train_over_train_precision_logit`: Coverage (eval→train) with logit transform
- `flow_train_to_eval_mean_dist_over_radius_eval`: Normalized mean distance (train→eval)
- `flow_eval_to_train_mean_dist_over_radius_train`: Normalized mean distance (eval→train)
- `flow_eval_to_train_mean_dist`: Raw mean distance (eval→train)
- `flow_eval_to_train_kl_div`: KL divergence using k-NN estimator
- `flow_eval_to_train_kl_div_hist`: KL divergence using histogram estimator
- `flow_eval_to_train_kl_div_hist_log1p_linear`: KL divergence with log1p + linear binning
- `flow_mmd`: Maximum Mean Discrepancy

### DINO Metrics
Same patterns with `dino_` prefix instead of `flow_`

---

## Troubleshooting

### "Warning: Failed to fit {predictor}: 'r2'"
**Fixed!** This was due to `run_group_cv` not returning an `r2` column. Now using `mae` and `rmse` instead.

### Empty results in univariate comparison
Check that:
1. Predictor names match columns in the merged dataframe
2. Sufficient non-NaN observations exist (need at least 10 per predictor)
3. Target column `auc_delta_rank` exists

### RAFT models grouped incorrectly
**Fixed!** Now using `model_family_encoder` composite column that correctly separates RAFT (no encoder variants) from CatsPP (has encoder variants).

---

## Example Workflow

```bash
# Step 1: Quick test with 3 predictors
python scripts/build_leakage_free_eval.py \
  [...base args...] \
  --predictors "flow_mmd,dino_mmd,flow_eval_to_train_kl_div" \
  --run-univariate-comparison \
  --skip-prediction \
  --output-dir analysis/test/quick

# Step 2: If successful, run full comprehensive analysis
bash scripts/run_comprehensive_analysis.sh

# Step 3: Review results
cat analysis/comprehensive/stability_all_predictors/stable_predictors.txt
cat analysis/comprehensive/univariate_all_predictors/univariate_predictor_comparison.csv

# Step 4: Run final analysis with stable predictors
STABLE=$(cat analysis/comprehensive/stability_all_predictors/stable_predictors.txt)
python scripts/build_leakage_free_eval.py \
  [...base args...] \
  --predictors "$STABLE" \
  --output-dir analysis/final/paper_results
```

---

## Performance Notes

- **Univariate Comparison**: Fast (~2-3 minutes for 26 predictors)
- **Stability Selection (200 bootstraps)**: Slow (~30-60 minutes for 26 predictors)
- Consider reducing `--stability-n-bootstrap` to 100 for faster testing

---

## Further Documentation

- See `FIXES_IMPLEMENTED.md` for technical details on what was changed
- See `.cursor_plan.md` for the overall project plan and experimental design
