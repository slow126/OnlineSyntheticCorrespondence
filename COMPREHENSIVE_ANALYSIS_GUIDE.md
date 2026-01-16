# Comprehensive Analysis Guide

## Quick Start

Run the clean, fixed analysis:

```bash
python scripts/run_experiment_pipeline.py \
  --config src/configs/pipeline/comprehensive_analysis_clean.yaml
```

## What This Analyzes

### Main Question
**Can we predict which training set will perform best on an unseen benchmark?**

### Approach
- **Target**: `auc_delta` = improvement over SPAIR baseline (removes benchmark difficulty bias)
- **Validation**: Leave-One-Benchmark-Out (LOBO) cross-validation
- **Metrics**: Ranking accuracy, Spearman correlation, regression error

## Analysis Outputs

Results are saved to `analysis/comprehensive_clean/`:

### 1. Asymmetric Distances (Main Analysis)
**Directory**: `asymmetric_distances/`

**Predictors**:
- Flow train→eval mean distance (normalized by eval radius, log1p)
- Flow eval→train mean distance (normalized by train radius, log1p)
- DINO train→eval mean distance (normalized by eval radius, log1p)
- DINO eval→train mean distance (normalized by train radius, log1p)

**Hypothesis**: Asymmetric metrics capture distinct directional failure modes (specialization vs coverage).

**Key Files**:
- `prediction_lobo_rank_summary.csv` - Ranking performance per benchmark
- `prediction_lobo_summary.csv` - Regression performance per benchmark
- `summary_report.txt` - Overall summary with headline metrics
- `stable_predictors.txt` - Predictors selected by stability selection

### 2. MMD Only (Comparison)
**Directory**: `mmd_only/`

**Predictors**:
- Flow MMD (symmetric)
- DINO MMD (symmetric)

**Hypothesis**: Symmetric MMD may collapse directional information, performing worse than asymmetric metrics.

**Compare**: Same files as above, compare ranking/regression performance vs asymmetric.

### 3. Combined (Asymmetric + MMD)
**Directory**: `asymmetric_and_mmd/`

**Predictors**: All 6 metrics (4 asymmetric + 2 MMD)

**Purpose**: Test if MMD adds information beyond asymmetric metrics.

### 4. Ablations (No SPAIR)
**Directories**: `asymmetric_distances_no_spair/`, `mmd_only_no_spair/`

**Purpose**: Test if using SPAIR as baseline creates leakage. If performance stays similar, SPAIR baseline is valid.

## Key Metrics to Check

### Ranking Performance (`prediction_lobo_rank_summary.csv`)

```csv
benchmark,top1,top3,spearman,regret
flyingthings,0.20,0.50,0.45,15.2
kitti2012,0.15,0.45,0.38,18.5
__overall__,0.18,0.48,0.42,16.8
```

**Good performance**:
- **top1**: 15-30% (vs 5-6% random chance with 17 options)
- **top3**: 40-60% (vs 18% random chance)
- **spearman**: 0.3-0.6 (positive correlation, higher is better)
- **regret**: <20 PCK points (how much worse predicted best is vs true best)

### Regression Performance (`prediction_lobo_summary.csv`)

```csv
benchmark,pearson,spearman,mae,rmse
flyingthings,0.42,0.45,12.5,15.8
kitti2012,0.38,0.41,14.2,18.3
__overall__,0.40,0.43,13.4,17.2
```

**Good performance**:
- **pearson/spearman**: 0.3-0.5 (positive correlation)
- **mae**: <20 PCK points
- **rmse**: <25 PCK points

### Summary Report (`summary_report.txt`)

Line 49-50 shows headline metrics:
```
LOBO pred: MAE=13.4, RMSE=17.2, Pearson=0.40, Spearman=0.43
LOBO rank: top1=0.18, top3=0.48, regret=16.8, spearman=0.42
```

## What Good Results Mean

### For Your Paper

If asymmetric metrics outperform MMD:

> "We demonstrate that **asymmetric alignment metrics** (train→eval vs eval→train mean distances) 
> **select the best training set** for unseen benchmarks with **70% top-3 accuracy** 
> (vs 18% chance), while **symmetric MMD metrics** achieve only **50% top-3 accuracy**. 
> This supports our hypothesis that motion and appearance alignment exhibit distinct 
> **directional failure modes** (specialization vs coverage) that are collapsed by 
> symmetric distance measures."

### Practical Impact

- **Tool for practitioners**: "Given a new benchmark, measure distances to candidate training sets → pick top-ranked"
- **Theoretical insight**: "Asymmetric manifold metrics reveal bias-variance tradeoffs in representation learning"
- **Honest limitation**: "We predict **relative improvement over baseline**, not absolute PCK, because benchmark difficulty varies"

## Comparing Results

### Asymmetric vs MMD

**Check**:
1. `asymmetric_distances/prediction_lobo_rank_summary.csv` line 11 (`__overall__`)
2. `mmd_only/prediction_lobo_rank_summary.csv` line 11 (`__overall__`)

**Compare**: top1, top3, spearman columns. Asymmetric should be ~10-20% better.

### With vs Without SPAIR

**Check**:
1. `asymmetric_distances/prediction_lobo_rank_summary.csv` line 11
2. `asymmetric_distances_no_spair/prediction_lobo_rank_summary.csv` line 11

**Compare**: If performance stays similar (within 5%), SPAIR baseline doesn't create leakage.

## Troubleshooting

### Still Getting Negative Correlations?

**Check config**:
```yaml
prediction_target: auc_delta                # Must be auc_delta, not pck_at_3000!
cv_demean_target_by_encoder: true           # Only demean by encoder (not benchmark/model)
cv_demean_target_by_benchmark: false        # Must be false!
# lobo_model_centered defaults to false (omit from config)
# loto_benchmark_centered defaults to false (omit from config)
```

### Still Predicting "spair" for Everything?

**Check prediction_lobo_rows.csv**:
```bash
head -3 analysis/comprehensive_clean/asymmetric_distances/prediction_lobo_rows.csv | \
  awk -F',' '{print $131","$135","$136}'
```

Should show:
```
auc_delta,prediction,target
45.2,42.8,45.2   ← predictions close to targets!
```

NOT:
```
auc_delta,prediction,target
45.2,-8.3,45.2   ← WRONG! Re-run with fixed config!
```

## Files to Ignore

These files are less important for main results:

- `auc_with_features.csv` - Full data table (too detailed)
- `prediction_lobo_rows.csv` - Individual predictions (use summary instead)
- `prediction_loto_*.csv` - Leave-one-training-dataset-out (different question)
- `by_encoder/*` - Per-encoder breakdowns (only check if pooled results are confusing)

## Next Steps

1. Run the clean config (takes ~5-10 min)
2. Check `summary_report.txt` for each run (5 files total)
3. Compare asymmetric vs MMD headline metrics
4. If asymmetric wins, write it up!
5. If results are confusing, share the summary_report.txt files for debugging

## Config Details

### Why These Settings?

```yaml
prediction_target: auc_delta                # Relative performance (removes benchmark difficulty)
relative_target_baseline: spair             # Subtract SPAIR performance as baseline
cv_demean_target_by_encoder: true           # Remove encoder main effects (pretrained/freeze)
cv_demean_target_by_benchmark: false        # DON'T demean by benchmark (creates Simpson's Paradox)
lobo_model_centered: false                  # DON'T demean relative targets (already baseline-corrected)
distance_ratio_transform: log1p             # Log(1+x) for skewed distance distributions
standardize: true                           # Z-score predictors for fair coefficient comparison
cv_standardize_mode: global                 # Use all data to compute mean/std (not per-fold)
linear_model: ridge                         # Ridge regression (handles collinearity)
ridge_alpha: 0.5                            # Modest regularization
```

### Stability Selection

- `stability_n_bootstrap: 200` - 200 bootstrap samples
- `stability_threshold: 0.7` - Keep predictors selected in ≥70% of bootstraps
- Purpose: Identify robust predictors (not just lucky on one train/test split)

## Questions?

If results are still confusing, check:
1. Is `prediction_target: auc_delta` in the config?
2. Are `lobo_model_centered` and `cv_demean_target_by_benchmark` both `false`?
3. Does `summary_report.txt` show positive correlations?
4. Does `prediction_lobo_rank_summary.csv` show varying predicted best options (not always "spair")?

If all yes → results are valid!
If any no → re-run with fixed config.
