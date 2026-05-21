# Transfer Analysis v3 Working Summary

Date: 2026-05-21

This is a provisional synthesis of the v3 results. It is meant to help steer writing,
not to lock final claims. The flow feature refresh is still running, and the DINO
analysis is not yet in the current v3 result set.

## Current Framing

The old paper story should probably not be "directed flow coverage is the best
transfer signal." The current evidence points to a more careful story:

> Distributional similarity helps predict optical-flow transfer, but the useful
> signal is not just directed coverage. Symmetric flow distances, especially FID,
> SW2, and MMD, are highly competitive and often stronger. Dataset/profile controls
> explain a meaningful part of the signal, so the real contribution is the residual
> value of flow geometry beyond sample/profile effects.

The most important methodological shift is that MAE and rank correlation must both
be reported. Some LOTO models get reasonable absolute MAE while producing poor or
reversed Spearman, meaning they identify the performance band but not the ordering
among training datasets.

## Result Sources Read

- `results/flow_only_pure_peak_pck/results.md`
- `results/flow_only_pure_peak_pck/summary_table.csv`
- `results/flow_only_pure_auc_normalized/results.md`
- `results/flow_only_pure_auc_normalized/summary_table.csv`
- `results/flow_only_pure_drop_spair_peak_pck/results.md`
- `results/flow_only_pure_drop_spair_peak_pck/summary_table.csv`
- `results/flow_only_pure_drop_spair_auc_normalized/results.md`
- `results/flow_only_pure_drop_spair_auc_normalized/summary_table.csv`
- `results/modeling_options_auc_normalized/summary_table.csv`
- Older broad summaries under `results/auc_normalized` and `results/peak_pck`

## Strongest Current Evidence

### 1. Peak PCK gives the cleanest current story

For pure-train `peak_pck`, real flow features beat density/random controls in the
main splits. Best current absolute-scale configurations:

| Split | Best config | MAE | Spearman |
|---|---:|---:|---:|
| LOTO | `idw_prior_two_way` + `flow_mmd_only` | 7.96 | 0.477 |
| LOBO | `ridge_pairwise_cross_resid_spline` + `flow_fid_only` | 11.16 | 0.689 |
| LOCO-cell | `ridge_pairwise_cross_resid_spline` + `flow_eps` | 6.55 | 0.426 |

The built-in objective summary also reports positive real-feature lift over
density/random controls in all selected model/split comparisons: 6/6 positive for
LOTO, LOBO, and LOCO-cell.

Interpretation: flow distribution metrics are not just operating as panel/density
borrowing controls, at least for final/peak transfer performance.

### 2. Symmetric flow metrics are no longer baselines

On pure `peak_pck`, symmetric metrics often beat directed coverage/NN/KL:

| Split | Median symmetric advantage | Symmetric wins | Strongest symmetric feature |
|---|---:|---:|---|
| LOTO | +1.15 MAE | 5/6 | MMD |
| LOBO | +2.18 MAE | 6/6 | FID |
| LOCO-cell | -0.04 MAE | 2/6 | FID |

Interpretation: directed coverage remains useful, especially for LOCO-cell, but
the paper should treat FID/SW2/MMD as primary competitors or primary methods, not
as weak baselines.

### 3. AUC-normalized has a split-dependent warning

For pure `auc_normalized`, LOBO is strong and consistent:

| Split | Best config | MAE | Spearman |
|---|---:|---:|---:|
| LOBO | `ridge_pairwise_cross_resid` + `flow_fid_only` | 8.25 | 0.786 |
| LOCO-cell | `ridge_pairwise_uniform` + `flow_kl` | 7.05 | 0.699 |
| LOTO | `idw_prior_context_local` + `flow_w2_only` | 8.72 | -0.084 |

Interpretation: early-learning transfer is predictable for held-out benchmarks and
cells, but LOTO remains hard. A low LOTO MAE can hide the fact that the model ranks
unseen training datasets incorrectly.

### 4. Profile controls are strong but do not fully explain the signal

The `modeling_options_auc_normalized` run is the current best check against
train/profile confounding. It compares `profile_simple`, `train_profile_simple`,
`flow_fid_only`, `flow_w2_only`, `flow_kl`, and `motion_km`.

Best real feature vs best profile control:

| Split | Best real feature | MAE / Spearman | Best profile control | MAE / Spearman |
|---|---|---:|---|---:|
| LOTO | `flow_fid_only` | 9.23 / -0.006 | `train_profile_simple` | 10.95 / -0.488 |
| LOBO | `flow_fid_only` | 8.58 / 0.781 | `train_profile_simple` | 12.32 / 0.758 |
| LOCO-cell | `flow_fid_only` | 7.20 / 0.696 | `profile_simple` | 7.67 / 0.629 |

Interpretation: profile controls are not strawmen. They are genuinely predictive,
especially in LOBO Spearman, but flow FID still improves MAE in every split checked
here. The LOCO-cell gain is modest, so this claim needs final refresh confirmation.

### 5. SPair appears to be a major LOTO stress case

Dropped-SPair runs materially change the LOTO picture.

For pure `peak_pck`, best LOTO MAE improves from 7.96 to 5.69 when SPair is removed.
For pure `auc_normalized`, dropped-SPair LOTO becomes dominated by profile/density
or context-local effects, and Spearman remains negative for several low-MAE
settings.

Interpretation: SPair is likely not just noise. It is a diagnostic outlier tied to
sample/vector-density mismatch. The paper can use SPair to motivate profile controls
and to explain why naive coverage conclusions were unstable.

## What This Invalidates From the Older Story

- Do not claim directed flow coverage is the clear winner.
- Do not select methods by MAE alone, especially for LOTO.
- Do not treat density/sample/profile baselines as trivial controls.
- Do not use broad all-dataset `auc_normalized`/`peak_pck` summaries as the final
  story without checking whether their feature CSVs predate the current refresh.
- Do not make final appearance-vs-motion claims until DINO features are complete.

## Modeling Interpretation

The current IDW-family models appear to be good at borrowing global or context-level
performance priors. That is useful for absolute prediction, but it can mask a harder
failure mode: within a benchmark/model context, the model may predict the right
performance band while ranking the candidate training datasets backwards.

This is the core MAE-vs-Spearman tension:

- Low MAE means the model is calibrated to the right global or context mean.
- Positive Spearman means the model orders candidates correctly within the held-out
  decision set.
- Low MAE with negative Spearman means the prior is doing useful calibration, but
  the feature-dependent residual has the wrong sign or is too weak relative to the
  context effect.

This is close to the Simpson's-paradox issue: aggregate trends across benchmarks,
training sets, or model variants can look predictive because they explain large
between-context shifts, while the within-context selection problem has a different
or reversed relationship.

For writing and analysis, the cleanest decomposition is probably:

1. Context/global prior: can we predict the approximate performance level?
2. Residual ordering: after removing benchmark/model/train-set priors, do flow or
   appearance features rank alternatives correctly?
3. Decision quality: does the model choose one of the top training datasets, even
   if full rank correlation is modest?

This also explains why `auc_normalized` and `peak_pck` behave differently. AUC is
about early training dynamics and can reward fast initial adaptation; `peak_pck`
is about the best reachable performance and is sensitive to convergence, collapse,
or training length. CATS++ fine-tuning collapse would make these targets diverge:
the same dataset can be excellent early and poor later if it overfits or collapses.

## Modeling Options To Consider

More flexible regressors may help, but the data regime is small enough that generic
random forests or XGBoost could easily overfit fold identity, dataset profile, or
benchmark effects. If tried, they should be framed as diagnostics unless they win
under strict LOTO/LOBO/joint validation.

Lower-risk options:

- Keep the two-stage prior/residual setup, but report residual Spearman separately
  from absolute MAE.
- Fit residual models on within-context centered targets, then add the prior back
  only for absolute prediction.
- Use simple nonlinearities with strong regularization, such as splines/GAM-style
  terms or kernel ridge, before moving to high-capacity tree ensembles.
- Evaluate top-k decision metrics, such as whether the predicted best dataset is
  actually in the top 3, because full Spearman may be harsher than the practical
  dataset-selection objective.
- Treat SPair as both an included stress test and an ablation. If longer training
  changes SPair substantially, the current results may partly reflect compute-limited
  non-convergence rather than intrinsic transfer mismatch.

If time permits, a longer convergence sweep would clarify whether `peak_pck` is
measuring final transfer potential or merely the best point reachable under the
current short budget. This is especially important for SPair and any CATS++ variants
that spike early and then collapse.

## Candidate Paper Story

### Short version

We study whether training-free distributional similarity predicts transfer among
synthetic optical-flow datasets. The answer is yes, but the strongest evidence is
not a simple nearest-neighbor coverage story. Flow distribution metrics improve
transfer prediction beyond profile controls, while symmetric flow distances are
surprisingly competitive with, and often stronger than, directed coverage. The
remaining hard case is novel training datasets, where absolute performance bands
are easier to predict than dataset ordering.

### More cautious version

Training-free transfer prediction is possible, but the signal is partly confounded
by dataset profile. After adding sample-count and vector-density controls, flow
geometry still contributes, especially for held-out benchmarks and final
performance, but no single metric dominates across all generalization regimes.
The right conclusion is a decomposition of transfer predictability into profile
effects, symmetric distribution mismatch, directed coverage, and residual hard
cases such as SPair.

## Suggested Writing Outline

1. Problem: choosing synthetic training data without training every candidate.
2. Original hypothesis: train/eval distributional match in flow or appearance
   should predict transfer.
3. Important correction from reviews/results: naive similarity can proxy dataset
   size, benchmark difficulty, or supervision density.
4. New evaluation setup: LOTO, LOBO, and LOCO-cell, with MAE and Spearman reported
   separately.
5. Baselines: global/context priors, density/profile controls, random-neighbor IDW,
   uniform-neighbor IDW.
6. Result 1: transfer prediction is feasible, especially for LOBO and LOCO-cell.
7. Result 2: real flow features improve over profile/density controls.
8. Result 3: symmetric flow distances are first-class signals, often beating
   directed coverage.
9. Result 4: LOTO exposes the difference between calibration and ranking.
10. Result 5: SPair/vector-density mismatch explains some earlier instability.
11. Pending result: DINO appearance features may change the appearance-vs-motion
    comparison and should be integrated before final claims.
12. Conclusion: the contribution is not a single magic metric, but a controlled
    analysis of which dataset-distribution signals survive realistic transfer
    confounds.

## Figures/Tables To Build Next

- Main table: best MAE and Spearman for `peak_pck` and `auc_normalized` across
  LOTO, LOBO, LOCO-cell.
- Control table: best real flow feature vs best density/profile/random control.
- Symmetric-vs-directed table: MMD/FID/SW2 vs NN/epsilon/KL/motion.
- LOTO diagnostic plot: MAE vs Spearman for top configs, highlighting cases with
  low MAE and negative Spearman.
- SPair ablation table: with-SPair vs drop-SPair for LOTO and LOBO.
- DINO table placeholder: same structure as flow once RC extraction is complete.

## Checks Before Final Claims

- Re-run the summary after the flow refresh completes and confirm no old
  k-means/FID/SW2/MMD artifacts are mixed into the tables.
- Add DINO feature families and compare against flow under the same profile-control
  setup.
- Prefer pure-train and clearly defined mixed-train analyses separately. Do not mix
  them in one claim without saying which training universe is used.
- Check confidence intervals for the headline improvements, especially LOCO-cell
  where profile controls are close to real features.
- For LOTO, report both calibration/MAE and rank ordering. Treat negative Spearman
  as a failure for dataset selection even if MAE looks acceptable.
