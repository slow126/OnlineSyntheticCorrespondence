# Transfer Analysis v3 — Agent Handoff

Briefing for an agent continuing work on this pipeline. Read fully before starting.

---

## Current state as of 2026-05-21

The flow feature refresh was reworked after repeated overnight failures in the raw
FAISS coverage job. The raw directed flow features used by the current sweep are now
materialized from `analysis_v3/pairwise_self_distances.csv` instead of recomputing the
huge raw FAISS coverage search. This is intentional:

- `pairwise_self_distances.csv` already has all 110 pure train/eval flow pairs.
- It contains `mean_nn_*`, epsilon coverage at 1/4/16px, and KL at k=5/20.
- `build_table.py` already reads `flow_kl` from this file.
- The legacy `analysis_v3/kl_flow_features.csv` path is not needed for the v3 sweep.

New helper:
`scripts/transfer_analysis_v3/materialize_flow_raw_coverage_from_pairwise.py`

Current feature status after materialization:

| Family | Source | Current status |
|---|---|---|
| raw flow NN/eps (`flow_nn`, `flow_eps`, `motion`) | `analysis/coverage_v2_flow_only_raw_joint_full.csv` materialized from pairwise self | complete, 0/110 missing |
| flow KL (`flow_kl`) | `analysis_v3/pairwise_self_distances.csv` | complete, 0/110 missing |
| k-means flow (`flow_km`, `motion_km`) | `analysis/coverage_v2_flow_only_raw_joint_kmeans_full.csv` | needs refresh |
| flow FID/SW2 | `analysis_v3/symmetric_distances.csv` | needs refresh |
| flow MMD | `flow_mmd_results_fast.csv` | old/incomplete; needs refresh |

The old CSVs were not deleted. Scratch refresh archives them as `.bak_<timestamp>`.
Known archives include old raw/kmeans files from 2026-02-03 and old FID/SW2 from
2026-05-15.

Important run-pipeline flags added recently:

- `FLOW_REFRESH_ONLY=1`: refresh/audit features and exit before model sweeps.
- `FLOW_REFRESH_MODE=scratch|append`: archive old feature CSVs or resume existing partials.
- `FLOW_REFRESH_PARALLEL=0|1`: optionally run k-means and FID/SW2 in parallel.
- `FLOW_KMEANS_CUDA_VISIBLE_DEVICES`, `FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES`,
  `FLOW_MMD_CUDA_VISIBLE_DEVICES`: per-stage GPU assignment.
- `FLOW_DIAGNOSTIC_FEATURE_GROUPS`: shell override for diagnostic feature groups.
- `FLOW_AUDIT_REQUIRE_FAMILIES`: shell override for which feature families must be complete.

After a crash in `calculate_coverage_faiss_flow_kmeans.py`, a refactor bug was fixed:
the script no longer references removed `train_vectors`/`eval_vectors` variables. It now
loads one dataset at a time, builds/loads its k-means codebook, frees vectors, and appends
each pair result immediately.

If continuing the full feature refresh, run in `screen` on GPU 1:

```bash
FLOW_ONLY=1 \
FLOW_REFRESH_ONLY=1 \
REFRESH_FLOW_FEATURES=1 \
REFRESH_FLOW_MMD=1 \
FLOW_REFRESH_MODE=append \
FLOW_REFRESH_PARALLEL=0 \
FLOW_KMEANS_CUDA_VISIBLE_DEVICES=1 \
FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES=1 \
FLOW_MMD_CUDA_VISIBLE_DEVICES=1 \
REQUIRE_FLOW_AUDIT_CLEAN=1 \
FLOW_AUDIT_REQUIRE_FAMILIES="flow_raw_coverage flow_kmeans_coverage flow_fid_sw2 flow_mmd pairwise_self_flow_train_eval" \
bash scripts/transfer_analysis_v3/run_pipeline.sh
```

Use `append` here unless deliberately starting over; `scratch` will archive partial CSVs.

While the refresh runs, it is reasonable to run a reduced model sweep on GPU 0 using only
currently complete feature families:

```bash
CUDA_VISIBLE_DEVICES=0 \
FLOW_ONLY=1 \
FLOW_SWEEP_MODE=diagnostic \
FLOW_SPLITS="loto lobo joint_cell" \
TARGETS="auc_normalized peak_pck" \
REFRESH_FLOW_FEATURES=0 \
REFRESH_FLOW_MMD=0 \
REQUIRE_FLOW_AUDIT_CLEAN=1 \
FLOW_AUDIT_REQUIRE_FAMILIES="flow_raw_coverage pairwise_self_flow_train_eval" \
FLOW_DIAGNOSTIC_FEATURE_GROUPS="density_train density_eval density_idw random_idw sample_count vector_density_simple train_profile_simple profile_simple flow_nn flow_eps motion flow_kl flow_kl_profile" \
CLEAN_RESULTS=1 \
bash scripts/transfer_analysis_v3/run_pipeline.sh
```

For a full feature sweep after the refresh audit is clean, restore the broader diagnostic
feature groups including `motion_km`, `flow_fid_only`, `flow_w2_only`, and their profile
composites.

---

## What the project is about

Building a **transfer predictor**: given a synthetic training dataset and an evaluation
benchmark, predict which training dataset will produce the best-performing optical flow
model — without actually training. The predictor uses distributional similarity features
(coverage, NN distance, KL divergence, FID, Wasserstein, etc.) computed between training
dataset and benchmark flow/DINO feature distributions.

**Core question:** Which distributional signal best predicts transfer performance?
**Current question:** Are flow/motion similarity features genuinely predictive, or are
they mostly proxying sample density / supervision density? Recent changes added explicit
sample-count and vectors-per-sample controls, plus non-IDW models, to separate three
possibilities: row/column/model-variant effects, density/profile confounds, and real
train/eval similarity geometry.

**Paper goal under revision:** Show which distributional signals add predictive value
beyond trivial dataset profile controls. Do not assume directed flow coverage beats
symmetric baselines; recent peak-PCK results suggest FID/SW2/KL and density/profile
controls can be surprisingly competitive. Also do not select by MAE alone: LOTO
can show low absolute MAE with reversed Spearman, meaning the model predicts the
right performance band but the wrong ordering.

---

## Experimental setup

### Models evaluated
- RAFT and CATS++ variants (pretrained=True/False, freeze=True/False)
- Each trained on ~25 datasets, evaluated on ~10 benchmarks
- `context_id` = `(benchmark, model_variant)` — this is the unit of Spearman evaluation

### Training datasets
**Pure (11):** flyingthings, imagenet2dwarp, movi_f, pointodyssey, sintel, spair,
synthetic, synthetic_2d_warp, synthetic_large_zoom, synthetic_random_flipping,
synthetic_small_zoom

**Mixed (14+):** e.g. spair_synthetic_30_70, flyingthings_synthetic_70_30, etc.

**Important:** The current transfer table includes all 25 datasets → loco_cell has
250 folds (25 × 10), not 99. If you want 99 folds (pure only), rebuild the table
with `--train-datasets` restricted to the 11 pure names.

### Benchmarks (~10)
flyingthings test, KITTI 2012/2015, SPair-71k, PF-Pascal, PF-Willow, PointOdyssey,
TSS, Middlebury (exact set varies by model variant).

### Target metrics
- **auc_normalized**: trapz(PCK, steps) / 5000 over first 5000 training steps (early)
- **peak_pck**: max PCK over full training run (final)

### Evaluation splits
| Split | What's held out | Folds | Primary use |
|-------|----------------|-------|-------------|
| `loto` | One training dataset | 25 | Hardest: novel training data |
| `lobo` | One benchmark | 10 | Realistic: novel eval domain |
| `loco_cell` | One (train, benchmark) cell | 250 | "Real use case" |
| `joint_cell` | One (train, benchmark) cell, with both train and benchmark axes excluded from fit | ~110 pure | Strict joint generalization diagnostic |
| `loto_grouped` | Grouped training variants | ~11 | Robustness check |
| `loco` | One context_id (bench×variant) | ~250 | Slow, rarely needed |
| `lomo` | One model family | ~4 | Model-type generalisation |

**Current primary splits to focus on: `loto`, `lobo`, `joint_cell`.**
`loco_cell` is informative but slower and got less scrutiny; do not run it by default
while the main architecture sweep is still moving.

---

## Feature groups

### Directed (asymmetric) — flow space
| Group | Measures | # cols | Status |
|-------|----------|--------|--------|
| `flow_nn` | Mean NN dist eval→train + train→eval | 2 | ✓ available |
| `flow_eps` | ε-coverage at 1/4/16 px both dirs | 6 | ✓ available |
| `flow_km` | K-means density-weighted ε-coverage | 6 | refresh pending after scratch archive |
| `flow_kl` | kNN KL divergence k=5,20 both dirs | 4 | ✓ in pairwise_self_distances.csv |

### Directed — DINO appearance space
| Group | Measures | # cols | Status |
|-------|----------|--------|--------|
| `dino_nn` | Mean NN dist in DINO space | 2 | ✓ available |
| `dino_cov` | Null-calibrated cosine coverage | 2 | ✓ available |
| `dino_kl` | kNN KL divergence in DINO space | 4 | ✓ available |

### Symmetric baselines
| Group | Contents | # cols |
|-------|----------|--------|
| `sym_flow` | flow MMD + flow FID + flow SW2 | 3 |
| `flow_mmd_only`, `flow_fid_only`, `flow_w2_only` | individual symmetric metrics | 1 each |

### Composites
- `motion` = flow_nn + flow_eps (8 cols)
- `motion_km` = flow_nn + flow_km (8 cols)
- `appearance` = dino_nn + dino_cov (4 cols)

### Density / confound baselines
| Group | Features | IDW distance | Purpose |
|-------|----------|-------------|---------|
| `density_eval` | log(N_eval) | \|log N_a − log N_b\| | Is model proxying benchmark difficulty? |
| `density_train` | log(N_train) | \|log N_a − log N_b\| | Is model proxying training set size? |
| `density_idw` | log(N_train) + log(N_eval) | \|log N_a − log N_b\| | Both size features, size IDW |
| `random_idw` | random_train + random_eval | shuffled random distances | Pure mechanism baseline — zero real signal |

**All density feature groups use size-based IDW** (`log_n_dist` = |log N_a − log N_b|),
consistent with how flow feature groups use flow-space distances for IDW.
`random_idw` uses shuffled `mean_nn_sym` values (seed=42) — same distribution, random assignment.

### Vector-profile controls (new)
These were added after noticing SPair/Sintel are major LOTO outliers and may differ mostly
in supervision/vector density. The old `log_*_n_vectors` columns are total vector counts,
not average vectors per sample.

Stats source:
`/mnt/nvme_1tb_b/coverage_vectors/stats/flow_counts_{dataset}_{split}_{space}.json`

`build_table.py` now prefers `*_flow.json` and falls back to `*_dino.json` when flow stats
are not present. The JSONs include `images_seen`, `images_with_zero`, `total_valid_vectors`,
`total_sampled_vectors`, `total_vectors_retained`, and `valid_counts` summaries.

New feature groups:
| Group | Features | IDW distance | Purpose |
|-------|----------|--------------|---------|
| `sample_count` | log train/eval image counts | `sample_count_dist` | Is sample count enough? |
| `sample_count_train` / `sample_count_eval` | one axis only | `sample_count_dist` | Axis-specific count controls |
| `vector_density` | vectors/sample summaries train+eval | `vector_density_dist` | Supervision density control |
| `vector_density_train` / `vector_density_eval` | one axis only | `vector_density_dist` | Axis-specific density controls |
| `train_profile` | train sample count + train vectors/sample | `profile_dist` | Main train-set profile control |
| `eval_profile` | eval sample count + eval vectors/sample | `profile_dist` | Main benchmark profile control |
| `profile_density` | train+eval sample counts + vectors/sample | `profile_dist` | Strongest profile-only control |

Composite feature groups for "does flow add anything after density?":
| Group | Contents |
|-------|----------|
| `flow_mmd_profile` | `flow_mmd_only` + `train_profile` |
| `flow_fid_profile` | `flow_fid_only` + `train_profile` |
| `flow_w2_profile` | `flow_w2_only` + `train_profile` |
| `flow_kl_profile` | `flow_kl` + `train_profile` |
| `motion_km_profile` | `flow_nn` + `flow_km` + `train_profile` |

Important sanity check from the rebuilt pure table:
- SPair: ~6.95 valid vectors/image (`log_train_valid_vectors_per_sample ≈ 2.07`)
- PointOdyssey: ~2,436 valid vectors/image
- Sintel: ~248,711 valid vectors/image
- FlyingThings/MOVI-F: ~262,000 valid vectors/image

This is likely relevant to the SPair low / Sintel high LOTO outliers.

---

## Models

### Ranking models (predict relative rank scores, not absolute AUC)
- **`ridge`**: RidgeCV on within-context rank scores. Fast, good Spearman baseline.
- **`bradley_terry`**: Pairwise logistic on score differences.
- **`plackett_luce`**: Listwise PL loss (torch). Slow with many features — 200 epochs.
- **`global_prior`**: Mean rank per training dataset from training fold. Strong LOBO baseline (0.62 Spearman).
- **`random`**: Random scores averaged over 1000 seeds. MAE ~30.4.

**Note:** ridge/BT/PL produce [0,1] rank scores. MAE vs absolute AUC is meaningless for them.

### Absolute models (predict actual AUC values)
- **`ridge_abs`**: Plain RidgeCV on the absolute target. Correct vanilla Ridge baseline.
  LOTO MAE ~15–18 depending on feature group. flow_km is consistently best (15.6).
- **`ridge_pairwise`**: Coupled Ridge + IDW. Feature group determines both the ridge
  input features AND the IDW neighbor distance (see `_PAIRWISE_FEATURE_GROUP_COLS`).
  LOTO MAE ~10.1–10.6 regardless of feature group — IDW dominates.
  Correctly handles LOTO and LOBO: builds neighbor lists from all datasets in test_df
  (not just fold datasets), so held-out training datasets and benchmarks both get IDW.
- **`ridge_pairwise_cross_resid`**: 2-axis IDW that subtracts source benchmark mean before
  borrowing across similar benchmarks, then adds eval-neighbor benchmark prior back.
- **`ridge_pairwise_cross_resid_spline`**: spline basis version of the residual ridge.
  Implemented with `SplineTransformer(n_knots=4, degree=3)`. Early results showed almost
  no benefit; do not include by default unless specifically testing nonlinear residuals.
- **`idw_prior_residual`**: Two-stage: LOO IDW prior (additive) + global ridge on residuals.
  Prior uses `_idw_both` combining train-side and eval-side IDW with "has specific data" flags.
  - LOTO test: train-side has benchmark-specific perf → use train-side IDW only
  - LOBO test: eval-side has per-td perf for similar benchmarks → use eval-side IDW only
  - Training rows (LOO): both sides have data → average train-side and eval-side
  Stage 2: RidgeCV on (y − prior) using flow features only.
- **`idw_prior_context`**: Two-stage: context_mean[(j, mv)] prior + global ridge.
  `context_mean` = mean_n y(n, j, mv) — removes both benchmark difficulty AND model-variant scale.
  For LOBO (novel benchmark): IDW over similar in-fold benchmarks' context_means via eval-eval distances.
  Residual ridge sees only within-context variation — exactly what Spearman measures.
- **`idw_prior_context_local`**: Same context_mean prior + per-benchmark local RidgeCVs (≥5 rows).
  Falls back to global ridge from IDWPriorContextModel for unseen benchmarks (LOBO).
  Captures benchmark-specific feature importance when slopes vary across benchmarks.
- **`krr_tp_flow_nn`**: Tensor product KRR on flow-space pairwise kernel.
  **Broken for LOTO** (mean kernel fallback → ~-0.9 Spearman inversion). Works for LOBO/loco_cell.
- **`idw_prior_two_way`**: Axis-aware prior combining train-axis and eval-axis IDW, then
  residual Ridge. This is currently the most coherent absolute model for the global-prior
  plus local-residual idea.
- **`idw_prior_two_way_spline`**, **`uniform_prior_two_way_spline`**, **`random_prior_two_way_spline`**:
  spline residual versions. Mostly diagnostic; expensive and not clearly better.
- **`two_way_mixed_ridge`**: Regularized two-way effects approximation. Ridge on selected
  numeric features plus one-hot train dataset, benchmark, and model-variant effects
  (`model_family`, `pretrained`, `freeze`). This is the stable row/column/variant baseline.
  It does not learn geometry for a novel LOTO train dataset; unseen one-hot levels fall
  back to numeric features + known axes.
- **`anchor_bilinear_ridge`**: Non-IDW learned interpolation baseline. Represents each
  train dataset by distances/similarities to in-fold train anchors, each eval benchmark
  by distances/similarities to in-fold eval anchors, then fits Ridge on
  train anchors, eval anchors, and their full outer product. This tests whether the
  similarity space is useful beyond inverse-distance weighting. It is high-variance and
  should be treated as diagnostic; full bilinear/random-control designs can be ill-conditioned.
- **`anchor_additive_ridge`**: Anchor model without train×eval interactions. Lower-capacity
  comparison for checking whether the full bilinear model is memorizing cells.
- **`anchor_lowrank_bilinear_ridge`**: Additive anchors plus low-rank PCA interaction
  features. Preferred bilinear-style compromise when overfit risk is a concern.
- **`anchor_bilinear_shrunk_ridge`**: Full bilinear features with stronger Ridge alphas.
- **`kernel_mixed_additive`**: Conservative additive kernel mixed-effects model:
  `K = wt*K_train + we*K_eval + wv*K_variant`. Uses the same feature-group→pairwise-distance
  mapping as IDW, so flow/density/random controls are directly comparable. This is the
  most principled robust non-IDW model to inspect first.
- **`kernel_mixed_interaction`**: Same as `kernel_mixed_additive`, with a capped
  train×eval interaction `wi*(K_train*K_eval)`. Interaction weights are intentionally small
  (`0.05`, `0.10`, `0.25`) and selected by inner CV to avoid the pure TP-KRR overfit mode.

### Which to run
```
Baselines (feature-group-independent, run once with any feature group):
  global_prior, random

Feature ablation models (run across all feature groups):
  ridge, ridge_abs

Coupled absolute calibration (run across feature groups, no cross product):
  two_way_mixed_ridge
  anchor_bilinear_ridge
  kernel_mixed_additive, kernel_mixed_interaction
  ridge_pairwise, ridge_pairwise_cross_resid
  idw_prior_residual, idw_prior_context, idw_prior_context_local
  idw_prior_two_way

Optional diagnostic:
  krr_tp_flow_nn (LOBO/loco_cell only)

For the current density-vs-geometry question, use the narrower diagnostic set:
```
Models:
  ridge_abs
  two_way_mixed_ridge
  anchor_bilinear_ridge
  kernel_mixed_additive
  kernel_mixed_interaction
  ridge_pairwise
  idw_prior_two_way
  uniform_prior_two_way
  random_prior_two_way

Feature groups:
  density_train density_eval density_idw random_idw
  sample_count vector_density_simple train_profile_simple profile_simple
  flow_fid_only flow_w2_only flow_kl motion_km
  flow_fid_profile flow_w2_profile flow_kl_profile motion_km_profile
```

The FLOW_ONLY pipeline now defaults to this narrow diagnostic grid and writes to
`results/flow_only_pure_diagnostic_<target>` when `PURE_ONLY=1`.

Profile controls are intentionally simple and saturated by default:
- `vector_density_simple` = capped log valid vectors/image for train + eval
- `train_profile_simple` = train sample count + capped train valid vectors/image
- `profile_simple` = train/eval sample counts + capped train/eval valid vectors/image

The cap is 10,000 valid vectors/image before `log1p`, so SPair-like sparse labels remain
distinct while fully dense flow datasets are treated as effectively dense enough. The old
high-dimensional `vector_density`, `train_profile`, and `profile_density` groups are still
available for manual diagnostics but should not be used as the default LOTO profile control;
they caused unstable residual-Ridge extrapolation.

Run it with:
```
FLOW_ONLY=1 bash scripts/transfer_analysis_v3/run_pipeline.sh
```

Use `FLOW_SWEEP_MODE=full` only when you explicitly want the old broad sweep. After profile
controls were added, that full IDW block is roughly
`16 models × ~26 feature groups × 3 splits × 2 targets`, with many folds per split.

---

## Key architectural insights

### IDW dominance and the benchmark-difficulty explanation
The IDW mechanism provides ~20 MAE improvement over global mean (30.4 → 10.4) even with
**completely random neighborhoods** (`random_idw` LOTO = 10.42). The mechanism works because:
- For LOTO, train-side IDW computes a **per-benchmark weighted average** of in-fold performances
- Even with random neighbor weights, this approximates the per-benchmark mean → much better than global mean
- Flow-based IDW (10.35) vs random IDW (10.42) = only 0.07 MAE difference

**Hierarchy:**
| Setup | LOTO MAE |
|---|---|
| Global mean | 30.4 |
| Random IDW (pure mechanism) | 10.42 |
| Size IDW (`density_idw`) | 11.7 |
| Flow IDW (`flow_km`) | 10.6 |

93% of the IDW gain comes from the mechanism itself; only 7% from flow-distance quality.

### Features DO carry real signal — but only visible in ridge_abs
In `ridge_abs` (features only, no IDW): `flow_km` (15.6) clearly beats `density_eval` (18.6),
which beats global mean (30.4). Flow coverage features are genuinely predictive.
But once IDW is added, all feature groups collapse to ~10.1–10.6 — IDW swamps the feature signal.
`idw_prior_residual` and `idw_prior_context` are designed to fix this by using IDW as an offset,
not a co-feature: Stage 1 captures benchmark difficulty, Stage 2 uses features to predict the residual.

### Two-stage model design intent
The design goal is to **isolate** what flow features know beyond benchmark-difficulty effects:
- Stage 1 prior: what can collaborative filtering tell us? (IDW over similar training datasets)
- Stage 2 residual: given the prior, can flow features predict above/below-average transfer?

For an **unseen benchmark (LOBO)**: Stage 1 uses eval-side IDW to interpolate benchmark difficulty
from similar known benchmarks. Stage 2 still predicts a residual — the question is whether features
carry signal about how THIS training dataset does relative to others on similar benchmarks.

For an **unseen training dataset (LOTO)**: Stage 1 uses train-side IDW (quality from similar datasets).
Stage 2 uses features to correct the interpolation.

### Scale mismatch at train vs test time
At training time (LOO), both train-side and eval-side IDW have in-fold data → good prior → small
residuals. At test time (LOTO/LOBO), one side degenerates to cross-entity mean fallback → weaker
prior → larger residuals. Ridge trained on small residuals may undercorrect at test. Practically:
this preserves Spearman direction (sign/magnitude rank order), may slightly inflate MAE. Check
empirically whether stage 2 ridge actually helps vs stage 1 alone.

### Density/profile confound — open question again
Earlier `density_eval`/`density_train` controls suggested flow features were not just
proxying total vector count. However, total vector count was too crude. The new profile
controls test average vectors per sample and sample count separately.

The key comparison is:
```
train_profile or profile_density
vs
flow_fid_profile / flow_w2_profile / flow_kl_profile / motion_km_profile
```
If profile-only controls match the profile+flow composites, the "flow" signal may mostly be
supervision density. If composites improve meaningfully, flow geometry adds real signal.

### LOBO vs LOTO — not contradictory
- **LOBO (0.62 Global Prior)**: "Given I've seen all training datasets on other benchmarks,
  predict ranking on a new benchmark." Easy because training dataset rankings are stable
  across benchmarks (cross-benchmark Spearman ~0.47).
- **LOTO (0.33 best)**: "Given I've never benchmarked this training dataset anywhere,
  predict from distribution alone." Hard because training dataset explains only 9.5% of
  AUC variance (benchmark identity explains 54%).

### Variance decomposition
- Benchmark identity alone: R² = 0.543
- Training dataset alone: R² = 0.095
- Model variant alone: R² = 0.076
- All three: R² = 0.716
- After removing benchmark effects: training dataset R² = 0.208

This is why LOTO is hard and why LOBO looks easy.

### TP-KRR LOTO failure
For novel training datasets, `TensorProductKRRModel` falls back to `K.mean(axis=0)` for
the training kernel row. Unusual datasets (e.g. `spair`) get slightly higher kernel values
when removed, causing systematic sign inversion (~-0.9 Spearman). Not a bug — fundamental
limitation of mean fallback. Only use TP-KRR results for LOBO/loco_cell.

---

## Recommended run commands

Standard 4-run set (two targets × with/without spair):
```bash
# With spair
FLOW_ONLY=1 PURE_ONLY=1 TARGETS="auc_normalized" \
  bash scripts/transfer_analysis_v3/run_pipeline.sh

FLOW_ONLY=1 PURE_ONLY=1 TARGETS="peak_pck" \
  bash scripts/transfer_analysis_v3/run_pipeline.sh

# Drop spair (robustness check)
FLOW_ONLY=1 PURE_ONLY=1 TARGETS="auc_normalized" DROP_TRAIN_DATASETS="spair" \
  bash scripts/transfer_analysis_v3/run_pipeline.sh

FLOW_ONLY=1 PURE_ONLY=1 TARGETS="peak_pck" DROP_TRAIN_DATASETS="spair" \
  bash scripts/transfer_analysis_v3/run_pipeline.sh
```

Pipeline is resumable — already-done experiments are skipped automatically.
Pipeline runs `compile_results.py` at the end; no separate compile step needed.

### Faster diagnostic command for the current question
Use this instead of the full pipeline when checking density/profile vs flow geometry
and the new non-IDW models:
```bash
for TARGET in auc_normalized peak_pck; do
  OUT_DIR="scripts/transfer_analysis_v3/results/flow_only_pure_diagnostic_${TARGET}"

  python scripts/transfer_analysis_v3/run_experiments.py \
    --splits loto lobo joint_cell \
    --models ridge_abs two_way_mixed_ridge \
             anchor_additive_ridge anchor_lowrank_bilinear_ridge \
             anchor_bilinear_ridge anchor_bilinear_shrunk_ridge \
             kernel_mixed_additive kernel_mixed_interaction \
             ridge_pairwise idw_prior_two_way idw_prior_two_way_rank \
             uniform_prior_two_way random_prior_two_way \
    --feature-groups density_train density_eval density_idw random_idw \
                     sample_count vector_density_simple \
                     train_profile_simple profile_simple \
                     flow_fid_only flow_w2_only flow_kl motion_km \
                     flow_fid_profile flow_w2_profile \
                     flow_kl_profile motion_km_profile \
    --pairwise-spaces flow \
    --self-dist-csv analysis_v3/pairwise_self_distances.csv \
    --target "$TARGET" \
    --output-dir "$OUT_DIR"

  python scripts/transfer_analysis_v3/compile_results.py \
    --results-dir "$OUT_DIR" \
    --output "$OUT_DIR/results.md" \
    --mi-csv "analysis_v3/feature_mi_${TARGET}/feature_mi.csv"
done
```

Pipeline equivalent for canonical reports:
```bash
FLOW_ONLY=1 FLOW_SWEEP_MODE=diagnostic FLOW_SPLITS="loto lobo joint_cell" \
  TARGETS="auc_normalized peak_pck" \
  bash scripts/transfer_analysis_v3/run_pipeline.sh
```

Do not set `CLEAN_RESULTS=1` when resuming a crashed run; completed experiments are
skipped automatically when their `metrics.csv` already exists.

---

## Code changes made across sessions

### `scripts/transfer_analysis_v3/run_experiments.py`

**1. flow_eps bug fix:**
`flow_eps` was accidentally including `_weighted` columns, giving 12 features instead of 6.
Fixed by filtering `if not c.endswith("_weighted")`.

**2. loco_cell split added** (`iter_loco_cell_folds`):
Holds out one `(train_dataset, benchmark)` cell. 250 folds with full table.

**3. Model-variant-aware IDW** (`RidgePairwiseDistModel`):
`_perf_lookup` stores `(td, bm, model_family, pretrained, freeze)` first, then family, then pair.

**4. MAE/RMSE added to evaluation** (`evaluate_context`):
`mae` and `rmse` fields added. Only meaningful for absolute-scale models.

**5. Density feature groups added** (`resolve_feature_groups`):
`density_train`, `density_eval`, `density_idw`, `random_idw` — see table above.
`random_idw` uses `random_train`/`random_eval` features (random scalars from `build_table.py`).

**6. `_PAIRWISE_FEATURE_GROUP_COLS` updated:**
- `density_train`, `density_eval`, `density_idw` → `("log_n_dist", False)` (size IDW)
- `random_idw` → `("random_dist", False)` (shuffled random IDW)
- `_REVERSE_COL` entries added for both `log_n_dist` and `random_dist` (symmetric)

**7. `IDWPriorResidualModel` added and fixed:**
Two-stage: LOO IDW prior (`_idw_both`) + ridge on residuals (flow features only).

Key bugs fixed this session:
- **`train_nbr` built from `fold_trains` instead of `all_trains`**: test-time neighbor lookup
  only contained in-fold datasets → held-out training datasets (LOTO) returned `[]` from
  `train_nbr.get(held_out_td, [])` → IDW degenerated to `eval_mean` fallback for all LOTO
  test rows. Fixed: `all_trains = list(dict.fromkeys(df["train_dataset"]))` — now covers
  every dataset appearing in the test dataframe, including held-out ones.
- **Same fix for eval-side**: `all_evals = list(dict.fromkeys(df["benchmark"]))` similarly.
- **`_idw_both` helper added**: combines train-side and eval-side IDW using "has specific data"
  flags. LOTO → train-side only; LOBO → eval-side only; training rows → average of both.
  Previously, only train-side IDW was implemented, so LOBO test rows had no eval-side
  interpolation of benchmark difficulty from similar benchmarks.
- **Training prior correctly uses LOO IDW**: an earlier session had changed the training prior
  to `eval_mean` to compensate for the broken test-time IDW (they degenerated to the same
  fallback, so residuals were scale-consistent but both stages were broken). Now both training
  and test use IDW; scale mismatch is an accepted limitation (see above).

**8. `IDWPriorContextModel` added** (inherits from `IDWPriorResidualModel`):
Prior = `context_mean[(bm, mv)]` = mean_n y(n, bm, mv) over all in-fold training datasets.
Removes both benchmark-difficulty AND model-variant scale. Residuals are exactly the
within-context signal that Spearman measures.
For LOBO (held-out benchmark not in `context_mean`): IDW over similar in-fold benchmarks'
context_means using eval-eval distances.
`fit()` calls `super().fit()` for setup, then computes `_context_mean`, then refits `_model`
with context-mean residuals.

**9. `IDWPriorContextLocalModel` added** (inherits from `IDWPriorContextModel`):
Per-benchmark RidgeCVs (≥5 rows each) on context-mean residuals.
`predict_score_df` starts with global model predictions and overrides with local models for
known benchmarks. LOBO falls back to global ridge automatically.

**10. `GENERIC_PAIRWISE_MODELS` and `MODEL_CLASSES` updated:**
Added `"idw_prior_context"` and `"idw_prior_context_local"`.

**11. Spline residual models added:**
- `ridge_pairwise_cross_resid_spline`
- `idw_prior_two_way_spline`
- `uniform_prior_two_way_spline`
- `random_prior_two_way_spline`

Implementation uses `SplineTransformer` before `RidgeCV` in the residual ridge stage.
Early peak-PCK result: spline barely helped and sometimes hurt. Treat as diagnostic only.

**12. Vector-profile feature groups added:**
`resolve_feature_groups()` now includes sample-count, vectors/sample, train/eval profile,
profile-only controls, and profile+flow composites. See "Vector-profile controls" above.

**13. Profile-based IDW distances added at experiment time:**
`add_profile_distance_columns()` derives:
- `sample_count_dist`
- `vector_density_dist`
- `profile_dist`

These are computed separately for `train_train` and `eval_eval` rows in the self-distance
DataFrame using standardized Euclidean distance over the corresponding profile columns.
This prevents profile-control IDW from accidentally using total-vector `log_n_dist`.

### `scripts/transfer_analysis_v3/build_table.py`

**Density supplement block** (after density_df is built):
Supplements missing density rows (movi_f, synthetic/val) from `n_vecs` columns in
`pairwise_self_distances.csv`. These datasets were added after the coverage CSV was built.

**Random scalar columns** (added after density join):
`random_train` and `random_eval` — seeded (seed=42) random normals, one per unique
train dataset and benchmark. Used by `random_idw` as a zero-signal feature baseline.

**Vector profile stats loader added:**
- Parses `/mnt/nvme_1tb_b/coverage_vectors/stats/flow_counts_*.json`
- Handles dataset names with underscores by parsing `{split}_{space}` from the right
- Prefers `space=flow`, falls back to `space=dino`
- Adds 24 profile columns to the pure transfer table:
  - `log_train_n_samples`, `log_eval_n_samples`
  - `log_*_valid_vectors_per_sample`, sampled/retained vectors per sample
  - valid vector mean/median/p10/p90/p95
  - sampled vector mean/median
  - `train_zero_image_frac`, `eval_zero_image_frac`

The pure rebuilt table had 540/540 matches for these profile columns.

### `scripts/transfer_analysis_v3/compile_results.py`

**1. `sharey=True` bug fix** in `_fig4_model_bar`:
Shared y-axis caused LOBO tick labels to overwrite LOTO. Fixed to `sharey=False`.

**2. Density display names** added to `FEATURE_DISPLAY`, `FEATURE_LEGEND`, `FEATURE_ORDER`.

**3. `table_density_confound_check`** function added:
Shows MAE for `ridge_abs` vs `ridge_pairwise` vs `idw_prior_residual` across density
and flow feature groups. Appears as §4.x in results.md.

**4. Two-stage models added** to display:
- `idw_prior_residual`, `idw_prior_context`, `idw_prior_context_local` added to
  `MODEL_ORDER`, `ABSOLUTE_MODELS`, `MODEL_DISPLAY`, `MODEL_COLORS`.
- Added to `_fig1_scatter` and `_fig5_calibration` hardcoded lists.
- `table_idw_prior_variants()`: new function showing MAE + Spearman for all 3 two-stage
  variants + `ridge_pairwise` as reference. Appears as §4.y in results.md.

**5. Profile controls added to display/reporting:**
`FEATURE_DISPLAY`, `FEATURE_LEGEND`, and `FEATURE_ORDER` include all profile groups.
`table_density_confound_check()` now includes profile controls and profile+flow composites.
The objective summary treats profile-only groups as null/control features and profile+flow
groups as real feature candidates.

### `scripts/transfer_analysis_v3/run_pipeline.sh`

FLOW_ONLY branch model list updated to include all three two-stage models:
```bash
--models ridge_pairwise ridge_pairwise_cross_resid \
         idw_prior_residual idw_prior_context idw_prior_context_local \
```

Later update: FLOW_ONLY now also includes spline models and the full profile-control
feature grid. This is useful for exhaustive sweeps, but too large for quick diagnosis.
For iterative work, use the narrower command above.

**Recent model additions and stability fix:**
- Added `two_way_mixed_ridge`, `anchor_bilinear_ridge`,
  `kernel_mixed_additive`, and `kernel_mixed_interaction`.
- `kernel_mixed_additive` / `kernel_mixed_interaction` are the preferred principled
  non-IDW tests. They use additive train/eval/model-variant kernels, with only a
  capped train×eval interaction in the interaction variant.
- **`idw_prior_two_way_rank`**: Axis-aware two-way prior plus a residual ranking stage.
  LOTO ranks training datasets within `(benchmark, model_variant)` contexts; LOBO ranks
  benchmarks within `(train_dataset, model_variant)` contexts; `joint_cell` uses both.
  Keeps absolute predictions while adding a rank-aware residual correction.
- `anchor_bilinear_ridge` is diagnostic and can overfit. A run crashed at
  `loco_cell / anchor_bilinear_ridge / random_idw` with
  `numpy.linalg.LinAlgError: SVD did not converge`. The shared ridge fitter now
  sanitizes non-finite columns and falls back to `Ridge(alpha=100, solver="lsqr")`
  if `RidgeCV` SVD fails. Exact failing model/feature/split was re-tested through
  all `loco_cell` folds and completed.

Logs for current flow-only runs live in:
`scripts/transfer_analysis_v3/logs/flow_only_*.log`

If `pipeline.log` shows `step0d_pairwise_self_distances`, the non-flow-only pipeline
was started; the flow-only diagnostic logs are the `flow_only_*` files.

Recent refresh-control updates:
- `VEC_DIR` default fixed to `/mnt/nvme_1tb_b/coverage_vectors` under `set -u`.
- `FLOW_REFRESH_ONLY=1` lets the feature refresh/audit run without launching model sweeps.
- `FLOW_REFRESH_MODE=scratch` archives existing feature CSVs as `.bak_<timestamp>`;
  `append` resumes partial current CSVs. Use `append` after any crash.
- `FLOW_REFRESH_PARALLEL=1` can run k-means and FID/SW2 concurrently, but the current
  recommendation is one refresh GPU and one model-sweep GPU.
- Per-stage GPU variables are available:
  `FLOW_KMEANS_CUDA_VISIBLE_DEVICES`, `FLOW_SYMMETRIC_CUDA_VISIBLE_DEVICES`,
  `FLOW_MMD_CUDA_VISIBLE_DEVICES`.
- Raw flow coverage refresh no longer calls `calculate_coverage_faiss_v2.py`; it calls
  `materialize_flow_raw_coverage_from_pairwise.py` instead.

### `scripts/transfer_analysis_v3/materialize_flow_raw_coverage_from_pairwise.py`

New helper that writes a legacy-shaped raw flow coverage CSV from
`analysis_v3/pairwise_self_distances.csv`. It writes the columns used by `flow_nn`,
`flow_eps`, and `motion`, with 110 pure train/eval rows. This avoids the raw FAISS
coverage job that repeatedly crashed in the legacy KL/GPU path.

### `scripts/calculate_coverage_faiss_flow_kmeans.py`

Refactored to avoid loading all flow vectors into RAM. It now:
- loads one dataset at a time, using mmap when enabled;
- normalizes/transforms just that dataset;
- trains or loads its k-means codebook;
- saves the codebook and frees raw vectors;
- computes k-means pair metrics from centroids/weights only;
- appends each pair row and curve rows immediately.

Bug fixed after refactor: removed stale references to deleted `train_vectors` and
`eval_vectors` variables in the eager space-transform block.

### `scripts/transfer_analysis_v3/compute_symmetric_distances.py`

FID/SW2 now flushes every pair instead of every 20 pairs, so partial runs are resumable
with at most one pair of lost work.

### `scripts/calculate_flow_mmd.py`

Flow MMD now appends each pair result as it is computed and skips cached pairs on resume.

### `analysis_v3/pairwise_self_distances.csv`

Two new columns added directly (no recomputation needed):
- `log_n_dist` = `|log(1 + n_vecs_a) − log(1 + n_vecs_b)|` — size-based distance
- `random_dist` = shuffled `mean_nn_sym` within each (space, pair_type) group, seed=42

---

## File locations

| What | Path |
|------|------|
| Transfer table (features × AUC) | `scripts/transfer_analysis_v3/transfer_table.csv` |
| Pipeline script | `scripts/transfer_analysis_v3/run_pipeline.sh` |
| Build table | `scripts/transfer_analysis_v3/build_table.py` |
| Experiments runner | `scripts/transfer_analysis_v3/run_experiments.py` |
| Results compiler | `scripts/transfer_analysis_v3/compile_results.py` |
| Current diagnostic results (auc_normalized) | `scripts/transfer_analysis_v3/results/flow_only_pure_diagnostic_auc_normalized/` |
| Current diagnostic results (peak_pck) | `scripts/transfer_analysis_v3/results/flow_only_pure_diagnostic_peak_pck/` |
| Older results (auc_normalized, with spair) | `scripts/transfer_analysis_v3/results/flow_only_pure_auc_normalized/` |
| Older results (auc_normalized, drop spair) | `scripts/transfer_analysis_v3/results/flow_only_pure_drop_spair_auc_normalized/` |
| Report | `results/<variant>/results.md` |
| Flow coverage CSV | `analysis/coverage_v2_flow_only_raw_joint_full.csv` |
| Pairwise self-distances (+ log_n_dist, random_dist) | `analysis_v3/pairwise_self_distances.csv` |
| Symmetric distances | `analysis_v3/symmetric_distances.csv` |
| Feature MI (auc_normalized) | `analysis_v3/feature_mi_auc_normalized/feature_mi.csv` |
| Coverage vectors | `/mnt/nvme_1tb_b/coverage_vectors/` |
| Vector profile stats | `/mnt/nvme_1tb_b/coverage_vectors/stats/flow_counts_*.json` |

---

## Pending work

1. **Resume or rerun the focused density-vs-geometry diagnostic.**
   The current canonical diagnostic includes `ridge_abs`, mixed/bilinear/kernel
   models, `ridge_pairwise`, `idw_prior_two_way`, and uniform/random null priors.
   Resume without `CLEAN_RESULTS=1` after crashes:
   ```bash
   FLOW_ONLY=1 FLOW_SWEEP_MODE=diagnostic TARGETS="auc_normalized peak_pck" \
     OUT_PREFIX=flow_only_pure_diagnostic \
     bash scripts/transfer_analysis_v3/run_pipeline.sh
   ```

2. **Interpret the model ladder, not individual table-cell winners.**
   - If `two_way_mixed_ridge` wins, row/column/model-variant effects dominate.
   - If `kernel_mixed_additive` beats density/random controls, similarity geometry helps
     without strict IDW.
   - If `kernel_mixed_interaction` beats additive, true train×eval compatibility helps.
   - If only `ridge_pairwise`/`idw_prior_two_way` wins, the hand-structured borrowing
     is doing useful work.
   - Reject any "winner" with strongly negative LOTO Spearman unless the question is
     strictly absolute-band calibration.

3. **Check density/profile confounds explicitly.**
   For every split and target, compare flow groups against `density_idw`, `random_idw`,
   `vector_density_simple`, `train_profile_simple`, and `profile_simple`. Flow should
   beat random/density controls in the same model family before claiming geometry matters.

4. **DINO extraction** — add `dino_nn`, `dino_cov`, `dino_kl` features to the pipeline
   once DINO feature extraction is complete. Likely a new feature group `appearance` or
   `motion_appearance` composite.
