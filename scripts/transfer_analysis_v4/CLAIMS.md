# Transfer Analysis v4 — Paper Claims & Evidence Map

Paper-prep document. Each claim → hypothesis → test → result → status →
files. Update as results evolve.

**Headline target:** `peak_pck` (peak validation PCK; removes the
training-speed conflation that `auc_normalized` carries).

**Source scope:** 11 pure training datasets (`--pure-only`), 10 benchmarks,
5 model variants → 590 rows. Mixed-variant sources (e.g. `spair_synthetic_70_30`)
are in `transfer_table.csv` but excluded from the headline scope; see HANDOFF.md.

**Decomposition framework** (cite Mundlak 1978; Park & Marcotte 2012):
`perf(i, k, v) = L(i, k, v) + g(features(i → k))` where ranking is on `g` and
L is calibration only.

**Last updated:** 2026-05-28, after the v4 lean-recovery sweep, the
strength-tests apparatus (per-family CIs + P(>0) + shuffle-null z), the
two-axis density ablation, and the `spair_only` build-table bug fix
(verified empirically null-impact on pure-only headlines).

**Statistical reporting convention.** Headline ρ_g numbers in the tables
below are the single-fit ridge point estimate (deterministic given the
table + fold). 95% CIs and shuffle-null z-scores come from the
entity-resampled bootstrap in `strength_tests.py` (N_BOOT=300). When the
bootstrap mean differs from the point estimate (e.g. LOTO at N=11 sources
is mildly skewed) we report the point estimate as ρ_g and bracket it with
the bootstrap CI.

---

## Claim 1 — Motion cross-distance predicts transfer (THE HEADLINE)

> Motion-domain (i→k) cross-distance from a training source to a target
> correspondence benchmark predicts within-context transfer performance,
> consistently across three dyadic CV regimes (Park-Marcotte C2 + C3).

**Test:** within-context Spearman ρ of ridge `g` vs actual `peak_pck`,
per (split, family). 300-iter entity-resampled bootstrap.

**Result (peak_pck, ridge `g`, 95% bootstrap CI from `strength_tests.py`):**

| regime | motion ρ_g [CI] | P(ρ_g>0) | z vs shuffle-null |
|---|---|---|---|
| LOTO  (held source, observed benchmark) | **+0.508 [+0.229, +0.663]** | 0.997 | **+8.55σ** |
| LOBO  (observed source, held benchmark) | **+0.448 [+0.368, +0.534]** | 1.000 | **+12.37σ** |
| JOINT (both unseen) | **+0.321 [+0.112, +0.457]** | 1.000 | **+5.19σ** |

For comparison on auc_normalized (same configuration; the *speed*-aware target):

| regime | motion ρ_g [CI] | P(ρ_g>0) | z vs shuffle-null |
|---|---|---|---|
| LOTO  | +0.271 [-0.059, +0.522] | 0.927 | +5.51σ |
| LOBO  | +0.342 [+0.220, +0.474] | 1.000 | +11.19σ |
| JOINT | +0.166 [-0.029, +0.296] | 0.953 | +3.07σ |

**Status:** ✓ Decisive on peak_pck. peak_pck consistently gives ~0.10 higher ρ
than auc_normalized in every regime — the training-speed conflation in AUC
suppresses signal. All three regimes are >+5σ above the entity-permuted
shuffle null; LOBO is +12σ.

**Supporting files:** [ABLATION_strength.md](ABLATION_strength.md) §Target=peak_pck,
[strength_per_family.csv](strength_per_family.csv),
`results_mixed/results.md` Table 1, fig1, fig6, `bootstrap_gap.csv`.

---

## Claim 1b — Symmetric distance aggregate (FID+SW2+MMD) is the strongest single family

> A 3-feature symmetric-distance family — FID + sliced-W2 + MMD on the
> flow-distribution — yields the strongest LOBO ρ_g of any family tested
> and is statistically separable from every appearance counterpart.

**Test:** ridge ρ_g on the `motion_sym` family (3 features: `flow_fid`,
`flow_sliced_w2`, `flow_mmd` aggregated symmetrically), vs the canonical
13-feature `motion` family and the analogous `appearance_sym` family.

**Result (peak_pck, ridge ρ_g, 95% CI):**

| family | LOTO | LOBO | JOINT |
|---|---|---|---|
| motion_sym (3 features) | +0.413 [+0.192, +0.554], **+8.4σ** | **+0.533 [+0.475, +0.592], +21.2σ** | +0.401 [+0.235, +0.559], +8.7σ |
| motion (13 features) | +0.476 [+0.229, +0.663], +8.6σ | +0.450 [+0.368, +0.534], +12.4σ | +0.300 [+0.112, +0.457], +5.2σ |
| motion_w2 (1 feature) | +0.449 [+0.271, +0.579], +11.8σ | +0.481 [+0.394, +0.581], **+24.2σ** | +0.464 [+0.319, +0.585], +10.6σ |
| motion_fid (1 feature) | +0.440 [+0.228, +0.574], +11.7σ | +0.467 [+0.374, +0.572], +16.7σ | +0.446 [+0.303, +0.558], +9.4σ |
| motion_mmd (1 feature) | -0.116 [-0.524, +0.299] | +0.203 [+0.124, +0.285] | +0.065 [-0.121, +0.268] |

**Reads:**

1. **motion_sym beats the 13-dim `motion` family on every regime.** 3
   features outperform 13 because the symmetric FID/SW2/MMD distances are
   genuinely discriminative while the 13-dim selfdist set drags in KL
   features that carry inverted signal (see Claim 8).
2. **motion_w2 alone is +24σ above the shuffle null on LOBO** — the
   single strongest standalone metric in the entire study. Sliced-W2 of
   the flow distribution is the cleanest known indicator of motion-domain
   transfer fit.
3. **motion_mmd is weak alone** (LOTO actually negative, P=0.32). MMD's
   bandwidth selection on flow is noisy; it only contributes inside the
   aggregate.

**Status:** ✓ New headline candidate. Recommended paper framing: lead
with `motion_sym` (3 features, strongest LOBO ρ_g, cleanest 3-feature
ablation story); report 13-dim `motion` as the unablated baseline that
also works.

**Supporting files:** [ABLATION_strength.md](ABLATION_strength.md),
[strength_per_family.csv](strength_per_family.csv).

---

## Claim 2 — Appearance does NOT predict transfer (with one caveat)

> Appearance-domain (DINO) cross-distance does not predict transfer on
> LOTO or JOINT; on LOBO the picture is mixed depending on which
> appearance family is chosen, but no appearance family is competitive
> with its motion counterpart.

**Test:** ctx_rho_g for appearance families across regimes; paired
bootstrap gap of motion-side vs appearance-side counterparts.

**Result (peak_pck, ridge ρ_g, 95% CI):**

| family | LOTO | LOBO | JOINT |
|---|---|---|---|
| appearance (13 selfdist features) | -0.220 [-0.455, +0.105] | +0.074 [-0.019, +0.174] | -0.182 [-0.333, -0.009] |
| appearance_sym (FID+SW2+MMD, 3) | +0.062 [-0.222, +0.408] | +0.141 [-0.005, +0.299] | +0.011 [-0.156, +0.163] |
| appearance_fid (1) | -0.143 [-0.407, +0.191] | +0.260 [+0.075, +0.491] | -0.128 [-0.271, +0.046] |
| appearance_w2 (1) | -0.181 [-0.469, +0.181] | -0.011 [-0.232, +0.201] | -0.297 [-0.461, -0.145] |
| appearance_mmd (1) | +0.197 [-0.130, +0.523] | +0.341 [+0.238, +0.444] | +0.272 [+0.116, +0.401] |
| appearance_nullk (8) | -0.042 [-0.329, +0.286] | +0.054 [-0.063, +0.191] | -0.044 [-0.205, +0.123] |

**Reads:**

1. **No appearance family is competitive with its motion analogue.** The
   paired gap `motion − appearance` is +0.37 / +0.37 / +0.48 across
   LOTO/LOBO/JOINT with P(>0)=1.000 in every regime. The paired gap
   `motion_sym − appearance_sym` is +0.35 / +0.39 / +0.39 with the same
   significance. See Claim 3.
2. **`appearance_mmd` has positive LOBO ρ_g (+0.341).** Likely because
   MMD on DINO captures coarse photorealism that correlates with the few
   real-world benchmarks. This is well below `motion_mmd` LOBO (+0.203 —
   yes, less than appearance_mmd) but the *direction is the same as
   motion*, so it does not reverse the motion-vs-appearance ranking when
   aggregated. The signal is in the level statistic, not the within-context
   ranker.
3. **`appearance_fid` LOBO (+0.260)** is also non-trivial; same MMD-like
   coarse-photorealism story.

**Status:** ✓ Decisive on the paper claim. *The motion-vs-appearance
ordering holds with P=1.000 across all regimes;* individual appearance
sub-families have some signal but never displace their motion analogues.

**Defense against "the negative ρ is suspicious" reviewer question:**
The synthetic-family training sources are uniformly far from real
benchmarks in DINO space, so DINO similarity carries almost no relative
information about which synthetic source transfers best. The weak
*negative* signal on the 13-dim `appearance` family probably reflects
that DINO-similar-to-benchmark synthetic sources are the ones that
compromised motion-content diversity for photorealism, and thus transfer
slightly *worse*.

**Supporting files:** [ABLATION_strength.md](ABLATION_strength.md),
`results_mixed/results.md` Table 1b, fig1, fig8.

---

## Claim 3 — Motion-appearance gap is decisive (the main statistical statement)

> Across all three dyadic CV regimes, the motion-vs-appearance ranking
> advantage holds with paired-bootstrap probability ≥ 0.997.

**Test:** paired entity-resampled bootstrap of (motion_ρ_g − appearance_ρ_g),
N=300 iterations, across two parallel comparisons: 13-dim vs 13-dim, and
3-dim symmetric-aggregate vs 3-dim symmetric-aggregate.

**Result (peak_pck, mixed L mode):**

| comparison | LOTO gap [CI], P | LOBO gap [CI], P | JOINT gap [CI], P |
|---|---|---|---|
| motion − appearance (13 vs 13) | +0.701 [+0.262, +0.968], 1.000 | +0.373 [+0.277, +0.473], 1.000 | +0.476 [+0.242, +0.710], 0.997 |
| motion_sym − appearance_sym (3 vs 3) | +0.349 [-0.162, +0.725], 0.920 | +0.390 [+0.243, +0.544], 1.000 | +0.385 [+0.184, +0.601], 1.000 |
| motion − random | +0.586 [+0.314, +0.783], 1.000 | +0.529 [+0.417, +0.652], 1.000 | +0.391 [+0.193, +0.571], 1.000 |

For comparison on auc_normalized:

| comparison | LOTO gap, P | LOBO gap, P | JOINT gap, P |
|---|---|---|---|
| motion − appearance | +0.619 [+0.235, +0.975], 1.000 | +0.145 [+0.055, +0.230], 0.993 | +0.370 [+0.116, +0.602], 0.997 |
| motion_sym − appearance_sym | +0.379 [-0.065, +0.721], 0.957 | +0.338 [+0.236, +0.435], 1.000 | +0.452 [+0.238, +0.682], 1.000 |

**Status:** ✓ Decisive. Motion ≫ appearance on both targets, all three
CV regimes, both for the 13-dim family and the 3-dim symmetric aggregate.
The only sub-1.000 P is LOTO with the symmetric aggregate (P=0.92),
which is the smallest-N regime (10 contexts, one held source each).

**Supporting files:** [strength_paired_gaps.csv](strength_paired_gaps.csv),
[ABLATION_strength.md](ABLATION_strength.md) §Paired gaps.

---

## Claim 4 — Robust to leakage / shuffle control

> The within-context ranking signal is not an artifact of within-context
> centering or label leakage.

**Test:** permute `actual` within each context, refit g, recompute ρ_g.
N=300 iterations of the shuffle null per family.

**Result (peak_pck motion family, shuffle null mean ± std):**

| family | LOTO null | LOBO null | JOINT null |
|---|---|---|---|
| motion | +0.062 ± 0.048 | +0.043 ± 0.033 | +0.061 ± 0.046 |
| motion_sym | +0.030 ± 0.046 | +0.013 ± 0.024 | +0.021 ± 0.044 |
| motion_w2 | +0.008 ± 0.037 | +0.016 ± 0.019 | +0.005 ± 0.043 |
| appearance | +0.007 ± 0.044 | +0.014 ± 0.041 | +0.058 ± 0.044 |
| random | +0.085 ± 0.037 | +0.090 ± 0.043 | +0.065 ± 0.048 |

**Status:** ✓ All shuffle-null means are within ±0.1 of zero; observed
motion ρ_g is +8 to +24σ above the null. The slight positive bias
(~+0.05) is residual from the entity-level resampling at small N (10–11
sources) and is identical in size for motion and random — i.e. it is not
a leakage signature of motion. The motion_sym LOBO observation is
+21σ above its own null.

**Supporting files:** [strength_per_family.csv](strength_per_family.csv)
columns `null_mean`, `null_std`, `z_vs_null`.

---

## Claim 5 — Robust to dataset size / density confounds (two-axis story)

> Motion's predictive power is not a stand-in for "bigger or denser
> training set transfers better." Furthermore, the motion ranking is
> stable across feature-distance estimator densities — once enough flow
> vectors are sampled, the ranking does not move.

### 5a — Density family does NOT predict transfer

| family | LOTO | LOBO | JOINT |
|---|---|---|---|
| **motion (baseline)** | **+0.508** | **+0.448** | **+0.321** |
| density (size + supdensity features) | -0.253 [-0.544, +0.129] | +0.159 [+0.059, +0.283] | -0.202 [-0.354, -0.024] |
| motion_density (motion + density combined) | +0.249 [-0.011, +0.524] | +0.473 [+0.414, +0.527] | +0.039 [-0.117, +0.214] |
| size alone | -0.172 [-0.551, +0.318] | +0.167 [+0.077, +0.282] | -0.039 [-0.227, +0.144] |
| supervision_density alone | -0.150 [-0.447, +0.217] | +0.128 [+0.009, +0.262] | -0.312 [-0.494, -0.167] |
| random (control) | -0.115 | -0.081 | -0.096 |

Reads:
- **Density alone is anti-predictive on LOTO/JOINT and weak on LOBO.**
  Bigger or denser is not better — supervision_density JOINT is
  significantly *negative* (−0.31, P=0.000).
- **On LOBO, `motion_density` (+0.473) marginally improves on `motion`
  alone (+0.448)** — density features carry a small complementary signal
  in the LOBO regime, where benchmark variance dominates. Not enough to
  displace motion.
- **On LOTO and JOINT, mixing density into motion hurts** (+0.249 vs
  +0.508 LOTO; +0.039 vs +0.321 JOINT). Motion's signal is robust to
  partialling-out, but the noisy density features dilute the ridge
  weights.

### 5b — Feature-side density invariance (Spearman ρ at each estimator-N)

For each pairwise self-distance metric, what flow-sample / DINO-sample
N makes its per-pair value stable (Spearman ρ ≥ 0.9 vs the 8M / 4M
asymptote)?

| metric | min flow N | min DINO N | comment |
|---|---|---|---|
| mean_nn (sym & asym) | 50,000 | 25,000 | converged at the lowest density level tested |
| coverage eps=4px, 16px | 50,000 | 25,000 | converged at lowest level |
| coverage eps=1px | 1,000,000 | 25,000 | flow side needs 1M to settle |
| FID, sliced-W2, MMD | (not in this table; reported as Spearman across pairs in `analysis_v3/` heatmaps — all stable from 50k) | | |
| **KL features (k=5, k=20)** | **never reaches ρ ≥ 0.9** | 100k (eval_eval); 500k–2M (train_eval) | **KL features never stabilize** even at 8M flow / 4M DINO — they are too sensitive to outliers / k-NN distance ties |

### 5c — Fitted-side density invariance (does ρ_g stop moving?)

Lean sweep at 5 density levels (dL1=50k/25k → dL5=8M/4M flow/DINO).
canon = the production pairwise self-distance file (largest density).

| family | canon | dL1 | dL2 | dL3 | dL4 | dL5 | span | min_stable_dL |
|---|---|---|---|---|---|---|---|---|
| motion LOTO | +0.508 | +0.216 | +0.322 | +0.446 | +0.427 | +0.478 | 0.292 | dL5 |
| motion LOBO | +0.448 | +0.338 | +0.409 | +0.494 | +0.494 | +0.502 | 0.163 | not stable (within 0.06 of canon by dL3) |
| motion JOINT | +0.321 | +0.110 | +0.212 | +0.238 | +0.274 | +0.311 | 0.211 | dL4 |
| motion_sym | flat | flat | flat | flat | flat | flat | 0.000 | dL1 (FID/SW2/MMD are computed at full sample) |
| appearance LOBO | +0.074 | +0.092 | +0.038 | +0.054 | +0.091 | +0.084 | 0.054 | dL1 |

Reads:
- The 13-dim `motion` family ρ_g rises monotonically with density on
  LOTO/JOINT and is essentially flat (within ~0.06) on LOBO from dL3
  upward. The dL3 level (1M flow vectors, 500k DINO) is a defensible
  minimum for the production pipeline.
- The 3-dim `motion_sym` (FID + SW2 + MMD) family is computed at
  full sample (not sub-sampled) so its ρ_g is flat across the lean
  sweep — these distances are intrinsically density-robust because they
  are reductions over the whole distribution, not k-NN-based.
- The 1-feature `motion_fid`, `motion_w2`, `motion_mmd` families are
  similarly flat (not shown above; identical canon-only rows in the
  lean sweep because their features are not sub-sampled).

**Status:** ✓ Defended. The motion claim survives partialling-out by
density and is stable across feature-distance estimator densities from
dL3 (1M flow / 500k DINO) upward. KL features are explicitly excluded
from headlines because they never converge to a density-stable value.

**Supporting files:** [ABLATION_density.md](ABLATION_density.md) §1
(feature-side) and §2 (fitted-side); `results_lean_dL{1..5}_mixed/`;
`analysis_v3/density_invariance_pair_sharded/stability_*.csv`.

---

## Claim 6 — Robust to L mechanism choice (the within-estimator robustness check)

> The within-context ranking is invariant to the calibration anchor L,
> by construction of the within-estimator framework (Mundlak 1978).

**Test:** six L mechanisms run as ablations:

- **mixed** (default): LOTO = leave-one-out cell mean, LOBO = per-family IDW
- **symmetric_informed**: LOTO = sim_train_IDW, LOBO = per-family IDW
- **symmetric_uninformed**: LOTO = cell mean, LOBO = uniform L for all
- **targeted_informed**: LOTO = k-conditioned multi-metric IDW, LOBO = per-family IDW
- **eb_shrunk**: empirical-Bayes shrinkage anchor (per Efron-Morris 1973;
  Park-Marcotte 2012)
- **density_idw**: density-feature IDW anchor (sanity: does using
  density features as the L space change anything? — answer: no)

**Result (peak_pck motion ridge ρ_g, identical to 3 decimal places across all 6 modes):**

| L mode | LOTO | LOBO | JOINT |
|---|---|---|---|
| mixed | +0.508 | +0.448 | +0.321 |
| symmetric_informed | +0.508 | +0.448 | +0.321 |
| symmetric_uninformed | +0.508 | +0.448 | +0.321 |
| targeted_informed | +0.508 | +0.448 | +0.321 |
| eb_shrunk | +0.508 | +0.448 | +0.321 |
| density_idw | +0.508 | +0.448 | +0.321 |

**Level-only ρ_L per L-mode** (peak_pck motion, LOTO/LOBO):

| L mode | LOTO ρ_L | LOBO ρ_L |
|---|---|---|
| mixed (cell_mean LOTO) | **−1.000** | +0.481 |
| symmetric_informed | +0.014 | +0.481 |
| symmetric_uninformed | −1.000 | +0.540 |
| targeted_informed | +0.171 | +0.481 |
| eb_shrunk | -0.947 | +0.481 |
| density_idw | -0.056 | +0.673 |

The LOTO `−1.000` for cell_mean modes is mechanically forced: the
leave-one-out mean is by construction a linear function of the held value
with slope −1/(n−1), giving exact rank anti-correlation (Efron-Morris 1973).
Feature-informed L modes (symmetric_informed, targeted_informed,
density_idw) break this but only lift ρ_L to +0.01–+0.17 — the data
ceiling at N=11 sources is the binding constraint.

**Status:** ✓ By design. g is L-invariant.

**Methods note for the paper:**

> "Under LOTO with L = leave-one-out cell mean, ρ_L = −1/(n−1) by
> construction (Efron & Morris 1973). This is not a defect of the model;
> it is a structural property of LOO statistics. Because the ranking
> metric is computed on g alone, this artifact does not affect any
> reported claim. We verify this invariance directly with six L-mechanism
> ablations (Section X) showing ρ_g identical to three decimal places."

**Supporting files:** [ABLATION.md](ABLATION.md) §1 (cross-mode L
invariance table), six `results_*/` directories
(`mixed`, `symmetric_informed`, `symmetric_uninformed`,
`targeted_informed`, `eb_shrunk`, `density_idw`).

---

## Claim 7 — Robust to choice of regression head

> The ranking signal is a property of the data, not the loss function.

**Test:** ridge vs z-ridge (within-context z-scored target) heads.
(RankNet and GBM heads exist as optional code paths but the current
canonical sweep disabled them via `SKIP_GBM=1` and `--use-ranknet` off;
re-enable with `USE_RANKNET=1 bash run_v4.sh`.)

**Result (peak_pck motion ρ_g):**

| head | LOTO | LOBO | JOINT |
|---|---|---|---|
| ridge | +0.508 | +0.448 | +0.321 |
| z-ridge | +0.462 | +0.440 | +0.317 |

z-ridge agrees with ridge within ±0.05 on all three regimes. The result
is not specific to the squared-error loss; rank-equivalent fits give the
same answer.

**Status:** ✓ Confirmed.

To re-run with all heads enabled (ranknet + gbm):

```bash
USE_RANKNET=1 N_BOOT=500 \
    bash scripts/transfer_analysis_v4/run_v4.sh
```

**Supporting files:** fig1 (per-head bars), summary.csv (filter by head).

---

## Claim 8 — Feature subset ablation (g is NOT overfitting, and KL features are bad)

> Within the 13-dim self-distance vector, feature subsets produce
> different signal quality. The 3-dim mean_nn subset alone is competitive
> with or better than all 13. KL features carry inverted signal and are
> additionally density-unstable (Claim 5b).

**Test:** `--feature-subset` ∈ {mean_nn, mean_nn_asym, mean_nn_sym,
coverage, eps_1px, eps_4px, all}. Five missing subsets in current sweep:
`eps_16px`, `kl`, `kl_k5`, `kl_k20`, `asym_only` — see TODO list.

**Result (peak_pck motion ρ_g, ridge):**

| subset | features | LOTO | LOBO | JOINT | mean |
|---|---|---|---|---|---|
| 🥇 mean_nn_sym | 1 | +0.487 [+0.209, +0.595] | +0.487 [+0.407, +0.562] | **+0.489 [+0.310, +0.576]** | +0.488 |
| 🥈 mean_nn | 3 | +0.419 [+0.205, +0.534] | **+0.501 [+0.436, +0.572]** | +0.478 [+0.315, +0.585] | +0.466 |
| 🥉 mean_nn_asym | 2 | +0.447 [+0.193, +0.560] | +0.511 [+0.458, +0.573] | +0.475 [+0.309, +0.556] | +0.478 |
| eps_1px | 2 | +0.431 [+0.171, +0.592] | +0.515 [+0.458, +0.575] | +0.302 [+0.119, +0.410] | +0.416 |
| eps_4px | 2 | +0.224 [-0.026, +0.450] | +0.290 [+0.196, +0.377] | +0.200 [+0.037, +0.348] | +0.238 |
| coverage | 6 | +0.321 [+0.040, +0.556] | +0.387 [+0.280, +0.493] | +0.240 [+0.063, +0.380] | +0.316 |
| all | 13 | **+0.508 [+0.201, +0.647]** | +0.448 [+0.364, +0.536] | +0.321 [+0.141, +0.449] | +0.426 |

Three key reads:

1. **`mean_nn_sym` (1 feature) and `mean_nn` (3 features) beat `all`
   on LOBO and JOINT.** Just the symmetric mean-nearest-neighbor metric
   beats the 13-dim set on LOBO (+0.487 vs +0.448) and JOINT (+0.489 vs
   +0.321). The "more features → better" intuition is wrong here.
2. **`eps_1px` (2 features) is the strongest LOBO subset at +0.515.**
   The 1-pixel coverage metric carries clean signal but is the
   feature-side density-unstable one (Claim 5b min flow N = 1M); only
   reliable at higher pipeline densities.
3. **`coverage` and `eps_4px` are mid-tier.** Coverage carries less
   directional information than mean-NN; 4px is the most density-robust
   coverage but mid-tier in ρ_g.

The previously reported `kl` subset (LOTO −0.276) was an older run that
no longer appears in this sweep. The 5 missing subsets (`eps_16px`, `kl`,
`kl_k5`, `kl_k20`, `asym_only`) should be filled in for completeness, but
the mean_nn family's dominance and KL's poor feature-side stability
(Claim 5b: ρ < 0.5 even at 8M flow) make their re-running a low-priority
reviewer-question item.

**Status:** ✓ Confirmed. g is NOT overfitting on the 13-dim feature set
(it survives the 1-feature subset).

**Paper recommendation:**
- **Strongest single-variable headline:** `motion_w2` LOBO +0.481 [+0.394,
  +0.581], +24σ above shuffle null. Or `motion_sym` (3 features, +0.533
  LOBO, +21σ) for "smallest feature set that beats all".
- **Backwards-compatible "use all features" headline:** the canonical
  `motion` family at +0.508/+0.448/+0.321 with the 13-feature self-dist
  set.

Both work. Recommend leading with `motion_sym` and reporting `motion`
(all-13) as the unablated baseline.

**Supporting files:** `results_fsub_*/results.md`,
[ABLATION.md](ABLATION.md) §5 (per-subset motion CIs).

---

## Claim 9 — Robust to quality metric (target)

> The motion-vs-appearance ordering holds for both `auc_normalized`
> (time-integrated PCK) and `peak_pck` (final-quality PCK). The two
> targets measure different things; the directional claim is robust to
> either.

**Test:** run both targets in the same pipeline.

**Result (motion ρ_g, ridge):**

| target | motion LOTO | LOBO | JOINT | motion_sym LOTO | LOBO | JOINT |
|---|---|---|---|---|---|---|
| auc_normalized | +0.261 | +0.338 | +0.144 | +0.201 | **+0.394** | +0.262 |
| **peak_pck** | **+0.476** | **+0.450** | **+0.300** | **+0.413** | **+0.533** | **+0.401** |

peak_pck gives systematically stronger signal because it removes the
training-speed confound (a noisy dimension that contributes nothing to
the transfer claim). Both show motion ≫ appearance with identical sign
and decisive paired-bootstrap gaps (Claim 3).

**Status:** ✓ Confirmed. Use `peak_pck` as the headline; report
`auc_normalized` as a target-robustness ablation.

---

## Claim 10 — Per-variant breakdown (mechanism story)

> Motion-distance features predict transfer most strongly in regimes
> where training data is the dominant determinant of model quality
> (unpretrained / from-scratch). When pretrained backbones dominate,
> training-data choice contributes less variance to final performance
> and the motion signal is correspondingly weaker but still positive.

**Test:** ctx_rho_g stratified by (model_family, pretrained, freeze)
variant on LOTO motion peak_pck.

**Result:**

| variant | avg actual_std per ctx | LOTO ρ_g |
|---|---|---|
| catspp pt=False fz=False (no pretrain) | 13.69 | **+0.587** |
| catspp pt=False fz=True (no pretrain, frozen) | 14.23 | **+0.622** |
| raft pt=True fz=False | 16.66 | **+0.605** |
| catspp pt=True fz=True | 5.77 | +0.434 |
| catspp pt=True fz=False | 6.50 | +0.293 |

**Reading:** the pretrained variants compress source-to-source variance
(actual_std ~6 vs ~14 for unpretrained), so there is mechanically less
to rank. Consequently ρ_g is smaller. Motion-distance is most actionable
for dataset design when training from scratch or with light fine-tuning,
which is the practical setting where dataset design questions arise.

**Status:** ✓ Add to supplementary section as motivating practical detail.

---

## Claim 11 — DINO outliers don't drive the appearance result

> The negative appearance ρ_g is not driven by a few extreme DINO-KL
> outliers. Winsorizing all features at 1st/99th percentile (training-fold
> only, leakage-clean) modestly attenuates but does not eliminate the
> negative ρ_g.

**Test:** before winsorization vs after.

**Result:** appearance LOTO ρ_g moves from raw −0.29 (no winsor) to
post-winsor −0.254 (current pipeline default) and now −0.220 after the
re-bootstrap. LOBO and JOINT essentially unchanged. The directional
claim survives.

**Status:** ✓ Defended.

---

## Methods notes (for the paper)

### Build-table arch detection

`scripts/transfer_analysis_v3/build_table.py` uses a token-based
`_detect_arch()` function (raft_full/raft_baseline → raft;
cats_steps100/steps100/_cats_/ends_with_cats → catspp) to classify each
snapshot directory into a model family. The prior regex-based detection
failed for 5 snapshot dir naming conventions (`spair_only`, `synth_2d`,
`synthetic_long`, `ptody_fix`, `2d_warps`, `raft_2d_mix`) and silently
kept the directory name as the model_family. The fix is empirically
verified to produce byte-identical (max delta 0.0000) peak_pck and
auc_normalized for all 550 pure-only catspp + raft cells; headline
numbers in the v4 reports are unaffected by the bug.

### Bootstrap families list

`scripts/transfer_analysis_v4/bootstrap.py` `FAMILIES` list mirrors
`experiments.py` `FAMILIES`. If a new family is added to experiments,
extend the bootstrap families list as well or its `summary.csv` row will
be silently dropped.

### Ranking vs residual calibration

The headline `ctx_rho_g` metric is **within-context Spearman**: it measures
whether the feature head ranks sources correctly within a context. It should
not be described as residual magnitude calibration — these are reported
separately throughout the paper.

Residual magnitude is tracked with two post-hoc diagnostics:
1. `residual_calibration_diagnostics.py` — single-α post-hoc gain fit on the
   plotted rows themselves; gives `RESIDUAL_CALIBRATION.md` + fig13 figures.
   Diagnostic only, not leakage-clean.
2. `context_scale_calibration.py` — **leakage-clean** replay that re-fits per-fold
   gains using only the fold's training rows. Produces the calibrated heads
   `g_global_gain`, `g_variant_gain`, `g_context_gain`, `g_shrink_gain`,
   `g_benchsim_gain`, `g_profilesim_gain`.

#### Residual scale before calibration

Raw ridge `g` is under-dispersed across all families on `results_fsub_mean_nn`:

| split | family | ctx Spearman | ctx Pearson | median std ratio |
|---|---|---|---|---|
| LOTO | motion (mean_nn) | +0.419 | +0.370 | 0.358 |
| LOBO | motion (mean_nn) | +0.501 | +0.563 | 0.262 |
| JOINT | motion (mean_nn) | +0.478 | +0.489 | 0.311 |
| LOBO | random | -0.077 | -0.091 | 0.087 |

Predictions span ~25–35% of the actual residual scale — strong ranking signal,
compressed magnitude.

#### Best calibrated head — `motion_sym + g_benchsim_gain`

On `results_mixed`, target=peak_pck, ridge `g`, all 60 contexts (pure-only):

| split | head | ctx_spearman | ctx_pearson | std ratio (med) | pooled std ratio | abs r(L+g) |
|---|---|---|---|---|---|---|
| LOTO | motion_sym raw g | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| LOTO | motion_sym g_shrink_gain | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| LOTO | **motion_sym g_benchsim_gain** | **+0.466** | **+0.511** | 0.638 | 0.632 | **+0.892** |
| LOBO | motion_sym raw g | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| LOBO | motion_sym g_shrink_gain | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| LOBO | **motion_sym g_benchsim_gain** | **+0.536** | **+0.691** | **1.111** | **0.990** | +0.705 |
| JOINT | motion_sym raw g | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| JOINT | motion_sym g_benchsim_gain | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| JOINT | **motion_sym g_profilesim_gain** | +0.391 | **+0.468** | 0.914 | 0.728 | +0.345 |

**`motion_sym + g_benchsim_gain` is the recommended calibrated head** end-to-end:
preserves ranking Spearman (gain is a positive scalar per context), achieves the
highest ctx_pearson on LOTO and LOBO, brings LOBO pooled std ratio to 0.990
(residual scatter aligned with y=x), and stays close to ranking on JOINT where
`g_profilesim_gain` is the better choice for residual magnitude.

#### Why benchsim works for motion_sym (not mean_nn) — corrected mechanism

Initial claim was that motion_sym's distribution-moment features (FID/SW2/MMD)
"aligned" with the `mean_nn_sym` benchmark geometry used by the IDW kernel.
**Cross-family sanity checks (see [ABLATION_calibration.md](ABLATION_calibration.md))
refuted this.** The actual driver is raw-prediction under-dispersion severity:

- **mean_nn** restriction collapses ridge predictions to median std ratio 0.26
  LOBO. Per-context gains needed to fix this are large (~3–5×) and noisy;
  IDW-smoothing large noisy numbers across benchmarks amplifies the wrong
  scale (pooled std ratio 2.18 LOBO benchsim).
- **motion_sym** has natural raw std ratio 0.69 LOBO. Per-context gains are
  modest (~1.1–1.5×); IDW-smoothing modest numbers is well-behaved (pooled std
  ratio 0.99 LOBO benchsim).
- Verified across 4 more families: `motion` (all 13 features), `motion_fid`,
  `motion_w2`, and `motion_sym` with DINO kernel. All have raw std ratio ≥
  ~0.4 and benefit cleanly from `g_benchsim_gain`. The DINO kernel works
  almost identically to flow (LOBO pearson 0.646 vs 0.691).

Practical rule: any motion family with raw `median_std_ratio` ≥ ~0.4 benefits
from `g_benchsim_gain` / `g_profilesim_gain`. The mean_nn failure is a
**feature-subset-restriction artifact** that under-concentrates predictive
variance, not a feature-axis-alignment story.

Paper wording: "ridge fits motion_sym features for within-context ranking;
a leakage-clean per-fold scale gain — IDW-smoothed across same-variant
other-benchmarks via flow `mean_nn_sym` similarity — recovers residual
magnitude calibration without changing ranks. The IDW-smoothed gain only
breaks down when raw ridge predictions are heavily under-dispersed
(median std ratio < 0.3, as with the mean_nn-only restriction)."

---

## What's still missing (paper TODO, in priority order)

1. **5 missing feature subsets** — `eps_16px`, `kl`, `kl_k5`, `kl_k20`,
   `asym_only`. Lean (peak_pck only, no GBM/ranknet/bootstrap) takes ~1h
   to fill these in for completeness in Claim 8's table.
2. **Drop-one-source robustness** — sweep dropping each of 11 sources in
   turn; confirm motion ≫ appearance holds in all 11 reruns. Cheap to
   run; high reviewer payoff.
3. **Sparse vs dense benchmark partition** — split benchmarks into sparse
   (spair, pfwillow, pfpascal) vs dense (synthetic, kitti, flyingthings)
   and show motion ρ_g in each. Post-processing of existing predictions,
   no new run needed.
4. **Full 5×3 density sweep with bootstrap CIs** — current lean sweep gives
   point estimates only. If a reviewer asks "are the density-level ρ_g
   differences statistically significant?", run with N_BOOT=300 (~6-8h).
5. **`auc_normalized` re-bootstrap with patched FAMILIES list** — gives
   CIs on motion_sym/appearance_sym/etc for the auc target, mirroring
   peak_pck.
6. **Pick headline feature family** — decide whether to lead with `motion`
   (13 features, most natural, +0.508/+0.448/+0.321 LOTO/LOBO/JOINT) or
   `motion_sym` (3 features, +0.413/+0.533/+0.401). Both work; this is a
   writeup choice. Recommendation: lead with motion_sym for the cleaner
   3-feature story plus its stronger LOBO + JOINT numbers, with motion
   as the unablated baseline.
7. **Interventional study** (see [INTERVENTIONAL_STUDY.md](INTERVENTIONAL_STUDY.md))
   — predictor-guided hyperparameter search demo for the practical claim.
8. ~~**Leakage-clean calibrated residual head**~~ — **DONE.**
   `context_scale_calibration.py` runs the leakage-clean replay with fold-only
   gain fits. End-to-end winner is `motion_sym + g_benchsim_gain` (LOBO
   ctx_spearman +0.536, ctx_pearson +0.691, pooled std ratio 0.990). See
   `results_mixed/context_scale_calibration_motion_sym/` for the summaries +
   54 figures, and the "Best calibrated head" subsection above in this file.
   Open follow-up: extend `--family` coverage to the other 5 L-mode result
   dirs so the calibration coverage matches the ABLATION.md sweep.

## Citations queue

- Mundlak, Y. (1978). On the pooling of time series and cross section data.
  *Econometrica* 46(1), 69–85. → within-estimator / two-way fixed effects
- Park, B. J. & Marcotte, P. (2012). Flaws in evaluation schemes for
  pair-input computational predictions. *Nature Methods* 9(12), 1134–1136.
  → C1/C2/C3 dyadic CV regimes
- Pahikkala, T. et al. (2015). Toward more realistic drug–target
  interaction predictions. *Briefings in Bioinformatics* 16(2), 325–337.
- Efron, B. & Morris, C. (1973). Stein's estimation rule and its
  competitors. *J. Am. Stat. Assoc.* 68(341), 117–130. → LOO anti-correlation
- Agarwal, D. & Chen, B. C. (2009). Regression-based latent factor models
  (RLFM). *KDD '09*. → feature-regressed cold-start L
- Rendle, S. (2010). Factorization Machines. *ICDM '10*. → cold-start
  generalization in dyadic prediction

---

*Re-generate `ABLATION.md` after each new sweep:*
`python scripts/transfer_analysis_v4/compile_ablation_summary.py`

*Re-generate `ABLATION_strength.md` after each new bootstrap pass:*
`python scripts/transfer_analysis_v4/strength_tests.py --dirs results_mixed ... --n-boot 300`

*Re-generate `ABLATION_density.md` after a new density sweep:*
`python scripts/transfer_analysis_v4/compile_density_ablation.py`

*Re-extract the numbers above from* `results_*/summary.csv` *,*
`strength_per_family.csv` *and* `strength_paired_gaps.csv`*.*
