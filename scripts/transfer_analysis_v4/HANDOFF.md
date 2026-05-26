# Transfer Analysis v4 — Agent / Self Handoff

Running log of what's been tried in v4 and the current state. Read fully
before continuing work.

---

## Current state as of 2026-05-24

**Pipeline runs end-to-end** via `bash scripts/transfer_analysis_v4/run_v4.sh`.

What's working / settled:
- 13 self-distance features per (source, benchmark) per space (flow + DINO),
  pulled from `analysis_v3/pairwise_self_distances.csv` via
  `add_selfdist_features()` in `transfer_predictor_prototype.py`.
- Density / size / supervision_density features pulled from `transfer_table.csv`
  (see `DENSITY_COLS`, `SIZE_COLS`, `SUPERVISION_DENSITY_COLS` in `experiments.py`).
- Spair_long catspp variants averaged into `transfer_table.csv` peak_pck via
  `patch_spair_long.py` (raft variant still broken; no val results).
- Two targets supported in one run: `auc_normalized` and `peak_pck`. **peak_pck
  is the headline target** — removes the training-speed conflation that AUC
  carries.
- Four heads available per fold: `ridge` (always), `z-ridge`, `RankNet` (opt-in),
  `GBM` (opt-in). All written into one per-row prediction CSV with multiple
  score columns.
- Four L modes via `--l-mode`: `mixed` (default), `symmetric_informed`,
  `symmetric_uninformed`, `targeted_informed`.
- Per-family L prior (motion/both/motion_density use flow IDW, appearance uses
  DINO IDW, random/density use uniform L). Random no longer piggybacks on
  motion's flow-IDW.
- Feature winsorization at 1st/99th percentile (training-fold only,
  leakage-clean) prevents DINO-KL outliers from dominating appearance ridge.
- Bootstrap CIs with entity-resampling (N_BOOT default 500–1000).
- 12 figures auto-generated per target; ABLATION.md cross-mode summary.
- CLAIMS.md as the paper-prep claim → evidence map.

What's pending — see "Pending work" at bottom.

---

## What v4 is and how it differs from v3

v4 is a **focused, claim-driven analysis layer on top of v3's features**.
v3 still produces `transfer_table.csv` and `pairwise_self_distances.csv`;
v4 does the modeling, bootstrap, figures, and report compilation.

| | v3 | v4 |
|---|---|---|
| Goal | "Which feature group predicts transfer best?" | "Does motion-distance predict transfer better than appearance-distance?" |
| Model count | 16+ model variants | One model (`L + g`) under three CV regimes |
| Feature groups | ~26 (density, sample_count, profile, flow_*, motion_*, etc.) | 6+ families (motion, appearance, both, random, density variants) — by feature SOURCE not technique |
| CV regimes | 8 (loto, lobo, loco_cell, joint_cell, loto_grouped, loco, lomo, ...) | 3 (LOTO, LOBO, JOINT) — aligned to Park-Marcotte C2 / C3 |
| Output | `flow_only_pure_diagnostic_*/results.md` | `results_<mode>/results.md`, ABLATION.md, CLAIMS.md |

**v4 is what the paper will be written from.** v3 stays as the feature
generator and as a historical record of model exploration.

---

## What we've tried in v4 (chronological log of ideas + their fate)

This is the "what was tried, why, what happened" log. Append to it after each
new session.

### Session: 2026-05-23 — Initial v4 setup + headline tightening

1. **Baseline `L + g` decomposition** wired up. L = observed-or-back-off cell
   band per regime. g = within-context ridge on demeaned features. Replicated
   v3's `transfer_predictor_prototype.py` headline numbers cleanly.

2. **Switched target from `auc_normalized` to `peak_pck`.**
   - Reason: auc integrates PCK over training, conflating speed-of-convergence
     with final quality. peak_pck is final-quality only.
   - Result: motion ρ_g jumped from +0.26 LOTO → +0.49 LOTO; LOBO +0.37 → +0.48;
     JOINT +0.21 → +0.36. The training-speed dimension was adding noise.
   - **Kept both targets in the pipeline** so the headline can be compared
     across them as a robustness check.

3. **RankNet head added** (pairwise logistic on within-context feature diffs).
   - Reason: in case ridge's loss function was specific to the result.
   - Result: ρ_g within ±0.05 of ridge on every regime. Useful as robustness check.
   - **Later demoted to opt-in (`--use-ranknet`)** because it adds nothing the
     headline doesn't already say and the rank scatter is unreadable at N=10.

4. **GBM head added** (HistGradientBoostingRegressor) as nonlinear ceiling check.
   - Reason: rule out "linear ridge is leaving signal on the table."
   - Result: GBM motion ρ_g within ±0.03 of ridge on every regime.
     Confirmed ridge is at-ceiling at this N.
   - **Kept as opt-in (`--no-gbm` available)** — adds 2-3 min to each run.

5. **Per-benchmark gain calibration** (g_cal: rescale ridge predictions by
   per-context std ratio).
   - Reason: visible heterogeneous shrinkage in residual scatter (spair tight,
     synthetic spread). Hoped to fix the visual.
   - Result: rank-invariant by construction (gain is positive scaling), so ρ_g
     unchanged. abs_r slightly hurt because per-fold gain estimate is noisy.
   - **Removed**, replaced with z-ridge (next session).

6. **Spair_long snapshot integration.**
   - Diagnosis: old spair training had a bug (catspp pt=True works fine; raft
     pt=True fz=False had peak_pck ≈ 3.4 on flyingthings, should be ~70).
   - Action: `patch_spair_long.py` averages peak_pck from
     `/mnt/nvme_1tb_b/spair_long/*` catspp variants into existing spair rows.
   - Raft variant has no val_results.csv (training cut short) — can't patch.
   - Original transfer_table backed up as `transfer_table.pre_spair_long.csv`.

7. **Bootstrap added** (entity-resample 500–1000 iters; resamples held-out
   axis: sources for LOTO, benchmarks for LOBO, (source, benchmark) pairs for
   JOINT). Paired bootstrap for motion − appearance gap.
   - Headline result: P(motion − appearance gap > 0) ≥ 0.997 on every regime.

8. **Figures added/refined**:
   - fig1: headline bars (ctx_rho_g per family × split × head)
   - fig2: pooled L+g vs actual scatter (relabeled "pooled r" later)
   - fig3a: within-context residual scatter (g)
   - fig3b: within-context residual scatter (L+g)
   - fig4: shuffle + random controls bars
   - fig5: density confound bars

### Session: 2026-05-24 — L-mode ablations + variance handling + new figures

9. **Per-family L prior** (`FAMILY_SPACE` mapping).
   - Reason: original code used flow-IDW for L on every family. So when
     appearance got `level_only ρ_L ≈ 0.72` on LOBO, that was random borrowing
     from flow geometry, not appearance's own signal. Reviewer attack vector.
   - Fix: motion → flow-IDW, appearance → DINO-IDW, random/density → uniform L
     (no benchmark-similarity prior at all). `--flat-flow-prior` available for
     legacy comparison.
   - Result: random LOBO abs_r dropped 0.79 → 0.34 — random no longer cheats.

10. **Winsorization at 1st/99th percentile.**
    - Diagnosis: DINO KL features have heavy tails (movi_f → others ≈ 271-345
      vs typical 100; synthetic→synthetic ≈ −112 from KL estimator noise).
      Caused appearance ridge std = 257 on JOINT (vs motion's 12), destroying
      pooled abs_r.
    - Fix: clip features at training-fold 1st/99th percentile in `_fit_ridge`.
      Reused by RankNet, z-ridge, GBM, and targeted_idw.
    - Result: appearance LOTO abs_r jumped 0.013 → 0.793.
      Motion ρ_g barely moved (0.487 → 0.468).

11. **Within-context z-score ridge (z-ridge)** as replacement for gain
    calibration.
    - Mechanism: target divided by per-context std before fitting; one global
      slope on z-scaled data; predictions un-standardized per context.
    - Result: ρ_g within ±0.02 of ridge in most cells; slightly stronger on
      JOINT (+0.28 → +0.37 for motion). Calibration sometimes better, sometimes
      worse. Kept as opt-in head.

12. **fig8 per-context ρ histogram** — every context contributes one ρ value;
    histogram colored by benchmark shows which contexts win vs lose.
    Replaces the unreadable rank scatter (fig6) as the "performance breakdown"
    view. Honest and informative.

13. **fig9 Top-K hit rate** — for each (benchmark, variant), what fraction of
    model's top-K predicted training datasets are in the actual top-K?
    Practical view: "if you used this predictor to pick training data, how
    often would you pick winners?" Top-5 ~0.63 on LOBO motion (random = 0.45).

14. **fig10 hexbin density** — density-view of fig3a. Diagonal density along
    the trend is the visual ctx_ρ.

15. **fig6 rank scatter DROPPED.** At N≈10 sources per context, the grid is
    fully populated → unreadable. fig8 + fig10 cover the same info more clearly.

16. **L-mode ablations** (`--l-mode`):
    - `mixed` (default): LOTO=cell_mean, LOBO=per-family IDW
    - `symmetric_informed`: LOTO=sim_train_IDW, LOBO=sim_eval_IDW (your
      symmetric variant)
    - `symmetric_uninformed`: LOTO=cell_mean, LOBO=uniform L for all
    - `targeted_informed`: LOTO=kNN-IDW in standardized (i→k) feature space
      (multi-metric, directional, k-conditioned). LOBO=per-family IDW.
    - **g_only ρ is IDENTICAL across all 4 modes by design** (g doesn't see L).
      That's the robustness check.

17. **`compile_ablation_summary.py`** scans `results*/` dirs and produces
    ABLATION.md with side-by-side per-mode tables.

18. **`--feature-subset` + `--targeted-subset` flags** for feature-set ablations:
    - `--feature-subset` restricts f_cols (affects g, z-ridge, ranknet, GBM,
      targeted_idw vector)
    - `--targeted-subset` further restricts the targeted_idw distance norm
    - Subsets: `all` / `mean_nn` / `coverage` / `kl` / `asym_only`
    - Smoke test: motion LOTO ρ_g with just 3 mean_nn features = +0.46
      (vs +0.47 with all 13). Ridge is NOT overfitting — features are mostly
      redundant. Mean_nn alone carries ~97% of the signal.

19. **Density family split into size + supervision_density + motion_size + motion_supdensity.**
    - `size`: log_train/eval_n_samples, log_train/eval_n_vectors (totals only)
    - `supervision_density`: log_*_valid_vectors_per_sample / mean / p90 (per-sample only)
    - `motion_size`: motion features + size
    - `motion_supdensity`: motion features + supervision_density
    - Smoke test finding:
      - size alone ρ_g = -0.17 LOTO / +0.19 LOBO — weak
      - supervision_density alone ρ_g = -0.15 LOTO / +0.13 LOBO — weak
      - motion_supdensity LOBO ridge = +0.49 (vs motion +0.48), z-ridge = +0.55
        (**best LOBO number seen so far**). Marginal lift but worth a sentence.
      - On LOTO, density features ADD NOISE to motion (drop ~0.06).

20. **Per-variant ρ_g breakdown** (paper-friendly mechanism story).
    - On LOTO motion peak_pck:
      - catspp pt=False (no pretrain): ρ_g ≈ +0.50, actual std ≈ 14
      - catspp pt=True: ρ_g ≈ +0.34–0.45, actual std ≈ 6
    - Pretrained variants COMPRESS source-to-source variance (pretraining
      dominates) → less to rank → lower ρ_g. Unpretrained variants are where
      motion-distance is most ACTIONABLE for dataset design.
    - Counter-intuitive but defensible finding. Add to supp.

21. **LOO anti-correlation framing settled.**
    - ρ_L = −1/(n−1) for LOTO with cell_mean L is structural (Efron-Morris 1973).
    - Symmetric_informed and targeted_informed lift it to +0.01 to +0.22 with
      feature-informed source-similarity, but the lift is modest (data ceiling,
      not method ceiling, is the constraint).
    - Paper framing: rank on g alone (within-estimator, Mundlak 1978); L is
      calibration only; cite the structural result and move on.
    - 30-line implementation of Empirical-Bayes shrinkage as 5th L mode is
      possible but unnecessary unless reviewers push.

22. **`CLAIMS.md`** drafted as paper-prep document. Per-claim hypothesis →
    test → result → status → file references. 11 claims, citations queue.

---

## Key findings so far

### Robust scientific claims
1. **Motion ≫ appearance** for transfer prediction on all 3 dyadic CV regimes
   (P(gap > 0) ≥ 0.997 paired bootstrap)
2. **Result is L-invariant** by design (within-estimator framework)
3. **Result is leakage-clean** (shuffle ρ_g ≈ 0)
4. **Result is not a density / size proxy** (motion survives partialling-out)
5. **Result is not loss-function specific** (ridge / z-ridge / RankNet / GBM
   all agree)
6. **Result holds for both `auc_normalized` and `peak_pck`** with peak_pck
   stronger

### Subordinate findings worth a sentence each
- **Flow vs DINO have different distance roles**: flow strong for
  cross-distance (i→k via g), DINO strong for source-identity (i↔j via L).
- **Per-sample supervision density** complements motion on LOBO (~+0.07
  with z-ridge); doesn't help on LOTO.
- **Pretrained variants are harder to rank** (lower source-to-source variance);
  motion-distance is most actionable for from-scratch training.
- **Mean_nn family carries ~97% of motion's signal** (3 of 13 features
  suffice); ridge correctly regularizes the redundant features.
- **DINO outliers (movi_f, synthetic→synthetic KL)** are real signal, not
  corrupted data; winsorization at 1%/99% prevents them from dominating.

### Things we tried and explicitly RULED OUT
- **Folding L into ranking on LOTO** — wrecks ρ (LOO anti-correlation).
  Confirmed; never do.
- **Sim_train_IDW with `mean_nn_sym` alone** for source clustering — flow
  source-similarity in this metric is essentially uninformative (ρ_L ≈ +0.01).
- **Per-benchmark gain calibration** — rank-invariant by construction so
  doesn't change ρ; high-variance gain estimates hurt abs_r. Removed.
- **In-sample fit** — trivially gives ρ_L = +1.0 by memorization; defeats
  cold-start evaluation. Not pursued.

---

## How to run things

### Standard
```bash
bash scripts/transfer_analysis_v4/run_v4.sh
```

### Useful env vars (overridable)
| Var | Default | Effect |
|---|---|---|
| `TARGETS` | `auc_normalized peak_pck` | Targets to run |
| `N_BOOT` | `1000` | Bootstrap iterations |
| `L_MODE` | `mixed` | L mechanism: `mixed`, `symmetric_informed`, `symmetric_uninformed`, `targeted_informed` |
| `FEATURE_SUBSET` | unset | Restrict f_cols: `all`, `mean_nn`, `coverage`, `kl`, `asym_only` |
| `TARGETED_SUBSET` | unset | Restrict targeted_idw norm: same options as `FEATURE_SUBSET` |
| `OUT_DIR` | `scripts/transfer_analysis_v4/results` | Output directory |
| `LOG_DIR` | `$OUT_DIR/logs` | Log directory |
| `SKIP_BOOTSTRAP` | `0` | Skip bootstrap (use point estimates only) |
| `SKIP_FIGURES` | `0` | Skip figure generation |
| `SKIP_GBM` | `0` | Skip GBM head |
| `USE_RANKNET` | `0` | Enable RankNet head |
| `FLAT_FLOW_PRIOR` | `0` | Legacy: use flow-IDW for all families |
| `FAMILY_MATCHED` | `0` | Legacy: appearance LOBO uses DINO-IDW (now default; flag is no-op) |

### Parallel ablation sweep (the canonical comparison run)
```bash
# 4 L modes
for mode in mixed symmetric_informed symmetric_uninformed targeted_informed; do
    OUT_DIR=scripts/transfer_analysis_v4/results_${mode} \
    L_MODE=${mode} N_BOOT=500 SKIP_GBM=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh \
        > /tmp/v4_${mode}.log 2>&1 &
done

# 4 within-targeted-IDW subsets
for sub in mean_nn coverage kl asym_only; do
    OUT_DIR=scripts/transfer_analysis_v4/results_targeted_${sub} \
    L_MODE=targeted_informed TARGETED_SUBSET=${sub} N_BOOT=500 SKIP_GBM=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh \
        > /tmp/v4_targeted_${sub}.log 2>&1 &
done

# 4 global feature subsets (g overfitting check)
for sub in mean_nn coverage kl asym_only; do
    OUT_DIR=scripts/transfer_analysis_v4/results_fsub_${sub} \
    FEATURE_SUBSET=${sub} N_BOOT=500 SKIP_GBM=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh \
        > /tmp/v4_fsub_${sub}.log 2>&1 &
done

wait
python scripts/transfer_analysis_v4/compile_ablation_summary.py
```

12 parallel runs; ~30 min wall time. Produces `ABLATION.md` at the end.

---

## File locations

| What | Path |
|---|---|
| Main entry point | `scripts/transfer_analysis_v4/run_v4.sh` |
| Experiments runner | `scripts/transfer_analysis_v4/experiments.py` |
| Bootstrap | `scripts/transfer_analysis_v4/bootstrap.py` |
| Figures | `scripts/transfer_analysis_v4/figures.py` |
| Per-run report compile | `scripts/transfer_analysis_v4/compile_v4.py` |
| Cross-mode ablation compile | `scripts/transfer_analysis_v4/compile_ablation_summary.py` |
| Spair_long patcher | `scripts/transfer_analysis_v4/patch_spair_long.py` |
| Paper-prep doc | `scripts/transfer_analysis_v4/CLAIMS.md` |
| Cross-mode ablation table | `scripts/transfer_analysis_v4/ABLATION.md` |
| Quick-start doc | `scripts/transfer_analysis_v4/README.md` |
| **This file** | `scripts/transfer_analysis_v4/HANDOFF.md` |
| Per-run results | `scripts/transfer_analysis_v4/results_<mode>/results.md` |
| Per-row predictions | `scripts/transfer_analysis_v4/results_<mode>/predictions/<target>/rows_*.csv` |
| Summary (long-form) | `scripts/transfer_analysis_v4/results_<mode>/summary.csv` |
| Bootstrap gap CIs | `scripts/transfer_analysis_v4/results_<mode>/bootstrap_gap.csv` |
| v3 transfer table | `scripts/transfer_analysis_v3/transfer_table.csv` |
| v3 backup (pre-spair_long) | `scripts/transfer_analysis_v3/transfer_table.pre_spair_long.csv` |
| v3 pairwise self-distances | `analysis_v3/pairwise_self_distances.csv` |
| Spair_long snapshots | `/mnt/nvme_1tb_b/spair_long/spair_cats_steps100_*_2026_05_21_*/` |
| Flow coverage vectors (still on disk) | `/mnt/nvme_1tb_b/coverage_vectors/` |

---

## Pending work

In priority order:

1. **Tonight's overnight sweep** (12 parallel runs from above) — populates
   `results_*/` dirs and the ABLATION.md cross-comparison.

2. **Fill in CLAIMS.md** with the new numbers once the sweep finishes.
   Particularly:
   - Claim 5 (density confound): finish the motion_size / motion_supdensity row
   - Claim 8 (feature-subset / g overfitting): fill in coverage / kl /
     asym_only rows
   - Claim 6 (L-mode ablation): confirm ρ_g identical across all 4 modes

3. **Drop-one-source robustness sweep** (paper-defensive). Run 11 times each
   dropping one source; confirm motion ≫ appearance every time.
   Implementation: not yet written. Would be a script that loops over sources
   and calls `experiments.py --drop-source X`. Add `--drop-source` flag.

4. **Sparse vs dense benchmark partition** (paper-defensive). Re-aggregate
   existing per-row predictions stratified by benchmark category. Should be a
   small post-processing script — no rerun needed.

5. **Empirical Bayes shrinkage as 5th L mechanism** (`--l-mode eb_shrunk`).
   ~30 lines. Adds Park & Marcotte / Efron-Morris citation backup. Worth a
   supp table.

6. **Mean motion magnitude per dataset** as a cheap heuristic baseline column.
   Requires reading flow .npy files from `/mnt/nvme_1tb_b/coverage_vectors/`.
   Optional; only if reviewers ask "what about a really simple baseline."

7. **Spair_long raft re-training.** The raft variant of spair training was
   never completed (cluster went down before final val). If/when it finishes,
   re-run `patch_spair_long.py` to fold it into the table. Currently the 10
   raft+spair rows in the transfer table are the OLD broken values (peak_pck
   ≈ 3 on flyingthings).

---

## Things to NOT do

- **Don't fold L into the ranking score on LOTO.** It's anti-correlated by
  LOO construction; ρ collapses.
- **Don't drop the random / shuffle controls.** They're the proof the result
  isn't an artifact of the modeling.
- **Don't blindly add features.** At N=540 (~10 sources × ~10 benchmarks × ~5
  variants) the feature-to-sample ratio is already borderline; ridge handles
  it with regularization but more features ≠ more signal.
- **Don't change CV regimes.** LOTO/LOBO/JOINT map cleanly to Park-Marcotte
  C2/C3. Adding more CV regimes (loco_cell, loco) was the v3 trap.
- **Don't use the rank scatter (`fig6`) anywhere.** Unreadable at N=10/context.
  Use fig8 (per-context ρ histogram) instead.

---

## Sessions / changelog

Append the highlights from each session here:

### 2026-05-23
- Initial v4 setup, headline tightening, peak_pck switch, GBM/RankNet
  added, spair_long patched, bootstrap+figures wired up.
- Headline: motion ≫ appearance with P > 0.997 paired bootstrap.

### 2026-05-24
- Per-family L prior (DINO-IDW for appearance, uniform for random).
- Feature winsorization fixing DINO outlier inflation.
- z-ridge replacing gain calibration.
- fig8 (per-context ρ histogram), fig9 (top-K hit rate),
  fig10 (residual hexbin) added; fig6 (rank scatter) dropped.
- Four L modes (mixed / symmetric_informed / symmetric_uninformed /
  targeted_informed). targeted_informed = kNN-IDW in standardized (i→k)
  feature space (the "k-conditioned multi-metric" version of source similarity).
- `--feature-subset` and `--targeted-subset` flags for ablations.
- Density family split into size + supervision_density + combos.
  Found that motion_supdensity slightly boosts LOBO ρ_g.
- Per-variant breakdown: unpretrained variants give the strongest ranking
  signal (more variance from training data).
- LOO anti-correlation framing settled (Mundlak + Efron-Morris citations).
- `CLAIMS.md` and `compile_ablation_summary.py` added.
