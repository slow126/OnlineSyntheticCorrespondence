# Transfer Analysis v4 — Status (handoff)

Living status doc. Read this first; falls back to [HANDOFF.md](HANDOFF.md) for
historical session-by-session log and design rationale.

**Last updated: 2026-05-28.** Open this file at the start of every new agent /
session. Append a session entry at the bottom when you stop.

---

## TL;DR for whoever picks this up

The headline scientific claims are settled with strong statistical evidence.
The paper section that uses this analysis can be written from the artifacts
already on disk. The remaining items are reviewer-question insurance, not
load-bearing.

**Most current results (from canonical pure-only sweep + re-bootstrap + strength tests):**

- `motion` ridge ρ_g LOBO = **+0.450 [+0.368, +0.534]**, P(ρ_g>0)=1.000, **+12σ vs shuffle-null**
- `motion_sym` (FID+SW2+MMD) ρ_g LOBO = **+0.533 [+0.475, +0.592]**, P=1.000, **+21σ vs shuffle-null** ← strongest single family
- `motion_w2` (sliced-W2 only) ρ_g LOBO = +0.481 [+0.394, +0.581], +24σ vs null
- `appearance` ρ_g LOBO = +0.074 [-0.019, +0.174], P=0.950 — weak
- `appearance_sym` ρ_g LOBO = +0.141 [-0.005, +0.299], P=0.963 — **not competitive with motion_sym**
- Paired gap `motion_sym − appearance_sym` LOBO = **+0.390 [+0.243, +0.544], P(gap>0)=1.000**

(target = `peak_pck`, L-mode = `mixed`, head = ridge, pure-only filter on 11 training
sources, N_BOOT=300 entity-resampled.)

**Calibrated head for the interventional study (new 2026-05-28 PM):** `motion_sym` ridge + `g_benchsim_gain` (leakage-clean fold-trained gain, IDW-smoothed over same-variant other-benchmarks via flow `mean_nn_sym`) is the recommended end-to-end head. LOBO: ctx_spearman +0.536 (ranking preserved) + ctx_pearson **+0.691** (calibrated magnitude) + median std ratio **1.11** + pooled std ratio **0.990** (scatter aligned with y=x). JOINT slightly prefers `g_profilesim_gain` (eval-side density/profile kernel) over benchsim.

**Mechanism (corrected after 2026-05-28 PM sanity checks — earlier "feature axis alignment" claim was wrong):** benchsim fails for `mean_nn` because raw ridge predictions there are severely under-dispersed (median std ratio 0.26 LOBO), so per-context gains are large (~3–5×) and noisy. IDW-smoothing large noisy gains across benchmarks amplifies the wrong scale (pooled std ratio 2.18). For families with raw std ratio ≥ ~0.4 — motion (all 13), motion_sym, motion_fid, motion_w2 — the per-context gains are modest (~1.1–1.5×) and benchsim cleanly recovers calibration. Flow vs DINO `mean_nn_sym` kernels give similar results for motion_sym (flow slightly better on LOTO/LOBO, identical on JOINT). The cross-family calibration sweep lives in [ABLATION_calibration.md](ABLATION_calibration.md). Best per-family / split breakdown + raw outputs: [results_mixed/context_scale_calibration_motion_sym/](results_mixed/context_scale_calibration_motion_sym/) and sibling dirs.

---

## What's done (in order of importance)

### Headlines / claims with statistical backing
| Claim | Evidence | Status |
|---|---|---|
| Motion ≫ appearance across LOTO/LOBO/JOINT | [ABLATION_strength.md](ABLATION_strength.md) § "Per-family strength", paired gaps | ✅ |
| `motion_sym` (FID+SW2+MMD) is the strongest single feature family on LOBO | [ABLATION_strength.md](ABLATION_strength.md), [strength_per_family.csv](strength_per_family.csv) | ✅ |
| Appearance counterparts (`appearance_sym/fid/w2`) are NOT competitive with their motion analogues | paired gap motion_sym−appearance_sym = +0.39 P=1.000 LOBO | ✅ |
| g is L-invariant by construction | [ABLATION.md](ABLATION.md) §1 — all 6 L-modes give identical per-family ρ_g | ✅ |
| Predictor is leakage-clean | shuffle-null mean ≈ 0 ± 0.04 across all families in strength table | ✅ |
| Predictor outperforms random control | `motion − random` gap LOBO = +0.529, P=1.000 | ✅ |

### Density-invariance (two-axis stability story)
| Axis | Where | Finding |
|---|---|---|
| Feature-side (Spearman ρ vs baseline at the largest N) | [ABLATION_density.md](ABLATION_density.md) §1, sources in `analysis_v3/density_invariance_pair_sharded/stability_*.csv` | mean_nn + coverage(eps≥4px) stable from 50k vectors; eps_1px coverage needs 200k–1M; **KL features never reach ρ≥0.9** at any N (up to 8M flow, 4M DINO) |
| Fitted-side (does ρ_g per family stop moving across density levels?) | [ABLATION_density.md](ABLATION_density.md) §2, sources `results_lean_canon_mixed/`, `results_lean_dL{1..5}_mixed/` | Most families stable by dL2-dL3 (200k flow / 100k DINO and up); densities below that show noisier ρ_g |

### Pipeline + reporting infrastructure built this session
| Component | What it does | Path |
|---|---|---|
| `--dist` env override in `run_v4.sh` | Lets you swap `pairwise_self_distances.csv` per density level without rebuilding the table | [run_v4.sh](run_v4.sh) |
| `run_density_sweep.sh` | Full-fat density sweep driver (5 levels × 17 modes, with bootstrap) — written but NOT run; lean version was preferred | [run_density_sweep.sh](run_density_sweep.sh) |
| `run_density_sweep_lean.sh` | Sequential 6-run sweep: canon + 5 dL levels, mixed L-mode only, peak_pck only, no GBM/ranknet/bootstrap. ~55min wall. | [run_density_sweep_lean.sh](run_density_sweep_lean.sh) |
| `strength_tests.py` | Per-family ρ_g CIs + P(>0) + shuffle-null comparison + 3 paired-gap tests (motion−appearance, motion_sym−appearance_sym, motion−random) | [strength_tests.py](strength_tests.py) |
| `compile_density_ablation.py` | Combines feature-side stability CSVs + fitted-side ρ_g(N) per family into one report | [compile_density_ablation.py](compile_density_ablation.py) |
| `bootstrap.py` FAMILIES list expanded | Now includes `motion_sym/fid/w2/mmd`, `appearance_sym/fid/w2/mmd/nullk`, size, supervision_density, etc. — was a 6-family list before | [bootstrap.py:210](bootstrap.py#L210) |
| `compile_ablation_summary.py` §2 fix | Renders sym/fid/w2 families with CIs (`with_ci=True`) | [compile_ablation_summary.py:212](compile_ablation_summary.py#L212) |

### Critical bug fix (don't lose this context)
**`spair_only` was being treated as a real model_family.** Affected only mixed-mode (`PURE_ONLY=0`) runs — all reported headline numbers used `PURE_ONLY=1` and were empirically verified unchanged after the fix.

- **Root cause:** [build_table.py:44](../transfer_analysis_v3/build_table.py#L44) had a regex that only matched `_(raft_full|raft_baseline|cats)_`. Older CATS++ runs use the `steps100` arch token without a `_cats_` prefix and fell through, so their snapshot-dir name (`spair_only`) was kept as `model_family`. Same issue for `synth_2d`, `synthetic_long`, `ptody_fix`, `2d_warps`, `raft_2d_mix` — these are all snapshot-directory names, not real model families.
- **Fix applied:** replaced regex with `_detect_arch()` function that searches for `{raft_full, raft_baseline}` → raft, `{cats_steps100, steps100, _cats_, ends_with _cats}` → catspp. All 2130 raw rows now classify; nothing falls through.
- **Impact verified:** rebuilt `transfer_table.csv` (1340 → 1180 rows). For all 550 pure-only catspp+raft cells, `peak_pck` and `auc_normalized` are byte-identical before vs after the fix (max delta = 0.0000). So all headline ρ_g numbers in current ABLATION reports are valid.
- **Backups exist:** `scripts/transfer_analysis_v3/transfer_table.csv.bak_pre_arch_fix_<timestamp>` and `analysis_v3/symmetric_distances.csv.bak_pre_dino_merge_<timestamp>`.

### Data merges done this session
| What | Where | Provenance |
|---|---|---|
| `dino_fid` + `dino_sliced_w2` columns | `transfer_table.csv` (590 non-null rows) | Pulled from RC: `analysis_v3/symmetric_distances_dino.csv`. Merged into `analysis_v3/symmetric_distances.csv` first. |
| 5 combined flow+dino per-density pairwise CSVs | [analysis_v3/density_invariance_pair_sharded/combined/](../../analysis_v3/density_invariance_pair_sharded/combined/) | Concatenated flow + dino shards from RC pair-sharded density sweep. Each: 210 train_eval + 210 eval_eval rows. Used by `--dist` override in lean density sweep. |

---

## Current artifacts to read (in priority order)

| File | Why read it |
|---|---|
| **[ABLATION_strength.md](ABLATION_strength.md)** | The single most important report. Per-family ρ_g with 95% CIs, P(ρ_g > 0), shuffle-null mean ± std + z-score, and 3 paired-gap tests. Source of all the headline numbers above. |
| **[ABLATION.md](ABLATION.md)** | Cross-mode L-invariance summary (§1) + sym/fid/w2 family rows with CIs (§2). 12 directories combined. |
| **[ABLATION_density.md](ABLATION_density.md)** | Two-axis density stability: feature-side (Spearman ρ at each N vs baseline) + fitted-side (ρ_g per family across canon + 5 density levels). |
| [ABLATION_lean_density.md](ABLATION_lean_density.md) | Side-output from the lean sweep itself; redundant with ABLATION_density.md §2 but uses the generic compile_ablation_summary renderer. |
| [strength_per_family.csv](strength_per_family.csv), [strength_paired_gaps.csv](strength_paired_gaps.csv) | Machine-readable source for ABLATION_strength.md. |
| [HANDOFF.md](HANDOFF.md) | Full historical log of v4 design choices, things tried + ruled out, "things to NOT do" list. Still authoritative for design rationale. |
| [CLAIMS.md](CLAIMS.md) | Paper-prep document. May be stale relative to new numbers — needs an update pass. |

---

## What's NOT done (lean-completion options)

All of these are **reviewer-question insurance**, not load-bearing for the paper headline. None block writing.

| Item | Why you might add it | Cost |
|---|---|---|
| 5 missing feature subsets: `eps_16px`, `kl`, `kl_k5`, `kl_k20`, `asym_only` | Completes G4 (feature-subset robustness). The 6 fsubs already done show ridge down-weights weak features regardless; KL ones in particular are confounded by the feature-side instability finding. | ~1h lean / ~5h with bootstrap |
| Full 5×3 density sweep with bootstrap CIs (currently lean = point estimates only) | If a reviewer asks "are those density-level ρ_g differences statistically significant?" | ~6-8h |
| `auc_normalized` re-bootstrap with patched FAMILIES list | If you want CIs on motion_sym/appearance_sym/etc for the auc target the same way peak_pck has them | ~2h |
| Mixed-mode (PURE_ONLY=0) sanity run with the fixed table | Only if a reviewer asks "what happens without your purity filter?" Bug fix unblocks this cleanly. | ~6h |
| Spair_long raft re-training + re-patch | Currently 10 raft+spair rows still have OLD broken peak_pck (~3 on flyingthings vs ~70 expected). Cluster ran out before val completed. Doesn't affect pure-only headline because patches were applied where validation finished. | external (waiting on training) |

If you only do **one** more thing: run the 5 missing fsubs lean (peak_pck only, no GBM/ranknet, no bootstrap) so the §1 table in ABLATION.md has all 11 columns filled. That's the only "looks-incomplete" item in the current reports.

---

## How to reproduce / extend

### From scratch (full pipeline)
```bash
# 1. Build the table (now uses fixed arch detection)
python scripts/transfer_analysis_v3/build_table.py
python scripts/transfer_analysis_v4/patch_spair_long.py

# 2. Run canonical L-mode sweep (this is what produced results_mixed etc.)
for mode in mixed symmetric_informed symmetric_uninformed targeted_informed eb_shrunk density_idw; do
    OUT_DIR=scripts/transfer_analysis_v4/results_${mode} \
    L_MODE=${mode} N_BOOT=500 USE_RANKNET=1 \
        bash scripts/transfer_analysis_v4/run_v4.sh \
        > /tmp/v4_${mode}.log 2>&1 &
done
wait

# 3. Strength tests on the 6 L-mode dirs
python scripts/transfer_analysis_v4/strength_tests.py \
    --dirs results_mixed results_symmetric_informed results_symmetric_uninformed \
           results_targeted_informed results_eb_shrunk results_density_idw \
    --n-boot 300

# 4. Lean density sweep + density compile
bash scripts/transfer_analysis_v4/run_density_sweep_lean.sh   # ~55 min
python scripts/transfer_analysis_v4/compile_density_ablation.py

# 5. Compile the master ABLATION.md
python scripts/transfer_analysis_v4/compile_ablation_summary.py
```

### Just re-bootstrap (when you change `FAMILIES` list or bootstrap impl)
```bash
for d in results_mixed results_symmetric_informed results_symmetric_uninformed \
         results_targeted_informed results_eb_shrunk results_density_idw; do
    python scripts/transfer_analysis_v4/bootstrap.py \
        --results "scripts/transfer_analysis_v4/$d" --n-boot 300 &
done
wait
python scripts/transfer_analysis_v4/compile_ablation_summary.py
```

### Add a new density level
The diagonal levels are baked into `run_density_sweep_lean.sh` as
`LEVELS_FLOW=(50000 200000 1000000 4000000 8000000)` and
`LEVELS_DINO=(25000 100000 500000 2000000 4000000)`. To add another level:
1. Compute its pairwise CSV on RC (or locally) — see `density_invariance_train_eval_only.py` + the RC slurm scripts.
2. Concatenate flow + dino versions into `analysis_v3/density_invariance_pair_sharded/combined/pairwise_self_combined_flowN_dinoM.csv`.
3. Append to the LEVELS arrays in the sweep driver.

---

## File map (post-this-session, focused on what's new or changed)

| Path | Purpose | New/changed this session? |
|---|---|---|
| `scripts/transfer_analysis_v3/build_table.py` | Builds `transfer_table.csv` from feature CSVs + auc results | **Changed**: `_detect_arch()` replaces the regex remap (line 40-58) |
| `scripts/transfer_analysis_v3/transfer_table.csv` | Predictor input | **Rebuilt** — 1180 rows, catspp + raft only |
| `analysis_v3/symmetric_distances.csv` | flow_fid/sw2 + dino_fid/sw2 | **Merged** dino_fid/sliced_w2 from RC |
| `scripts/transfer_analysis_v4/run_v4.sh` | Single-mode driver | **Changed**: added `DIST` env override |
| `scripts/transfer_analysis_v4/bootstrap.py` | Entity-resampled bootstrap + summary.csv writer | **Changed**: FAMILIES list expanded from 6 to 20 |
| `scripts/transfer_analysis_v4/compile_ablation_summary.py` | Cross-mode ABLATION.md compiler | **Changed**: §2 sym/fid/w2 table renders with CIs |
| `scripts/transfer_analysis_v4/strength_tests.py` | NEW: per-family CI + P(>0) + shuffle-null + paired gaps | **NEW** |
| `scripts/transfer_analysis_v4/compile_density_ablation.py` | NEW: combines feature-side + fitted-side stability | **NEW** |
| `scripts/transfer_analysis_v4/zscore_residual_diagnostics.py` | NEW: z-scored residual scatter/hexbin from existing prediction rows; visualization diagnostic only | **NEW** |
| `scripts/transfer_analysis_v4/residual_calibration_diagnostics.py` | NEW: residual Pearson/slope/std-ratio diagnostics + post-hoc gain-calibrated figures from existing rows | **NEW** |
| `scripts/transfer_analysis_v4/run_density_sweep.sh` | Full-fat density sweep driver (5×17, with bootstrap) | **NEW** but unused — too expensive (~12-17h on a 32-core box) |
| `scripts/transfer_analysis_v4/run_density_sweep_lean.sh` | Lean density sweep driver | **NEW** — what actually got run |
| `scripts/transfer_analysis_v4/ABLATION_strength.md` | Output of strength_tests | **NEW** |
| `scripts/transfer_analysis_v4/ABLATION_density.md` | Output of compile_density_ablation | **NEW** |
| `scripts/transfer_analysis_v4/ABLATION_lean_density.md` | Side-output of run_density_sweep_lean.sh | **NEW** |
| `scripts/transfer_analysis_v4/ABLATION.md` | Cross-mode summary | **Refreshed** after re-bootstrap |
| `scripts/transfer_analysis_v4/strength_per_family.csv` | Source data for ABLATION_strength.md | **NEW** |
| `scripts/transfer_analysis_v4/strength_paired_gaps.csv` | Source data for ABLATION_strength.md | **NEW** |
| `scripts/transfer_analysis_v4/_archive_partial_<ts>/` | 2 partial fsub dirs (`eps_16px`, `kl`) from the killed full sweep | **NEW** |
| `scripts/transfer_analysis_v4/_archive_pre_density_sweep_<ts>/` | All 17 results_* dirs from before the killed full sweep | **NEW** (the full sweep was killed after 23h at 12/17 modes done; those 12 became the new canonical results_* dirs) |
| `analysis_v3/density_invariance_pair_sharded/combined/` | 5 combined flow+dino pairwise CSVs (one per density level) | **NEW** |

Old reference files (unchanged this session, still authoritative for what they cover):
- [HANDOFF.md](HANDOFF.md) — historical session log + design rationale + things-to-not-do list
- [README.md](README.md) — quick-start
- [CLAIMS.md](CLAIMS.md) — paper-prep claim → evidence map (may be stale; needs update pass)
- [INTERVENTIONAL_STUDY.md](INTERVENTIONAL_STUDY.md) — downstream goal (predictor-guided hyperparameter search for kubric/SDF generation)
- [RC_CLUSTER_PLAN.md](RC_CLUSTER_PLAN.md) — RC cluster compute plan (mostly done; Step 5 = merge back = done)
- [REVIEW_RESPONSE_ASSESSMENT.md](REVIEW_RESPONSE_ASSESSMENT.md) — reviewer-attack inventory

---

## Things to NOT do (carried forward from HANDOFF.md, still apply)

- **Don't fold L into the ranking score on LOTO.** LOO anti-correlation; ρ collapses.
- **Don't drop the random / shuffle controls.** They're the proof the result isn't an artifact.
- **Don't blindly add features.** N≈540 means the feature-to-sample ratio is borderline; ridge handles it with regularization but more features ≠ more signal.
- **Don't change CV regimes.** LOTO/LOBO/JOINT map cleanly to Park-Marcotte C2/C3. The v3 trap was adding more.
- **Don't use the rank scatter (`fig6`).** Unreadable at N=10/context. Use fig8 (per-context ρ histogram) instead.

**Added this session:**
- **Don't run with PURE_ONLY=0 before confirming the `_detect_arch()` fix is in place.** Pre-fix mixed-mode runs would have included spurious `spair_only` contexts contributing duplicate-but-unpatched peak_pck signal. Current `build_table.py` is correct; just be aware if you check out an older revision.
- **Don't add new model families** like `appearance_<X>` or `motion_<X>` without also extending `bootstrap.py`'s `families` list. The old bug (6-family hardcode) silently dropped sym/fid/w2 families from `summary.csv` for months; the patched list at [bootstrap.py:210](bootstrap.py#L210) now mirrors `experiments.py`'s FAMILIES.
- **Don't interpret residual scatter as the headline claim.** `ctx_rho_g` is Spearman ranking. Residual magnitude calibration is now tracked separately in `RESIDUAL_CALIBRATION.md`; raw ridge residuals can be under-dispersed even when the ranking signal is real.

---

## Sessions / changelog (append on new sessions)

### 2026-05-27 — RC sweep merge + density sweep planning
- Pulled RC density-invariance results: 5 flow levels (50k–8M), 5 dino levels (25k–4M); stability CSVs + heatmaps merged to `analysis_v3/density_invariance_pair_sharded/`.
- RC `symmetric_distances_dino.csv` pulled and merged into `analysis_v3/symmetric_distances.csv` → `dino_fid` + `dino_sliced_w2` now in `transfer_table.csv`.
- Launched a full 5×17 density sweep (canonical 17 modes + 5 density levels × 3 headline L-modes). Estimated ~17h. **Reality**: ~6h per mode-pair due to BLAS + bootstrap costs; killed at 23h with 12/17 canonical modes done.

### 2026-05-28 — Lean recovery + strength tests + spair_only bug fix
- Killed the over-running full sweep at 12/17 canonical modes done. Archived the partial fsub dirs (`_archive_partial_*`).
- Compiled an interim ABLATION.md from the 12 completed L-mode + fsub dirs.
- Discovered [bootstrap.py:210](bootstrap.py#L210) FAMILIES list was missing `motion_sym/appearance_sym/fid/w2/mmd/nullk/size/supervision_density/motion_size/motion_supdensity/motion_km`. Patched.
- Wrote `strength_tests.py` (per-family CIs + P(>0) + shuffle-null + 3 paired-gap tests) and `compile_density_ablation.py` (feature-side + fitted-side stability into one report).
- Ran lean density sweep (`run_density_sweep_lean.sh`): 6 sequential runs, ~55 min wall.
- Ran chained post-sweep: re-bootstrap 6 L-mode dirs (parallel-2), recompile ABLATION.md, run strength tests, compile density ablation. Total ~7h.
- **Found the `spair_only` bug**: 160 rows had `model_family="spair_only"` (a snapshot-dir name) due to incomplete regex in `build_table.py`. Fixed with `_detect_arch()` token-based detection.
- **Verified no impact on headline numbers**: rebuilt `transfer_table.csv` (1340 → 1180 rows); all 550 pure-only catspp+raft cells have byte-identical peak_pck/auc_normalized between buggy and fixed tables (max delta = 0.0000). Headline numbers in the existing ABLATION_strength.md / ABLATION.md are valid as-is.
- Decision: did NOT re-run the post-sweep chain after the fix (waste of compute for empirically zero change). The fix protects future mixed-mode runs and is the correct data model going forward.

#### Headline numbers as of 2026-05-28 end of session
(target=peak_pck, mixed L-mode, ridge head, pure-only, N_BOOT=300, entity-resampled)

| family | LOTO ρ_g [CI] | LOBO ρ_g [CI] | JOINT ρ_g [CI] |
|---|---|---|---|
| motion | +0.476 [+0.229, +0.663] | +0.450 [+0.368, +0.534] | +0.300 [+0.112, +0.457] |
| **motion_sym** | +0.413 [+0.192, +0.554] | **+0.533 [+0.475, +0.592]** | +0.401 [+0.235, +0.559] |
| motion_w2 | +0.449 [+0.271, +0.579] | +0.481 [+0.394, +0.581] | +0.464 [+0.319, +0.585] |
| motion_fid | +0.440 [+0.228, +0.574] | +0.467 [+0.374, +0.572] | +0.446 [+0.303, +0.558] |
| appearance | -0.220 [-0.455, +0.105] | +0.074 [-0.019, +0.174] | -0.182 [-0.333, -0.009] |
| appearance_sym | +0.062 [-0.222, +0.408] | +0.141 [-0.005, +0.299] | +0.011 [-0.156, +0.163] |
| appearance_mmd | +0.197 | +0.341 | +0.272 |

Paired gaps (LOBO):
- motion − appearance: +0.373 [+0.277, +0.473], P(>0)=1.000
- motion_sym − appearance_sym: +0.390 [+0.243, +0.544], P(>0)=1.000
- motion − random: +0.529 [+0.417, +0.652], P(>0)=1.000

Reproducibility hash points: `transfer_table.csv` MD5 (post-fix) and `bootstrap.py` FAMILIES list version are the two things that matter for matching these numbers. If your re-run differs by more than ~0.01, check those first.

### 2026-05-28 — Residual calibration diagnostics for the mean_nn concern
- Did **not** relaunch `/tmp/v4_post_sweep.sh`; no full pipeline rerun was needed.
- Added `zscore_residual_diagnostics.py` to make standardized residual scatter/hexbin figures from existing prediction rows. This answers whether the apparent pile-up at zero is from cross-context scale pooling.
- Added `residual_calibration_diagnostics.py` to compute residual Pearson, calibration slope, std(pred residual)/std(actual residual), and diagnostic post-hoc gain-calibrated scatter/hexbin figures.
- Ran both diagnostics on `results_fsub_mean_nn`, `target=peak_pck`, heads `g` and `g_zridge`.
- Output report: [results_fsub_mean_nn/RESIDUAL_CALIBRATION.md](results_fsub_mean_nn/RESIDUAL_CALIBRATION.md). Machine-readable CSV: `results_fsub_mean_nn/figures/residual_calibration__peak_pck__g_g_zridge.csv`.
- Key read: mean_nn ridge has real residual association for motion, especially LOBO (`ctx_spearman=+0.501`, `ctx_pearson=+0.563`), but it is under-dispersed (`ctx_std_ratio_median=0.262` on LOBO motion). This supports the wording: **ranking signal is stronger than residual magnitude calibration**.
- For interventional search, treat `g` as a ranking prior, not a calibrated oracle. If the search uses residual magnitudes across variants, add target-context calibration/support checks rather than changing the scientific estimand.

### 2026-05-28 PM (later) — sanity checks corrected the mechanism, wiring decision: keep separate
- Ran 3 sanity checks on the motion_sym + benchsim story:
  - **#1**: `--family motion --feature-subset all` (all 13 motion features). Raw std ratio 0.72 LOTO / 0.87 LOBO / 0.63 JOINT. Benchsim improves pearson (LOBO 0.520 → 0.626) without over-amplifying.
  - **#2**: `--family motion_fid` and `--family motion_w2` (single-feature). Both have raw std ratio ≥ 0.49 and benefit cleanly from benchsim (LOBO pearson 0.384 → 0.518 for fid; 0.370 → 0.505 for w2).
  - **#3**: `motion_sym --kernel-space dino` (changed flow → DINO `mean_nn_sym` kernel via new `--kernel-space` flag in [context_scale_calibration.py:394](context_scale_calibration.py)). LOBO pearson 0.646 vs flow's 0.691. **DINO kernel works almost as well as flow** — the "geometric axis alignment" story I wrote earlier doesn't hold up.
- **Corrected mechanism**: benchsim's failure on `mean_nn` was about **under-dispersion severity** (raw median std ratio 0.26 LOBO — about 2× worse than any other tested family), not about feature axis. With raw std ratio < ~0.4 the per-context gains the kernel tries to smooth are large and noisy; smoothing produces over-amplified results. Any motion family with raw std ratio ≥ ~0.4 benefits cleanly.
- **Wiring decision: keep calibration as a separate study (not folded into v4 sweep).** Ranking ρ_g stays in ABLATION.md / ABLATION_strength.md as the headline scientific claim. Residual calibration lives in [ABLATION_calibration.md](ABLATION_calibration.md) as a methodological deliverable for the interventional search.
- **New compiler script**: [compile_calibration_ablation.py](compile_calibration_ablation.py) scans `results_*/context_scale_calibration*/` dirs and emits ABLATION_calibration.md aggregating 6 calibration runs (mean_nn + motion all 13 + motion_fid + motion_w2 + motion_sym flow-kernel + motion_sym dino-kernel) with per-(family, split) raw-vs-best-head tables + full per-head detail + mechanism note.
- **STATUS.md / CLAIMS.md / HANDOFF.md edits earlier in this session were partially wrong** about the "geometric alignment" mechanism. Those have been corrected to the under-dispersion-threshold story above.

### 2026-05-28 PM — motion_sym calibration extension (end-to-end winner found)
- Extended `FAMILIES` in `residual_calibration_diagnostics.py` + `zscore_residual_diagnostics.py` to include `motion_sym, motion_fid, motion_w2, appearance_sym`. The 8-family residual calibration is now in [results_mixed/RESIDUAL_CALIBRATION.md](results_mixed/RESIDUAL_CALIBRATION.md) with fig11/12/13_* for both heads.
- Ran the full leakage-clean replay (`context_scale_calibration.py`) with `--family motion_sym --feature-subset all` on `results_mixed`, both `all_variants` and `drop_false_true` variant filters. Outputs in [results_mixed/context_scale_calibration_motion_sym/](results_mixed/context_scale_calibration_motion_sym/) (summary CSVs + 54 figures incl. `grid_*` overview plots).
- **Headline shift: the calibrated head winner for the interventional study is `motion_sym + g_benchsim_gain`** — not `mean_nn`'s shrink/variant gain. End-to-end on LOBO:

  | family / head | ctx_spearman | ctx_pearson | std ratio (med) | pooled std ratio | abs r(L+g) |
  |---|---|---|---|---|---|
  | motion mean_nn raw g | +0.501 | +0.413 | 0.262 | 0.403 | +0.737 |
  | motion_sym raw g | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
  | motion_sym g_shrink_gain | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
  | **motion_sym g_benchsim_gain** | **+0.536** | **+0.691** | **1.111** | **0.990** | +0.705 |
  | motion_sym g_profilesim_gain | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |

- **Why benchsim works for motion_sym but failed on mean_nn (CORRECTED by 2026-05-28 PM later sanity checks)**: the earlier claim was "motion_sym features (FID/SW2/MMD) align with the `mean_nn_sym` flow geometry." That doesn't hold up. The real driver is **raw-prediction under-dispersion severity**. mean_nn's restriction collapses ridge predictions to median std ratio 0.26 LOBO; per-context gains needed to fix that are large (~3–5×) and noisy, so IDW-smoothing across benchmarks amplifies the wrong scale. motion_sym's natural std ratio is 0.69 LOBO — per-context gains are modest (~1.2×), IDW-smoothing well-behaved. Verified: motion (all 13), motion_fid, motion_w2 all work with benchsim; DINO kernel for motion_sym also works (LOBO pearson 0.646 vs flow's 0.691). See [ABLATION_calibration.md](ABLATION_calibration.md) and the next session entry below.
- LOTO: same ordering, `g_benchsim_gain` wins both ranking (+0.466) and pearson (+0.511). JOINT: profilesim slightly beats benchsim on pearson (+0.468 vs +0.448) but with worse pooled std ratio (0.728 vs 0.693). Recommendation: benchsim for LOTO/LOBO, profilesim as JOINT fallback.
- **Mechanism, corrected for the paper:** the motion_sym ridge fits three separate sym-distance coefficients (`flow_fid`, `flow_sliced_w2`, `flow_mmd`), giving the ranking signal. A leakage-clean per-fold scale gain — IDW-smoothed across same-variant other-benchmarks using flow `mean_nn_sym` as the kernel — recovers residual magnitude calibration without changing ranks. The kernel just needs the raw predictions to be in a reasonable scale range (raw std ratio ≥ ~0.4); given that, the IDW-smoothed per-context gain produces well-calibrated predictions.
- For the interventional search: use `g_benchsim_gain` as the predicted residual head for kubric/SDF candidates; predicted residual magnitudes now span the actual residual range, fixing the "saturated bad regions" pile-up at zero. Falls back to `variant_gain` cleanly when there are no neighbor benchmarks (see `_benchsim_gain` in `context_scale_calibration.py:251-254`).

### 2026-05-28 — Leakage-clean context-scale calibration quick test
- Added `context_scale_calibration.py`, a small replay diagnostic for `peak_pck / motion / mean_nn` that fits residual scale gains using only each fold's training rows.
- Ran both all-variant and `False|True`-dropped versions; outputs live under `results_fsub_mean_nn/context_scale_calibration/`.
- Main read: raw ridge is under-dispersed, and a simple fold-training gain fixes much of the scale without changing the basic ranking story.
  - All variants: LOTO median std ratio `0.358 -> 1.148` with shrink gain; context-centered Pearson `0.243 -> 0.395`.
  - Drop `pretrained=False, freeze=True`: LOTO median std ratio `0.323 -> 1.149`; context-centered Pearson `0.207 -> 0.366`.
  - LOBO/JOINT also improve in scale; LOBO drop-`False|True` median std ratio `0.248 -> 0.591`, Pearson `0.367 -> 0.426`.
- Added a first feature-informed calibration attempt, `g_benchsim_gain`: same-variant gains smoothed over other benchmarks using flow mean-NN benchmark similarity, excluding the same benchmark. This is the "closer to zero-shot" calibration test.
- Result: `g_benchsim_gain` is **not** the default recommendation. It over-amplifies residuals in this form (all-variant LOTO pooled std ratio `2.758`; drop-`False|True` LOTO pooled std ratio `2.577`) and lowers absolute `L+g` Pearson versus shrink/variant gains. The feature-informed idea is worth keeping as an ablation, but needs stronger partial pooling before use.
- Added `g_profilesim_gain`, using standardized eval-side dataset profile/density distance (`log_eval_n_samples`, `log_eval_n_vectors`, valid-vector-per-sample summaries) to smooth gains across same-variant benchmarks.
- Result: profile-sim is more sensible than flow-benchmark-sim but still not best. Drop-`False|True` LOTO profile-sim has Pearson `0.325`, pooled std ratio `1.467`, absolute `L+g` Pearson `0.785`; shrink gain remains better (`0.366`, `1.109`, `0.839`). LOBO profile-sim is close but still trails variant/shrink.
- Printed the flow mean-NN benchmark neighborhoods: KITTI2012/KITTI2015 are very close (`0.0008`) and PF-PASCAL/PF-WILLOW are close (`0.0184`), but SPair is closer to synthetic (`0.0047`) and TSS (`0.0231`) than PF-PASCAL (`0.0505`). The intuitive semantic grouping was not fully reflected by flow mean-NN.
- Interpretation: calibration is possible and leakage-clean as a diagnostic, but it should remain separate from the headline Spearman ranking claim. For the interventional search, log calibrated scores plus clipping/support diagnostics and use calibrated magnitude as a tie-breaker/triage signal, not a standalone oracle.

### 2026-06-04 — Calibration ceiling: recalibration + residual feature search (both NULL)

Two leakage-clean diagnostics to answer "can we fix absolute-magnitude calibration?" Both say no, and explain why. **Ranking stays the deliverable.**

**Ridge alpha context:** `_fit_ridge` uses `RidgeCV(alphas=[0.01..1000])` ([experiments.py:467](experiments.py#L467)); it selects **α=100** (MSE-optimal → deliberately under-dispersed = the calibration miss). ρ_g is scale-invariant so a lower α would expand g at little ranking cost, but at N=11 sources/context it risks OOD-candidate overfit. Don't de-tune α; the residual search below is the real answer.

**[`calibrate.py`](calibrate.py)** (NEW) — leakage-clean recalibration of `L+g`, leave-one-source-out. Methods: raw, global_gain (L+b·g), global_affine, percell_affine, intercept_cell+slope_g. Run: `python scripts/transfer_analysis_v4/calibrate.py --split LOTO --family motion`. Output: `results_mixed/calibration_LOTO_motion.csv` (reliability deciles).
- **No method beats raw `L+g`** (RMSE 13.96, slope 0.955). `percell_affine` OVERFITS (11 sources/cell) and destroys ranking (ctx_rho 0.396 → −0.120). Random control dev_R² −5.3 (method validated).
- In-distribution absolute calibration is ALREADY good — reliability deciles on the diagonal, slope 0.955. The ~−10 interventional misses are **OOD optimized candidates (extrapolation)**, not general miscalibration.
- Within-cell magnitude is weak: out-of-fold dev_R² **0.035** (g ranks at ctx_rho 0.40 but explains ~3% of within-cell magnitude variance).

**[`residual_feature_search.py`](residual_feature_search.py)** (NEW) — regress residual `r = actual − L − g` (within-cell demeaned) on every dataset descriptor in `transfer_table.csv`, leave-one-source-out. Run: `python scripts/transfer_analysis_v4/residual_feature_search.py --split LOTO`.
- **Nothing explains r out-of-fold.** appearance(dino) R² −1.08, density/size −1.12, flow/dino isolation −0.18/−0.14, zero_flow −0.04, ALL-combined −0.58. Control `motion(g-saw)` −0.03 ≈ 0 (validates method — g already used the motion features).
- Within-cell signal is REAL, not noise: across-source σ ≈ **7–17 PCK** (KITTI ~17) ≫ ±1–2 seed noise — but it is **not a function of any static dataset statistic** → training dynamics / unencoded source quality.

**Conclusion:** transfer magnitude beyond the motion-distance ranking is not predictable from current features. **Ranking (ρ≈0.4–0.5) is the ceiling.** The `g_benchsim/shrink_gain` only extracts the rank-implied ~0.15 R² (useful for EXTREME/optimized candidates, e.g. KITTI 87.55→93.70).
**One stone unturned:** higher-order flow self-moments (mean magnitude, kurtosis) are NOT in `transfer_table.csv` — the only untested feature avenue; compute from coverage vectors + re-run before declaring an exhaustive negative.
**Caveat:** p/n severe (19–26 feats / 11 sources) so the −1.0 magnitudes are overfit-inflated; conclusion robust (real signal → positive oof R²). A top-3-per-family reduced run would give cleaner numbers.

---

## 2026-06-04 — Robustness tooling: generator-family hold-out + cluster bootstrap

**Motivation.** The 11 "pure" sources are not 11 independent draws — they collapse
to ~5 **generator families** (`FAMILY_MAP` in `experiments.py`): `sdf3d`
(synthetic + large_zoom/small_zoom/random_flipping), `warp2d` (synthetic_2d_warp +
imagenet2dwarp), `kubric` (movi_f), `realflow` (flyingthings/pointodyssey/sintel),
`semantic` (spair). Treating them as independent overstates robustness. Two tests
were implemented (Tier 1 drop-one-source also available but de-prioritized).

**Feature-scale sanity (settles the "tiny DINO" worry).** Raw scales differ wildly
(flow_fid≈16k vs dino_fid≈0.6) but this does NOT bias the model: `_fit_ridge`
(experiments.py:460) does winsorize→median-impute→**StandardScaler**→RidgeCV, so
every feature is z-scored before α selection — the appearance-null is not a scaling
artifact. DINO is not saturated globally (dino_fid CV 0.40–0.65 across sources), but
among the 5 SYNTHETIC sources→kitti2015 **dino_fid CV=0.028 vs flow_fid CV=0.612** —
appearance is degenerate among synthetic sources while motion spreads 61×. That is
the *mechanism* of the appearance-null, not a bug. (diag: read
`analysis_v3/symmetric_distances.csv`, CV=std/mean across the 11 sources.)

**Tier 2 — leave-one-generator-family-out** (`experiments.py --drop-family FAM`):
refit the WHOLE pipeline with each family removed. Headline survives iff motion
`ctx_rho_g` stays clearly + and the motion−appearance gap stays >0 under every drop.
  - `python experiments.py --pure-only --drop-family sdf3d --targets peak_pck --out <dir>`
  - also `--drop-source SRC...` for Tier 1.

**Tier 3 — cluster bootstrap** (`bootstrap.py --cluster`): resamples at the family
level (~5 clusters) not the source level (11), via `Prepared(cluster=True)`. Honest
(wider) CIs under within-family correlation. Writes `summary_cluster.csv` +
`bootstrap_gap_cluster.csv` (does not clobber the source-level files). Smoke-checked:
motion ρ_g `+0.482 [-0.116,+0.620]` at n_boot=50 — legitimately wider lower tail.

**Driver + compile:**
  - `bash scripts/transfer_analysis_v4/run_robustness.sh`  (env: TARGETS, N_BOOT,
    FAMILIES, SKIP_FIGURES) — runs the full-data cluster bootstrap + all 5 family
    drops, then compiles.
  - `compile_robustness.py` → `results_robust/ROBUSTNESS_SUMMARY.{csv,md}`: one row
    per perturbation with motion ρ [CI], appearance ρ, gap [CI], P(gap>0), n_rows.
  - `run_v4.sh` now honors env: `DROP_FAMILY`, `DROP_SOURCE`, `CLUSTER=1`.
  - Est. ~60–70 min for the full suite (6 × experiments+bootstrap at N_BOOT=1000).
