# Transfer Analysis v4 — focused, claim-driven sweep

v4 is a thin analysis layer on top of v3's feature outputs. **v3 still produces
the features** (`transfer_table.csv`, `pairwise_self_distances.csv`); v4 just
does the modeling, bootstrap, figures, and report compilation — re-organized
around one scientific claim instead of dozens of model variants.

## The scientific claim

> Motion-domain (i→k) cross-distance from a training source to a target
> benchmark predicts within-context transfer performance of correspondence
> models. Appearance-domain (DINO) cross-distance does not.

The headline is **within-context Spearman ρ on `g`** (the within estimator).
Calibration (`L + g`) is reported as a sanity check; the level term `L` is
largely identity memorization and is family-blind on LOBO.

Residual magnitude calibration is deliberately reported separately from the
ranking claim. Use `residual_calibration_diagnostics.py` on existing
prediction rows to measure Pearson residual association, calibration slope,
and predicted-vs-actual residual dispersion.

## The model

One model under three regimes:

```
perf(i, k, v) = L(i, k, v) + g(features(i → k))
```

- `g` = within-context fixed-effects ridge on within-`cv` demeaned features.
        cv = (benchmark, variant). This is the estimand.
- `L` = observed-or-back-off cell band:
        LOTO: in-fold cell mean (benchmark seen)
        LOBO: eval-side IDW over neighbor benchmarks
        JOINT: grand + γ_v (only variant observed)
- rank_score = g (the claim; identity-free; leakage-clean)
- abs_pred   = L + g (calibrated; mostly identity-memorization on LOBO)

## What runs

`bash scripts/transfer_analysis_v4/run_v4.sh`

Order:
1. `experiments.py` — sweep families × splits, write per-row predictions.
2. `bootstrap.py` — entity-resampled 95% CIs on ctx_rho, cent_rho, abs_r.
3. `figures.py` — headline bars + global scatter + residual scatter + controls.
4. `compile_v4.py` — `results/results.md` with claim-structured sections.

Optional diagnostics, no full rerun needed:

```bash
# (1) Standardized residual scatter/hexbin — within-context centered visualization.
python scripts/transfer_analysis_v4/zscore_residual_diagnostics.py \
    --results-dir scripts/transfer_analysis_v4/results_mixed \
    --target peak_pck --head g

# (2) Single-α post-hoc residual gain — diagnostic only, not leakage-clean.
python scripts/transfer_analysis_v4/residual_calibration_diagnostics.py \
    --results-dir scripts/transfer_analysis_v4/results_mixed \
    --target peak_pck --heads g g_zridge

# (3) Leakage-clean replay with per-fold gain heads (global / variant /
# context / shrink / benchsim / profilesim). This is the right one to use
# when you need a calibrated residual head for downstream search.
python scripts/transfer_analysis_v4/context_scale_calibration.py \
    --family motion_sym --feature-subset all \
    --out-dir scripts/transfer_analysis_v4/results_mixed/context_scale_calibration_motion_sym

python scripts/transfer_analysis_v4/plot_context_scale_calibration.py \
    --calib-dir scripts/transfer_analysis_v4/results_mixed/context_scale_calibration_motion_sym \
    --variant-filter all_variants \
    --heads g g_shrink_gain g_benchsim_gain g_profilesim_gain
```

End-to-end winner from (3) on motion_sym, LOBO peak_pck: `g_benchsim_gain`
preserves ranking (ctx_spearman +0.536) while bringing residual magnitude
calibration to ctx_pearson +0.691 and pooled std ratio 0.990. See
[CLAIMS.md](CLAIMS.md) §"Best calibrated head" for the full split breakdown.

After running calibration for any new family, regenerate the cross-family
summary report:

```bash
python scripts/transfer_analysis_v4/compile_calibration_ablation.py
# scans results_*/context_scale_calibration*/ and writes ABLATION_calibration.md
```

[ABLATION_calibration.md](ABLATION_calibration.md) is the canonical
calibration summary — kept deliberately separate from the headline ranking
claims in ABLATION.md / ABLATION_strength.md. Mechanism note: benchsim works
when the raw `g` median std ratio is ≥ ~0.4; the only failure mode tested
(mean_nn-only feature subset, std ratio 0.26) is a feature-restriction
artifact.

Inputs (must already exist from v3):
- `scripts/transfer_analysis_v3/transfer_table.csv`
- `analysis_v3/pairwise_self_distances.csv`

Outputs (under `scripts/transfer_analysis_v4/results/`):
- `predictions/rows_<split>_<family>.csv` per (split, family)
- `summary.csv` — point estimates and CIs
- `bootstrap_gap.csv` — motion − appearance gap CIs
- `figures/*.png`
- `results.md` — the report
- `RESIDUAL_CALIBRATION.md` — optional post-hoc residual calibration report
  when `residual_calibration_diagnostics.py` is run

## Families

- `motion` — 13 matched flow self-distance metrics (se_flow_*)
- `appearance` — 13 matched DINO self-distance metrics (se_dino_*)
- `both` — both (26 features)
- `random` — 13 dim-matched gaussian noise (the honest g-only floor)

## Splits

- `LOTO` — Park-Marcotte C2 (new source)
- `LOBO` — Park-Marcotte C2 (new benchmark)
- `JOINT` — Park-Marcotte C3 / Pahikkala S4 (both unseen)

## Controls

- **shuffle-target** — permute outcomes within context; refit g; ctx_rho should ≈ 0
- **random features** — 13 gaussian columns; ctx_rho should ≈ 0 (honest floor)
- **level_only** — rank by L alone; on LOBO ≈ 0.72 *identical across families*
  (proves the level is identity-borrowing, family-blind)

## Ablations (opt-in, behind flags)

- `--levels A,B,C` — alternative cold-start level estimators (B = RLFM-style
  feature-regressed, C = empirical-Bayes shrinkage). Default: A only.
- `--family-matched-prior` — appearance LOBO with DINO-IDW level instead of
  FLOW-IDW. Demonstrates the motion advantage is in g, not the prior space.
