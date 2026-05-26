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

Inputs (must already exist from v3):
- `scripts/transfer_analysis_v3/transfer_table.csv`
- `analysis_v3/pairwise_self_distances.csv`

Outputs (under `scripts/transfer_analysis_v4/results/`):
- `predictions/rows_<split>_<family>.csv` per (split, family)
- `summary.csv` — point estimates and CIs
- `bootstrap_gap.csv` — motion − appearance gap CIs
- `figures/*.png`
- `results.md` — the report

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
