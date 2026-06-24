# Transfer Analysis v5 — Regime-Direction Law

One clean, reproducible run producing every table in
`Paper 2 - Outline v2 (Regime-Direction)` (Obsidian, Project/). Finding +
verification narrative: `REGIME_DIRECTION_FINDING.md` (repo root) and
`../transfer_analysis_v4/regime_direction_verification/REPORT.md`.

```
bash scripts/transfer_analysis_v5/run_v5.sh                 # full (~25 min, N_BOOT=200)
SKIP_EXPERIMENTS=1 bash scripts/transfer_analysis_v5/run_v5.sh   # reuse fits (~5 min)
```

## Outline-v2 table → artifact map

| outline | content | artifact |
|---|---|---|
| Table 1 | per-variant precision/recall/sym + d CIs; permutation/LOBO/self-pair checks; DINO control | `v4/regime_direction_verification/REPORT.md`, `REPORT_dino.md`, `master_table_mean_nn.csv` |
| Table 2 | rule vs all predictors under LOTO/LOBO/JOINT (+ CIs) | `v4/results_rule_v5core/summary.csv`, `bootstrap_gap.csv` (motion_rule vs appearance gap) |
| Table 3 | ceiling: rule consensus fraction 0.79/0.80/0.81 of 0.731 | `v4/results_rule_v5core/CONSENSUS_RULE.csv`; per-variant fraction in `results/rule_holdout_checks.csv` + finding doc |
| Table 4 | absolute prediction L+rule (MAE 8.99 / r 0.874 LOTO) | `v4/results_rule_v5core/summary_points_peak_pck.csv` (MAE_Lg, abs_r_Lg rows) |
| Table 5 | selection regret + gap-stratified accuracy | `results/selection_regret_rule.csv`, `results/pairwise_gap_rule.csv` |
| Table 6 | asym vs sym (rule vs mean_nn_sym/FID/W2/MMD) | `results/asym_vs_sym.csv` |
| §7.3 | pre-registered out-of-sample test on the kubric grid | `results/intervention_oos.csv` (FF verdict: PASS) |

## Key numbers (2026-06-10 run)

- Rule (fit-free): mean ctx ρ +0.50; under folds +0.49/+0.50/+0.52; ceiling fraction ≈0.99 (inter-variant), 0.79/0.80/0.81 (cross-arch consensus).
- Absolutes: L+rule MAE 8.99 PCK, Pearson 0.874 (LOTO).
- Regret (rule): median 2.3–3.4 PCK vs 13.3 random; P(best in top3) 0.56–0.59.
- Asym vs sym: rule +0.51 vs sym +0.47 / FID +0.43 / W2 +0.42 / MMD +0.16; rule wins by +0.30–0.40 in flip-extreme (pretrained GLU-Net) cells.
- OOS (kubric grid, FF arm, 5-seed-averaged distances): **precision +0.62 > sym +0.10 > recall −0.13 — DECISIVE PASS.** TT arm n=3, non-discriminative. KITTI-family cells are matched-motion appearance ablations by construction (motion near-tied; appearance dominates those cells).
- Sampling stability (fresh, 2026-06-10): near-tied sources need seed-averaged distances (single-subsample precision rank-corr across seeds 0.49 → averaged over 5 seeds; recall 0.88). `intervention_distances_directional.py` output is now 5-seed-averaged.
- Rule CIs (N=200): LOTO +0.49 [+0.29,+0.59], LOBO +0.50 [+0.46,+0.54], JOINT +0.52 [+0.36,+0.57]; abs_r LOTO 0.874 [0.793,0.949]. Gap-stratified (rule): >10 PCK pairs 0.79–0.82, above cross-arch retraining (0.77).

## Upstream inputs (not regenerated here)

- `scripts/transfer_analysis_v3/transfer_table.csv` (build_table.py, v3)
- `analysis_v3/pairwise_self_distances.csv` (v3 self-distances; orientation verified a=train 110/110)
- Intervention grid: `/mnt/nvme_1tb_a/snapshots/transfer_grid/` + directional distances via
  `le-wm/outputs/intervention_distances_directional.py` (cached source vectors in `_cache_src_vectors/`)
