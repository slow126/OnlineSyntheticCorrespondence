# Hard Results — Regime-Direction Law (consolidated, 2026-06-10)

Every number below is on disk now; nothing pending. Source artifact cited per table.
Regenerate everything: `bash scripts/transfer_analysis_v5/run_v5.sh`

## T1. The Law — per-variant directional ρ (fit-free, peak_pck, benchmark-bootstrap CIs)
`scripts/transfer_analysis_v4/regime_direction_verification/REPORT.md`

| variant | precision (train→tgt) | recall (tgt→train) | sym | d [95% CI] |
|---|---|---|---|---|
| catspp F/F (scratch) | **+0.558** | +0.034 | +0.494 | +0.53 [+0.29,+0.77] |
| catspp F/T (scratch) | **+0.533** | +0.070 | +0.490 | +0.46 [+0.22,+0.69] |
| raft (scratch — config-verified, see T9) | **+0.508** | +0.285 | +0.542 | +0.22 [−0.12,+0.57] |
| glunet F/F (scratch) | +0.335 | +0.420 | +0.479 | −0.09 [−0.44,+0.32] |
| glunet F/T (scratch) | +0.421 | +0.313 | +0.609 | +0.11 [−0.25,+0.47] |
| catspp T/F (pretr) | +0.084 | **+0.441** | +0.455 | −0.36 [−0.67,+0.02] |
| catspp T/T (pretr) | +0.169 | **+0.438** | +0.546 | −0.27 [−0.54,+0.01] |
| glunet T/F (pretr) | −0.135 | **+0.605** | +0.307 | −0.74 [−1.08,−0.33] |
| glunet T/T (pretr) | −0.163 | **+0.734** | +0.334 | −0.90 [−1.11,−0.65] |

Flip statistic +0.818; exact permutation **p=0.0286** (RAFT excluded, C(8,4)=70) and
**p=0.0079** with config-verified RAFT included as scratch (C(9,5)=126). LOBO-stable
[+0.73,+0.90]; self-pair robust; replicates in eps4px/eps16px/KL-k20; auc consistent.
DINO control: NO flip (p=0.54) — motion-specific.

## T2. Rule vs every learned predictor (ctx ρ, all-variant mean; CIs N=200)
`results_rule_v5core/summary.csv`; pooling runs in results_perarch_*/results_regime_*/results_hier_*

| predictor | LOTO | LOBO | JOINT |
|---|---|---|---|
| **fit-free regime rule** | **+0.49 [+0.29,+0.59]** | **+0.50 [+0.46,+0.54]** | **+0.52 [+0.36,+0.57]** |
| fit-free symmetric | +0.47 | +0.47 | +0.47 |
| shared 3-arch ridge (13 feat) | +0.31 | +0.36 | +0.18 [−0.01,+0.29] |
| arch pools (catspp+raft \| glunet) | +0.35 | +0.43 | +0.23 |
| regime pools | +0.28 | +0.43 | +0.20 |
| hierarchical ridge (arch interactions) | +0.30 | +0.46 | +0.29 |
| depth-3 tree (+ regime flags) | +0.07 | — | — |
| appearance (13 feat) | −0.22 | +0.16 | −0.21 |

## T3. Ceilings — how much recoverable signal the rule recovers
- Per-variant: rule +0.498 ≈ oracle direction +0.502 vs inter-variant agreement
  +0.528 → **fraction ≈ 0.99** (3 variants >1). `rule_holdout_checks.csv` + finding doc.
- Cross-architecture consensus (held-arch, 3 archs): rule captures **0.79 / 0.80 / 0.81**
  of the 0.731 ceiling (LOTO/LOBO/JOINT) — uniform across splits.
  `results_rule_v5core/CONSENSUS_RULE.csv`. (Shared-ridge had dropped this to 0.52/0.56/0.30.)
- Continuum: spearman(d, mean transfer level) = **−0.80** (n=9): the flip tracks the
  appearance floor. LOVO direction selection: unanimous 9/9.

## T4. Absolute prediction (L + rule; mixed L = best point config)
`results_rule_v5core/summary_points_peak_pck.csv`; comparison in l_directional_check.py output

| regime | MAE (PCK) | Pearson r | reading |
|---|---|---|---|
| LOTO (new source, known benchmark — the deployment case) | **8.94** | **+0.874** | few-shot: anchors from sibling sources |
| LOBO (new benchmark) | 18.3–18.7 | +0.69–0.70 | coarse; symmetric IDW borrowing |
| JOINT (new source + new benchmark) | 29.2 | +0.01 | zero-shot: **rank-only** (+0.52 ρ!) |

"Anchors buy units, not order." Directionality does NOT help L (symmetric IDW wins
borrowing: LOBO-L MAE 16.9 sym vs 18.9/24.1 directional; triangle ranking ≈ 0 in all
directions — triangle adds level smoothing only, fixes the ρ_L=−1 artifact).

## T5. Decision metrics (rule predictor)
`transfer_analysis_v5/results/selection_regret_rule.csv`, `pairwise_gap_rule.csv`

| split | median regret | mean regret | random | P(within 1 PCK) | P(best in top-3) |
|---|---|---|---|---|---|
| LOTO | **2.30** | 7.64 | 13.27 | 0.40 | 0.56 |
| LOBO | **3.36** | 8.70 | 13.27 | 0.37 | 0.59 |
| JOINT | **2.78** | 6.46 | 13.27 | 0.38 | 0.59 |
| appearance (best split) | 6.43 | 11.5 | 13.27 | 0.20 | 0.40 |

Gap-stratified pairwise accuracy (rule): >10 PCK pairs **0.79–0.82** vs cross-arch
retraining 0.77, same-arch 0.84; <2 PCK pairs ~0.52–0.65 (irreducible: same-arch
retraining itself only agrees 0.58–0.62 there).

## T6. Asymmetric vs symmetric (why FID/W2 made this look washed out)
`transfer_analysis_v5/results/asym_vs_sym.csv`

| | rule | precision | recall | sym | FID | W2 | MMD |
|---|---|---|---|---|---|---|---|
| mean (9 variants) | **+0.51** | +0.26 | +0.37 | +0.47 | +0.43 | +0.42 | +0.16 |
| glunet T/T | **+0.73** | −0.16 | +0.73 | +0.33 | +0.33 | +0.29 | +0.07 |
| glunet T/F | **+0.61** | −0.14 | +0.61 | +0.31 | +0.25 | +0.19 | −0.03 |

Symmetric metrics hedge the flip: decent pooled, −0.3 to −0.4 behind in flip-extreme
cells. Pooled directional means (+0.26/+0.37) < sym (+0.47) = the historical washout.

## T7. PRE-REGISTERED OUT-OF-SAMPLE TEST — kubric intervention grid (never seen by the law)
`transfer_analysis_v5/results/intervention_oos.csv` (5-seed-averaged distances)

FF arm (9 sources × 4 benchmarks; law predicts precision wins):
| benchmark | precision | recall | sym |
|---|---|---|---|
| flyingthings | **+0.667** | −0.383 | −0.283 |
| kitti2012 | **+0.917** | +0.300 | +0.883 |
| kitti2015 | **+0.383** | −0.700 | −0.633 |
| middlebury | **+0.517** | +0.283 | +0.417 |
| **MEAN** | **+0.621** | **−0.125** | +0.096 |

**Precision wins on ALL FOUR benchmarks; wrong direction negative on average; PASS.**
TT arm: n=3, orderings coincide → non-discriminative. Sampling protocol: 5-seed-averaged
distances required for these near-tied sources (single-seed rank-corr 0.49 → stable).

## T8. The intervention grid itself (peak PCK@0.05, 50 epochs, finished 2026-06-09)
`/mnt/nvme_1tb_a/snapshots/transfer_grid/*/validation_results.csv`

From-scratch (FF) arm:
| source | FT | K2012 | K2015 | MB |
|---|---|---|---|---|
| trial19 (frozen camera) | **56.2** | 90.9 | **87.2** | 52.0 |
| lowtex_matte | 52.1 | 90.4 | 86.0 | 53.5 |
| kitti_badmotion_ft_gso_hq | 54.4 | 89.5 | 84.2 | 53.1 |
| kitti_recovered_gso_hq | 48.9 | 96.5 | 83.2 | 52.9 |
| kitti_recovered_hq | 47.6 | 96.2 | 80.2 | 51.3 |
| kitti_recovered_matte | 43.5 | 88.6 | 76.6 | 45.9 |
| kitti_recovered_gso_matte | 43.4 | 91.1 | 73.8 | 48.2 |
| ft_recovered_hq | 52.0 | 88.4 | 83.2 | **53.6** |
| ft_recovered_matte | 41.4 | 76.7 | 72.5 | 50.5 |
| synthetic_fractal_trial76 (TPE) | — | — | **94.4** | — |

Pretrained-frozen (TT) arm: gso_hq 96.6 / gso_matte 93.8 / badmotion 94.3 (K2015);
trial19 TT (June-1 run) 96.1. **trial19 case: FF rank 1/9 → TT rank 2/4** — its
frozen-camera profile (max precision, worst-in-family recall b→a 0.0105 vs ~0.0065)
predicts exactly this regime-dependence.

## T9. RAFT coding correction (2026-06-10)
RAFT trained **fully from scratch**: `synthetic_raft.yaml` has `pretrained_path: null`,
`paths.pretrained: null`; `models/RAFT/raft_wrapper.py` loads weights only from that
path; encoder is RAFT's own BasicEncoder (no ResNet, no ImageNet). The
`pretrained=True` in the transfer table came from the harness summary template
(CATs++ defaults). Action for paper: footnote the coding; RAFT joins the scratch
group → permutation p improves to 0.0079 (T1).

## Figures (also in Obsidian Attachments)
- F2 direction-preference bars: `scripts/transfer_analysis_v5/results/figures/F2_direction_preference.png`
- F4 gap-stratified curves: `.../F4_gap_stratified.png`
- F5 absolute scatters: `.../F5_absolute_scatter.png`
