# Code Map — Experiments

**Purpose:** one pointer index for all Project-2 experiment code. v4 was meant to hold everything but ballooned into post-processing + ablation + calibration + robustness families. This maps the canonical pipeline order, flags **[v3]/[v4]**, marks **CANONICAL / one-off / SUPERSEDED**, and notes input->output.

Paths relative to project root `~/Projects/OnlineSyntheticCorrespondence/`.
Planning mirror (with paper cross-links): `~/Documents/Obsidian/Correspondence/Project/Code Map - Experiments.md`.

> The 30-second mental model:
> **train runs -> AUC/PCK** -> **extract flow/DINO vectors** -> **pairwise distances** -> **build_table.csv** -> **experiments.py (predictor) -> results*/** -> **{strength, density, calibration, robustness} ablations** -> **compile_*/figures -> ABLATION_*.md / CLAIMS.md**. The intervention (camera search) lives in *separate repos*.

---

## Stage 0 — Upstream data (outside the analysis dirs)
The training + feature extraction that everything downstream consumes.

- **CANONICAL Training runs** -> snapshots `/mnt/nvme_1tb_*/snapshots/...` -> per-run `validation_results.csv` (PCK/AUC). Configs in `src/configs/`.
- **CANONICAL** `transfer_analysis_v3/compute_kubric_auc.py` **[v3]** — kubric/movi_f snapshots -> appends rows to `analysis/leakage_free_flow_kmeans_manifold/auc_results.csv`.
- **CANONICAL Flow/DINO coverage vectors** -> `/mnt/nvme_1tb_b/coverage_vectors` (per-dataset sampled descriptors; raw material for all distances).

**Key produced CSVs (the table's inputs):**
`analysis/leakage_free_flow_kmeans_manifold/auc_results.csv` · `analysis/coverage_v2_flow_only_raw_joint_full.csv` · `..._kmeans_full.csv` · `analysis_v3/coverage_dino_full.csv` · `analysis_v3/dino_null_coverage.csv`

---

## Stage 1 — Distances & features  (Paper 2 §4)  **[v3]**
Turn coverage vectors into source->target / benchmark<->benchmark distances.

- **CANONICAL** `compute_pairwise_self_distances.py` — **the metric source.** train-train, eval-eval, train-eval pairs -> `mean_nn`, coverage@{1,4,16}px, kNN-KL@{5,20}. In: `--vec-dir coverage_vectors`. Out: `analysis_v3/pairwise_self_distances.csv`. (`compute_pair_metrics` here is reused by the kubric oracle.)
- **CANONICAL** `compute_symmetric_distances.py` — symmetric distributional distances: `flow_fid`, sliced-W2. Out: `symmetric_distances.csv`.
- one-off `build_symmetric_self_distances.py` — additive merge of symmetric self-distances for IDW neighborhoods.
- one-off `merge_pairwise_distances.py` — merge per-rank shards from the parallel/cluster distance run; symmetrize.
- one-off `compute_dino_null_coverage.py` — null-calibrated DINO cosine coverage -> `dino_null_coverage.csv`.
- one-off `materialize_flow_raw_coverage_from_pairwise.py` — legacy raw-flow coverage CSV from pairwise.
- one-off `audit_flow_feature_coverage.py` — checks table pairs have cached features.
- SUPERSEDED `compute_feature_mi.py` — MI(feature; AUC) + redundancy matrix (exploratory).

**Cluster drivers:** `slurm_density_pair_shard_rc.sh`, `slurm_density_merge_level_rc.sh`, `slurm_dino_sym_rc.sh`, `run_pairwise_slurm.sh`.

---

## Stage 2 — Table assembly  (Paper 2 §4)  **[v3]**

- **CANONICAL** `build_table.py` — **joins AUC/PCK + all feature CSVs -> the modeling table.** Out: `transfer_analysis_v3/transfer_table.csv` (the single object every v4 script reads). Backups: `transfer_table.csv.bak_*`.
- one-off `patch_spair_long.py` **[v4]** — averages spair_long catspp peak_pck into spair rows; backs up `.pre_spair_long.csv`.

---

## Stage 3 — Core predictor  (Paper 2 §5, §8)  **[v4]**

- **CANONICAL** `experiments.py` — **THE main sweep.** g (within-context ridge) + L (cell band), multiple `--targets` (auc_normalized, peak_pck), all families, all CV regimes. Out: `results*/predictions/<target>/`, `summary.csv`. *This is the script v4 was supposed to "be."*
- **CANONICAL** `bootstrap.py` — entity-resampled CIs for ctx_rho / cent_rho / abs_r per target×head. Walks `predictions/<target>/`. **Has the FAMILIES list — extend when adding families.**
- **CANONICAL** `strength_tests.py` — ctx_rho_g point + bootstrap CI + P(rho_g>0) + shuffle null per family. -> `strength_per_family.csv`, `strength_paired_gaps.csv`.

**Top-level orchestrator:** `run_v4.sh`.

---

## Stage 4 — Ablation families  (Paper 2 §5–§8)  **[v4]**
Each is a re-run of Stage 3 over a variation, then a `compile_*` -> `ABLATION_*.md`.

### 4a. Strength / headline  (§5)
- **CANONICAL** `compile_ablation_summary.py` -> `ABLATION.md` + `ABLATION_strength.md` (scans `results*/summary.csv`).

### 4b. Density  (§6, Table 4b)
- **CANONICAL** `density_invariance.py` / `density_invariance_train_eval_only.py` — **feature-side** stability (rho vs N).
- **CANONICAL** `run_density_sweep.sh` / `run_density_sweep_lean.sh` — **fitted-side** (rerun predictor at dL1–dL5) -> `results_lean_dL{1..5}_mixed/`.
- **CANONICAL** `compile_density_ablation.py` -> `ABLATION_density.md` (+ `ABLATION_lean_density.md`).

### 4c. Calibration / dispersion  (§8 — benchsim)
- **CANONICAL** `context_scale_calibration.py` — the gain heads (`g_benchsim_gain`, `g_profilesim_gain`, …). Out: per-family `summary_all_variants.csv`. **Imports IDW utils from `../transfer_analysis_v3/triangle_prior_prototype.py`.**
- **CANONICAL** `compile_calibration_ablation.py` -> `ABLATION_calibration.md`.
- **CANONICAL** `calibrate.py` — leakage-clean L+g recalibration (the deattenuation harness would extend this).
- one-off `plot_context_scale_calibration.py` — plots the heads.
- one-off `residual_calibration_diagnostics.py` / `zscore_residual_diagnostics.py` — post-hoc residual diagnostics (read existing prediction rows).
- **CANONICAL** `residual_feature_search.py` — "does any descriptor explain the residual magnitude?" -> the **ceiling** result.

### 4d. Pure-only vs mixed mode
- **CANONICAL** results in `results_pure0_*` ; write-up `ABLATION_pure0_mixed_mode.md`.

### 4e. Feature subsets  (§6, Table 8)
- **CANONICAL** `results_fsub_*` (one dir per subset: mean_nn, coverage, eps_{1,4,16}px, kl, kl_k{5,20}, asym_only, sym). Compiled via `compile_ablation_summary.py`.

---

## Stage 5 — Robustness  (Paper 2 §6)  **[v4]**

- **CANONICAL** `run_robustness.sh` -> `results_robust_drop_sdf3d/` (drop-one-source / leave-one-generator-family-out).
- **CANONICAL** `compile_robustness.py` — robustness comparison table.
- **CANONICAL** `benchmark_stratification.py` — sparse vs dense per-benchmark rho_g (reads existing LOTO rows, no rerun).

---

## Stage 6 — Intervention (camera search)  (Paper 2 §7)  **[separate repos]**
- **CANONICAL** `transfer_analysis_v4/INTERVENTIONAL_STUDY.md` — the plan + predicted-vs-actual.
- **CANONICAL** `~/Projects/interventional-study/` — `search_loop.py`, `search_spaces.py` (TPE/predictor-guided search).
- **CANONICAL** `~/Projects/kubric/interface/` — generator + render (also Paper 3 infra).

---

## Stage 7 — Compile / figures / docs  **[v4]**
- **CANONICAL** `compile_v4.py` — `summary.csv` + `bootstrap_gap.csv` -> `results.md`.
- **CANONICAL** `figures.py` — `fig1_headline_bars`, `fig2_global_scatter`, … per target×split×family×head.
- **Living docs:** `STATUS.md` (living state) · `CLAIMS.md` (claims+evidence) · `README.md` · `HANDOFF.md` · `REVIEW_RESPONSE_ASSESSMENT.md` · `RC_CLUSTER_PLAN.md`.

---

## Appendix — v3 prototypes  (SUPERSEDED by `experiments.py`; keep for provenance)
Historical modeling prototypes; **don't run these for the paper**, but `triangle_prior_prototype.py` is still a live import for benchsim.
- SUPERSEDED `transfer_predictor_prototype.py` · `marginal_decomp_prototype.py` · `level_mechanism_sweep.py` · `run_experiments.py` (156k monolith) · `compile_results.py` (166k) · `run_few_shot_analysis.py` · `run_subsampling_stability.py` · `run_symmetric_stability.py`
- one-off `triangle_prior_prototype.py` — the failed triangle-prior test **and** the IDW utility module imported by `context_scale_calibration.py`. **Don't delete.**
- one-off `toy_lobo_loto_simpson.py` — synthetic illustration figure (no real data).
- **Big v3 orchestrators:** `run_pipeline.sh`, `run_fresh_features.sh`, `run_clean_flow_rerun.sh`.

---

## `results_*` directory decoder
| prefix | meaning |
|---|---|
| `results_mixed` / `results` | main run (mixed L-mode) |
| `results_pure0_*` | PURE_ONLY=0 (mixed-source) variants |
| `results_fsub_*` | single feature-subset runs |
| `results_lean_dL{1..5}_mixed` | density-sweep levels |
| `results_{density_idw,eb_shrunk,symmetric_*,targeted_*}` | L-mechanism / prior variants |
| `results_robust_drop_sdf3d` | robustness (drop-source) |
| `_archive_*` | snapshots of prior states — ignore |

> Cleanup candidates (post-draft): `_archive_*` dirs, `transfer_table.csv.bak_*`, and the SUPERSEDED v3 prototypes are the bulk of the "balloon." Safe to archive off-tree once the draft is frozen — except `triangle_prior_prototype.py` (live import) and `transfer_table.csv` (active table).
