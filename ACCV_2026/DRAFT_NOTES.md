# ACCV 2026 draft — provenance & regeneration notes (2026-06-10, overnight draft)

> ✅ **CURRENT STATUS — 2026-06-13 (~18:50 UTC). CATs++-FF "anomaly" RESOLVED; two follow-up experiments in flight.**
>
> **RESOLVED — it was never a rogue config; it's a TWO-CAMP structure.** The clean
> CATs++ FF/FT pure re-run finished. Regime-agreement test (new FF/FT vs table TT/TF,
> Spearman of per-benchmark source ranking): **scratch {FF,FT} agree ρ≈0.90; pretrained
> {TF,TT} agree ρ≈0.78–0.86; cross-camp only ρ≈0.07–0.54.** Deconfounded — the same
> split appears WITHIN the table alone (one pipeline), and new-FF vs old-FF ρ=0.76 (so
> not a pipeline artifact, not undertraining). Reading: **CATs++-FF agrees with its
> scratch sibling FT, not the pretrained consensus.** Pretrained regimes rank sources by
> motion COVERAGE (the Round-12 law); scratch regimes rank by source LEARNABILITY — a
> second coherent consensus. This REFINES (not contradicts) Round 12: coverage is the
> *pretrained-camp* law. [memory: regime_agreement_two_camps]
>
> **Rank-stability:** FF source ranking locks by ep30–40 (pooled ρ≈0.9) → the anomaly
> is NOT an undertraining artifact. BUT FF barely beats a FROZEN-RANDOM encoder (FT) in
> absolute PCK (movi_f FF 45.85 vs FT 45.74; avg gap ~+2) → the scratch encoder at the
> DEFAULT `lr_backbone=3e-6` (a fine-tuning LR, the only knob unchanged from stock CATs++)
> is ≈noise. [memory: catspp_ff_undertraining_rankstable]
>
> **IN FLIGHT #1 (RC) — high-LR re-run.** `catspp_ff_ft_hilr_rc.yaml`: lr_backbone 3e-4
> (100×), 200 ep, step `[140,160,180]`, 22 jobs (`--exclude=cs-1-2`). Tests whether a
> properly-trained FF encoder DIVERGES from FT (and maybe joins the pretrained/coverage
> camp). 1st attempt died disk-full; re-launched 2026-06-13 after archiving (338G free).
> DECISION pending: FF pulls away from FT → encoder was undertrained (maybe collapses to
> one law); FF still ≈ FT → scratch CATs++ is encoder-bottlenecked in this compute budget.
> Baseline (3e-6/50ep) curves preserved: `snapshots_archive/undertrained_ff_ft_baseline_2026_06_13` (RC).
>
> **IN FLIGHT #2 (LOCAL) — motion×appearance intervention heatmap.** On 11 canonical
> sources, transfer is driven by MOTION distance (β=−0.73, partial r=−0.80), NOT
> appearance (β=+0.05); axes separable (r=−0.28). Fig:
> `scripts/transfer_analysis_v5/results/motion_vs_appearance_heatmap.png`. Now extracting
> BFV+DINO vectors for 11 materialized kubric interventions (recovered / badmotion /
> deplete_d05–15 × hq/matte ±gso, + 2 flyingthings; `poison_v*` skipped = broken 67-scene
> renders) to build the CONTROLLED-grid version. Configs:
> `coverage_faiss_{flow,dino}_interventions.yaml`; DINO PCA loaded from cache (consistent
> with existing vectors). Flow ≈done, DINO ≈5h. Idea: at MATCHED total distance, does
> shifting toward motion beat appearance (iso-transfer contour tilt)?
>
> **FlowFormer:** grid finished (walltime-truncated, 55/56 cells), downloaded to
> `scripts/transfer_analysis_v5/flowformer_rc_results/` (1 hole: synthetic_2d_warp/TT).
> No crashes. TODO: fold peak-PCK rows into the analysis / stratified Table 1 (shares the
> synthetic-source distances).
>
> **Infra lessons:** (1) TF (via kubric/TFDS) + PyTorch teardown segfault dumps a ~20GB
> core on SUCCESSFUL exit (exit 134, `free(): invalid pointer`) — `ulimit -c 0` is now in
> the SLURM job template. (2) cs-1-2 has a bad GPU (~10 jobs killed) — exclude it.
> (3) disk-full kills CATs++ right after a validation (post-val checkpoint-save fails,
> `maxep=none`) — keep headroom; FlowFormer's 115GB of checkpoints were the main hog.
>
> **Still pending (unchanged):** §5 body prose (main.tex ~L437–488) still tells the OLD
> flip/two-cost story and must be rewritten to the Round-12 coverage law + the two-camp
> refinement above. Don't finalize until the hilr re-run lands.

> ⚠️⚠️ **READ THIS FIRST — NARRATIVE SUPERSEDED (2026-06-12, Round 12).**
> The current headline is **NOT** the "regime-direction flip" or the "two-cost
> model" (Rounds 5–11 below). Those are **superseded** and kept only as an
> audit trail. A rewrite should follow the **stratified / coverage** narrative
> in the next section and must not re-import the flip framing, the
> "scratch pays both costs / pretrained forgives off-target" story, or the
> anti-correlation-of-directions paragraph as live claims. Table 1 and Table 2
> in the paper were replaced on 2026-06-12 with the stratified versions; the
> body prose of §5 (main.tex ~L437–488) still tells the OLD story and is the
> main thing a rewrite must replace.

## Round 12 (2026-06-12): STRATIFIED REFRAME — coverage is the law; the flip was a pooling artifact

**One-line thesis.** BFV — a deliberately crude 4-D motion descriptor
(x, y, dx, dy) — is a good proxy for 3-D scene motion, and it enables fast,
motion-centric *dataset* selection through **recall / coverage** (d_{B→T}):
pick the synthetic source whose motion distribution *covers* the target's.
Because the descriptor is cheap and the signal is coverage, you can (i)
rediscover plausible motion re-parameterizations of a target by searching BFV
alone (construct validity), and (ii) **close the loop**: generate render-free
candidate datasets (geometry still required) and run much faster TPE searches
over dataset parameterizations, scored by BFV coverage instead of by training
a network.

**What changed and why (the honest story).**
- The old paper claimed a **regime flip**: from-scratch favors *precision*
  (off-target mass, d_{T→B}); pretrained favors *recall* (missing support,
  d_{B→T}). Round 6 dressed this up as a "two-cost" model.
- Stratifying the benchmarks into **real-motion** (KITTI-12/15, FlyingThings,
  PointOdyssey, synthetic) vs **semantic** (SPair, PF-PASCAL, PF-WILLOW, TSS)
  **dissolves the flip**: on real-motion targets, **recall is positive for
  every architecture × regime** (+0.29 … +0.73), including from scratch
  (GLU-Net +0.69, RAFT +0.44, FlowFormer +0.47; CATs++ the lone off-target
  leaner at +0.29 vs +0.54). The pooled "scratch prefers precision" was the
  **semantic benchmarks dragging scratch-recall down**, not a motion fact.
- **The single real exception** is **from-scratch on semantic targets**, where
  recall collapses (CATs++ −0.31; GLU-Net +0.07, RAFT +0.13, FlowFormer +0.08).
  Stated explanation = **floor saturation**: a from-scratch network never
  learns semantic features, so it sits near the benchmark floor and *no*
  property of the source's motion (coverage or otherwise) can move it. Under a
  pretrained backbone, recall on semantic targets returns (CATs++ +0.50,
  GLU-Net +0.80).
- **DINO (appearance) is the clean negative control** (Table 2): null-to-
  wrong-signed in every cell, especially wrong-signed on semantic targets.
  Whatever Table 1 captures lives in motion (and, see below, possibly in
  spatial sampling), not in what the frames look like.

**Tables 1 & 2 (paper) are now the stratified versions.**
- `ACCV_2026/tables/tab_law.tex` (motion) + `tab_law_dino.tex` (DINO),
  regenerated by `scripts/transfer_analysis_v5/make_stratified_law_tables.py`
  (writes straight into `ACCV_2026/tables/`; old ones backed up
  `*.bak_pre_strat_2026-06-12`). Rows = arch × regime (scratch=FF, pretrained=TT);
  cols = {d_{T→B}, d_{B→T}, sym, W2} × {real-motion | semantic}. The recall
  (d_{B→T}) column is **bolded** as "which part of motion." RAFT now correctly
  shows as scratch-only (no backbone column).
- FlowFormer rows use the partial RC pull at matched epoch (cap 150); only
  `synthetic` source is still missing and is expected to fall in line. Re-run
  the generator at cap 200 once FlowFormer converges.

**Still TODO for the rewrite (prose, not tables).**
- Rewrite §5 (main.tex ~L437–488): replace the flip/two-cost/anti-correlation
  argument with the coverage-is-universal + floor-saturation reading. Keep the
  defensible robustness machinery (estimator-invariance, leave-one-benchmark-
  out, self-pair exclusion, the DINO specificity control) but re-anchor it to
  coverage rather than to "the flip."
- Decide what survives of Sec 6/7 (policy, OOS grid, two-cost predictions) —
  the "select with the policy" slogan and Eq. (1)'s regime-conditioned form
  are downstream of the flip and need re-checking against the stratified view.

## OPEN PROBE (2026-06-12): is the SEMANTIC "precision" preference really MOTION, or just SPATIAL SAMPLING / DENSITY?

**The puzzle.** In stratified Table 1, semantic targets prefer the *precision*
metric d_{T→B} from scratch (CATs++ +0.51, GLU-Net +0.58, RAFT +0.50,
FlowFormer +0.20) while recall collapses. Taken at face value this says
"semantics want low off-target motion mass." But BFV is **4-D (x, y, dx, dy)**,
and `spaces.normalize_flow_vectors` maps **x, y → [−1, 1]** (range 2) while
scaling **dx, dy by only 2/W** (range ≈0.25 at KITTI motion magnitudes). So the
**joint nearest-neighbour distance is mechanically dominated by spatial
position**, and semantic benchmarks are **sparsely keypoint-labelled** → their
(x, y) marginal is a peculiar, object-centric point pattern.

**Hypothesis (Spencer, 2026-06-12).** The semantic "precision" signal may be a
**spatial-sampling artifact**, not motion. Mechanistically: for semantic
targets the from-scratch encoder must *learn aligned semantic representations*;
what helps it train is **where** in the image the source supplies
correspondence supervision — **spatial sampling alignment**, or even just
**sampling density** — not how well the source's *motion vectors* match. So
d_{T→B} may be ranking sources by spatial overlap with the sparse keypoint
layout, dressed up as a motion metric.

**Probe (running):** `scripts/transfer_analysis_v5/bfv_spatial_vs_flow_probe.py`
recomputes the two directed mean-NN distances in three sub-spaces of the *same*
BFV cloud — **full [x,y,dx,dy]**, **spatial-only [x,y]**, **flow-only [dx,dy]** —
and re-runs Table 1's stratified within-context Spearman cells per sub-space.
Reads → `scripts/transfer_analysis_v5/results/bfv_spatial_vs_flow_distances.csv`.
Decision rule: if on **semantic** targets the d_{T→B} signal is reproduced by
**xy-only** and **collapses under flow-only**, the semantic preference is
spatial; on **real-motion** targets we expect the opposite (flow carries it).

  RESULT (2026-06-12, N_sub=120k seed 0, pooled archs: FF=catspp/glunet/raft,
  TT=catspp/glunet; FlowFormer NOT in this probe; semantic stratum = 4 bench):

    regime     stratum |  dTB_full  dTB_xy  dTB_flow | dBT_full  dBT_xy  dBT_flow
    scratch real-motion |   +0.35   -0.31   +0.37   |   +0.66   +0.25   +0.18
    scratch    semantic |   +0.55   +0.02   +0.25   |   -0.17   +0.43   -0.20
    pretrnd real-motion |   +0.06   -0.10   +0.04   |   +0.70   +0.25   +0.26
    pretrnd    semantic |   -0.11   +0.38   -0.44   |   +0.61   +0.38   +0.53

  Three findings, and they REFINE (don't simply confirm) the hypothesis:

  (1) The semantic *precision* cell (dTB_full +0.55) is NOT a pure spatial
      artifact: **xy-only precision ≈ 0 (+0.02)**, flow-only +0.25, and the
      joint is larger than either. So "semantics prefer precision *because of*
      spatial sampling" is NOT supported in the d_{T→B} metric the table shows.

  (2) The spatial story instead lives in **RECALL**, and it is exactly the
      from-scratch-semantic floor cell: there **spatial coverage predicts
      (dBT_xy +0.43)** while **motion coverage is anti-predictive (dBT_flow
      −0.20)**, and the full-space recall is dragged negative (−0.17) — i.e.
      the "floor-saturation exception" to the coverage law decomposes into
      *motion coverage is unusable from scratch on semantic, only spatial
      sampling coverage helps.* This is Spencer's mechanism, measured.

  (3) The clean axis is **flow-coverage (dBT_flow)**: it is positive whenever
      the network can exploit motion — real-motion in BOTH regimes (+0.18
      scratch / +0.26 pretrained) and semantic ONLY under a pretrained backbone
      (+0.53), but NEGATIVE for semantic-from-scratch (−0.20). So motion
      coverage helps iff the encoder has (or is given) the features to use it;
      from scratch on semantic it cannot, and spatial alignment/density takes
      over. This is a cleaner claim than "one coverage law for all benchmarks":
      **real-motion transfer is predicted by motion coverage; semantic-from-
      scratch transfer is predicted by spatial sampling coverage; a pretrained
      backbone re-enables motion coverage on semantic targets too.**

  NOTE the table's d_{T→B}/d_{B→T} are full-4D, so the bolded "recall" column
  of paper Table 1 is full-space; the decomposition above is what's behind it.
  Density-proper features (point count, xy-entropy/support area) still untested
  (follow-ups below) — dBT_xy is a coverage proxy for density-alignment but not
  a pure count.

**Follow-up probes to add (sampling DENSITY angle):**
- Per-dataset **spatial density** features independent of flow: GT point count,
  spatial entropy / effective support area of the (x,y) marginal, and a
  density-matched directed distance. Test whether a pure density/coverage
  feature explains semantic transfer as well as d_{T→B}.
- **Density-control**: re-weight or resample the source BFV so its (x,y)
  marginal matches the benchmark's, then ask whether the residual *flow-only*
  distance still predicts. If not, semantic transfer is spatial all the way down.
- If confirmed, the paper should say semantic and real-motion transfer are
  predicted by **different facets of BFV** (spatial sampling vs motion
  coverage), which is a cleaner and more defensible claim than one law over all
  benchmarks — and explains *why* from-scratch-semantic breaks the coverage law.

**DENSITY EXPERIMENT RESULT (2026-06-12) — `bfv_density_experiment.py`.**
Ran the full density vs alignment vs matching experiment (source-only richness
features; statistical partial-correlation control; physical |flow|-magnitude
equalization). Pooled-arch within-source Spearman, per regime × stratum:

```
              | rich:flent rich:mov% rich:logN | M:dBTflow  M|rich  rich|M | M_eq   xy(sat)
SCRATCH  real |   -0.41    -0.37    +0.32      |  +0.18    +0.24   -0.51  | +0.19  +0.25
SCRATCH  sem  |   -0.38    -0.29    +0.38      |  -0.20    +0.14   -0.39  | -0.18  +0.43
PRETRN   real |   -0.16    -0.12    +0.33      |  +0.26    +0.26   -0.33  | +0.27  +0.25
PRETRN   sem  |   +0.22    +0.30    +0.18      |  +0.53    +0.59   -0.27  | +0.54  +0.38
```
Three conclusions, and the headline is a NEGATIVE result for the strict density
hypothesis:

(1) **"Denser/richer labels help from-scratch-semantic" is NOT supported as
    stated.** The motion-richness features (flow_entropy, moving_frac) are
    NEGATIVE on semantic-scratch (−0.38/−0.29) — and *equally* negative on
    real-motion-scratch (−0.41/−0.37). They are not a semantic-specific
    benefit; they track *unnatural/extreme* motion (the high-entropy sources
    are imagenet2dwarp / spair / synthetic_random_flipping), which hurts
    everywhere. The only positive source-only feature is raw count (logN,
    +0.32…+0.38) but it helps in *every* cell — a dataset-scale confound, not a
    from-scratch-semantic regularization effect.

(2) **Spatial alignment is a saturated non-signal.** dBT_xy ≈ 0.0024 for every
    dense source (image space is filled); the +0.43 semantic cell rides on the
    semantic-source outlier, not a graded axis. Not selectable.

(3) **The real structure is the motion-matching dissociation across regimes.**
    Motion coverage (M) predicts and *survives both controls* on real-motion
    (M +0.18, M|rich +0.24, M_eq +0.19) and on pretrained-semantic (M +0.53,
    M|rich +0.59, M_eq +0.54) — i.e. it is motion STRUCTURE, not amount
    (magnitude-equalization barely moves it). On **scratch-semantic it is a
    genuine floor**: raw −0.20, statistical-partial +0.14, physical-equalized
    −0.18 → indistinguishable from zero. Nothing source-side (motion, spatial,
    richness, count-beyond-scale) cleanly ranks sources there.

**CAVEAT / why the observational test can't fully settle it.** In Spencer's
actual source set, "label/motion density" is *entangled* with "motion
naturalness" — the densest-motion sources ARE the unnatural warps — so an
observational feature cannot isolate a pure density→gradient-richness effect.
A clean causal test needs a TRAINING ablation: take ONE natural source (e.g.
movi_f or flyingthings), vary supervised-label density (mask/subsample flow to
e.g. 100/50/25/10% of pixels, or vary moving-pixel fraction) holding motion
content fixed, train from scratch, and read semantic vs real-motion transfer.
If denser labels lift *semantic*-from-scratch but not real-motion, the
gradient-richness/regularization story is real and causal; otherwise the
floor is a floor. (Mirror of the existing precision/coverage ladders — a
"label-density ladder.")

**Net for the paper:** the from-scratch-semantic cell is best described as a
**floor where motion is unusable** (the encoder never learns semantic
features), NOT as "prefers spatial sampling / denser labels." The coverage law
is clean wherever motion is usable (all real-motion; semantic once a backbone
supplies features). Artifacts: `results/bfv_density_source_features.csv`,
`results/bfv_density_matched_distances.csv`, `results/bfv_spatial_vs_flow_distances.csv`.

## Build
```
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex supp_main.tex  (x2)
```
Main: 16 pp (review mode, deliberately long per Spencer's instruction).
Supp: 7 pp. Zero LaTeX errors, no undefined refs.
Template original backed up at `main_template_backup.tex`.

## Everything is regenerable — no hand-typed numbers in tables
- **Tables** (`tables/*.tex`): `python scripts/transfer_analysis_v5/make_paper_tables.py`
  renders all 12 from the artifact CSVs (same sources as the Obsidian docs).
  Main: tab_law, tab_predictors, tab_regime_linear, tab_oos, tab_utility.
  Supp: tab_supp_{asym,eps,oracles,gap,controls,featuresets,dino}.
- **Result figures** (`figures/results/`): `make_figures_v3.py` (F2–F6, F5supp,
  F5grid) + `make_intervention_summary.py` (F7); copied from
  `scripts/transfer_analysis_v5/results/figures/`.
- **Splats** (`figures/splats/`): reused from the ECCV submission
  (`ECCV_Reviews/eccv2026/figures/eccv26/section3_direction_panels_base_clustered_rerun/`)
  plus three NEW panels rendered tonight with the same code/style:
  MOVi-F, trial19, lowtex-matte —
  `python gaussian_splat/export_final_direction_panels.py
   --datasets movi_f_train trial19_train lowtex_matte_train --out-dir ...`
  (specs added to the DATASETS list in that script).
- All numbers exclude Middlebury (eval bug; documented in supp Sec. 5 with
  sensitivity analysis).

## Choices Spencer should review in the morning
1. **Title** — went with the regime-direction framing, not "Beyond Realism".
2. **Narrative** — law → rule → ceiling → absolute → interventions (not
   chronological). SDF-Fractal3D demoted from headline to instrument
   (addresses ECCV reviews P-001/P-002/S-004).
3. **Review-point coverage** — P-007/P-013 ("0.70 not useful") answered by the
   ceiling section + Table 5; P-012/S-015/S-016 (causal overreach) by the
   pre-registered test + explicit Limitations; P-009 (absolute vs ranking) by
   Sec. 6 "anchors buy units, the rule buys order"; S-022 (baselines) by
   Table 2's symmetric/learned/appearance rows.
4. **Prose numbers quoted in text** were taken from the current no-Middlebury
   artifacts; if any pipeline rerun changes them, grep main.tex for the
   number (tables auto-regen, prose doesn't).
5. Closed-loop section quotes KITTI +0.93→+0.87 and the FT null; the n=3
   seed-CI runs (act2_seeds) were still training — fold in when harvested.
6. Author block is the anonymous placeholder; ID needs the real submission
   number in `\usepackage[review,year=2026,ID=*****]{accv}`.

## Morning-feedback revision (2026-06-10, round 2)
- Table 1: flip-Δ column REMOVED; symmetric contrast columns (sym, FID) added;
  paired with Table 2 = identical analysis in DINO space (tab_law_dino.tex).
  GLU-Net F/F framed in caption: "the principle concerns which direction is
  BINDING, not that the other carries zero signal."
- Oracle claim reframed as PARITY, never ">100% / better than retraining";
  explains why a fraction slightly above 1 is an artifact of oracle noise.
- ρ≈0.5 → ρ²≈25% variance bound now properly explained in Sec. 6.
- Terminology: regimes are now "from scratch" vs "pretrained image-feature
  backbone" (ResNet/VGG/DINO); "fine-tuning" wording removed everywhere.
- Fig 1 = polished composite splat grid (make_splat_composite.py).
- Fig 2b = NEW direction-plane (ρ[dTB] vs ρ[dBT] per variant, regimes split
  across the y=x diagonal) — no derived statistic.
- Fig 4 (was confusing) relabeled in plain language.
- F5 color variants staged: F5_absolute_by_benchmark.png /
  F5_absolute_by_trainset.png (pick one; regime version currently in paper).
- Unrankable variant = RAFT-no-backbone config trained on ONE source
  (movi_f exploratory run) — incomplete, not buggy; now explained in Setup.

## ⚠ OPEN: trial19/lowtex flow caches are CORRUPTED
/mnt/nvme_1tb_b/coverage_vectors/kitti2015_hq_trial19_train_flow.npy and
kitti2015_lowtex_matte_train_flow.npy contain ONE constant vector repeated
(angle std 0.07°, single unique (dx,dy)). Real training data is fine (models
train to 87+ PCK). Other caches (movi_f, flyingthings) are healthy.
Consequences: trial19/lowtex splat panels invalid (pulled from paper);
any distance computed from these caches is suspect — the le-wm grid
distances' provenance must be checked / recomputed from real flow before the
"trial19 = worst-in-family coverage" measured claim is used (case study
currently softened to "by construction"). TODO: re-extract both caches from
the actual dataset dirs, recompute grid distances, re-render splats.

## Round 3 (2026-06-10, cache + RAFT + scatter)
- CACHE SCAN: scanned all 33 flow + 33 dino caches. ONLY 3 flow caches were
  the one-vector corruption: trial19, lowtex_matte, middlebury_val (all in
  coverage_vectors/). DINO all clean. The OOS intervention distances use a
  SEPARATE healthy cache (_cache_src_vectors/, all 9 sources fine) — so NO
  published number was ever wrong; only the trial19 splat. Deleted the 3 bad
  caches; restored trial19/lowtex from the clean _cache_src_vectors (raw
  flow). trial19 splat now shows correct radial camera-dolly; re-added Fig 7.
  middlebury_val_flow still needs re-extraction during the heavy grid rerun.
- RAFT FIX: movi_f RAFT was mislabeled F/F (stranded 1-source variant). RAFT
  ignores the pretrained flag (pretrained_path: null, no backbone), so F/F and
  T/F are the same model. Relabeled movi_f F/F -> T/F in both transfer tables
  (backups *_preraftfix.csv); RAFT now uses all 11 pure sources (rho
  0.40/0.30, still scratch-consistent). Reran no-mid pipeline + tables +
  figures. "Unrankable variant" language removed from paper.
- SCATTER: F5 is now HYBRID — dots colored by benchmark (shows LOTO clustering
  / LOBO striation), red+blue regime calibration lines on top. Regime-only
  version saved as F5_absolute_by_regime.png if wanted.

## Round 6 (2026-06-10 late): TWO-COST reframe (Spencer's hierarchy question)
Trigger: Spencer asked "is there a better way to combine precision+recall?"
Key discovery: the two directed distances are ANTI-correlated within contexts
on canonical sources (mean spearman -0.35, tight-vs-broad tradeoff). This
upgrades the mechanism from "which direction is more informative" to a
TWO-COST model: scratch pays BOTH costs (utility = dtb+dbt = the symmetric
average, now motivated as total cost, not a naive hedge); pretrained backbone
FORGIVES off-target mass (sole cost = dbt; anti-correlation explains why dtb
goes NEGATIVE pretrained and why sym collapses 0.39 vs 0.61). The model
PREDICTS the OOS grid result (coverage fixed -> scratch sum reduces to
precision; sym blind -0.01 vs +0.66) — formerly an awkward exception, now a
confirmation. New slogan: "select with the policy, design with the
direction."
- Supporting analyses (all persisted): conditional_combination.py (lex
  hierarchy FAILS pretrained +0.15; GBM/interaction/level-conditioning all
  <= 2-coef linear -> frontier is linear); loao_weight_transfer.py (scratch
  weights transfer ~1:1 across archs; pretrained fitted weights DO NOT
  transfer (n=2 archs), fit-free recall does); policy_vs_fit_regret.py.
- Eq. (1) is now the POLICY: D = dtb+dbt scratch / dbt pretrained. Policy
  tops tab_predictors (+0.52/+0.52/+0.50, green) — make_paper_tables.py FAMS
  updated. Ceiling: policy scratch frac 0.715, pretrained 1.085 (parity),
  pooled per-variant mean 0.549 vs inter-variant proxy ~0.53.
- CONSENSUS_POLICY.csv generated (n_boot 500): fractions 0.73/0.75/0.76.
  STALE PROSE FIXED: paper said "80--82%" but CONSENSUS_RULE.csv was
  regenerated today to 0.71-0.75 (README still quotes 0.79/0.80/0.81 from an
  older generation — update README or rerun with original settings).
- Table 4 (regime-linear) 7x/4x coefficient-ratio claims REMOVED (ratios are
  normalization-sensitive; scratch is ~1:1 under within-context scaling;
  sign pattern + single-direction flip are normalization-free).
- Limitations now carry THREE pre-registered FlowFormer predictions: scratch
  cells ranked by the sum, pretrained cells by recall alone, fitted weights
  transfer without improving. Abstract/intro/Secs 4-5-7/conclusion rewritten
  around the two-cost framing. 23pp, compiles clean.

## ⚠ LIVE ARTIFACT: intervention_oos.csv moves while the local grid trains
Regenerating intervention_oos_test.py on 2026-06-10 late gave slightly
different TT numbers than the morning (kitti2015 TT n 9->8; TT means now
prec +0.64 / rec +0.31 / sym +0.43, prose synced). FF arm unchanged
(+0.66/-0.26/-0.01, PASS). The script now also emits a `scratch_fit` column
(canonical-fitted scratch 2-coef model applied to the grid, fully OOS):
per-bench +0.18/+0.88/-0.25, MEAN +0.27 — between pure direction (+0.66)
and blind sum (-0.01), i.e. attenuates in proportion to its recall weight,
another two-cost confirmation (quoted in Sec 7 prose). RE-RUN the OOS script
+ grep prose numbers when the grid finishes/freezes.

## Round 11 (2026-06-10 late): Table 6 + spread columns; "coverage frozen" OVERCLAIM FIXED
Spencer: reviewers will see the control's -0.70 and call it garbage; show the
VARIATION next to each correlation. Checking the spreads exposed that the
inherited "grid holds coverage fixed" claim is only true vs FLYINGTHINGS
(d_TB spread 3.1x vs d_BT 1.3x); on KITTI it's INVERTED (d_TB near-tied
1.02-1.04x — precision-matched family by design, per the original
verification doc's caveat — d_BT 1.7-2.6x). Table 6 now has varies(max/min)
columns next to each rho; FlyingThings = the discriminative cell (bold);
KITTI rows labeled sign-consistent-not-load-bearing in caption. ALL
"coverage held fixed/frozen" phrasings replaced with defensible wording
(abstract, Sec 4 prediction, Sec 7 caption+prose, Limitations) — claim is
now "where the off-target cost carries the dynamic range, it alone ranks".
intervention_oos_test.py emits prec_spread/rec_spread (max/min) per row.
costs ~decoupled on grid (corr(dTB,dBT) -0.23/+0.25/-0.23 per benchmark vs
-0.35 natural). GLU-Net launcher pre-registration re-worded: prediction is
FlyingThings-specific (off-target-ranked there).

## Round 10 (2026-06-10 late): Table 6 FINAL FORM = minimal prediction/control
Spencer rejected the dose-response version too ("not how I understood it;
wall-of-text caption = bad table"). Root cause: the grid froze one cost, so
no table of grid numbers can showcase the two-cost story — scope must live
in the TITLE. Final form: "Pre-registered out-of-sample test of the
OFF-TARGET COST", 2 columns only (prediction: dtb, bold, positive every
benchmark / control: dbt, erratic), 3 benchmarks + mean, 5-line caption.
The dose-response gradient (+0.66 -> fit +0.27 -> sum -0.01) lives in ONE
prose sentence in Sec 7 (i), which was also cut ~in half. sym/scratch_fit
columns stay in intervention_oos.csv + the unused tab_dirtest.tex (supp
candidates). GLU-Net generality check referenced in (i) prose —
scripts/run_transfer_grid_glunet.py is the launcher (separate snapshot dir
transfer_grid_glunet/, glunet_cos recipe, FF_BATCH knob, pre-registration in
its docstring; dry-run verified, badmotion_gso_matte at 4890/4900 scenes).

## Round 9 (2026-06-10 late): Table 6 = DOSE-RESPONSE table
Spencer: "make table 6 good — 3x3 barely says anything; every table should
build the two-cost story." tab_oos rebuilt as a dose-response readout: 4
score columns ordered by missing-support weight (none = off-target alone /
small = canonical-fitted scratch model [new scratch_fit column in
intervention_oos.csv] / equal = summed cost, the policy's scratch arm / all
= missing support alone); mean row decays monotonically +0.66 -> +0.27 ->
-0.01 -> -0.26 exactly as the two-cost model predicts on a coverage-frozen
grid. kitti2012 = the directions-agree exception, pre-explained in caption.
Remaining "rule"->"policy" prose stragglers fixed in Secs 5-6. 24pp.

## Round 8 (2026-06-10 late): OOS direction-test table SIMPLIFIED
Spencer: the 4-col dirtest table (canonical+grid x 2 regimes, un-greyed,
composite rows) "shows almost noise — nothing works in that table." Agreed:
it was carrying 4 messages. REPLACED in main with tab_oos (per-benchmark FF
arm only, previously generated-but-unused): matched direction bold-positive
on every benchmark (+0.67/+0.92/+0.38, mean +0.66), sym blind on net (-0.01;
kitti2012 +0.88 cell pre-explained: directions agree there), mismatched
negative. Pretrained-arm numbers (+0.66/+0.28/+0.39, seed-noise band) +
policy-blind-on-grid (-0.01, by construction; select/design split) moved to
the Sec 7 (i) prose, fully quoted, NOT hidden; Limitations updated ("reported
undiscounted in Sec 7"). tab_dirtest.tex (the comprehensive 4-col version w/
composite rows) is still GENERATED but no longer \input — candidate for supp.
\label{tab:dirtest} kept on the new table so refs didn't move. 23pp.

## Round 7 (2026-06-10 late): ALL artifacts switched to the policy
Every table/figure that previously showed the rule now shows the policy:
- tab_predictors: policy row top (green +0.52/+0.52/+0.50); sym row renamed
  "Summed distance everywhere" (matches Eq. 1 language; same in tab_utility).
- tab_regime_linear: last column is now "policy rho (fit-free)" per regime
  (scratch=sum 0.50, pretrained=recall 0.61, pooled=policy 0.55); scratch row
  bolds BOTH coefficients (the two costs); fit bolded only if it beats
  fit-free by >0.02 (ties -> fit-free).
- F4 gap-stratified: motion_policy curves, retitled "The policy's errors...";
  policy now sits ON the cross-arch retraining line.
- F5 absolute scatter (+F5supp, F5grid): rebuilt from NEW
  results/benchsim_policy/ = scratch rows from a fresh
  context_scale_calibration --family motion_meannn_sym run (MUST use
  --table transfer_table_nomid.csv — first run used the default table,
  middlebury contaminated the LOBO/JOINT IDW anchors, MAE blew to 21/30 vs
  rule's... actually rule's LOBO/JOINT MAE is also 21/30 in this view; the
  contamination signature was only ~0.6 MAE) + pretrained rows from
  benchsim_rule; assembled by make_policy_rows.py --benchsim-* flags
  (normalize drops middlebury + raft|False|False strays). Panel stats now
  LOTO r .86/MAE 10.4, LOBO .64/21.8, JOINT .56/29.6 (≈ rule values).
- Prose: LOTO L+g for policy = MAE 9.3/r .88 (was rule 9.2/.88); fig:gap +
  fig:abs captions + Sec5/Sec6 wording policy-ified ("summed distance"
  replaces "symmetric average" where it names the policy arm).
- NOT rebuilt (rule == policy there or data-only): F2/F3 (signature), F6/F7
  (closed loop / interventions), tab_dirtest, tab_law, recovery/decomp.

## Round 5 (2026-06-10): claim-inversion reframe (Spencer's table audit)
Trigger: tables contradicted prose (sym best in 4/9 tab_law rows; rule's
aggregate edge over sym only +0.03-0.06; grid pretrained arm nominally favors
the WRONG direction; tab_regime_linear pretrained fit < rule).
- CLAIM HIERARCHY INVERTED: headline = motion-at-ceiling + appearance-null
  (no counterfactual columns anywhere); the flip = "which direction is MORE
  informative" (8/9, ninth=scratch GLU-Net direction-neutral), NOT "matched
  direction beats everything per cell"; symmetric = honest hedge whose cost
  is concentrated in pretrained cells (+0.39 vs +0.61; GLU-Net +0.26/0.33 vs
  +0.67/0.76).
- tab_law/tab_predictors/tab_regime_linear captions now state the
  counterevidence themselves (sym wins direction-agreeing rows; pooled
  margins modest; pretrained 2-coef fit < fit-free rule = fitting noise where
  one direction carries everything).
- tab_dirtest MOVED Sec4->Sec7 (it is OOS evidence, not observational
  adjudication); Sec7 restructured as 3 questions (causal? actionable?
  meaningful?); pretrained grid arm = "honestly unscoreable", and Limitations
  now DISCLOSES its unstable point estimates nominally favor the unmatched
  direction.
- NEW Table 6 (tab_utility): rows = selection strategies incl. NEW
  regime-aware policy (sym from scratch, d_B->T pretrained; built by
  make_policy_rows.py -> rows_*_motion_policy.csv, wired into run_v5.sh
  stage 6 + make_paper_tables.py). Policy median regret 1.2/2.1/2.8 vs rule
  3.9/4.7/3.9, sym-everywhere 3.1/3.1/2.7, random 13.9; per-regime split:
  scratch sym 3.9-4.1 < rule 5.2-7.9, pretrained rule 0.7-2.1 < sym ~3.0.
  Pairwise acc gap>10: policy 0.77-0.78 = cross-arch retraining 0.78
  (same-arch 0.84). Regret CSVs regenerated with 4 families.
- Abstract/contributions/limitations/conclusion rewritten to match. 22pp now.

## Round 4 (2026-06-10): BFV construct-validity added
- New main-text item in Sec 7 (interventions): "The recovered motion is
  physically meaningful (construct validity)" — Fig 7 (F8_bfv_recovery.png) +
  compact Table 7 (tab_recovery.tex). TPE search on BFV alone recovers
  camera-centric motion for KITTI, object-centric for FlyingThings.
- Middlebury DROPPED from this story (eval bug); KITTI+FT only. The
  camera/object ambiguity (was the Middlebury "interpretable miss") kept as a
  one-line caveat + full treatment in supp Sec 3.7 (tab_recovery_full.tex).
- Data: scripts/transfer_analysis_v5/recovered_theta.csv (the 2 recovered θ).
  Figure: scripts/transfer_analysis_v5/make_recovery_figure.py.
- Main now 19pp, supp 8pp (still over 14 limit — expected, trim later).
- This is "cut last" tier: validates BFV but not load-bearing for the
  regime-direction principle.
