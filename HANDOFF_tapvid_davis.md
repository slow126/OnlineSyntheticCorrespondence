# Handoff: Add TAP-Vid-DAVIS as a real-motion benchmark

**Author:** prior agent (codebase mapped 2026-06-24)
**For:** an implementing agent picking this up cold
**Goal:** Evaluate the existing trained correspondence models on **TAP-Vid-DAVIS** and
compute its **motion-coverage (BFV) descriptor**, so the ACCV paper has a *second
real-world dense/sparse-motion target besides KITTI* (reviewer concern "R1": the
real-motion law currently rests almost entirely on KITTI driving scenes).

> **Read this first, then verify everything against the real `pointodyssey` code paths.**
> Parts of the code snippets below were reconstructed by exploration agents and may have
> wrong field names. **PointOdyssey is your Rosetta Stone**: it is sparse, real-derived
> motion, and is *already wired into both the eval harness and the coverage pipeline*.
> The safest implementation strategy is: **find every place `pointodyssey` appears and add a
> `tapvid_davis` sibling next to it.** `grep -rin "pointodyssey" src/ scripts/ models/ slurm/`.

---

## 0. What TAP-Vid-DAVIS is (and is not)

- **Benchmark only — NOT a training source.** TAP-Vid (Doersch et al., NeurIPS 2022).
  TAP-Vid-DAVIS = 30 real DAVIS videos with *manually annotated* ground-truth point
  tracks. We only *evaluate* on it; we do **not** retrain any source model.
- Real images, real motion, diverse **non-driving** scenes (animals, people, sports,
  objects) — this is precisely why it answers the "KITTI is your only real motion" critique.
- **Sparse** ground truth (point tracks with occlusion flags), so it goes down the
  **sparse-keypoint eval path** (the one SPair / PF-Pascal / **PointOdyssey** use), not the
  dense-flow path (KITTI/FlyingThings).
- Distributed as a pickle. Each record ≈
  `{'video': [S,H,W,3] uint8, 'points': [N,S,2] normalized (x,y) in [0,1], 'occluded': [N,S] bool}`.
  Download: https://github.com/google-deepmind/tapnet (TAP-Vid data). **Verify the exact
  pickle schema after download** before trusting the snippet below.

There are **three** pieces of work, in dependency order:
1. **Eval adapter** — load TAP-Vid frame pairs as correspondence examples → PCK.
2. **Snapshot selection** — decide *which checkpoint* of each already-trained model to
   evaluate (the hard part you flagged).
3. **Coverage BFV** — compute TAP-Vid's motion descriptor so it joins the coverage tables.

---

## 1. Key design decisions (resolve these first)

### 1a. Frame-pair sampling / stride  ⚠️ affects both eval AND coverage
The models are *image-pair* correspondence; TAP-Vid is *long-range tracking*. You must
reduce videos to frame **pairs**:
- For a pair `(t1, t2)`: a point is a usable correspondence iff it is **visible (not
  occluded) in BOTH** frames. `src_kps = points@t1`, `trg_kps = points@t2`, both
  denormalized to pixels.
- **Stride is a hyperparameter.** Small stride (e.g. `Δ=1` or a few frames) keeps
  displacements inside the flow models' operating range; large stride is harder and more
  "transfer-relevant" but invites occlusion/large-motion failure that confounds the test.
  **Recommendation:** start with a small fixed stride (Δ=1), get the pipeline green, then
  optionally report a second stride. **Document the choice in the paper.**
- **Critical consistency rule:** the BFV/coverage descriptor (piece 3) must be built from
  the **same (t1,t2) pairs** used for eval, so the measured motion matches the evaluated
  motion. Use one shared pair-list.

### 1b. Snapshot / checkpoint selection  ⚠️ THE HARD PART
The existing "peak PCK" metric = `max PCK over training epochs`, computed from each run's
`validation_results.csv`, which only logs benchmarks that were in the eval set **during
training**. TAP-Vid was not. So you must evaluate *saved checkpoints* post-hoc.

**✅ DISK CHECK DONE (2026-06-24) — the answer is verified, don't re-derive it.**

The premise "ls `snapshots/*/...`" is **WRONG**: the repo-local snapshot dirs
(`snapshots/`, `cats_ff_ft_snapshtos/`, `glunet_fullgrid_snapshots/`, `ladder_snapshots/`,
`snapshots_mixed/`, …) have been **stripped to `validation_results.csv` + `training_summary.txt`**
— **0 weight files** for the whole CATs++/GLU-Net/ladder/mixed grids. Do not look there for weights.

**The actual weights live OFF-REPO on the NVMe mounts** (confirmed):
- `/mnt/nvme_1tb_a/cats_lolr_ckpts/` — the **CATs++ FF/FT source grid**: 22 dirs = 11 sources × {FF,FT},
  named `movi_f_FF/`, `pointodyssey_FT/`, `synthetic_large_zoom_FF/`, etc. **Each dir has only
  `model_best.pth`** (no epoch sweep, no per-benchmark bests).
- `/mnt/nvme_1tb_a/snapshots/` — 55 dirs, 1129 weight files: the **in-design grids**
  `transfer_grid/` (27), `transfer_grid_glunet/` (28), `flowformer_{movif,kittirecov}_{ff,tt}_b8/`,
  `catspp_movif_b8/`, `tssgrid_*`, etc. Each in-design run dir holds
  `{model_best.pth, last.pth, ONE epoch=N-step=N.ckpt, per-benchmark <bench>_best.pth}`.
- `/mnt/nvme_1tb_b/snapshots_synth_2d/` (336), `snapshots_synthetic_long/` (288),
  `snapshots_ptody_fix/` (240), etc. — older synthetic-source runs (full `epoch_N.pth` sweeps survive here).

**KEY CONSTRAINT — no uniform peak-sweep is possible.** A full per-epoch `epoch_N.pth` sweep
survives only for some *older* runs; the **CATs++ FF/FT grid has ONLY `model_best.pth`**, and the
newer grid runs keep only the single *last* Lightning `epoch=N-step=N.ckpt` (not every epoch).
So "evaluate every checkpoint, take max" is **physically impossible** across all 76 → drop that row.

**The only rule achievable uniformly across all 76 is `model_best.pth`.** Use it.

| Situation | Achievable? | Approach |
|---|---|---|
| ~~All epoch_*.pth survive~~ | **NO** — CATs grid has only model_best; new grids keep only last.ckpt | ~~peak sweep~~ — impossible, drop |
| **model_best.pth** (uniform) | **YES, everywhere** | Eval each run's `model_best.pth` (best *avg* PCK over its original benchmarks) on TAP-Vid. One eval/run. **This is the rule to use.** |
| **Proxy-peak via pre-saved per-bench best** | **PARTIAL** | Some runs already saved `<bench>_best.pth` (e.g. `flyingthings_best.pth`, `kitti2015_best.pth`); grab that file directly (no sweep needed). BUT availability is heterogeneous — `pointodyssey_best.pth` exists in older spair/2d_warp baselines, **not** the transfer_grid runs. Not uniform → use only as a robustness check, not the headline. |

**Framing tip (turn the constraint into a virtue):** `model_best.pth` is selected *without ever
looking at TAP-Vid* → TAP-Vid becomes a **true held-out test with no target-peeking**, which
pre-empts the "peak PCK is test-set selection" critique. State in the paper: "TAP-Vid is evaluated
at each model's avg-PCK-best checkpoint, selected on the *original* benchmarks — a stricter, no-peek
evaluation." Apply this one rule uniformly; do not mix.

**Practical note:** point the eval at the off-repo mount paths above, not the repo's stripped
`snapshots/` dirs. The repo dir name in `validation_results.csv` may not match the mount dir name
(repo uses `{dataset}_steps{N}_pretrained{T/F}_freeze{T/F}_{ts}/`; the mount uses short names like
`movi_f_FF/`, `transfer_grid/deplete_d05_pt0_fz0_.../`) — build an explicit run→checkpoint map.

### 1c. Stratum placement
TAP-Vid is **real-motion but sparse**. Put it in the **real-motion** stratum (broadening it
beyond KITTI), tagged as sparse GT. Check how the stratification is defined
(`tab:study` / the analysis scripts group `real-motion` vs `semantic`) and add `tapvid_davis`
to the real-motion list.

---

## 2. Piece 1 — Eval adapter (mirror PointOdyssey)

**Contract** (`src/data/synth/common/common_sample.py`, `CommonSample` dataclass). For a
sparse benchmark, `__getitem__` returns a `CommonSample` with:
- `src_img`, `trg_img` — `[3,H,W]`
- `src_kps`, `trg_kps` — `[2, N]` pixel coords (x,y) of co-visible points
- `n_pts` — int
- leave `flow_full` / `pckthres` **None** — the collate pipeline
  (`src/data/synth/collate_pipeline.py` → `flow_from_kps` in
  `src/data/synth/datasets/flow_utils.py`) builds dense flow + PCK threshold automatically.

**Files to touch (verify against the real PointOdyssey adapter, do not trust verbatim):**
- New loader: `src/data/synth/datasets/TapVidDavisDataset.py` — load pickle, build the
  shared (t1,t2) pair-list (1a), return the dict PointOdyssey's loader returns.
- Adapter + registration: `src/data/synth/adapters.py`
  - `PointOdysseyAdapter` (~lines 85-94) is the template.
  - Add `TapVidDavisAdapter`, then register `"tapvid_davis": TapVidDavisAdapter` in
    `ADAPTER_REGISTRY` (~line 537) and handle any special-casing in `build_adapter` (~564).

**Eval/PCK path (should need no change):**
`models/CATs_PlusPlus/utils_training/optimize_multi.py::validate_epoch_multi_benchmark`
runs `pred = net(trg_img, src_img)` → `flow2kps(trg_kps, pred, n_pts)` →
`eval_instance.py::eval_kps_transfer` computes PCK = % of points within `alpha * pckthres`.
**Get the src/trg ordering exactly right by copying PointOdyssey's convention.** Use
`alpha=0.05` to match the other sparse benchmarks (confirm against the configs).

---

## 3. Piece 2 — Wire into the eval run / results table

- Add `tapvid_davis` to the `eval_benchmarks` list in the relevant eval/train config
  (e.g. under `slurm/experiment_configs/`). **Inspect a working config that already lists
  `pointodyssey` and copy its exact keys** (the exploration agent guessed
  `eval_alphas` / `split_to_use_for_validation` — verify, don't assume).
- Add the eval split to `EVAL_SPLITS` in
  `scripts/transfer_analysis_v3/build_table.py` (~lines 337-341):
  `"tapvid_davis": "test",`.
- Results flow: per-run eval → `validation_results.csv` → `compute_curve_stats` (peak) in
  `scripts/build_leakage_free_eval.py` → `auc_results.csv` → joined on
  `(train_dataset, benchmark)` in `build_table.py` → `transfer_table.csv`. A new
  `benchmark="tapvid_davis"` row per (source, config) appears automatically **iff** that
  run logged a TAP-Vid PCK (see snapshot decision 1b).

---

## 4. Piece 3 — Coverage BFV for the new target

The coverage pipeline already has a **sparse path** (SPair keypoints → flow → BFV), so this
mirrors PointOdyssey/SPair:

- BFV builder: `scripts/coverage/spaces.py::normalize_flow_vectors` — `[x,y,dx,dy]`,
  positions `2*c/dim - 1`, flow `2*disp/dim` (no offset). Vectors are extracted from
  `batch['flow']` by `src/coreset/validation.py::extract_flow_vectors_from_batch`.
- Register `tapvid_davis` (split `test`, `representation: flow`, `is_eval: true`) in the
  coverage config `src/configs/coverage_configs/coverage_faiss_flow_full_v2.yaml`
  (mirror the `spair` / `pointodyssey` entries).
- Run `scripts/calculate_coverage_faiss_v2.py`; it extracts vectors → caches
  (`/mnt/nvme_1tb_b/coverage_vectors/tapvid_davis_test_flow.npy`) → self-radius →
  directed distances `d_{B→T}` / `d_{T→B}` (`scripts/coverage/faiss_ops.py`) →
  `analysis/coverage_v2_flow_full.csv`.
- **Consistency:** build these flow vectors from the **same (t1,t2) pairs** as the eval
  (decision 1a). The simplest route is to feed the eval loader's pairs straight into the
  coverage extractor so motion and PCK refer to identical correspondences.

---

## 5. Suggested order of work (with a smoke test before scale)

1. **Acquire data**; verify pickle schema; write `TapVidDavisDataset` with a fixed small
   stride; unit-check shapes (a pair returns sane `src_kps/trg_kps/n_pts`, points inside the
   image, displacements non-degenerate).
2. **Adapter + registration**; load one batch through the collate pipeline; confirm
   `flow_from_kps` produces a finite flow at keypoints.
3. **Smoke eval:** run ONE already-trained checkpoint (pick any `model_best.pth`) on
   TAP-Vid; confirm PCK is a plausible number (not 0, not 100). Sanity-check the src/trg
   ordering by flipping it and seeing PCK collapse.
4. **Resolve snapshot decision (1b)** based on what's on disk; script the chosen selection
   over all 76 runs.
5. **Coverage BFV:** register in coverage config, run, eyeball the splat / distance — a
   diverse real set should NOT look like KITTI's forward-dolly fingerprint.
6. **Full eval sweep** → results table → coverage join. Confirm a `tapvid_davis` column
   appears in `transfer_table.csv` with coverage features attached.

---

## 6. Risks / things to verify (do not trust blindly)

- **Field names / yaml keys** in §2–§4 were partly reconstructed by exploration agents.
  Verify every `CommonSample` field and config key against the **real** `pointodyssey`
  adapter and a **working** eval config. PointOdyssey is the ground truth.
- **src/trg convention** in `net(trg_img, src_img)` + `flow2kps(trg_kps, ...)` — copy
  PointOdyssey exactly; getting it backwards silently halves PCK.
- **Occlusion handling**: evaluate co-visible points only; an occluded point at t2 is not a
  valid correspondence.
- **Coordinate convention**: TAP-Vid points are normalized [0,1]; confirm (x,y) vs (y,x)
  order before denormalizing.
- **Snapshot consistency**: whichever selection rule (peak / model_best / proxy) you pick,
  apply it uniformly and state it in the paper. Mixing rules across benchmarks is a
  reviewer trap.
- **Coverage pair-consistency**: BFV must come from the same pairs as the eval, or the
  coverage score won't describe the motion you actually tested.

## 7. File map (from codebase exploration)

| Purpose | File |
|---|---|
| Sample contract | `src/data/synth/common/common_sample.py` (`CommonSample`) |
| Adapter template + registry | `src/data/synth/adapters.py` (`PointOdysseyAdapter`, `ADAPTER_REGISTRY` ~537, `build_adapter` ~564) |
| Collate / kps→flow | `src/data/synth/collate_pipeline.py`, `src/data/synth/datasets/flow_utils.py` (`flow_from_kps`, `downsample_flow`) |
| Eval / PCK | `models/CATs_PlusPlus/utils_training/optimize_multi.py` (`validate_epoch_multi_benchmark`), `.../eval_instance.py` (`eval_kps_transfer`, `classify_prd`) |
| Checkpoints | `snapshots/{run}/epoch_*.pth`, `model_best.pth`; selection logic `train_cats_unified.py` (best-avg), peak in `scripts/build_leakage_free_eval.py` (`compute_curve_stats`) |
| Results table | `scripts/build_leakage_free_eval.py` → `scripts/transfer_analysis_v3/build_table.py` (`EVAL_SPLITS` ~337) → `transfer_table.csv` |
| Coverage BFV | `scripts/coverage/spaces.py` (`normalize_flow_vectors`), `src/coreset/validation.py` (`extract_flow_vectors_from_batch`), `scripts/coverage/faiss_ops.py` (`compute_directed_distances`, `compute_self_radius`), config `src/configs/coverage_configs/coverage_faiss_flow_full_v2.yaml`, runner `scripts/calculate_coverage_faiss_v2.py` |
