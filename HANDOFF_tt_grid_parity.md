# Handoff: make the intervention-grid TT (pretrained) arm reach parity with FF

## TL;DR of the problem
The intervention grid's **FF arm** (from-scratch, `pt0_fz0`) has **10 sources**;
the **TT arm** (pretrained+frozen backbone, `pt1_fz1`) only surfaces **3** in the
harvested results (`/mnt/nvme_1tb_a/snapshots/transfer_grid/`). That asymmetry is
why trial19 reads "1/9 from scratch but 2/4 pretrained" — the TT arm is
under-populated, so the OOS pretrained test is non-discriminative (the paper
currently flags it as `n=3`).

**Root cause (verified):** the grid driver `scripts/run_transfer_grid.py` only
queued TT for the 3 new GSO sources. Its comment claims TT "already exists in
./snapshots" for the rest — and that is **literally true**: 6 of the missing TT
runs are fully trained (50 epochs, pretrained_backbone=True, freeze=True, with
`validation_results.csv`), they just live in `./snapshots/<source>/...` and were
never harvested into the `transfer_grid/` dir that the analysis reads. Only **1**
source is genuinely untrained on TT.

## The exact inventory (per FF source -> is there a TT run?)
| source | TT in transfer_grid? | TT trained in ./snapshots? | action |
|---|---|---|---|
| kitti_badmotion_ft_gso_hq | YES | — | none |
| kitti_recovered_gso_hq | YES | — | none |
| kitti_recovered_gso_matte | YES | — | none |
| ft_recovered_hq | no | `snapshots/ft_recovered_hq/kubric_ft_recovered_hq_2026_06_06_11_27` | **HARVEST** |
| ft_recovered_matte | no | `snapshots/ft_recovered_matte/kubric_ft_recovered_matte_2026_06_06_11_27` | **HARVEST** |
| kitti_recovered_hq | no | `snapshots/kitti_recovered_hq/kubric_kitti_recovered_hq_2026_06_07_12_07` | **HARVEST** |
| kitti_recovered_matte | no | `snapshots/kitti_recovered_matte/kubric_kitti_recovered_matte_2026_06_07_11_21` | **HARVEST** |
| trial19 | no | `snapshots/kubric_intervention_kitti2015_trial19_2026_06_01_14_04` | **HARVEST** (this is the source of the hardcoded 96.12) |
| synthetic_fractal_trial76 | no | `snapshots/sdf_kitti2015_trial76_widebnds_2026_05_29_14_10` | **HARVEST after VERIFY** (this dir is the TPE *search* snapshot; confirm its `config.yaml` is a real 50-epoch TT transfer run and its `validation_results.csv` covers the eval benchmarks, not just a search log) |
| lowtex_matte | no | — none — | **TRAIN** (the only genuine retrain) |

So: **6 harvest, 1 verify-then-harvest, 1 train.** No 7-run retrain needed.

## How the harvest works (so you know what "harvest" means here)
`scripts/transfer_analysis_v5/blocks.py` (and `intervention_oos_test.py`) build
the grid PCK table by iterating `/mnt/nvme_1tb_a/snapshots/transfer_grid/`,
reading each subdir's `validation_results.csv`, and keying:
- `src = dirname.rsplit("_pt",1)[0]`
- `arm = "TT" if "_pt1_fz1" in dirname else "FF" if "_pt0_fz0" in dirname else skip`
- `peak_pck = max over rows per benchmark`

There is also ONE hardcoded line in `blocks.py` (~line 213):
`rows.append(("trial19","TT","kitti2015",96.1158))  # June-1 T/T snapshot`
— remove this once trial19 TT is harvested properly (it currently only injects
the kitti2015 number, which is half the "4").

The harvest keys purely off directory NAME (`<source>_pt1_fz1*`), so the clean
fix is to make each old TT run visible under that naming.

## Recommended fix (do in this order)

### 1. Harvest the 6 (+1 verify) existing TT runs — no GPU
For each harvestable source, symlink its trained dir into the grid under the
expected `<source>_pt1_fz1` name (symlink, don't copy — checkpoints are large):

```bash
GRID=/mnt/nvme_1tb_a/snapshots/transfer_grid
ln -s "$PWD/snapshots/ft_recovered_hq/kubric_ft_recovered_hq_2026_06_06_11_27"      $GRID/ft_recovered_hq_pt1_fz1_harvested
ln -s "$PWD/snapshots/ft_recovered_matte/kubric_ft_recovered_matte_2026_06_06_11_27" $GRID/ft_recovered_matte_pt1_fz1_harvested
ln -s "$PWD/snapshots/kitti_recovered_hq/kubric_kitti_recovered_hq_2026_06_07_12_07" $GRID/kitti_recovered_hq_pt1_fz1_harvested
ln -s "$PWD/snapshots/kitti_recovered_matte/kubric_kitti_recovered_matte_2026_06_07_11_21" $GRID/kitti_recovered_matte_pt1_fz1_harvested
ln -s "$PWD/snapshots/kubric_intervention_kitti2015_trial19_2026_06_01_14_04"       $GRID/trial19_pt1_fz1_harvested
# trial76: VERIFY first, then:
# ln -s "$PWD/snapshots/sdf_kitti2015_trial76_widebnds_2026_05_29_14_10"            $GRID/synthetic_fractal_trial76_pt1_fz1_harvested
```
CRITICAL CHECKS before trusting each symlinked run:
- `validation_results.csv` exists and has rows for the right benchmarks
  (kitti2015, kitti2012, flyingthings — middlebury is excluded, eval-bugged).
- `config.yaml` has `model.pretrained_backbone: true` and `model.freeze: true`
  and `training.epochs: 50` (matches the 3 grid TT runs for apples-to-apples).
- The dataset `datapath` points at the SAME `*_5000` materialized set the FF
  run used (so motion is identical across arms).

### 2. Train the 1 missing cell (lowtex_matte TT)
Add `("lowtex_matte","TT")` to the `GRID` list in `scripts/run_transfer_grid.py`
(around line 100, the `GRID: list[tuple[str,str]]` block), then:
```bash
python scripts/run_transfer_grid.py --only lowtex_matte --variants TT      # dry run / config check
python scripts/run_transfer_grid.py --only lowtex_matte --variants TT --run --gpus 0
```
TT is frozen-backbone so it fits batch 8 (no OOM); ~50 epochs x 100 steps, fast.
NOTE: a long-running act2_seeds training may still be on GPU 0/1 — check
`nvidia-smi` and use a free GPU, or wait for it to finish.

### 3. Remove the hardcoded trial19 line + drop the corrupt-cache guard
In `scripts/transfer_analysis_v5/blocks.py`:
- delete the `rows.append(("trial19","TT","kitti2015",96.1158))` line (now
  harvested from the symlink).
Confirm the middlebury flow-cache is still excluded (it was deleted as corrupt;
see DRAFT_NOTES round 3).

### 4. Recompute distances for the new TT cells (if needed)
The OOS distances (`le-wm/outputs/intervention_motion_distances_directional.csv`)
are already computed for all sources from the clean `_cache_src_vectors/` — the TT
arm uses the SAME source distances as FF (distance is a property of the dataset,
not the model), so **no distance recompute is needed**. Only the PCK harvest
changes.

### 5. Regenerate + recompile
```bash
python scripts/transfer_analysis_v5/intervention_oos_test.py --out scripts/transfer_analysis_v5/results/intervention_oos.csv
python scripts/transfer_analysis_v5/make_paper_tables.py        # tab_oos.tex etc.
python scripts/transfer_analysis_v5/make_intervention_summary.py # F7 (trial19 panel c)
python scripts/transfer_analysis_v5/make_figures_v3.py
cp scripts/transfer_analysis_v5/results/figures/F7_interventions_summary.png ACCV_2026/figures/results/
cd ACCV_2026 && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
Then the TT arm has ~9-10 sources and the OOS pretrained test becomes
discriminative — update the paper's "TT arm n=3, non-discriminative" caveat
(Sec. 7 / Table tab_oos caption / supp Sec. 3.4) to report the real result.
trial19 should land mid-pack in TT (the principle predicts unremarkable
pretrained), turning "1/9 vs 2/4" into "1/N vs ~mid/N" with a real N.

## Watch-outs / gotchas
- The 6 harvestable runs were trained at DIFFERENT times with possibly slightly
  different harnesses than the 3 grid TT runs. Before claiming parity, sanity-
  check that all TT runs share epochs=50, steps_per_epoch=100, batch=8,
  pretrained_backbone+freeze, and the same eval benchmark list. If any differ,
  retrain that cell instead of harvesting.
- Do NOT re-introduce middlebury: its eval is bugged AND its flow cache was
  corrupt (deleted). The OOS test already excludes it.
- `synthetic_fractal_trial76` is the SDF online generator, not a kubric `*_5000`
  set — its TT run, if harvested, must come from a real transfer-training
  snapshot, not the TPE search dir. Verify carefully or retrain via the
  SYNTH_TEMPLATE path in run_transfer_grid.py.
- The FF "9 vs 10": the OOS test drops sources to whatever has both a distance
  row and a PCK; trial19's FF rank is "1/9" because one of the 10 FF sources
  (synthetic_fractal_trial76) is K2015-only / filtered on some benchmarks.
  After harvest, re-check the per-benchmark `n` in `intervention_oos.csv`.

## Key files
- driver: `scripts/run_transfer_grid.py` (GRID list ~L100, VARIANTS, build_config)
- harvest + hardcode: `scripts/transfer_analysis_v5/blocks.py` (~L205-213)
- OOS test: `scripts/transfer_analysis_v5/intervention_oos_test.py`
- grid dir: `/mnt/nvme_1tb_a/snapshots/transfer_grid/`
- old TT snapshots: `./snapshots/<source>/...`
- TT config template: `src/configs/lightning/transfer_grid/*_pt1_fz1.yaml`
