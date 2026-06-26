# Renderer-agnostic harness (SDF-fractal: default vs tuned/trial76)

End-to-end harness to (re)train the 6 cells of the renderer-agnostic test and
evaluate them at strict / multi-alpha / multi-stride settings. Built because the
original default + FlowFormer checkpoints were stripped to `validation_results.csv`,
so the table can't be re-scored from disk — it must be retrained.

## The 6 cells
`{cats, glunet, flowformer} x {default, tuned}`, all pretrained-backbone + frozen (TT).

The **only** difference between default and tuned is `dataset.geometry_config_overrides`:
- **tuned**  = the trial76 TPE sampler block (KITTI coverage-search winner)
- **default** = `null` (generator defaults from `OnlineGeometryConfig.yaml`)

Each config is derived from that architecture's existing trial76 run config, so model /
lr / scheduler are identical across the pair — a clean controlled comparison.

## Files
- `gen_configs.py` — emit `configs/<arch>_<source>.yaml` for all 6 cells.
- `score_transfer_cell.py` — post-hoc eval of one trained cell:
  - KITTI-2012/2015 @ α ∈ {0.05, 0.03, 0.01} (single pass; extra α on same predictions)
  - TSS @ α ∈ {0.10, 0.05, 0.03}  (`tss_a10/a05/a03`)
  - TAP-Vid-DAVIS @ stride ∈ {1,2,4,8,16} (`tapvid_davis_s{N}`, α 0.05)
  - peak PCK over epochs per (benchmark, alpha/stride) → `transfer_cell_eval.csv`
- `build_table.py` — assemble `tables/tab_ra_{kitti,tss,tapvid}.tex` (default/tuned/Δ).
- `run_table.sh` — driver: gen → train → eval → tables.

## Run
```bash
# dry-run (prints the plan, launches nothing):
renderer_agnostic_harness/run_table.sh

# actually train + eval all 6 cells, sequential, on GPU 0:
GO=1 GPU=0 renderer_agnostic_harness/run_table.sh

# subset / split across GPUs:
GO=1 GPU=0 CELLS="cats_default cats_tuned glunet_default" renderer_agnostic_harness/run_table.sh
GO=1 GPU=1 CELLS="glunet_tuned flowformer_default flowformer_tuned" renderer_agnostic_harness/run_table.sh

# eval already-trained dirs only (skip training):
GO=1 EVAL_ONLY=1 renderer_agnostic_harness/run_table.sh
```
Outputs land in `$RA_OUT` (default `/mnt/nvme_1tb_a/renderer_agnostic/<cell>_<timestamp>/`).

## Training recipe (edit in gen_configs.py before launching)
- 50 epochs, validate every epoch, 100 steps/epoch, constant LR within the window
  (matches the CATs trial76 run that produced the published @0.05 table).
- `save_epoch_checkpoints: true` → per-epoch `epoch_*.pth` for peak-over-epochs.
- Per-arch batch: cats/glunet `bs=8`; **flowformer `bs=1` + `accum=8`** (512² won't fit
  `bs≥2` on a 24 GB 3090; effective batch stays 8).
- Cost is non-trivial — FlowFormer especially. Run subsets per GPU.

## Fidelity
`score_transfer_cell.py` reuses the exact training-time `validate_epoch_multi_benchmark`
+ `flow2kps` + `classify_prd`. The KITTI multi-alpha path is verified against logged
numbers (CATs tuned ep33 K2015@0.05 = 98.08 = logged 98.0818).

## Scope
RAFT excluded (no pretrained backbone; no trial76 SDF source). Middlebury excluded
everywhere (eval bug).
