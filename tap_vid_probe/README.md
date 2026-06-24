# tap_vid_probe

Quarantined TAP-Vid-DAVIS work: a new **eval benchmark** wired into the correspondence
pipeline, plus a **MOVi-F training probe** to test whether TAP-Vid PCK peaks at the same
epoch as an existing benchmark (which would give us a **proxy** so we don't have to retrain
every model just to find its TAP-Vid peak).

## What this answers
TAP-Vid-DAVIS = 30 real, non-driving DAVIS videos with manual point tracks — a second
real-motion target besides KITTI (reviewer concern R1). It is **eval-only**; no source
model trains on it. The probe trains 4 architectures from a MOVi-F baseline while
evaluating TAP-Vid alongside kitti/tss/pf every epoch, so we can read off peak-epoch
co-occurrence.

## Layout
| File | Purpose |
|---|---|
| `preprocess_davis.py` | one-time: raw `tapvid_davis.pkl` → compact 512px mmap cache (`frames.npy`+`index.pkl`) |
| `tapvid_davis_dataset.py` | `TapVidDavisSimpleDataset` + `TapVidDavisAdapter` (mirror of PointOdyssey) |
| `test_dataset.py` | shape/collate/flow-direction sanity check (no model) |
| `smoke_eval.py` | run one trained checkpoint on TAP-Vid; PCK + flip diagnostic |
| `calibrate_stride.py` | pick a (stride, alpha) where PCK is discriminative (model ≫ identity) |
| `configs/*.yaml` | 4 full-convergence MOVi-F configs (catspp/flowformer/glunet **TF**, raft **FF**) |
| `run_probe.sh` | launch the 4 runs across 2 GPUs (2 lanes) |
| `monitor.py` | per-run, per-benchmark PCK curves + peak epoch |

## Data
- Source pickle: `/mnt/nvme_1tb_a/tapvid/tapvid_davis/tapvid_davis.pkl` (CC-BY, 30 videos)
- mmap cache (what training reads): `/mnt/nvme_1tb_a/tapvid/probe_cache/` (frames.npy ≈1.5 GB)
- Schema (verified): per video `points (N,S,2)` float32 norm[0,1] (x,y), `occluded (N,S)` bool,
  `video (S,H,W,3)` uint8. Frames resized to 512², points scaled `*512`.

## Integration into shared code (the only two non-quarantined edits)
1. `src/data/synth/adapters.py` — guarded block registers `tapvid_davis` → `TapVidDavisAdapter`
   (no-ops if this dir is removed).
2. `train_cats_unified.py::create_validation_datasets` — one `elif benchmark == 'tapvid_davis'`
   branch maps `tapvid_davis_root` + stride knobs to the adapter.

Eval routing needs nothing else: `EvaluatorInstance` sends every non-`caltech` benchmark to
the sparse-keypoint PCK path (`eval_kps_transfer`), same as PointOdyssey/SPair.

## Calibration finding (why stride=20, alpha=0.03)
At small stride the point motion is **below** the PCK threshold, so a zero-flow "identity"
prediction already scores ~67% → saturated, won't move during training. Sweep (CATs
movi_f_FT reference):

| stride | identity@.03 | model@.03 | gap |
|---|---|---|---|
| 5  | 47.9% | — | — (saturated) |
| 20 | **16.3%** | **41.6%** | **+25.3** |
| 30 | 12.3% | 33.2% | +20.8 |

stride=20 / alpha=0.03 gives the largest model-over-identity gap with a low floor and
~280 pairs / 3300 points — TAP-Vid tracks training instead of sitting saturated. The
trained model beats identity by +25 (it cannot if src/trg were swapped) → direction is
correct.

## Run
```bash
# sanity (no GPU): dataset + collate
python tap_vid_probe/test_dataset.py
# one trained ckpt on TAP-Vid (GPU)
CUDA_VISIBLE_DEVICES=0 python tap_vid_probe/smoke_eval.py
# full probe: 4 runs / 2 GPUs, full convergence
bash tap_vid_probe/run_probe.sh all
# watch curves + peak epochs
python tap_vid_probe/monitor.py
```

## Reading the result
`monitor.py` prints a `PEAK@` row per run. If `tapvid_davis` peaks at ~the same epoch as
`kitti2015` (or tss/pf), that benchmark is a viable **proxy**: select each model's TAP-Vid
checkpoint by the proxy's peak, no TAP-Vid retrain needed. If TAP-Vid peaks elsewhere, the
retrain route (RC grid) is required — and the no-peek `model_best.pth` rule still applies
(see HANDOFF_tapvid_davis.md §1b).
