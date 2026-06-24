#!/usr/bin/env python3
"""Sequential CATs++ transfer-grid launcher (kubric + synthetic interventions).

Trains the materialized intervention datasets through CATs++ at two backbone
regimes — True_True (ImageNet ResNet, frozen) and False_False (random ResNet,
trainable = no feature-extractor pretraining bias) — plus the TPE-found
synthetic-fractal source, into ONE dedicated snapshot dir so the grid is easy to
find and the transfer cells are easy to harvest afterward.

Design choices baked in:
  * trainer       : train_lightning.py --config <yaml>  (per the trial19 header)
  * horizon       : trial19's (50 ep x 100 steps, val every epoch) -> bounded.
  * snapshots     : SNAP_DIR (on /mnt/nvme_1tb_a, NOT root which is ~93% full).
  * readiness     : a kubric source is skipped until it has >= MIN_SCENES scenes
                    (so a still-materializing dataset is not trained on a partial).
  * resumable     : a cell whose snapshot already exists is skipped.
  * SAFE default  : prints the plan and exits; pass --run to actually launch.

Edit SOURCES / GRID below to taste. Run:
    python scripts/run_transfer_grid.py                 # dry-run plan
    python scripts/run_transfer_grid.py --run --gpu 0   # launch sequentially
    python scripts/run_transfer_grid.py --run --gpu 1 --only ft_recovered   # subset
"""
from __future__ import annotations

import argparse
import copy
import glob
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import yaml

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "train_lightning.py"
KUBRIC_ROOT = Path("/mnt/nvme_1tb_a/kubric_interventions/datasets")

# Dedicated, easy-to-find output dir (759 G free here; root is ~93% full).
SNAP_DIR = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")
GEN_CFG_DIR = REPO / "src/configs/lightning/transfer_grid"
LOG_DIR = SNAP_DIR / "logs"

# Templates (real configs we deep-copy + override).
KUBRIC_TEMPLATE = REPO / "src/configs/lightning/kubric_kitti_recovered_hq.yaml"
# Synthetic-fractal template = the TPE-found run's saved config (carries the
# geometry_config_overrides theta). Point this at your preferred fractal config.
SYNTH_TEMPLATE = REPO / "snapshots/sdf_kitti2015_trial76_widebnds_2026_05_29_14_10/config.yaml"

# Bounded training horizon (trial19's; converges fast so 50 ep is plenty).
HORIZON = {"epochs": 50, "steps_per_epoch": 100, "check_val_every_n_epoch": 1}

MIN_SCENES = 4900  # a kubric source must have at least this many scene_* dirs

# From-scratch (freeze=False) trains the ResNet-101 backbone, so backprop at 512^2
# OOMs at batch 8 on 24 GB. Trainable-backbone cells use a much smaller batch.
FF_BATCH = 2

# CPU budget: 2 concurrent runs share 32 cores. The templates' val_num_workers=32
# oversubscribes badly (2x32=64 loader procs -> load ~75, starves the GPUs and
# spikes RAM). Cap loaders so peak concurrency stays well under the core count.
VAL_WORKERS = 8
TRAIN_WORKERS = 4

# --- Sources -----------------------------------------------------------------
# key -> materialized dataset folder under KUBRIC_ROOT.
KUBRIC_SOURCES = {
    # new GSO sets (materializing now) -- need BOTH TT and FF
    "kitti_recovered_gso_hq":      "kitti_recovered_gso_hq_5000",
    "kitti_recovered_gso_matte":   "kitti_recovered_gso_matte_5000",
    "kitti_badmotion_ft_gso_hq":   "kitti_badmotion_ft_gso_hq_5000",
    # 2026-06-10: matte twin of badmotion_ft_gso_hq -> completes the GSO 2x2
    # (object*matte*GSO, the only cell never rendered). FF only.
    "kitti_badmotion_ft_gso_matte": "kitti_badmotion_ft_gso_matte_5000",
    # kubasic sets (already have TT in ./snapshots) -- need FF only
    "kitti_recovered_hq":          "kitti_recovered_hq_5000",
    "kitti_recovered_matte":       "kitti_recovered_matte_5000",
    "ft_recovered_hq":             "flyingthings_recovered_hq_5000",
    "ft_recovered_matte":          "flyingthings_recovered_matte_5000",
    "trial19":                     "kitti2015_hq_trial19_5000",
    "lowtex_matte":                "kitti2015_lowtex_matte_5000",
    # (future) jitter set -- uncomment once materialized:
    # "kitti_recovered_gso_hq_jitter": "kitti_recovered_gso_hq_jitter_5000",
}
SYNTH_SOURCES = {"synthetic_fractal_trial76": SYNTH_TEMPLATE}

# Backbone regimes: (pretrained_backbone, freeze)
VARIANTS = {"TT": (True, True), "FF": (False, False)}

# --- The grid: which (source, variant) cells to run --------------------------
# TT already exists in ./snapshots for the kubasic + trial19 + lowtex + synthetic
# sources, so only the new GSO sets need TT. FF is missing for everything kubric
# + the synthetic fractal, so FF for all. (movi_f already has all 4 in the table.)
GRID: list[tuple[str, str]] = (
    [(s, "TT") for s in ("kitti_recovered_gso_hq", "kitti_recovered_gso_matte",
                         "kitti_badmotion_ft_gso_hq")]
    + [(s, "FF") for s in KUBRIC_SOURCES]
    + [("synthetic_fractal_trial76", "FF")]
    # 2026-06-10 TT-arm parity retrains: lowtex_matte had no TT anywhere, and
    # trial76's only TT snapshot was the 19-epoch kitti2015-only TPE search run.
    # These two close the grid so the OOS pretrained test is discriminative.
    + [("lowtex_matte", "TT"), ("synthetic_fractal_trial76", "TT")]
    # 2026-06-10 (cont.): trial19's only TT snapshot was the kitti2015-ONLY June-1
    # harvested run, so the pretrained grid was missing trial19 on flyingthings +
    # kitti2012 (the discriminative cells). Retrain it through the kubric template
    # so all three benchmarks are logged; the harvested dir is moved aside first.
    + [("trial19", "TT")]
)


def load_yaml(p: Path) -> dict:
    return yaml.safe_load(Path(p).read_text())


def scene_count(datapath: Path) -> int:
    return len(glob.glob(str(datapath / "train" / "scene_*")))


def snapshot_exists(basename: str, snap: Path) -> bool:
    return bool(glob.glob(str(snap / f"{basename}_*")))


def build_config(source: str, variant: str, snap: Path, horizon: dict,
                 ) -> tuple[str, dict, str]:
    """Return (basename, config_dict, status) for a cell."""
    pretrained, freeze = VARIANTS[variant]
    basename = f"{source}_pt{int(pretrained)}_fz{int(freeze)}"

    if source in KUBRIC_SOURCES:
        cfg = copy.deepcopy(load_yaml(KUBRIC_TEMPLATE))
        datapath = KUBRIC_ROOT / KUBRIC_SOURCES[source]
        if not datapath.is_dir():
            return basename, cfg, f"MISSING ({datapath} not found)"
        n = scene_count(datapath)
        if n < MIN_SCENES:
            return basename, cfg, f"NOT-READY ({n}/{MIN_SCENES} scenes)"
        cfg["dataset"]["datapath"] = str(datapath)
    else:  # synthetic fractal -- uses the online generator + geometry overrides
        cfg = copy.deepcopy(load_yaml(SYNTH_SOURCES[source]))
        # The TPE search config only evaluates kitti2015. Make its eval block
        # apples-to-apples with the kubric TT runs: kitti2015 + kitti2012 +
        # flyingthings (middlebury excluded — eval bugged, dropped at harvest).
        # Only the eval block is replaced; the SDF training/geometry config is
        # untouched. roots/splits/val_dataset settings now match the other cells.
        kub_eval = copy.deepcopy(load_yaml(KUBRIC_TEMPLATE)["evaluation"])
        keep = ["kitti2015", "kitti2012", "flyingthings"]
        kub_eval["eval_benchmarks"] = keep
        kub_eval["eval_alphas"] = [0.05] * len(keep)
        kub_eval["val_datasets"] = {b: kub_eval["val_datasets"][b] for b in keep}
        cfg["evaluation"] = kub_eval

    cfg["model"]["pretrained_backbone"] = bool(pretrained)
    cfg["model"]["freeze"] = bool(freeze)
    cfg["training"].update(horizon)
    if not freeze:  # trainable backbone OOMs at batch 8 (512^2 ResNet-101 on 24 GB)
        ref_batch = int(cfg["training"]["batch_size"])      # template batch (what TT uses)
        scale = max(1, round(ref_batch / FF_BATCH))
        cfg["training"]["batch_size"] = FF_BATCH
        cfg["evaluation"]["val_batch_size"] = FF_BATCH
        # keep it apples-to-apples: steps scale 1/batch so samples/epoch (and the
        # total budget + per-epoch val cadence) match the batch-8 TT runs.
        cfg["training"]["steps_per_epoch"] = int(cfg["training"]["steps_per_epoch"]) * scale
    # cap dataloader workers to avoid CPU oversubscription across the 2-run pool
    cfg["training"]["n_threads"] = TRAIN_WORKERS
    cfg["evaluation"]["val_num_workers"] = VAL_WORKERS
    cfg.setdefault("paths", {})
    cfg["paths"]["snapshots"] = str(snap)
    cfg["paths"]["pretrained"] = None

    status = "SKIP (exists)" if snapshot_exists(basename, snap) else "RUN"
    return basename, cfg, status


def run_pool(runnable: list, gpus: list[str], log_dir: Path) -> list[tuple[str, int]]:
    """Dynamic GPU pool: each GPU worker pulls the next queued cell when it frees up.
    A poor-man's slurm queue — no static split, both GPUs saturated until drained."""
    q: Queue = Queue()
    for cell in runnable:
        q.put(cell)
    plock = threading.Lock()
    results: list[tuple[str, int]] = []

    def worker(gpu: str) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # reduce fragmentation
        while True:
            try:
                basename, variant, status, cfg_path = q.get_nowait()
            except Empty:
                return
            log = log_dir / f"{basename}.log"
            with plock:
                print(f"[gpu {gpu}] START {basename}  ({q.qsize()} still queued) -> {log}", flush=True)
            t0 = time.time()
            with log.open("w") as fh:
                rc = subprocess.run(
                    [sys.executable, "-u", str(TRAINER), "--config", str(cfg_path)],
                    cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT,
                ).returncode
            dt = (time.time() - t0) / 60
            with plock:
                print(f"[gpu {gpu}] {'OK  ' if rc == 0 else f'FAIL rc={rc}'} {basename} "
                      f"in {dt:.1f}m" + ("" if rc == 0 else f"  -- see {log}"), flush=True)
            results.append((basename, rc))

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true",
                    help="actually launch training (default: dry-run plan only).")
    ap.add_argument("--gpus", default="0",
                    help="comma list of GPU ids forming the worker pool, e.g. 0,1.")
    ap.add_argument("--smoke", action="store_true",
                    help="1-epoch/10-step sanity pass for ALL cells into a _smoke subdir "
                         "(validates configs + the queue; does NOT touch the real grid).")
    ap.add_argument("--only", default=None, help="substring filter on the cell name.")
    ap.add_argument("--variants", default=None,
                    help="comma list to restrict variants, e.g. FF or TT,FF.")
    args = ap.parse_args()

    snap = SNAP_DIR / "_smoke" if args.smoke else SNAP_DIR
    horizon = ({"epochs": 1, "steps_per_epoch": 10, "check_val_every_n_epoch": 1}
               if args.smoke else HORIZON)
    gen_dir = GEN_CFG_DIR / "_smoke" if args.smoke else GEN_CFG_DIR
    log_dir = snap / "logs"
    for d in (snap, gen_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    want_variants = set(args.variants.split(",")) if args.variants else None
    plan = []
    for source, variant in GRID:
        if want_variants and variant not in want_variants:
            continue
        if args.only and args.only not in source:
            continue
        basename, cfg, status = build_config(source, variant, snap, horizon)
        cfg_path = gen_dir / f"{basename}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        plan.append((basename, variant, status, cfg_path))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    mode = "SMOKE 1ep/10step" if args.smoke else "FULL"
    print(f"\nTransfer grid [{mode}] -> {snap}")
    print(f"trainer: {TRAINER.name}  horizon: {horizon}  gpu pool: {gpus}\n")
    print(f"{'cell':40s} {'var':4s} status")
    print("-" * 78)
    for basename, variant, status, _ in plan:
        print(f"{basename:40s} {variant:4s} {status}")
    runnable = [p for p in plan if p[2] == "RUN"]
    print(f"\n{len(runnable)} to run, "
          f"{sum(1 for p in plan if p[2].startswith('SKIP'))} already done, "
          f"{sum(1 for p in plan if p[2].startswith(('NOT-READY', 'MISSING')))} not ready.")

    if not args.run:
        print(f"\n(dry run — configs in {gen_dir}. Add --run to launch on GPUs {gpus}.)")
        return

    print(f"\nLaunching {len(runnable)} runs across GPU pool {gpus}...\n")
    results = run_pool(runnable, gpus, log_dir)
    ok = sum(1 for _, rc in results if rc == 0)
    print(f"\nDone [{mode}]. {ok}/{len(results)} succeeded. Snapshots in {snap}")
    if results and ok < len(results):
        print("FAILED: " + ", ".join(b for b, rc in results if rc != 0))
    if not args.smoke and ok:
        print("Next: point the AUC step at this dir, e.g.\n"
              f"  python scripts/transfer_analysis_v3/compute_kubric_auc.py --snapshots {snap} ...")


if __name__ == "__main__":
    main()
