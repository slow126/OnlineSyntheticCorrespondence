#!/usr/bin/env python3
"""Sequential GLU-Net transfer-grid launcher — the architecture-generality arm.

Mirror of run_transfer_grid.py (the CATs++ grid) for GLU-Net, so the
dose-response table (paper Table 6) stops resting on one architecture.
Trains the SAME materialized intervention sources from scratch (FF =
pretrained_backbone False, freeze False) through GLU-Net via the unified
train_lightning (model.type: glunet).

Pre-registered predictions (write the date down before harvest — see
ACCV_2026/DRAFT_NOTES.md): on this grid the off-target cost carries the
dynamic range against FlyingThings (3.1x spread vs 1.3x for coverage; the
KITTI cells are precision-matched by family design). So GLU-Net-FF should
be off-target-ranked on FlyingThings (positive rho), the summed cost should
attenuate, and missing support should stay ~negative — EVEN THOUGH scratch
GLU-Net carries the missing-support cost on canonical sources — because a
low-variation cost cannot drive a ranking regardless of consumer. If
GLU-Net-FF came out recall-ranked on FlyingThings, the two-cost model would
be wrong.

Differences from the CATs++ script, all deliberate:
  * model block      : glunet (ResNet-50 encoder, local_window_size 9), per
                       src/configs/lightning/glunet_training_base_rc.yaml.
  * training recipe  : the glunet_cos canonical recipe (100 ep x 100 steps,
                       cosine anneal, gradient_clip 1.0, lr 1e-3/1e-5), so
                       these cells follow the same convention as GLU-Net's
                       canonical transfer-table rows. Val every 5 epochs
                       (canonical used 20; the grid ranking wants a finer
                       peak_pck estimate, and all cells share the cadence).
  * downsample_flow  : null (GLU-Net trains its native multi-scale loss on
                       full-resolution flow; CATs++ used 32).
  * snapshots        : SNAP_DIR is transfer_grid_glunet/ — SEPARATE from the
                       CATs++ grid dir, because the harvest
                       (intervention_oos_test.py) parses the source name from
                       the snapshot dir name; mixing architectures in one dir
                       would corrupt it. Harvest these cells by pointing the
                       same parser at this dir.
  * FF memory        : the encoder trains in FF. ResNet-50 at 512^2 batch 8
                       should fit 24 GB, but it is untested locally — run
                       --smoke first; if it OOMs set FF_BATCH below (steps
                       are rescaled automatically to keep the sample budget).

Run:
    python scripts/run_transfer_grid_glunet.py                 # dry-run plan
    python scripts/run_transfer_grid_glunet.py --smoke --run --gpus 0
    python scripts/run_transfer_grid_glunet.py --run --gpus 0,1
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
KUBRIC_ROOT = Path("/mnt/nvme_1tb_a/kubric_interventions/ladder")

# SEPARATE dir from the CATs++ grid (see docstring).
SNAP_DIR = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid_glunet_ladder")
GEN_CFG_DIR = REPO / "src/configs/lightning/transfer_grid_glunet"
LOG_DIR = SNAP_DIR / "logs"

# Templates: kubric template supplies dataset+evaluation blocks (local paths);
# the synthetic-fractal template carries the TPE theta. Model/training blocks
# are REPLACED with the GLU-Net recipe below.
KUBRIC_TEMPLATE = REPO / "src/configs/lightning/kubric_kitti_recovered_hq.yaml"
SYNTH_TEMPLATE = REPO / "snapshots/sdf_kitti2015_trial76_widebnds_2026_05_29_14_10/config.yaml"

MIN_SCENES = 1250  # ladder datasets are 1250 by design (was 4900 for 5000-scene natural sets)

# GLU-Net FF trains the ResNet-50 encoder. Batch 8 at 512^2 is expected to fit
# 24 GB but is locally untested — smoke first; drop to 4 (steps auto-rescale)
# if it OOMs.
FF_BATCH = 8
REF_BATCH = 8  # the canonical glunet batch the steps budget is defined at

VAL_WORKERS = 8
TRAIN_WORKERS = 4

# The glunet_cos canonical recipe (slurm/experiment_configs/
# glunet_cos_canonical_rc.yaml), with a finer val cadence for grid peak_pck.
GLUNET_TRAINING = {
    "epochs": 100,
    "batch_size": REF_BATCH,
    "n_threads": TRAIN_WORKERS,
    "seed": 2021,
    "lr": 1.0e-3,
    "lr_backbone": 1.0e-5,
    "weight_decay": 5.0e-4,
    "momentum": 0.9,
    "scheduler": "cosine",
    "step": "[70, 80, 90]",
    "step_gamma": 0.5,
    "gradient_clip_val": 1.0,
    "steps_per_epoch": 100,
    "check_val_every_n_epoch": 5,
    "augmentation": False,
    "min_flow_length": None,
    "max_flow_length": None,
    "mmd_every_n_epochs": 0,
    "eval_initial": False,
    "enable_debug": False,
    "debug_visualization_benchmarks": None,
    "debug_visualization_persist": False,
}

GLUNET_MODEL_BASE = {
    "type": "glunet",
    "glunet": {
        "model_name": "resnet50",
        "local_window_size": 9,
        "decoder_dense_connect": False,
    },
}

# --- Sources (same keys/dirs as the CATs++ grid) ------------------------------
KUBRIC_SOURCES = {
    "kitti_m025_hq": "kitti_m025_hq",
    "kitti_m050_hq": "kitti_m050_hq",
    "kitti_m100_hq": "kitti_m100_hq",
    "kitti_m150_hq": "kitti_m150_hq",
    "kitti_m200_hq": "kitti_m200_hq",
    "kitti_m025_matte": "kitti_m025_matte",
    "kitti_m050_matte": "kitti_m050_matte",
    "kitti_m100_matte": "kitti_m100_matte",
    "kitti_m150_matte": "kitti_m150_matte",
    "kitti_m200_matte": "kitti_m200_matte",
    "kitti_m025_gsohq": "kitti_m025_gsohq",
    "kitti_m050_gsohq": "kitti_m050_gsohq",
    "kitti_m100_gsohq": "kitti_m100_gsohq",
    "kitti_m150_gsohq": "kitti_m150_gsohq",
    "kitti_m200_gsohq": "kitti_m200_gsohq",
    "kitti_m025_gsomatte": "kitti_m025_gsomatte",
    "kitti_m050_gsomatte": "kitti_m050_gsomatte",
    "kitti_m100_gsomatte": "kitti_m100_gsomatte",
    "kitti_m150_gsomatte": "kitti_m150_gsomatte",
    "kitti_m200_gsomatte": "kitti_m200_gsomatte",
}
SYNTH_SOURCES = {"synthetic_fractal_trial76": SYNTH_TEMPLATE}

VARIANTS = {"TT": (True, True), "FF": (False, False), "TF": (True, False)}

# The generality arm: FF (from scratch) for every grid source. TT cells were
# originally deliberately absent, but 2026-06-14 we add the pretrained (TT =
# frozen ImageNet backbone) complement to complete the transfer table's
# pretrained branch — recovered-vs-MOVi-F needs both arms, and GLU-Net movi_f
# already has all four configs (glunet_cos_clean / transfer_table.csv).
# Excludes the dropped trial19/lowtex_matte sources; add them here for full
# FF/TT symmetry if ever wanted.
PRETRAINED_TT_SOURCES = [
    "kitti_recovered_gso_hq", "kitti_recovered_gso_matte",
    "kitti_badmotion_ft_gso_hq", "kitti_badmotion_ft_gso_matte",
    "kitti_recovered_hq", "kitti_recovered_matte",
    "ft_recovered_hq", "ft_recovered_matte",
]
GRID: list[tuple[str, str]] = (
    [(s, "FF") for s in KUBRIC_SOURCES]
    + [(s, "TF") for s in KUBRIC_SOURCES]
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
    # same naming convention as the CATs++ grid; the architecture lives in the
    # snapshot DIR, not the cell name, so the harvest parser works unchanged
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
        synth_threads = None
    else:  # synthetic fractal — online generator + geometry overrides
        cfg = copy.deepcopy(load_yaml(SYNTH_SOURCES[source]))
        kub_eval = copy.deepcopy(load_yaml(KUBRIC_TEMPLATE)["evaluation"])
        keep = ["kitti2015", "kitti2012", "flyingthings"]
        kub_eval["eval_benchmarks"] = keep
        kub_eval["eval_alphas"] = [0.05] * len(keep)
        kub_eval["val_datasets"] = {b: kub_eval["val_datasets"][b] for b in keep}
        cfg["evaluation"] = kub_eval
        # the single-process renderer needs its template's loader setting
        synth_threads = cfg.get("training", {}).get("n_threads", 0)

    # GLU-Net model block (replaces the CATs++ one wholesale)
    cfg["model"] = copy.deepcopy(GLUNET_MODEL_BASE)
    cfg["model"]["pretrained_backbone"] = bool(pretrained)
    cfg["model"]["freeze"] = bool(freeze)

    # GLU-Net training recipe (replaces the CATs++ one), then the horizon
    cfg["training"] = copy.deepcopy(GLUNET_TRAINING)
    cfg["training"].update(horizon)
    if synth_threads is not None:
        cfg["training"]["n_threads"] = synth_threads

    # GLU-Net trains its native multi-scale loss on full-resolution flow
    cfg["dataset"]["downsample_flow"] = None

    if not freeze and FF_BATCH != REF_BATCH:
        scale = max(1, round(REF_BATCH / FF_BATCH))
        cfg["training"]["batch_size"] = FF_BATCH
        cfg["evaluation"]["val_batch_size"] = FF_BATCH
        cfg["training"]["steps_per_epoch"] = (
            int(cfg["training"]["steps_per_epoch"]) * scale)

    cfg["evaluation"]["val_num_workers"] = VAL_WORKERS
    cfg.setdefault("paths", {})
    cfg["paths"]["snapshots"] = str(snap)
    cfg["paths"]["pretrained"] = None
    cfg["paths"]["save_epoch_checkpoints"] = False

    status = "SKIP (exists)" if snapshot_exists(basename, snap) else "RUN"
    return basename, cfg, status


def run_pool(runnable: list, gpus: list[str], log_dir: Path) -> list[tuple[str, int]]:
    """Dynamic GPU pool: each GPU worker pulls the next queued cell when it
    frees up. A poor-man's slurm queue — both GPUs saturated until drained."""
    q: Queue = Queue()
    for cell in runnable:
        q.put(cell)
    plock = threading.Lock()
    results: list[tuple[str, int]] = []

    def worker(gpu: str) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
                    help="1-epoch/10-step sanity pass for ALL cells into a _smoke "
                         "subdir (validates configs + memory; not the real grid).")
    ap.add_argument("--only", default=None, help="substring filter on the cell name.")
    ap.add_argument("--variants", default=None,
                    help="comma list to restrict variants, e.g. FF.")
    args = ap.parse_args()

    snap = SNAP_DIR / "_smoke" if args.smoke else SNAP_DIR
    horizon = ({"epochs": 1, "steps_per_epoch": 10, "check_val_every_n_epoch": 1}
               if args.smoke else {})
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
    mode = "SMOKE 1ep/10step" if args.smoke else "FULL (glunet_cos recipe)"
    print(f"\nGLU-Net transfer grid [{mode}] -> {snap}")
    print(f"trainer: {TRAINER.name}  gpu pool: {gpus}\n")
    print(f"{'cell':44s} {'var':4s} status")
    print("-" * 78)
    for basename, variant, status, _ in plan:
        print(f"{basename:44s} {variant:4s} {status}")
    runnable = [p for p in plan if p[2] == "RUN"]
    print(f"\n{len(runnable)} cell(s) to run.")
    if not args.run:
        print("Dry run only — pass --run to launch.")
        return
    if not runnable:
        print("Nothing to do.")
        return
    results = run_pool(runnable, gpus, log_dir)
    fails = [b for b, rc in results if rc != 0]
    print(f"\nDone: {len(results) - len(fails)} ok, {len(fails)} failed"
          + (f" ({', '.join(fails)})" if fails else ""))


if __name__ == "__main__":
    main()
