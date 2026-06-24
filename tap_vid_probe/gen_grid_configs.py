"""Generate the RC retrain-route grid: MOVi-F x {models} x {regimes}, eval on 9 + TAP-Vid.

The local probe (configs/movif_*.yaml) trains only the TF/FF cells we needed tonight. The
retrain route needs the full regime sweep so every cell gets a real TAP-Vid PCK curve
(then pick each model's checkpoint by the no-peek model_best rule; see HANDOFF §1b).

Grid (10 cells):
  catspp     x {FF, TF, TT}
  flowformer x {FF, TF, TT}
  glunet     x {FF, TF, TT}
  raft       x {FF}            (RAFT has no pretrained backbone -> FF only)

Eval set = the canonical 9 + tapvid_davis (tapvid at alpha 0.03, rest 0.05).

Each cell is built by loading the validated base config for that model and toggling the
regime flags + eval block + snapshot dir. Emits to configs/grid/ plus run_grid.sh.

  python tap_vid_probe/gen_grid_configs.py            # write configs + launcher (dry plan)
  bash   tap_vid_probe/configs/grid/run_grid.sh 0     # run all cells sequentially on GPU0
RC: copy probe_cache to the cluster, set GRID_PATHS below (or inject via
slurm/machine_configs/remote.yaml) and submit through the usual slurm pipeline.
"""
import os
import copy
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "configs")
OUT = os.path.join(BASE, "grid")

# --- EDIT FOR RC: data roots + snapshot base (defaults are local) ---------------
GRID_PATHS = {
    "tapvid_davis_root": "/mnt/nvme_1tb_a/tapvid/probe_cache",
    "kitti_root": "/home/spencer/Data/correspondence/kitti",
    "tss_root": "/home/spencer/Data/correspondence/TSS_CVPR2016",
    "flyingthings_root": "/home/spencer/Data/FlyingThings3D_tiny",
    "pointodyssey_root": "/home/spencer/Data/PointOdyssey",
    "middlebury_root": "/home/spencer/Data/middlebury/all",
    "datapath": "./models/Datasets_CATs",
    "snap_base": "/mnt/nvme_1tb_a/snapshots/tapvid_grid",
}

EVAL_BENCHMARKS = ["tapvid_davis", "kitti2015", "kitti2012", "tss", "pfpascal",
                   "pfwillow", "spair", "flyingthings", "pointodyssey", "middlebury"]
EVAL_ALPHAS = [0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

# model -> (base config, regime -> flag overrides on model block)
MODELS = {
    "catspp": ("movif_catspp_tf.yaml", {
        "FF": {"pretrained_backbone": False, "freeze": False},
        "TF": {"pretrained_backbone": True,  "freeze": False},
        "TT": {"pretrained_backbone": True,  "freeze": True},
    }),
    "flowformer": ("movif_flowformer_tf.yaml", {
        "FF": {"pretrain": False, "freeze": False},
        "TF": {"pretrain": True,  "freeze": False},
        "TT": {"pretrain": True,  "freeze": True},
    }),
    "glunet": ("movif_glunet_tf.yaml", {
        "FF": {"pretrained_backbone": False, "freeze": False},
        "TF": {"pretrained_backbone": True,  "freeze": False},
        "TT": {"pretrained_backbone": True,  "freeze": True},
    }),
    "raft": ("movif_raft_ff.yaml", {
        "FF": {"pretrained_backbone": False, "freeze": False},
    }),
}


def eval_block():
    return {
        "eval_benchmarks": list(EVAL_BENCHMARKS),
        "eval_alphas": list(EVAL_ALPHAS),
        "thres": "img",
        "split_to_use_for_validation": "test",
        "val_batch_size": 2,
        "val_num_workers": 8,
        "prefetch_factor": 2,
        "use_motion_aware": False,
        "min_motion_pixels": 5.0,
        "zero_threshold": 0.5,
        "datapath": GRID_PATHS["datapath"],
        "tss_root": GRID_PATHS["tss_root"],
        "kitti_root": GRID_PATHS["kitti_root"],
        "kitti_val_use_full_training": True,
        "flyingthings_root": GRID_PATHS["flyingthings_root"],
        "pointodyssey_root": GRID_PATHS["pointodyssey_root"],
        "middlebury_root": GRID_PATHS["middlebury_root"],
        "tapvid_davis_root": GRID_PATHS["tapvid_davis_root"],
        "val_datasets": {
            "tapvid_davis": {"tapvid_stride": 20, "tapvid_frame_step": 5,
                             "tapvid_min_pts": 1, "reverse_flow": True, "normalize_images": True},
            "kitti2012": {"split": "val", "normalize_images": True},
            "kitti2015": {"split": "val", "normalize_images": True},
            "pointodyssey": {"pointodyssey_sequence_length": 4, "pointodyssey_num_pts_to_track": 32,
                             "use_all_valid": True, "val_sequence_fraction": 0.2,
                             "normalize_images": True, "split": "test"},
            "flyingthings": {"split": "test", "normalize_images": True, "val_dataset_fraction": 0.2},
            "middlebury": {"normalize_images": True, "reverse_flow": False},
        },
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = []
    for model, (base_name, regimes) in MODELS.items():
        with open(os.path.join(BASE, base_name)) as f:
            base = yaml.safe_load(f)
        for regime, flags in regimes.items():
            cfg = copy.deepcopy(base)
            cfg["model"].update(flags)
            cfg["evaluation"] = eval_block()
            name = f"movif_{model}_{regime.lower()}"
            cfg["paths"]["snapshots"] = os.path.join(GRID_PATHS["snap_base"], name)
            cfg["paths"]["save_epoch_checkpoints"] = False
            cfg["training"]["eval_initial"] = True
            out_path = os.path.join(OUT, name + ".yaml")
            with open(out_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            cells.append(name)
            print(f"  wrote {out_path}")

    # sequential local launcher
    launcher = os.path.join(OUT, "run_grid.sh")
    with open(launcher, "w") as f:
        f.write("#!/usr/bin/env bash\n# Sequential local grid runner. Usage: bash run_grid.sh <gpu>\n")
        f.write("set -u\nREPO=/home/spencer/Projects/OnlineSyntheticCorrespondence\n")
        f.write('GPU="${1:-0}"\ncd "$REPO"\n')
        for name in cells:
            f.write(f'echo "[$(date +%T)] START {name}"\n')
            f.write(f'CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py '
                    f'--config tap_vid_probe/configs/grid/{name}.yaml '
                    f'> tap_vid_probe/logs/grid_{name}.log 2>&1\n')
        f.write('echo "grid done"\n')
    print(f"\n{len(cells)} cells -> {OUT}\nlauncher: {launcher}")
    print("cells:", ", ".join(cells))


if __name__ == "__main__":
    main()
