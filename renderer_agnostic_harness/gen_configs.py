#!/usr/bin/env python3
"""Generate the 6 training configs for the SDF-fractal renderer-agnostic table.

Cells = {cats, glunet, flowformer} x {default, tuned}, all pretrained-frozen (TT).
Each config is derived from that architecture's EXISTING trial76 (tuned) run config,
so model / lr / scheduler / batch are bit-identical to the published runs and the ONLY
thing that varies between default and tuned is `dataset.geometry_config_overrides`:

    tuned   -> the trial76 TPE sampler block (kept from the template)
    default -> null  (falls back to OnlineGeometryConfig.yaml defaults)

This is the controlled comparison: hold everything fixed, toggle the source.

Training eval is kept MINIMAL (kitti2012/kitti2015 @0.05, for checkpoint monitoring).
The full strict-alpha KITTI + multi-alpha TSS + multi-stride TAP-Vid evaluation is done
POST-HOC by score_transfer_cell.py on the saved epoch_*.pth, so every reported number
comes from one verified evaluator.

Writes configs to renderer_agnostic_harness/configs/<arch>_<source>.yaml.
NOTHING is launched. Review the configs, then run run_table.sh.
"""
import os, sys, copy, yaml

HARNESS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HARNESS)

# Architecture -> existing trial76 (tuned) config used as the template.
TEMPLATES = {
    'cats':       '/mnt/nvme_1tb_a/snapshots/transfer_grid/synthetic_fractal_trial76_pt1_fz1_2026_06_10_13_28/config.yaml',
    'glunet':     '/mnt/nvme_1tb_a/snapshots/transfer_grid_glunet/synthetic_fractal_trial76_pt1_fz1_2026_06_15_04_41/config.yaml',
    'flowformer': os.path.join(REPO, 'scripts/transfer_analysis_v5/flowformer_rc_results/'
                  'sdf_kitti2015_trial76_widebnds_flowformer_steps100_pretrainTrue_freezeTrue_2026_06_11_15_01/config.yaml'),
}

# Local data roots (templates carry RC /home/slow1 paths).
LOCAL_ROOTS = {
    'kitti_root':        '/home/spencer/Data/correspondence/kitti',
    'tss_root':          '/home/spencer/Data/correspondence/TSS_CVPR2016',
    'tapvid_davis_root': '/mnt/nvme_1tb_a/tapvid/probe_cache',
    'flyingthings_root': '/home/spencer/Data/FlyingThings3D_tiny',
    'middlebury_root':   '/home/spencer/Data/middlebury/all',
}

# Common training overrides for a bounded, comparable re-run (matches the CATs
# trial76 recipe: 50 epochs, validate every epoch, constant LR within the window).
COMMON_TRAIN = {
    'epochs': 30,                  # matched across all archs: 30 ep x 100 steps = 3000 train steps
    'check_val_every_n_epoch': 1,
    'steps_per_epoch': 100,
    'eval_initial': False,
    'mmd_every_n_epochs': 0,
}

# Per-arch training overrides. FlowFormer at 512^2 does NOT fit batch>=2 on a 24GB
# 3090, so use batch1 + accum8 (effective batch 8) — the established local recipe.
PER_ARCH_TRAIN = {
    'cats':       {'batch_size': 8, 'accumulate_grad_batches': 1},
    'glunet':     {'batch_size': 8, 'accumulate_grad_batches': 1},
    'flowformer': {'batch_size': 1, 'accumulate_grad_batches': 8},
}

OUT_SNAPSHOTS = os.environ.get('RA_OUT', '/mnt/nvme_1tb_a/renderer_agnostic')


def fix_rc_paths(obj):
    """Recursively rewrite RC /home/slow1 -> local /home/spencer in string values."""
    if isinstance(obj, dict):
        return {k: fix_rc_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_rc_paths(v) for v in obj]
    if isinstance(obj, str) and obj.startswith('/home/slow1'):
        return obj.replace('/home/slow1', '/home/spencer')
    return obj


def build_cell(arch, source):
    cfg = yaml.safe_load(open(TEMPLATES[arch]))
    cfg = fix_rc_paths(cfg)

    # --- the one variable: source ---
    assert 'geometry_config_overrides' in cfg['dataset'], "template missing overrides key"
    if source == 'default':
        cfg['dataset']['geometry_config_overrides'] = None
    elif source == 'tuned':
        assert cfg['dataset']['geometry_config_overrides'], "tuned template must carry trial76 overrides"
    else:
        raise ValueError(source)

    # --- model: enforce pretrained-frozen (TT) ---
    cfg['model']['freeze'] = True
    if arch == 'flowformer':
        cfg['model']['pretrain'] = True
    else:
        cfg['model']['pretrained_backbone'] = True

    # --- training: bounded comparable recipe ---
    cfg.setdefault('training', {})
    cfg['training'].update(COMMON_TRAIN)
    cfg['training'].update(PER_ARCH_TRAIN[arch])

    # --- paths: per-cell output dir, keep per-epoch checkpoints ---
    cfg.setdefault('paths', {})
    cfg['paths']['snapshots'] = OUT_SNAPSHOTS
    cfg['paths']['save_epoch_checkpoints'] = True
    cfg['paths']['pretrained'] = None
    cfg['paths']['start_epoch'] = 0

    # --- evaluation: local roots + MINIMAL monitoring suite (post-hoc does the rest) ---
    ev = cfg['evaluation']
    for k, v in LOCAL_ROOTS.items():
        ev[k] = v
    ev['kitti_val_use_full_training'] = True
    ev['use_motion_aware'] = False
    ev['eval_benchmarks'] = ['kitti2015', 'kitti2012']
    ev['eval_alphas'] = [0.05, 0.05]
    # FlowFormer at 512^2 OOMs on batched validation (template ships val_batch_size=32);
    # force a tiny val batch. Training stays batch1+accum8.
    if arch == 'flowformer':
        ev['val_batch_size'] = 1
    ev['val_datasets'] = {
        'kitti2015': {'split': 'val', 'normalize_images': True},
        'kitti2012': {'split': 'val', 'normalize_images': True},
    }
    return cfg


def main():
    out_dir = os.path.join(HARNESS, 'configs')
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for arch in TEMPLATES:
        for source in ('default', 'tuned'):
            cfg = build_cell(arch, source)
            # run-name so each cell lands in its own snapshot subdir
            name = f'{arch}_{source}'
            cfg.setdefault('paths', {})['run_name'] = name
            path = os.path.join(out_dir, f'{name}.yaml')
            with open(path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
            ov = cfg['dataset']['geometry_config_overrides']
            written.append((name, path, 'OVERRIDES' if ov else 'null'))
    print(f"snapshots base = {OUT_SNAPSHOTS}")
    print(f"wrote {len(written)} configs to {out_dir}:")
    for name, path, ov in written:
        print(f"  {name:<20} overrides={ov:<10} {path}")


if __name__ == '__main__':
    main()
