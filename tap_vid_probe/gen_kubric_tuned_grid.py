#!/usr/bin/env python3
"""Generate the all-architecture Kubric "tuned-source" transfer grid for Table 8.

For each (architecture x source-stratum) it emits a training config + slurm job.
Each arch's model/training block is loaded from a baseline config in _base/ so the
recipe is IDENTICAL to the MOVi-F baselines (movif_*_tt / raft_movif_ff) -- only the
training SOURCE changes. This makes the tuned-vs-MOVi-F comparison in Table 8
apples-to-apples (same recipe, vary only the data source).

Evaluation (all cells): TSS @{.10,.05,.03}, TAP-Vid-DAVIS strides {1,2,4,8,16},
and KITTI-2012/2015 @{.05,.03,.01} -- the alpha variants are now first-class
benchmark names (see train_cats_unified.py), so no post-hoc re-scoring.

Source-parameterized: add an entry to SOURCES to retarget at a new render (e.g.
the incoming higher-variability kitti-tuned set) without touching anything else.

Usage:  python3 tap_vid_probe/gen_kubric_tuned_grid.py --source tss_dense
"""
import argparse
import copy
import os

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(HERE, 'configs', 'grid_kubric_tuned', '_base')
OUT_DIR = os.path.join(HERE, 'configs', 'grid_kubric_tuned')
JOB_DIR = os.path.join(HERE, '..', 'slurm', 'kubric_tuned_jobs')
RC_REPO = '/home/slow1/Projects/OnlineSyntheticCorrespondence'

# arch -> (base config, slurm walltime). TT = frozen pretrained backbone; RAFT = end-to-end scratch.
ARCHS = {
    'cats_tt':       ('base_cats_tt.yaml',       '12:00:00'),
    'glunet_tt':     ('base_glunet_tt.yaml',     '08:00:00'),
    'flowformer_tt': ('base_flowformer_tt.yaml', '14:00:00'),
    'raft_ff':       ('base_raft_ff.yaml',       '10:00:00'),
}

# source -> datapath(s) on RC. strata=None => single pooled dataset (one cell per arch).
# strata=[...] => per-stratum matrix ({s} substituted into datapath).
SOURCES = {
    # Pooled: all 5 strata shuffled into one dataset (symlinks in tss_dense_pooled/train).
    # 4 archs x 1 dataset = 4 jobs. This is the primary "can Kubric beat MOVi-F" test.
    'tss_dense_pool': {
        'datapath': '/home/slow1/Data/tss_dense_pooled',
        'strata': None,
    },
    # Per-stratum matrix (kept for reference; not the current plan).
    'tss_dense': {
        'datapath': '/home/slow1/Data/tss_dense/tss_dense_{s}',
        'strata': ['s1', 's2', 's3', 's4', 's5'],
    },
    # Higher-variability kitti-tuned set: 12 family-deduped objects (4 moving),
    # grounded floor_scatter, forward dolly with per-scene magnitude spread (median
    # 2.18, range 0-8). On the group FS to dodge the /home/slow1 inode cap.
    'kitti_fitted12': {
        'datapath': '/home/slow1/fsl_groups/grp_farrell/slow1/Data/kitti_fitted12',
        'strata': None,
    },
    # Same render as kitti_fitted12, but the loader swaps source/target on 50% of
    # samples (src_tgt_flip, matching the SDF synthetic_flow_warp_swap rate), so a
    # zoom-in pair is presented as zoom-out half the time. Tests whether scale-up
    # plus scale-down matching closes the gap to the SDF source.
    'kitti_fitted12_flip': {
        'datapath': '/home/slow1/fsl_groups/grp_farrell/slow1/Data/kitti_fitted12',
        'strata': None,
        'src_tgt_flip': 0.5,
    },
    # Dataset-size-matched MOVi-F baseline. The standard MOVi-F source streams
    # ~6k videos x 23 pairs (~138k unique pairs); tss_dense_pooled has 4691
    # scenes with one pair each. This caps MOVi-F to 4691 videos x 1 pair = 4691
    # unique pairs from 4691 distinct scenes, matching tss_dense's size so the
    # MOVi-F-vs-tuned comparison in Table 6 needs no dataset-size caveat.
    'movif_matched': {
        'dataset_name': 'movi_f',
        'datapath': '/home/slow1/Data/movi_f/512x512/1.0.0',
        'strata': None,
        'movi_f_max_videos': 4691,
        'movi_f_pairs_per_video': 1,
    },
    # Full MOVi-F through this exact grid (same _base recipe + eval) -- a clean,
    # same-harness reproduction of the "Kubric, generic" baseline, to check
    # whether the table's full-MOVi-F rows are recipe-comparable to the tuned grid.
    'movif_full': {
        'dataset_name': 'movi_f',
        'datapath': '/home/slow1/Data/movi_f/512x512/1.0.0',
        'strata': None,
        'movi_f_max_videos': None,
        'movi_f_pairs_per_video': None,
    },
    # Large-motion MOVi-F "tuned" source: same 4691 size as movif_matched, but each
    # video contributes its HIGHEST-motion pair instead of a representative one.
    # Identical renderer/assets/lighting to the generic; only motion magnitude
    # differs -> an appearance-locked, size-matched motion intervention that
    # replaces the re-rendered Kubric-tuned line in Table 6.
    'movif_himotion': {
        'dataset_name': 'movi_f',
        'datapath': '/home/slow1/Data/movi_f/512x512/1.0.0',
        'strata': None,
        'movi_f_max_videos': 4691,
        'movi_f_pairs_per_video': 1,
        'movi_f_pair_select': 'max_motion',
    },
}

# Shared evaluation block (same for every cell). KITTI alpha variants are registered
# benchmark names; the alpha is supplied positionally via eval_alphas.
EVAL_BENCHMARKS = [
    'tss_a10', 'tss_a05', 'tss_a03',
    'tapvid_davis_s1', 'tapvid_davis_s2', 'tapvid_davis_s4', 'tapvid_davis_s8', 'tapvid_davis_s16',
    'kitti2012_a05', 'kitti2012_a03', 'kitti2012_a01',
    'kitti2015_a05', 'kitti2015_a03', 'kitti2015_a01',
]
EVAL_ALPHAS = [0.1, 0.05, 0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.03, 0.01, 0.05, 0.03, 0.01]


def _val_datasets():
    vd = {}
    for a in ('tss_a10', 'tss_a05', 'tss_a03'):
        vd[a] = {'normalize_images': True, 'reverse_flow': False}
    for s in (1, 2, 4, 8, 16):
        vd[f'tapvid_davis_s{s}'] = {'tapvid_stride': s, 'tapvid_frame_step': 5,
                                    'tapvid_min_pts': 1, 'reverse_flow': True, 'normalize_images': True}
    for b in ('kitti2012_a05', 'kitti2012_a03', 'kitti2012_a01',
              'kitti2015_a05', 'kitti2015_a03', 'kitti2015_a01'):
        vd[b] = {'split': 'val', 'normalize_images': True}
    return vd


def shared_eval():
    return {
        'datapath': './models/Datasets_CATs',
        'eval_benchmarks': list(EVAL_BENCHMARKS),
        'eval_alphas': list(EVAL_ALPHAS),
        'thres': 'img',
        'split_to_use_for_validation': 'test',
        'use_motion_aware': False,
        'min_motion_pixels': 5.0,
        'zero_threshold': 0.5,
        'prefetch_factor': 2,
        'val_batch_size': 2,
        'val_num_workers': 4,
        'kitti_root': '/home/slow1/Data/correspondence/kitti',
        'kitti_val_use_full_training': True,
        'tss_root': '/home/slow1/Data/correspondence/TSS_CVPR2016',
        'tapvid_davis_root': '/home/slow1/Data/tapvid/probe_cache',
        'val_datasets': _val_datasets(),
    }


def build_config(arch, base_file, source_name, src, stratum):
    cfg = yaml.safe_load(open(os.path.join(BASE_DIR, base_file)))
    if stratum is None:
        cell = f'{source_name}_{arch}'
        datapath = src['datapath']
    else:
        cell = f'{source_name}_{stratum}_{arch}'
        datapath = src['datapath'].format(s=stratum)
    # Source (the only thing that varies vs the MOVi-F baseline recipe).
    cfg['dataset']['datapath'] = datapath
    cfg['dataset']['split'] = 'train'
    if src.get('dataset_name') == 'movi_f':
        # Size-matched MOVi-F baseline: stream the TFDS shards but cap the unique
        # videos / pairs so the training set matches a rendered source's size.
        cfg['dataset']['dataset_name'] = 'movi_f'
        cfg['dataset']['negate_flow'] = True
        cfg['dataset']['reverse_flow'] = False
        cfg['dataset']['kubric_dir'] = f'{os.path.dirname(RC_REPO)}/kubric'  # RC kubric repo
        cfg['dataset']['movi_f_config'] = '512x512'
        cfg['dataset']['movi_f_shuffle_buffer'] = 16
        # Size cap is optional: present => size-matched subset (one frozen, motion-
        # representative pair per video); absent => full MOVi-F (all videos/pairs).
        if src.get('movi_f_max_videos') is not None:
            cfg['dataset']['movi_f_max_videos'] = int(src['movi_f_max_videos'])
            cfg['dataset']['movi_f_pairs_per_video'] = int(src['movi_f_pairs_per_video'])
        else:
            cfg['dataset']['movi_f_max_videos'] = None
            cfg['dataset']['movi_f_pairs_per_video'] = None
        # 'random' = representative (generic); 'max_motion' = large-motion (tuned)
        cfg['dataset']['movi_f_pair_select'] = src.get('movi_f_pair_select', 'random')
        cfg['dataset'].pop('occlusion_mask', None)
    else:
        cfg['dataset']['dataset_name'] = 'kubric_intervention'
        cfg['dataset']['negate_flow'] = True   # kubric flow is sign-flipped vs synthetic
        cfg['dataset']['occlusion_mask'] = True
        cfg['dataset']['src_tgt_flip'] = float(src.get('src_tgt_flip', 0.0))
    cfg.setdefault('paths', {})
    cfg['paths']['snapshots'] = f'{RC_REPO}/snapshots_kubric_tuned/{cell}'
    cfg['paths']['pretrained'] = cfg['paths'].get('pretrained', None)
    cfg['paths']['start_epoch'] = cfg['paths'].get('start_epoch', 0)
    cfg['paths']['save_epoch_checkpoints'] = False
    cfg['evaluation'] = shared_eval()
    return cell, cfg


def job_script(cell, arch, walltime):
    return f"""#!/bin/bash
#SBATCH --job-name=kt_{cell}
#SBATCH --output={RC_REPO}/slurm/kubric_tuned_jobs/logs/{cell}_%j.out
#SBATCH --error={RC_REPO}/slurm/kubric_tuned_jobs/logs/{cell}_%j.err
#SBATCH --time={walltime}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --qos=cs
#SBATCH --exclude=cs-1-2
cd {RC_REPO}
source "$HOME/.bashrc"; conda activate cuda
export PATH=/apps/slurm/latest/bin:$PATH
ulimit -c 0; export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/home/slow1/Data/hf_cache/datasets
mkdir -p {RC_REPO}/snapshots_kubric_tuned/{cell}
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST: {cell} start $(date)"
srun --ntasks=1 bash -c "ulimit -c 0; python3 -u train_lightning.py --config tap_vid_probe/configs/grid_kubric_tuned/{cell}.yaml"
ec=$?; echo "{cell} done $(date) exit=$ec"; exit $ec
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='tss_dense', choices=list(SOURCES))
    ap.add_argument('--archs', nargs='+', default=list(ARCHS), choices=list(ARCHS))
    args = ap.parse_args()

    src = SOURCES[args.source]
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(JOB_DIR, exist_ok=True)
    cells = []
    strata = src['strata'] if src['strata'] is not None else [None]
    for arch in args.archs:
        base_file, walltime = ARCHS[arch]
        for stratum in strata:
            cell, cfg = build_config(arch, base_file, args.source, src, stratum)
            with open(os.path.join(OUT_DIR, f'{cell}.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)
            with open(os.path.join(JOB_DIR, f'job_{cell}.sh'), 'w') as f:
                f.write(job_script(cell, arch, walltime))
            cells.append(cell)

    submit = os.path.join(JOB_DIR, f'submit_{args.source}.sh')
    with open(submit, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write(f'cd {RC_REPO}\n')
        f.write('export PATH=/apps/slurm/latest/bin:$PATH\n')
        f.write('mkdir -p slurm/kubric_tuned_jobs/logs\n')
        f.write(f'for j in slurm/kubric_tuned_jobs/job_{args.source}_*.sh; do\n')
        f.write('  echo "$(basename $j): $(sbatch --parsable "$j")"\n')
        f.write('done\n')
        f.write('echo "submitted. squeue -u $USER"\n')
    print(f'wrote {len(cells)} configs to {OUT_DIR}')
    print(f'wrote {len(cells)} jobs + {os.path.basename(submit)} to {JOB_DIR}')
    for c in cells:
        print('  ', c)


if __name__ == '__main__':
    main()
