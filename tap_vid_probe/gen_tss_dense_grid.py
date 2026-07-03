#!/usr/bin/env python3
"""Generate the per-stratum tss_dense GLU-Net+TT transfer grid (Track 2).

5 strata (tss_dense_s1..s5) -> 5 configs + 5 slurm jobs, mirroring the
grid_ladder/*_glunet_tt.yaml template (same arch/regime as the DAVIS ladder so
the y-axis is apples-to-apples). Each cell trains on one stratum and evaluates
TSS (a10/a05/a03), TAP-Vid-DAVIS (strides 1/2/4/8/16), and KITTI-2012/2015 @.05.

Baseline = snapshots_mm_rc/movif_glunet_tt (already on RC), so no MOVi-F cell here.
"""
import os

RC_REPO = '/home/slow1/Projects/OnlineSyntheticCorrespondence'
DATA_ROOT = '/home/slow1/Data/tss_dense'
STRATA = ['s1', 's2', 's3', 's4', 's5']

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'grid_tss_dense')
JOB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'slurm', 'tss_dense_jobs')


def config_yaml(stratum):
    cell = f'tss_dense_{stratum}_glunet_tt'
    return f"""dataset:
  dataset_name: kubric_intervention
  datapath: {DATA_ROOT}/tss_dense_{stratum}
  kubric_dir: /home/spencer/Projects/kubric
  movi_f_config: 512x512
  movi_f_shuffle_buffer: 16
  size:
  - 512
  - 512
  downsample_flow: null
  dense_kps_use_all: true
  max_kps: null
  normalize_images: true
  reverse_flow: false
  split: train
  debug: false
  verbose: false
  occlusion_mask: true
  negate_flow: true
  kubric_max_pairs: null
  mirror_flip: 0.0
model:
  type: glunet
  pretrained_backbone: true
  freeze: true
  glunet:
    model_name: resnet50
    local_window_size: 9
    decoder_dense_connect: false
training:
  epochs: 25
  steps_per_epoch: 400
  batch_size: 8
  check_val_every_n_epoch: 1
  eval_initial: true
  momentum: 0.9
  scheduler: cosine
  step: '[70, 80, 90]'
  step_gamma: 0.5
  augmentation: false
  mmd_every_n_epochs: 0
  min_flow_length: null
  max_flow_length: null
  n_threads: 2
  seed: 2021
  lr: 0.001
  lr_backbone: 1.0e-05
paths:
  snapshots: {RC_REPO}/snapshots_tss_dense/{cell}
  pretrained: null
  start_epoch: 0
  save_epoch_checkpoints: false
evaluation:
  eval_benchmarks:
  - tss_a10
  - tss_a05
  - tss_a03
  - tapvid_davis_s1
  - tapvid_davis_s2
  - tapvid_davis_s4
  - tapvid_davis_s8
  - tapvid_davis_s16
  - kitti2012_a05
  - kitti2012_a03
  - kitti2012_a01
  - kitti2015_a05
  - kitti2015_a03
  - kitti2015_a01
  eval_alphas:
  - 0.1
  - 0.05
  - 0.03
  - 0.05
  - 0.05
  - 0.05
  - 0.05
  - 0.05
  - 0.05
  - 0.03
  - 0.01
  - 0.05
  - 0.03
  - 0.01
  thres: img
  split_to_use_for_validation: test
  val_batch_size: 2
  val_num_workers: 2
  prefetch_factor: 2
  use_motion_aware: false
  min_motion_pixels: 5.0
  zero_threshold: 0.5
  datapath: ./models/Datasets_CATs
  tss_root: /home/slow1/Data/correspondence/TSS_CVPR2016
  kitti_root: /home/slow1/Data/correspondence/kitti
  kitti_val_use_full_training: true
  tapvid_davis_root: /home/slow1/Data/tapvid/probe_cache
  val_datasets:
    tss_a10:
      reverse_flow: false
      normalize_images: true
    tss_a05:
      reverse_flow: false
      normalize_images: true
    tss_a03:
      reverse_flow: false
      normalize_images: true
    tapvid_davis_s1:
      tapvid_stride: 1
      tapvid_frame_step: 5
      tapvid_min_pts: 1
      reverse_flow: true
      normalize_images: true
    tapvid_davis_s2:
      tapvid_stride: 2
      tapvid_frame_step: 5
      tapvid_min_pts: 1
      reverse_flow: true
      normalize_images: true
    tapvid_davis_s4:
      tapvid_stride: 4
      tapvid_frame_step: 5
      tapvid_min_pts: 1
      reverse_flow: true
      normalize_images: true
    tapvid_davis_s8:
      tapvid_stride: 8
      tapvid_frame_step: 5
      tapvid_min_pts: 1
      reverse_flow: true
      normalize_images: true
    tapvid_davis_s16:
      tapvid_stride: 16
      tapvid_frame_step: 5
      tapvid_min_pts: 1
      reverse_flow: true
      normalize_images: true
    kitti2012_a05:
      split: val
      normalize_images: true
    kitti2012_a03:
      split: val
      normalize_images: true
    kitti2012_a01:
      split: val
      normalize_images: true
    kitti2015_a05:
      split: val
      normalize_images: true
    kitti2015_a03:
      split: val
      normalize_images: true
    kitti2015_a01:
      split: val
      normalize_images: true
"""


def job_sh(stratum):
    cell = f'tss_dense_{stratum}_glunet_tt'
    return f"""#!/bin/bash
#SBATCH --job-name=tssd_{stratum}_gtt
#SBATCH --output={RC_REPO}/slurm/tss_dense_jobs/logs/{cell}_%j.out
#SBATCH --error={RC_REPO}/slurm/tss_dense_jobs/logs/{cell}_%j.err
#SBATCH --time=06:00:00
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
mkdir -p {RC_REPO}/snapshots_tss_dense/{cell}
echo "Job $SLURM_JOB_ID on $SLURM_NODELIST: {cell} start $(date)"
srun --ntasks=1 bash -c "ulimit -c 0; python3 -u train_lightning.py --config tap_vid_probe/configs/grid_tss_dense/{cell}.yaml"
ec=$?; echo "{cell} done $(date) exit=$ec"; exit $ec
"""


def submit_sh():
    return f"""#!/bin/bash
cd {RC_REPO}
export PATH=/apps/slurm/latest/bin:$PATH
mkdir -p slurm/tss_dense_jobs/logs
for j in slurm/tss_dense_jobs/job_tss_dense_*_glunet_tt.sh; do
  echo "$(basename $j): $(sbatch --parsable "$j")"
done
echo "submitted. squeue -u $USER"
"""


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(JOB_DIR, exist_ok=True)
    for s in STRATA:
        cell = f'tss_dense_{s}_glunet_tt'
        with open(os.path.join(CONFIG_DIR, f'{cell}.yaml'), 'w') as f:
            f.write(config_yaml(s))
        with open(os.path.join(JOB_DIR, f'job_{cell}.sh'), 'w') as f:
            f.write(job_sh(s))
    with open(os.path.join(JOB_DIR, 'submit_tss_dense.sh'), 'w') as f:
        f.write(submit_sh())
    print(f'wrote {len(STRATA)} configs to {CONFIG_DIR}')
    print(f'wrote {len(STRATA)} jobs + submit_tss_dense.sh to {JOB_DIR}')


if __name__ == '__main__':
    main()
