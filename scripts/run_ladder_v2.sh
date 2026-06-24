#!/usr/bin/env bash
# Tier 1: GLU-Net precision ladder (FF f0/.1/.25/.5 + TT f0/.5) -> arch-specificity of off-target.
# Tier 2: CATs++ FF recall ladder (recovered/d15/d10/d05) + re-run crashed CATs++ TT recall (d05/d10).
# Fixed loader (skips corrupt scenes). Detached, niced. ~13h/GPU.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
CFG=src/configs/lightning/precision_ladder
LOG=/mnt/nvme_1tb_a/snapshots/precision_ladder/logs; mkdir -p "$LOG"
run(){ echo "[$(date +%H:%M)] START $1 (gpu $2)"; OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$2 nice -n 10 $PY -u train_lightning.py --config "$CFG/$1.yaml" > "$LOG/$1.log" 2>&1; echo "[$(date +%H:%M)] DONE $1 rc=$?"; }
GPU0=(glunet_precision_f000_ff glunet_precision_f050_ff glunet_precision_f025_ff glunet_precision_f010_ff glunet_precision_f000_tt glunet_precision_f050_tt coverage_recovered_catspp_ff)
GPU1=(coverage_d05_catspp_ff coverage_d15_catspp_ff coverage_d10_catspp_ff coverage_d05_catspp_tt coverage_d10_catspp_tt)
( for c in "${GPU0[@]}"; do run "$c" 0; done ) &
( for c in "${GPU1[@]}"; do run "$c" 1; done ) &
wait
echo "[$(date +%H:%M)] ALL LADDER-V2 RUNS DONE"
