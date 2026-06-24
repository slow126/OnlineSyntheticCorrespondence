#!/usr/bin/env bash
# Two causal ladders, 12 runs, CATs++ (batch2, 50ep x 400 steps, ~2.6h each):
#   PRECISION (mirror off-target): FF should DROP with f; TT control should be FLAT.
#   RECALL (depleted coverage, TT, size-matched 2500): transfer should RISE with dolly.
# GPU0 = 6 runs, GPU1 = 6 runs. Detached, niced.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
CFG=src/configs/lightning/precision_ladder
LOG=/mnt/nvme_1tb_a/snapshots/precision_ladder/logs; mkdir -p "$LOG"
run(){ echo "[$(date +%H:%M)] START $1 (gpu $2)"; OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$2 nice -n 10 $PY -u train_lightning.py --config "$CFG/$1.yaml" > "$LOG/$1.log" 2>&1; echo "[$(date +%H:%M)] DONE $1 rc=$?"; }
GPU0=(precision_f000_catspp_ff precision_f050_catspp_ff precision_f025_catspp_ff precision_f010_catspp_ff precision_f005_catspp_ff coverage_d05_catspp_tt)
GPU1=(precision_f000_catspp_tt precision_f050_catspp_tt precision_f025_catspp_tt coverage_recovered_catspp_tt coverage_d15_catspp_tt coverage_d10_catspp_tt)
( for c in "${GPU0[@]}"; do run "$c" 0; done ) &
( for c in "${GPU1[@]}"; do run "$c" 1; done ) &
wait
echo "[$(date +%H:%M)] ALL 12 LADDER RUNS DONE"
