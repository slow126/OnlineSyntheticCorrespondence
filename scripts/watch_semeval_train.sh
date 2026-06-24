#!/usr/bin/env bash
# Re-train all 4 sources x {TF,TT} on the SAME 1000-scene datasets, but eval the
# FULL semantic suite (TSS, PF-Pascal, PF-Willow, SPair). 4 waves of 2 (one/GPU).
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
LOG=/mnt/nvme_1tb_a/snapshots/semeval_quicktrain
mkdir -p "$LOG"; WLOG="$LOG/watcher.log"
log(){ echo "[$(date +%F_%H:%M:%S)] $*" | tee -a "$WLOG"; }

run_pair(){  # $1=gpu0 cfg, $2=gpu1 cfg
  log "wave: $1 (gpu0) + $2 (gpu1)"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 5 \
    $PY -u train_lightning.py --config src/configs/lightning/$1.yaml > "$LOG/$1.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nice -n 5 \
    $PY -u train_lightning.py --config src/configs/lightning/$2.yaml > "$LOG/$2.log" 2>&1 &
  wait
}
log "starting full-semantic-suite retrain (8 runs, 4 waves)"
run_pair semeval_ns4_tf   semeval_ns4_tt
run_pair semeval_mm4_tf   semeval_mm4_tt
run_pair semeval_movif_tf semeval_movif_tt
run_pair semeval_tssv2_tf semeval_tssv2_tt
log "ALL SEMEVAL TRAINING DONE"
$PY scripts/harvest_semeval.py 2>&1 | tee -a "$WLOG"
