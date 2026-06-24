#!/usr/bin/env bash
# Poor-man's 2-GPU queue for the 4 FlowFormer intervention runs (Table 8):
# tuned (KITTI-recovered) vs MOVi-F, x {scratch FF, pretrained-frozen TT}.
# Two workers (GPU 0 and GPU 1) pull jobs from a shared queue until empty, so a
# GPU that finishes early steals the next job (proper work-stealing).
# GPU 1 coexists with the FlyingThings TPE search (~1.8GB); FlowFormer ~16.7GB fits.
# Effective batch 8 = batch_size 2 x accumulate_grad_batches 4.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
CFG=src/configs/lightning
LOG=/mnt/nvme_1tb_a/snapshots/flowformer_interv_logs; mkdir -p "$LOG"
QUEUE="$LOG/queue.txt"
LOCK="$LOG/queue.lock"

cat > "$QUEUE" <<'EOF'
kitti_recov_flowformer_ff_b8
movif_flowformer_ff_b8
kitti_recov_flowformer_tt_b8
movif_flowformer_tt_b8
EOF

pop_job(){
  ( flock 9
    job=$(head -n1 "$QUEUE" 2>/dev/null)
    [ -n "$job" ] && sed -i '1d' "$QUEUE"
    printf '%s' "$job"
  ) 9>"$LOCK"
}

worker(){   # $1 = gpu id
  local gpu=$1 job
  while job=$(pop_job); [ -n "$job" ]; do
    echo "[$(date +%F_%H:%M)] gpu$gpu START $job" >> "$LOG/driver.log"
    CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 nice -n 10 \
      $PY -u train_lightning.py --config "$CFG/$job.yaml" > "$LOG/$job.log" 2>&1
    echo "[$(date +%F_%H:%M)] gpu$gpu DONE  $job rc=$?" >> "$LOG/driver.log"
  done
}

worker 0 &
worker 1 &
wait
echo "[$(date +%F_%H:%M)] ALL FLOWFORMER INTERVENTION RUNS DONE" >> "$LOG/driver.log"
