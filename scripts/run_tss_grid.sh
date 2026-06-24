#!/usr/bin/env bash
# Poor-man's 2-GPU queue for the TSS design grid: tss_v1 (tuned) vs MOVi-F,
# x {CATs++, GLU-Net, FlowFormer}, all pretrained-frozen, eval on semantic suite.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
CFG=src/configs/lightning
LOG=/mnt/nvme_1tb_a/snapshots/tssgrid_logs; mkdir -p "$LOG"
QUEUE="$LOG/queue.txt"; LOCK="$LOG/queue.lock"
cat > "$QUEUE" <<'JOBS'
tss_v1_catspp_tt_tssgrid
movif_catspp_tt_tssgrid
tss_v1_glunet_tt_tssgrid
movif_glunet_tt_tssgrid
tss_v1_flowformer_tt_tssgrid
movif_flowformer_tt_tssgrid
JOBS
pop(){ ( flock 9; j=$(head -n1 "$QUEUE" 2>/dev/null); [ -n "$j" ] && sed -i '1d' "$QUEUE"; printf '%s' "$j"; ) 9>"$LOCK"; }
worker(){ local g=$1 j; while j=$(pop); [ -n "$j" ]; do
  echo "[$(date +%F_%H:%M)] gpu$g START $j" >> "$LOG/driver.log"
  CUDA_VISIBLE_DEVICES=$g OMP_NUM_THREADS=8 nice -n 10 $PY -u train_lightning.py --config "$CFG/$j.yaml" > "$LOG/$j.log" 2>&1
  echo "[$(date +%F_%H:%M)] gpu$g DONE  $j rc=$?" >> "$LOG/driver.log"; done; }
worker 0 & worker 1 & wait
echo "[$(date +%F_%H:%M)] TSS GRID DONE" >> "$LOG/driver.log"
