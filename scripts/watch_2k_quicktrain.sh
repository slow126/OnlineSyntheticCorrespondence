#!/usr/bin/env bash
# At ~2000 rendered scenes: snapshot the complete scenes and run a quick CATs++ TF
# training (eval TSS) to sanity-check the tss_v2 source is learning, WITHOUT
# disturbing the still-running render (trains on a copy, GPU + few CPU workers).
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
SRC=/mnt/nvme_1tb_a/kubric_interventions/datasets/tss_v2_matched_5000/train
SNAP=/mnt/nvme_1tb_a/kubric_interventions/datasets/tss_v2_partial2k
LOG=/mnt/nvme_1tb_a/snapshots/tss_v2_quicktrain
mkdir -p "$LOG"
WLOG="$LOG/watcher.log"
log(){ echo "[$(date +%F_%H:%M:%S)] $*" | tee -a "$WLOG"; }
TARGET=1000

log "watcher started (pid $$); waiting for $TARGET complete scenes"
while true; do
  n=$(ls -d "$SRC"/scene_* 2>/dev/null | wc -l)
  [ "$n" -ge "$TARGET" ] && { log "$n scenes present — proceeding"; break; }
  sleep 60
done

# snapshot only fully-complete scenes (avoid copying one mid-write)
log "copying complete scenes -> $SNAP/train"
mkdir -p "$SNAP/train"
copied=0
for d in "$SRC"/scene_*/; do
  s=$(basename "$d")
  [ -d "$SNAP/train/$s" ] && continue
  if [ -f "$d/rgba_00000.png" ] && [ -f "$d/rgba_00001.png" ] \
     && [ -f "$d/forward_flow_00000.png" ] && [ -f "$d/data_ranges.json" ]; then
    cp -r "$d" "$SNAP/train/$s" && copied=$((copied+1))
  fi
done
have=$(ls -d "$SNAP"/train/scene_* 2>/dev/null | wc -l)
log "snapshot ready: $have complete scenes (copied $copied this run)"
if [ "$have" -lt 1000 ]; then log "ABORT: only $have scenes copied"; exit 1; fi

# quick sanity training: CATs++ TF (GPU0) + TT (GPU1) in PARALLEL on the snapshot
log "launching CATs++ TF (GPU0) + TT (GPU1) in parallel (eval TSS)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 5 \
  $PY -u train_lightning.py --config src/configs/lightning/tss_v2_quicktrain_catspp_tf.yaml \
  > "$LOG/train_tf.log" 2>&1 &
PTF=$!
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nice -n 5 \
  $PY -u train_lightning.py --config src/configs/lightning/tss_v2_quicktrain_catspp_tt.yaml \
  > "$LOG/train_tt.log" 2>&1 &
PTT=$!
wait $PTF; rtf=$?
wait $PTT; rtt=$?
log "both finished (TF rc=$rtf, TT rc=$rtt)"
log "TF (trained enc) TSS PCK trend: $(grep -oE 'tss: PCK=[0-9.]+%' "$LOG/train_tf.log" | tr '\n' ' ')"
log "TT (frozen  enc) TSS PCK trend: $(grep -oE 'tss: PCK=[0-9.]+%' "$LOG/train_tt.log" | tr '\n' ' ')"
log "TF best: $(grep -E 'New best tss PCK' "$LOG/train_tf.log" | tail -1)"
log "TT best: $(grep -E 'New best tss PCK' "$LOG/train_tt.log" | tail -1)"
log "(ref: MOVi-F CATs++ TSS = TF 58.6 / TT 55.7; this is a partial-data sanity run. Climbing PCK = learning; flat/zero = crash.)"
