#!/usr/bin/env bash
# When both 4-object renders finish (ns4 near-static, mm4 motion-matched, 1000 each),
# train CATs++ TF+TT on each and report TSS PCK vs MOVi-F-1000 / tss_v2-1000.
# Isolates motion magnitude with composition held clean (4 objects).
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
LOG=/mnt/nvme_1tb_a/snapshots/ns_mm_quicktrain
mkdir -p "$LOG"; WLOG="$LOG/watcher.log"
log(){ echo "[$(date +%F_%H:%M:%S)] $*" | tee -a "$WLOG"; }

log "waiting for both renders (ns4 + mm4, 1000 each)"
for i in $(seq 1 300); do
  grep -q ALLDONE /tmp/ns_mm_render.progress 2>/dev/null && break
  sleep 60
done
log "renders done: $(grep DONE /tmp/ns_mm_render.progress 2>/dev/null | tr '\n' ' ')"

run_wave () {  # $1=variant
  local v=$1
  log "training $v: TF(gpu0) + TT(gpu1)"
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 5 \
    $PY -u train_lightning.py --config src/configs/lightning/tss_${v}_catspp_tf.yaml > "$LOG/${v}_tf.log" 2>&1 &
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nice -n 5 \
    $PY -u train_lightning.py --config src/configs/lightning/tss_${v}_catspp_tt.yaml > "$LOG/${v}_tt.log" 2>&1 &
  wait
}
run_wave ns4
run_wave mm4
log "ALL TRAINING DONE"
for x in ns4_tf ns4_tt mm4_tf mm4_tt; do
  log "$x TSS PCK: $(grep -oE 'tss: PCK=[0-9.]+%' "$LOG/$x.log" 2>/dev/null | tr '\n' ' ')"
done
log "================ COMPARISON (TSS PCK, all 1000 scenes, CATs++) ================"
log "MOVi-F-1000  (clean comp, ~2px motion):   TF 57.8 / TT 54.1"
log "tss_v2-1000  (CLUTTERED 14-obj, ~50px):   TF 28.7 / TT 38.2"
log "ns4-1000     (clean 4-obj, near-static ~7px): see ns4_tf/ns4_tt above"
log "mm4-1000     (clean 4-obj, matched ~43px):    see mm4_tf/mm4_tt above"
log "Read: if ns4/mm4 >> tss_v2 -> COMPOSITION was the lever. ns4 vs mm4 -> does motion magnitude matter once composition is clean."
