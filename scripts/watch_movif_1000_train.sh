#!/usr/bin/env bash
# When the MOVi-F 1000-example extraction finishes, train CATs++ TF (GPU0) + TT
# (GPU1) on it with the SAME recipe as the tss_v2 quick runs — the object-diversity
# control. Logs the TSS PCK trends + the verdict.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
PY=/home/spencer/miniconda3/envs/cuda/bin/python
EXT=/mnt/nvme_1tb_a/kubric_interventions/datasets/movif_1000_extracted/train
LOG=/mnt/nvme_1tb_a/snapshots/movif_1000_quicktrain
mkdir -p "$LOG"
WLOG="$LOG/watcher.log"
log(){ echo "[$(date +%F_%H:%M:%S)] $*" | tee -a "$WLOG"; }

log "waiting for MOVi-F extraction (~1010 scenes)"
for i in $(seq 1 80); do
  n=$(ls -d "$EXT"/scene_* 2>/dev/null | wc -l)
  [ "$n" -ge 1005 ] && { log "extraction ready: $n scenes"; break; }
  grep -q "^DONE" /tmp/extract_movif.log 2>/dev/null && { log "extraction DONE ($n scenes)"; break; }
  sleep 15
done

rm -rf /mnt/nvme_1tb_a/snapshots/movif_1000_tf /mnt/nvme_1tb_a/snapshots/movif_1000_tt
log "launching MOVi-F-1000 CATs++ TF (GPU0) + TT (GPU1) in parallel"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 nice -n 5 \
  $PY -u train_lightning.py --config src/configs/lightning/movif_1000_catspp_tf.yaml > "$LOG/train_tf.log" 2>&1 &
PTF=$!
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 nice -n 5 \
  $PY -u train_lightning.py --config src/configs/lightning/movif_1000_catspp_tt.yaml > "$LOG/train_tt.log" 2>&1 &
PTT=$!
wait $PTF; rtf=$?
wait $PTT; rtt=$?
log "both finished (TF rc=$rtf, TT rc=$rtt)"
log "MOVi-F-1000 TF TSS PCK: $(grep -oE 'tss: PCK=[0-9.]+%' "$LOG/train_tf.log" | tr '\n' ' ')"
log "MOVi-F-1000 TT TSS PCK: $(grep -oE 'tss: PCK=[0-9.]+%' "$LOG/train_tt.log" | tr '\n' ' ')"
log "--- COMPARISON ---"
log "tss_v2-1000:   TF peak 28.7 / TT peak 38.2"
log "MOVi-F-full:   TF 58.6     / TT 55.7"
log "Verdict: MOVi-F-1000 ~ tss_v2-1000  => advantage was DIVERSITY/SIZE."
log "         MOVi-F-1000 >> tss_v2-1000 => advantage is CONTENT (per-scene quality/motion)."
