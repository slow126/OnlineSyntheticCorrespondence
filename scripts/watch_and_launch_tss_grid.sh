#!/usr/bin/env bash
# Watcher: wait for the tss_v1_2500 materialization to finish, then launch the
# 6-cell TSS design grid (tss_v1 vs MOVi-F x {CATs++,GLU-Net,FlowFormer}, TT).
# Designed to run unattended overnight via nohup. Idempotent-ish: it only fires
# the grid once.
set -uo pipefail
cd /home/spencer/Projects/OnlineSyntheticCorrespondence

DATA=/mnt/nvme_1tb_a/kubric_interventions/datasets/tss_v1_2500/train
RENDER_PID=1940978
NEED=2500
LOG=/mnt/nvme_1tb_a/snapshots/tssgrid_logs
mkdir -p "$LOG"
WLOG="$LOG/watcher.log"

log(){ echo "[$(date +%F_%H:%M:%S)] $*" | tee -a "$WLOG"; }

log "watcher started (pid $$); waiting for $NEED scenes in $DATA"
while true; do
  n=$(ls "$DATA" 2>/dev/null | grep -c '^scene_')
  alive=$(ps -p "$RENDER_PID" -o pid= --no-headers 2>/dev/null || true)
  if [ "$n" -ge "$NEED" ]; then
    log "$n/$NEED scenes present — materialization complete"; break
  fi
  if [ -z "$alive" ]; then
    if [ "$n" -ge 2490 ]; then
      log "render process gone, $n/$NEED scenes (>=2490) — proceeding"; break
    else
      log "WARNING: render process gone but only $n/$NEED scenes — proceeding anyway with what we have"; break
    fi
  fi
  sleep 60
done

# let the renderer flush its last scene + release the box
sleep 45

# sanity: confirm the tss_v1 data dir is non-trivial before burning GPU hours
final_n=$(ls "$DATA" 2>/dev/null | grep -c '^scene_')
if [ "$final_n" -lt 2000 ]; then
  log "ABORT: only $final_n scenes — refusing to launch the grid on a truncated source"
  exit 1
fi
log "launching TSS grid on $final_n scenes"
bash scripts/run_tss_grid.sh
log "run_tss_grid.sh returned (rc=$?). TSS grid driver finished."
log "harvest: validation_results.csv under /mnt/nvme_1tb_a/snapshots/tssgrid_* ; compare tss_v1 vs movif per architecture."
