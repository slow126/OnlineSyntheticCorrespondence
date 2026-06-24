#!/usr/bin/env bash
# Launch the TAP-Vid probe: 4 MOVi-F full-convergence runs across 2 GPUs.
# Two lanes, sequential within a lane (4 jobs / 2 GPUs):
#   lane0 (GPU0): catspp_tf  -> glunet_tf
#   lane1 (GPU1): flowformer_tf -> raft_ff
#
# Usage:
#   bash tap_vid_probe/run_probe.sh lane0   # just GPU0 lane
#   bash tap_vid_probe/run_probe.sh lane1   # just GPU1 lane
#   bash tap_vid_probe/run_probe.sh all     # both lanes in parallel (default)
#   bash tap_vid_probe/run_probe.sh one movif_catspp_tf 0   # single config on a given GPU
set -u
REPO=/home/spencer/Projects/OnlineSyntheticCorrespondence
CFG="$REPO/tap_vid_probe/configs"
LOG="$REPO/tap_vid_probe/logs"
cd "$REPO"
mkdir -p "$LOG"

run() {  # gpu config_basename
  local gpu=$1 name=$2
  echo "[$(date '+%F %T')] START $name on GPU$gpu" | tee -a "$LOG/probe_driver.log"
  CUDA_VISIBLE_DEVICES="$gpu" TAPVID_PROBE_DEBUG=0 \
    python -u train_lightning.py --config "$CFG/$name.yaml" > "$LOG/$name.log" 2>&1
  echo "[$(date '+%F %T')] DONE  $name on GPU$gpu (exit $?)" | tee -a "$LOG/probe_driver.log"
}

lane0() { run 0 movif_catspp_tf; run 0 movif_glunet_tf; }
lane1() { run 1 movif_flowformer_tf; run 1 movif_raft_ff; }

case "${1:-all}" in
  lane0) lane0 ;;
  lane1) lane1 ;;
  one)   run "${3:-0}" "${2:?need config basename}" ;;
  all)   lane0 & lane1 & wait ;;
  *)     echo "unknown mode: $1"; exit 1 ;;
esac
