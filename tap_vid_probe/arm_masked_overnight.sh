#!/bin/bash
# Overnight arming script: wait for the masked render to reach 1000 scenes, stop the
# render cleanly, then launch BOTH GLU-Net runs -- masked (gpu0) + unmasked control
# (gpu1) -- at a long horizon with dense (per-epoch) eval. Fully detached; needs no
# attached agent. Logs to tap_vid_probe/logs/kubvar/.
set -u
OUT=/mnt/nvme_1tb_a/kubric_interventions/datasets/tss_matte_masked_1000
REPO=/home/spencer/Projects/OnlineSyntheticCorrespondence
LOG="$REPO/tap_vid_probe/logs/kubvar"
mkdir -p "$LOG"
echo "[arm $(date '+%m-%d %H:%M:%S')] waiting for 1000 scenes in $OUT/train"
while true; do
  n=$(ls "$OUT/train" 2>/dev/null | grep -c scene_)
  [ "$n" -ge 1000 ] && { echo "[arm] reached $n scenes"; break; }
  pgrep -f render_kubric_dataset >/dev/null 2>&1 || { echo "[arm] render process exited at $n scenes -- proceeding"; break; }
  sleep 30
done
# make sure no straggler render workers contend with training
pkill -f render_kubric_dataset 2>/dev/null
sleep 8
FINAL=$(ls "$OUT/train" 2>/dev/null | grep -c scene_)
echo "[arm $(date '+%m-%d %H:%M:%S')] launching training on $FINAL scenes"
cd "$REPO" || exit 1
( ulimit -c 0; CUDA_VISIBLE_DEVICES=0 python -u train_lightning.py \
    --config tap_vid_probe/configs/kubvar/kv_matte_masked_glunet_tf.yaml ) \
    > "$LOG/kv_matte_masked_glunet_tf.log" 2>&1 &
PIDM=$!
echo "[arm] masked GLU-Net PID=$PIDM on gpu0 -> $LOG/kv_matte_masked_glunet_tf.log"
sleep 25  # stagger init/eval-cache so the two don't collide on startup
( ulimit -c 0; CUDA_VISIBLE_DEVICES=1 python -u train_lightning.py \
    --config tap_vid_probe/configs/kubvar/kv_matte_ctrl_glunet_tf.yaml ) \
    > "$LOG/kv_matte_ctrl_glunet_tf.log" 2>&1 &
PIDC=$!
echo "[arm] control GLU-Net PID=$PIDC on gpu1 -> $LOG/kv_matte_ctrl_glunet_tf.log"
wait
echo "[arm $(date '+%m-%d %H:%M:%S')] both training runs finished"
