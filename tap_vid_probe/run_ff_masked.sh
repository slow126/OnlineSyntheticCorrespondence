#!/bin/bash
# Launch FlowFormer masked (gpu0) + unmasked control (gpu1) on tss_matte_masked_1000.
# Run AFTER the GLU-Net overnight runs finish (GPUs free).
#   bash tap_vid_probe/run_ff_masked.sh         # launch now (assumes GPUs free)
#   bash tap_vid_probe/run_ff_masked.sh wait     # block until GLU-Net runs finish, then launch
set -u
REPO=/home/spencer/Projects/OnlineSyntheticCorrespondence
LOG="$REPO/tap_vid_probe/logs/kubvar"; mkdir -p "$LOG"
cd "$REPO" || exit 1
if [ "${1:-}" = "wait" ]; then
  echo "[ff $(date '+%m-%d %H:%M:%S')] waiting for GLU-Net runs to finish..."
  while pgrep -f arm_masked_overnight.sh >/dev/null 2>&1; do sleep 120; done
  while pgrep -f "train_lightning.py.*glunet" >/dev/null 2>&1; do sleep 60; done
  sleep 30
fi
echo "[ff $(date '+%m-%d %H:%M:%S')] launching FlowFormer masked + control"
( ulimit -c 0; CUDA_VISIBLE_DEVICES=0 python -u train_lightning.py \
    --config tap_vid_probe/configs/kubvar/kv_matte_masked_flowformer_tf.yaml ) \
    > "$LOG/kv_matte_masked_flowformer_tf.log" 2>&1 &
echo "[ff] masked FF PID=$! (gpu0) -> $LOG/kv_matte_masked_flowformer_tf.log"
sleep 25
( ulimit -c 0; CUDA_VISIBLE_DEVICES=1 python -u train_lightning.py \
    --config tap_vid_probe/configs/kubvar/kv_matte_ctrl_flowformer_tf.yaml ) \
    > "$LOG/kv_matte_ctrl_flowformer_tf.log" 2>&1 &
echo "[ff] control FF PID=$! (gpu1) -> $LOG/kv_matte_ctrl_flowformer_tf.log"
wait
echo "[ff $(date '+%m-%d %H:%M:%S')] FlowFormer runs finished"
