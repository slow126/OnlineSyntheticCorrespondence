#!/bin/bash
# Table-8 grid (14 cells). Eval = TSS@{0.10,0.05,0.03} + TAP-Vid strides {1,2,4,8,16}@0.05.
# GPU1 (free now): synth cells, DEFAULT-synth GLU-Net anchor FIRST (reproduces the June-8 ~62%).
# GPU0 (waits for coverage FAISS job): 8 fast diverse/movi-f cells, key cells (catspp_tt, glunet_tf) first.
cd /home/spencer/Projects/OnlineSyntheticCorrespondence
LOGDIR=tap_vid_probe/logs/grid8
mkdir -p $LOGDIR
GPU1_LIST="g8_synthdef_glunet_tt g8_synthdef_glunet_tf g8_synth_glunet_tt g8_synth_glunet_tf g8_synth_catspp_tt g8_synth_catspp_tf"
GPU0_LIST="g8_diverse_catspp_tt g8_movif_catspp_tt g8_diverse_glunet_tf g8_movif_glunet_tf g8_diverse_catspp_tf g8_movif_catspp_tf g8_diverse_glunet_tt g8_movif_glunet_tt"
WAIT_PID_GPU0=375715   # coverage FAISS job; let it finish before grabbing GPU0

run_queue(){
  local gpu=$1 wpid=$2; shift 2
  if [ -n "$wpid" ] && [ "$wpid" != "0" ]; then
    while kill -0 "$wpid" 2>/dev/null; do sleep 15; done
    echo "[$(date +%H:%M)] GPU$gpu: pid $wpid finished, starting queue" >> $LOGDIR/_queue.log
  fi
  for name in "$@"; do
    echo "[$(date +%H:%M)] GPU$gpu START $name" >> $LOGDIR/_queue.log
    ( ulimit -c 0; CUDA_VISIBLE_DEVICES=$gpu python -u train_lightning.py \
        --config tap_vid_probe/configs/grid8/$name.yaml > $LOGDIR/$name.log 2>&1 )
    echo "[$(date +%H:%M)] GPU$gpu END   $name exit=$?" >> $LOGDIR/_queue.log
  done
  echo "[$(date +%H:%M)] GPU$gpu QUEUE COMPLETE" >> $LOGDIR/_queue.log
}

run_queue 1 0            $GPU1_LIST &
run_queue 0 $WAIT_PID_GPU0 $GPU0_LIST &
wait
echo "[$(date +%H:%M)] ALL COMPLETE" >> $LOGDIR/_queue.log
