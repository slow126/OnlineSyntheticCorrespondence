#!/bin/bash
# 12-run grid: {movif, diverse} x {catspp,glunet,flowformer} x {tt,tf}.
# CATs -> GLU-Net -> FlowFormer(last). Each pair = movif(cuda0)+diverse(cuda1), wait, next.
set -u
REPO=/home/spencer/Projects/OnlineSyntheticCorrespondence
CFG=$REPO/tap_vid_probe/configs/grid8; LOG=$REPO/tap_vid_probe/logs/grid8
mkdir -p "$LOG"; cd "$REPO" || exit 1
PAIRS=(
 "movif_catspp_tt diverse_catspp_tt"
 "movif_catspp_tf diverse_catspp_tf"
 "movif_glunet_tt diverse_glunet_tt"
 "movif_glunet_tf diverse_glunet_tf"
 "movif_flowformer_tt diverse_flowformer_tt"
 "movif_flowformer_tf diverse_flowformer_tf"
)
for pair in "${PAIRS[@]}"; do
  set -- $pair; a=$1; b=$2
  echo "[grid8 $(date '+%m-%d %H:%M:%S')] START pair: $a (cuda0) + $b (cuda1)"
  ( ulimit -c 0; CUDA_VISIBLE_DEVICES=0 python -u train_lightning.py --config "$CFG/g8_$a.yaml" > "$LOG/g8_$a.log" 2>&1 ) & P0=$!
  ( ulimit -c 0; CUDA_VISIBLE_DEVICES=1 python -u train_lightning.py --config "$CFG/g8_$b.yaml" > "$LOG/g8_$b.log" 2>&1 ) & P1=$!
  wait $P0 $P1
  echo "[grid8 $(date '+%m-%d %H:%M:%S')] DONE pair: $a, $b"
done
echo "[grid8 $(date '+%m-%d %H:%M:%S')] ALL 12 RUNS DONE"
