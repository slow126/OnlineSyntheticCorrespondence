#!/bin/bash
# Render a MOTION-DIVERSE kubric dataset: turns ON the per-scene variance "ladder"
# (object translation/rotation + camera distance/focal/azimuth/elevation jitter)
# so the flow targets have variance -> dense flow models can no longer regress to
# the mean flow field. Tests the fix for the GLU-Net/FlowFormer kubric collapse.
#
# Usage:
#   bash tap_vid_probe/render_diverse_kubric.sh 5     # cheap theta check (5 scenes)
#   bash tap_vid_probe/render_diverse_kubric.sh 300   # test training set
#   GPU=1 bash tap_vid_probe/render_diverse_kubric.sh 1000
set -e
N=${1:-300}
GPU=${GPU:-0}
THETA=/home/spencer/Projects/OnlineSyntheticCorrespondence/tap_vid_probe/kubric_motion_diverse_theta.json
DS=/mnt/nvme_1tb_a/kubric_interventions/datasets/tss_zoom1obj_diverse_${N}
mkdir -p "$DS"
cp "$THETA" "$DS/theta.json"; cp "$THETA" "$DS/_render_theta.json"
echo "[render] $N scenes -> $DS/train  (gpu $GPU)"
cd /home/spencer/Projects/kubric
CUDA_VISIBLE_DEVICES="$GPU" python interface/render_intervention_scenelets.py \
  --theta-json "$THETA" \
  --output-dir "$DS/train" \
  --n-pairs "$N" --seed 0 --gpu-backend CUDA
echo "[done] $(ls "$DS/train" 2>/dev/null | grep -c scene_) scenes in $DS"
