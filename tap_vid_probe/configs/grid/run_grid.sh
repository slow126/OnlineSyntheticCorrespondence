#!/usr/bin/env bash
# Sequential local grid runner. Usage: bash run_grid.sh <gpu>
set -u
REPO=/home/spencer/Projects/OnlineSyntheticCorrespondence
GPU="${1:-0}"
cd "$REPO"
echo "[$(date +%T)] START movif_catspp_ff"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_catspp_ff.yaml > tap_vid_probe/logs/grid_movif_catspp_ff.log 2>&1
echo "[$(date +%T)] START movif_catspp_tf"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_catspp_tf.yaml > tap_vid_probe/logs/grid_movif_catspp_tf.log 2>&1
echo "[$(date +%T)] START movif_catspp_tt"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_catspp_tt.yaml > tap_vid_probe/logs/grid_movif_catspp_tt.log 2>&1
echo "[$(date +%T)] START movif_flowformer_ff"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_flowformer_ff.yaml > tap_vid_probe/logs/grid_movif_flowformer_ff.log 2>&1
echo "[$(date +%T)] START movif_flowformer_tf"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_flowformer_tf.yaml > tap_vid_probe/logs/grid_movif_flowformer_tf.log 2>&1
echo "[$(date +%T)] START movif_flowformer_tt"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_flowformer_tt.yaml > tap_vid_probe/logs/grid_movif_flowformer_tt.log 2>&1
echo "[$(date +%T)] START movif_glunet_ff"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_glunet_ff.yaml > tap_vid_probe/logs/grid_movif_glunet_ff.log 2>&1
echo "[$(date +%T)] START movif_glunet_tf"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_glunet_tf.yaml > tap_vid_probe/logs/grid_movif_glunet_tf.log 2>&1
echo "[$(date +%T)] START movif_glunet_tt"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_glunet_tt.yaml > tap_vid_probe/logs/grid_movif_glunet_tt.log 2>&1
echo "[$(date +%T)] START movif_raft_ff"
CUDA_VISIBLE_DEVICES=$GPU python -u train_lightning.py --config tap_vid_probe/configs/grid/movif_raft_ff.yaml > tap_vid_probe/logs/grid_movif_raft_ff.log 2>&1
echo "grid done"
