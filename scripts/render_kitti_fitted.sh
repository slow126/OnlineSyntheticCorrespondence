#!/usr/bin/env bash
# Track 1: re-render the KITTI-recovered Kubric source with per-scene dolly
# spread (theta_kitti_fitted_gso_hq.json) so it covers KITTI's motion-magnitude
# distribution, incl. the small-displacement tail that @.01 needs.
# Mirrors materialize_kitti_gso_ablation.sh (single dataset).
set -euo pipefail

INTERVENTION_ROOT="/home/spencer/Projects/interventional-study"
PYTHON="/home/spencer/miniconda3/envs/cuda/bin/python"
DATA_ROOT="/mnt/nvme_1tb_a/kubric_interventions/datasets"
LOG_ROOT="/mnt/nvme_1tb_a/kubric_interventions/materialize_logs"

NAME="kitti_fitted_gso_hq_5000"
THETA="/mnt/nvme_1tb_a/theta_kitti_fitted_gso_hq.json"
N_PAIRS="${N_PAIRS:-5000}"
RENDER_WORKERS="${RENDER_WORKERS:-6}"
SPP="${SPP:-16}"

mkdir -p "$DATA_ROOT" "$LOG_ROOT"
out_root="$DATA_ROOT/$NAME"
log="$LOG_ROOT/${NAME}.log"

if pgrep -f "render_kubric_dataset.py .*${out_root}" >/dev/null; then
  echo "already running: $NAME"
  exit 0
fi

echo "launching $NAME -> $log  (n_pairs=$N_PAIRS, workers=$RENDER_WORKERS, spp=$SPP, CPU)"
nohup setsid --fork --wait \
  "$PYTHON" -u "$INTERVENTION_ROOT/render_kubric_dataset.py" \
  --theta "$THETA" \
  --out-root "$out_root" \
  --n-pairs "$N_PAIRS" \
  --render-workers "$RENDER_WORKERS" \
  --samples-per-pixel "$SPP" \
  --resume \
  >"$log" 2>&1 < /dev/null &
echo "$!" >"$LOG_ROOT/${NAME}.pid"
echo "pid $(cat "$LOG_ROOT/${NAME}.pid")  | tail -f $log"
