#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/spencer/Projects/OnlineSyntheticCorrespondence"
PYTHON="/home/spencer/miniconda3/envs/cuda/bin/python"
DATA_ROOT="/mnt/nvme_1tb_a/kubric_interventions/datasets"
LOG_ROOT="/mnt/nvme_1tb_a/kubric_interventions/training_logs"
EXPECTED_SCENES=5000

mkdir -p "$LOG_ROOT"
cd "$ROOT"

count_complete_scenes() {
  local split_dir="$1"
  find "$split_dir" \
    -mindepth 1 -maxdepth 1 -type d -name 'scene_*' \
    -exec test -f '{}/rgba_00000.png' \; \
    -exec test -f '{}/rgba_00001.png' \; \
    -exec test -f '{}/backward_flow_00001.png' \; \
    -exec test -f '{}/forward_flow_00000.png' \; \
    -exec test -f '{}/data_ranges.json' \; \
    -print | wc -l
}

wait_for_dataset() {
  local name="$1"
  local split_dir="$DATA_ROOT/$name/train"
  local count
  while true; do
    count="$(count_complete_scenes "$split_dir")"
    printf '%s %s: %s/%s complete\n' "$(date --iso-8601=seconds)" \
      "$name" "$count" "$EXPECTED_SCENES"
    if [[ "$count" -eq "$EXPECTED_SCENES" ]]; then
      return
    fi
    sleep 60
  done
}

launch_training() {
  local name="$1"
  local gpu="$2"
  local config="$3"
  local log="$LOG_ROOT/${name}.log"

  wait_for_dataset "$name"
  printf '%s launching %s on GPU %s\n' "$(date --iso-8601=seconds)" \
    "$name" "$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" MPLCONFIGDIR=/tmp/matplotlib-"$name" \
    "$PYTHON" -u train_lightning.py --config "$config" \
    >"$log" 2>&1
}

case "${1:-}" in
  hq)
    launch_training \
      kitti_recovered_hq_5000 \
      0 \
      src/configs/lightning/kubric_kitti_recovered_hq.yaml
    ;;
  matte)
    launch_training \
      kitti_recovered_matte_5000 \
      1 \
      src/configs/lightning/kubric_kitti_recovered_matte.yaml
    ;;
  *)
    echo "usage: $0 {hq|matte}" >&2
    exit 2
    ;;
esac
