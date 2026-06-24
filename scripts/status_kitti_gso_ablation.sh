#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="/mnt/nvme_1tb_a/kubric_interventions/datasets"
LOG_ROOT="/mnt/nvme_1tb_a/kubric_interventions/materialize_logs"

count_complete() {
  local root="$1"
  find "$root/train" \
    -mindepth 1 -maxdepth 1 -type d -name 'scene_*' \
    -exec test -f '{}/rgba_00000.png' \; \
    -exec test -f '{}/rgba_00001.png' \; \
    -exec test -f '{}/backward_flow_00001.png' \; \
    -exec test -f '{}/forward_flow_00000.png' \; \
    -exec test -f '{}/data_ranges.json' \; \
    -print 2>/dev/null | wc -l
}

for name in \
  kitti_recovered_gso_hq_5000 \
  kitti_recovered_gso_matte_5000 \
  kitti_badmotion_ft_gso_hq_5000
do
  count="$(count_complete "$DATA_ROOT/$name")"
  printf '%-38s %4d/5000 complete\n' "$name" "$count"
  tail -n 1 "$LOG_ROOT/$name.log" 2>/dev/null || true
done
