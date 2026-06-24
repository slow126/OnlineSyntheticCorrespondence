#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/spencer/Projects/OnlineSyntheticCorrespondence"
INTERVENTION_ROOT="/home/spencer/Projects/interventional-study"
PYTHON="/home/spencer/miniconda3/envs/cuda/bin/python"
DATA_ROOT="/mnt/nvme_1tb_a/kubric_interventions/datasets"
LOG_ROOT="/mnt/nvme_1tb_a/kubric_interventions/materialize_logs"
N_PAIRS="${N_PAIRS:-5000}"
RENDER_WORKERS="${RENDER_WORKERS:-4}"
SPP="${SPP:-16}"

mkdir -p "$DATA_ROOT" "$LOG_ROOT"

"$PYTHON" - <<'PY'
import json
from pathlib import Path

out = Path("/mnt/nvme_1tb_a")

def load(path):
    return json.loads(Path(path).read_text())

def dump(name, theta):
    path = out / name
    path.write_text(json.dumps(theta, indent=2) + "\n")
    print(path)

kitti_hq = load("/mnt/nvme_1tb_a/theta_kitti_hq.json")
kitti_matte = load("/mnt/nvme_1tb_a/theta_kitti_matte.json")
ft_hq = load("/mnt/nvme_1tb_a/theta_ft_hq.json")

# Same recovered KITTI motion as the current KuBasic pair, but with GSO assets.
gso_hq = dict(kitti_hq)
gso_hq.update({
    "asset_source": "gso",
    "background_mode": "hdri",
    "keep_asset_materials": True,
})
dump("theta_kitti_recovered_gso_hq.json", gso_hq)

gso_matte = dict(kitti_matte)
gso_matte.update({
    "asset_source": "gso",
    "background_mode": "matte",
    "keep_asset_materials": False,
})
dump("theta_kitti_recovered_gso_matte.json", gso_matte)

# Good appearance, intentionally non-KITTI/object-dominant motion baseline.
# This uses the recovered FlyingThings-style motion setting, but swaps in GSO+HDRI.
bad_motion = dict(ft_hq)
bad_motion.update({
    "asset_source": "gso",
    "background_mode": "hdri",
    "keep_asset_materials": True,
})
dump("theta_kitti_badmotion_ft_gso_hq.json", bad_motion)
PY

launch_one() {
  local name="$1"
  local theta="$2"
  local out_root="$DATA_ROOT/$name"
  local log="$LOG_ROOT/${name}.log"

  if pgrep -f "render_kubric_dataset.py .*${out_root}" >/dev/null; then
    echo "already running: $name"
    return
  fi

  echo "launching $name -> $log"
  nohup setsid --fork --wait \
    "$PYTHON" -u "$INTERVENTION_ROOT/render_kubric_dataset.py" \
    --theta "$theta" \
    --out-root "$out_root" \
    --n-pairs "$N_PAIRS" \
    --render-workers "$RENDER_WORKERS" \
    --samples-per-pixel "$SPP" \
    --resume \
    >"$log" 2>&1 < /dev/null &
  echo "$!" >"$LOG_ROOT/${name}.pid"
}

launch_one \
  "kitti_recovered_gso_hq_5000" \
  "/mnt/nvme_1tb_a/theta_kitti_recovered_gso_hq.json"

launch_one \
  "kitti_recovered_gso_matte_5000" \
  "/mnt/nvme_1tb_a/theta_kitti_recovered_gso_matte.json"

launch_one \
  "kitti_badmotion_ft_gso_hq_5000" \
  "/mnt/nvme_1tb_a/theta_kitti_badmotion_ft_gso_hq.json"

echo
echo "Logs:"
echo "  $LOG_ROOT/kitti_recovered_gso_hq_5000.log"
echo "  $LOG_ROOT/kitti_recovered_gso_matte_5000.log"
echo "  $LOG_ROOT/kitti_badmotion_ft_gso_hq_5000.log"
