#!/usr/bin/env bash
# Driver for the SDF-fractal renderer-agnostic table: train each of the 6 cells
# (cats/glunet/flowformer x default/tuned), then post-hoc eval KITTI(alphas) +
# TSS(alphas) + TAP-Vid(strides). Finally assemble the LaTeX tables.
#
# SAFE BY DEFAULT: prints the plan and exits. It only trains/evals when you pass GO=1.
#   ./run_table.sh                 # dry-run: print what would happen
#   GO=1 ./run_table.sh            # actually train + eval all cells (sequential)
#   GO=1 CELLS="cats_default glunet_default" ./run_table.sh   # subset
#   GO=1 EVAL_ONLY=1 ./run_table.sh                            # skip training, eval existing dirs
#   GPU=1 GO=1 ./run_table.sh      # pin CUDA_VISIBLE_DEVICES
set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HARNESS")"
cd "$REPO"

SNAP_BASE="${RA_OUT:-/mnt/nvme_1tb_a/renderer_agnostic}"
GPU="${GPU:-0}"
ALL_CELLS="cats_default cats_tuned glunet_default glunet_tuned flowformer_default flowformer_tuned"
CELLS="${CELLS:-$ALL_CELLS}"
GO="${GO:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
NOGEN="${NOGEN:-0}"     # skip gen_configs (when running parallel queues that share configs)
NOBUILD="${NOBUILD:-0}" # skip the final build_table (assemble once, after all queues finish)

echo "== renderer-agnostic harness =="
echo "  REPO=$REPO"
echo "  snapshots base=$SNAP_BASE   GPU=$GPU   GO=$GO   EVAL_ONLY=$EVAL_ONLY"
echo "  cells: $CELLS"

# 1) (re)generate configs
if [ "$NOGEN" = "1" ]; then
  echo "  [NOGEN] skipping gen_configs (using existing configs/)"
elif [ "$GO" = "1" ]; then
  RA_OUT="$SNAP_BASE" python "$HARNESS/gen_configs.py"
else
  echo "  [dry-run] would run: RA_OUT=$SNAP_BASE python gen_configs.py"
fi

run() { echo "  + $*"; if [ "$GO" = "1" ]; then CUDA_VISIBLE_DEVICES="$GPU" "$@"; fi; }

for cell in $CELLS; do
  cfg="$HARNESS/configs/${cell}.yaml"
  echo "---- cell: $cell ----"

  # locate or create the output dir (snapshots/<cell>_<timestamp>)
  outdir=""
  if [ "$EVAL_ONLY" != "1" ]; then
    run python train_lightning.py --config "$cfg"
  fi
  # newest matching dir
  if [ "$GO" = "1" ]; then
    outdir="$(ls -dt "$SNAP_BASE/${cell}_"* 2>/dev/null | head -1 || true)"
    if [ -z "$outdir" ]; then echo "  !! no output dir for $cell (training failed?)"; continue; fi
    echo "  outdir=$outdir"
  else
    echo "  [dry-run] would train from $cfg then eval newest $SNAP_BASE/${cell}_*"
    outdir="$SNAP_BASE/${cell}_<timestamp>"
  fi

  run python "$HARNESS/score_transfer_cell.py" --cell "$outdir" \
      --families kitti tss tapvid \
      --kitti-alphas 0.05 0.03 0.01 \
      --tss-alphas 0.10 0.05 0.03 \
      --tapvid-strides 1 2 4 8 16 --tapvid-alpha 0.05
done

if [ "$NOBUILD" = "1" ]; then
  echo "  [NOBUILD] skipping build_table (run it once after all queues finish)"
else
  echo "---- assemble tables ----"
  run python "$HARNESS/build_table.py" --snap-base "$SNAP_BASE" --cells $CELLS
fi
echo "== done =="
