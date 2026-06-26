#!/bin/bash
# Kubric-variant grid: GLU-Net + FlowFormer x 8 tss_* motion variants x {tt,tf} = 32 runs.
# The discriminating flow models on every controlled kubric motion mode (CATs couldn't tell
# them apart). Evals on TSS + TAP-Vid(stride6) + kitti. GLU-Net runs FIRST (fast), then FlowFormer.
#
# Usage:
#   bash tap_vid_probe/run_kubvar.sh glunet       # 16 glunet runs across 2 GPUs (~3-5h)
#   bash tap_vid_probe/run_kubvar.sh flowformer   # 16 flowformer runs across 2 GPUs (~18-24h)
#   bash tap_vid_probe/run_kubvar.sh all          # glunet then flowformer
#   GPUS="0 1" bash tap_vid_probe/run_kubvar.sh all
cd /home/spencer/Projects/OnlineSyntheticCorrespondence || exit 1
CFG=tap_vid_probe/configs/kubvar
LOG=tap_vid_probe/logs/kubvar
mkdir -p "$LOG"
read -r -a GPU <<< "${GPUS:-0 1}"
MODE="${1:-all}"

run_one() {  # $1=name $2=gpu
  echo "[$(date +%H:%M:%S)] START $1 on gpu$2"
  ( ulimit -c 0; CUDA_VISIBLE_DEVICES="$2" python -u train_lightning.py --config "$CFG/$1.yaml" ) \
      > "$LOG/$1.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE  $1 (exit $?)"
}
run_lane() { local gpu=$1; shift; for n in "$@"; do run_one "$n" "$gpu"; done; }

# motion-priority order: clean LARGE-motion first (fast vs-movi-f comparison), then small control, then rest
VPRIORITY="zoom1obj_big zoom2obj ns4 mm4 camonly zoom1obj zoom1obj_focal so1"
listm() { local m=$1 v r name; for v in $VPRIORITY; do for r in tf tt; do
  name="kv_${v}_${m}_${r}"; [ -f "$CFG/$name.yaml" ] && echo "$name"; done; done; }
case "$MODE" in
  glunet)     mapfile -t ALL < <(listm glunet) ;;
  flowformer) mapfile -t ALL < <(listm flowformer) ;;
  all)        mapfile -t ALL < <(listm glunet; listm flowformer) ;;
  *) echo "mode must be glunet|flowformer|all"; exit 1 ;;
esac
echo "launching ${#ALL[@]} runs on GPUs: ${GPU[*]}  (mode=$MODE)"

# N-GPU general: assign run i to GPU[i % ngpu], one lane per GPU (handles 1 or 2+ GPUs)
ngpu=${#GPU[@]}
for ((g=0; g<ngpu; g++)); do
  lane=()
  for i in "${!ALL[@]}"; do (( i % ngpu == g )) && lane+=("${ALL[i]}"); done
  echo "  lane gpu${GPU[g]}: ${#lane[@]} runs"
  run_lane "${GPU[g]}" "${lane[@]}" &
done
wait
echo "[$(date +%H:%M:%S)] kubvar grid ($MODE) complete"
