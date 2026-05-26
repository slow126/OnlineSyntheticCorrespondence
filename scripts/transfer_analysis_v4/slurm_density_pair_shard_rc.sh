#!/usr/bin/env bash
# Compute one shard of a density-invariance level on RC.
#
# Submit one array per (SPACE, LEVEL):
#   sbatch --array=0-164%8 \
#     --export=ALL,SPACE=dino,LEVEL=200000,SHARD_COUNT=165 \
#     scripts/transfer_analysis_v4/slurm_density_pair_shard_rc.sh

#SBATCH --partition=cs2
#SBATCH --qos=cs
#SBATCH --job-name=dens_pair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=scripts/transfer_analysis_v4/logs/density_pair_%x_%A_%a.log

set -euo pipefail

REPO="${REPO:-/home/slow1/Projects/OnlineSyntheticCorrespondence}"
VEC_DIR="${VEC_DIR:-/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors}"
OUT_DIR="${OUT_DIR:-$REPO/analysis_v3/density_invariance_pair_sharded}"
SPACE="${SPACE:?set SPACE=flow or dino}"
LEVEL="${LEVEL:?set LEVEL, e.g. 200000}"
PAIR_TYPES="${PAIR_TYPES:-train_eval eval_eval}"
SHARD_COUNT="${SHARD_COUNT:-165}"
RANK="${SLURM_ARRAY_TASK_ID:-${RANK:-0}}"

if [ "$SPACE" != "flow" ] && [ "$SPACE" != "dino" ]; then
    echo "SPACE must be flow or dino, got: $SPACE" >&2
    exit 2
fi
if [ "$RANK" -ge "$SHARD_COUNT" ]; then
    echo "RANK $RANK must be < SHARD_COUNT $SHARD_COUNT" >&2
    exit 2
fi

read -r -a PAIR_TYPE_ARGS <<< "$PAIR_TYPES"

LEVEL_DIR="$OUT_DIR/shards/${SPACE}_N${LEVEL}"
OUT_CSV="$LEVEL_DIR/rank_${RANK}.csv"

cd "$REPO"
mkdir -p scripts/transfer_analysis_v4/logs "$LEVEL_DIR"

echo "host       : $(hostname)"
echo "repo       : $REPO"
echo "vec_dir    : $VEC_DIR"
echo "out_csv    : $OUT_CSV"
echo "space      : $SPACE"
echo "level      : $LEVEL"
echo "pair_types : ${PAIR_TYPE_ARGS[*]}"
echo "rank/stride: $RANK / $SHARD_COUNT"
nvidia-smi -L || true

MAX_ARG="--max-${SPACE}"

srun python -u scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
    --vec-dir "$VEC_DIR" \
    --output "$OUT_CSV" \
    --spaces "$SPACE" \
    --pair-types "${PAIR_TYPE_ARGS[@]}" \
    "$MAX_ARG" "$LEVEL" \
    --stride "$SHARD_COUNT" \
    --rank "$RANK" \
    --gpu
