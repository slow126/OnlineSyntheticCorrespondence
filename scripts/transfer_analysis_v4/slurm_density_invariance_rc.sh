#!/usr/bin/env bash
# Density-invariance RC job for BYU cs nodes.
#
# Submit compute arrays:
#   sbatch --array=0-4 --export=ALL,SPACE=flow,LEVELS="50000 200000 1000000 4000000 8000000" \
#     scripts/transfer_analysis_v4/slurm_density_invariance_rc.sh
#
#   sbatch --array=0-4 --export=ALL,SPACE=dino,LEVELS="25000 100000 500000 2000000 4000000" \
#     scripts/transfer_analysis_v4/slurm_density_invariance_rc.sh
#
# Submit analysis after an array finishes:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#     --export=ALL,SPACE=flow,LEVELS="50000 200000 1000000 4000000 8000000",ANALYZE_ONLY=1 \
#     scripts/transfer_analysis_v4/slurm_density_invariance_rc.sh

#SBATCH --partition=cs2
#SBATCH --qos=cs
#SBATCH --job-name=dens_inv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/transfer_analysis_v4/logs/density_invariance_%x_%A_%a.log

set -euo pipefail

REPO="${REPO:-/home/slow1/Projects/OnlineSyntheticCorrespondence}"
VEC_DIR="${VEC_DIR:-/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors}"
BASELINE="${BASELINE:-$REPO/analysis_v3/pairwise_self_distances.csv}"
OUT_DIR="${OUT_DIR:-$REPO/analysis_v3/density_invariance_pair_sharded}"
SPACE="${SPACE:-flow}"
PAIR_TYPES="${PAIR_TYPES:-train_eval eval_eval}"
ANALYZE_ONLY="${ANALYZE_ONLY:-0}"
COMPUTE_ONLY="${COMPUTE_ONLY:-0}"

if [ "$SPACE" = "flow" ]; then
    LEVELS="${LEVELS:-50000 200000 1000000 4000000 8000000}"
elif [ "$SPACE" = "dino" ]; then
    # Match the same fractions of the extracted-vector cap as flow:
    # flow cap 16M -> 50k/200k/1M/4M/8M;
    # DINO cap 8M -> 25k/100k/500k/2M/4M.
    LEVELS="${LEVELS:-25000 100000 500000 2000000 4000000}"
else
    echo "SPACE must be flow or dino, got: $SPACE" >&2
    exit 2
fi

read -r -a LEVEL_ARGS <<< "$LEVELS"
read -r -a PAIR_TYPE_ARGS <<< "$PAIR_TYPES"

if [ "${SLURM_ARRAY_TASK_ID:-}" != "" ] && [ "$ANALYZE_ONLY" != "1" ]; then
    if [ "$SLURM_ARRAY_TASK_ID" -ge "${#LEVEL_ARGS[@]}" ]; then
        echo "Array task $SLURM_ARRAY_TASK_ID has no matching level in: $LEVELS" >&2
        exit 2
    fi
    LEVEL_ARGS=("${LEVEL_ARGS[$SLURM_ARRAY_TASK_ID]}")
    COMPUTE_ONLY=1
fi

MODE_ARGS=()
if [ "$ANALYZE_ONLY" = "1" ]; then
    MODE_ARGS+=(--analyze-only)
elif [ "$COMPUTE_ONLY" = "1" ]; then
    MODE_ARGS+=(--compute-only)
fi

cd "$REPO"
mkdir -p scripts/transfer_analysis_v4/logs "$OUT_DIR"

echo "host      : $(hostname)"
echo "repo      : $REPO"
echo "vec_dir   : $VEC_DIR"
echo "baseline  : $BASELINE"
echo "out_dir   : $OUT_DIR"
echo "space     : $SPACE"
echo "levels    : ${LEVEL_ARGS[*]}"
echo "pair_types: ${PAIR_TYPE_ARGS[*]}"
echo "mode_args : ${MODE_ARGS[*]:-(compute+analyze)}"
nvidia-smi -L || true

srun python -u scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    --space "$SPACE" \
    --levels "${LEVEL_ARGS[@]}" \
    --pair-types "${PAIR_TYPE_ARGS[@]}" \
    --vec-dir "$VEC_DIR" \
    --baseline "$BASELINE" \
    --output-dir "$OUT_DIR" \
    "${MODE_ARGS[@]}"
