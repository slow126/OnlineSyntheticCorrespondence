#!/usr/bin/env bash
# Density-invariance RC job for BYU cs nodes.
#
# Submit examples:
#   sbatch --export=ALL,SPACE=flow,LEVELS="50000 200000 1000000" \
#     scripts/transfer_analysis_v4/slurm_density_invariance_rc.sh
#
#   sbatch --export=ALL,SPACE=dino,LEVELS="10000 50000 200000" \
#     scripts/transfer_analysis_v4/slurm_density_invariance_rc.sh
#
# Optional while cs-2-2 is idle:
#   sbatch --nodelist=cs-2-2 --export=ALL,SPACE=flow,LEVELS="50000 200000 1000000" \
#     scripts/transfer_analysis_v4/slurm_density_invariance_rc.sh

#SBATCH --partition=cs2
#SBATCH --qos=cs
#SBATCH --job-name=dens_inv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=scripts/transfer_analysis_v4/logs/density_invariance_%x_%j.log

set -euo pipefail

REPO="${REPO:-/home/slow1/Projects/OnlineSyntheticCorrespondence}"
VEC_DIR="${VEC_DIR:-/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors}"
BASELINE="${BASELINE:-$REPO/analysis_v3/pairwise_self_distances.csv}"
OUT_DIR="${OUT_DIR:-$REPO/analysis_v3/density_invariance}"
SPACE="${SPACE:-flow}"
PAIR_TYPES="${PAIR_TYPES:-train_eval eval_eval}"

if [ "$SPACE" = "flow" ]; then
    LEVELS="${LEVELS:-50000 200000 1000000}"
elif [ "$SPACE" = "dino" ]; then
    LEVELS="${LEVELS:-10000 50000 200000}"
else
    echo "SPACE must be flow or dino, got: $SPACE" >&2
    exit 2
fi

read -r -a LEVEL_ARGS <<< "$LEVELS"
read -r -a PAIR_TYPE_ARGS <<< "$PAIR_TYPES"

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
nvidia-smi -L || true

python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
    --space "$SPACE" \
    --levels "${LEVEL_ARGS[@]}" \
    --pair-types "${PAIR_TYPE_ARGS[@]}" \
    --vec-dir "$VEC_DIR" \
    --baseline "$BASELINE" \
    --output-dir "$OUT_DIR"
