#!/usr/bin/env bash
# Merge pair-shard CSVs for one density-invariance level.
#
# Submit after a shard array:
#   sbatch --dependency=afterok:<ARRAY_JOB_ID> \
#     --export=ALL,SPACE=dino,LEVEL=200000 \
#     scripts/transfer_analysis_v4/slurm_density_merge_level_rc.sh

#SBATCH --partition=cs2
#SBATCH --qos=cs
#SBATCH --job-name=dens_merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=scripts/transfer_analysis_v4/logs/density_merge_%x_%j.log

set -euo pipefail

REPO="${REPO:-/home/slow1/Projects/OnlineSyntheticCorrespondence}"
OUT_DIR="${OUT_DIR:-$REPO/analysis_v3/density_invariance_pair_sharded}"
SPACE="${SPACE:?set SPACE=flow or dino}"
LEVEL="${LEVEL:?set LEVEL, e.g. 200000}"

LEVEL_DIR="$OUT_DIR/shards/${SPACE}_N${LEVEL}"
OUT_CSV="$OUT_DIR/pairwise_self_${SPACE}_N${LEVEL}.csv"

cd "$REPO"
mkdir -p scripts/transfer_analysis_v4/logs "$OUT_DIR"

echo "host     : $(hostname)"
echo "repo     : $REPO"
echo "level_dir: $LEVEL_DIR"
echo "out_csv  : $OUT_CSV"
ls "$LEVEL_DIR"/rank_*.csv | wc -l

python scripts/transfer_analysis_v3/merge_pairwise_distances.py \
    --inputs "$LEVEL_DIR"/rank_*.csv \
    --output "$OUT_CSV"
