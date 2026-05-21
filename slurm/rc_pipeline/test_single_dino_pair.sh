#!/bin/bash
#SBATCH --job-name=test_dino_pair
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --qos=cs
#SBATCH --output=scripts/transfer_analysis_v3/logs/test_dino_pair_%j.log

REPO=/home/slow1/Projects/OnlineSyntheticCorrespondence
VEC_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors
SEED_CSV=$REPO/analysis_v3/pairwise_self_distances_local.csv
OUT_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/pairwise_ranks

mkdir -p $OUT_DIR $REPO/scripts/transfer_analysis_v3/logs

cd $REPO

# Give Faiss 16 GB scratch on the 80 GB A100 for efficient large-tile computation.
export FAISS_GPU_TEMP_GB=56

# Smoke test: process rank 0 only (1 pair from the 231-pair dino list).
# Check logs/test_dino_pair_<jobid>.log to verify it ran correctly.
python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
    --vec-dir    $VEC_DIR \
    --output     $OUT_DIR/test_rank_0.csv \
    --seed-csv   $SEED_CSV \
    --spaces     dino \
    --stride     231 \
    --rank       0 \
    --max-dino   8000000 \
    --batch-size 500000 \
    --gpu

echo "Test complete. Output: $OUT_DIR/test_rank_0.csv"
