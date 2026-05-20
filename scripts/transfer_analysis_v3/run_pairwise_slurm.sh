#!/bin/bash
#SBATCH --job-name=pairwise_dist
#SBATCH --array=0-7               # 8 workers; adjust to match --stride below
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pairwise_%A_%a.log

# ---------------------------------------------------------------------------
# Paths — edit before submitting
# ---------------------------------------------------------------------------
VEC_DIR=/scratch/$USER/coverage_vectors        # where you rsync'd the .npy files
KNN_DIR=$VEC_DIR/knn_self                      # included inside VEC_DIR
SEED_CSV=/scratch/$USER/pairwise_self_distances_local.csv  # copy of local CSV
OUT_DIR=/scratch/$USER/pairwise_ranks
REPO=/scratch/$USER/OnlineSyntheticCorrespondence

mkdir -p $OUT_DIR logs

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python $REPO/scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
    --vec-dir    $VEC_DIR \
    --output     $OUT_DIR/rank_${SLURM_ARRAY_TASK_ID}.csv \
    --seed-csv   $SEED_CSV \
    --spaces     dino \
    --stride     8 \
    --rank       $SLURM_ARRAY_TASK_ID \
    --max-dino   8000000 \
    --gpu
