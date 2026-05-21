#!/bin/bash
#SBATCH --job-name=pairwise_dino
#SBATCH --array=0-230
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --qos=cs
#SBATCH --output=scripts/transfer_analysis_v3/logs/pairwise_%A_%a.log

VEC_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors
SEED_CSV=/home/slow1/Projects/OnlineSyntheticCorrespondence/analysis_v3/pairwise_self_distances_local.csv
OUT_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/pairwise_ranks
REPO=/home/slow1/Projects/OnlineSyntheticCorrespondence

mkdir -p $OUT_DIR $REPO/scripts/transfer_analysis_v3/logs

cd $REPO

# Give Faiss 16 GB scratch on the 80 GB A100 for efficient large-tile computation.
export FAISS_GPU_TEMP_GB=16

python scripts/transfer_analysis_v3/compute_pairwise_self_distances.py \
    --vec-dir    $VEC_DIR \
    --output     $OUT_DIR/rank_${SLURM_ARRAY_TASK_ID}.csv \
    --seed-csv   $SEED_CSV \
    --spaces     dino \
    --stride     231 \
    --rank       $SLURM_ARRAY_TASK_ID \
    --max-dino   8000000 \
    --gpu
