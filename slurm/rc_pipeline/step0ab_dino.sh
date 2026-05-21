#!/bin/bash
#SBATCH --job-name=dino_coverage
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --qos=cs
#SBATCH --output=scripts/transfer_analysis_v3/logs/step0ab_%j.log

REPO=/home/slow1/Projects/OnlineSyntheticCorrespondence
VEC_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors

cd $REPO
mkdir -p scripts/transfer_analysis_v3/logs analysis_v3

# Step 0a: DINO pairwise coverage + vector extraction
# Uses RC config: RC dataset paths, synthetic/val added, movi_f loaded from cache.
python scripts/calculate_coverage_faiss_v2.py \
    --config src/configs/coverage_configs/coverage_faiss_dino_v3_minimal_rc.yaml

# Step 0b: Null-calibrated cosine coverage
python scripts/transfer_analysis_v3/compute_dino_null_coverage.py \
    --coverage-csv analysis_v3/coverage_dino_full.csv \
    --vec-dir $VEC_DIR \
    --output analysis_v3/dino_null_coverage.csv \
    --null-percentiles 80 90 95 99 \
    --gpu
