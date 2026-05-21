#!/bin/bash
#SBATCH --job-name=symmetric_dist
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --qos=cs
#SBATCH --output=scripts/transfer_analysis_v3/logs/step0c_%j.log

REPO=/home/slow1/Projects/OnlineSyntheticCorrespondence
VEC_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors

cd $REPO
mkdir -p scripts/transfer_analysis_v3/logs analysis_v3

# Step 0c: FID + sliced Wasserstein between train/eval distributions.
# Needs flow vectors (transferred separately, ~2.7 GB).
python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
    --flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv \
    --vec-dir $VEC_DIR \
    --output analysis_v3/symmetric_distances.csv \
    --n-proj 200 \
    --sw-samples 100000 \
    --fid-samples 200000
