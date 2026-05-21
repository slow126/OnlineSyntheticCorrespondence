#!/bin/bash
# Submit the full Transfer Analysis v3 pipeline to the RC cluster.
# Run from the repo root: bash slurm/rc_pipeline/submit_all.sh
#
# Job dependency chain:
#   step0ab (GPU)  ─┐
#   step0c  (CPU)  ─┤─► step0e_merge_experiments (CPU)
#   step0d  (GPU array) ─┘

set -euo pipefail
REPO=/home/slow1/Projects/OnlineSyntheticCorrespondence
cd $REPO

mkdir -p scripts/transfer_analysis_v3/logs

# Step 0a+0b: DINO vector coverage + null calibration (GPU, up to 8h)
JOB0AB=$(sbatch --parsable slurm/rc_pipeline/step0ab_dino.sh)
echo "Submitted step0ab (DINO coverage): $JOB0AB"

# Step 0c: FID + sliced Wasserstein (CPU, ~30 min)
JOB0C=$(sbatch --parsable slurm/rc_pipeline/step0c_symmetric.sh)
echo "Submitted step0c (symmetric distances): $JOB0C"

# Step 0d: pairwise self-distances job array (GPU, 231 jobs × ~30 min each)
JOB0D=$(sbatch --parsable scripts/transfer_analysis_v3/run_pairwise_slurm.sh)
echo "Submitted step0d array (pairwise distances): $JOB0D"

# Step 0e + merge + experiments: runs after ALL of the above finish
JOB_EXP=$(sbatch --parsable \
    --dependency=afterok:${JOB0AB}:${JOB0C}:${JOB0D} \
    slurm/rc_pipeline/step0e_merge_experiments.sh)
echo "Submitted step0e+experiments (after all GPU jobs): $JOB_EXP"

echo ""
echo "Pipeline submitted. Monitor with: squeue -u \$USER"
echo "Logs in: $REPO/scripts/transfer_analysis_v3/logs/"
