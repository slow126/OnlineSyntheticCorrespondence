#!/bin/bash
#SBATCH --job-name=spair_lr1e-4_lrbackbone1e-6_pretrainedTrue
#SBATCH --output=/home/spencer/Projects/OnlineSyntheticCorrespondence/slurm_jobs/logs/spair_lr1e-4_lrbackbone1e-6_pretrainedTrue_%j.out
#SBATCH --error=/home/spencer/Projects/OnlineSyntheticCorrespondence/slurm_jobs/logs/spair_lr1e-4_lrbackbone1e-6_pretrainedTrue_%j.err
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=1
#SBATCH --mem=64g
#SBATCH --qos=cs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: spair_lr1e-4_lrbackbone1e-6_pretrainedTrue"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "Memory: 64g"
echo "Training Dataset: spair"
echo "Config: /home/spencer/Projects/OnlineSyntheticCorrespondence/slurm/experiment_configs/generated/spair_lr1e-4_lrbackbone1e-6_pretrainedTrue.yaml"

# Change to project directory
cd /home/spencer/Projects/OnlineSyntheticCorrespondence

# Activate conda environment if specified

# Run training with srun (allows sattach and real-time output)
# --ntasks=1 ensures only one instance runs (not multiple per CPU)
# -u flag disables Python buffering for real-time log output
echo "Starting training..."
echo "Command: python3 -u train_lightning.py --config /home/spencer/Projects/OnlineSyntheticCorrespondence/slurm/experiment_configs/generated/spair_lr1e-4_lrbackbone1e-6_pretrainedTrue.yaml"
srun --ntasks=1 python3 -u train_lightning.py --config /home/spencer/Projects/OnlineSyntheticCorrespondence/slurm/experiment_configs/generated/spair_lr1e-4_lrbackbone1e-6_pretrainedTrue.yaml

echo "Training completed at: $(date)"
