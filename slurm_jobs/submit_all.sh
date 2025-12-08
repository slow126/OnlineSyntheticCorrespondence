#!/bin/bash
# Submit all generated SLURM jobs
# Get the directory where this script is located (works from any location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo 'Submitting job_flyingthings_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_freezeTrue_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_freezeTrue_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_freezeTrue_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_freezeTrue_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_freezeFalse_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_freezeFalse_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_freezeFalse_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_freezeFalse_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_logstepslogarithmic_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_logstepslogarithmic_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_flyingthings_logstepslogarithmic_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_flyingthings_logstepslogarithmic_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_freezeTrue_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_freezeTrue_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_freezeTrue_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_freezeTrue_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_freezeFalse_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_freezeFalse_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_freezeFalse_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_freezeFalse_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_logstepslogarithmic_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_spair_logstepslogarithmic_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_spair_logstepslogarithmic_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_spair_logstepslogarithmic_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_freezeTrue_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_freezeTrue_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_freezeTrue_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_freezeTrue_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_freezeFalse_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_freezeFalse_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_freezeFalse_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_freezeFalse_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_logstepslogarithmic_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_logstepslogarithmic_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_pointodyssey_logstepslogarithmic_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_pointodyssey_logstepslogarithmic_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr3e-4_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr3e-4_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr3e-4_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr3e-4_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-3_lrbackbone1e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-3_lrbackbone1e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-3_lrbackbone3e-6_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_lr1e-3_lrbackbone3e-6_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_freezeTrue_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_freezeTrue_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_freezeTrue_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_freezeTrue_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_freezeFalse_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_freezeFalse_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_freezeFalse_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_freezeFalse_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_logstepslogarithmic_pretrainedTrue.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_logstepslogarithmic_pretrainedTrue.sh"
sleep 1  # Small delay between submissions

echo 'Submitting job_synthetic_logstepslogarithmic_pretrainedFalse.sh...'
sbatch "$SCRIPT_DIR/job_synthetic_logstepslogarithmic_pretrainedFalse.sh"
sleep 1  # Small delay between submissions

echo 'All jobs submitted!'
