#!/usr/bin/env bash
# DINO FID + sliced-W2 RC job for BYU cs nodes.
#
# Submit:
#   sbatch scripts/transfer_analysis_v4/slurm_dino_sym_rc.sh
#
# Optional while cs-2-2 is idle:
#   sbatch --nodelist=cs-2-2 scripts/transfer_analysis_v4/slurm_dino_sym_rc.sh

#SBATCH --partition=cs2
#SBATCH --qos=cs
#SBATCH --job-name=dino_sym
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=scripts/transfer_analysis_v4/logs/dino_sym_%j.log

set -euo pipefail

REPO="${REPO:-/home/slow1/Projects/OnlineSyntheticCorrespondence}"
VEC_DIR="${VEC_DIR:-/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors}"
FLOW_CSV="${FLOW_CSV:-$REPO/analysis/coverage_v2_flow_only_raw_joint_full.csv}"
OUT_CSV="${OUT_CSV:-$REPO/analysis_v3/symmetric_distances_dino.csv}"

cd "$REPO"
mkdir -p scripts/transfer_analysis_v4/logs analysis_v3

echo "host    : $(hostname)"
echo "repo    : $REPO"
echo "vec_dir : $VEC_DIR"
echo "flow_csv: $FLOW_CSV"
echo "out_csv : $OUT_CSV"
nvidia-smi -L || true

srun python -u scripts/transfer_analysis_v3/compute_symmetric_distances.py \
    --flow-csv "$FLOW_CSV" \
    --vec-dir "$VEC_DIR" \
    --output "$OUT_CSV" \
    --skip-flow \
    --n-proj 200 \
    --sw-samples 100000 \
    --fid-samples 200000
