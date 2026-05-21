#!/bin/bash
#SBATCH --job-name=merge_and_experiments
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --qos=cs
#SBATCH --output=scripts/transfer_analysis_v3/logs/step0e_exp_%j.log

REPO=/home/slow1/Projects/OnlineSyntheticCorrespondence
VEC_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/coverage_vectors
RANKS_DIR=/home/slow1/fsl_groups/grp_farrell/slow1/pairwise_ranks
SEED_CSV=$REPO/analysis_v3/pairwise_self_distances_local.csv

cd $REPO
mkdir -p scripts/transfer_analysis_v3/logs analysis_v3

# Step 0d merge: combine per-rank CSVs with existing local results, symmetrize.
python scripts/transfer_analysis_v3/merge_pairwise_distances.py \
    --inputs $RANKS_DIR/rank_*.csv \
    --seed-csv $SEED_CSV \
    --output analysis_v3/pairwise_self_distances.csv

# Step 0e: Symmetric self-distances (flow FID/SW2/MMD for IDW neighborhoods).
python scripts/transfer_analysis_v3/build_symmetric_self_distances.py \
    --self-dist analysis_v3/pairwise_self_distances.csv \
    --output analysis_v3/pairwise_symmetric_distances.csv \
    --vec-dir $VEC_DIR \
    --flow-mmd-csv flow_mmd_results_fast.csv \
    --pair-types train_train eval_eval \
    --fid-samples 200000 \
    --sw-samples 100000 \
    --n-proj 200

# Steps 1–3: build table → experiments → compile (pure CPU, fast)
MINIMAL=1 LOCO=0 FLOW_ONLY=1 \
  VEC_DIR=$VEC_DIR \
  DINO_CONFIG=src/configs/coverage_configs/coverage_faiss_dino_v3_minimal_rc.yaml \
  N_EVAL=10 \
  bash scripts/transfer_analysis_v3/run_pipeline.sh
