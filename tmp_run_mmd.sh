#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="analysis/full_mmd_coverage_logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=1
python scripts/calculate_flow_mmd.py \
  --config src/configs/mmd_configs/flow_mmd_config_full.yaml \
  |& tee "$LOG_DIR/flow_mmd_full.log"

python scripts/calculate_feature_mmd.py \
  --config src/configs/mmd_configs/feature_mmd_config_full.yaml \
  |& tee "$LOG_DIR/feature_mmd_full.log"

python scripts/calculate_feature_mmd.py \
  --config src/configs/mmd_configs/feature_mmd_config_dino_full.yaml \
  |& tee "$LOG_DIR/dino_mmd_full.log"

python scripts/calculate_coverage_faiss.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full.yaml \
  |& tee "$LOG_DIR/coverage_flow_full.log"

python scripts/calculate_coverage_faiss.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full.yaml \
  |& tee "$LOG_DIR/coverage_dino_full.log"

