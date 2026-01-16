#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="analysis/full_faiss_coverage_logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
python scripts/calculate_coverage_faiss.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full.yaml \
  |& tee "$LOG_DIR/coverage_flow_full.log"

python scripts/calculate_coverage_faiss.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet.yaml \
  |& tee "$LOG_DIR/coverage_resnet_full.log"

python scripts/calculate_coverage_faiss.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full.yaml \
  |& tee "$LOG_DIR/coverage_dino_full.log"
