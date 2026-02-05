#!/usr/bin/env bash
# Coverage Pipeline v2.2 Runner (Manifold self-radius with exact dedup)

set -u

LOG_DIR="analysis/full_faiss_coverage_logs_v2_manifold_dedup"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES=0
export COVERAGE_SELF_RADIUS_DEDUP=1
export COVERAGE_ALPHA_DEDUP=1
export COVERAGE_DISABLE_ALPHA=1

echo "========================================="
echo "Coverage Pipeline v2.2 (Manifold, Dedup Self-Radius)"
echo "========================================="
echo "Features:"
echo "  - Flow: manifold self-radius (exact dedup for radius only)"
echo "  - ResNet/DINO: PCA + L2 normalization, manifold self-radius (exact dedup for radius only)"
echo "  - Cross-dataset distances keep duplicates (density signal)"
echo "========================================="
echo

run_step() {
  local name="$1"
  shift
  local log_file="$1"
  shift
  echo "$name"
  "$@" |& tee "$LOG_DIR/$log_file"
  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "  ⚠️  Step failed (exit=$status), continuing..."
  fi
  return 0
}

# Flow representation (manifold v2)
run_step "[1/3] Running FLOW manifold coverage analysis..." \
  coverage_flow_full_v2_dedup.log \
  python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_flow_full_v2_joint.yaml

echo
echo "[1/3] FLOW manifold complete ✓"
echo

# ResNet representation (single feature space)
run_step "[2/3] Running RESNET manifold coverage analysis..." \
  coverage_resnet_full_v2_dedup.log \
  python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_resnet_v2.yaml

echo
echo "[2/3] RESNET complete ✓"
echo

# DINO representation (single feature space)
run_step "[3/3] Running DINO manifold coverage analysis..." \
  coverage_dino_full_v2_dedup.log \
  python -u scripts/calculate_coverage_faiss_v2.py \
  --config src/configs/coverage_configs/coverage_faiss_dino_full_v2.yaml

echo
echo "[3/3] DINO complete ✓"
echo

echo "========================================="
echo "All coverage analyses complete!"
echo "========================================="
echo "Logs:"
echo "  - $LOG_DIR/"
echo "========================================="
