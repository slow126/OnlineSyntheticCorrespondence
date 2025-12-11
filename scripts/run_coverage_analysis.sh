#!/bin/bash
# Workflow script for coverage analysis
# 
# This script:
# 1. Builds coresets for datasets (if not already built)
# 2. Computes pairwise coverage metrics
# 3. Generates plots with plot_benchmark_metrics.py
#
# Usage:
#   bash scripts/run_coverage_analysis.sh

set -e  # Exit on error

echo "============================================"
echo "COVERAGE ANALYSIS WORKFLOW"
echo "============================================"

# Step 1: Build coresets (if config exists)
CORESET_CONFIG="src/configs/coreset_configs/build_datasets.yaml"

if [ -f "$CORESET_CONFIG" ]; then
    echo ""
    echo "Step 1: Building coresets..."
    echo "--------------------------------------------"
    
    # Check if coresets directory has files
    if [ -z "$(ls -A coresets/*.pt 2>/dev/null)" ]; then
        echo "Building coresets from config: $CORESET_CONFIG"
        python scripts/build_coresets.py --config "$CORESET_CONFIG"
    else
        echo "Coresets already exist in coresets/. Skipping build."
        echo "To rebuild, delete coresets/*.pt and re-run this script."
    fi
else
    echo "WARNING: Coreset config not found: $CORESET_CONFIG"
    echo "Skipping coreset build. Make sure coresets/*.pt files exist."
fi

# Step 2: Calculate pairwise coverage
echo ""
echo "Step 2: Computing pairwise coverage metrics..."
echo "--------------------------------------------"

if [ -z "$(ls -A coresets/*.pt 2>/dev/null)" ]; then
    echo "ERROR: No coreset files found in coresets/"
    echo "Please build coresets first or check the directory."
    exit 1
fi

python scripts/calculate_coverage.py \
    --coresets-dir coresets/ \
    --output coverage_results.csv \
    --epsilon eps_base \
    --min-count 0

echo ""
echo "Coverage results saved to: coverage_results.csv"

# Step 3: Generate plots (if snapshots exist)
echo ""
echo "Step 3: Generating plots..."
echo "--------------------------------------------"

if [ -d "snapshots" ] && [ -n "$(ls -A snapshots/)" ]; then
    echo "Running plot_benchmark_metrics.py to generate coverage plots..."
    python plot_benchmark_metrics.py \
        --snapshots_dir snapshots/ \
        --output-dir benchmark_plots/
    
    echo ""
    echo "Plots saved to: benchmark_plots/"
    echo "  - training_coverage_vs_best_pck.png (scatter plot)"
    echo "  - training_coverage_vs_best_pck_errorbars.png (error bars)"
else
    echo "No snapshots directory found. Skipping plot generation."
    echo "Run training to generate snapshots, then re-run this script."
fi

echo ""
echo "============================================"
echo "COVERAGE ANALYSIS COMPLETE!"
echo "============================================"
echo ""
echo "Results:"
echo "  - Coverage CSV: coverage_results.csv"
if [ -d "benchmark_plots" ]; then
    echo "  - Plots: benchmark_plots/"
fi
echo ""
