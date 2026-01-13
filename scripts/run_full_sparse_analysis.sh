#!/bin/bash
# Master script to run complete sparse regularization analysis
# This runs all analysis scripts in the correct order

set -e  # Exit on error

echo "================================================================================"
echo "SPARSE REGULARIZATION ANALYSIS - FULL PIPELINE"
echo "================================================================================"
echo ""

# Configuration
SPAIR_ONLY_DIR="snapshots_spair_only"
DWARP_DIR="snapshots_2d_warps"
OUTPUT_DIR="analysis/sparse_regularization"
BENCHMARK="spair"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Main analysis (learning curves, final performance, summary)
echo "Step 1/4: Running main analysis (learning curves, performance comparison)..."
echo "--------------------------------------------------------------------------------"
python scripts/analyze_sparse_regularization.py \
    --spair-only-dir "$SPAIR_ONLY_DIR" \
    --2dwarp-dir "$DWARP_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --benchmark "$BENCHMARK"

echo ""
echo "✓ Main analysis complete"
echo ""

# Step 2: Smoothness analysis (OPTIONAL - requires GPU and time)
echo "Step 2/4: Smoothness analysis (OPTIONAL - GPU intensive)..."
echo "--------------------------------------------------------------------------------"
echo "NOTE: This step loads checkpoints and runs inference on SPAIR test set."
echo "      It may take significant time and requires GPU."
echo ""
read -p "Run smoothness analysis? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python scripts/run_smoothness_comparison.py \
        --snapshot-dirs "$SPAIR_ONLY_DIR" "$DWARP_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --benchmarks "$BENCHMARK" \
        --batch-size 8 \
        --num-workers 4 \
        --device cuda
    echo ""
    echo "✓ Smoothness analysis complete"
else
    echo "⊘ Skipping smoothness analysis"
    echo "  (You can run it later with: python scripts/run_smoothness_comparison.py ...)"
fi
echo ""

# Step 3: Dense dataset evaluation (OPTIONAL - requires GPU and time)
echo "Step 3/4: Dense dataset evaluation (OPTIONAL - GPU intensive)..."
echo "--------------------------------------------------------------------------------"
echo "NOTE: This step evaluates checkpoints on KITTI and Middlebury."
echo "      It may take significant time and requires GPU + datasets."
echo ""
read -p "Run dense dataset evaluation? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    python scripts/evaluate_dense_datasets.py \
        --snapshot-dirs "$SPAIR_ONLY_DIR" "$DWARP_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --benchmarks kitti2015 kitti2012 middlebury \
        --device cuda
    echo ""
    echo "✓ Dense dataset evaluation complete"
else
    echo "⊘ Skipping dense dataset evaluation"
    echo "  (You can run it later with: python scripts/evaluate_dense_datasets.py ...)"
fi
echo ""

# Step 4: Aggregate results
echo "Step 4/4: Aggregating results into publication figure..."
echo "--------------------------------------------------------------------------------"
python scripts/aggregate_sparse_analysis.py \
    --analysis-dir "$OUTPUT_DIR" \
    --output aggregate_figure.png

echo ""
echo "✓ Aggregation complete"
echo ""

echo "================================================================================"
echo "ANALYSIS PIPELINE COMPLETE"
echo "================================================================================"
echo ""
echo "Results available in: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - learning_curves_spair.png         : Training dynamics comparison"
echo "  - final_performance_spair.png       : Final performance with statistics"
echo "  - summary_report.txt                : Text summary with conclusions"
if [ -f "$OUTPUT_DIR/smoothness_comparison_spair.png" ]; then
    echo "  - smoothness_comparison_spair.png   : Flow smoothness metrics"
fi
if [ -f "$OUTPUT_DIR/dense_eval_comparison.png" ]; then
    echo "  - dense_eval_comparison.png         : Dense dataset generalization"
fi
echo "  - aggregate_figure.png              : Publication-ready 4-panel figure"
echo "  - results_table.tex                 : LaTeX table for paper"
echo "  - ANALYSIS_REPORT.md                : Comprehensive markdown report"
echo ""
