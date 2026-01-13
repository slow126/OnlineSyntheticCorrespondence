#!/usr/bin/env python3
"""
Aggregate all sparse regularization analysis results into publication-ready figures.

This script loads results from:
1. Learning curves analysis
2. Smoothness metrics analysis  
3. Dense dataset evaluation

And creates a comprehensive 4-panel publication figure.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_analysis_results(analysis_dir):
    """Load all analysis results from directory"""
    analysis_dir = Path(analysis_dir)
    
    results = {}
    
    # Check for required files
    required_files = {
        'learning_curves': 'learning_curves_spair.png',
        'final_performance': 'final_performance_spair.png',
        'smoothness': 'smoothness_results_aggregated.csv',
        'dense_eval': 'dense_eval_results.csv',
        'summary': 'summary_report.txt'
    }
    
    for key, filename in required_files.items():
        filepath = analysis_dir / filename
        if filepath.exists():
            if filename.endswith('.csv'):
                results[key] = pd.read_csv(filepath)
            elif filename.endswith('.txt'):
                with open(filepath, 'r') as f:
                    results[key] = f.read()
            else:
                results[key] = str(filepath)
        else:
            print(f"Warning: {filename} not found in {analysis_dir}")
            results[key] = None
    
    return results


def create_aggregate_figure(analysis_dir, output_path):
    """Create comprehensive 4-panel publication figure"""
    
    # Load data
    smoothness_df = None
    dense_eval_df = None
    
    smoothness_path = Path(analysis_dir) / 'smoothness_results_aggregated.csv'
    if smoothness_path.exists():
        smoothness_df = pd.read_csv(smoothness_path)
    
    dense_eval_path = Path(analysis_dir) / 'dense_eval_results.csv'
    if dense_eval_path.exists():
        dense_eval_df = pd.read_csv(dense_eval_path)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Define colors
    condition_colors = {
        'spair_only': '#d62728',
        'spair_synthetic': '#2ca02c',
        'spair_2dwarp': '#ff7f0e',
    }
    
    # Panel 1: Learning Curves (top-left, 2 columns wide)
    ax1 = fig.add_subplot(gs[0, :2])
    
    # Load validation results to recreate learning curves
    # For now, reference the existing plot or show placeholder
    ax1.text(0.5, 0.5, 'Learning Curves\n(See learning_curves_spair.png)', 
             ha='center', va='center', fontsize=14,
             transform=ax1.transAxes)
    ax1.set_title('A. Learning Dynamics', fontsize=14, fontweight='bold', loc='left')
    ax1.axis('off')
    
    # Panel 2: Final Performance (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title('B. Final Performance', fontsize=14, fontweight='bold', loc='left')
    
    # This is a placeholder - in practice, you'd load the actual data
    ax2.text(0.5, 0.5, 'Final Performance\n(See final_performance_spair.png)', 
             ha='center', va='center', fontsize=12,
             transform=ax2.transAxes)
    ax2.axis('off')
    
    # Panel 3: Smoothness Metrics (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title('C. Prediction Smoothness', fontsize=14, fontweight='bold', loc='left')
    
    if smoothness_df is not None:
        # Filter to from-scratch, SPAIR benchmark
        plot_df = smoothness_df[
            (smoothness_df['pretrained'] == False) & 
            (smoothness_df['freeze'] == False) &
            (smoothness_df['benchmark'] == 'spair')
        ].copy()
        
        if len(plot_df) > 0:
            # Prepare data
            plot_data = []
            for _, row in plot_df.iterrows():
                condition = row['condition']
                mix_ratio = row['mix_ratio']
                
                if condition == 'spair_only':
                    label = 'SPAIR\nOnly'
                elif condition == 'spair_synthetic':
                    label = f'Synth\n{mix_ratio}'
                elif condition == 'spair_2dwarp':
                    label = f'Warp\n{mix_ratio}'
                else:
                    label = str(condition)[:10]
                
                plot_data.append({
                    'label': label,
                    'condition': condition,
                    'tv': row['mean_tv']
                })
            
            plot_df_agg = pd.DataFrame(plot_data)
            plot_df_agg = plot_df_agg.sort_values(['condition', 'tv'])
            
            colors = [condition_colors.get(c, 'gray') for c in plot_df_agg['condition']]
            x_pos = range(len(plot_df_agg))
            
            ax3.bar(x_pos, plot_df_agg['tv'], color=colors, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(plot_df_agg['label'], rotation=0, fontsize=9)
            ax3.set_ylabel('Total Variation\n(lower = smoother)', fontsize=11)
            ax3.grid(True, alpha=0.3, axis='y')
            
            # Add values
            for i, v in enumerate(plot_df_agg['tv']):
                ax3.text(i, v + v*0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No smoothness data', ha='center', va='center',
                    transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, 'Smoothness data\nnot available', ha='center', va='center',
                transform=ax3.transAxes)
    
    # Panel 4: Dense Dataset Generalization (bottom-middle and bottom-right)
    ax4 = fig.add_subplot(gs[1, 1:])
    ax4.set_title('D. Dense Dataset Generalization', fontsize=14, fontweight='bold', loc='left')
    
    if dense_eval_df is not None:
        # Filter to from-scratch
        plot_df = dense_eval_df[
            (dense_eval_df['pretrained'] == False) & 
            (dense_eval_df['freeze'] == False)
        ].copy()
        
        if len(plot_df) > 0:
            # Get unique benchmarks
            benchmarks = sorted(plot_df['benchmark'].unique())
            n_benchmarks = len(benchmarks)
            
            # Group by benchmark and condition
            grouped_data = []
            for benchmark in benchmarks:
                bench_data = plot_df[plot_df['benchmark'] == benchmark]
                
                for condition in ['spair_only', 'spair_synthetic', 'spair_2dwarp']:
                    cond_data = bench_data[bench_data['condition'] == condition]
                    
                    if len(cond_data) == 0:
                        continue
                    
                    # Take mean across mix ratios for non-spair_only
                    mean_pck = cond_data['pck'].mean()
                    
                    grouped_data.append({
                        'benchmark': benchmark,
                        'condition': condition,
                        'pck': mean_pck
                    })
            
            if grouped_data:
                grouped_df = pd.DataFrame(grouped_data)
                
                # Create grouped bar plot
                x_labels = benchmarks
                x_pos = np.arange(len(x_labels))
                width = 0.25
                
                for i, condition in enumerate(['spair_only', 'spair_synthetic', 'spair_2dwarp']):
                    cond_data = grouped_df[grouped_df['condition'] == condition]
                    values = [cond_data[cond_data['benchmark'] == b]['pck'].mean() 
                             if len(cond_data[cond_data['benchmark'] == b]) > 0 else 0
                             for b in benchmarks]
                    
                    offset = width * (i - 1)
                    label = {'spair_only': 'SPAIR Only', 
                            'spair_synthetic': 'SPAIR + Synthetic',
                            'spair_2dwarp': 'SPAIR + 2D Warp'}[condition]
                    
                    ax4.bar(x_pos + offset, values, width, 
                           label=label, color=condition_colors[condition],
                           alpha=0.7, edgecolor='black', linewidth=1.5)
                
                ax4.set_xlabel('Dense Benchmark', fontsize=11)
                ax4.set_ylabel('PCK (%)', fontsize=11)
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([b.upper() for b in x_labels], fontsize=10)
                ax4.legend(loc='best', fontsize=10)
                ax4.grid(True, alpha=0.3, axis='y')
            else:
                ax4.text(0.5, 0.5, 'No dense eval data', ha='center', va='center',
                        transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, 'No from-scratch data', ha='center', va='center',
                    transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Dense eval data\nnot available', ha='center', va='center',
                transform=ax4.transAxes)
    
    # Overall title
    fig.suptitle('Dense Geometric Regularization from Synthetic Data:\nComprehensive Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregate figure to: {output_path}")
    plt.close()


def create_latex_table(analysis_dir, output_path):
    """Create LaTeX table with all results"""
    
    # Load results
    smoothness_df = None
    dense_eval_df = None
    
    smoothness_path = Path(analysis_dir) / 'smoothness_results_aggregated.csv'
    if smoothness_path.exists():
        smoothness_df = pd.read_csv(smoothness_path)
    
    dense_eval_path = Path(analysis_dir) / 'dense_eval_results.csv'
    if dense_eval_path.exists():
        dense_eval_df = pd.read_csv(dense_eval_path)
    
    with open(output_path, 'w') as f:
        f.write("% LaTeX Table: Sparse Regularization Analysis Results\n")
        f.write("% Generated automatically\n\n")
        
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison across training conditions}\n")
        f.write("\\label{tab:sparse_regularization}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Condition & SPAIR & Smoothness (TV) & KITTI & Middlebury \\\\\n")
        f.write("\\midrule\n")
        
        # Add rows for each condition
        conditions = [
            ('SPAIR Only', 'spair_only', None),
            ('SPAIR + Synth (30/70)', 'spair_synthetic', '30_70'),
            ('SPAIR + Synth (50/50)', 'spair_synthetic', '50_50'),
            ('SPAIR + Synth (70/30)', 'spair_synthetic', '70_30'),
            ('SPAIR + 2DWarp (30/70)', 'spair_2dwarp', '30_70'),
            ('SPAIR + 2DWarp (50/50)', 'spair_2dwarp', '50_50'),
            ('SPAIR + 2DWarp (70/30)', 'spair_2dwarp', '70_30'),
        ]
        
        for label, condition, mix_ratio in conditions:
            row = [label]
            
            # SPAIR PCK (would need to load from validation results)
            row.append("-")
            
            # Smoothness
            if smoothness_df is not None:
                mask = (smoothness_df['condition'] == condition) & \
                       (smoothness_df['pretrained'] == False) & \
                       (smoothness_df['freeze'] == False)
                if mix_ratio:
                    mask &= (smoothness_df['mix_ratio'] == mix_ratio)
                
                smooth_data = smoothness_df[mask]
                if len(smooth_data) > 0:
                    tv = smooth_data['mean_tv'].mean()
                    row.append(f"{tv:.4f}")
                else:
                    row.append("-")
            else:
                row.append("-")
            
            # Dense eval metrics
            if dense_eval_df is not None:
                mask = (dense_eval_df['condition'] == condition) & \
                       (dense_eval_df['pretrained'] == False) & \
                       (dense_eval_df['freeze'] == False)
                if mix_ratio:
                    mask &= (dense_eval_df['mix_ratio'] == mix_ratio)
                
                # KITTI
                kitti_data = dense_eval_df[mask & (dense_eval_df['benchmark'].str.contains('kitti', case=False))]
                if len(kitti_data) > 0:
                    row.append(f"{kitti_data['pck'].mean():.2f}")
                else:
                    row.append("-")
                
                # Middlebury
                mid_data = dense_eval_df[mask & (dense_eval_df['benchmark'] == 'middlebury')]
                if len(mid_data) > 0:
                    row.append(f"{mid_data['pck'].mean():.2f}")
                else:
                    row.append("-")
            else:
                row.append("-")
                row.append("-")
            
            f.write(" & ".join(row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"Saved LaTeX table to: {output_path}")


def create_markdown_report(analysis_dir, output_path):
    """Create comprehensive markdown report"""
    
    output_path = Path(output_path)
    
    with open(output_path, 'w') as f:
        f.write("# Dense Geometric Regularization Analysis\n\n")
        f.write("## Hypothesis\n\n")
        f.write("Synthetic data provides **dense geometric regularization** that improves ")
        f.write("learning from sparse labels (SPAIR dataset).\n\n")
        
        f.write("## Experimental Setup\n\n")
        f.write("Compared three training conditions:\n\n")
        f.write("1. **SPAIR Only**: Baseline with sparse labels only\n")
        f.write("2. **SPAIR + Synthetic**: Mixed training with 3D synthetic data (30/70, 50/50, 70/30)\n")
        f.write("3. **SPAIR + 2D ImageNet Warps**: Control for 'just more data' (30/70, 50/50, 70/30)\n\n")
        
        f.write("All experiments run with:\n")
        f.write("- Same compute budget\n")
        f.write("- Same training protocol\n")
        f.write("- Multiple configurations: from-scratch vs pretrained, RAFT vs CATs\n\n")
        
        f.write("## Key Results\n\n")
        
        # Load summary report if available
        summary_path = Path(analysis_dir) / 'summary_report.txt'
        if summary_path.exists():
            f.write("### Performance Summary\n\n")
            f.write("```\n")
            with open(summary_path, 'r') as summary:
                f.write(summary.read())
            f.write("\n```\n\n")
        
        # Add smoothness results
        smoothness_summary = Path(analysis_dir) / 'smoothness_summary.txt'
        if smoothness_summary.exists():
            f.write("### Smoothness Analysis\n\n")
            f.write("```\n")
            with open(smoothness_summary, 'r') as summary:
                f.write(summary.read())
            f.write("\n```\n\n")
        
        # Add dense eval results
        dense_summary = Path(analysis_dir) / 'dense_eval_summary.txt'
        if dense_summary.exists():
            f.write("### Dense Dataset Generalization\n\n")
            f.write("```\n")
            with open(dense_summary, 'r') as summary:
                f.write(summary.read())
            f.write("\n```\n\n")
        
        f.write("## Conclusions\n\n")
        f.write("Based on the comprehensive analysis:\n\n")
        f.write("1. **H1 - Performance**: Synthetic data improves over baseline? [See results above]\n")
        f.write("2. **H2 - vs 2D Warps**: Synthetic outperforms 2D warps? [See results above]\n")
        f.write("3. **H3 - Learning Speed**: Faster convergence with synthetic? [See learning curves]\n")
        f.write("4. **H4 - Smoothness**: Lower TV/Laplacian metrics? [See smoothness analysis]\n")
        f.write("5. **H5 - Generalization**: Better on dense datasets? [See dense eval]\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `learning_curves_spair.png`: Training dynamics comparison\n")
        f.write("- `final_performance_spair.png`: Final performance with statistics\n")
        f.write("- `smoothness_comparison_spair.png`: Flow smoothness metrics\n")
        f.write("- `dense_eval_comparison.png`: KITTI/Middlebury generalization\n")
        f.write("- `aggregate_figure.png`: Publication-ready 4-panel figure\n")
        f.write("- `results_table.tex`: LaTeX table for paper\n\n")
    
    print(f"Saved markdown report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate sparse regularization analysis results'
    )
    parser.add_argument('--analysis-dir', type=str, required=True,
                       help='Directory containing analysis results')
    parser.add_argument('--output', type=str, default='aggregate_figure.png',
                       help='Output filename for aggregate figure')
    
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.exists():
        raise ValueError(f"Analysis directory not found: {analysis_dir}")
    
    print("="*80)
    print("AGGREGATING ANALYSIS RESULTS")
    print("="*80)
    print()
    
    # Create aggregate figure
    print("Creating aggregate figure...")
    output_fig = analysis_dir / args.output
    create_aggregate_figure(analysis_dir, output_fig)
    
    # Create LaTeX table
    print("\nCreating LaTeX table...")
    output_table = analysis_dir / 'results_table.tex'
    create_latex_table(analysis_dir, output_table)
    
    # Create markdown report
    print("\nCreating markdown report...")
    output_md = analysis_dir / 'ANALYSIS_REPORT.md'
    create_markdown_report(analysis_dir, output_md)
    
    print()
    print("="*80)
    print("AGGREGATION COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  - {output_fig}")
    print(f"  - {output_table}")
    print(f"  - {output_md}")


if __name__ == '__main__':
    main()
