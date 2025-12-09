#!/usr/bin/env python3
"""
Script to parse snapshot validation results and create plots.

Usage:
    python plot_snapshot_metrics.py <snapshot_path>
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Correlation analysis will be limited.")


def parse_training_dataset(snapshot_path):
    """Extract training dataset name from training_summary.txt"""
    summary_path = os.path.join(snapshot_path, 'training_summary.txt')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"training_summary.txt not found in {snapshot_path}")
    
    with open(summary_path, 'r') as f:
        for line in f:
            if line.startswith('Train dataset:'):
                return line.split(':', 1)[1].strip()
    
    raise ValueError("Could not find 'Train dataset:' in training_summary.txt")


def load_validation_results(snapshot_path):
    """Load validation results CSV"""
    csv_path = os.path.join(snapshot_path, 'validation_results.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"validation_results.csv not found in {snapshot_path}")
    
    df = pd.read_csv(csv_path)
    
    # Convert empty strings to NaN for MMD columns
    mmd_cols = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
    for col in mmd_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def load_pairwise_mmd_comparisons(training_dataset, csv_path='flow_mmd_results.csv'):
    """
    Load pairwise MMD² comparisons from CSV file and find all comparisons
    involving the training dataset. Uses train split for training dataset.
    
    Args:
        training_dataset: Name of the training dataset (e.g., 'synthetic', 'pointodyssey')
        csv_path: Path to the CSV file with pairwise comparisons
        
    Returns:
        Dictionary mapping benchmark name -> mmd2 value for training vs benchmark comparison
        Uses train split for training dataset to compare with benchmark eval sets
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Skipping pairwise MMD² comparisons.")
        return {}
    
    df = pd.read_csv(csv_path)
    pairwise_mmd = {}
    
    # Normalize training dataset name for matching (case-insensitive comparison)
    training_lower = str(training_dataset).lower()
    
    # Check if CSV has split columns (new format)
    has_splits = 'split1' in df.columns and 'split2' in df.columns
    
    if has_splits:
        # New format with splits: prioritize train split for training dataset
        training_dataset_train = f"{training_lower}_train"
        
        for _, row in df.iterrows():
            dataset1 = str(row['dataset1']).lower()
            dataset2 = str(row['dataset2']).lower()
            split1 = str(row['split1']).lower()
            split2 = str(row['split2']).lower()
            
            dataset1_id = f"{dataset1}_{split1}"
            dataset2_id = f"{dataset2}_{split2}"
            
            # Match training dataset with train split
            if dataset1_id == training_dataset_train:
                # training_dataset_train vs dataset2_split2
                eval_set = row['dataset2']  # Use dataset name without split for benchmark lookup
                pairwise_mmd[eval_set] = row['mmd2']
            elif dataset2_id == training_dataset_train:
                # dataset1_split1 vs training_dataset_train
                eval_set = row['dataset1']  # Use dataset name without split for benchmark lookup
                pairwise_mmd[eval_set] = row['mmd2']
            # Also try matching without explicit split (for backward compatibility)
            elif dataset1 == training_lower and split1 == 'train':
                eval_set = row['dataset2']
                pairwise_mmd[eval_set] = row['mmd2']
            elif dataset2 == training_lower and split2 == 'train':
                eval_set = row['dataset1']
                pairwise_mmd[eval_set] = row['mmd2']
    else:
        # Old format without splits: use exact match
        for _, row in df.iterrows():
            dataset1_lower = str(row['dataset1']).lower()
            dataset2_lower = str(row['dataset2']).lower()
            
            if dataset1_lower == training_lower:
                # training_dataset vs dataset2
                eval_set = row['dataset2']
                pairwise_mmd[eval_set] = row['mmd2']
            elif dataset2_lower == training_lower:
                # dataset1 vs training_dataset
                eval_set = row['dataset1']
                pairwise_mmd[eval_set] = row['mmd2']
    
    return pairwise_mmd


def compute_minmax_normalized_pck(df, use_baseline=True):
    """
    Compute min-max (0-1) normalized PCK across benchmarks.
    
    Args:
        df: DataFrame with validation results
        use_baseline: If True, normalize relative to epoch 0 (baseline).
                      If False, normalize using all epochs' min/max.
    
    Returns:
        DataFrame with added 'pck_minmax' column
        Dictionary of benchmark statistics
    """
    df = df.copy()
    
    # Check if 'epoch' column exists (for backward compatibility)
    has_epoch = 'epoch' in df.columns
    
    # Compute per-benchmark statistics
    benchmark_stats = {}
    benchmarks = df['benchmark'].unique()
    
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]['pck']
        
        if use_baseline and has_epoch:
            # Use epoch 0 as baseline (min)
            baseline_data = df[(df['benchmark'] == benchmark) & (df['epoch'] == 0)]['pck']
            if len(baseline_data) > 0:
                baseline_min = baseline_data.iloc[0]  # Use epoch 0 value as min
                baseline_max = bench_data.max()  # Use max across all epochs
            else:
                # Fallback: use first available epoch
                if 'training_steps' in df.columns:
                    first_step_data = df[df['benchmark'] == benchmark].sort_values('training_steps')
                    if len(first_step_data) > 0:
                        baseline_min = first_step_data['pck'].iloc[0]
                    else:
                        baseline_min = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
                else:
                    baseline_min = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
                baseline_max = bench_data.max()
        elif use_baseline and not has_epoch:
            # No epoch column, use first row by training_steps or first available
            if 'training_steps' in df.columns:
                first_step_data = df[df['benchmark'] == benchmark].sort_values('training_steps')
                if len(first_step_data) > 0:
                    baseline_min = first_step_data['pck'].iloc[0]
                else:
                    baseline_min = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
            else:
                baseline_min = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
            baseline_max = bench_data.max()
        else:
            # Use all epochs' statistics
            baseline_min = bench_data.min()
            baseline_max = bench_data.max()
        
        benchmark_stats[benchmark] = {
            'min': baseline_min,
            'max': baseline_max,
            'range': baseline_max - baseline_min if baseline_max > baseline_min else 1.0,  # Avoid division by zero
        }
    
    # Normalize PCK per benchmark using min-max: (x - min) / (max - min)
    normalized_pck = []
    for _, row in df.iterrows():
        benchmark = row['benchmark']
        pck = row['pck']
        stats = benchmark_stats[benchmark]
        
        # Min-max normalization: (x - min) / (max - min)
        if stats['range'] > 0:
            normalized = (pck - stats['min']) / stats['range']
        else:
            normalized = 0.0  # All values are the same
        
        normalized_pck.append(normalized)
    
    df['pck_minmax'] = normalized_pck
    return df, benchmark_stats


def compute_zscore_normalized_pck(df, use_baseline=True):
    """
    Compute z-score normalized PCK across benchmarks.
    
    Args:
        df: DataFrame with validation results
        use_baseline: If True, normalize relative to epoch 0 (baseline).
                      If False, normalize using all epochs' statistics.
    
    Returns:
        DataFrame with added 'pck_zscore' column
        Dictionary of benchmark statistics
    """
    df = df.copy()
    
    # Check if 'epoch' column exists (for backward compatibility)
    has_epoch = 'epoch' in df.columns
    
    # Compute per-benchmark statistics
    benchmark_stats = {}
    benchmarks = df['benchmark'].unique()
    
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]['pck']
        
        if use_baseline and has_epoch:
            # Use epoch 0 as baseline
            baseline_data = df[(df['benchmark'] == benchmark) & (df['epoch'] == 0)]['pck']
            if len(baseline_data) > 0:
                baseline_mean = baseline_data.iloc[0]  # Use epoch 0 value as mean
                baseline_std = bench_data.std()  # Use std across all epochs
            else:
                # Fallback: use first available epoch (by training_steps)
                if 'training_steps' in df.columns:
                    first_step_data = df[df['benchmark'] == benchmark].sort_values('training_steps')
                    if len(first_step_data) > 0:
                        baseline_mean = first_step_data['pck'].iloc[0]
                    else:
                        baseline_mean = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
                else:
                    baseline_mean = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
                baseline_std = bench_data.std()
        elif use_baseline and not has_epoch:
            # No epoch column, use first row by training_steps or first available
            if 'training_steps' in df.columns:
                first_step_data = df[df['benchmark'] == benchmark].sort_values('training_steps')
                if len(first_step_data) > 0:
                    baseline_mean = first_step_data['pck'].iloc[0]
                else:
                    baseline_mean = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
            else:
                baseline_mean = bench_data.iloc[0] if len(bench_data) > 0 else 0.0
            baseline_std = bench_data.std()
        else:
            # Use all epochs' statistics
            baseline_mean = bench_data.mean()
            baseline_std = bench_data.std()
        
        benchmark_stats[benchmark] = {
            'mean': baseline_mean,
            'std': baseline_std if baseline_std > 0 else 1.0,  # Avoid division by zero
        }
    
    # Standardize PCK per benchmark using z-score
    standardized_pck = []
    for _, row in df.iterrows():
        benchmark = row['benchmark']
        pck = row['pck']
        stats = benchmark_stats[benchmark]
        
        # Z-score normalization: (x - mean) / std
        standardized = (pck - stats['mean']) / stats['std']
        standardized_pck.append(standardized)
    
    df['pck_zscore'] = standardized_pck
    return df, benchmark_stats


def _plot_with_first_highlight(ax, x, y, label=None, color=None, marker='o', markersize=3, linewidth=1.5):
    """
    Plot a line/marker series and highlight the first point with a bolder marker to show direction.
    """
    line = ax.plot(
        x, y, marker=marker, label=label, markersize=markersize,
        linewidth=linewidth, color=color, alpha=0.8
    )[0]
    first_color = line.get_color()
    # Highlight the first point (direction origin) with a bold marker
    if len(x) > 0:
        ax.scatter(
            x.iloc[0], y.iloc[0],
            s=(markersize * 3.0) ** 2,
            facecolor=first_color,
            edgecolor='k',
            linewidth=1.0,
            zorder=5
        )
    return line


def create_combined_summary_figure(df, df_normalized, training_dataset, benchmarks, correlations, normalization_type='zscore', pairwise_mmd_dict=None):
    """Create a combined 12-plot summary figure with both normalized and unnormalized PCK
    
    Args:
        normalization_type: 'zscore' or 'minmax' - type of normalization used
        pairwise_mmd_dict: Optional dictionary mapping benchmark -> mmd2 value for training vs benchmark comparisons
    """
    
    # Create figure with 12 subplots (3 rows, 4 columns)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    norm_label = 'Z-score' if normalization_type == 'zscore' else 'Min-Max (0-1)'
    fig.suptitle(f'Training Dataset: {training_dataset}', fontsize=16, fontweight='bold')
    
    # Determine which normalized column to use
    pck_norm_col = 'pck_zscore' if normalization_type == 'zscore' else 'pck_minmax'
    pck_norm_label = 'PCK (Z-score)' if normalization_type == 'zscore' else 'PCK (Min-Max)'
    has_normalized = pck_norm_col in df_normalized.columns
    
    # Row 1: Training steps vs metrics
    # Plot 1: training_steps vs pck (raw)
    ax = axes[0, 0]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        ax.plot(bench_data['training_steps'], bench_data['pck'], 
                marker='o', label=benchmark, markersize=3, linewidth=1.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('PCK (%)')
    ax.set_title('Training Steps vs PCK (Raw)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: training_steps vs pck (normalized) - if available
    ax = axes[0, 1]
    if has_normalized:
        for benchmark in benchmarks:
            bench_data = df_normalized[df_normalized['benchmark'] == benchmark]
            ax.plot(bench_data['training_steps'], bench_data[pck_norm_col], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(pck_norm_label)
        ax.set_title(f'Training Steps vs PCK ({norm_label})')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        if normalization_type == 'zscore':
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        elif normalization_type == 'minmax':
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.axis('off')
    
    # Plot 3: training_steps vs mmd2_pred_corr_vs_pred_miss
    ax = axes[0, 2]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_pred_miss'])
        if len(valid_data) > 0:
            ax.plot(valid_data['training_steps'], valid_data['mmd2_pred_corr_vs_pred_miss'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('MMD² (pred_corr vs pred_miss)')
    ax.set_title('Training Steps vs MMD² (pred_corr vs pred_miss)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: training_steps vs mmd2_pred_corr_vs_gt
    ax = axes[0, 3]
    benchmark_colors = {}
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_gt'])
        if len(valid_data) > 0:
            lines = ax.plot(valid_data['training_steps'], valid_data['mmd2_pred_corr_vs_gt'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
            benchmark_colors[benchmark] = lines[0].get_color()
    # Add horizontal dotted lines for training vs eval set MMD²
    if pairwise_mmd_dict:
        for benchmark in benchmarks:
            if benchmark in pairwise_mmd_dict:
                line_color = benchmark_colors.get(benchmark, 'gray')
                ax.axhline(y=pairwise_mmd_dict[benchmark], linestyle=':', linewidth=1.5, 
                          color=line_color, alpha=0.6)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('MMD² (pred_corr vs gt)')
    ax.set_title('Training Steps vs MMD² (pred_corr vs gt)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Row 2: Unnormalized PCK vs MMD²
    # Plot 5: PCK (unnormalized) vs mmd2_pred_corr_vs_pred_miss
    ax = axes[1, 0]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_pred_miss', 'pck'])
        if 'training_steps' in valid_data.columns:
            valid_data = valid_data.sort_values('training_steps')
        if len(valid_data) > 0:
            _plot_with_first_highlight(ax, valid_data['pck'], valid_data['mmd2_pred_corr_vs_pred_miss'],
                                       label=benchmark, markersize=3, linewidth=1.5)
    if correlations and HAS_SCIPY:
        if 'mmd2_pred_corr_vs_pred_miss' in correlations['unnormalized'] and 'overall' in correlations['unnormalized']['mmd2_pred_corr_vs_pred_miss']:
            corr_info = correlations['unnormalized']['mmd2_pred_corr_vs_pred_miss']['overall']
            corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('PCK (%)')
    ax.set_ylabel('MMD² (pred_corr vs pred_miss)')
    ax.set_title('PCK (Unnormalized) vs MMD² (pred_corr vs pred_miss)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 6: PCK (unnormalized) vs mmd2_pred_corr_vs_gt
    ax = axes[1, 1]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_gt', 'pck'])
        if 'training_steps' in valid_data.columns:
            valid_data = valid_data.sort_values('training_steps')
        if len(valid_data) > 0:
            _plot_with_first_highlight(ax, valid_data['pck'], valid_data['mmd2_pred_corr_vs_gt'],
                                       label=benchmark, markersize=3, linewidth=1.5)
    if correlations and HAS_SCIPY:
        if 'mmd2_pred_corr_vs_gt' in correlations['unnormalized'] and 'overall' in correlations['unnormalized']['mmd2_pred_corr_vs_gt']:
            corr_info = correlations['unnormalized']['mmd2_pred_corr_vs_gt']['overall']
            corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('PCK (%)')
    ax.set_ylabel('MMD² (pred_corr vs gt)')
    ax.set_title('PCK (Unnormalized) vs MMD² (pred_corr vs gt)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 7: PCK (unnormalized) vs mmd2_pred_miss_vs_gt
    ax = axes[1, 2]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_miss_vs_gt', 'pck'])
        if 'training_steps' in valid_data.columns:
            valid_data = valid_data.sort_values('training_steps')
        if len(valid_data) > 0:
            _plot_with_first_highlight(ax, valid_data['pck'], valid_data['mmd2_pred_miss_vs_gt'],
                                       label=benchmark, markersize=3, linewidth=1.5)
    if correlations and HAS_SCIPY:
        if 'mmd2_pred_miss_vs_gt' in correlations['unnormalized'] and 'overall' in correlations['unnormalized']['mmd2_pred_miss_vs_gt']:
            corr_info = correlations['unnormalized']['mmd2_pred_miss_vs_gt']['overall']
            corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('PCK (%)')
    ax.set_ylabel('MMD² (pred_miss vs gt)')
    ax.set_title('PCK (Unnormalized) vs MMD² (pred_miss vs gt)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Plot 8: training_steps vs mmd2_pred_miss_vs_gt
    ax = axes[1, 3]
    benchmark_colors = {}
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_miss_vs_gt'])
        if len(valid_data) > 0:
            lines = ax.plot(valid_data['training_steps'], valid_data['mmd2_pred_miss_vs_gt'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
            benchmark_colors[benchmark] = lines[0].get_color()
    # Add horizontal dotted lines for training vs eval set MMD²
    if pairwise_mmd_dict:
        for benchmark in benchmarks:
            if benchmark in pairwise_mmd_dict:
                line_color = benchmark_colors.get(benchmark, 'gray')
                ax.axhline(y=pairwise_mmd_dict[benchmark], linestyle=':', linewidth=1.5, 
                          color=line_color, alpha=0.6)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('MMD² (pred_miss vs gt)')
    ax.set_title('Training Steps vs MMD² (pred_miss vs gt)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # Row 3: Normalized PCK vs MMD²
    # Plot 9: PCK (normalized) vs mmd2_pred_corr_vs_pred_miss
    ax = axes[2, 0]
    if has_normalized:
        for benchmark in benchmarks:
            bench_data = df_normalized[df_normalized['benchmark'] == benchmark]
            valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_pred_miss', pck_norm_col])
            if 'training_steps' in valid_data.columns:
                valid_data = valid_data.sort_values('training_steps')
            if len(valid_data) > 0:
                _plot_with_first_highlight(ax, valid_data[pck_norm_col], valid_data['mmd2_pred_corr_vs_pred_miss'],
                                           label=benchmark, markersize=3, linewidth=1.5)
            if correlations and HAS_SCIPY:
                if 'mmd2_pred_corr_vs_pred_miss' in correlations['normalized'] and 'overall' in correlations['normalized']['mmd2_pred_corr_vs_pred_miss']:
                    corr_info = correlations['normalized']['mmd2_pred_corr_vs_pred_miss']['overall']
                    corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
                    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                            fontsize=7, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(pck_norm_label)
        ax.set_ylabel('MMD² (pred_corr vs pred_miss)')
        ax.set_title(f'PCK ({norm_label}) vs MMD² (pred_corr vs pred_miss)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        if normalization_type == 'zscore':
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        elif normalization_type == 'minmax':
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.axis('off')
    
    # Plot 10: PCK (normalized) vs mmd2_pred_corr_vs_gt
    ax = axes[2, 1]
    if has_normalized:
        for benchmark in benchmarks:
            bench_data = df_normalized[df_normalized['benchmark'] == benchmark]
            valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_gt', pck_norm_col])
            if 'training_steps' in valid_data.columns:
                valid_data = valid_data.sort_values('training_steps')
            if len(valid_data) > 0:
                _plot_with_first_highlight(ax, valid_data[pck_norm_col], valid_data['mmd2_pred_corr_vs_gt'],
                                           label=benchmark, markersize=3, linewidth=1.5)
            if correlations and HAS_SCIPY:
                if 'mmd2_pred_corr_vs_gt' in correlations['normalized'] and 'overall' in correlations['normalized']['mmd2_pred_corr_vs_gt']:
                    corr_info = correlations['normalized']['mmd2_pred_corr_vs_gt']['overall']
                    corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
                    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                           fontsize=7, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(pck_norm_label)
        ax.set_ylabel('MMD² (pred_corr vs gt)')
        ax.set_title(f'PCK ({norm_label}) vs MMD² (pred_corr vs gt)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        if normalization_type == 'zscore':
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        elif normalization_type == 'minmax':
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.axis('off')
    
    # Plot 11: PCK (normalized) vs mmd2_pred_miss_vs_gt
    ax = axes[2, 2]
    if has_normalized:
        for benchmark in benchmarks:
            bench_data = df_normalized[df_normalized['benchmark'] == benchmark]
            valid_data = bench_data.dropna(subset=['mmd2_pred_miss_vs_gt', pck_norm_col])
            if 'training_steps' in valid_data.columns:
                valid_data = valid_data.sort_values('training_steps')
            if len(valid_data) > 0:
                _plot_with_first_highlight(ax, valid_data[pck_norm_col], valid_data['mmd2_pred_miss_vs_gt'],
                                           label=benchmark, markersize=3, linewidth=1.5)
            if correlations and HAS_SCIPY:
                if 'mmd2_pred_miss_vs_gt' in correlations['normalized'] and 'overall' in correlations['normalized']['mmd2_pred_miss_vs_gt']:
                    corr_info = correlations['normalized']['mmd2_pred_miss_vs_gt']['overall']
                    corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
                    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                           fontsize=7, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlabel(pck_norm_label)
        ax.set_ylabel('MMD² (pred_miss vs gt)')
        ax.set_title(f'PCK ({norm_label}) vs MMD² (pred_miss vs gt)')
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        if normalization_type == 'zscore':
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        elif normalization_type == 'minmax':
            ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.axis('off')
    
    # Plot 12: Empty or additional plot
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    return fig


def create_single_summary_figure(df, df_plot, training_dataset, benchmarks, pck_col, pck_label, 
                                 correlations, corr_key, use_zscore, suffix='', pairwise_mmd_dict=None):
    """Create a single 8-plot summary figure with specified PCK column
    
    Args:
        pairwise_mmd_dict: Optional dictionary mapping benchmark -> mmd2 value for training vs benchmark comparisons
    """
    
    # Create figure with 8 subplots (2 rows, 4 columns)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title_suffix = f' ({suffix})' if suffix else ''
    fig.suptitle(f'Training Dataset: {training_dataset}{title_suffix}', fontsize=16, fontweight='bold')
    
    # Plot 1: training_steps vs pck (raw)
    ax = axes[0, 0]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        ax.plot(bench_data['training_steps'], bench_data['pck'], 
                marker='o', label=benchmark, markersize=3, linewidth=1.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('PCK (%)')
    ax.set_title('Training Steps vs PCK (Raw)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: training_steps vs mmd2_pred_corr_vs_pred_miss
    ax = axes[0, 1]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        # Filter out NaN values
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_pred_miss'])
        if len(valid_data) > 0:
            ax.plot(valid_data['training_steps'], valid_data['mmd2_pred_corr_vs_pred_miss'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('MMD² (pred_corr vs pred_miss)')
    ax.set_title('Training Steps vs MMD² (pred_corr vs pred_miss)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: training_steps vs mmd2_pred_corr_vs_gt
    ax = axes[0, 2]
    benchmark_colors = {}
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_gt'])
        if len(valid_data) > 0:
            lines = ax.plot(valid_data['training_steps'], valid_data['mmd2_pred_corr_vs_gt'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
            benchmark_colors[benchmark] = lines[0].get_color()
    # Add horizontal dotted lines for training vs eval set MMD²
    if pairwise_mmd_dict:
        for benchmark in benchmarks:
            if benchmark in pairwise_mmd_dict:
                line_color = benchmark_colors.get(benchmark, 'gray')
                ax.axhline(y=pairwise_mmd_dict[benchmark], linestyle=':', linewidth=1.5, 
                          color=line_color, alpha=0.6)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('MMD² (pred_corr vs gt)')
    ax.set_title('Training Steps vs MMD² (pred_corr vs gt)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: training_steps vs mmd2_pred_miss_vs_gt
    ax = axes[0, 3]
    benchmark_colors = {}
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_miss_vs_gt'])
        if len(valid_data) > 0:
            lines = ax.plot(valid_data['training_steps'], valid_data['mmd2_pred_miss_vs_gt'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
            benchmark_colors[benchmark] = lines[0].get_color()
    # Add horizontal dotted lines for training vs eval set MMD²
    if pairwise_mmd_dict:
        for benchmark in benchmarks:
            if benchmark in pairwise_mmd_dict:
                line_color = benchmark_colors.get(benchmark, 'gray')
                ax.axhline(y=pairwise_mmd_dict[benchmark], linestyle=':', linewidth=1.5, 
                          color=line_color, alpha=0.6)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('MMD² (pred_miss vs gt)')
    ax.set_title('Training Steps vs MMD² (pred_miss vs gt)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 5: pck vs mmd2_pred_corr_vs_pred_miss
    ax = axes[1, 0]
    for benchmark in benchmarks:
        bench_data = df_plot[df_plot['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_pred_miss', pck_col])
        if len(valid_data) > 0:
            ax.plot(valid_data[pck_col], valid_data['mmd2_pred_corr_vs_pred_miss'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
    
    # Add correlation annotation if available
    if correlations and HAS_SCIPY:
        if 'mmd2_pred_corr_vs_pred_miss' in correlations[corr_key] and 'overall' in correlations[corr_key]['mmd2_pred_corr_vs_pred_miss']:
            corr_info = correlations[corr_key]['mmd2_pred_corr_vs_pred_miss']['overall']
            corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(pck_label)
    ax.set_ylabel('MMD² (pred_corr vs pred_miss)')
    ax.set_title(f'{pck_label} vs MMD² (pred_corr vs pred_miss)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    if use_zscore:
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 6: pck vs mmd2_pred_corr_vs_gt
    ax = axes[1, 1]
    benchmark_colors = {}
    for benchmark in benchmarks:
        bench_data = df_plot[df_plot['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_corr_vs_gt', pck_col])
        if len(valid_data) > 0:
            lines = ax.plot(valid_data[pck_col], valid_data['mmd2_pred_corr_vs_gt'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
            benchmark_colors[benchmark] = lines[0].get_color()
    
    # Add horizontal dotted lines for training vs eval set MMD²
    if pairwise_mmd_dict:
        for benchmark in benchmarks:
            if benchmark in pairwise_mmd_dict:
                line_color = benchmark_colors.get(benchmark, 'gray')
                ax.axhline(y=pairwise_mmd_dict[benchmark], linestyle=':', linewidth=1.5, 
                          color=line_color, alpha=0.6)
    
    # Add correlation annotation if available
    if correlations and HAS_SCIPY:
        if 'mmd2_pred_corr_vs_gt' in correlations[corr_key] and 'overall' in correlations[corr_key]['mmd2_pred_corr_vs_gt']:
            corr_info = correlations[corr_key]['mmd2_pred_corr_vs_gt']['overall']
            corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(pck_label)
    ax.set_ylabel('MMD² (pred_corr vs gt)')
    ax.set_title(f'{pck_label} vs MMD² (pred_corr vs gt)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    if use_zscore:
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 7: pck vs mmd2_pred_miss_vs_gt
    ax = axes[1, 2]
    benchmark_colors = {}
    for benchmark in benchmarks:
        bench_data = df_plot[df_plot['benchmark'] == benchmark]
        valid_data = bench_data.dropna(subset=['mmd2_pred_miss_vs_gt', pck_col])
        if len(valid_data) > 0:
            lines = ax.plot(valid_data[pck_col], valid_data['mmd2_pred_miss_vs_gt'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
            benchmark_colors[benchmark] = lines[0].get_color()
    
    # Add horizontal dotted lines for training vs eval set MMD²
    if pairwise_mmd_dict:
        for benchmark in benchmarks:
            if benchmark in pairwise_mmd_dict:
                line_color = benchmark_colors.get(benchmark, 'gray')
                ax.axhline(y=pairwise_mmd_dict[benchmark], linestyle=':', linewidth=1.5, 
                          color=line_color, alpha=0.6)
    
    # Add correlation annotation if available
    if correlations and HAS_SCIPY:
        if 'mmd2_pred_miss_vs_gt' in correlations[corr_key] and 'overall' in correlations[corr_key]['mmd2_pred_miss_vs_gt']:
            corr_info = correlations[corr_key]['mmd2_pred_miss_vs_gt']['overall']
            corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(pck_label)
    ax.set_ylabel('MMD² (pred_miss vs gt)')
    ax.set_title(f'{pck_label} vs MMD² (pred_miss vs gt)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    if use_zscore:
        ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Plot 8: training_steps vs PCK (raw or normalized)
    ax = axes[1, 3]
    if use_zscore and pck_col == 'pck_zscore':
        for benchmark in benchmarks:
            bench_data = df_plot[df_plot['benchmark'] == benchmark]
            ax.plot(bench_data['training_steps'], bench_data['pck_zscore'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('PCK (Z-score)')
        ax.set_title('Training Steps vs PCK (Z-score normalized)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    else:
        # Show raw PCK over time
        for benchmark in benchmarks:
            bench_data = df[df['benchmark'] == benchmark]
            ax.plot(bench_data['training_steps'], bench_data['pck'], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('PCK (%)')
        ax.set_title('Training Steps vs PCK (Raw)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_plots(df, training_dataset, output_dir=None, normalization_type='zscore', correlations=None, df_normalized=None, benchmark_stats=None, pairwise_mmd_dict=None):
    """Create plots from validation results - creates both normalized and unnormalized versions
    
    Args:
        df: DataFrame with validation results
        training_dataset: Name of training dataset
        output_dir: Output directory for plots
        normalization_type: 'zscore', 'minmax', or None - type of normalization to use
        correlations: Optional pre-computed correlation statistics
        df_normalized: Optional pre-computed normalized dataframe
        benchmark_stats: Optional pre-computed benchmark statistics
        pairwise_mmd_dict: Optional dictionary mapping benchmark -> mmd2 value for training vs benchmark comparisons
    """
    
    use_normalization = normalization_type is not None
    
    # Compute normalized PCK if not provided
    if df_normalized is None:
        if normalization_type == 'zscore':
            df_normalized, benchmark_stats = compute_zscore_normalized_pck(df, use_baseline=True)
        elif normalization_type == 'minmax':
            df_normalized, benchmark_stats = compute_minmax_normalized_pck(df, use_baseline=True)
        else:
            df_normalized = df.copy()
            benchmark_stats = None
    
    # Get unique benchmarks
    benchmarks = df['benchmark'].unique()
    
    # Ensure output directory exists
    if output_dir is None:
        output_dir = '.'
    
    # Determine which normalized column exists
    pck_norm_col = None
    if normalization_type == 'zscore' and 'pck_zscore' in df_normalized.columns:
        pck_norm_col = 'pck_zscore'
    elif normalization_type == 'minmax' and 'pck_minmax' in df_normalized.columns:
        pck_norm_col = 'pck_minmax'
    
    # Create COMBINED summary figure with all 12 plots (default)
    if pck_norm_col:
        fig_combined = create_combined_summary_figure(
            df=df,
            df_normalized=df_normalized,
            training_dataset=training_dataset,
            benchmarks=benchmarks,
            correlations=correlations,
            normalization_type=normalization_type,
            pairwise_mmd_dict=pairwise_mmd_dict
        )
        output_path_default = os.path.join(output_dir, f'{training_dataset}_metrics.png')
        fig_combined.savefig(output_path_default, dpi=150, bbox_inches='tight')
        print(f"Saved combined (12-plot) summary to: {output_path_default}")
        plt.close(fig_combined)
        default_fig = fig_combined
    else:
        # If no normalization available, create combined figure without normalized plots
        fig_combined = create_combined_summary_figure(
            df=df,
            df_normalized=df.copy(),
            training_dataset=training_dataset,
            benchmarks=benchmarks,
            correlations=correlations,
            normalization_type=None,
            pairwise_mmd_dict=pairwise_mmd_dict
        )
        output_path_default = os.path.join(output_dir, f'{training_dataset}_metrics.png')
        fig_combined.savefig(output_path_default, dpi=150, bbox_inches='tight')
        print(f"Saved combined (12-plot) summary to: {output_path_default}")
        plt.close(fig_combined)
        default_fig = fig_combined
    
    # Also create separate unnormalized and normalized figures for reference
    fig_unnorm = create_single_summary_figure(
        df=df,
        df_plot=df,
        training_dataset=training_dataset,
        benchmarks=benchmarks,
        pck_col='pck',
        pck_label='PCK (%)',
        correlations=correlations,
        corr_key='unnormalized',
        use_zscore=False,
        suffix='Unnormalized',
        pairwise_mmd_dict=pairwise_mmd_dict
    )
    output_path_unnorm = os.path.join(output_dir, f'{training_dataset}_metrics_unnormalized.png')
    fig_unnorm.savefig(output_path_unnorm, dpi=150, bbox_inches='tight')
    print(f"Saved unnormalized plots to: {output_path_unnorm}")
    plt.close(fig_unnorm)
    
    # Create NORMALIZED summary figure (if normalized data is available)
    if pck_norm_col:
        norm_label = 'Z-score' if normalization_type == 'zscore' else 'Min-Max (0-1)'
        pck_label = f'PCK ({norm_label})'
        fig_norm = create_single_summary_figure(
            df=df,
            df_plot=df_normalized,
            training_dataset=training_dataset,
            benchmarks=benchmarks,
            pck_col=pck_norm_col,
            pck_label=pck_label,
            correlations=correlations,
            corr_key='normalized',
            use_zscore=(normalization_type == 'zscore'),
            suffix=f'Normalized ({norm_label})',
            pairwise_mmd_dict=pairwise_mmd_dict
        )
        output_path_norm = os.path.join(output_dir, f'{training_dataset}_metrics_normalized.png')
        fig_norm.savefig(output_path_norm, dpi=150, bbox_inches='tight')
        print(f"Saved normalized plots to: {output_path_norm}")
        plt.close(fig_norm)
    
    return default_fig
    
    # Also save individual plots
    save_individual_plots(df, df_normalized, training_dataset, output_dir, benchmarks, use_zscore)
    
    # Save standardized comparison plot if normalization is enabled
    if normalization_type and benchmark_stats:
        create_standardized_comparison_plot(df, df_normalized, training_dataset, output_dir, benchmark_stats, normalization_type)
    
    # Compute correlation statistics (needed for plot annotations)
    print("\nComputing correlation statistics...")
    correlations = compute_correlation_stats(df, df_normalized, normalization_type)
    
    # Create correlation analysis plots
    create_correlation_analysis_plots(df, df_normalized, training_dataset, output_dir, correlations, normalization_type)
    
    # Print summary of correlations
    if HAS_SCIPY:
        print("\nCorrelation Summary:")
        print("-" * 60)
        mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
        mmd_labels = {
            'mmd2_pred_corr_vs_pred_miss': 'MMD² (pred_corr vs pred_miss)',
            'mmd2_pred_corr_vs_gt': 'MMD² (pred_corr vs gt)',
            'mmd2_pred_miss_vs_gt': 'MMD² (pred_miss vs gt)'
        }
        
        for mmd_col in mmd_metrics:
            print(f"\n{mmd_labels[mmd_col]}:")
            if mmd_col in correlations['unnormalized'] and 'overall' in correlations['unnormalized'][mmd_col]:
                corr_info = correlations['unnormalized'][mmd_col]['overall']
                print(f"  Unnormalized: r={corr_info['corr']:.4f}, p={corr_info['p_value']:.4f}")
            if use_zscore and mmd_col in correlations['normalized'] and 'overall' in correlations['normalized'][mmd_col]:
                corr_info = correlations['normalized'][mmd_col]['overall']
                print(f"  Normalized:   r={corr_info['corr']:.4f}, p={corr_info['p_value']:.4f}")
    
    return fig


def save_individual_plots(df, df_normalized, training_dataset, output_dir, benchmarks, use_zscore=True):
    """Save each plot as a separate figure"""
    
    # Use normalized PCK for PCK-related plots if z-score is enabled
    pck_col = 'pck_zscore' if use_zscore else 'pck'
    pck_label = 'PCK (Z-score)' if use_zscore else 'PCK (%)'
    
    plots = [
        ('training_steps', 'pck', 'Training Steps', 'PCK (%)', 'Training Steps vs PCK (Raw)', df),
        ('training_steps', 'mmd2_pred_corr_vs_pred_miss', 'Training Steps', 'MMD² (pred_corr vs pred_miss)', 
         'Training Steps vs MMD² (pred_corr vs pred_miss)', df),
        ('training_steps', 'mmd2_pred_corr_vs_gt', 'Training Steps', 'MMD² (pred_corr vs gt)', 
         'Training Steps vs MMD² (pred_corr vs gt)', df),
        ('training_steps', 'mmd2_pred_miss_vs_gt', 'Training Steps', 'MMD² (pred_miss vs gt)', 
         'Training Steps vs MMD² (pred_miss vs gt)', df),
        (pck_col, 'mmd2_pred_corr_vs_pred_miss', pck_label, 'MMD² (pred_corr vs pred_miss)', 
         f'{pck_label} vs MMD² (pred_corr vs pred_miss)', df_normalized),
        (pck_col, 'mmd2_pred_corr_vs_gt', pck_label, 'MMD² (pred_corr vs gt)', 
         f'{pck_label} vs MMD² (pred_corr vs gt)', df_normalized),
        (pck_col, 'mmd2_pred_miss_vs_gt', pck_label, 'MMD² (pred_miss vs gt)', 
         f'{pck_label} vs MMD² (pred_miss vs gt)', df_normalized),
    ]
    
    # Add standardized PCK plot if enabled
    if use_zscore:
        plots.append(('training_steps', 'pck_zscore', 'Training Steps', 'PCK (Z-score)', 
                     'Training Steps vs PCK (Z-score normalized)', df_normalized))
    
    for x_col, y_col, x_label, y_label, title, data_df in plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for benchmark in benchmarks:
            bench_data = data_df[data_df['benchmark'] == benchmark]
            # Filter out NaN values for y column
            valid_data = bench_data.dropna(subset=[y_col])
            if len(valid_data) > 0:
                ax.plot(valid_data[x_col], valid_data[y_col], 
                        marker='o', label=benchmark, markersize=4, linewidth=2)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f'{title}\nTraining Dataset: {training_dataset}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add reference lines for z-score plots
        if use_zscore and 'zscore' in title.lower():
            if x_col == 'pck_zscore':
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            if y_col == 'pck_zscore':
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        
        # Save individual plot
        safe_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('²', '2')
        output_path = os.path.join(output_dir, f'{training_dataset}_{safe_title}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved individual plots to: {output_dir}")


def create_standardized_comparison_plot(df, df_normalized, training_dataset, output_dir, benchmark_stats, normalization_type='zscore'):
    """Create comparison plot showing raw vs standardized PCK"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    benchmarks = df['benchmark'].unique()
    
    # Plot 1: Raw PCK over time
    ax = axes[0]
    for benchmark in benchmarks:
        bench_data = df[df['benchmark'] == benchmark]
        ax.plot(bench_data['training_steps'], bench_data['pck'], 
                marker='o', label=benchmark, markersize=3, linewidth=1.5)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('PCK (%)')
    ax.set_title('Raw PCK Across Benchmarks')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Standardized PCK over time
    ax = axes[1]
    # Determine which normalized column to use
    pck_norm_col = 'pck_zscore' if normalization_type == 'zscore' else 'pck_minmax'
    pck_norm_label = 'PCK (Z-score)' if normalization_type == 'zscore' else 'PCK (Min-Max)'
    has_normalized = pck_norm_col in df_normalized.columns
    
    if has_normalized:
        for benchmark in benchmarks:
            bench_data = df_normalized[df_normalized['benchmark'] == benchmark]
            ax.plot(bench_data['training_steps'], bench_data[pck_norm_col], 
                    marker='o', label=benchmark, markersize=3, linewidth=1.5)
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(pck_norm_label)
        norm_label = 'Z-score' if normalization_type == 'zscore' else 'Min-Max (0-1)'
        ax.set_title(f'Standardized PCK ({norm_label} normalized, baseline=epoch 0)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        if normalization_type == 'zscore':
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
        elif normalization_type == 'minmax':
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=1)
    else:
        ax.text(0.5, 0.5, 'Normalized PCK not available', 
               transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title('Standardized PCK (N/A)', fontsize=14, fontweight='bold')
    
    # Add benchmark statistics as text
    if benchmark_stats:
        norm_label = 'Z-score' if normalization_type == 'zscore' else 'Min-Max'
        stats_text = f"Benchmark Statistics ({norm_label}, baseline=epoch 0):\n"
        for benchmark in sorted(benchmarks):
            if benchmark in benchmark_stats:
                stats = benchmark_stats[benchmark]
                if normalization_type == 'zscore':
                    stats_text += f"{benchmark}: μ={stats['mean']:.1f}, σ={stats['std']:.1f}\n"
                else:
                    stats_text += f"{benchmark}: min={stats['min']:.1f}, max={stats['max']:.1f}\n"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{training_dataset}_standardized_pck_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved standardized PCK comparison to: {output_path}")
    plt.close()


def compute_correlation_stats(df, df_normalized, normalization_type='zscore'):
    """
    Compute correlation between PCK and MMD² metrics for both normalized and un-normalized PCK.
    
    Args:
        df: DataFrame with raw validation results
        df_normalized: DataFrame with normalized PCK
        normalization_type: 'zscore' or 'minmax' - type of normalization used
    
    Returns:
        Dictionary with correlation statistics for both normalized and un-normalized PCK
    """
    correlations = {
        'normalized': {},
        'unnormalized': {}
    }
    
    benchmarks = df['benchmark'].unique()
    mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
    
    if not HAS_SCIPY:
        return correlations
    
    # Determine which normalized column to use
    pck_norm_col = 'pck_zscore' if normalization_type == 'zscore' else 'pck_minmax'
    
    # Analyze unnormalized PCK
    for mmd_col in mmd_metrics:
        correlations['unnormalized'][mmd_col] = {}
        
        # Overall correlation (all benchmarks combined)
        combined_data = df.dropna(subset=['pck', mmd_col])
        if len(combined_data) > 2:
            corr, p_value = stats.pearsonr(combined_data['pck'], combined_data[mmd_col])
            correlations['unnormalized'][mmd_col]['overall'] = {
                'corr': corr, 
                'p_value': p_value, 
                'n': len(combined_data)
            }
        
        # Per-benchmark correlations
        for benchmark in benchmarks:
            bench_data = df[(df['benchmark'] == benchmark)].dropna(subset=['pck', mmd_col])
            if len(bench_data) > 2:
                corr, p_value = stats.pearsonr(bench_data['pck'], bench_data[mmd_col])
                correlations['unnormalized'][mmd_col][benchmark] = {
                    'corr': corr, 
                    'p_value': p_value, 
                    'n': len(bench_data)
                }
    
    # Analyze normalized PCK (if available)
    if pck_norm_col in df_normalized.columns:
        for mmd_col in mmd_metrics:
            correlations['normalized'][mmd_col] = {}
            
            # Overall correlation (all benchmarks combined)
            combined_data = df_normalized.dropna(subset=[pck_norm_col, mmd_col])
            if len(combined_data) > 2:
                corr, p_value = stats.pearsonr(combined_data[pck_norm_col], combined_data[mmd_col])
                correlations['normalized'][mmd_col]['overall'] = {
                    'corr': corr, 
                    'p_value': p_value, 
                    'n': len(combined_data)
                }
            
            # Per-benchmark correlations
            for benchmark in benchmarks:
                bench_data = df_normalized[(df_normalized['benchmark'] == benchmark)].dropna(subset=[pck_norm_col, mmd_col])
                if len(bench_data) > 2:
                    corr, p_value = stats.pearsonr(bench_data[pck_norm_col], bench_data[mmd_col])
                    correlations['normalized'][mmd_col][benchmark] = {
                        'corr': corr, 
                        'p_value': p_value, 
                        'n': len(bench_data)
                    }
    
    return correlations


def create_correlation_analysis_plots(df, df_normalized, training_dataset, output_dir, correlations, normalization_type='zscore'):
    """Create comprehensive correlation analysis plots for both normalized and un-normalized PCK"""
    
    benchmarks = df['benchmark'].unique()
    mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
    mmd_labels = {
        'mmd2_pred_corr_vs_pred_miss': 'MMD² (pred_corr vs pred_miss)',
        'mmd2_pred_corr_vs_gt': 'MMD² (pred_corr vs gt)',
        'mmd2_pred_miss_vs_gt': 'MMD² (pred_miss vs gt)'
    }
    
    # Determine which normalized column to use
    pck_norm_col = 'pck_zscore' if normalization_type == 'zscore' else 'pck_minmax'
    pck_norm_label = 'PCK (Z-score)' if normalization_type == 'zscore' else 'PCK (Min-Max)'
    has_normalized = pck_norm_col in df_normalized.columns
    
    # Create comparison plots: unnormalized vs normalized side by side
    for mmd_col in mmd_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        mmd_label = mmd_labels[mmd_col]
        
        # Plot 1: Unnormalized PCK vs MMD²
        ax = axes[0]
        for benchmark in benchmarks:
            bench_data = df[df['benchmark'] == benchmark]
            valid_data = bench_data.dropna(subset=['pck', mmd_col])
            if len(valid_data) > 0:
                ax.plot(valid_data['pck'], valid_data[mmd_col], 
                       marker='o', label=benchmark, markersize=4, linewidth=2, alpha=0.7)
        
        # Add correlation info
        if mmd_col in correlations['unnormalized'] and 'overall' in correlations['unnormalized'][mmd_col]:
            corr_info = correlations['unnormalized'][mmd_col]['overall']
            corr_text = f"Overall: r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
            ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('PCK (%)', fontsize=12)
        ax.set_ylabel(mmd_label, fontsize=12)
        ax.set_title(f'Unnormalized PCK vs {mmd_label}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Normalized PCK vs MMD² (if available)
        ax = axes[1]
        pck_norm_col = 'pck_zscore' if normalization_type == 'zscore' else 'pck_minmax'
        pck_norm_label = 'PCK (Z-score)' if normalization_type == 'zscore' else 'PCK (Min-Max)'
        has_normalized = pck_norm_col in df_normalized.columns
        
        if has_normalized:
            for benchmark in benchmarks:
                bench_data = df_normalized[df_normalized['benchmark'] == benchmark]
                valid_data = bench_data.dropna(subset=[pck_norm_col, mmd_col])
                if len(valid_data) > 0:
                    ax.plot(valid_data[pck_norm_col], valid_data[mmd_col], 
                           marker='o', label=benchmark, markersize=4, linewidth=2, alpha=0.7)
            
            # Add correlation info
            if mmd_col in correlations['normalized'] and 'overall' in correlations['normalized'][mmd_col]:
                corr_info = correlations['normalized'][mmd_col]['overall']
                corr_text = f"Overall: r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
                ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            if normalization_type == 'zscore':
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
            elif normalization_type == 'minmax':
                ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
                ax.axvline(x=1, color='r', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlabel(pck_norm_label, fontsize=12)
            ax.set_title(f'Normalized PCK vs {mmd_label}', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'Normalized PCK not available', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title('Normalized PCK vs MMD² (N/A)', fontsize=14, fontweight='bold')
        
        ax.set_ylabel(mmd_label, fontsize=12)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comparison plot
        safe_mmd = mmd_col.replace('mmd2_', '').replace('_', '_')
        output_path = os.path.join(output_dir, f'{training_dataset}_correlation_{safe_mmd}_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create correlation summary table/plot
    create_correlation_summary(correlations, training_dataset, output_dir, normalization_type)


def create_correlation_summary(correlations, training_dataset, output_dir, normalization_type='zscore'):
    """Create a summary plot/table showing all correlation coefficients"""
    
    if not HAS_SCIPY:
        return
    
    mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
    mmd_labels = {
        'mmd2_pred_corr_vs_pred_miss': 'pred_corr\nvs\npred_miss',
        'mmd2_pred_corr_vs_gt': 'pred_corr\nvs\ngt',
        'mmd2_pred_miss_vs_gt': 'pred_miss\nvs\ngt'
    }
    
    # Collect all benchmarks
    all_benchmarks = set()
    for mmd_col in mmd_metrics:
        if mmd_col in correlations['unnormalized']:
            all_benchmarks.update([k for k in correlations['unnormalized'][mmd_col].keys() if k != 'overall'])
    
    all_benchmarks = sorted(list(all_benchmarks))
    
    # Create figure with subplots for normalized and unnormalized
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, len(all_benchmarks) * 0.5)))
    
    # Plot 1: Unnormalized correlations
    ax = axes[0]
    corr_matrix = np.full((len(all_benchmarks) + 1, len(mmd_metrics)), np.nan)
    
    for j, mmd_col in enumerate(mmd_metrics):
        if mmd_col in correlations['unnormalized']:
            # Overall correlation
            if 'overall' in correlations['unnormalized'][mmd_col]:
                corr_matrix[0, j] = correlations['unnormalized'][mmd_col]['overall']['corr']
            
            # Per-benchmark correlations
            for i, benchmark in enumerate(all_benchmarks, 1):
                if benchmark in correlations['unnormalized'][mmd_col]:
                    corr_matrix[i, j] = correlations['unnormalized'][mmd_col][benchmark]['corr']
    
    im = ax.imshow(corr_matrix, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(mmd_metrics)))
    ax.set_xticklabels([mmd_labels[m] for m in mmd_metrics], fontsize=10)
    ax.set_yticks(range(len(all_benchmarks) + 1))
    ax.set_yticklabels(['Overall'] + all_benchmarks, fontsize=9)
    
    # Add text annotations
    for i in range(len(all_benchmarks) + 1):
        for j in range(len(mmd_metrics)):
            if not np.isnan(corr_matrix[i, j]):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax.set_title('Unnormalized PCK Correlations', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Pearson Correlation Coefficient')
    
    # Plot 2: Normalized correlations (if available)
    ax = axes[1]
    if normalization_type:
        norm_label = 'Z-score' if normalization_type == 'zscore' else 'Min-Max (0-1)'
        corr_matrix_norm = np.full((len(all_benchmarks) + 1, len(mmd_metrics)), np.nan)
        
        for j, mmd_col in enumerate(mmd_metrics):
            if mmd_col in correlations['normalized']:
                # Overall correlation
                if 'overall' in correlations['normalized'][mmd_col]:
                    corr_matrix_norm[0, j] = correlations['normalized'][mmd_col]['overall']['corr']
                
                # Per-benchmark correlations
                for i, benchmark in enumerate(all_benchmarks, 1):
                    if benchmark in correlations['normalized'][mmd_col]:
                        corr_matrix_norm[i, j] = correlations['normalized'][mmd_col][benchmark]['corr']
        
        im = ax.imshow(corr_matrix_norm, cmap='RdYlGn', aspect='auto', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(mmd_metrics)))
        ax.set_xticklabels([mmd_labels[m] for m in mmd_metrics], fontsize=10)
        ax.set_yticks(range(len(all_benchmarks) + 1))
        ax.set_yticklabels(['Overall'] + all_benchmarks, fontsize=9)
        
        # Add text annotations
        for i in range(len(all_benchmarks) + 1):
            for j in range(len(mmd_metrics)):
                if not np.isnan(corr_matrix_norm[i, j]):
                    text = ax.text(j, i, f'{corr_matrix_norm[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontsize=8, fontweight='bold')
        
        ax.set_title(f'Normalized PCK ({norm_label}) Correlations', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Pearson Correlation Coefficient')
    else:
        ax.text(0.5, 0.5, 'Normalized correlations\nnot available\n(normalization disabled)', 
               transform=ax.transAxes, ha='center', va='center', fontsize=14)
        ax.set_title('Normalized PCK Correlations (N/A)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{training_dataset}_correlation_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved correlation summary to: {output_path}")
    plt.close()
    
    # Also save correlation statistics to text file
    save_correlation_stats_to_file(correlations, training_dataset, output_dir, normalization_type)


def save_correlation_stats_to_file(correlations, training_dataset, output_dir, normalization_type='zscore'):
    """Save detailed correlation statistics to a text file"""
    
    output_path = os.path.join(output_dir, f'{training_dataset}_correlation_stats.txt')
    
    with open(output_path, 'w') as f:
        f.write(f"Correlation Analysis: {training_dataset}\n")
        f.write("=" * 80 + "\n\n")
        
        mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
        mmd_labels = {
            'mmd2_pred_corr_vs_pred_miss': 'MMD² (pred_corr vs pred_miss)',
            'mmd2_pred_corr_vs_gt': 'MMD² (pred_corr vs gt)',
            'mmd2_pred_miss_vs_gt': 'MMD² (pred_miss vs gt)'
        }
        
        # Unnormalized correlations
        f.write("UNNORMALIZED PCK CORRELATIONS\n")
        f.write("-" * 80 + "\n\n")
        
        for mmd_col in mmd_metrics:
            f.write(f"{mmd_labels[mmd_col]}:\n")
            if mmd_col in correlations['unnormalized']:
                if 'overall' in correlations['unnormalized'][mmd_col]:
                    corr_info = correlations['unnormalized'][mmd_col]['overall']
                    f.write(f"  Overall: r={corr_info['corr']:.4f}, p={corr_info['p_value']:.4f}, n={corr_info['n']}\n")
                
                for benchmark in sorted([k for k in correlations['unnormalized'][mmd_col].keys() if k != 'overall']):
                    corr_info = correlations['unnormalized'][mmd_col][benchmark]
                    f.write(f"  {benchmark}: r={corr_info['corr']:.4f}, p={corr_info['p_value']:.4f}, n={corr_info['n']}\n")
            f.write("\n")
        
        # Normalized correlations
        if normalization_type:
            norm_label = 'Z-SCORE' if normalization_type == 'zscore' else 'MIN-MAX (0-1)'
            f.write(f"\nNORMALIZED PCK ({norm_label}) CORRELATIONS\n")
            f.write("-" * 80 + "\n\n")
            
            for mmd_col in mmd_metrics:
                f.write(f"{mmd_labels[mmd_col]}:\n")
                if mmd_col in correlations['normalized']:
                    if 'overall' in correlations['normalized'][mmd_col]:
                        corr_info = correlations['normalized'][mmd_col]['overall']
                        f.write(f"  Overall: r={corr_info['corr']:.4f}, p={corr_info['p_value']:.4f}, n={corr_info['n']}\n")
                    
                    for benchmark in sorted([k for k in correlations['normalized'][mmd_col].keys() if k != 'overall']):
                        corr_info = correlations['normalized'][mmd_col][benchmark]
                        f.write(f"  {benchmark}: r={corr_info['corr']:.4f}, p={corr_info['p_value']:.4f}, n={corr_info['n']}\n")
                f.write("\n")
    
    print(f"Saved correlation statistics to: {output_path}")


def compute_motion_binned_correlations(df):
    """
    Compute correlations between MMD² metrics and motion-binned PCK.
    Returns a dictionary with correlation statistics.
    """
    motion_bins = ['small', 'medium', 'large']
    mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
    
    correlations = {'motion_binned': {}}
    
    if not HAS_SCIPY:
        return correlations
    
    from scipy.stats import pearsonr
    
    for mmd_metric in mmd_metrics:
        for motion_bin in motion_bins:
            pck_col = f'pck_motion_{motion_bin}'
            
            if pck_col not in df.columns:
                continue
            
            # Overall correlation
            all_data = df.dropna(subset=[mmd_metric, pck_col])
            if len(all_data) > 2:
                corr, p_val = pearsonr(all_data[pck_col], all_data[mmd_metric])
                corr_key = f'{mmd_metric}_vs_{pck_col}'
                correlations['motion_binned'][corr_key] = {
                    'overall': {
                        'corr': corr,
                        'p_value': p_val,
                        'n_samples': len(all_data)
                    }
                }
                
                # Per-benchmark correlations
                benchmarks = df['benchmark'].unique()
                per_benchmark = {}
                for benchmark in benchmarks:
                    bench_data = df[df['benchmark'] == benchmark]
                    bench_valid = bench_data.dropna(subset=[mmd_metric, pck_col])
                    if len(bench_valid) > 2:
                        bench_corr, bench_p_val = pearsonr(bench_valid[pck_col], bench_valid[mmd_metric])
                        per_benchmark[benchmark] = {
                            'corr': bench_corr,
                            'p_value': bench_p_val,
                            'n_samples': len(bench_valid)
                        }
                
                if per_benchmark:
                    correlations['motion_binned'][corr_key]['per_benchmark'] = per_benchmark
    
    return correlations


def create_mmd_vs_motion_pck_plots(df, training_dataset, output_dir, correlations=None):
    """
    Create plots showing MMD² vs motion-binned PCK (small, medium, large).
    This helps understand if MMD² correlates better with motion-aware metrics.
    """
    motion_bins = ['small', 'medium', 'large']
    mmd_metrics = ['mmd2_pred_corr_vs_pred_miss', 'mmd2_pred_corr_vs_gt', 'mmd2_pred_miss_vs_gt']
    mmd_labels = {
        'mmd2_pred_corr_vs_pred_miss': 'MMD² (pred_corr vs pred_miss)',
        'mmd2_pred_corr_vs_gt': 'MMD² (pred_corr vs gt)',
        'mmd2_pred_miss_vs_gt': 'MMD² (pred_miss vs gt)'
    }
    
    benchmarks = df['benchmark'].unique()
    
    # Check if motion-binned columns exist
    has_motion_data = any(f'pck_motion_{bin_name}' in df.columns for bin_name in motion_bins)
    if not has_motion_data:
        print("Warning: Motion-binned PCK columns not found. Skipping MMD² vs motion-binned PCK plots.")
        return
    
    # Create a figure with 3 rows (one per MMD metric) and 3 columns (one per motion bin)
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'MMD² vs Motion-Binned PCK - {training_dataset}', fontsize=16, fontweight='bold')
    
    for mmd_idx, mmd_metric in enumerate(mmd_metrics):
        for bin_idx, motion_bin in enumerate(motion_bins):
            ax = axes[mmd_idx, bin_idx]
            pck_col = f'pck_motion_{motion_bin}'
            
            # Check if column exists
            if pck_col not in df.columns:
                ax.text(0.5, 0.5, f'{pck_col} not available', 
                       transform=ax.transAxes, ha='center', va='center', fontsize=12)
                ax.set_title(f'{mmd_labels[mmd_metric]} vs PCK Motion {motion_bin.capitalize()}')
                continue
            
            # Plot for each benchmark
            for benchmark in benchmarks:
                bench_data = df[df['benchmark'] == benchmark]
                valid_data = bench_data.dropna(subset=[mmd_metric, pck_col])
                if len(valid_data) > 0:
                    ax.plot(valid_data[pck_col], valid_data[mmd_metric], 
                           marker='o', label=benchmark, markersize=4, linewidth=2, alpha=0.7)
            
            # Compute correlation if scipy available
            if correlations and HAS_SCIPY:
                # Try to get correlation from precomputed stats
                corr_key = f'{mmd_metric}_vs_{pck_col}'
                if 'motion_binned' in correlations and corr_key in correlations['motion_binned']:
                    corr_info = correlations['motion_binned'][corr_key]['overall']
                    corr_text = f"r={corr_info['corr']:.3f}, p={corr_info['p_value']:.3f}"
                    ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                           fontsize=8, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                else:
                    # Compute on-the-fly correlation
                    all_data = df.dropna(subset=[mmd_metric, pck_col])
                    if len(all_data) > 2:
                        from scipy.stats import pearsonr
                        corr, p_val = pearsonr(all_data[pck_col], all_data[mmd_metric])
                        corr_text = f"r={corr:.3f}, p={p_val:.3f}"
                        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
                               fontsize=8, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'PCK Motion {motion_bin.capitalize()} (%)', fontsize=10)
            ax.set_ylabel(mmd_labels[mmd_metric], fontsize=10)
            ax.set_title(f'{mmd_labels[mmd_metric]} vs PCK Motion {motion_bin.capitalize()}', fontsize=11)
            ax.legend(loc='best', fontsize=7)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{training_dataset}_mmd_vs_motion_pck.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved MMD² vs Motion-Binned PCK plots to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot validation metrics from snapshot')
    parser.add_argument('snapshot_path', type=str, help='Path to snapshot directory')
    parser.add_argument('--output-dir', type=str, default=None, 
                       help='Output directory for plots (default: same as snapshot parent)')
    parser.add_argument('--normalization', type=str, default='zscore', 
                       choices=['zscore', 'minmax', 'none'],
                       help='Type of normalization to use: zscore (default), minmax (0-1), or none')
    parser.add_argument('--no-zscore', action='store_true', default=False,
                       help='[Deprecated] Disable z-score normalization (use --normalization none instead)')
    
    args = parser.parse_args()
    
    snapshot_path = os.path.abspath(args.snapshot_path)
    
    if not os.path.isdir(snapshot_path):
        raise ValueError(f"Snapshot path does not exist: {snapshot_path}")
    
    print(f"Parsing snapshot: {snapshot_path}")
    
    # Extract training dataset name
    training_dataset = parse_training_dataset(snapshot_path)
    print(f"Training dataset: {training_dataset}")
    
    # Load validation results
    df = load_validation_results(snapshot_path)
    print(f"Loaded {len(df)} validation records")
    print(f"Benchmarks: {', '.join(df['benchmark'].unique())}")
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(snapshot_path) if os.path.dirname(snapshot_path) else '.'
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine normalization type (handle deprecated --no-zscore flag)
    if args.no_zscore:
        normalization_type = None
        print("Warning: --no-zscore is deprecated. Use --normalization none instead.")
    else:
        normalization_type = args.normalization if args.normalization != 'none' else None
    
    if normalization_type == 'zscore':
        print("Using z-score normalization for PCK (baseline=epoch 0)")
    elif normalization_type == 'minmax':
        print("Using min-max (0-1) normalization for PCK (baseline=epoch 0)")
    else:
        print("Using raw PCK values (normalization disabled)")
    
    # Pre-compute normalized dataframe and statistics
    if normalization_type == 'zscore':
        df_normalized, benchmark_stats = compute_zscore_normalized_pck(df, use_baseline=True)
    elif normalization_type == 'minmax':
        df_normalized, benchmark_stats = compute_minmax_normalized_pck(df, use_baseline=True)
    else:
        df_normalized = df.copy()
        benchmark_stats = None
    
    # Compute correlations (needed for plot annotations)
    print("\nComputing correlation statistics...")
    correlations = compute_correlation_stats(df, df_normalized, normalization_type)
    
    # Compute motion-binned correlations
    print("Computing motion-binned correlations...")
    motion_correlations = compute_motion_binned_correlations(df)
    
    # Merge with existing correlations
    if correlations is None:
        correlations = {}
    correlations.update(motion_correlations)
    
    # Load pairwise MMD² comparisons
    pairwise_mmd_dict = load_pairwise_mmd_comparisons(training_dataset)
    if pairwise_mmd_dict:
        print(f"Loaded pairwise MMD² comparisons for {len(pairwise_mmd_dict)} benchmarks")
    
    # Create plots with correlation info
    create_plots(df, training_dataset, output_dir, normalization_type=normalization_type, 
                correlations=correlations, df_normalized=df_normalized, benchmark_stats=benchmark_stats,
                pairwise_mmd_dict=pairwise_mmd_dict)
    
    # Create MMD² vs motion-binned PCK plots
    print("Creating MMD² vs motion-binned PCK plots...")
    create_mmd_vs_motion_pck_plots(df, training_dataset, output_dir, correlations=correlations)
    
    print("Done!")


if __name__ == '__main__':
    main()
