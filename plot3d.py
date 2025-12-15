#!/usr/bin/env python3
"""
3D visualization script for coverage metrics vs PCK performance.

Creates 3D scatter plots showing:
- x-axis: ResNet recall coverage
- y-axis: Flow recall coverage  
- z-axis: PCK performance (raw and z-scored)

Also creates 2D color-mapped versions for easier interpretation.

Usage:
    python plot3d.py --snapshots_dir snapshots/ --coverage-csv coverage_results.csv --coverage-resnet-csv coverage_resnet_results.csv --output-dir plots3d/
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy import stats

# Try to import statsmodels for mixed-effects regression
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Mixed-effects regression will be skipped.")
    print("Install with: pip install statsmodels")

# Import functions from existing plotting scripts
from plot_metrics import (
    parse_training_summary,
    parse_validation_results,
    format_training_dataset_label,
    parse_directory_name,
    parse_snapshot_directory,
)
from plot_benchmark_metrics import (
    parse_training_dataset_from_summary,
    parse_best_performance_from_summary,
    load_mmd_lookup,
)


def load_coverage_lookup(csv_path='coverage_results.csv'):
    """
    Load coverage metrics from CSV file.
    
    Args:
        csv_path: Path to coverage CSV file
        
    Returns:
        Dictionary mapping (train_dataset_split, eval_dataset_split) -> coverage_metrics dict
    """
    coverage_lookup = {}
    
    if not os.path.exists(csv_path):
        print(f"Warning: Coverage CSV not found: {csv_path}")
        return coverage_lookup
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse dataset names and splits (normalize to lowercase for consistent matching)
                train_dataset = str(row.get('dataset1', '')).strip().lower()
                train_split = str(row.get('split1', '')).strip().lower()
                eval_dataset = str(row.get('dataset2', '')).strip().lower()
                eval_split = str(row.get('split2', '')).strip().lower()
                
                # Skip rows with empty dataset names
                if not train_dataset or not eval_dataset:
                    continue
                
                train_id = f"{train_dataset}_{train_split}" if train_split else train_dataset
                eval_id = f"{eval_dataset}_{eval_split}" if eval_split else eval_dataset
                
                # Get recall value
                recall_val = row.get('recall', np.nan)
                try:
                    recall_val = float(recall_val) if recall_val else np.nan
                except (ValueError, TypeError):
                    recall_val = np.nan
                
                if not pd.isna(recall_val):
                    coverage_lookup[(train_id, eval_id)] = {
                        'recall': recall_val,
                        'precision': float(row.get('precision', np.nan)) if row.get('precision') else np.nan,
                        'outside': float(row.get('outside', np.nan)) if row.get('outside') else np.nan,
                    }
                    
                    # Also store without explicit split for backward compatibility
                    coverage_lookup[(train_dataset, eval_dataset)] = {
                        'recall': recall_val,
                        'precision': float(row.get('precision', np.nan)) if row.get('precision') else np.nan,
                        'outside': float(row.get('outside', np.nan)) if row.get('outside') else np.nan,
                    }
    except Exception as e:
        print(f"Error loading coverage CSV {csv_path}: {e}")
        import traceback
        traceback.print_exc()
    
    # Debug: print sample of loaded keys
    if coverage_lookup:
        print(f"  Loaded {len(coverage_lookup)} coverage entries from {csv_path}")
        sample_keys = list(coverage_lookup.keys())[:3]
        print(f"  Sample keys (first 3): {sample_keys}")
    
    return coverage_lookup




def collect_3d_data_points(snapshots_data, flow_coverage_lookup, resnet_coverage_lookup, debug=False):
    """
    Collect data points for 3D plotting.
    
    Returns:
        List of dicts with keys: 'resnet_recall', 'flow_recall', 'pck', 'training_dataset', 'benchmark', 'snapshot_path'
    """
    data_points = []
    missing_flow = defaultdict(int)
    missing_resnet = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        if not summary_path.exists():
            continue
        
        # Get base training dataset name
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Warning: Could not parse training dataset from {summary_path}")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        # For each benchmark, look up both flow and resnet coverage
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up coverage with splits
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Get flow recall
            flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
            flow_key_used = (training_dataset_train, benchmark_test)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
                flow_key_used = (training_dataset_train, benchmark_val)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((base_training_dataset, benchmark_lower))
                flow_key_used = (base_training_dataset, benchmark_lower)
            
            # Get resnet recall
            resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_test))
            resnet_key_used = (training_dataset_train, benchmark_test)
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_val))
                resnet_key_used = (training_dataset_train, benchmark_val)
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((base_training_dataset, benchmark_lower))
                resnet_key_used = (base_training_dataset, benchmark_lower)
            
            # Track missing metrics for summary (no per-item warnings)
            if debug:
                if not flow_metrics:
                    missing_flow[flow_key_used] += 1
                if not resnet_metrics:
                    missing_resnet[resnet_key_used] += 1
            
            # Only add if we have both coverage values
            if (flow_metrics and 'recall' in flow_metrics and not pd.isna(flow_metrics['recall']) and
                resnet_metrics and 'recall' in resnet_metrics and not pd.isna(resnet_metrics['recall'])):
                data_points.append({
                    'resnet_recall': resnet_metrics['recall'],
                    'flow_recall': flow_metrics['recall'],
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
    
    if debug and (missing_flow or missing_resnet):
        print(f"\nDebug: Missing flow recall keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print(f"\nDebug: Missing resnet recall keys (top 10):")
        for key, count in sorted(missing_resnet.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        
        # Show sample of available keys
        print(f"\nDebug: Sample of available flow recall keys (first 5):")
        for key in list(flow_coverage_lookup.keys())[:5]:
            print(f"  {key}")
        print(f"\nDebug: Sample of available resnet recall keys (first 5):")
        for key in list(resnet_coverage_lookup.keys())[:5]:
            print(f"  {key}")
    
    return data_points


def create_3d_scatter_plot(data_points, output_path, dataset_color_map, zscore=False):
    """Create 3D scatter plot"""
    if not data_points:
        print("Warning: No data points for 3D plot")
        return
    
    # Extract data
    resnet_recall = [p['resnet_recall'] for p in data_points]
    flow_recall = [p['flow_recall'] for p in data_points]
    pck_values = [p['pck'] for p in data_points]
    
    # Z-score PCK if requested
    if zscore:
        pck_values = stats.zscore(pck_values)
        z_label = 'PCK (Z-scored)'
    else:
        z_label = 'PCK (%)'
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Group by training dataset
    datasets_points = defaultdict(list)
    for point in data_points:
        datasets_points[point['training_dataset']].append(point)
    
    # Plot each dataset
    for training_dataset, points in datasets_points.items():
        resnet_vals = [p['resnet_recall'] for p in points]
        flow_vals = [p['flow_recall'] for p in points]
        pck_vals = [p['pck'] for p in points]
        
        if zscore:
            # Recompute z-score for this dataset's points
            pck_vals = stats.zscore(pck_vals)
        
        color = dataset_color_map.get(training_dataset, 'black')
        ax.scatter(resnet_vals, flow_vals, pck_vals,
                  color=color, label=training_dataset,
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('ResNet Recall Coverage', fontsize=12, labelpad=10)
    ax.set_ylabel('Flow Recall Coverage', fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    ax.set_title('3D: ResNet Recall vs Flow Recall vs PCK', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    suffix = '_zscore' if zscore else '_raw'
    output_file = output_path / f'3d_resnet_flow_pck{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 3D plot: {output_file}")
    plt.close()


def zscore_pck_by_benchmark(data_points):
    """
    Z-score PCK within each benchmark to remove baseline difficulty differences.
    
    Returns:
        List of z-scored PCK values (same order as data_points)
    """
    # Convert to DataFrame for easier groupby operations
    df = pd.DataFrame(data_points)
    
    # Z-score PCK within each benchmark
    def standard_zscore(x):
        if x.std() > 0:
            return (x - x.mean()) / x.std()
        return x * 0  # Return zeros if no variance
    
    df['pck_z'] = df.groupby('benchmark')['pck'].transform(standard_zscore)
    
    return df['pck_z'].values


def create_faceted_by_benchmark_plot(data_points, output_path, zscore_by_benchmark=False):
    """
    Create faceted 2D colormap plots with one panel per benchmark.
    Each panel shows resnet_recall vs flow_recall colored by PCK.
    """
    if not data_points:
        print("Warning: No data points for faceted by benchmark plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Prepare PCK values
    if zscore_by_benchmark:
        df['pck_plot'] = zscore_pck_by_benchmark(data_points)
        color_label = 'PCK (Z-scored by benchmark)'
        suffix = '_zscore_by_benchmark'
    else:
        df['pck_plot'] = df['pck']
        color_label = 'PCK (%)'
        suffix = '_raw'
    
    benchmarks = sorted(df['benchmark'].unique())
    n_benchmarks = len(benchmarks)
    
    if n_benchmarks == 0:
        return
    
    # Calculate grid size
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Get global color scale for consistency
    vmin = df['pck_plot'].min()
    vmax = df['pck_plot'].max()
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        subset = df[df['benchmark'] == benchmark]
        
        if len(subset) == 0:
            continue
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(subset['resnet_recall'], subset['flow_recall'],
                           c=subset['pck_plot'], s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Add training dataset labels
        for _, row in subset.iterrows():
            ax.annotate(row['training_dataset'],
                       (row['resnet_recall'], row['flow_recall']),
                       fontsize=6, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
        
        ax.set_xlabel('ResNet Recall', fontsize=10)
        ax.set_ylabel('Flow Recall', fontsize=10)
        ax.set_title(benchmark.upper(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar (shared across all subplots)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(color_label, fontsize=11)
    
    # Add overall title
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else ''
    fig.suptitle(f'ResNet Recall vs Flow Recall{title_suffix} - By Benchmark',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    
    # Save
    output_file = output_path / f'2d_resnet_flow_pck_by_benchmark{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved faceted by benchmark plot: {output_file}")
    plt.close()


def create_faceted_by_training_set_plot(data_points, output_path, dataset_color_map, zscore_by_benchmark=False):
    """
    Create faceted 2D colormap plots with one panel per training dataset.
    Each panel shows resnet_recall vs flow_recall colored by PCK.
    """
    if not data_points:
        print("Warning: No data points for faceted by training set plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Prepare PCK values
    if zscore_by_benchmark:
        df['pck_plot'] = zscore_pck_by_benchmark(data_points)
        color_label = 'PCK (Z-scored by benchmark)'
        suffix = '_zscore_by_benchmark'
    else:
        df['pck_plot'] = df['pck']
        color_label = 'PCK (%)'
        suffix = '_raw'
    
    training_datasets = sorted(df['training_dataset'].unique())
    n_datasets = len(training_datasets)
    
    if n_datasets == 0:
        return
    
    # Calculate grid size
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Get global color scale for consistency
    vmin = df['pck_plot'].min()
    vmax = df['pck_plot'].max()
    
    for idx, training_dataset in enumerate(training_datasets):
        ax = axes[idx]
        subset = df[df['training_dataset'] == training_dataset]
        
        if len(subset) == 0:
            continue
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(subset['resnet_recall'], subset['flow_recall'],
                           c=subset['pck_plot'], s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Add benchmark labels
        for _, row in subset.iterrows():
            ax.annotate(row['benchmark'],
                       (row['resnet_recall'], row['flow_recall']),
                       fontsize=6, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
        
        ax.set_xlabel('ResNet Recall', fontsize=10)
        ax.set_ylabel('Flow Recall', fontsize=10)
        ax.set_title(training_dataset, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(n_datasets, len(axes)):
        axes[idx].set_visible(False)
    
    # Add colorbar (shared across all subplots)
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label(color_label, fontsize=11)
    
    # Add overall title
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else ''
    fig.suptitle(f'ResNet Recall vs Flow Recall{title_suffix} - By Training Dataset',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    
    # Save
    output_file = output_path / f'2d_resnet_flow_pck_by_training_set{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved faceted by training set plot: {output_file}")
    plt.close()


def create_2d_colormap_plot(data_points, output_path, dataset_color_map, zscore=False, zscore_by_benchmark=False):
    """Create 2D scatter plot with PCK as color"""
    if not data_points:
        print("Warning: No data points for 2D colormap plot")
        return
    
    # Extract data
    resnet_recall = [p['resnet_recall'] for p in data_points]
    flow_recall = [p['flow_recall'] for p in data_points]
    pck_values = np.array([p['pck'] for p in data_points])
    
    # Z-score PCK if requested
    if zscore_by_benchmark:
        pck_values = zscore_pck_by_benchmark(data_points)
        color_label = 'PCK (Z-scored by benchmark)'
        suffix = '_zscore_by_benchmark'
    elif zscore:
        pck_values = stats.zscore(pck_values)
        color_label = 'PCK (Z-scored)'
        suffix = '_zscore'
    else:
        color_label = 'PCK (%)'
        suffix = '_raw'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(resnet_recall, flow_recall, c=pck_values,
                        s=150, alpha=0.7, edgecolors='black', linewidth=1,
                        cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12)
    
    # Add benchmark labels
    for point in data_points:
        ax.annotate(point['benchmark'],
                   (point['resnet_recall'], point['flow_recall']),
                   fontsize=7, alpha=0.6,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('ResNet Recall Coverage', fontsize=12)
    ax.set_ylabel('Flow Recall Coverage', fontsize=12)
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else (' (PCK z-scored)' if zscore else '')
    ax.set_title(f'ResNet Recall vs Flow Recall{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = output_path / f'2d_resnet_flow_pck_colormap{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D colormap plot: {output_file}")
    plt.close()


def collect_precision_data_points(snapshots_data, flow_coverage_lookup, resnet_coverage_lookup, debug=False):
    """
    Collect data points for precision plotting.
    
    Returns:
        List of dicts with keys: 'resnet_precision', 'flow_precision', 'pck', 'training_dataset', 'benchmark', 'snapshot_path'
    """
    data_points = []
    missing_flow = defaultdict(int)
    missing_resnet = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        if not summary_path.exists():
            continue
        
        # Get base training dataset name
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Warning: Could not parse training dataset from {summary_path}")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        # For each benchmark, look up both flow and resnet precision
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up coverage with splits
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Get flow precision
            flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
            flow_key_used = (training_dataset_train, benchmark_test)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
                flow_key_used = (training_dataset_train, benchmark_val)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((base_training_dataset, benchmark_lower))
                flow_key_used = (base_training_dataset, benchmark_lower)
            
            # Get resnet precision
            resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_test))
            resnet_key_used = (training_dataset_train, benchmark_test)
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_val))
                resnet_key_used = (training_dataset_train, benchmark_val)
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((base_training_dataset, benchmark_lower))
                resnet_key_used = (base_training_dataset, benchmark_lower)
            
            # Debug output
            if debug:
                if not flow_metrics or pd.isna(flow_metrics.get('precision', np.nan)):
                    missing_flow[flow_key_used] += 1
                if not resnet_metrics or pd.isna(resnet_metrics.get('precision', np.nan)):
                    missing_resnet[resnet_key_used] += 1
            
            # Only add if we have both precision values
            flow_precision = flow_metrics.get('precision', np.nan) if flow_metrics else np.nan
            resnet_precision = resnet_metrics.get('precision', np.nan) if resnet_metrics else np.nan
            
            if (not pd.isna(flow_precision) and not pd.isna(resnet_precision)):
                data_points.append({
                    'resnet_precision': resnet_precision,
                    'flow_precision': flow_precision,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
    
    if debug and (missing_flow or missing_resnet):
        print(f"\nDebug: Missing flow precision keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print(f"\nDebug: Missing resnet precision keys (top 10):")
        for key, count in sorted(missing_resnet.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
    
    return data_points


def create_2d_precision_colormap_plot(data_points, output_path, dataset_color_map, zscore=False, zscore_by_benchmark=False):
    """Create 2D scatter plot with ResNet Precision vs Flow Precision, colored by PCK"""
    if not data_points:
        print("Warning: No data points for 2D precision colormap plot")
        return
    
    # Extract data
    resnet_precision = [p['resnet_precision'] for p in data_points]
    flow_precision = [p['flow_precision'] for p in data_points]
    pck_values = np.array([p['pck'] for p in data_points])
    
    # Z-score PCK if requested
    if zscore_by_benchmark:
        pck_values = zscore_pck_by_benchmark(data_points)
        color_label = 'PCK (Z-scored by benchmark)'
        suffix = '_zscore_by_benchmark'
    elif zscore:
        pck_values = stats.zscore(pck_values)
        color_label = 'PCK (Z-scored)'
        suffix = '_zscore'
    else:
        color_label = 'PCK (%)'
        suffix = '_raw'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(resnet_precision, flow_precision, c=pck_values,
                        s=150, alpha=0.7, edgecolors='black', linewidth=1,
                        cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12)
    
    # Add benchmark labels
    for point in data_points:
        ax.annotate(point['benchmark'],
                   (point['resnet_precision'], point['flow_precision']),
                   fontsize=7, alpha=0.6,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('ResNet Precision Coverage', fontsize=12)
    ax.set_ylabel('Flow Precision Coverage', fontsize=12)
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else (' (PCK z-scored)' if zscore else '')
    ax.set_title(f'ResNet Precision vs Flow Precision{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = output_path / f'2d_resnet_flow_precision_pck_colormap{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D precision colormap plot: {output_file}")
    plt.close()


def collect_mmd_data_points(snapshots_data, flow_mmd_lookup, feature_mmd_lookup, debug=False):
    """
    Collect data points for MMD plotting.
    
    Returns:
        List of dicts with keys: 'flow_mmd', 'feature_mmd', 'pck', 'training_dataset', 'benchmark', 'snapshot_path'
    """
    data_points = []
    missing_flow = defaultdict(int)
    missing_feature = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        if not summary_path.exists():
            continue
        
        # Get base training dataset name
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Warning: Could not parse training dataset from {summary_path}")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        # For each benchmark, look up both flow and feature MMD
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up MMD with splits
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Get flow MMD
            flow_mmd = flow_mmd_lookup.get((training_dataset_train, benchmark_test))
            flow_key_used = (training_dataset_train, benchmark_test)
            if flow_mmd is None:
                flow_mmd = flow_mmd_lookup.get((training_dataset_train, benchmark_val))
                flow_key_used = (training_dataset_train, benchmark_val)
            if flow_mmd is None:
                flow_mmd = flow_mmd_lookup.get((base_training_dataset, benchmark_lower))
                flow_key_used = (base_training_dataset, benchmark_lower)
            
            # Get feature MMD
            feature_mmd = feature_mmd_lookup.get((training_dataset_train, benchmark_test))
            feature_key_used = (training_dataset_train, benchmark_test)
            if feature_mmd is None:
                feature_mmd = feature_mmd_lookup.get((training_dataset_train, benchmark_val))
                feature_key_used = (training_dataset_train, benchmark_val)
            if feature_mmd is None:
                feature_mmd = feature_mmd_lookup.get((base_training_dataset, benchmark_lower))
                feature_key_used = (base_training_dataset, benchmark_lower)
            
            # Debug output
            if debug:
                if flow_mmd is None:
                    missing_flow[flow_key_used] += 1
                    print(f"  Missing flow MMD for: {flow_key_used} (train={base_training_dataset}, bench={benchmark_lower})")
                if feature_mmd is None:
                    missing_feature[feature_key_used] += 1
                    print(f"  Missing feature MMD for: {feature_key_used} (train={base_training_dataset}, bench={benchmark_lower})")
            
            # Only add if we have both MMD values
            if flow_mmd is not None and feature_mmd is not None:
                data_points.append({
                    'flow_mmd': flow_mmd,
                    'feature_mmd': feature_mmd,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
    
    if debug and (missing_flow or missing_feature):
        print(f"\nDebug: Missing flow MMD keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print(f"\nDebug: Missing feature MMD keys (top 10):")
        for key, count in sorted(missing_feature.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        
        # Show sample of available keys
        print(f"\nDebug: Sample of available flow MMD keys (first 5):")
        for key in list(flow_mmd_lookup.keys())[:5]:
            print(f"  {key}")
        print(f"\nDebug: Sample of available feature MMD keys (first 5):")
        for key in list(feature_mmd_lookup.keys())[:5]:
            print(f"  {key}")
    
    return data_points


def create_2d_mmd_colormap_plot(data_points, output_path, dataset_color_map, zscore=False, zscore_by_benchmark=False):
    """Create 2D scatter plot with Flow MMD vs Feature MMD, colored by PCK"""
    if not data_points:
        print("Warning: No data points for 2D MMD colormap plot")
        return
    
    # Extract data
    flow_mmd = [p['flow_mmd'] for p in data_points]
    feature_mmd = [p['feature_mmd'] for p in data_points]
    pck_values = np.array([p['pck'] for p in data_points])
    
    # Z-score PCK if requested
    if zscore_by_benchmark:
        pck_values = zscore_pck_by_benchmark(data_points)
        color_label = 'PCK (Z-scored by benchmark)'
        suffix = '_zscore_by_benchmark'
    elif zscore:
        pck_values = stats.zscore(pck_values)
        color_label = 'PCK (Z-scored)'
        suffix = '_zscore'
    else:
        color_label = 'PCK (%)'
        suffix = '_raw'
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create scatter plot with color mapping
    scatter = ax.scatter(flow_mmd, feature_mmd, c=pck_values,
                        s=150, alpha=0.7, edgecolors='black', linewidth=1,
                        cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12)
    
    # Add benchmark labels
    for point in data_points:
        ax.annotate(point['benchmark'],
                   (point['flow_mmd'], point['feature_mmd']),
                   fontsize=7, alpha=0.6,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Flow MMD²', fontsize=12)
    ax.set_ylabel('Feature MMD²', fontsize=12)
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else (' (PCK z-scored)' if zscore else '')
    ax.set_title(f'Flow MMD² vs Feature MMD²{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = output_path / f'2d_flow_feature_mmd_pck_colormap{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D MMD colormap plot: {output_file}")
    plt.close()


def collect_all_predictors_data_points(snapshots_data, flow_mmd_lookup, feature_mmd_lookup, 
                                       flow_coverage_lookup, resnet_coverage_lookup, debug=False):
    """
    Collect data points with all predictors: flow MMD, feature MMD, flow recall, resnet recall,
    flow precision, resnet precision.
    
    Returns:
        List of dicts with keys: 'flow_mmd', 'feature_mmd', 'flow_recall', 'resnet_recall',
        'flow_precision', 'resnet_precision', 'pck', 'training_dataset', 'benchmark', 'snapshot_path'
    """
    data_points = []
    missing = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        if not summary_path.exists():
            continue
        
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            continue
        
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Get flow MMD
            flow_mmd = flow_mmd_lookup.get((training_dataset_train, benchmark_test))
            if flow_mmd is None:
                flow_mmd = flow_mmd_lookup.get((training_dataset_train, benchmark_val))
            if flow_mmd is None:
                flow_mmd = flow_mmd_lookup.get((base_training_dataset, benchmark_lower))
            
            # Get feature MMD
            feature_mmd = feature_mmd_lookup.get((training_dataset_train, benchmark_test))
            if feature_mmd is None:
                feature_mmd = feature_mmd_lookup.get((training_dataset_train, benchmark_val))
            if feature_mmd is None:
                feature_mmd = feature_mmd_lookup.get((base_training_dataset, benchmark_lower))
            
            # Get flow recall
            flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((base_training_dataset, benchmark_lower))
            flow_recall = flow_metrics.get('recall', np.nan) if flow_metrics else np.nan
            
            # Get resnet recall
            resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_test))
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_val))
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((base_training_dataset, benchmark_lower))
            resnet_recall = resnet_metrics.get('recall', np.nan) if resnet_metrics else np.nan
            resnet_precision = resnet_metrics.get('precision', np.nan) if resnet_metrics else np.nan
            
            # Get flow precision
            flow_precision = flow_metrics.get('precision', np.nan) if flow_metrics else np.nan
            
            # Only add if we have all required metrics (MMD and recall are required, precision is optional)
            if (flow_mmd is not None and feature_mmd is not None and 
                not pd.isna(flow_recall) and not pd.isna(resnet_recall)):
                data_points.append({
                    'flow_mmd': flow_mmd,
                    'feature_mmd': feature_mmd,
                    'flow_recall': flow_recall,
                    'resnet_recall': resnet_recall,
                    'flow_precision': flow_precision,
                    'resnet_precision': resnet_precision,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
            elif debug:
                missing_key = []
                if flow_mmd is None:
                    missing_key.append('flow_mmd')
                if feature_mmd is None:
                    missing_key.append('feature_mmd')
                if pd.isna(flow_recall):
                    missing_key.append('flow_recall')
                if pd.isna(resnet_recall):
                    missing_key.append('resnet_recall')
                missing[tuple(missing_key)] += 1
    
    if debug and missing:
        print(f"\nDebug: Missing metrics (top 10):")
        for key, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
    
    return data_points


def compare_predictors_with_mixed_effects(df, output_path=None, create_plots=True):
    """
    Compare all predictors using multiple mixed-effects regression models.
    
    This function:
    1. Runs models with each predictor individually
    2. Runs a full model with all predictors
    3. Compares models using AIC/BIC
    4. Shows standardized coefficients for fair comparison
    
    Args:
        df: DataFrame with columns ['flow_mmd', 'feature_mmd', 'flow_recall', 'resnet_recall',
            'flow_precision', 'resnet_precision', 'pck', 'benchmark']
        output_path: Optional path to save results to file
    """
    if not HAS_STATSMODELS:
        print("Error: statsmodels not installed. Cannot run mixed-effects regression.")
        print("Install with: pip install statsmodels")
        return None
    
    df = df.copy()
    # Required predictors
    required_predictors = ['flow_mmd', 'feature_mmd', 'flow_recall', 'resnet_recall', 'pck', 'benchmark']
    df = df.dropna(subset=required_predictors)
    
    if len(df) < 10:
        print(f"Error: Insufficient data ({len(df)} points). Need at least 10 points for reliable analysis.")
        return None
    
    print(f"\n{'='*80}")
    print(f"PREDICTOR COMPARISON: Which metric is most predictive of PCK?")
    print(f"{'='*80}")
    print(f"\nData: {len(df)} observations across {df['benchmark'].nunique()} benchmarks")
    print(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}")
    
    # Standardize predictors for fair comparison (mean=0, std=1)
    # This allows comparing coefficients on the same scale
    try:
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("Error: sklearn not installed. Install with: pip install scikit-learn")
        return None
    
    scaler = StandardScaler()
    # Base predictors (always included)
    predictors = ['flow_mmd', 'feature_mmd', 'flow_recall', 'resnet_recall']
    # Add precision if available
    if 'flow_precision' in df.columns and 'resnet_precision' in df.columns:
        # Check if we have enough non-NaN precision values
        precision_available = df[['flow_precision', 'resnet_precision']].notna().all(axis=1).sum()
        if precision_available >= 10:
            predictors.extend(['flow_precision', 'resnet_precision'])
            print(f"  Including precision metrics ({precision_available} observations with precision data)")
        else:
            print(f"  Skipping precision metrics (only {precision_available} observations with precision data)")
    
    df_scaled = df.copy()
    # Only standardize predictors that exist and have data
    predictors_to_scale = [p for p in predictors if p in df.columns]
    df_scaled[[f'{p}_std' for p in predictors_to_scale]] = scaler.fit_transform(df[predictors_to_scale])
    
    results = {}
    
    # 1. Individual predictor models
    print(f"\n{'='*80}")
    print("1. INDIVIDUAL PREDICTOR MODELS")
    print(f"{'='*80}")
    print(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'AIC':>10} {'BIC':>10}")
    print(f"{'-'*70}")
    
    for predictor in predictors:
        try:
            model = smf.mixedlm(f"pck ~ {predictor}_std", data=df_scaled, groups=df_scaled["benchmark"])
            result = model.fit(method='lbfgs', reml=False)  # Use ML instead of REML for AIC/BIC
            
            # Check if model converged
            if not result.converged:
                print(f"{predictor:<20} {'NO CONV':>12} {'NO CONV':>12} {'NO CONV':>10} {'NO CONV':>10}")
                results[predictor] = None
                continue
            
            coef = result.fe_params.get(f'{predictor}_std', np.nan)
            pval = result.pvalues.get(f'{predictor}_std', np.nan)
            
            # Get AIC/BIC - handle potential NaN values
            aic = result.aic if hasattr(result, 'aic') and not np.isnan(result.aic) else np.nan
            bic = result.bic if hasattr(result, 'bic') and not np.isnan(result.bic) else np.nan
            
            # If AIC/BIC are NaN, try to compute manually
            if np.isnan(aic) or np.isnan(bic):
                llf = result.llf if hasattr(result, 'llf') and not np.isnan(result.llf) else np.nan
                n_params = len(result.fe_params) + 1  # fixed effects + 1 for random effect variance
                n_obs = len(df_scaled)
                if not np.isnan(llf):
                    aic = -2 * llf + 2 * n_params
                    bic = -2 * llf + np.log(n_obs) * n_params
            
            results[predictor] = {
                'std_coef': coef,
                'pvalue': pval,
                'aic': aic,
                'bic': bic,
                'converged': result.converged,
                'significant': pval < 0.05 if not np.isnan(pval) else False,
                'model': result
            }
            
            sig_marker = '*' if results[predictor]['significant'] else ''
            aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
            print(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {aic_str:>10} {bic_str:>10}")
        except Exception as e:
            print(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10} {'ERROR':>10}")
            print(f"  Error: {e}")
            results[predictor] = None
    
    # 2. Full model with all predictors
    print(f"\n{'='*80}")
    print("2. FULL MODEL (All predictors together)")
    print(f"{'='*80}")
    
    try:
        # Build formula with available predictors
        formula_parts = [f"{p}_std" for p in predictors_to_scale]
        formula = "pck ~ " + " + ".join(formula_parts)
        model_full = smf.mixedlm(formula, data=df_scaled, groups=df_scaled["benchmark"])
        result_full = model_full.fit(method='lbfgs', reml=False)  # Use ML instead of REML for AIC/BIC
        
        # Check if model converged
        if not result_full.converged:
            print(f"Warning: Full model did not converge!")
            results['full_model'] = None
        else:
            predictor_names = {
                'flow_mmd': 'Flow MMD',
                'feature_mmd': 'Feature MMD',
                'flow_recall': 'Flow Recall',
                'resnet_recall': 'ResNet Recall',
                'flow_precision': 'Flow Precision',
                'resnet_precision': 'ResNet Precision'
            }
            formula_display = " + ".join([predictor_names.get(p, p) for p in predictors_to_scale])
            print(f"Model: PCK ~ {formula_display} + (1|benchmark)")
            print(f"\n{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'Significant':>12}")
            print(f"{'-'*60}")
            
            full_results = {}
            for predictor in predictors_to_scale:
                coef = result_full.fe_params.get(f'{predictor}_std', np.nan)
                pval = result_full.pvalues.get(f'{predictor}_std', np.nan)
                sig = pval < 0.05 if not np.isnan(pval) else False
                sig_marker = '*' if sig else ''
                
                full_results[predictor] = {
                    'std_coef': coef,
                    'pvalue': pval,
                    'significant': sig
                }
                
                print(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {'Yes' if sig else 'No':>12}")
            
            # Get AIC/BIC - handle potential NaN values
            aic = result_full.aic if hasattr(result_full, 'aic') and not np.isnan(result_full.aic) else np.nan
            bic = result_full.bic if hasattr(result_full, 'bic') and not np.isnan(result_full.bic) else np.nan
            
            # If AIC/BIC are NaN, try to compute manually
            if np.isnan(aic) or np.isnan(bic):
                llf = result_full.llf if hasattr(result_full, 'llf') and not np.isnan(result_full.llf) else np.nan
                n_params = len(result_full.fe_params) + 1  # fixed effects + 1 for random effect variance
                n_obs = len(df_scaled)
                if not np.isnan(llf):
                    aic = -2 * llf + 2 * n_params
                    bic = -2 * llf + np.log(n_obs) * n_params
            
            print(f"\nModel fit:")
            print(f"  Converged: {result_full.converged}")
            aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
            llf_str = f"{result_full.llf:.2f}" if hasattr(result_full, 'llf') and not np.isnan(result_full.llf) else "N/A"
            print(f"  AIC: {aic_str}")
            print(f"  BIC: {bic_str}")
            print(f"  Log-likelihood: {llf_str}")
            if hasattr(result_full, 'cov_re') and hasattr(result_full.cov_re, 'iloc'):
                print(f"  Random effect variance (benchmark): {result_full.cov_re.iloc[0, 0]:.4f}")
            
            results['full_model'] = {
                'result': result_full,
                'predictors': full_results,
                'aic': aic,
                'bic': bic,
                'converged': result_full.converged,
                'df_scaled': df_scaled,
                'predictors_to_scale': predictors_to_scale
            }
            
            # Create visualizations if requested
            if create_plots and output_path:
                print("\nCreating model fit diagnostic plots...")
                visualize_model_fit(
                    result_full,
                    df_scaled,
                    output_path,
                    predictors_to_scale
                )
    except Exception as e:
        print(f"Error fitting full model: {e}")
        import traceback
        traceback.print_exc()
        results['full_model'] = None
    
    # 3. Model comparison
    print(f"\n{'='*80}")
    print("3. MODEL COMPARISON (Lower AIC/BIC is better)")
    print(f"{'='*80}")
    
    # Sort individual models by AIC (filter out NaN AIC values)
    valid_models = [(k, v) for k, v in results.items() 
                    if v is not None and k != 'full_model' 
                    and not np.isnan(v.get('aic', np.nan))]
    valid_models.sort(key=lambda x: x[1]['aic'])
    
    print(f"{'Model':<25} {'AIC':>10} {'BIC':>10} {'ΔAIC vs Best':>15}")
    print(f"{'-'*60}")
    
    if valid_models:
        best_aic = valid_models[0][1]['aic']
        for name, model_data in valid_models:
            delta_aic = model_data['aic'] - best_aic
            aic_str = f"{model_data['aic']:.1f}" if not np.isnan(model_data['aic']) else "N/A"
            bic_str = f"{model_data['bic']:.1f}" if not np.isnan(model_data['bic']) else "N/A"
            delta_str = f"{delta_aic:.1f}" if not np.isnan(delta_aic) else "N/A"
            print(f"{name:<25} {aic_str:>10} {bic_str:>10} {delta_str:>15}")
        
        if results.get('full_model') and not np.isnan(results['full_model'].get('aic', np.nan)):
            delta_aic_full = results['full_model']['aic'] - best_aic
            aic_str = f"{results['full_model']['aic']:.1f}" if not np.isnan(results['full_model']['aic']) else "N/A"
            bic_str = f"{results['full_model']['bic']:.1f}" if not np.isnan(results['full_model']['bic']) else "N/A"
            delta_str = f"{delta_aic_full:.1f}" if not np.isnan(delta_aic_full) else "N/A"
            print(f"{'Full model (all predictors)':<25} {aic_str:>10} {bic_str:>10} {delta_str:>15}")
    
    # 4. Summary and recommendations
    print(f"\n{'='*80}")
    print("4. SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Find best individual predictor
    if valid_models:
        best_predictor = valid_models[0][0]
        best_model = valid_models[0][1]
        print(f"\nBest individual predictor: {best_predictor}")
        print(f"  Standardized coefficient: {best_model['std_coef']:.4f}")
        print(f"  p-value: {best_model['pvalue']:.4f}")
        print(f"  {'✓ Statistically significant' if best_model['significant'] else '✗ Not statistically significant'}")
    
    # Check if full model is better
    if results.get('full_model') and not np.isnan(results['full_model'].get('aic', np.nan)):
        full_aic = results['full_model']['aic']
        if valid_models and not np.isnan(best_aic) and full_aic < best_aic:
            print(f"\n✓ Full model (AIC={full_aic:.1f}) is better than best individual model (AIC={best_aic:.1f})")
            print(f"  → Multiple predictors together improve prediction")
            
            # Show which predictors remain significant in full model
            sig_in_full = [p for p, data in results['full_model']['predictors'].items() if data['significant']]
            if sig_in_full:
                print(f"  Significant predictors in full model: {', '.join(sig_in_full)}")
        elif valid_models and not np.isnan(best_aic):
            print(f"\n✗ Full model (AIC={full_aic:.1f}) is NOT better than best individual model (AIC={best_aic:.1f})")
            print(f"  → Single predictor is sufficient")
    
    # Compare standardized coefficients in full model
    if results.get('full_model'):
        print(f"\nRelative importance (standardized coefficients in full model):")
        pred_importance = [(p, abs(data['std_coef'])) for p, data in results['full_model']['predictors'].items()]
        pred_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (pred, abs_coef) in enumerate(pred_importance, 1):
            coef = results['full_model']['predictors'][pred]['std_coef']
            sig = results['full_model']['predictors'][pred]['significant']
            sig_marker = '*' if sig else ''
            print(f"  {i}. {pred:<20} {coef:>8.4f}{sig_marker:>1}")
    
    print(f"\n{'='*80}\n")
    
    # Save summary to file if requested
    if output_path:
        output_file = output_path / 'predictor_comparison_analysis.txt'
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PREDICTOR COMPARISON ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Data: {len(df)} observations across {df['benchmark'].nunique()} benchmarks\n")
            f.write(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}\n")
            
            # Write individual predictor models section
            f.write(f"\n{'='*80}\n")
            f.write("1. INDIVIDUAL PREDICTOR MODELS\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'AIC':>10} {'BIC':>10}\n")
            f.write(f"{'-'*70}\n")
            
            for predictor in predictors:
                if results.get(predictor) and results[predictor] is not None:
                    model_data = results[predictor]
                    if model_data.get('converged', False):
                        coef = model_data['std_coef']
                        pval = model_data['pvalue']
                        aic = model_data['aic']
                        bic = model_data['bic']
                        sig_marker = '*' if model_data.get('significant', False) else ''
                        aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
                        bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
                        f.write(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {aic_str:>10} {bic_str:>10}\n")
                    else:
                        f.write(f"{predictor:<20} {'NO CONV':>12} {'NO CONV':>12} {'NO CONV':>10} {'NO CONV':>10}\n")
                else:
                    f.write(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10} {'ERROR':>10}\n")
            
            # Write full model section
            f.write(f"\n{'='*80}\n")
            f.write("2. FULL MODEL (All predictors together)\n")
            f.write(f"{'='*80}\n")
            
            if results.get('full_model') and results['full_model'] is not None:
                predictor_names = {
                    'flow_mmd': 'Flow MMD',
                    'feature_mmd': 'Feature MMD',
                    'flow_recall': 'Flow Recall',
                    'resnet_recall': 'ResNet Recall',
                    'flow_precision': 'Flow Precision',
                    'resnet_precision': 'ResNet Precision'
                }
                formula_display = " + ".join([predictor_names.get(p, p) for p in predictors_to_scale])
                f.write(f"Model: PCK ~ {formula_display} + (1|benchmark)\n")
                f.write(f"\n{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'Significant':>12}\n")
                f.write(f"{'-'*60}\n")
                
                for predictor in predictors_to_scale:
                    if predictor in results['full_model']['predictors']:
                        data = results['full_model']['predictors'][predictor]
                        coef = data['std_coef']
                        pval = data['pvalue']
                        sig = data['significant']
                        sig_marker = '*' if sig else ''
                        f.write(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {'Yes' if sig else 'No':>12}\n")
                
                f.write(f"\nModel fit:\n")
                f.write(f"  Converged: {results['full_model']['converged']}\n")
                aic = results['full_model']['aic']
                bic = results['full_model'].get('bic', np.nan)
                aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
                bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
                f.write(f"  AIC: {aic_str}\n")
                f.write(f"  BIC: {bic_str}\n")
                if results['full_model'].get('result'):
                    result_full = results['full_model']['result']
                    llf_str = f"{result_full.llf:.2f}" if hasattr(result_full, 'llf') and not np.isnan(result_full.llf) else "N/A"
                    f.write(f"  Log-likelihood: {llf_str}\n")
                    if hasattr(result_full, 'cov_re') and hasattr(result_full.cov_re, 'iloc'):
                        f.write(f"  Random effect variance (benchmark): {result_full.cov_re.iloc[0, 0]:.4f}\n")
            else:
                f.write("Full model did not converge or encountered an error.\n")
            
            # Write model comparison section
            f.write(f"\n{'='*80}\n")
            f.write("3. MODEL COMPARISON (Lower AIC/BIC is better)\n")
            f.write(f"{'='*80}\n")
            
            if valid_models:
                f.write(f"{'Model':<25} {'AIC':>10} {'BIC':>10} {'ΔAIC vs Best':>15}\n")
                f.write(f"{'-'*60}\n")
                
                best_aic = valid_models[0][1]['aic']
                for name, model_data in valid_models:
                    delta_aic = model_data['aic'] - best_aic
                    aic_str = f"{model_data['aic']:.1f}" if not np.isnan(model_data['aic']) else "N/A"
                    bic_str = f"{model_data['bic']:.1f}" if not np.isnan(model_data['bic']) else "N/A"
                    delta_str = f"{delta_aic:.1f}" if not np.isnan(delta_aic) else "N/A"
                    f.write(f"{name:<25} {aic_str:>10} {bic_str:>10} {delta_str:>15}\n")
                
                if results.get('full_model') and not np.isnan(results['full_model'].get('aic', np.nan)):
                    delta_aic_full = results['full_model']['aic'] - best_aic
                    aic_str = f"{results['full_model']['aic']:.1f}" if not np.isnan(results['full_model']['aic']) else "N/A"
                    bic_str = f"{results['full_model']['bic']:.1f}" if not np.isnan(results['full_model'].get('bic', np.nan)) else "N/A"
                    delta_str = f"{delta_aic_full:.1f}" if not np.isnan(delta_aic_full) else "N/A"
                    f.write(f"{'Full model (all predictors)':<25} {aic_str:>10} {bic_str:>10} {delta_str:>15}\n")
            
            # Write summary and recommendations section
            f.write(f"\n{'='*80}\n")
            f.write("4. SUMMARY & RECOMMENDATIONS\n")
            f.write(f"{'='*80}\n")
            
            if valid_models:
                best_predictor = valid_models[0][0]
                best_model = valid_models[0][1]
                f.write(f"\nBest individual predictor: {best_predictor}\n")
                f.write(f"  Standardized coefficient: {best_model['std_coef']:.4f}\n")
                f.write(f"  p-value: {best_model['pvalue']:.4f}\n")
                f.write(f"  {'✓ Statistically significant' if best_model['significant'] else '✗ Not statistically significant'}\n")
            
            # Check if full model is better
            if results.get('full_model') and not np.isnan(results['full_model'].get('aic', np.nan)):
                full_aic = results['full_model']['aic']
                if valid_models and not np.isnan(best_aic) and full_aic < best_aic:
                    f.write(f"\n✓ Full model (AIC={full_aic:.1f}) is better than best individual model (AIC={best_aic:.1f})\n")
                    f.write(f"  → Multiple predictors together improve prediction\n")
                    
                    # Show which predictors remain significant in full model
                    sig_in_full = [p for p, data in results['full_model']['predictors'].items() if data['significant']]
                    if sig_in_full:
                        f.write(f"  Significant predictors in full model: {', '.join(sig_in_full)}\n")
                elif valid_models and not np.isnan(best_aic):
                    f.write(f"\n✗ Full model (AIC={full_aic:.1f}) is NOT better than best individual model (AIC={best_aic:.1f})\n")
                    f.write(f"  → Single predictor is sufficient\n")
            
            # Compare standardized coefficients in full model
            if results.get('full_model'):
                f.write(f"\nRelative importance (standardized coefficients in full model):\n")
                pred_importance = [(p, abs(data['std_coef'])) for p, data in results['full_model']['predictors'].items()]
                pred_importance.sort(key=lambda x: x[1], reverse=True)
                for i, (pred, abs_coef) in enumerate(pred_importance, 1):
                    coef = results['full_model']['predictors'][pred]['std_coef']
                    sig = results['full_model']['predictors'][pred]['significant']
                    sig_marker = '*' if sig else ''
                    f.write(f"  {i}. {pred:<20} {coef:>8.4f}{sig_marker:>1}\n")
            
            f.write(f"\n{'='*80}\n")
        
        print(f"Saved predictor comparison summary to: {output_file}")
    
    return results


def visualize_model_fit(result_full, df_scaled, output_path, predictors_to_scale):
    """
    Create diagnostic plots for the mixed-effects model.
    
    Creates:
    1. Predicted vs Observed PCK
    2. Residuals vs Fitted values
    3. Q-Q plot of residuals
    4. Random effects (benchmark intercepts) visualization
    
    Args:
        result_full: Fitted mixed-effects model result
        df_scaled: DataFrame with standardized predictors
        output_path: Path to save plots
        predictors_to_scale: List of predictor names
    """
    if result_full is None or not result_full.converged:
        print("Warning: Cannot visualize model - model did not converge or is None")
        return
    
    # Get predictions and residuals
    predicted = result_full.fittedvalues
    observed = df_scaled['pck'].values
    residuals = result_full.resid
    
    # Get random effects (benchmark intercepts)
    # Try multiple methods to extract random effects
    benchmark_names = sorted(df_scaled['benchmark'].unique())
    benchmark_intercepts = None
    
    # Method 1: Try to get from result_full.random_effects
    try:
        if hasattr(result_full, 'random_effects'):
            re = result_full.random_effects
            # Handle different formats
            if isinstance(re, dict):
                # Dict format: {group_name: value or array}
                extracted = {}
                for bm in benchmark_names:
                    if bm in re:
                        val = re[bm]
                        if isinstance(val, (list, np.ndarray)):
                            extracted[bm] = val[0] if len(val) > 0 else 0
                        elif isinstance(val, (int, float, np.number)):
                            extracted[bm] = val
                        else:
                            extracted[bm] = 0
                    else:
                        extracted[bm] = 0
                if any(v != 0 for v in extracted.values()):
                    benchmark_intercepts = [extracted[bm] for bm in benchmark_names]
            elif hasattr(re, 'iloc') or hasattr(re, '__getitem__'):
                # DataFrame or Series format
                extracted = {}
                for bm in benchmark_names:
                    try:
                        val = re[bm] if bm in re.index else (re.iloc[0] if hasattr(re, 'iloc') else 0)
                        if isinstance(val, (list, np.ndarray)):
                            extracted[bm] = val[0] if len(val) > 0 else 0
                        else:
                            extracted[bm] = float(val) if not pd.isna(val) else 0
                    except:
                        extracted[bm] = 0
                if any(v != 0 for v in extracted.values()):
                    benchmark_intercepts = [extracted[bm] for bm in benchmark_names]
    except Exception as e:
        pass  # Will try method 2
    
    # Method 2: Compute manually from residuals and group means
    if benchmark_intercepts is None:
        try:
            # Get overall intercept
            overall_intercept = result_full.fe_params.get('Intercept', 0)
            
            # For each benchmark, compute mean residual (which approximates random effect)
            benchmark_intercepts = []
            for bm in benchmark_names:
                bm_mask = df_scaled['benchmark'] == bm
                bm_residuals = residuals[bm_mask]
                # Mean residual for this benchmark (approximates random effect)
                bm_random_effect = np.mean(bm_residuals) if len(bm_residuals) > 0 else 0
                benchmark_intercepts.append(bm_random_effect)
        except Exception as e:
            # Fallback: compute from observed - predicted means
            try:
                benchmark_intercepts = []
                for bm in benchmark_names:
                    bm_mask = df_scaled['benchmark'] == bm
                    bm_observed = observed[bm_mask]
                    bm_predicted = predicted[bm_mask]
                    if len(bm_observed) > 0:
                        bm_re = np.mean(bm_observed) - np.mean(bm_predicted)
                    else:
                        bm_re = 0
                    benchmark_intercepts.append(bm_re)
            except:
                benchmark_intercepts = [0] * len(benchmark_names)
    
    # Final fallback: ensure we have valid data
    if benchmark_intercepts is None or len(benchmark_intercepts) != len(benchmark_names):
        # Compute from observed - predicted means (most reliable method)
        benchmark_intercepts = []
        for bm in benchmark_names:
            bm_mask = df_scaled['benchmark'] == bm
            bm_observed = observed[bm_mask]
            bm_predicted = predicted[bm_mask]
            if len(bm_observed) > 0:
                # Random effect = mean(observed) - mean(predicted) for this benchmark
                # This shows how much each benchmark deviates from the model's predictions
                bm_re = np.mean(bm_observed) - np.mean(bm_predicted)
            else:
                bm_re = 0
            benchmark_intercepts.append(bm_re)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Predicted vs Observed
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(observed, predicted, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Add perfect prediction line
    min_val = min(observed.min(), predicted.min())
    max_val = max(observed.max(), predicted.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    # Calculate R²
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    ax1.set_xlabel('Observed PCK', fontsize=11)
    ax1.set_ylabel('Predicted PCK', fontsize=11)
    ax1.set_title(f'Predicted vs Observed PCK\nR² = {r_squared:.3f}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals vs Fitted
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(predicted, residuals, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Fitted Values (Predicted PCK)', fontsize=11)
    ax2.set_ylabel('Residuals', fontsize=11)
    ax2.set_title('Residuals vs Fitted Values', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Q-Q plot of residuals
    ax3 = plt.subplot(2, 3, 3)
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals\n(Normality Check)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Random effects (benchmark intercepts)
    ax4 = plt.subplot(2, 3, 4)
    
    # benchmark_intercepts should already be computed by this point
    # Sort for better visualization
    sorted_indices = np.argsort(benchmark_intercepts)
    sorted_benchmarks = [benchmark_names[i] for i in sorted_indices]
    sorted_intercepts = [benchmark_intercepts[i] for i in sorted_indices]
    
    # Only plot if we have non-zero values
    if len(sorted_intercepts) > 0 and not all(abs(x) < 1e-10 for x in sorted_intercepts):
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_intercepts)))
        bars = ax4.barh(range(len(sorted_benchmarks)), sorted_intercepts, color=colors, edgecolor='black', linewidth=0.5)
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax4.set_yticks(range(len(sorted_benchmarks)))
        ax4.set_yticklabels(sorted_benchmarks, fontsize=9)
        ax4.set_xlabel('Random Effect (Benchmark Intercept)', fontsize=11)
        ax4.set_title('Random Effects by Benchmark\n(Deviation from Overall Intercept)', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, sorted_intercepts)):
            if abs(val) > 0.1:  # Only label if significant
                ax4.text(val, i, f' {val:.1f}', va='center', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'Random effects could not be extracted\n(All values are zero or unavailable)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=10)
        ax4.set_title('Random Effects by Benchmark\n(Data Unavailable)', fontsize=12, fontweight='bold')
    
    # 5. Residuals distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(residuals, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_xlabel('Residuals', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    ax5.text(0.05, 0.95, f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}',
             transform=ax5.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. Observed PCK by benchmark (with model predictions)
    ax6 = plt.subplot(2, 3, 6)
    benchmarks = sorted(df_scaled['benchmark'].unique())
    benchmark_observed_means = [df_scaled[df_scaled['benchmark'] == bm]['pck'].mean() for bm in benchmarks]
    benchmark_predicted_means = [predicted[df_scaled['benchmark'] == bm].mean() for bm in benchmarks]
    
    x_pos = np.arange(len(benchmarks))
    width = 0.35
    
    ax6.bar(x_pos - width/2, benchmark_observed_means, width, label='Observed Mean', 
           alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax6.bar(x_pos + width/2, benchmark_predicted_means, width, label='Predicted Mean',
           alpha=0.7, color='coral', edgecolor='black', linewidth=0.5)
    
    ax6.set_xlabel('Benchmark', fontsize=11)
    ax6.set_ylabel('Mean PCK', fontsize=11)
    ax6.set_title('Observed vs Predicted Mean PCK\nby Benchmark', fontsize=12, fontweight='bold')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(benchmarks, rotation=45, ha='right', fontsize=8)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Mixed-Effects Model Diagnostic Plots', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_file = output_path / 'model_fit_diagnostics.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved model fit diagnostics to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create 3D visualizations of coverage metrics vs PCK'
    )
    parser.add_argument(
        '--snapshots-dir', type=str, default='snapshots/',
        help='Directory containing snapshot subdirectories (default: snapshots/)'
    )
    parser.add_argument(
        '--coverage-csv', type=str, default='coverage_results.csv',
        help='Path to flow coverage CSV (default: coverage_results.csv)'
    )
    parser.add_argument(
        '--coverage-resnet-csv', type=str, default='coverage_resnet_results.csv',
        help='Path to resnet coverage CSV (default: coverage_resnet_results.csv)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='plots3d/',
        help='Output directory for plots (default: plots3d/)'
    )
    parser.add_argument(
        '--zscore', action='store_true',
        help='Also create z-scored versions of PCK'
    )
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load snapshots
    print("Loading snapshots...")
    snapshots_dir_path = Path(args.snapshots_dir).expanduser()
    if not snapshots_dir_path.exists():
        print(f"Error: Snapshots directory does not exist: {args.snapshots_dir}")
        return
    
    # Collect snapshot directories
    snapshot_dirs = []
    for item in snapshots_dir_path.iterdir():
        if item.is_dir():
            # Check if it looks like a snapshot directory (has training_summary.txt)
            if (item / 'training_summary.txt').exists():
                snapshot_dirs.append(str(item))
    
    if not snapshot_dirs:
        print(f"Error: No snapshot directories found in {args.snapshots_dir}")
        print("  (Looking for directories containing training_summary.txt)")
        return
    
    print(f"Found {len(snapshot_dirs)} snapshot directories")
    
    snapshots_data = []
    for snapshot_dir in snapshot_dirs:
        training_dataset, validation_data, metrics = parse_snapshot_directory(snapshot_dir)
        if validation_data:
            snapshots_data.append((training_dataset, validation_data, metrics, snapshot_dir))
    
    print(f"Loaded {len(snapshots_data)} snapshots")
    
    # Load coverage data
    print("\nLoading coverage data...")
    flow_coverage_lookup = load_coverage_lookup(args.coverage_csv)
    resnet_coverage_lookup = load_coverage_lookup(args.coverage_resnet_csv)
    
    print(f"Loaded {len(flow_coverage_lookup)} flow coverage entries from {args.coverage_csv}")
    print(f"Loaded {len(resnet_coverage_lookup)} resnet coverage entries from {args.coverage_resnet_csv}")
    
    # Collect data points
    print("\nCollecting data points...")
    data_points = collect_3d_data_points(snapshots_data, flow_coverage_lookup, resnet_coverage_lookup, debug=True)
    print(f"Collected {len(data_points)} data points with both flow and resnet coverage")
    
    if not data_points:
        print("Error: No data points found. Make sure both coverage CSV files exist and have matching entries.")
        return
    
    # Create color map
    all_datasets = set(p['training_dataset'] for p in data_points)
    num_datasets = len(all_datasets)
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
        if num_datasets > 20:
            colors = list(colors) * ((num_datasets // 20) + 1)
            colors = colors[:num_datasets]
    
    dataset_color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}
    
    # Create plots
    print("\nCreating 3D scatter plots...")
    create_3d_scatter_plot(data_points, output_path, dataset_color_map, zscore=False)
    if args.zscore:
        create_3d_scatter_plot(data_points, output_path, dataset_color_map, zscore=True)
    
    print("\nCreating 2D colormap plots...")
    create_2d_colormap_plot(data_points, output_path, dataset_color_map, zscore=False)
    create_2d_colormap_plot(data_points, output_path, dataset_color_map, zscore_by_benchmark=True)
    if args.zscore:
        create_2d_colormap_plot(data_points, output_path, dataset_color_map, zscore=True)
    
    print("\nCreating faceted by benchmark plots...")
    create_faceted_by_benchmark_plot(data_points, output_path, zscore_by_benchmark=False)
    create_faceted_by_benchmark_plot(data_points, output_path, zscore_by_benchmark=True)
    
    print("\nCreating faceted by training set plots...")
    create_faceted_by_training_set_plot(data_points, output_path, dataset_color_map, zscore_by_benchmark=False)
    create_faceted_by_training_set_plot(data_points, output_path, dataset_color_map, zscore_by_benchmark=True)
    
    # Create precision plots
    print("\nCollecting precision data points...")
    precision_data_points = collect_precision_data_points(snapshots_data, flow_coverage_lookup, resnet_coverage_lookup, debug=True)
    print(f"Collected {len(precision_data_points)} data points with both flow and resnet precision")
    
    if precision_data_points:
        print("\nCreating 2D precision colormap plots...")
        create_2d_precision_colormap_plot(precision_data_points, output_path, dataset_color_map, zscore=False)
        create_2d_precision_colormap_plot(precision_data_points, output_path, dataset_color_map, zscore_by_benchmark=True)
        if args.zscore:
            create_2d_precision_colormap_plot(precision_data_points, output_path, dataset_color_map, zscore=True)
    else:
        print("  Warning: No precision data points found. Make sure coverage CSV files have precision values.")
    
    # Load MMD data and create MMD vs PCK plots
    print("\nLoading MMD data...")
    flow_mmd_lookup = load_mmd_lookup('flow_mmd_results.csv')
    feature_mmd_lookup = load_mmd_lookup('feature_mmd_results.csv')
    
    print(f"Loaded {len(flow_mmd_lookup)} flow MMD entries from flow_mmd_results.csv")
    print(f"Loaded {len(feature_mmd_lookup)} feature MMD entries from feature_mmd_results.csv")
    
    if flow_mmd_lookup and feature_mmd_lookup:
        # Collect MMD data points
        print("\nCollecting MMD data points...")
        mmd_data_points = collect_mmd_data_points(snapshots_data, flow_mmd_lookup, feature_mmd_lookup, debug=True)
        print(f"Collected {len(mmd_data_points)} data points with both flow and feature MMD")
        
        if mmd_data_points:
            # Update color map to include all datasets from MMD data
            mmd_datasets = set(p['training_dataset'] for p in mmd_data_points)
            all_datasets_mmd = all_datasets | mmd_datasets
            num_datasets_mmd = len(all_datasets_mmd)
            if num_datasets_mmd <= 10:
                colors_mmd = plt.cm.tab10(np.linspace(0, 1, num_datasets_mmd))
            else:
                colors_mmd = plt.cm.tab20(np.linspace(0, 1, min(num_datasets_mmd, 20)))
                if num_datasets_mmd > 20:
                    colors_mmd = list(colors_mmd) * ((num_datasets_mmd // 20) + 1)
                    colors_mmd = colors_mmd[:num_datasets_mmd]
            
            dataset_color_map_mmd = {dataset: colors_mmd[i] for i, dataset in enumerate(sorted(all_datasets_mmd))}
            
            # Create 2D MMD colormap plots
            print("\nCreating 2D MMD colormap plots...")
            create_2d_mmd_colormap_plot(mmd_data_points, output_path, dataset_color_map_mmd, zscore=False)
            create_2d_mmd_colormap_plot(mmd_data_points, output_path, dataset_color_map_mmd, zscore_by_benchmark=True)
            if args.zscore:
                create_2d_mmd_colormap_plot(mmd_data_points, output_path, dataset_color_map_mmd, zscore=True)
        else:
            print("  Warning: No MMD data points found. Make sure both MMD CSV files exist and have matching entries.")
    else:
        print("  Skipping MMD plots (one or both MMD lookup files not available)")
    
    # Compare all predictors using mixed-effects regression
    if (flow_mmd_lookup and feature_mmd_lookup and 
        flow_coverage_lookup and resnet_coverage_lookup):
        print("\n" + "="*80)
        print("COMPARING ALL PREDICTORS (Flow MMD, Feature MMD, Flow Recall, ResNet Recall, Flow Precision, ResNet Precision)")
        print("="*80)
        all_predictors_data = collect_all_predictors_data_points(
            snapshots_data, flow_mmd_lookup, feature_mmd_lookup,
            flow_coverage_lookup, resnet_coverage_lookup, debug=True
        )
        
        if len(all_predictors_data) >= 10:
            df_all = pd.DataFrame(all_predictors_data)
            compare_predictors_with_mixed_effects(df_all, output_path=output_path, create_plots=True)
        else:
            print(f"\nWarning: Only {len(all_predictors_data)} data points with all four predictors.")
            print("  Need at least 10 points for reliable comparison.")
            print("  Make sure all CSV files (flow_mmd_results.csv, feature_mmd_results.csv,")
            print("  coverage_results.csv, coverage_resnet_results.csv) exist and have matching entries.")
    else:
        print("\nSkipping predictor comparison (missing required data files)")
        print("  Required: flow_mmd_results.csv, feature_mmd_results.csv,")
        print("            coverage_results.csv, coverage_resnet_results.csv")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
