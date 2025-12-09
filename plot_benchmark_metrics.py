#!/usr/bin/env python3
"""
Script to plot benchmark metrics across multiple snapshots.
For each benchmark, creates plots showing metric pairs (e.g., PCK vs MMD²)
with different training datasets shown by color and training progression
shown with line plots featuring bold first points.

Usage:
    python plot_benchmark_metrics.py --snapshots_dir snapshots/ --output-dir benchmark_plots/
"""

import argparse
import csv
import os
import sys
import re
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import functions from plot_metrics.py
from plot_metrics import (
    parse_training_summary,
    parse_validation_results,
    format_training_dataset_label,
    parse_directory_name,
    parse_snapshot_directory,
    collect_snapshot_directories
)

# Import plotting helper from plot_snapshot_metrics.py
from plot_snapshot_metrics import _plot_with_first_highlight


def load_snapshots(snapshot_dirs):
    """
    Load and parse multiple snapshots.
    
    Args:
        snapshot_dirs: List of snapshot directory paths
        
    Returns:
        List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
    """
    snapshots_data = []
    all_metrics = set()
    
    for snapshot_dir in snapshot_dirs:
        print(f"  Parsing: {snapshot_dir}")
        training_dataset, validation_data, metrics = parse_snapshot_directory(snapshot_dir)
        if validation_data:
            snapshots_data.append((training_dataset, validation_data, metrics, snapshot_dir))
            all_metrics.update(metrics)
            print(f"    Training dataset: {training_dataset}")
            print(f"    Metrics found: {len(metrics)}")
        else:
            print(f"    Warning: No validation data found, skipping")
    
    return snapshots_data, sorted(list(all_metrics))


def organize_by_benchmark(snapshots_data):
    """
    Organize data by benchmark, then by metric pairs.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        where validation_data_dict maps (benchmark, metric) -> list of (training_steps, value) tuples
        
    Returns:
        Dictionary mapping benchmark -> metric_pair -> list of (training_dataset, x_values, y_values) tuples
        where metric_pair is (metric1, metric2) and values are sorted by training_steps
    """
    benchmark_data = defaultdict(lambda: defaultdict(list))
    
    # Define metric pairs to plot
    metric_pairs = [
        ('pck', 'mmd2_pred_corr_vs_pred_miss'),
        ('pck', 'mmd2_pred_corr_vs_gt'),
        ('pck', 'mmd2_pred_miss_vs_gt'),
    ]
    
    for training_dataset, validation_data, metrics, _ in snapshots_data:
        # Group data by benchmark
        benchmark_metrics = defaultdict(dict)
        for (benchmark, metric), data_points in validation_data.items():
            if benchmark not in benchmark_metrics:
                benchmark_metrics[benchmark] = {}
            benchmark_metrics[benchmark][metric] = data_points
        
        # For each benchmark, extract metric pairs
        for benchmark, metrics_dict in benchmark_metrics.items():
            for metric1, metric2 in metric_pairs:
                if metric1 in metrics_dict and metric2 in metrics_dict:
                    # Get data points for both metrics
                    metric1_data = metrics_dict[metric1]
                    metric2_data = metrics_dict[metric2]
                    
                    # Create dictionaries keyed by training_steps for alignment
                    metric1_dict = {step: value for step, value in metric1_data}
                    metric2_dict = {step: value for step, value in metric2_data}
                    
                    # Find common training_steps
                    common_steps = sorted(set(metric1_dict.keys()) & set(metric2_dict.keys()))
                    
                    if len(common_steps) > 0:
                        # Extract aligned values
                        x_values = [metric1_dict[step] for step in common_steps]
                        y_values = [metric2_dict[step] for step in common_steps]
                        
                        # Store as pandas Series for compatibility with _plot_with_first_highlight
                        x_series = pd.Series(x_values)
                        y_series = pd.Series(y_values)
                        
                        benchmark_data[benchmark][(metric1, metric2)].append(
                            (training_dataset, x_series, y_series)
                        )
    
    return benchmark_data


def plot_metric_pair(ax, benchmark, metric1, metric2, training_datasets_data, dataset_color_map):
    """
    Plot one metric pair for a benchmark.
    
    Args:
        ax: Matplotlib axis object
        benchmark: Benchmark name
        metric1: First metric name (x-axis)
        metric2: Second metric name (y-axis)
        training_datasets_data: List of (training_dataset, x_values, y_values) tuples
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    for training_dataset, x_values, y_values in training_datasets_data:
        color = dataset_color_map.get(training_dataset, 'black')
        label = training_dataset
        
        # Use _plot_with_first_highlight for bold first point
        _plot_with_first_highlight(
            ax, x_values, y_values,
            label=label,
            color=color,
            marker='o',
            markersize=4,
            linewidth=1.5
        )


def create_benchmark_plots(benchmark_data, output_dir, benchmarks_filter=None, metrics_filter=None, dataset_color_map=None):
    """
    Create plots for each benchmark.
    
    Args:
        benchmark_data: Dictionary mapping benchmark -> metric_pair -> list of (dataset, x_values, y_values)
        output_dir: Output directory path
        benchmarks_filter: Optional list of benchmarks to plot (None = all)
        metrics_filter: Optional list of metric pairs to plot (None = all)
        dataset_color_map: Optional dictionary mapping training_dataset -> color (if None, will create one)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create by_benchmark subdirectory
    benchmark_output_path = output_path / 'by_benchmark'
    benchmark_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all training datasets for color mapping (if not provided)
    if dataset_color_map is None:
        all_datasets = set()
        for benchmark_dict in benchmark_data.values():
            for metric_pair_data in benchmark_dict.values():
                for dataset, _, _ in metric_pair_data:
                    all_datasets.add(dataset)
        
        # Create color map (same as plot_metrics.py)
        num_datasets = len(all_datasets)
        if num_datasets <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
            if num_datasets > 20:
                colors = list(colors) * ((num_datasets // 20) + 1)
                colors = colors[:num_datasets]
        
        dataset_color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}
    
    # Filter benchmarks if specified
    benchmarks_to_plot = list(benchmark_data.keys())
    if benchmarks_filter:
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]
    
    # Create plots for each benchmark
    for benchmark in benchmarks_to_plot:
        if benchmark not in benchmark_data:
            continue
        
        benchmark_dir = benchmark_output_path / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating plots for benchmark: {benchmark}")
        
        # Get metric pairs for this benchmark
        metric_pairs = list(benchmark_data[benchmark].keys())
        if metrics_filter:
            metric_pairs = [mp for mp in metric_pairs if mp in metrics_filter]
        
        for metric1, metric2 in metric_pairs:
            if (metric1, metric2) not in benchmark_data[benchmark]:
                continue
            
            training_datasets_data = benchmark_data[benchmark][(metric1, metric2)]
            
            if not training_datasets_data:
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 7))
            
            # Plot the metric pair
            plot_metric_pair(
                ax, benchmark, metric1, metric2,
                training_datasets_data, dataset_color_map
            )
            
            # Format metric names for display
            metric1_display = metric1.replace('_', ' ').replace('mmd2', 'MMD²').title()
            metric2_display = metric2.replace('_', ' ').replace('mmd2', 'MMD²').title()
            
            # Set labels and title
            ax.set_xlabel(metric1_display, fontsize=12)
            ax.set_ylabel(metric2_display, fontsize=12)
            ax.set_title(f'{metric1_display} vs {metric2_display} - {benchmark.upper()}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save plot
            safe_metric1 = metric1.replace('mmd2_', 'mmd2').replace('_', '_')
            safe_metric2 = metric2.replace('mmd2_', 'mmd2').replace('_', '_')
            output_file = benchmark_dir / f'{safe_metric1}_vs_{safe_metric2}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"  Saved: {output_file}")
            plt.close()


def organize_pck_vs_steps_by_benchmark(snapshots_data):
    """
    Organize PCK vs training_steps data by benchmark.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        where validation_data_dict maps (benchmark, metric) -> list of (training_steps, value) tuples
        
    Returns:
        Dictionary mapping benchmark -> list of (training_dataset, steps_array, pck_array) tuples
    """
    benchmark_pck_data = defaultdict(list)
    
    for training_dataset, validation_data, _, _ in snapshots_data:
        # Group data by benchmark
        benchmark_metrics = defaultdict(dict)
        for (benchmark, metric), data_points in validation_data.items():
            if benchmark not in benchmark_metrics:
                benchmark_metrics[benchmark] = {}
            benchmark_metrics[benchmark][metric] = data_points
        
        # Extract PCK vs training_steps for each benchmark
        for benchmark, metrics_dict in benchmark_metrics.items():
            if 'pck' in metrics_dict:
                pck_data = metrics_dict['pck']
                # Sort by training_steps
                pck_data_sorted = sorted(pck_data, key=lambda x: x[0])
                steps = [point[0] for point in pck_data_sorted]
                pck_values = [point[1] for point in pck_data_sorted]
                
                benchmark_pck_data[benchmark].append((training_dataset, steps, pck_values))
    
    return benchmark_pck_data


def create_pck_vs_steps_plots(benchmark_pck_data, output_dir, benchmarks_filter=None, dataset_color_map=None):
    """
    Create PCK vs training_steps plots for each benchmark.
    
    Args:
        benchmark_pck_data: Dictionary mapping benchmark -> list of (training_dataset, steps, pck_values) tuples
        output_dir: Output directory path
        benchmarks_filter: Optional list of benchmarks to plot (None = all)
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create by_benchmark subdirectory
    benchmark_output_path = output_path / 'by_benchmark'
    benchmark_output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter benchmarks if specified
    benchmarks_to_plot = list(benchmark_pck_data.keys())
    if benchmarks_filter:
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]
    
    # Create plots for each benchmark
    for benchmark in benchmarks_to_plot:
        if benchmark not in benchmark_pck_data:
            continue
        
        training_datasets_data = benchmark_pck_data[benchmark]
        
        if not training_datasets_data:
            continue
        
        benchmark_dir = benchmark_output_path / benchmark
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Plot each training dataset
        for training_dataset, steps, pck_values in training_datasets_data:
            color = dataset_color_map.get(training_dataset, 'black') if dataset_color_map else None
            
            ax.plot(steps, pck_values,
                   marker='o', label=training_dataset,
                   markersize=4, linewidth=1.5,
                   color=color, alpha=0.8)
        
        # Set labels and title
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('PCK (%)', fontsize=12)
        ax.set_title(f'PCK vs Training Steps - {benchmark.upper()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_file = benchmark_dir / 'pck_vs_steps.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()


def load_mmd_lookup(csv_path='flow_mmd_results.csv'):
    """
    Load MMD² data from CSV file and create a bidirectional lookup.
    Handles new format with splits: treats dataset+split as unique identifiers.
    Skips identical comparisons (same dataset AND same split).
    
    Args:
        csv_path: Path to flow_mmd_results.csv file
        
    Returns:
        Dictionary mapping (dataset1_split1, dataset2_split2) -> mmd2 value
        Also includes mappings for (dataset1, dataset2) without splits for backward compatibility
        Works for both orderings
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot create MMD lookup.")
        return {}
    
    mmd_lookup = {}
    
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            dataset1 = str(row['dataset1']).lower()
            dataset2 = str(row['dataset2']).lower()
            
            # Check if split columns exist (new format)
            if 'split1' in df.columns and 'split2' in df.columns:
                split1 = str(row['split1']).lower()
                split2 = str(row['split2']).lower()
                
                # Skip identical comparisons (same dataset AND same split)
                if dataset1 == dataset2 and split1 == split2:
                    continue
                
                # Create unique identifiers with splits
                dataset1_id = f"{dataset1}_{split1}"
                dataset2_id = f"{dataset2}_{split2}"
                
                mmd2 = float(row['mmd2'])
                
                # Store both orderings with split identifiers
                mmd_lookup[(dataset1_id, dataset2_id)] = mmd2
                mmd_lookup[(dataset2_id, dataset1_id)] = mmd2
                
                # Also store without explicit split in key for backward compatibility
                # This allows lookups like (dataset1, dataset2) to work
                # But prioritize the more specific (dataset1_split1, dataset2_split2) format
                mmd_lookup[(dataset1, dataset2)] = mmd2
                mmd_lookup[(dataset2, dataset1)] = mmd2
            else:
                # Old format without splits
                mmd2 = float(row['mmd2'])
                
                # Skip identical comparisons
                if dataset1 == dataset2:
                    continue
                
                # Store both orderings
                mmd_lookup[(dataset1, dataset2)] = mmd2
                mmd_lookup[(dataset2, dataset1)] = mmd2
            
    except Exception as e:
        print(f"Warning: Could not load MMD lookup from {csv_path}: {e}")
        return {}
    
    return mmd_lookup


def parse_training_dataset_from_summary(summary_path):
    """
    Parse training_summary.txt to extract the base training dataset name.
    
    Args:
        summary_path: Path to training_summary.txt file
        
    Returns:
        Base training dataset name (string) or None if not found
    """
    if not os.path.exists(summary_path):
        return None
    
    try:
        with open(summary_path, 'r') as f:
            for line in f:
                if line.startswith('Train dataset:'):
                    dataset = line.split('Train dataset:')[1].strip()
                    return dataset.lower()  # Normalize to lowercase for lookup
    except Exception as e:
        print(f"Warning: Could not parse training dataset from {summary_path}: {e}")
        return None
    
    return None


def parse_best_performance_from_summary(summary_path):
    """
    Parse training_summary.txt to extract best performance per benchmark.
    
    Args:
        summary_path: Path to training_summary.txt file
        
    Returns:
        Dictionary mapping benchmark -> best_pck_value (float)
    """
    if not os.path.exists(summary_path):
        return {}
    
    best_performance = {}
    
    try:
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        
        in_best_section = False
        for line in lines:
            line = line.strip()
            
            # Find the "BEST PERFORMANCE PER BENCHMARK:" section
            if 'BEST PERFORMANCE PER BENCHMARK:' in line:
                in_best_section = True
                continue
            
            if in_best_section:
                # Stop when we hit the next section
                if line.startswith('-') and len(line) > 10:  # Separator line
                    continue
                if line.startswith('MOTION-AWARE') or line.startswith('TRAINING CONFIGURATION'):
                    break
                
                # Parse lines like "tss         : 75.28% PCK (epoch 6, epoch_6.pth)"
                if ':' in line and '%' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        benchmark = parts[0].strip()
                        value_part = parts[1].strip()
                        # Extract percentage value
                        match = re.search(r'(\d+\.?\d*)%', value_part)
                        if match:
                            pck_value = float(match.group(1))
                            best_performance[benchmark] = pck_value
                            
    except Exception as e:
        print(f"Warning: Could not parse best performance from {summary_path}: {e}")
        return {}
    
    return best_performance


def create_mmd_vs_pck_scatter_plot(snapshots_data, mmd_lookup, output_dir, dataset_color_map):
    """
    Create scatter plot showing training dataset MMD² vs best PCK.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        mmd_lookup: Dictionary mapping (dataset1_split1, dataset2_split2) -> mmd2 value
                    Also supports (dataset1, dataset2) format for backward compatibility
        output_dir: Output directory path
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all data points: (mmd2, best_pck, training_dataset, benchmark)
    data_points = []
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get base training dataset name from summary (for MMD lookup)
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            print(f"  Warning: Could not parse training dataset from {summary_path}")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            print(f"  Warning: No best performance data found in {summary_path}")
            continue
        
        # For each benchmark, look up MMD² and store data point
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up MMD² with splits
            # Training datasets are typically "train" split, benchmarks are typically "test" or "val" split
            mmd2 = None
            
            # Try with explicit splits (new format)
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Try test split first, then val split
            mmd2 = mmd_lookup.get((training_dataset_train, benchmark_test))
            if mmd2 is None:
                mmd2 = mmd_lookup.get((training_dataset_train, benchmark_val))
            
            # Fall back to old format without splits (backward compatibility)
            if mmd2 is None:
                mmd2 = mmd_lookup.get((base_training_dataset, benchmark_lower))
            
            if mmd2 is not None:
                data_points.append({
                    'mmd2': mmd2,
                    'best_pck': best_pck,
                    'training_dataset': training_dataset_label,  # Use formatted label for display
                    'benchmark': benchmark
                })
            else:
                print(f"  Warning: MMD² not found for ({base_training_dataset}, {benchmark_lower})")
    
    if not data_points:
        print("Warning: No data points collected for MMD vs PCK scatter plot")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group points by training dataset for plotting
    datasets_points = defaultdict(list)
    for point in data_points:
        datasets_points[point['training_dataset']].append(point)
    
    # Plot each training dataset with different color
    for training_dataset, points in datasets_points.items():
        mmd2_values = [p['mmd2'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        color = dataset_color_map.get(training_dataset, 'black')
        
        ax.scatter(mmd2_values, pck_values, 
                  color=color, label=training_dataset, 
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add benchmark labels (optional - can be commented out if too cluttered)
        for point in points:
            ax.annotate(point['benchmark'], 
                       (point['mmd2'], point['best_pck']),
                       fontsize=7, alpha=0.6,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Dataset MMD² vs Benchmark', fontsize=12)
    ax.set_ylabel('Best PCK (%)', fontsize=12)
    ax.set_title('Training Dataset MMD² vs Best PCK Performance', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / 'training_mmd_vs_best_pck.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved MMD vs PCK scatter plot: {output_file}")
    plt.close()


def create_feature_mmd_vs_pck_scatter_plot(snapshots_data, mmd_lookup, output_dir, dataset_color_map):
    """
    Create scatter plot showing training dataset Feature MMD² vs best PCK.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        mmd_lookup: Dictionary mapping (dataset1_split1, dataset2_split2) -> mmd2 value
                    Also supports (dataset1, dataset2) format for backward compatibility
        output_dir: Output directory path
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all data points: (mmd2, best_pck, training_dataset, benchmark)
    data_points = []
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get base training dataset name from summary (for MMD lookup)
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            print(f"  Warning: Could not parse training dataset from {summary_path}")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            print(f"  Warning: No best performance data found in {summary_path}")
            continue
        
        # For each benchmark, look up MMD² and store data point
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up MMD² with splits
            # Training datasets are typically "train" split, benchmarks are typically "test" or "val" split
            mmd2 = None
            
            # Try with explicit splits (new format)
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Try test split first, then val split
            mmd2 = mmd_lookup.get((training_dataset_train, benchmark_test))
            if mmd2 is None:
                mmd2 = mmd_lookup.get((training_dataset_train, benchmark_val))
            
            # Fall back to old format without splits (backward compatibility)
            if mmd2 is None:
                mmd2 = mmd_lookup.get((base_training_dataset, benchmark_lower))
            
            if mmd2 is not None:
                data_points.append({
                    'mmd2': mmd2,
                    'best_pck': best_pck,
                    'training_dataset': training_dataset_label,  # Use formatted label for display
                    'benchmark': benchmark
                })
            else:
                print(f"  Warning: Feature MMD² not found for ({base_training_dataset}, {benchmark_lower})")
    
    if not data_points:
        print("Warning: No data points collected for Feature MMD vs PCK scatter plot")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group points by training dataset for plotting
    datasets_points = defaultdict(list)
    for point in data_points:
        datasets_points[point['training_dataset']].append(point)
    
    # Plot each training dataset with different color
    for training_dataset, points in datasets_points.items():
        mmd2_values = [p['mmd2'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        color = dataset_color_map.get(training_dataset, 'black')
        
        ax.scatter(mmd2_values, pck_values, 
                  color=color, label=training_dataset, 
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add benchmark labels (optional - can be commented out if too cluttered)
        for point in points:
            ax.annotate(point['benchmark'], 
                       (point['mmd2'], point['best_pck']),
                       fontsize=7, alpha=0.6,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Dataset Feature MMD² vs Benchmark', fontsize=12)
    ax.set_ylabel('Best PCK (%)', fontsize=12)
    ax.set_title('Training Dataset Feature MMD² vs Best PCK Performance', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / 'training_feature_mmd_vs_best_pck.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved Feature MMD vs PCK scatter plot: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot benchmark metrics across multiple snapshots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all benchmarks from snapshots directory
  python plot_benchmark_metrics.py --snapshots_dir snapshots/ --output-dir benchmark_plots/
  
  # Plot specific benchmarks
  python plot_benchmark_metrics.py --snapshots_dir snapshots/ --benchmarks spair pointodyssey
  
  # Explicit list of snapshots
  python plot_benchmark_metrics.py --snapshots snapshots/exp1 snapshots/exp2 --output-dir plots/
        """
    )
    
    parser.add_argument(
        '--snapshots',
        nargs='+',
        default=None,
        help='List of snapshot directory paths (can be combined with --snapshots_dir or stdin)'
    )
    
    parser.add_argument(
        '--snapshots_dir',
        type=str,
        default=None,
        help='Directory containing snapshot subdirectories (auto-detects all snapshots)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./benchmark_plots/',
        help='Output directory for plots (default: ./benchmark_plots/)'
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=None,
        help='Filter specific benchmarks to plot (default: all benchmarks)'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='Filter specific metric pairs to plot (format: "metric1,metric2") (default: all metric pairs)'
    )
    
    args = parser.parse_args()
    
    # Collect snapshot directories from all sources
    snapshot_dirs = collect_snapshot_directories(args)
    
    if not snapshot_dirs:
        print("Error: No snapshot directories found!")
        print("Please provide snapshots via:")
        print("  --snapshots <dir1> <dir2> ...")
        print("  --snapshots_dir <parent_dir>")
        print("  Or pipe directory paths via stdin")
        sys.exit(1)
    
    print(f"Found {len(snapshot_dirs)} snapshot directory(ies)")
    
    # Load all snapshots
    print("\nLoading snapshots...")
    snapshots_data, all_metrics = load_snapshots(snapshot_dirs)
    
    if not snapshots_data:
        print("Error: No valid snapshot data found!")
        sys.exit(1)
    
    print(f"\nFound {len(all_metrics)} metrics: {all_metrics}")
    
    # Organize data by benchmark
    print("\nOrganizing data by benchmark...")
    benchmark_data = organize_by_benchmark(snapshots_data)
    
    if not benchmark_data:
        print("Error: No benchmark data found!")
        sys.exit(1)
    
    print(f"Found {len(benchmark_data)} benchmarks: {list(benchmark_data.keys())}")
    
    # Parse metrics filter if provided
    metrics_filter = None
    if args.metrics:
        metrics_filter = []
        for metric_str in args.metrics:
            parts = metric_str.split(',')
            if len(parts) == 2:
                metrics_filter.append((parts[0].strip(), parts[1].strip()))
            else:
                print(f"Warning: Invalid metric pair format '{metric_str}', expected 'metric1,metric2'")
    
    # Create plots
    print("\nCreating plots...")
    
    # Get all training datasets for color mapping (needed for scatter plot)
    all_datasets = set()
    for training_dataset, _, _, _ in snapshots_data:
        all_datasets.add(training_dataset)
    
    # Create color map (same as plot_metrics.py)
    num_datasets = len(all_datasets)
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
        if num_datasets > 20:
            colors = list(colors) * ((num_datasets // 20) + 1)
            colors = colors[:num_datasets]
    
    dataset_color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}
    
    # Create benchmark plots
    create_benchmark_plots(
        benchmark_data,
        args.output_dir,
        benchmarks_filter=args.benchmarks,
        metrics_filter=metrics_filter,
        dataset_color_map=dataset_color_map
    )
    
    # Create PCK vs training steps plots for each benchmark
    print("\nCreating PCK vs Training Steps plots...")
    benchmark_pck_data = organize_pck_vs_steps_by_benchmark(snapshots_data)
    create_pck_vs_steps_plots(
        benchmark_pck_data,
        args.output_dir,
        benchmarks_filter=args.benchmarks,
        dataset_color_map=dataset_color_map
    )
    
    # Load MMD lookup and create scatter plot
    print("\nCreating Flow MMD vs PCK scatter plot...")
    mmd_lookup = load_mmd_lookup('flow_mmd_results.csv')
    if mmd_lookup:
        create_mmd_vs_pck_scatter_plot(
            snapshots_data,
            mmd_lookup,
            args.output_dir,
            dataset_color_map
        )
    else:
        print("  Skipping Flow MMD vs PCK scatter plot (MMD lookup not available)")
    
    # Load Feature MMD lookup and create scatter plot
    print("\nCreating Feature MMD vs PCK scatter plot...")
    feature_mmd_lookup = load_mmd_lookup('feature_mmd_results.csv')
    if feature_mmd_lookup:
        create_feature_mmd_vs_pck_scatter_plot(
            snapshots_data,
            feature_mmd_lookup,
            args.output_dir,
            dataset_color_map
        )
    else:
        print("  Skipping Feature MMD vs PCK scatter plot (Feature MMD lookup not available)")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
