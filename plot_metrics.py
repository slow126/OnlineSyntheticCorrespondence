#!/usr/bin/env python3
"""
Visualize all metrics vs training steps for multiple training datasets.
Creates one plot per metric comparing different training datasets/benchmarks.
"""

import argparse
import csv
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_training_summary(summary_path):
    """
    Parse training_summary.txt to extract training dataset name.
    
    Args:
        summary_path: Path to training_summary.txt file
        
    Returns:
        Training dataset name, or None if not found
    """
    if not os.path.exists(summary_path):
        return None
    
    try:
        with open(summary_path, 'r') as f:
            for line in f:
                if 'Train dataset:' in line:
                    # Extract dataset name after colon
                    dataset = line.split('Train dataset:')[1].strip()
                    return dataset
    except Exception as e:
        print(f"Warning: Could not parse {summary_path}: {e}")
        return None
    
    return None


def parse_validation_results(csv_path):
    """
    Parse validation_results.csv to extract all metrics vs training_steps data.
    
    Args:
        csv_path: Path to validation_results.csv file
        
    Returns:
        Dictionary mapping (benchmark, metric) -> list of (training_steps, value) tuples
        Also returns list of all available metrics
    """
    if not os.path.exists(csv_path):
        return {}, []
    
    data = defaultdict(lambda: defaultdict(list))
    metrics_set = set()
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    benchmark = row['benchmark']
                    training_steps = int(row['training_steps'])
                    
                    # Extract all metrics (skip non-metric columns)
                    skip_columns = {'epoch', 'training_steps', 'benchmark'}
                    for metric_name, metric_value in row.items():
                        if metric_name in skip_columns:
                            continue
                        
                        # Skip empty values
                        if not metric_value or metric_value.strip() == '':
                            continue
                        
                        try:
                            value = float(metric_value)
                            data[benchmark][metric_name].append((training_steps, value))
                            metrics_set.add(metric_name)
                        except ValueError:
                            # Skip non-numeric values
                            continue
                            
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping row in {csv_path}: {e}")
                    continue
        
        # Sort by training_steps for each benchmark/metric combination
        for benchmark in data:
            for metric in data[benchmark]:
                data[benchmark][metric].sort(key=lambda x: x[0])
        
        # Convert to simpler structure: (benchmark, metric) -> list of (steps, value)
        result = {}
        for benchmark, metrics_dict in data.items():
            for metric, data_points in metrics_dict.items():
                result[(benchmark, metric)] = data_points
                
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
        return {}, []
    
    return result, sorted(list(metrics_set))


def parse_snapshot_directory(snapshot_dir):
    """
    Parse a snapshot directory to extract training dataset and validation results.
    
    Args:
        snapshot_dir: Path to snapshot directory
        
    Returns:
        Tuple of (training_dataset_name, validation_data_dict, metrics_list)
        Returns (None, {}, []) if parsing fails
    """
    snapshot_path = Path(snapshot_dir)
    
    # Get training dataset name from training_summary.txt
    summary_path = snapshot_path / 'training_summary.txt'
    training_dataset = parse_training_summary(summary_path)
    
    # Fallback to directory name if summary not found
    if training_dataset is None:
        training_dataset = snapshot_path.name
        print(f"Warning: Could not find training dataset in {summary_path}, using directory name: {training_dataset}")
    
    # Parse validation results
    csv_path = snapshot_path / 'validation_results.csv'
    validation_data, metrics = parse_validation_results(csv_path)
    
    if not validation_data:
        print(f"Warning: No validation data found in {csv_path}")
    
    return training_dataset, validation_data, metrics


def organize_data_by_metric(snapshots_data, zero_train_step_data=None):
    """
    Organize data by metric, then by benchmark.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list) tuples
        zero_train_step_data: Optional (validation_data_dict, metrics_list) from zero_train_step
        
    Returns:
        Dictionary mapping metric -> benchmark -> list of (training_dataset, steps_array, values_array) tuples
        Also returns list of all metrics
    """
    metric_data = defaultdict(lambda: defaultdict(list))
    all_metrics = set()
    
    # Add zero_train_step baseline if provided
    if zero_train_step_data:
        zero_data, zero_metrics = zero_train_step_data
        all_metrics.update(zero_metrics)
        for (benchmark, metric), data_points in zero_data.items():
            if data_points:
                steps = [point[0] for point in data_points]
                values = [point[1] for point in data_points]
                metric_data[metric][benchmark].append(('zero_train_step', steps, values))
    
    # Add data from each snapshot
    for training_dataset, validation_data, metrics in snapshots_data:
        all_metrics.update(metrics)
        for (benchmark, metric), data_points in validation_data.items():
            if data_points:
                steps = [point[0] for point in data_points]
                values = [point[1] for point in data_points]
                metric_data[metric][benchmark].append((training_dataset, steps, values))
    
    return metric_data, sorted(list(all_metrics))


def create_plots(metric_data, output_dir, metrics_filter=None, benchmarks_filter=None):
    """
    Create visualization plots for each metric.
    
    Args:
        metric_data: Dictionary mapping metric -> benchmark -> list of (dataset, steps, values) tuples
        output_dir: Output directory for plots
        metrics_filter: Optional list of metrics to plot (None = all)
        benchmarks_filter: Optional list of benchmarks to plot (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter metrics if specified
    metrics_to_plot = list(metric_data.keys())
    if metrics_filter:
        metrics_to_plot = [m for m in metrics_to_plot if m in metrics_filter]
    
    # Get distinct colors for each training dataset
    all_datasets = set()
    all_benchmarks = set()
    for metric_dict in metric_data.values():
        for benchmark, data_list in metric_dict.items():
            all_benchmarks.add(benchmark)
            for dataset, _, _ in data_list:
                all_datasets.add(dataset)
    
    # Use a colormap for distinct colors - one color per training dataset
    # Use tab10 for up to 10 datasets, then cycle through tab20
    num_datasets = len(all_datasets)
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
    else:
        # Use tab20 for more datasets
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
        if num_datasets > 20:
            # For more than 20, cycle through tab20
            colors = list(colors) * ((num_datasets // 20) + 1)
            colors = colors[:num_datasets]
    
    # Create color map: each training dataset gets a unique color
    dataset_color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}
    
    # Create marker styles for benchmarks to further distinguish them
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'X', 'd']
    benchmark_marker_map = {benchmark: markers[i % len(markers)] for i, benchmark in enumerate(sorted(all_benchmarks))}
    
    # Create one plot per metric
    for metric in metrics_to_plot:
        if metric not in metric_data or not metric_data[metric]:
            print(f"Warning: No data for metric {metric}, skipping")
            continue
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Filter benchmarks if specified
        benchmarks_to_plot = list(metric_data[metric].keys())
        if benchmarks_filter:
            benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]
        
        # Plot each benchmark and training dataset combination
        for benchmark in benchmarks_to_plot:
            if benchmark not in metric_data[metric]:
                continue
                
            for training_dataset, steps, values in metric_data[metric][benchmark]:
                # Use different line style for zero_train_step
                linestyle = '--' if training_dataset == 'zero_train_step' else '-'
                linewidth = 2 if training_dataset == 'zero_train_step' else 1.5
                label = f"{benchmark} ({training_dataset})" if training_dataset != 'zero_train_step' else f"{benchmark} (Untrained baseline)"
                
                # Use color based on training dataset (primary distinction)
                color = dataset_color_map.get(training_dataset, 'black')
                # Use marker based on benchmark (secondary distinction)
                marker = benchmark_marker_map.get(benchmark, 'o')
                
                ax.plot(steps, values, 
                       color=color,
                       linestyle=linestyle,
                       linewidth=linewidth,
                       marker=marker,
                       markersize=5,
                       label=label,
                       alpha=0.8)
        
        # Format metric name for display
        metric_display = metric.replace('_', ' ').title()
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel(metric_display, fontsize=12)
        ax.set_title(f'{metric_display} vs Training Steps', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9, ncol=2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save combined plot (all benchmarks together)
        output_file = output_path / f'{metric}_vs_steps.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        plt.close()
    
    # Create individual benchmark plots in subdirectory
    create_individual_benchmark_plots(metric_data, output_path, metrics_to_plot, benchmarks_filter, dataset_color_map)


def create_individual_benchmark_plots(metric_data, output_path, metrics_to_plot, benchmarks_filter, dataset_color_map):
    """
    Create individual plots for each benchmark (one plot per metric per benchmark).
    Each plot shows all training datasets for that specific benchmark.
    
    Args:
        metric_data: Dictionary mapping metric -> benchmark -> list of (dataset, steps, values) tuples
        output_path: Base output directory path
        metrics_to_plot: List of metrics to plot
        benchmarks_filter: Optional list of benchmarks to plot (None = all)
        dataset_color_map: Color map for training datasets
    """
    # Create subdirectory for individual benchmark plots
    benchmark_output_path = output_path / 'by_benchmark'
    benchmark_output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all benchmarks
    all_benchmarks = set()
    for metric_dict in metric_data.values():
        all_benchmarks.update(metric_dict.keys())
    
    # Filter benchmarks if specified
    benchmarks_to_plot = list(all_benchmarks)
    if benchmarks_filter:
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]
    
    # Create one plot per metric per benchmark
    for metric in metrics_to_plot:
        if metric not in metric_data:
            continue
        
        for benchmark in benchmarks_to_plot:
            if benchmark not in metric_data[metric] or not metric_data[metric][benchmark]:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot all training datasets for this benchmark
            for training_dataset, steps, values in metric_data[metric][benchmark]:
                # Use different line style for zero_train_step
                linestyle = '--' if training_dataset == 'zero_train_step' else '-'
                linewidth = 2 if training_dataset == 'zero_train_step' else 1.5
                label = training_dataset if training_dataset != 'zero_train_step' else 'Untrained (baseline)'
                
                # Use color based on training dataset
                color = dataset_color_map.get(training_dataset, 'black')
                
                ax.plot(steps, values, 
                       color=color,
                       linestyle=linestyle,
                       linewidth=linewidth,
                       marker='o',
                       markersize=5,
                       label=label,
                       alpha=0.8)
            
            # Format metric name for display
            metric_display = metric.replace('_', ' ').title()
            
            ax.set_xlabel('Training Steps', fontsize=12)
            ax.set_ylabel(metric_display, fontsize=12)
            ax.set_title(f'{metric_display} vs Training Steps - {benchmark.upper()}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Adjust layout
            plt.tight_layout()
            
            # Create benchmark subdirectory
            benchmark_dir = benchmark_output_path / benchmark
            benchmark_dir.mkdir(parents=True, exist_ok=True)
            
            # Save plot
            output_file = benchmark_dir / f'{metric}_vs_steps.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {output_file}")
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize all metrics vs training steps for multiple training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--snapshots',
        nargs='+',
        required=True,
        help='List of snapshot directory paths'
    )
    
    parser.add_argument(
        '--zero_train_step',
        type=str,
        default=None,
        help='Path to zero_train_step directory (for untrained baseline)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./plots/',
        help='Output directory for plots (default: ./plots/)'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='Filter specific metrics to plot (default: all metrics)'
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=None,
        help='Filter specific benchmarks to plot (default: all benchmarks)'
    )
    
    args = parser.parse_args()
    
    # Parse all snapshot directories
    print("Parsing snapshot directories...")
    snapshots_data = []
    all_metrics = set()
    for snapshot_dir in args.snapshots:
        print(f"  Parsing: {snapshot_dir}")
        training_dataset, validation_data, metrics = parse_snapshot_directory(snapshot_dir)
        if validation_data:
            snapshots_data.append((training_dataset, validation_data, metrics))
            all_metrics.update(metrics)
            print(f"    Training dataset: {training_dataset}")
            print(f"    Metrics found: {len(metrics)}")
        else:
            print(f"    Warning: No validation data found, skipping")
    
    # Parse zero_train_step if provided
    zero_train_step_data = None
    if args.zero_train_step:
        print(f"\nParsing zero_train_step directory: {args.zero_train_step}")
        _, zero_data, zero_metrics = parse_snapshot_directory(args.zero_train_step)
        if zero_data:
            zero_train_step_data = (zero_data, zero_metrics)
            all_metrics.update(zero_metrics)
            print(f"  Metrics found: {len(zero_metrics)}")
        else:
            print(f"  Warning: No validation data found in zero_train_step")
    
    # Organize data by metric
    print("\nOrganizing data by metric...")
    metric_data, metrics_list = organize_data_by_metric(snapshots_data, zero_train_step_data)
    
    print(f"Found {len(metrics_list)} metrics: {metrics_list}")
    for metric, benchmark_dict in metric_data.items():
        total_series = sum(len(data_list) for data_list in benchmark_dict.values())
        print(f"  {metric}: {total_series} data series across {len(benchmark_dict)} benchmark(s)")
    
    # Create plots
    print(f"\nCreating plots...")
    create_plots(metric_data, args.output_dir, args.metrics, args.benchmarks)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

