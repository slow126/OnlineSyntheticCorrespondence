#!/usr/bin/env python3
"""
Visualize PCK accuracy vs training steps for multiple training datasets.
Creates one plot per validation benchmark comparing different training datasets.
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
    Parse validation_results.csv to extract PCK vs training_steps data.
    
    Args:
        csv_path: Path to validation_results.csv file
        
    Returns:
        Dictionary mapping benchmark -> list of (training_steps, pck) tuples
    """
    if not os.path.exists(csv_path):
        return {}
    
    data = defaultdict(list)
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    benchmark = row['benchmark']
                    training_steps = int(row['training_steps'])
                    pck = float(row['pck'])
                    data[benchmark].append((training_steps, pck))
                except (KeyError, ValueError) as e:
                    print(f"Warning: Skipping row in {csv_path}: {e}")
                    continue
        
        # Sort by training_steps for each benchmark
        for benchmark in data:
            data[benchmark].sort(key=lambda x: x[0])
            
    except Exception as e:
        print(f"Warning: Could not parse {csv_path}: {e}")
        return {}
    
    return data


def parse_snapshot_directory(snapshot_dir):
    """
    Parse a snapshot directory to extract training dataset and validation results.
    
    Args:
        snapshot_dir: Path to snapshot directory
        
    Returns:
        Tuple of (training_dataset_name, validation_data_dict)
        Returns (None, {}) if parsing fails
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
    validation_data = parse_validation_results(csv_path)
    
    if not validation_data:
        print(f"Warning: No validation data found in {csv_path}")
    
    return training_dataset, validation_data


def organize_data_by_benchmark(snapshots_data, zero_train_step_data=None):
    """
    Organize data by validation benchmark.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict) tuples
        zero_train_step_data: Optional validation_data_dict from zero_train_step
        
    Returns:
        Dictionary mapping benchmark -> list of (training_dataset, steps_array, pck_array) tuples
    """
    benchmark_data = defaultdict(list)
    
    # Add zero_train_step baseline if provided
    if zero_train_step_data:
        for benchmark, data_points in zero_train_step_data.items():
            if data_points:
                steps = [point[0] for point in data_points]
                pcks = [point[1] for point in data_points]
                benchmark_data[benchmark].append(('zero_train_step', steps, pcks))
    
    # Add data from each snapshot
    for training_dataset, validation_data in snapshots_data:
        for benchmark, data_points in validation_data.items():
            if data_points:
                steps = [point[0] for point in data_points]
                pcks = [point[1] for point in data_points]
                benchmark_data[benchmark].append((training_dataset, steps, pcks))
    
    return benchmark_data


def create_plots(benchmark_data, output_dir, benchmarks_filter=None):
    """
    Create visualization plots for each benchmark.
    
    Args:
        benchmark_data: Dictionary mapping benchmark -> list of (dataset, steps, pcks) tuples
        output_dir: Output directory for plots
        benchmarks_filter: Optional list of benchmarks to plot (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Filter benchmarks if specified
    benchmarks_to_plot = benchmark_data.keys()
    if benchmarks_filter:
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]
    
    # Get distinct colors for each training dataset
    datasets = set()
    for benchmark_data_list in benchmark_data.values():
        for dataset, _, _ in benchmark_data_list:
            datasets.add(dataset)
    
    # Use a colormap for distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets)))
    color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(datasets))}
    
    # Create one plot per benchmark
    for benchmark in benchmarks_to_plot:
        if benchmark not in benchmark_data or not benchmark_data[benchmark]:
            print(f"Warning: No data for benchmark {benchmark}, skipping")
            continue
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each training dataset
        for training_dataset, steps, pcks in benchmark_data[benchmark]:
            # Use different line style for zero_train_step
            linestyle = '--' if training_dataset == 'zero_train_step' else '-'
            linewidth = 2 if training_dataset == 'zero_train_step' else 1.5
            label = training_dataset if training_dataset != 'zero_train_step' else 'Untrained (baseline)'
            
            ax.plot(steps, pcks, 
                   color=color_map.get(training_dataset, 'black'),
                   linestyle=linestyle,
                   linewidth=linewidth,
                   marker='o',
                   markersize=4,
                   label=label)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('PCK Accuracy (%)', fontsize=12)
        ax.set_title(f'PCK Accuracy vs Training Steps - {benchmark.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_file = output_path / f'pck_vs_steps_{benchmark}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize PCK accuracy vs training steps for multiple training datasets',
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
        default='.',
        help='Output directory for plots (default: current directory)'
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
    for snapshot_dir in args.snapshots:
        print(f"  Parsing: {snapshot_dir}")
        training_dataset, validation_data = parse_snapshot_directory(snapshot_dir)
        if validation_data:
            snapshots_data.append((training_dataset, validation_data))
            print(f"    Training dataset: {training_dataset}")
            print(f"    Benchmarks: {list(validation_data.keys())}")
        else:
            print(f"    Warning: No validation data found, skipping")
    
    # Parse zero_train_step if provided
    zero_train_step_data = None
    if args.zero_train_step:
        print(f"\nParsing zero_train_step directory: {args.zero_train_step}")
        _, zero_train_step_data = parse_snapshot_directory(args.zero_train_step)
        if zero_train_step_data:
            print(f"  Benchmarks: {list(zero_train_step_data.keys())}")
        else:
            print(f"  Warning: No validation data found in zero_train_step")
    
    # Organize data by benchmark
    print("\nOrganizing data by benchmark...")
    benchmark_data = organize_data_by_benchmark(snapshots_data, zero_train_step_data)
    
    print(f"Found data for benchmarks: {list(benchmark_data.keys())}")
    for benchmark, data_list in benchmark_data.items():
        print(f"  {benchmark}: {len(data_list)} training dataset(s)")
    
    # Create plots
    print(f"\nCreating plots...")
    create_plots(benchmark_data, args.output_dir, args.benchmarks)
    
    print("\nDone!")


if __name__ == '__main__':
    main()

