#!/usr/bin/env python3
"""
Visualize all metrics vs training steps for multiple training datasets.
Creates one plot per metric comparing different training datasets/benchmarks.
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def parse_training_summary(summary_path):
    """
    Parse training_summary.txt to extract training dataset name and parameters.
    
    Args:
        summary_path: Path to training_summary.txt file
        
    Returns:
        Dictionary with 'dataset' and optional 'stride', 'sequence_length', 'freeze' keys
        Returns None if parsing fails
    """
    if not os.path.exists(summary_path):
        return None
    
    result = {}
    
    try:
        with open(summary_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip STATUS lines to avoid matching "S:" in "STATUS:"
                if 'STATUS:' in line:
                    continue
                # Extract dataset name
                if 'Train dataset:' in line:
                    dataset = line.split('Train dataset:')[1].strip()
                    result['dataset'] = dataset
                # Extract stride (for PointOdyssey)
                elif 'strides_pointodyssey:' in line or 'Strides:' in line:
                    # Handle both formats
                    if 'strides_pointodyssey:' in line:
                        stride_str = line.split('strides_pointodyssey:')[1].strip()
                    else:
                        stride_str = line.split('Strides:')[1].strip()
                    # Parse list format like "[1, 2, 4]" or single value
                    try:
                        # Remove brackets and split
                        stride_str = stride_str.strip('[]')
                        strides = [int(s.strip()) for s in stride_str.split(',') if s.strip()]
                        if strides:
                            # Use first stride or join if multiple
                            result['stride'] = '_'.join(map(str, strides)) if len(strides) > 1 else str(strides[0])
                    except:
                        result['stride'] = stride_str
                # Extract sequence length (S)
                # Be careful not to match "STATUS:" - only match standalone "S:" or explicit sequence length
                elif 'sequence_length_pointodyssey:' in line or 'Sequence length:' in line:
                    if 'sequence_length_pointodyssey:' in line:
                        seq_len = line.split('sequence_length_pointodyssey:')[1].strip()
                    else:
                        seq_len = line.split('Sequence length:')[1].strip()
                    # Extract just the number (handle cases like "4" or "4, Training in progress")
                    try:
                        # Take first token (should be the number)
                        seq_len = seq_len.split()[0].strip()
                        # Remove any trailing commas or other punctuation
                        seq_len = seq_len.rstrip(',')
                        result['sequence_length'] = seq_len
                    except:
                        # If parsing fails, try to extract number with regex-like approach
                        import re
                        match = re.search(r'\d+', seq_len)
                        if match:
                            result['sequence_length'] = match.group()
                        else:
                            result['sequence_length'] = seq_len.strip()
                # Match standalone "S:" (but not "STATUS:" - already filtered above)
                elif line.startswith('S:') or ' S:' in line:
                    # Extract value after "S:"
                    if line.startswith('S:'):
                        seq_len = line.split('S:', 1)[1].strip()
                    else:
                        seq_len = line.split(' S:', 1)[1].strip()
                    # Extract just the number
                    try:
                        seq_len = seq_len.split()[0].strip().rstrip(',')
                        result['sequence_length'] = seq_len
                    except:
                        import re
                        match = re.search(r'\d+', seq_len)
                        if match:
                            result['sequence_length'] = match.group()
                # Extract freeze
                elif 'freeze:' in line or 'Freeze:' in line:
                    if 'freeze:' in line:
                        freeze_str = line.split('freeze:')[1].strip()
                    else:
                        freeze_str = line.split('Freeze:')[1].strip()
                    # Normalize to True/False
                    freeze_str = freeze_str.lower()
                    if freeze_str in ['true', '1', 'yes']:
                        result['freeze'] = 'T'
                    elif freeze_str in ['false', '0', 'no']:
                        result['freeze'] = 'F'
                    else:
                        result['freeze'] = freeze_str
    except Exception as e:
        print(f"Warning: Could not parse {summary_path}: {e}")
        return None
    
    return result if result else None


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


def format_training_dataset_label(summary_info):
    """
    Format training dataset label with parameters.
    
    Args:
        summary_info: Dictionary from parse_training_summary or string dataset name
        
    Returns:
        Formatted label string
    """
    if summary_info is None:
        return "Unknown"
    
    # Handle legacy case where summary_info is just a string
    if isinstance(summary_info, str):
        dataset = summary_info
        summary_info = {'dataset': dataset}
    
    dataset = summary_info.get('dataset', 'Unknown')
    
    # Handle mixed datasets: convert "+" to "_" for display consistency
    # e.g., "spair+synthetic" -> "spair_synthetic"
    if '+' in dataset:
        dataset = dataset.replace('+', '_')
    
    # Abbreviate PointOdyssey to PtOd
    if 'pointodyssey' in dataset.lower():
        dataset = dataset.replace('pointodyssey', 'PtOd').replace('PointOdyssey', 'PtOd')
    
    # Build label with parameters
    parts = [dataset]
    
    # Add stride if present
    if 'stride' in summary_info:
        parts.append(f"stride{summary_info['stride']}")
    
    # Add sequence length (S) if present
    if 'sequence_length' in summary_info:
        parts.append(f"S{summary_info['sequence_length']}")
    
    # Add freeze if present
    if 'freeze' in summary_info:
        parts.append(f"freeze{summary_info['freeze']}")
    
    return '_'.join(parts)


def parse_directory_name(directory_name):
    """
    Parse directory name to extract dataset and parameters.
    
    Directory names follow pattern like:
    pointodyssey_stride1_sequence_length16_freezeFalse_eval...
    spair_synthetic_70_30_pretrainedTrue_freezeFalse_eval...
    spair_synthetic_pretrainedTrue_freezeFalse_eval...
    
    Args:
        directory_name: Name of the directory
        
    Returns:
        Dictionary with 'dataset' and optional 'stride', 'sequence_length', 'freeze' keys
    """
    import re
    result = {}
    
    # Known parameter keywords that indicate end of dataset name
    param_keywords = ['stride', 'sequence_length', 'freeze', 'pretrained', 'eval']
    
    # Check for mixed dataset pattern: dataset1_dataset2 or dataset1_dataset2_X_Y
    # Pattern 1: dataset1_dataset2_X_Y (with percentages)
    mixed_with_percent_match = re.match(r'^([a-zA-Z]+)_([a-zA-Z]+)_(\d+)_(\d+)(?:_|$)', directory_name)
    if mixed_with_percent_match:
        dataset1 = mixed_with_percent_match.group(1)
        dataset2 = mixed_with_percent_match.group(2)
        percent1 = mixed_with_percent_match.group(3)
        percent2 = mixed_with_percent_match.group(4)
        result['dataset'] = f"{dataset1}_{dataset2}_{percent1}_{percent2}"
    else:
        # Pattern 2: dataset1_dataset2 (without percentages, assumed 50/50)
        # Check if we have two words before hitting a parameter keyword
        parts = directory_name.split('_')
        if len(parts) >= 2:
            # Check if first two parts look like dataset names (not numbers, not parameter keywords)
            part1 = parts[0].lower()
            part2 = parts[1].lower()
            
            # Check if part2 is a parameter keyword or number
            is_part2_param = (part2 in param_keywords or 
                            part2.startswith('stride') or 
                            part2.startswith('sequence') or
                            part2.startswith('freeze') or
                            part2.startswith('pretrained') or
                            part2.isdigit())
            
            if not is_part2_param:
                # Likely a mixed dataset: dataset1_dataset2
                result['dataset'] = f"{parts[0]}_{parts[1]}"
            else:
                # Single dataset
                result['dataset'] = parts[0]
        else:
            # Single word dataset name
            result['dataset'] = parts[0] if parts else directory_name
    
    # Extract stride (stride{value})
    stride_match = re.search(r'stride(\d+)', directory_name)
    if stride_match:
        result['stride'] = stride_match.group(1)
    
    # Extract sequence_length (sequence_length{value})
    seq_match = re.search(r'sequence_length(\d+)', directory_name)
    if seq_match:
        result['sequence_length'] = seq_match.group(1)
    
    # Extract freeze (freezeTrue or freezeFalse)
    freeze_match = re.search(r'freeze(True|False)', directory_name)
    if freeze_match:
        freeze_val = freeze_match.group(1)
        result['freeze'] = 'T' if freeze_val == 'True' else 'F'
    
    return result if result else None


def parse_snapshot_directory(snapshot_dir):
    """
    Parse a snapshot directory to extract training dataset and validation results.
    
    Args:
        snapshot_dir: Path to snapshot directory
        
    Returns:
        Tuple of (training_dataset_label, validation_data_dict, metrics_list)
        Returns (None, {}, []) if parsing fails
    """
    snapshot_path = Path(snapshot_dir)
    
    # Get training dataset info from training_summary.txt
    summary_path = snapshot_path / 'training_summary.txt'
    summary_info = parse_training_summary(summary_path)
    
    # If summary doesn't have parameters, try parsing from directory name
    if summary_info:
        # Check if we're missing parameters that might be in directory name
        dir_info = parse_directory_name(snapshot_path.name)
        if dir_info:
            # Merge: use summary for dataset name, but fill in missing params from directory
            if 'stride' not in summary_info and 'stride' in dir_info:
                summary_info['stride'] = dir_info['stride']
            if 'sequence_length' not in summary_info and 'sequence_length' in dir_info:
                summary_info['sequence_length'] = dir_info['sequence_length']
            if 'freeze' not in summary_info and 'freeze' in dir_info:
                summary_info['freeze'] = dir_info['freeze']
    else:
        # No summary file, try parsing from directory name
        summary_info = parse_directory_name(snapshot_path.name)
        if not summary_info:
            # Fallback to directory name if parsing fails
            summary_info = {'dataset': snapshot_path.name}
            print(f"Warning: Could not parse training dataset from {summary_path} or directory name, using: {snapshot_path.name}")
    
    # Format label
    training_dataset = format_training_dataset_label(summary_info)
    
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


def collect_snapshot_directories(args):
    """
    Collect snapshot directories from various sources:
    1. --snapshots argument (explicit list)
    2. --snapshots_dir argument (directory containing snapshots)
    3. stdin (piped from ls or similar)
    
    Returns:
        List of snapshot directory paths
    """
    snapshot_dirs = []
    
    # Method 1: Explicit list via --snapshots
    if args.snapshots:
        snapshot_dirs.extend(args.snapshots)
    
    # Method 2: Directory containing snapshots
    if args.snapshots_dir:
        snapshots_dir_path = Path(args.snapshots_dir).expanduser()  # Expand ~ to home directory
        if not snapshots_dir_path.exists():
            print(f"Warning: Snapshots directory does not exist: {args.snapshots_dir}")
        else:
            print(f"Scanning directory: {snapshots_dir_path}")
            found_count = 0
            
            def is_snapshot_directory(path):
                """Check if a directory looks like a snapshot directory."""
                has_csv = (path / 'validation_results.csv').exists()
                has_summary = (path / 'training_summary.txt').exists()
                return has_csv or has_summary
            
            def find_snapshots_recursive(root_path, max_depth=3, current_depth=0):
                """Recursively find snapshot directories."""
                found = []
                if current_depth >= max_depth:
                    return found
                
                try:
                    for subdir in sorted(root_path.iterdir()):
                        if subdir.is_dir():
                            # Check if this directory is a snapshot
                            if is_snapshot_directory(subdir):
                                found.append(str(subdir))
                                print(f"  Found snapshot: {subdir} (depth={current_depth})")
                            # Also check for zero_train_step
                            elif subdir.name == 'zero_train_step' and not args.zero_train_step:
                                if (subdir / 'validation_results.csv').exists():
                                    found.append(str(subdir))
                                    print(f"  Found zero_train_step: {subdir}")
                            else:
                                # Recursively search deeper
                                found.extend(find_snapshots_recursive(subdir, max_depth, current_depth + 1))
                except PermissionError:
                    pass  # Skip directories we can't access
                
                return found
            
            # Search recursively
            found_snapshots = find_snapshots_recursive(snapshots_dir_path)
            snapshot_dirs.extend(found_snapshots)
            found_count = len(found_snapshots)
            
            if found_count == 0:
                print(f"  No snapshot directories found in {snapshots_dir_path}")
                print(f"  Looking for subdirectories containing 'validation_results.csv' or 'training_summary.txt'")
            else:
                print(f"  Found {found_count} snapshot directory(ies)")
    
    # Method 3: Read from stdin (for piping)
    if not sys.stdin.isatty():  # Check if stdin is not a terminal (i.e., piped)
        for line in sys.stdin:
            line = line.strip()
            if line:
                path = Path(line)
                if path.exists() and path.is_dir():
                    snapshot_dirs.append(line)
                else:
                    # Try as relative path from current directory
                    rel_path = Path(line)
                    if rel_path.exists() and rel_path.is_dir():
                        snapshot_dirs.append(line)
                    else:
                        print(f"Warning: Skipping invalid path from stdin: {line}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_dirs = []
    for dir_path in snapshot_dirs:
        if dir_path not in seen:
            seen.add(dir_path)
            unique_dirs.append(dir_path)
    
    return unique_dirs


def main():
    parser = argparse.ArgumentParser(
        description='Visualize all metrics vs training steps for multiple training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explicit list of snapshots
  python plot_metrics.py --snapshots snapshots/exp1 snapshots/exp2
  
  # Directory containing snapshots
  python plot_metrics.py --snapshots_dir snapshots/
  
  # Pipe from ls
  ls -d snapshots/*/ | python plot_metrics.py
  
  # Combine methods
  python plot_metrics.py --snapshots_dir snapshots/ --snapshots snapshots/special_exp
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
        '--zero_train_step',
        type=str,
        default=None,
        help='Path to zero_train_step directory (for untrained baseline). If not specified and found in snapshots_dir, will auto-detect.'
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
    
    # Collect snapshot directories from all sources
    snapshot_dirs = collect_snapshot_directories(args)
    
    if not snapshot_dirs:
        print("Error: No snapshot directories found!")
        print("Please provide snapshots via:")
        print("  --snapshots <dir1> <dir2> ...")
        print("  --snapshots_dir <parent_dir>")
        print("  Or pipe directory paths via stdin")
        sys.exit(1)
    
    # Separate zero_train_step if it's in the list
    zero_train_step_dir = args.zero_train_step
    if not zero_train_step_dir:
        # Check if zero_train_step is in the collected directories
        for dir_path in snapshot_dirs:
            if Path(dir_path).name == 'zero_train_step':
                zero_train_step_dir = dir_path
                snapshot_dirs.remove(dir_path)
                print(f"Auto-detected zero_train_step: {zero_train_step_dir}")
                break
    
    print(f"Found {len(snapshot_dirs)} snapshot directory(ies)")
    
    # Parse all snapshot directories
    print("Parsing snapshot directories...")
    snapshots_data = []
    all_metrics = set()
    for snapshot_dir in snapshot_dirs:
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
    if zero_train_step_dir:
        print(f"\nParsing zero_train_step directory: {zero_train_step_dir}")
        _, zero_data, zero_metrics = parse_snapshot_directory(zero_train_step_dir)
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

