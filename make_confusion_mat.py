#!/usr/bin/env python3
"""
Create confusion matrix visualizations comparing training datasets vs evaluation benchmarks.
Creates four matrices: 
  - Best performance (absolute values)
  - Average performance (absolute values)
  - Best performance (column-standardized, mean 0 std 1 within each benchmark)
  - Average performance (column-standardized, mean 0 std 1 within each benchmark)
Orders rows and columns to align matching training datasets and benchmarks on the diagonal.
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Reuse parsing functions from plot_metrics.py
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


def parse_directory_name(directory_name):
    """
    Parse directory name to extract dataset and parameters.
    
    Directory names follow patterns like:
    - pointodyssey_stride1_sequence_length16_freezeTrue_eval...
    - flyingthings_freezeTrue
    - spair_freezeTrue
    - synthetic_large_angle_small_scale
    - synthetic_small_flow_centered
    
    Args:
        directory_name: Name of the directory
        
    Returns:
        Dictionary with 'dataset' and optional 'stride', 'sequence_length', 'freeze' keys
    """
    result = {}
    
    # Remove eval suffix if present (everything from 'eval' onwards)
    name_to_parse = directory_name
    eval_match = re.search(r'_eval', name_to_parse)
    if eval_match:
        name_to_parse = name_to_parse[:eval_match.start()]
    
    # Extract freeze (freezeTrue or freezeFalse) - do this first as it affects dataset name
    freeze_match = re.search(r'freeze(True|False)', name_to_parse)
    freeze_val = None
    if freeze_match:
        freeze_val = freeze_match.group(1)
        result['freeze'] = 'T' if freeze_val == 'True' else 'F'
        # Remove freeze part from name for further parsing
        name_to_parse = re.sub(r'_freeze(True|False)', '', name_to_parse)
    
    # Extract stride (stride{value})
    stride_match = re.search(r'stride(\d+)', name_to_parse)
    if stride_match:
        result['stride'] = stride_match.group(1)
    
    # Extract sequence_length (sequence_length{value})
    seq_match = re.search(r'sequence_length(\d+)', name_to_parse)
    if seq_match:
        result['sequence_length'] = seq_match.group(1)
    
    # Extract dataset name
    # For synthetic variants, preserve the full variant name
    # For others, take the first part before underscore
    parts = name_to_parse.split('_')
    if parts:
        if parts[0] == 'synthetic':
            # For synthetic, preserve the full variant name (e.g., "synthetic_large_angle_small_scale")
            # Remove already-extracted parameters
            dataset_parts = [parts[0]]  # Start with "synthetic"
            i = 1
            while i < len(parts):
                # Skip parameters we've already extracted
                if re.match(r'stride\d+', parts[i]) or re.match(r'sequence_length\d+', parts[i]):
                    i += 1
                    continue
                # Add remaining parts as variant description
                dataset_parts.append(parts[i])
                i += 1
            result['dataset'] = '_'.join(dataset_parts)
        else:
            # For other datasets, just take the first part
            result['dataset'] = parts[0]
    
    return result if result else None


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


def parse_average_performance_from_csv(csv_path, metric='pck'):
    """
    Parse validation_results.csv to compute average performance per benchmark for specified metric.
    
    Args:
        csv_path: Path to validation_results.csv file
        metric: Metric name to extract (default: 'pck')
        
    Returns:
        Dictionary mapping benchmark -> average_metric_value (float)
    """
    if not os.path.exists(csv_path):
        return {}
    
    benchmark_data = defaultdict(list)
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                benchmark = row.get('benchmark', '').strip()
                metric_str = row.get(metric, '').strip()
                
                if not benchmark or not metric_str:
                    continue
                
                try:
                    metric_value = float(metric_str)
                    benchmark_data[benchmark].append(metric_value)
                except ValueError:
                    continue
        
        # Compute averages
        result = {}
        for benchmark, values in benchmark_data.items():
            if values:
                result[benchmark] = np.mean(values)
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not parse average performance from {csv_path}: {e}")
        return {}


def parse_snapshot_for_confusion_matrix(snapshot_dir):
    """
    Parse a snapshot directory to extract training dataset label and performance data.
    
    Args:
        snapshot_dir: Path to snapshot directory
        
    Returns:
        Dictionary with:
        - 'training_label': formatted training dataset label
        - 'best_performance': dict mapping benchmark -> best_pck
        - 'avg_performance': dict mapping benchmark -> avg_pck
        Returns None if parsing fails
    """
    snapshot_path = Path(snapshot_dir)
    
    # Parse training summary for dataset info
    summary_path = snapshot_path / 'training_summary.txt'
    summary_info = parse_training_summary(summary_path)
    
    # Parse directory name - this is important for synthetic variants
    dir_info = parse_directory_name(snapshot_path.name)
    
    # Merge: prefer directory name for dataset (preserves synthetic variants)
    # but use summary for other parameters if missing
    if dir_info:
        if summary_info:
            # If directory has dataset info, prefer it (especially for synthetic variants)
            if 'dataset' in dir_info:
                summary_info['dataset'] = dir_info['dataset']
            # Fill in missing parameters from directory
            if 'stride' not in summary_info and 'stride' in dir_info:
                summary_info['stride'] = dir_info['stride']
            if 'sequence_length' not in summary_info and 'sequence_length' in dir_info:
                summary_info['sequence_length'] = dir_info['sequence_length']
            if 'freeze' not in summary_info and 'freeze' in dir_info:
                summary_info['freeze'] = dir_info['freeze']
        else:
            # No summary file, use directory info
            summary_info = dir_info
    elif not summary_info:
        # Fallback to directory name if parsing fails
        summary_info = {'dataset': snapshot_path.name}
        print(f"Warning: Could not parse training dataset from {summary_path} or directory name, using: {snapshot_path.name}")
    
    # Format training label
    training_label = format_training_dataset_label(summary_info)
    
    # Get best performance from summary
    best_performance = parse_best_performance_from_summary(summary_path)
    
    # Get average performance from CSV (will be updated with metric in main)
    # For now, we'll parse it later with the specific metric
    
    return {
        'training_label': training_label,
        'best_performance': best_performance,
        'snapshot_path': snapshot_path  # Store path for later CSV parsing with metric
    }


def extract_base_dataset_name(training_label):
    """
    Extract base dataset name from training label.
    
    Examples:
        "pointodyssey_stride1_S16_freezeF" -> "pointodyssey"
        "PtOd_stride1_S16" -> "pointodyssey"
        "pfpascal" -> "pfpascal"
        "synthetic" -> "synthetic"
    
    Args:
        training_label: Formatted training dataset label
        
    Returns:
        Base dataset name (normalized to match benchmark names)
    """
    # Remove parameters (stride, S, freeze, etc.)
    # Split by underscore and take first part
    base = training_label.split('_')[0]
    
    # Normalize abbreviations
    if base.lower() in ['ptod', 'ptodyssey']:
        return 'pointodyssey'
    
    return base.lower()


def order_for_diagonal_alignment(training_labels, benchmark_labels):
    """
    Order training labels and benchmark labels so matching ones align on diagonal.
    
    Strategy:
    1. Group training datasets by base dataset name
    2. For each benchmark, find matching training dataset groups
    3. Order: matched pairs first (training variants grouped with their benchmark)
    4. Then unmatched training datasets
    5. Then unmatched benchmarks
    
    Args:
        training_labels: List of training dataset labels
        benchmark_labels: List of benchmark names
        
    Returns:
        Tuple of (ordered_training_labels, ordered_benchmark_labels)
    """
    # Extract base names for matching
    training_base_map = {label: extract_base_dataset_name(label) for label in training_labels}
    
    # Group training labels by base dataset name
    training_groups = defaultdict(list)
    for label in training_labels:
        base = training_base_map[label]
        training_groups[base].append(label)
    
    # Sort each group for consistent ordering
    for base in training_groups:
        training_groups[base].sort()
    
    # Find matches: benchmarks that have corresponding training datasets
    matched_training = []
    matched_benchmarks = []
    unmatched_training = []
    unmatched_benchmarks = []
    
    # Track which benchmarks and training bases have been matched
    matched_benchmark_set = set()
    matched_training_base_set = set()
    
    # First pass: find exact matches
    for benchmark in benchmark_labels:
        benchmark_lower = benchmark.lower()
        
        # Find training dataset groups that match this benchmark
        if benchmark_lower in training_groups:
            # Add all training variants for this benchmark
            matched_training.extend(training_groups[benchmark_lower])
            matched_benchmarks.append(benchmark)
            matched_benchmark_set.add(benchmark)
            matched_training_base_set.add(benchmark_lower)
    
    # Collect unmatched training datasets
    for label in training_labels:
        base = training_base_map[label]
        if base not in matched_training_base_set:
            unmatched_training.append(label)
    
    # Collect unmatched benchmarks
    for benchmark in benchmark_labels:
        if benchmark not in matched_benchmark_set:
            unmatched_benchmarks.append(benchmark)
    
    # Combine: matched pairs first, then unmatched
    # For matched pairs, we want training variants grouped together
    # and aligned with their benchmark
    ordered_training = matched_training + sorted(unmatched_training)
    ordered_benchmarks = matched_benchmarks + sorted(unmatched_benchmarks)
    
    return ordered_training, ordered_benchmarks


def collect_snapshots(snapshots_dir):
    """
    Collect all snapshot directories from the given directory.
    
    Args:
        snapshots_dir: Directory containing snapshot subdirectories
        
    Returns:
        List of snapshot directory paths
    """
    snapshots_path = Path(snapshots_dir).expanduser()
    if not snapshots_path.exists():
        print(f"Warning: Snapshots directory does not exist: {snapshots_dir}")
        return []
    
    snapshot_dirs = []
    for subdir in sorted(snapshots_path.iterdir()):
        if subdir.is_dir():
            has_csv = (subdir / 'validation_results.csv').exists()
            has_summary = (subdir / 'training_summary.txt').exists()
            
            if has_csv or has_summary:
                snapshot_dirs.append(str(subdir))
    
    return snapshot_dirs


def build_confusion_matrix_data(snapshots_data, metric='pck', use_best=True):
    """
    Build confusion matrix data structure with diagonal alignment.
    
    Args:
        snapshots_data: List of parsed snapshot data dictionaries
        metric: Which metric to use ('pck', 'pck_motion_aware', etc.)
        use_best: If True, use best performance; if False, use average
        
    Returns:
        Dictionary mapping:
        - 'matrix': 2D numpy array (training_datasets x benchmarks)
        - 'training_labels': list of training dataset labels (rows)
        - 'benchmark_labels': list of benchmark names (columns)
    """
    # Collect all training labels and benchmarks
    all_training_labels = set()
    all_benchmarks = set()
    
    # Build data dictionary: (training_label, benchmark) -> value
    data_dict = {}
    
    for snapshot in snapshots_data:
        training_label = snapshot['training_label']
        all_training_labels.add(training_label)
        
        if use_best:
            performance_dict = snapshot['best_performance']
        else:
            # Parse average performance from CSV with the specified metric
            csv_path = snapshot['snapshot_path'] / 'validation_results.csv'
            performance_dict = parse_average_performance_from_csv(csv_path, metric=metric)
        
        for benchmark, value in performance_dict.items():
            all_benchmarks.add(benchmark)
            data_dict[(training_label, benchmark)] = value
    
    # Order for diagonal alignment
    training_labels_list = sorted(list(all_training_labels))
    benchmark_labels_list = sorted(list(all_benchmarks))
    ordered_training, ordered_benchmarks = order_for_diagonal_alignment(
        training_labels_list, benchmark_labels_list
    )
    
    # Build matrix
    matrix = np.full((len(ordered_training), len(ordered_benchmarks)), np.nan)
    
    for i, training_label in enumerate(ordered_training):
        for j, benchmark in enumerate(ordered_benchmarks):
            key = (training_label, benchmark)
            if key in data_dict:
                matrix[i, j] = data_dict[key]
    
    return {
        'matrix': matrix,
        'training_labels': ordered_training,
        'benchmark_labels': ordered_benchmarks
    }


def standardize_matrix_columns(matrix):
    """
    Standardize each column (benchmark) independently to have mean 0 and std 1.
    For each column, transforms values using z-score standardization:
    - mean -> 0.0
    - std -> 1.0
    - Formula: (x - mean) / std
    
    Handles NaN values by ignoring them in mean/std calculation.
    
    Args:
        matrix: 2D numpy array (training_datasets x benchmarks)
        
    Returns:
        Standardized matrix with same shape
    """
    standardized = matrix.copy()
    
    for j in range(matrix.shape[1]):
        column = matrix[:, j]
        # Get valid (non-NaN) values
        valid_mask = ~np.isnan(column)
        if not np.any(valid_mask):
            # All NaN, leave as is
            continue
        
        valid_values = column[valid_mask]
        col_mean = np.mean(valid_values)
        col_std = np.std(valid_values)
        
        # Handle case where std is 0 (all values are the same)
        if col_std == 0:
            # Set all valid values to 0 (mean-centered)
            standardized[valid_mask, j] = 0.0
        else:
            # Z-score standardization: (x - mean) / std
            standardized[valid_mask, j] = (valid_values - col_mean) / col_std
    
    return standardized


def save_matrix_data(matrix_data, output_path, metric='pck', use_best=True):
    """
    Save confusion matrix data to a JSON file (with numpy arrays saved separately).
    
    Args:
        matrix_data: Dictionary from build_confusion_matrix_data
        output_path: Path to save the data (will create .json and .npy files)
        metric: Metric name
        use_best: Whether this is best or average performance
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save numpy array separately
    npy_path = output_path.with_suffix('.npy')
    np.save(npy_path, matrix_data['matrix'])
    
    # Save metadata as JSON
    json_data = {
        'metric': metric,
        'use_best': use_best,
        'training_labels': matrix_data['training_labels'],
        'benchmark_labels': matrix_data['benchmark_labels'],
        'matrix_shape': list(matrix_data['matrix'].shape),
        'matrix_file': npy_path.name
    }
    
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Saved matrix data: {json_path} (matrix: {npy_path})")


def load_matrix_data(input_path):
    """
    Load confusion matrix data from JSON and numpy files.
    
    Args:
        input_path: Path to the JSON file (or .json/.npy base path)
        
    Returns:
        Dictionary with 'matrix', 'training_labels', 'benchmark_labels', 'metric', 'use_best'
    """
    input_path = Path(input_path)
    
    # If .npy was provided, find the .json
    if input_path.suffix == '.npy':
        json_path = input_path.with_suffix('.json')
    else:
        json_path = input_path.with_suffix('.json')
        npy_path = input_path.with_suffix('.npy')
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if not npy_path.exists():
        raise FileNotFoundError(f"NumPy file not found: {npy_path}")
    
    # Load JSON metadata
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Load numpy matrix
    matrix = np.load(npy_path)
    
    return {
        'matrix': matrix,
        'training_labels': json_data['training_labels'],
        'benchmark_labels': json_data['benchmark_labels'],
        'metric': json_data.get('metric', 'pck'),
        'use_best': json_data.get('use_best', True)
    }


def create_confusion_matrix_plot(matrix_data, output_path, title_suffix="", metric='pck', standardized=False):
    """
    Create and save a confusion matrix heatmap visualization.
    
    Args:
        matrix_data: Dictionary from build_confusion_matrix_data
        output_path: Path to save the plot
        title_suffix: Additional text for title (e.g., "Best Performance")
        metric: Metric name for title
        standardized: If True, matrix is standardized (mean 0, std 1) and colorbar label will reflect this
    """
    matrix = matrix_data['matrix']
    training_labels = matrix_data['training_labels']
    benchmark_labels = matrix_data['benchmark_labels']
    
    fig, ax = plt.subplots(figsize=(max(12, len(benchmark_labels) * 1.2), 
                                    max(8, len(training_labels) * 0.5)))
    
    # Create heatmap
    # Use a colormap that works well for performance metrics (higher is better)
    # RdYlGn: Red (low/bad) -> Yellow (medium) -> Green (high/good)
    cbar_label = 'Standardized Score (mean=0, std=1)' if standardized else f'{metric.upper()} (%)'
    fmt_str = '.3f' if standardized else '.2f'
    
    sns.heatmap(matrix, 
                annot=True, 
                fmt=fmt_str,
                cmap='RdYlGn',  # Red-Yellow-Green: green for high (good), red for low (bad)
                cbar_kws={'label': cbar_label},
                xticklabels=benchmark_labels,
                yticklabels=training_labels,
                ax=ax,
                linewidths=0.5,
                linecolor='gray',
                mask=np.isnan(matrix),
                vmin=None,
                vmax=None)
    
    # Highlight diagonal cells where training dataset base matches benchmark
    # Only highlight the first matching training variant for each benchmark
    training_base_map = {label: extract_base_dataset_name(label) for label in training_labels}
    benchmark_positions = {benchmark.lower(): j for j, benchmark in enumerate(benchmark_labels)}
    
    # Track which benchmarks have been highlighted
    highlighted_benchmarks = set()
    
    for i, training_label in enumerate(training_labels):
        base = training_base_map[training_label]
        if base in benchmark_positions:
            j = benchmark_positions[base]
            # Only highlight the first training variant for each benchmark
            if base not in highlighted_benchmarks and not np.isnan(matrix[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                         edgecolor='blue', lw=2, zorder=10))
                highlighted_benchmarks.add(base)
    
    metric_display = metric.replace('_', ' ').title()
    std_suffix = " (Column-Standardized)" if standardized else ""
    ax.set_title(f'{metric_display} Confusion Matrix - {title_suffix}{std_suffix}', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Evaluation Benchmark', fontsize=12)
    ax.set_ylabel('Training Dataset', fontsize=12)
    
    # Rotate labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create confusion matrix visualizations for training datasets vs evaluation benchmarks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python make_confusion_mat.py --snapshots_dir snapshots/ --output_dir ./confusion_matrices/
  python make_confusion_mat.py --snapshots_dir snapshots/ --metric pck_motion_aware --benchmarks tss pointodyssey
        """
    )
    
    parser.add_argument(
        '--snapshots_dir',
        type=str,
        required=True,
        help='Directory containing snapshot subdirectories'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./confusion_matrices/',
        help='Output directory for plots (default: ./confusion_matrices/)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='pck',
        help='Metric to visualize (default: pck). Options: pck, pck_motion_aware, loss, etc.'
    )
    
    parser.add_argument(
        '--benchmarks',
        nargs='+',
        default=None,
        help='Filter specific benchmarks to include (default: all benchmarks)'
    )
    
    parser.add_argument(
        '--save_data',
        action='store_true',
        help='Save matrix data to JSON/numpy files for faster reloading'
    )
    
    parser.add_argument(
        '--load_data',
        type=str,
        default=None,
        help='Load matrix data from saved files (provide base path, e.g., ./data/pck_best)'
    )
    
    args = parser.parse_args()
    
    # 1. Collect all snapshot directories
    print(f"Scanning snapshots directory: {args.snapshots_dir}")
    snapshot_dirs = collect_snapshots(args.snapshots_dir)
    
    if not snapshot_dirs:
        print("Error: No snapshot directories found!")
        return
    
    print(f"Found {len(snapshot_dirs)} snapshot directory(ies)")
    
    # 2. Parse each snapshot
    print("Parsing snapshot directories...")
    snapshots_data = []
    for snapshot_dir in snapshot_dirs:
        print(f"  Parsing: {Path(snapshot_dir).name}")
        data = parse_snapshot_for_confusion_matrix(snapshot_dir)
        if data and (data['best_performance'] or data.get('snapshot_path')):
            snapshots_data.append(data)
            print(f"    Training dataset: {data['training_label']}")
        else:
            print(f"    Warning: No performance data found, skipping")
    
    if not snapshots_data:
        print("Error: No valid snapshot data found!")
        return
    
    # 3. Build confusion matrix for best performance
    print(f"\nBuilding confusion matrices for metric: {args.metric}...")
    
    # Try loading saved data if requested
    if args.load_data:
        try:
            print(f"Loading saved data from: {args.load_data}")
            best_matrix_data = load_matrix_data(f"{args.load_data}_best")
            avg_matrix_data = load_matrix_data(f"{args.load_data}_average")
            print("Successfully loaded saved data!")
        except FileNotFoundError as e:
            print(f"Warning: Could not load saved data: {e}")
            print("Computing matrices from scratch...")
            best_matrix_data = build_confusion_matrix_data(
                snapshots_data, 
                metric=args.metric, 
                use_best=True
            )
            avg_matrix_data = build_confusion_matrix_data(
                snapshots_data, 
                metric=args.metric, 
                use_best=False
            )
    else:
        best_matrix_data = build_confusion_matrix_data(
            snapshots_data, 
            metric=args.metric, 
            use_best=True
        )
        
        # 4. Build confusion matrix for average performance
        avg_matrix_data = build_confusion_matrix_data(
            snapshots_data, 
            metric=args.metric, 
            use_best=False
        )
    
    # Filter benchmarks if specified
    if args.benchmarks:
        # Filter both matrices
        for matrix_data in [best_matrix_data, avg_matrix_data]:
            benchmark_indices = [i for i, b in enumerate(matrix_data['benchmark_labels']) 
                               if b in args.benchmarks]
            if benchmark_indices:
                matrix_data['matrix'] = matrix_data['matrix'][:, benchmark_indices]
                matrix_data['benchmark_labels'] = [matrix_data['benchmark_labels'][i] 
                                                  for i in benchmark_indices]
    
    # 5. Create standardized versions
    best_matrix_data_standardized = {
        'matrix': standardize_matrix_columns(best_matrix_data['matrix']),
        'training_labels': best_matrix_data['training_labels'],
        'benchmark_labels': best_matrix_data['benchmark_labels']
    }
    
    avg_matrix_data_standardized = {
        'matrix': standardize_matrix_columns(avg_matrix_data['matrix']),
        'training_labels': avg_matrix_data['training_labels'],
        'benchmark_labels': avg_matrix_data['benchmark_labels']
    }
    
    # 6. Save data if requested
    if args.save_data:
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving matrix data...")
        save_matrix_data(
            best_matrix_data,
            output_path / f'{args.metric}_best_matrix_data',
            metric=args.metric,
            use_best=True
        )
        save_matrix_data(
            avg_matrix_data,
            output_path / f'{args.metric}_average_matrix_data',
            metric=args.metric,
            use_best=False
        )
    
    # 7. Create and save plots
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Original plots (absolute values)
    create_confusion_matrix_plot(
        best_matrix_data, 
        output_path / f'{args.metric}_best_confusion_matrix.png',
        title_suffix="Best Performance",
        metric=args.metric,
        standardized=False
    )
    
    create_confusion_matrix_plot(
        avg_matrix_data, 
        output_path / f'{args.metric}_average_confusion_matrix.png',
        title_suffix="Average Performance",
        metric=args.metric,
        standardized=False
    )
    
    # Standardized plots (mean 0, std 1 within each benchmark)
    create_confusion_matrix_plot(
        best_matrix_data_standardized, 
        output_path / f'{args.metric}_best_confusion_matrix_standardized.png',
        title_suffix="Best Performance",
        metric=args.metric,
        standardized=True
    )
    
    create_confusion_matrix_plot(
        avg_matrix_data_standardized, 
        output_path / f'{args.metric}_average_confusion_matrix_standardized.png',
        title_suffix="Average Performance",
        metric=args.metric,
        standardized=True
    )
    
    print("\nDone!")


if __name__ == '__main__':
    main()

