#!/usr/bin/env python3
"""
Plot correlation between PCK performance and Wasserstein-1 distance for magnitude histogram.
X-axis: Wasserstein-1 distance (between training dataset and evaluation benchmark fingerprints)
Y-axis: PCK performance (from confusion matrix)
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import functions from existing scripts
from make_confusion_mat import (
    collect_snapshots,
    parse_snapshot_for_confusion_matrix,
    build_confusion_matrix_data,
    format_training_dataset_label,
    extract_base_dataset_name,
    load_matrix_data,
    normalize_matrix_columns
)

# Import fingerprint functions
sys.path.insert(0, str(project_root / 'src' / 'fingerprints'))
try:
    from make_fingerprint_mat import (
        load_all_fingerprints,
        compute_distance_matrix,
        load_distance_matrix_data,
        EXPERIMENT_CONFIG,
        MACHINE_CONFIG,
        FINGERPRINT_DIR
    )
except ImportError:
    # Fallback if import fails
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "make_fingerprint_mat",
        project_root / 'src' / 'fingerprints' / 'make_fingerprint_mat.py'
    )
    make_fingerprint_mat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_fingerprint_mat)
    load_all_fingerprints = make_fingerprint_mat.load_all_fingerprints
    compute_distance_matrix = make_fingerprint_mat.compute_distance_matrix
    load_distance_matrix_data = make_fingerprint_mat.load_distance_matrix_data
    EXPERIMENT_CONFIG = make_fingerprint_mat.EXPERIMENT_CONFIG
    MACHINE_CONFIG = make_fingerprint_mat.MACHINE_CONFIG
    FINGERPRINT_DIR = make_fingerprint_mat.FINGERPRINT_DIR


def normalize_training_label(label):
    """
    Normalize training label to match between confusion matrix and fingerprint formats.
    Both scripts use similar formatting, but we need to ensure consistency.
    """
    # Both use similar formats, but fingerprint labels might use full names
    # while confusion matrix might abbreviate (e.g., PtOd vs pointodyssey)
    label_lower = label.lower()
    
    # Normalize pointodyssey abbreviations
    if label_lower.startswith('ptod'):
        label_lower = label_lower.replace('ptod', 'pointodyssey', 1)
    
    return label_lower


def match_labels(confusion_labels, fingerprint_labels):
    """
    Create a mapping between confusion matrix labels and fingerprint labels.
    
    Returns:
        Dictionary mapping confusion_label -> fingerprint_label
    """
    mapping = {}
    
    # Create normalized versions for matching
    confusion_normalized = {normalize_training_label(l): l for l in confusion_labels}
    fingerprint_normalized = {normalize_training_label(l): l for l in fingerprint_labels}
    
    # Try exact match first
    for norm_label, orig_label in confusion_normalized.items():
        if norm_label in fingerprint_normalized:
            mapping[orig_label] = fingerprint_normalized[norm_label]
        else:
            # Try partial match (e.g., if one has more parameters)
            for fp_norm, fp_orig in fingerprint_normalized.items():
                if norm_label in fp_norm or fp_norm in norm_label:
                    mapping[orig_label] = fp_orig
                    break
    
    return mapping


def extract_correlation_data(pck_matrix_data, w1_matrix_data, label_mapping):
    """
    Extract (distance, performance) pairs for all matching (training_dataset, benchmark) pairs.
    
    Args:
        pck_matrix_data: Dictionary with 'matrix', 'training_labels', 'benchmark_labels' from confusion matrix
        w1_matrix_data: Dictionary with 'matrix', 'training_labels', 'benchmark_labels' from fingerprint matrix
        label_mapping: Dictionary mapping confusion matrix training labels -> fingerprint training labels
    
    Returns:
        Tuple of (distances, performances, labels) where labels are (training_label, benchmark_label) tuples
    """
    distances = []
    performances = []
    labels = []
    
    pck_matrix = pck_matrix_data['matrix']
    pck_train_labels = pck_matrix_data['training_labels']
    pck_bench_labels = pck_matrix_data['benchmark_labels']
    
    w1_matrix = w1_matrix_data['matrix']
    w1_train_labels = w1_matrix_data['training_labels']
    w1_bench_labels = w1_matrix_data['benchmark_labels']
    
    # Create index maps for quick lookup
    w1_train_idx = {label: i for i, label in enumerate(w1_train_labels)}
    w1_bench_idx = {label: j for j, label in enumerate(w1_bench_labels)}
    
    # Iterate through all confusion matrix entries
    for i, train_label in enumerate(pck_train_labels):
        # Map to fingerprint label
        fp_train_label = label_mapping.get(train_label)
        if fp_train_label is None:
            continue
        
        fp_train_idx = w1_train_idx.get(fp_train_label)
        if fp_train_idx is None:
            continue
        
        for j, bench_label in enumerate(pck_bench_labels):
            # Check if both matrices have valid data for this pair
            pck_value = pck_matrix[i, j]
            if np.isnan(pck_value):
                continue
            
            # Find matching benchmark in fingerprint matrix
            # Benchmarks should match directly (they're the same names)
            fp_bench_idx = w1_bench_idx.get(bench_label)
            if fp_bench_idx is None:
                continue
            
            w1_value = w1_matrix[fp_train_idx, fp_bench_idx]
            if np.isnan(w1_value):
                continue
            
            # Both values are valid, add to lists
            distances.append(w1_value)
            performances.append(pck_value)
            labels.append((train_label, bench_label))
    
    return np.array(distances), np.array(performances), labels


def plot_correlation(distances, performances, labels, output_path, metric='pck', normalized=False):
    """
    Create scatter plot of PCK performance vs Wasserstein-1 distance.
    
    Args:
        distances: Array of Wasserstein-1 distances
        performances: Array of PCK performances
        labels: List of (training_label, benchmark_label) tuples
        output_path: Path to save the plot
        metric: Metric name for y-axis label
        normalized: Whether performances are normalized (0-1 range)
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot
    scatter = ax.scatter(distances, performances, alpha=0.6, s=50, edgecolors='black', linewidths=0.5)
    
    # Add labels for each point (optional, might be cluttered)
    # Uncomment if you want to see labels
    # for i, (dist, perf, (train, bench)) in enumerate(zip(distances, performances, labels)):
    #     ax.annotate(f'{train}\n{bench}', (dist, perf), fontsize=6, alpha=0.7)
    
    # Compute correlation
    if len(distances) > 1:
        corr, p_value = pearsonr(distances, performances)
        ax.text(0.05, 0.95, f'Pearson r = {corr:.3f}\np-value = {p_value:.4f}',
                transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels and title
    ax.set_xlabel('Wasserstein-1 Distance (Magnitude Histogram)', fontsize=12)
    if normalized:
        ax.set_ylabel(f'{metric.upper()} Performance (Normalized, 0-1)', fontsize=12)
        title_suffix = " [Normalized]"
    else:
        ax.set_ylabel(f'{metric.upper()} Performance (%)', fontsize=12)
        title_suffix = ""
    ax.set_title(f'PCK Performance vs Wasserstein-1 Distance{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot correlation between PCK performance and Wasserstein-1 distance',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--snapshots_dir',
        type=str,
        required=True,
        help='Directory containing snapshot subdirectories (for confusion matrix)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='pck_vs_w1_correlation.png',
        help='Output path for the plot (default: pck_vs_w1_correlation.png)'
    )
    
    parser.add_argument(
        '--metric',
        type=str,
        default='pck',
        help='Metric to use from confusion matrix (default: pck)'
    )
    
    parser.add_argument(
        '--use_best',
        action='store_true',
        help='Use best performance instead of average (default: False, uses average)'
    )
    
    parser.add_argument(
        '--fingerprint_config',
        type=str,
        default=EXPERIMENT_CONFIG,
        help=f'Path to fingerprint experiment config (default: {EXPERIMENT_CONFIG})'
    )
    
    parser.add_argument(
        '--fingerprint_dir',
        type=str,
        default=FINGERPRINT_DIR,
        help=f'Directory containing fingerprint JSON files (default: {FINGERPRINT_DIR})'
    )
    
    parser.add_argument(
        '--load_pck_data',
        type=str,
        default=None,
        help='Load PCK matrix data from saved files (provide base path, e.g., ./confusion_matrices/pck_average_matrix_data)'
    )
    
    parser.add_argument(
        '--load_w1_data',
        type=str,
        default=None,
        help='Load W1 distance matrix data from saved files (provide base path, e.g., ./fingerprint_matrices/data/w1_hist_distance_matrix_data)'
    )
    
    parser.add_argument(
        '--normalized',
        action='store_true',
        help='Use column-normalized PCK performance (normalized within each benchmark)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Computing Correlation: PCK Performance vs W1 Distance")
    print("="*60)
    
    # 1. Build confusion matrix (PCK performance)
    print("\n1. Building confusion matrix (PCK performance)...")
    
    if args.load_pck_data:
        try:
            print(f"   Loading saved PCK data from: {args.load_pck_data}")
            pck_matrix_data = load_matrix_data(args.load_pck_data)
            print("   Successfully loaded saved PCK data!")
        except FileNotFoundError as e:
            print(f"   Warning: Could not load saved data: {e}")
            print("   Computing from scratch...")
            print(f"   Snapshots directory: {args.snapshots_dir}")
            snapshot_dirs = collect_snapshots(args.snapshots_dir)
            
            if not snapshot_dirs:
                print("Error: No snapshot directories found!")
                return
            
            print(f"   Found {len(snapshot_dirs)} snapshot directory(ies)")
            
            snapshots_data = []
            for snapshot_dir in snapshot_dirs:
                data = parse_snapshot_for_confusion_matrix(snapshot_dir)
                if data and (data['best_performance'] or data.get('snapshot_path')):
                    snapshots_data.append(data)
            
            if not snapshots_data:
                print("Error: No valid snapshot data found!")
                return
            
            pck_matrix_data = build_confusion_matrix_data(
                snapshots_data,
                metric=args.metric,
                use_best=args.use_best
            )
    else:
        print(f"   Snapshots directory: {args.snapshots_dir}")
        snapshot_dirs = collect_snapshots(args.snapshots_dir)
        
        if not snapshot_dirs:
            print("Error: No snapshot directories found!")
            return
        
        print(f"   Found {len(snapshot_dirs)} snapshot directory(ies)")
        
        snapshots_data = []
        for snapshot_dir in snapshot_dirs:
            data = parse_snapshot_for_confusion_matrix(snapshot_dir)
            if data and (data['best_performance'] or data.get('snapshot_path')):
                snapshots_data.append(data)
        
        if not snapshots_data:
            print("Error: No valid snapshot data found!")
            return
        
        pck_matrix_data = build_confusion_matrix_data(
            snapshots_data,
            metric=args.metric,
            use_best=args.use_best
        )
    
    print(f"   Confusion matrix shape: {pck_matrix_data['matrix'].shape}")
    print(f"   Training datasets: {len(pck_matrix_data['training_labels'])}")
    print(f"   Benchmarks: {len(pck_matrix_data['benchmark_labels'])}")
    
    # Normalize PCK matrix if requested
    if args.normalized:
        print("\n   Normalizing PCK matrix (column-normalized)...")
        pck_matrix_data = {
            'matrix': normalize_matrix_columns(pck_matrix_data['matrix']),
            'training_labels': pck_matrix_data['training_labels'],
            'benchmark_labels': pck_matrix_data['benchmark_labels']
        }
    
    # 2. Build fingerprint distance matrix (Wasserstein-1)
    print("\n2. Building fingerprint distance matrix (Wasserstein-1)...")
    
    if args.load_w1_data:
        try:
            print(f"   Loading saved W1 data from: {args.load_w1_data}")
            w1_matrix, w1_train_labels, w1_eval_labels, _ = load_distance_matrix_data(args.load_w1_data)
            w1_matrix_data = {
                'matrix': w1_matrix,
                'training_labels': w1_train_labels,
                'benchmark_labels': w1_eval_labels
            }
            print("   Successfully loaded saved W1 data!")
        except FileNotFoundError as e:
            print(f"   Warning: Could not load saved data: {e}")
            print("   Computing from scratch...")
            print(f"   Fingerprint directory: {args.fingerprint_dir}")
            
            training_fingerprints, eval_fingerprints = load_all_fingerprints(
                args.fingerprint_config,
                MACHINE_CONFIG,
                args.fingerprint_dir
            )
            
            if not training_fingerprints or not eval_fingerprints:
                print("Error: Could not load fingerprints!")
                return
            
            print(f"   Training fingerprints: {len(training_fingerprints)}")
            print(f"   Evaluation fingerprints: {len(eval_fingerprints)}")
            
            w1_matrix, w1_train_labels, w1_eval_labels = compute_distance_matrix(
                training_fingerprints,
                eval_fingerprints,
                distance_name="w1_hist"
            )
            
            w1_matrix_data = {
                'matrix': w1_matrix,
                'training_labels': w1_train_labels,
                'benchmark_labels': w1_eval_labels
            }
    else:
        print(f"   Fingerprint directory: {args.fingerprint_dir}")
        
        training_fingerprints, eval_fingerprints = load_all_fingerprints(
            args.fingerprint_config,
            MACHINE_CONFIG,
            args.fingerprint_dir
        )
        
        if not training_fingerprints or not eval_fingerprints:
            print("Error: Could not load fingerprints!")
            return
        
        print(f"   Training fingerprints: {len(training_fingerprints)}")
        print(f"   Evaluation fingerprints: {len(eval_fingerprints)}")
        
        w1_matrix, w1_train_labels, w1_eval_labels = compute_distance_matrix(
            training_fingerprints,
            eval_fingerprints,
            distance_name="w1_hist"
        )
        
        w1_matrix_data = {
            'matrix': w1_matrix,
            'training_labels': w1_train_labels,
            'benchmark_labels': w1_eval_labels
        }
    
    print(f"   Distance matrix shape: {w1_matrix.shape}")
    
    # 3. Match labels between the two matrices
    print("\n3. Matching labels between matrices...")
    label_mapping = match_labels(
        pck_matrix_data['training_labels'],
        w1_matrix_data['training_labels']
    )
    
    print(f"   Matched {len(label_mapping)} training dataset labels")
    if len(label_mapping) < len(pck_matrix_data['training_labels']):
        unmatched = set(pck_matrix_data['training_labels']) - set(label_mapping.keys())
        print(f"   Warning: {len(unmatched)} labels could not be matched: {list(unmatched)[:5]}")
    
    # 4. Extract correlation data
    print("\n4. Extracting correlation data...")
    distances, performances, labels = extract_correlation_data(
        pck_matrix_data,
        w1_matrix_data,
        label_mapping
    )
    
    print(f"   Found {len(distances)} valid (distance, performance) pairs")
    
    if len(distances) == 0:
        print("Error: No matching data points found!")
        return
    
    # 5. Create plot
    print("\n5. Creating plot...")
    output_path = Path(args.output)
    print(f"   Output path: {output_path}")
    plot_correlation(distances, performances, labels, output_path, metric=args.metric, normalized=args.normalized)
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"  Performance range: [{performances.min():.2f}, {performances.max():.2f}]")
    if len(distances) > 1:
        corr, p_value = pearsonr(distances, performances)
        print(f"  Pearson correlation: {corr:.4f} (p-value: {p_value:.4f})")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

