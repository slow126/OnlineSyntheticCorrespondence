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
from scipy import stats

# Try to import statsmodels for mixed-effects regression
try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Mixed-effects regression will be skipped.")
    print("Install with: pip install statsmodels")

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
        snapshot_path = Path(snapshot_dir)
        summary_path = snapshot_path / 'training_summary.txt'
        
        # Skip snapshots without training summary
        if not summary_path.exists():
            print(f"  Skipping {snapshot_dir}: training_summary.txt not found")
            continue
        
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
        For mixed datasets, converts "+" to "_" for consistency with CSV lookups
    """
    if not os.path.exists(summary_path):
        return None
    
    dataset = None
    try:
        with open(summary_path, 'r') as f:
            for line in f:
                if line.startswith('Train dataset:'):
                    dataset = line.split('Train dataset:')[1].strip()
                    dataset = dataset.lower()  # Normalize to lowercase for lookup
                    # Convert "+" to "_" for mixed datasets (e.g., "spair+synthetic" -> "spair_synthetic")
                    # This ensures consistency with CSV lookup formats
                    dataset = dataset.replace('+', '_')
                    break
    except Exception as e:
        print(f"Warning: Could not parse training dataset from {summary_path}: {e}")
        return None

    try:
        dir_info = parse_directory_name(Path(summary_path).parent.name)
        dir_dataset = dir_info.get("dataset", "").lower() if dir_info else None
    except Exception:
        dir_dataset = None

    if dir_dataset:
        if dir_dataset.startswith("synthetic_") and dataset == "synthetic":
            dataset = dir_dataset
        elif (
            "_synthetic_" in dir_dataset
            and re.search(r"_\d+_\d+$", dir_dataset)
            and (dataset is None or "synthetic" in dataset)
        ):
            dataset = dir_dataset
        elif dataset is None:
            dataset = dir_dataset

    return dataset


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


def compute_within_benchmark_correlations(df):
    """
    Compute correlation between MMD² and PCK within each benchmark.
    
    Args:
        df: DataFrame with columns ['mmd2', 'best_pck', 'benchmark', 'training_dataset']
        
    Returns:
        DataFrame with correlation results per benchmark
    """
    results = []
    
    for benchmark in df['benchmark'].unique():
        subset = df[df['benchmark'] == benchmark]
        n = len(subset)
        
        if n >= 3:  # Need at least 3 points for meaningful correlation
            # Check for zero variance
            if subset['mmd2'].std() > 0 and subset['best_pck'].std() > 0:
                r, p = stats.pearsonr(subset['mmd2'], subset['best_pck'])
                results.append({
                    'benchmark': benchmark,
                    'correlation': r,
                    'p_value': p,
                    'n_points': n,
                    'significant': p < 0.05
                })
            else:
                results.append({
                    'benchmark': benchmark,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'n_points': n,
                    'significant': False
                })
        else:
            results.append({
                'benchmark': benchmark,
                'correlation': np.nan,
                'p_value': np.nan,
                'n_points': n,
                'significant': False
            })
    
    return pd.DataFrame(results)


def detect_outliers_iqr(series, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Args:
        series: pandas Series of values
        multiplier: IQR multiplier (default 1.5, use 3.0 for more extreme outliers)
        
    Returns:
        Boolean Series indicating which values are outliers
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    if IQR == 0:
        # No variance, no outliers
        return pd.Series([False] * len(series), index=series.index)
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (series < lower_bound) | (series > upper_bound)


def filter_outliers_by_benchmark(df, columns=['best_pck', 'mmd2'], multiplier=1.5):
    """
    Filter outliers within each benchmark using IQR method.
    
    Args:
        df: DataFrame with columns to check for outliers
        columns: List of column names to check for outliers
        multiplier: IQR multiplier for outlier detection
        
    Returns:
        DataFrame with outliers removed, and info about removed outliers
    """
    df = df.copy()
    original_len = len(df)
    
    # Track which rows are outliers
    is_outlier = pd.Series([False] * len(df), index=df.index)
    
    for benchmark in df['benchmark'].unique():
        bench_mask = df['benchmark'] == benchmark
        bench_data = df[bench_mask]
        
        # Check each column for outliers
        for col in columns:
            if col in bench_data.columns:
                col_outliers = detect_outliers_iqr(bench_data[col], multiplier=multiplier)
                is_outlier[bench_data.index[col_outliers]] = True
    
    # Remove outliers
    df_filtered = df[~is_outlier].copy()
    
    n_removed = original_len - len(df_filtered)
    if n_removed > 0:
        print(f"  Removed {n_removed} outlier(s) using IQR method (multiplier={multiplier})")
        # Print which benchmarks had outliers removed
        removed_data = df[is_outlier]
        if len(removed_data) > 0:
            for benchmark in removed_data['benchmark'].unique():
                bench_removed = removed_data[removed_data['benchmark'] == benchmark]
                print(f"    {benchmark}: removed {len(bench_removed)} point(s)")
    
    return df_filtered


def compute_zscore_correlation(df, robust=False, filter_outliers=True, outlier_multiplier=1.5):
    """
    Compute correlation using z-score normalized PCK within each benchmark.
    Only PCK is z-scored; the other metric (MMD/coverage) remains in original scale.
    This removes baseline difficulty differences between benchmarks for PCK.
    
    Args:
        df: DataFrame with columns ['mmd2', 'best_pck', 'benchmark']
        robust: If True, use robust z-score (median/MAD) which is less sensitive to outliers.
                If False, use standard z-score (mean/std).
        filter_outliers: If True, remove outliers using IQR method before computing z-scores.
        outlier_multiplier: IQR multiplier for outlier detection (default 1.5, use 3.0 for more extreme)
        
    Returns:
        Tuple of (correlation, p_value, df_with_zscores)
        Note: df will have 'pck_z' (z-scored PCK) and 'mmd_z' (original mmd2/coverage values)
    """
    df = df.copy()
    
    # Filter outliers if requested
    if filter_outliers:
        df = filter_outliers_by_benchmark(df, columns=['best_pck', 'mmd2'], multiplier=outlier_multiplier)
    
    if robust:
        # Robust z-score using median and MAD (Median Absolute Deviation)
        # MAD is scaled by 1.4826 to make it comparable to std for normal distributions
        def robust_zscore(x):
            median = x.median()
            mad = (x - median).abs().median()
            # Scale MAD to be comparable to standard deviation
            mad_scaled = mad * 1.4826 if mad > 0 else 1.0
            if mad_scaled > 0:
                return (x - median) / mad_scaled
            return x * 0  # Return zeros if no variance
        zscore_func = robust_zscore
    else:
        # Standard z-score using mean and std
        def standard_zscore(x):
            if x.std() > 0:
                return (x - x.mean()) / x.std()
            return x * 0  # Return zeros if no variance
        zscore_func = standard_zscore
    
    # Only z-score PCK, keep the other metric (mmd2/coverage) in original scale
    df['pck_z'] = df.groupby('benchmark')['best_pck'].transform(zscore_func)
    # Keep original metric for x-axis (not z-scored)
    df['mmd_z'] = df['mmd2']  # Keep original scale
    
    # Remove any NaN values
    df_clean = df.dropna(subset=['pck_z', 'mmd_z'])
    
    if len(df_clean) >= 3:
        # Correlation: original metric (x) vs z-scored PCK (y)
        r, p = stats.pearsonr(df_clean['mmd_z'], df_clean['pck_z'])
        return r, p, df
    else:
        return np.nan, np.nan, df


def run_mixed_effects_regression(df):
    """
    Run mixed-effects regression with benchmark as random effect.
    This controls for baseline difficulty differences between benchmarks.
    
    Model: best_pck ~ mmd2 + (1|benchmark)
    
    Args:
        df: DataFrame with columns ['mmd2', 'best_pck', 'benchmark']
        
    Returns:
        Dictionary with regression results or None if statsmodels not available
    """
    if not HAS_STATSMODELS:
        return None
    
    df = df.copy()
    df = df.dropna(subset=['mmd2', 'best_pck', 'benchmark'])
    
    if len(df) < 5:
        return None
    
    try:
        # Random intercept model: benchmark as grouping variable
        model = smf.mixedlm("best_pck ~ mmd2", data=df, groups=df["benchmark"])
        result = model.fit(method='lbfgs')  # Use LBFGS optimizer for stability
        
        return {
            'mmd2_coef': result.fe_params.get('mmd2', np.nan),
            'mmd2_pvalue': result.pvalues.get('mmd2', np.nan),
            'intercept': result.fe_params.get('Intercept', np.nan),
            'intercept_pvalue': result.pvalues.get('Intercept', np.nan),
            'random_effect_var': result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else np.nan,
            'n_observations': len(df),
            'n_groups': df['benchmark'].nunique(),
            'converged': result.converged,
            'llf': result.llf,  # Log-likelihood
            'aic': result.aic,
            'bic': result.bic,
            'summary': str(result.summary())
        }
    except Exception as e:
        print(f"  Warning: Mixed-effects regression failed: {e}")
        return None


def print_statistical_analysis(df, analysis_name="MMD"):
    """
    Print comprehensive statistical analysis results.
    
    Args:
        df: DataFrame with columns ['mmd2', 'best_pck', 'benchmark', 'training_dataset']
        analysis_name: Name of the analysis (e.g., "Flow MMD" or "Feature MMD")
    """
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS: {analysis_name} vs PCK")
    print(f"{'='*70}")
    
    # 1. Overall correlation (naive - for reference)
    if len(df) >= 3:
        r_naive, p_naive = stats.pearsonr(df['mmd2'], df['best_pck'])
        print(f"\n1. NAIVE OVERALL CORRELATION (pooled, ignoring benchmark):")
        print(f"   r = {r_naive:.4f}, p = {p_naive:.4f}, n = {len(df)}")
        print(f"   ⚠️  This may be confounded by benchmark difficulty!")
    
    # 2. Within-benchmark correlations
    print(f"\n2. WITHIN-BENCHMARK CORRELATIONS:")
    print(f"   (Controls for benchmark difficulty by analyzing each benchmark separately)")
    within_corr = compute_within_benchmark_correlations(df)
    print(f"   {'Benchmark':<15} {'r':>8} {'p-value':>10} {'n':>5} {'Sig?':>6}")
    print(f"   {'-'*45}")
    for _, row in within_corr.iterrows():
        sig_marker = '*' if row['significant'] else ''
        r_str = f"{row['correlation']:.4f}" if not np.isnan(row['correlation']) else 'N/A'
        p_str = f"{row['p_value']:.4f}" if not np.isnan(row['p_value']) else 'N/A'
        print(f"   {row['benchmark']:<15} {r_str:>8} {p_str:>10} {row['n_points']:>5} {sig_marker:>6}")
    
    # Summary of within-benchmark
    valid_corrs = within_corr.dropna(subset=['correlation'])
    if len(valid_corrs) > 0:
        mean_r = valid_corrs['correlation'].mean()
        print(f"\n   Mean within-benchmark correlation: r = {mean_r:.4f}")
        n_sig = valid_corrs['significant'].sum()
        print(f"   Significant correlations: {n_sig}/{len(valid_corrs)}")
    
    # 3. Z-score normalized correlation
    print(f"\n3. Z-SCORE NORMALIZED CORRELATION:")
    print(f"   (Z-scores PCK within each benchmark to remove difficulty differences)")
    print(f"   (Other metric remains in original scale)")
    print(f"   (Filtering outliers using IQR method before computing z-scores)")
    r_z, p_z, df_z = compute_zscore_correlation(df, robust=False, filter_outliers=True, outlier_multiplier=1.5)
    if not np.isnan(r_z):
        print(f"   r = {r_z:.4f}, p = {p_z:.4f}")
    else:
        print(f"   Could not compute (insufficient data)")
    
    # 4. Mixed-effects regression
    print(f"\n4. MIXED-EFFECTS REGRESSION:")
    print(f"   (Model: PCK ~ MMD² + (1|benchmark) - random intercept per benchmark)")
    me_results = run_mixed_effects_regression(df)
    if me_results:
        print(f"   MMD² coefficient: {me_results['mmd2_coef']:.4f}")
        print(f"   MMD² p-value: {me_results['mmd2_pvalue']:.4f}")
        print(f"   Interpretation: 1 unit increase in MMD² → {me_results['mmd2_coef']:.2f} change in PCK%")
        print(f"   Random effect variance (benchmark): {me_results['random_effect_var']:.4f}")
        print(f"   Model fit: AIC={me_results['aic']:.1f}, BIC={me_results['bic']:.1f}")
        print(f"   Observations: {me_results['n_observations']}, Groups: {me_results['n_groups']}")
        if me_results['mmd2_pvalue'] < 0.05:
            print(f"   ✓ MMD² effect is statistically significant (p < 0.05)")
        else:
            print(f"   ✗ MMD² effect is NOT statistically significant (p >= 0.05)")
    else:
        if HAS_STATSMODELS:
            print(f"   Could not fit model (insufficient data or convergence failure)")
        else:
            print(f"   Skipped (statsmodels not installed)")
    
    print(f"\n{'='*70}\n")
    
    return {
        'naive_correlation': (r_naive, p_naive) if len(df) >= 3 else (np.nan, np.nan),
        'within_benchmark': within_corr,
        'zscore_correlation': (r_z, p_z),
        'mixed_effects': me_results
    }


def save_statistical_analysis_to_file(df, output_path, analysis_name="MMD"):
    """
    Save statistical analysis results to a text file.
    
    Args:
        df: DataFrame with analysis data
        output_path: Path object for output directory
        analysis_name: Name of analysis for filename
    """
    import io
    from contextlib import redirect_stdout
    
    # Capture print output
    f = io.StringIO()
    with redirect_stdout(f):
        print_statistical_analysis(df, analysis_name)
    
    output_text = f.getvalue()
    
    # Save to file
    safe_name = analysis_name.lower().replace(' ', '_')
    output_file = output_path / f'{safe_name}_statistical_analysis.txt'
    with open(output_file, 'w') as file:
        file.write(output_text)
    
    print(f"Saved statistical analysis to: {output_file}")
    return output_file


def create_faceted_scatter_plot(df, output_path, analysis_name="MMD", dataset_color_map=None):
    """
    Create faceted scatter plot with one panel per benchmark.
    
    Args:
        df: DataFrame with columns ['mmd2', 'best_pck', 'benchmark', 'training_dataset']
        output_path: Path object for output directory
        analysis_name: Name for plot title and filename
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    benchmarks = sorted(df['benchmark'].unique())
    n_benchmarks = len(benchmarks)
    
    if n_benchmarks == 0:
        return
    
    # Calculate grid size
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    axes = axes.flatten()
    
    # Compute within-benchmark correlations for annotations
    within_corr = compute_within_benchmark_correlations(df)
    corr_dict = {row['benchmark']: row for _, row in within_corr.iterrows()}
    
    for idx, benchmark in enumerate(benchmarks):
        ax = axes[idx]
        subset = df[df['benchmark'] == benchmark]
        
        # Plot each training dataset with its color
        for training_dataset in subset['training_dataset'].unique():
            ds_subset = subset[subset['training_dataset'] == training_dataset]
            color = dataset_color_map.get(training_dataset, 'black') if dataset_color_map else None
            ax.scatter(ds_subset['mmd2'], ds_subset['best_pck'], 
                      color=color, label=training_dataset,
                      s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add regression line if enough points
        if len(subset) >= 3:
            z = np.polyfit(subset['mmd2'], subset['best_pck'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['mmd2'].min(), subset['mmd2'].max(), 100)
            ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2)
        
        # Add correlation annotation
        corr_info = corr_dict.get(benchmark, {})
        r = corr_info.get('correlation', np.nan)
        p_val = corr_info.get('p_value', np.nan)
        n = corr_info.get('n_points', len(subset))
        
        if not np.isnan(r):
            sig_star = '*' if p_val < 0.05 else ''
            ax.text(0.05, 0.95, f'r={r:.3f}{sig_star}\nn={n}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(f'{analysis_name}²', fontsize=10)
        ax.set_ylabel('Best PCK (%)', fontsize=10)
        ax.set_title(benchmark.upper(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'{analysis_name}² vs Best PCK - By Benchmark\n(* indicates p < 0.05)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = analysis_name.lower().replace(' ', '_')
    output_file = output_path / f'{safe_name}_vs_pck_by_benchmark.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved faceted plot: {output_file}")
    plt.close()


def create_zscore_scatter_plot(df, output_path, analysis_name="MMD", dataset_color_map=None):
    """
    Create scatter plot with z-score normalized PCK (y-axis) and original metric (x-axis).
    Only PCK is z-scored within each benchmark; the other metric remains in original scale.
    Filters outliers using IQR method before computing z-scores.
    
    Args:
        df: DataFrame with analysis data
        output_path: Path object for output directory
        analysis_name: Name for plot title and filename
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    r_z, p_z, df_z = compute_zscore_correlation(df, robust=False, filter_outliers=True, outlier_multiplier=1.5)
    
    if np.isnan(r_z):
        print(f"  Skipping z-score plot (insufficient data)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot by training dataset
    for training_dataset in df_z['training_dataset'].unique():
        subset = df_z[df_z['training_dataset'] == training_dataset]
        color = dataset_color_map.get(training_dataset, 'black') if dataset_color_map else None
        ax.scatter(subset['mmd_z'], subset['pck_z'],
                  color=color, label=training_dataset,
                  s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add regression line
    df_clean = df_z.dropna(subset=['mmd_z', 'pck_z'])
    if len(df_clean) >= 3:
        z = np.polyfit(df_clean['mmd_z'], df_clean['pck_z'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_clean['mmd_z'].min(), df_clean['mmd_z'].max(), 100)
        ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Trend')
    
    # Add correlation annotation
    ax.text(0.05, 0.95, f'r = {r_z:.4f}\np = {p_z:.4f}', 
           transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel(f'{analysis_name}² (original scale)', fontsize=12)
    ax.set_ylabel('Best PCK (z-scored within benchmark)', fontsize=12)
    ax.set_title(f'{analysis_name}² vs Z-Score Normalized PCK\n(PCK normalized to control for benchmark difficulty, outliers filtered)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = analysis_name.lower().replace(' ', '_')
    output_file = output_path / f'{safe_name}_vs_pck_zscore.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved z-score plot: {output_file}")
    plt.close()


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
            print(f"  Skipping {snapshot_path}: Could not parse training dataset from summary")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            print(f"  Skipping {snapshot_path}: No best performance data found in summary")
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
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)  # Track which snapshot this came from
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
    # Multiple snapshots with same training dataset will show as multiple points
    for training_dataset, points in datasets_points.items():
        mmd2_values = [p['mmd2'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        color = dataset_color_map.get(training_dataset, 'black')
        
        # Count how many trials we have for this dataset
        num_trials = len(set(p.get('snapshot_path', '') for p in points if 'snapshot_path' in p))
        if num_trials == 0:
            num_trials = len(points)  # Fallback: use number of points
        
        # Use label only once per dataset (to avoid duplicate legend entries)
        # If multiple trials, show count in label
        if num_trials > 1:
            label = f"{training_dataset} ({num_trials} trials)"
        else:
            label = training_dataset
        
        ax.scatter(mmd2_values, pck_values, 
                  color=color, label=label, 
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add benchmark labels (optional - can be commented out if too cluttered)
        for point in points:
            ax.annotate(point['benchmark'], 
                       (point['mmd2'], point['best_pck']),
                       fontsize=7, alpha=0.6,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Training Dataset Flow MMD² vs Benchmark', fontsize=12)
    ax.set_ylabel('Best PCK (%)', fontsize=12)
    ax.set_title('Training Dataset Flow MMD² vs Best PCK Performance', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / 'training_mmd_vs_best_pck.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved MMD vs PCK scatter plot: {output_file}")
    plt.close()
    
    # Create DataFrame for statistical analysis
    df = pd.DataFrame(data_points)
    
    # Run and print statistical analysis
    print_statistical_analysis(df, "Flow MMD")
    
    # Save statistical analysis to file
    save_statistical_analysis_to_file(df, output_path, "Flow MMD")
    
    # Create faceted plot (one panel per benchmark)
    create_faceted_scatter_plot(df, output_path, "Flow MMD", dataset_color_map)
    
    # Create z-score normalized plot
    create_zscore_scatter_plot(df, output_path, "Flow MMD", dataset_color_map)


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
            print(f"  Skipping {snapshot_path}: Could not parse training dataset from summary")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            print(f"  Skipping {snapshot_path}: No best performance data found in summary")
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
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)  # Track which snapshot this came from
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
    # Multiple snapshots with same training dataset will show as multiple points
    for training_dataset, points in datasets_points.items():
        mmd2_values = [p['mmd2'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        color = dataset_color_map.get(training_dataset, 'black')
        
        # Count how many trials we have for this dataset
        num_trials = len(set(p.get('snapshot_path', '') for p in points if 'snapshot_path' in p))
        if num_trials == 0:
            num_trials = len(points)  # Fallback: use number of points
        
        # Use label only once per dataset (to avoid duplicate legend entries)
        # If multiple trials, show count in label
        if num_trials > 1:
            label = f"{training_dataset} ({num_trials} trials)"
        else:
            label = training_dataset
        
        ax.scatter(mmd2_values, pck_values, 
                  color=color, label=label, 
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
    
    # Create DataFrame for statistical analysis
    df = pd.DataFrame(data_points)
    
    # Run and print statistical analysis
    print_statistical_analysis(df, "Feature MMD")
    
    # Save statistical analysis to file
    save_statistical_analysis_to_file(df, output_path, "Feature MMD")
    
    # Create faceted plot (one panel per benchmark)
    create_faceted_scatter_plot(df, output_path, "Feature MMD", dataset_color_map)
    
    # Create z-score normalized plot
    create_zscore_scatter_plot(df, output_path, "Feature MMD", dataset_color_map)


def create_mmd_vs_pck_errorbar_plot(snapshots_data, mmd_lookup, output_dir, dataset_color_map, mmd_type="Flow"):
    """
    Create error bar plot showing MMD² vs PCK with error bars for multiple configs.
    
    Since MMD² is a property of the dataset pair (not the model), we group all model
    configurations trained on the same dataset and show mean PCK ± std for each
    (training_dataset, benchmark) pair.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        mmd_lookup: Dictionary mapping (dataset1_split1, dataset2_split2) -> mmd2 value
        output_dir: Output directory path
        dataset_color_map: Dictionary mapping training_dataset -> color
        mmd_type: Type of MMD analysis ("Flow" or "Feature")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all data points grouped by (training_dataset, benchmark)
    # Key: (base_training_dataset, benchmark), Value: list of (mmd2, pck) tuples
    grouped_data = defaultdict(list)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get base training dataset name from summary (for MMD lookup)
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        # For each benchmark, look up MMD² and store data point
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up MMD² with splits
            mmd2 = None
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Try test split first, then val split
            mmd2 = mmd_lookup.get((training_dataset_train, benchmark_test))
            if mmd2 is None:
                mmd2 = mmd_lookup.get((training_dataset_train, benchmark_val))
            
            # Fall back to old format without splits
            if mmd2 is None:
                mmd2 = mmd_lookup.get((base_training_dataset, benchmark_lower))
            
            if mmd2 is not None:
                key = (training_dataset_label, benchmark)
                grouped_data[key].append({
                    'mmd2': mmd2,
                    'best_pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark
                })
    
    if not grouped_data:
        print(f"  Warning: No data collected for {mmd_type} MMD error bar plot")
        return
    
    # Compute statistics for each group
    plot_data = []
    for (training_dataset, benchmark), points in grouped_data.items():
        mmd2_values = [p['mmd2'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        
        # All mmd2 values should be identical (since it's a dataset property)
        mmd2_mean = np.mean(mmd2_values)
        
        # Compute statistics for PCK across different model configs
        pck_mean = np.mean(pck_values)
        pck_std = np.std(pck_values, ddof=1) if len(pck_values) > 1 else 0
        pck_stderr = pck_std / np.sqrt(len(pck_values)) if len(pck_values) > 1 else 0
        
        plot_data.append({
            'training_dataset': training_dataset,
            'benchmark': benchmark,
            'mmd2': mmd2_mean,
            'pck_mean': pck_mean,
            'pck_std': pck_std,
            'pck_stderr': pck_stderr,
            'n_configs': len(pck_values)
        })
    
    # Create error bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by training dataset for coloring
    dataset_groups = defaultdict(list)
    for data in plot_data:
        dataset_groups[data['training_dataset']].append(data)
    
    # Plot each training dataset with error bars
    for training_dataset, group_data in dataset_groups.items():
        mmd2_vals = [d['mmd2'] for d in group_data]
        pck_means = [d['pck_mean'] for d in group_data]
        pck_stds = [d['pck_std'] for d in group_data]
        
        color = dataset_color_map.get(training_dataset, 'black')
        
        # Plot with error bars (using std dev)
        ax.errorbar(mmd2_vals, pck_means, yerr=pck_stds,
                   fmt='o', label=training_dataset,
                   color=color, markersize=8, capsize=5, capthick=2,
                   alpha=0.7, elinewidth=2)
        
        # Optionally add benchmark labels
        for data in group_data:
            # Only label if there are multiple configs (error bars are meaningful)
            if data['n_configs'] > 1:
                ax.annotate(f"{data['benchmark']}\n(n={data['n_configs']})", 
                           (data['mmd2'], data['pck_mean']),
                           fontsize=7, alpha=0.6,
                           xytext=(5, 5), textcoords='offset points')
            else:
                ax.annotate(data['benchmark'], 
                           (data['mmd2'], data['pck_mean']),
                           fontsize=7, alpha=0.6,
                           xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel(f'Training Dataset {mmd_type} MMD² vs Benchmark', fontsize=12)
    ax.set_ylabel('Best PCK (%) - Mean ± Std Dev', fontsize=12)
    ax.set_title(f'{mmd_type} MMD² vs Best PCK Performance\n(Error bars show variation across model configurations)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    safe_name = mmd_type.lower().replace(' ', '_')
    output_file = output_path / f'training_{safe_name}_mmd_vs_best_pck_errorbars.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved {mmd_type} MMD error bar plot: {output_file}")
    plt.close()


def load_coverage_lookup(csv_path='coverage_results.csv', representation_filter=None):
    """
    Load coverage data from CSV file and create a normalized lookup.
    Automatically detects train vs eval/test/val splits and normalizes to train -> eval direction.
    
    Args:
        csv_path: Path to coverage_results.csv file
        representation_filter: Optional filter for representation type
        
    Returns:
        Dictionary mapping (train_dataset_split, eval_dataset_split) -> dict of coverage metrics
        Always normalized so train is first, eval is second
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot create coverage lookup.")
        return {}
    
    coverage_lookup = {}
    
    # Define split categories
    train_splits = {"train", "training"}
    eval_splits = {"val", "test", "validation", "eval"}
    
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            # Some CSVs (e.g., flow) don't include a representation column.
            # Treat missing/empty representation as the requested filter (if provided)
            # so that flow/resnet CSVs both load correctly.
            rep = str(row.get('representation', '')).lower()
            if (not rep or rep == 'nan') and representation_filter:
                rep = representation_filter.lower()
            if representation_filter and rep != representation_filter.lower():
                continue

            dataset1 = str(row['dataset1']).lower()
            dataset2 = str(row['dataset2']).lower()
            split1 = str(row.get('split1', '')).lower()
            split2 = str(row.get('split2', '')).lower()
            
            # Skip identical comparisons (same dataset AND same split)
            if dataset1 == dataset2 and split1 == split2:
                continue
            
            # Determine which is train and which is eval
            is_split1_train = split1 in train_splits
            is_split2_train = split2 in train_splits
            is_split1_eval = split1 in eval_splits
            is_split2_eval = split2 in eval_splits
            
            # Normalize to train -> eval direction
            if is_split1_train and is_split2_eval:
                # Already in correct direction: dataset1 (train) -> dataset2 (eval)
                train_dataset = dataset1
                train_split = split1
                eval_dataset = dataset2
                eval_split = split2
            elif is_split2_train and is_split1_eval:
                # Reversed: swap them
                train_dataset = dataset2
                train_split = split2
                eval_dataset = dataset1
                eval_split = split1
            elif is_split1_train and not is_split2_eval:
                # dataset1 is train, dataset2 might not have explicit eval split
                # Assume dataset2 is eval
                train_dataset = dataset1
                train_split = split1
                eval_dataset = dataset2
                eval_split = split2 if split2 else 'unknown'
            elif is_split2_train and not is_split1_eval:
                # dataset2 is train, dataset1 might not have explicit eval split
                # Swap them
                train_dataset = dataset2
                train_split = split2
                eval_dataset = dataset1
                eval_split = split1 if split1 else 'unknown'
            else:
                # Neither is clearly train/eval, try to infer from dataset names
                # Common pattern: training datasets often have "train" in name or are synthetic
                # For now, assume dataset1 -> dataset2 and store both directions
                train_dataset = dataset1
                train_split = split1 if split1 else 'train'
                eval_dataset = dataset2
                eval_split = split2 if split2 else 'unknown'
            
            # Create unique identifiers with splits
            train_id = f"{train_dataset}_{train_split}" if train_split else train_dataset
            eval_id = f"{eval_dataset}_{eval_split}" if eval_split else eval_dataset
            
            # Handle both legacy and new column names
            coverage_abs_val = row.get('coverage_abs', np.nan)
            coverage_rel_val = row.get('coverage_rel', np.nan)
            recall_val = row.get('recall', np.nan)
            precision_val = row.get('precision', np.nan)
            outside_val = row.get('outside', np.nan)

            if pd.isna(coverage_abs_val) and not pd.isna(recall_val):
                coverage_abs_val = recall_val
            if pd.isna(coverage_rel_val) and not pd.isna(recall_val):
                coverage_rel_val = recall_val
            if pd.isna(recall_val) and not pd.isna(coverage_abs_val):
                recall_val = coverage_abs_val
            if pd.isna(precision_val) and not pd.isna(outside_val):
                precision_val = 1.0 - outside_val

            # Store coverage metrics
            metrics = {
                'coverage_abs': float(coverage_abs_val),
                'coverage_rel': float(coverage_rel_val),
                'rho_95': float(row.get('rho_95', 0.0)),
                'rho_median': float(row.get('rho_median', 0.0)),
                'rho_mean': float(row.get('rho_mean', 0.0)),
                'epsilon': float(row.get('epsilon', 0.0)),
                'recall': float(recall_val) if not pd.isna(recall_val) else np.nan,
                'precision': float(precision_val) if not pd.isna(precision_val) else (float(row.get('precision', np.nan)) if 'precision' in row else np.nan),
                'outside': float(outside_val) if not pd.isna(outside_val) else (float(row.get('outside', np.nan)) if 'outside' in row else np.nan),
                'representation': rep,
            }
            
            # Store normalized direction: train -> eval
            coverage_lookup[(train_id, eval_id)] = metrics
            
            # Also store without explicit split in key for backward compatibility
            coverage_lookup[(train_dataset, eval_dataset)] = metrics
            
    except Exception as e:
        print(f"Warning: Could not load coverage lookup from {csv_path}: {e}")
        return {}
    
    return coverage_lookup


def create_coverage_vs_pck_scatter_plot(
    snapshots_data,
    coverage_lookup,
    output_dir,
    dataset_color_map,
    score_key='recall',
    score_label='Recall',
    coverage_tag=None,
):
    """
    Create scatter plot showing training dataset recall/precision vs best PCK.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which coverage metric to use
    coverage_label = score_label
    if coverage_tag:
        coverage_label = f"{coverage_label} ({coverage_tag})"
    
    # Collect all data points: (coverage, best_pck, training_dataset, benchmark)
    data_points = []
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get base training dataset name from summary (for coverage lookup)
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            print(f"  Skipping {snapshot_path}: Could not parse training dataset from summary")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            print(f"  Skipping {snapshot_path}: No best performance data found in summary")
            continue
        
        # For each benchmark, look up coverage and store data point
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up coverage with splits
            # The lookup is now normalized to always be train -> eval
            coverage_metrics = None
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Try test split first, then val split
            coverage_metrics = coverage_lookup.get((training_dataset_train, benchmark_test))
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((training_dataset_train, benchmark_val))
            
            # Fall back to old format without splits
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((base_training_dataset, benchmark_lower))
            
            if coverage_metrics is not None and score_key in coverage_metrics and not pd.isna(coverage_metrics[score_key]):
                data_points.append({
                    'coverage': coverage_metrics[score_key],
                    'best_pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
            else:
                print(f"  Warning: Coverage not found for ({base_training_dataset}, {benchmark_lower})")
    
    if not data_points:
        print(f"Warning: No data points collected for {coverage_label} vs PCK scatter plot")
        return
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group points by training dataset for plotting
    datasets_points = defaultdict(list)
    for point in data_points:
        datasets_points[point['training_dataset']].append(point)
    
    # Plot each training dataset with different color
    for training_dataset, points in datasets_points.items():
        coverage_values = [p['coverage'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        color = dataset_color_map.get(training_dataset, 'black')
        
        # Count how many trials we have for this dataset
        num_trials = len(set(p.get('snapshot_path', '') for p in points if 'snapshot_path' in p))
        if num_trials == 0:
            num_trials = len(points)
        
        # Use label only once per dataset
        if num_trials > 1:
            label = f"{training_dataset} ({num_trials} trials)"
        else:
            label = training_dataset
        
        ax.scatter(coverage_values, pck_values, 
                  color=color, label=label, 
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add benchmark labels
        for point in points:
            ax.annotate(point['benchmark'], 
                       (point['coverage'], point['best_pck']),
                       fontsize=7, alpha=0.6,
                       xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel(f'Training Dataset {coverage_label} of Benchmark', fontsize=12)
    ax.set_ylabel('Best PCK (%)', fontsize=12)
    ax.set_title(f'Training Dataset {coverage_label} vs Best PCK Performance', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    tag_suffix = f"_{coverage_tag}" if coverage_tag else ""
    output_file = output_path / f'training_{score_key}{tag_suffix}_vs_best_pck.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved {coverage_label} vs PCK scatter plot: {output_file}")
    plt.close()
    
    # Create DataFrame for statistical analysis
    df = pd.DataFrame(data_points)
    # Rename for compatibility with existing analysis functions
    df['mmd2'] = df['coverage']
    
    # Tag-aware analysis name so flow/resnet (and others) don't overwrite files
    analysis_name = f"{score_label} {coverage_tag}" if coverage_tag else f"{score_label}"
    
    # Run and print statistical analysis
    print_statistical_analysis(df, analysis_name)
    
    # Save statistical analysis to file
    save_statistical_analysis_to_file(df, output_path, analysis_name)
    
    # Create faceted plot (one panel per benchmark)
    create_faceted_scatter_plot(df, output_path, analysis_name, dataset_color_map)
    
    # Create z-score normalized plot
    create_zscore_scatter_plot(df, output_path, analysis_name, dataset_color_map)


def create_coverage_vs_pck_errorbar_plot(
    snapshots_data,
    coverage_lookup,
    output_dir,
    dataset_color_map,
    score_key='recall',
    score_label='Recall',
    coverage_tag=None,
):
    """
    Create error bar plot showing coverage vs PCK with error bars for multiple configs.
    
    Since coverage is a property of the dataset pair (not the model), we group all model
    configurations trained on the same dataset and show mean PCK ± std for each
    (training_dataset, benchmark) pair.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        coverage_lookup: Dictionary mapping (dataset1_split1, dataset2_split2) -> coverage metrics dict
        output_dir: Output directory path
        dataset_color_map: Dictionary mapping training_dataset -> color
        score_key: which metric to plot ('recall', 'precision', etc.)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which coverage metric to use
    coverage_label = score_label
    if coverage_tag:
        coverage_label = f"{coverage_label} ({coverage_tag})"
    
    # Collect all data points grouped by (training_dataset, benchmark)
    grouped_data = defaultdict(list)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get base training dataset name from summary
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        # For each benchmark, look up coverage and store data point
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up coverage with splits
            coverage_metrics = None
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Try test split first, then val split
            coverage_metrics = coverage_lookup.get((training_dataset_train, benchmark_test))
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((training_dataset_train, benchmark_val))
            
            # Fall back to old format without splits
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((base_training_dataset, benchmark_lower))
            # As a last resort, try reverse direction to detect swapped roles in CSVs
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((benchmark_test, training_dataset_train))
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((benchmark_val, training_dataset_train))
            if coverage_metrics is None:
                coverage_metrics = coverage_lookup.get((benchmark_lower, base_training_dataset))
            
            if coverage_metrics is not None and score_key in coverage_metrics and not pd.isna(coverage_metrics[score_key]):
                key = (training_dataset_label, benchmark)
                grouped_data[key].append({
                    'coverage': coverage_metrics[score_key],
                    'best_pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark
                })
    
    if not grouped_data:
        print(f"  Warning: No data collected for {coverage_label} error bar plot")
        return
    
    # Compute statistics for each group
    plot_data = []
    for (training_dataset, benchmark), points in grouped_data.items():
        coverage_values = [p['coverage'] for p in points]
        pck_values = [p['best_pck'] for p in points]
        
        # All coverage values should be identical
        coverage_mean = np.mean(coverage_values)
        
        # Compute statistics for PCK across different model configs
        pck_mean = np.mean(pck_values)
        pck_std = np.std(pck_values, ddof=1) if len(pck_values) > 1 else 0
        
        plot_data.append({
            'training_dataset': training_dataset,
            'benchmark': benchmark,
            'coverage': coverage_mean,
            'pck_mean': pck_mean,
            'pck_std': pck_std,
            'n_configs': len(pck_values)
        })
    
    # Create error bar plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by training dataset for coloring
    dataset_groups = defaultdict(list)
    for data in plot_data:
        dataset_groups[data['training_dataset']].append(data)
    
    # Plot each training dataset with error bars
    for training_dataset, group_data in dataset_groups.items():
        coverage_vals = [d['coverage'] for d in group_data]
        pck_means = [d['pck_mean'] for d in group_data]
        pck_stds = [d['pck_std'] for d in group_data]
        
        color = dataset_color_map.get(training_dataset, 'black')
        
        # Plot with error bars (using std dev)
        ax.errorbar(coverage_vals, pck_means, yerr=pck_stds,
                   fmt='o', label=training_dataset,
                   color=color, markersize=8, capsize=5, capthick=2,
                   alpha=0.7, elinewidth=2)
        
        # Add benchmark labels
        for data in group_data:
            if data['n_configs'] > 1:
                ax.annotate(f"{data['benchmark']}\n(n={data['n_configs']})", 
                           (data['coverage'], data['pck_mean']),
                           fontsize=7, alpha=0.6,
                           xytext=(5, 5), textcoords='offset points')
            else:
                ax.annotate(data['benchmark'], 
                           (data['coverage'], data['pck_mean']),
                           fontsize=7, alpha=0.6,
                           xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel(f'Training Dataset {coverage_label} of Benchmark', fontsize=12)
    ax.set_ylabel('Best PCK (%) - Mean ± Std Dev', fontsize=12)
    ax.set_title(f'{coverage_label} vs Best PCK Performance\n(Error bars show variation across model configurations)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    tag_suffix = f"_{coverage_tag}" if coverage_tag else ""
    output_file = output_path / f'training_{score_key}{tag_suffix}_vs_best_pck_errorbars.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved {coverage_label} error bar plot: {output_file}")
    plt.close()


def normalize_dataset_name_for_grouping(dataset_name):
    """
    Normalize dataset name by removing training configuration parameters.
    Removes things like 'logsteps100', 'steps100', 'stride1', etc. that are training configs,
    not dataset variations.
    
    Args:
        dataset_name: Dataset name string (e.g., 'synthetic_small_zoom_stride1')
        
    Returns:
        Normalized dataset name (e.g., 'synthetic_small_zoom')
    """
    import re
    # Remove training config parameters
    # Remove stride, sequence_length, freeze patterns
    name = re.sub(r'_stride\d+', '', dataset_name)
    name = re.sub(r'_sequence_length\d+', '', name)
    name = re.sub(r'_freeze[TF]', '', name)
    name = re.sub(r'_freezeTrue|_freezeFalse', '', name)
    # Remove logsteps/steps patterns (these are training configs, not dataset variations)
    name = re.sub(r'_logsteps\d+', '', name)
    name = re.sub(r'_steps\d+', '', name)
    name = re.sub(r'_S\d+', '', name)  # Remove sequence length shorthand
    
    return name


def collect_best_pck_by_benchmark(snapshots_data):
    """
    Collect best PCK per benchmark per training dataset.
    Groups by normalized dataset name (removes training config variations).
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        
    Returns:
        Dictionary mapping benchmark -> training_dataset -> best_pck
        Also returns a list of all unique training datasets
    """
    benchmark_data = defaultdict(lambda: defaultdict(list))
    all_datasets = set()
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue
        
        # Normalize dataset name (remove training config variations)
        normalized_dataset = normalize_dataset_name_for_grouping(training_dataset_label)
        all_datasets.add(normalized_dataset)
        
        # Store best PCK for each benchmark
        for benchmark, best_pck in best_performance.items():
            benchmark_data[benchmark][normalized_dataset].append(best_pck)
    
    # For each benchmark-dataset pair, take the maximum (best) PCK
    # (in case there are multiple runs with same normalized dataset name)
    benchmark_best_pck = {}
    for benchmark, datasets_dict in benchmark_data.items():
        benchmark_best_pck[benchmark] = {}
        for dataset, pck_values in datasets_dict.items():
            benchmark_best_pck[benchmark][dataset] = max(pck_values)
    
    return benchmark_best_pck, sorted(list(all_datasets))


def collect_best_pck_stats_by_benchmark(snapshots_data):
    """
    Collect best PCK stats per benchmark per training dataset.

    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples

    Returns:
        Tuple of (benchmark_stats, all_datasets)
        benchmark_stats: benchmark -> dataset -> stats dict (mean/std/max/n/values)
    """
    benchmark_data = defaultdict(lambda: defaultdict(list))
    all_datasets = set()

    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            continue

        normalized_dataset = normalize_dataset_name_for_grouping(training_dataset_label)
        all_datasets.add(normalized_dataset)

        for benchmark, best_pck in best_performance.items():
            benchmark_data[benchmark][normalized_dataset].append(best_pck)

    benchmark_stats = {}
    for benchmark, datasets_dict in benchmark_data.items():
        benchmark_stats[benchmark] = {}
        for dataset, pck_values in datasets_dict.items():
            values = np.array(pck_values, dtype=float)
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            benchmark_stats[benchmark][dataset] = {
                "values": values,
                "mean": float(np.mean(values)),
                "std": std,
                "max": float(np.max(values)),
                "n": int(len(values)),
            }

    return benchmark_stats, sorted(list(all_datasets))


def get_encoder_regime_label(snapshot_path):
    """
    Parse encoder regime label from snapshot path.

    Returns:
        String label like 'pretrainedTrue_freezeFalse', or 'unknown' if not detected.
    """
    path_lower = str(snapshot_path).lower()
    pretrained = None
    freeze = None

    if "pretrainedtrue" in path_lower:
        pretrained = "pretrainedTrue"
    elif "pretrainedfalse" in path_lower:
        pretrained = "pretrainedFalse"

    if "freezetrue" in path_lower:
        freeze = "freezeTrue"
    elif "freezefalse" in path_lower:
        freeze = "freezeFalse"

    if pretrained and freeze:
        return f"{pretrained}_{freeze}"
    if pretrained or freeze:
        return f"{pretrained or 'pretrainedUnknown'}_{freeze or 'freezeUnknown'}"
    return "unknown"


def group_snapshots_by_encoder_regime(snapshots_data):
    grouped = defaultdict(list)
    for entry in snapshots_data:
        _, _, _, snapshot_path = entry
        regime = get_encoder_regime_label(snapshot_path)
        grouped[regime].append(entry)
    return grouped


def build_color_map_for_datasets(datasets):
    num_datasets = len(datasets)
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
        if num_datasets > 20:
            colors = list(colors) * ((num_datasets // 20) + 1)
            colors = colors[:num_datasets]
    return {dataset: colors[i] for i, dataset in enumerate(sorted(datasets))}


def create_pck_bar_plots_by_benchmark(
    benchmark_stats,
    output_dir,
    dataset_color_map,
    benchmarks_filter=None,
    value_key="max",
    error_key=None,
    title_prefix="Best PCK",
    filename_suffix="best_pck",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    benchmarks_to_plot = sorted(benchmark_stats.keys())
    if benchmarks_filter:
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]

    for benchmark in benchmarks_to_plot:
        dataset_stats = benchmark_stats.get(benchmark, {})
        if not dataset_stats:
            continue

        datasets = sorted(dataset_stats.keys())
        values = [dataset_stats[ds][value_key] for ds in datasets]
        errors = [dataset_stats[ds][error_key] for ds in datasets] if error_key else None

        fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 0.6), 6))
        x = np.arange(len(datasets))
        colors = [dataset_color_map.get(ds, "gray") for ds in datasets]

        ax.bar(x, values, yerr=errors, capsize=4 if errors else 0,
               color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)

        ax.set_title(f"{title_prefix} - {benchmark.upper()}", fontsize=13, fontweight='bold')
        ax.set_xlabel("Training Dataset", fontsize=11, fontweight='bold')
        ax.set_ylabel("PCK (%)", fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = output_path / f"{benchmark}_{filename_suffix}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved bar plot: {output_file}")
        plt.close()


def write_ranked_outputs_by_benchmark(benchmark_stats, output_dir, prefix):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_rows = []
    avg_rows = []
    for benchmark, dataset_stats in benchmark_stats.items():
        best_sorted = sorted(
            dataset_stats.items(),
            key=lambda kv: kv[1]["max"],
            reverse=True,
        )
        for rank, (dataset, stats) in enumerate(best_sorted, start=1):
            best_rows.append({
                "benchmark": benchmark,
                "train_dataset": dataset,
                "best_pck": stats["max"],
                "n_runs": stats["n"],
                "rank": rank,
            })

        avg_sorted = sorted(
            dataset_stats.items(),
            key=lambda kv: kv[1]["mean"],
            reverse=True,
        )
        for rank, (dataset, stats) in enumerate(avg_sorted, start=1):
            avg_rows.append({
                "benchmark": benchmark,
                "train_dataset": dataset,
                "mean_pck": stats["mean"],
                "std_pck": stats["std"],
                "n_runs": stats["n"],
                "rank": rank,
            })

    if best_rows:
        best_csv = output_path / f"{prefix}_ranked_best_pck_by_benchmark.csv"
        pd.DataFrame(best_rows).to_csv(best_csv, index=False)
        print(f"  Saved rankings: {best_csv}")

        best_txt = output_path / f"{prefix}_ranked_best_pck_by_benchmark.txt"
        with open(best_txt, "w") as f:
            for benchmark in sorted(benchmark_stats.keys()):
                f.write(f"{benchmark}\n")
                rows = [r for r in best_rows if r["benchmark"] == benchmark]
                for row in rows:
                    f.write(
                        f"  {row['rank']:>2}. {row['train_dataset']}: {row['best_pck']:.2f}% (n={row['n_runs']})\n"
                    )
                f.write("\n")
        print(f"  Saved rankings: {best_txt}")

    if avg_rows:
        avg_csv = output_path / f"{prefix}_ranked_avg_pck_by_benchmark.csv"
        pd.DataFrame(avg_rows).to_csv(avg_csv, index=False)
        print(f"  Saved rankings: {avg_csv}")

        avg_txt = output_path / f"{prefix}_ranked_avg_pck_by_benchmark.txt"
        with open(avg_txt, "w") as f:
            for benchmark in sorted(benchmark_stats.keys()):
                f.write(f"{benchmark}\n")
                rows = [r for r in avg_rows if r["benchmark"] == benchmark]
                for row in rows:
                    f.write(
                        f"  {row['rank']:>2}. {row['train_dataset']}: "
                        f"{row['mean_pck']:.2f}% ± {row['std_pck']:.2f} (n={row['n_runs']})\n"
                    )
                f.write("\n")
        print(f"  Saved rankings: {avg_txt}")


def extract_encoder_regime_from_snapshots(snapshots_data):
    """
    Extract encoder regime (pretrained/freeze) from snapshots.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        
    Returns:
        Tuple of (pretrained_status, freeze_status) or (None, None) if cannot determine or mixed
        pretrained_status: 'pretrained' or 'not_pretrained' or None
        freeze_status: 'frozen' or 'unfrozen' or None
    """
    regimes = set()
    
    for _, _, _, snapshot_path in snapshots_data:
        path_str = str(snapshot_path)
        path_lower = path_str.lower()
        
        # Extract pretrained and freeze status
        pretrained = None
        freeze = None
        
        if 'pretrainedtrue' in path_lower:
            pretrained = 'pretrained'
        elif 'pretrainedfalse' in path_lower:
            pretrained = 'not_pretrained'
        
        if 'freezetrue' in path_lower:
            freeze = 'frozen'
        elif 'freezefalse' in path_lower:
            freeze = 'unfrozen'
        
        if pretrained and freeze:
            regimes.add((pretrained, freeze))
    
    # If all snapshots have the same regime, return it
    if len(regimes) == 1:
        return regimes.pop()
    else:
        # Mixed regimes or cannot determine
        return None, None


def create_best_pck_bar_plot(snapshots_data, output_dir, benchmarks_filter=None, dataset_color_map=None):
    """
    Create a bar plot showing best PCK per benchmark for each training dataset.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        output_dir: Output directory path
        benchmarks_filter: Optional list of benchmarks to plot (None = all)
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect best PCK data
    benchmark_best_pck, all_datasets = collect_best_pck_by_benchmark(snapshots_data)
    
    if not benchmark_best_pck:
        print("  Warning: No best PCK data found for bar plot")
        return
    
    # Filter benchmarks if specified
    benchmarks_to_plot = sorted(benchmark_best_pck.keys())
    if benchmarks_filter:
        benchmarks_to_plot = [b for b in benchmarks_to_plot if b in benchmarks_filter]
    
    if not benchmarks_to_plot:
        print("  Warning: No benchmarks to plot after filtering")
        return
    
    # Create normalized color map
    # If a color map was provided, we need to map normalized names to colors
    # by finding the first original dataset name that normalizes to each normalized name
    normalized_color_map = {}
    
    if dataset_color_map is not None:
        # Build mapping from normalized names to colors
        for original_name, color in dataset_color_map.items():
            normalized_name = normalize_dataset_name_for_grouping(original_name)
            # Only set if not already set (first match wins)
            if normalized_name not in normalized_color_map:
                normalized_color_map[normalized_name] = color
    
    # Fill in any missing colors for normalized datasets
    num_datasets = len(all_datasets)
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
        if num_datasets > 20:
            colors = list(colors) * ((num_datasets // 20) + 1)
            colors = colors[:num_datasets]
    
    # Assign colors to datasets that don't have them yet
    for i, dataset in enumerate(all_datasets):
        if dataset not in normalized_color_map:
            normalized_color_map[dataset] = colors[i % len(colors)]
    
    # Use the normalized color map
    dataset_color_map = normalized_color_map
    
    # Prepare data for plotting
    # Create a matrix: rows = datasets, columns = benchmarks
    plot_data = []
    dataset_labels = []
    
    for dataset in all_datasets:
        row = []
        for benchmark in benchmarks_to_plot:
            pck = benchmark_best_pck[benchmark].get(dataset, None)
            row.append(pck)
        # Only include datasets that have at least one benchmark result
        if any(p is not None for p in row):
            plot_data.append(row)
            dataset_labels.append(dataset)
    
    if not plot_data:
        print("  Warning: No data to plot")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(max(12, len(benchmarks_to_plot) * 1.5), 8))
    
    # Set up bar positions
    x = np.arange(len(benchmarks_to_plot))
    width = 0.8 / len(dataset_labels)  # Bar width
    offset = (len(dataset_labels) - 1) * width / 2
    
    # Plot bars for each dataset
    for i, (dataset, row_data) in enumerate(zip(dataset_labels, plot_data)):
        positions = x - offset + i * width
        color = dataset_color_map.get(dataset, 'gray')
        
        # Create bars, handling None values
        for j, (pos, pck) in enumerate(zip(positions, row_data)):
            if pck is not None:
                # Only add label for first benchmark to avoid duplicate legend entries
                label = dataset if j == 0 else ''
                ax.bar(pos, pck, width, label=label, 
                       color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Extract encoder regime for title
    pretrained, freeze = extract_encoder_regime_from_snapshots(snapshots_data)
    title = 'Best PCK by Benchmark and Training Dataset'
    if pretrained and freeze:
        # Format regime for display
        pretrained_display = 'Pretrained' if pretrained == 'pretrained' else 'Not Pretrained'
        freeze_display = 'Frozen' if freeze == 'frozen' else 'Unfrozen'
        title = f'Best PCK by Benchmark and Training Dataset\n({pretrained_display}, {freeze_display})'
    elif pretrained:
        pretrained_display = 'Pretrained' if pretrained == 'pretrained' else 'Not Pretrained'
        title = f'Best PCK by Benchmark and Training Dataset\n({pretrained_display})'
    elif freeze:
        freeze_display = 'Frozen' if freeze == 'frozen' else 'Unfrozen'
        title = f'Best PCK by Benchmark and Training Dataset\n({freeze_display})'
    
    # Formatting
    ax.set_xlabel('Benchmark', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best PCK (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.upper() for b in benchmarks_to_plot], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_file = output_path / 'best_pck_by_benchmark_barplot.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved bar plot: {output_file}")
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
        '--coverage-csv',
        type=str,
        default='coverage_results.csv',
        help='Coverage CSV to load (default: coverage_results.csv)'
    )
    parser.add_argument(
        '--coverage-representation',
        type=str,
        default=None,
        help='If provided, filter coverage rows by representation (e.g., flow, resnet). '
             'If omitted, the script will attempt to load both flow and resnet coverage CSVs if present.'
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

    output_root = Path(args.output_dir)
    overview_dir = output_root / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    
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
        str(overview_dir),
        benchmarks_filter=args.benchmarks,
        metrics_filter=metrics_filter,
        dataset_color_map=dataset_color_map
    )
    
    # Create PCK vs training steps plots for each benchmark
    print("\nCreating PCK vs Training Steps plots...")
    benchmark_pck_data = organize_pck_vs_steps_by_benchmark(snapshots_data)
    create_pck_vs_steps_plots(
        benchmark_pck_data,
        str(overview_dir),
        benchmarks_filter=args.benchmarks,
        dataset_color_map=dataset_color_map
    )
    
    # Create best PCK bar plot
    print("\nCreating Best PCK Bar Plot...")
    create_best_pck_bar_plot(
        snapshots_data,
        str(overview_dir),
        benchmarks_filter=args.benchmarks,
        dataset_color_map=dataset_color_map
    )

    # Create per-encoder regime bar plots and rankings
    print("\nCreating Per-Encoder Regime Bar Plots and Rankings...")
    snapshots_by_regime = group_snapshots_by_encoder_regime(snapshots_data)
    for regime, regime_snapshots in sorted(snapshots_by_regime.items()):
        if not regime_snapshots:
            continue
        print(f"  Encoder regime: {regime} ({len(regime_snapshots)} snapshots)")

        benchmark_stats, datasets = collect_best_pck_stats_by_benchmark(regime_snapshots)
        if not benchmark_stats:
            print(f"    Warning: No benchmark stats found for regime {regime}")
            continue

        regime_output = Path(args.output_dir) / "by_encoder" / regime
        barplot_dir = regime_output / "barplots"
        best_dir = barplot_dir / "best_pck"
        avg_dir = barplot_dir / "avg_pck"
        rankings_dir = regime_output / "rankings"

        regime_color_map = build_color_map_for_datasets(datasets)
        create_pck_bar_plots_by_benchmark(
            benchmark_stats,
            best_dir,
            regime_color_map,
            benchmarks_filter=args.benchmarks,
            value_key="max",
            title_prefix="Best PCK",
            filename_suffix="best_pck"
        )
        create_pck_bar_plots_by_benchmark(
            benchmark_stats,
            avg_dir,
            regime_color_map,
            benchmarks_filter=args.benchmarks,
            value_key="mean",
            error_key="std",
            title_prefix="Average Best PCK",
            filename_suffix="avg_pck"
        )
        write_ranked_outputs_by_benchmark(benchmark_stats, rankings_dir, prefix=regime)
    
    # Load MMD lookup and create scatter plot
    print("\nCreating Flow MMD vs PCK scatter plot...")
    mmd_lookup = load_mmd_lookup('flow_mmd_results.csv')
    if mmd_lookup:
        create_mmd_vs_pck_scatter_plot(
            snapshots_data,
            mmd_lookup,
            str(overview_dir),
            dataset_color_map
        )
        # Create error bar plot version
        print("\nCreating Flow MMD vs PCK error bar plot (grouped by training dataset)...")
        create_mmd_vs_pck_errorbar_plot(
            snapshots_data,
            mmd_lookup,
            str(overview_dir),
            dataset_color_map,
            mmd_type="Flow"
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
            str(overview_dir),
            dataset_color_map
        )
        # Create error bar plot version
        print("\nCreating Feature MMD vs PCK error bar plot (grouped by training dataset)...")
        create_mmd_vs_pck_errorbar_plot(
            snapshots_data,
            feature_mmd_lookup,
            str(overview_dir),
            dataset_color_map,
            mmd_type="Feature"
        )
    else:
        print("  Skipping Feature MMD vs PCK scatter plot (Feature MMD lookup not available)")
    
    # Load coverage lookup and create scatter plots
    print("\nCreating Coverage vs PCK scatter plots...")
    coverage_sources = []
    # Always attempt flow and resnet defaults if present
    for path, rep in [('coverage_results.csv', 'flow'), ('coverage_resnet_results.csv', 'resnet')]:
        if os.path.exists(path):
            coverage_sources.append((path, rep))
    # Include user-provided CSV if different
    if args.coverage_csv and os.path.exists(args.coverage_csv) and args.coverage_csv not in [c[0] for c in coverage_sources]:
        coverage_sources.append((args.coverage_csv, args.coverage_representation))

    if not coverage_sources:
        print("  Skipping Coverage vs PCK scatter plots (no coverage CSVs found)")
    else:
        for cov_path, cov_rep in coverage_sources:
            print(f"\nCreating Coverage vs PCK scatter plots from {cov_path} (rep={cov_rep})...")
            coverage_lookup = load_coverage_lookup(cov_path, representation_filter=cov_rep)
            if not coverage_lookup:
                print(f"  Skipping {cov_path}: coverage lookup empty")
                continue
            tag = cov_rep or Path(cov_path).stem

            # Recall plots
            print("\nCreating Recall vs PCK scatter plot...")
            create_coverage_vs_pck_scatter_plot(
                snapshots_data,
                coverage_lookup,
                str(overview_dir),
                dataset_color_map,
                score_key='recall',
                score_label='Recall',
                coverage_tag=tag
            )
            print("\nCreating Recall vs PCK error bar plot (grouped by training dataset)...")
            create_coverage_vs_pck_errorbar_plot(
                snapshots_data,
                coverage_lookup,
                str(overview_dir),
                dataset_color_map,
                score_key='recall',
                score_label='Recall',
                coverage_tag=tag
            )

            # Precision plots
            print("\nCreating Precision vs PCK scatter plot...")
            create_coverage_vs_pck_scatter_plot(
                snapshots_data,
                coverage_lookup,
                str(overview_dir),
                dataset_color_map,
                score_key='precision',
                score_label='Precision',
                coverage_tag=tag
            )
            print("\nCreating Precision vs PCK error bar plot (grouped by training dataset)...")
            create_coverage_vs_pck_errorbar_plot(
                snapshots_data,
                coverage_lookup,
                str(overview_dir),
                dataset_color_map,
                score_key='precision',
                score_label='Precision',
                coverage_tag=tag
            )

    print("\nDone!")


if __name__ == '__main__':
    main()
