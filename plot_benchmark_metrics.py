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


def compute_zscore_correlation(df):
    """
    Compute correlation using z-score normalized values within each benchmark.
    This removes baseline difficulty differences between benchmarks.
    
    Args:
        df: DataFrame with columns ['mmd2', 'best_pck', 'benchmark']
        
    Returns:
        Tuple of (correlation, p_value, df_with_zscores)
    """
    df = df.copy()
    
    # Z-score normalize within each benchmark
    def zscore(x):
        if x.std() > 0:
            return (x - x.mean()) / x.std()
        return x * 0  # Return zeros if no variance
    
    df['pck_z'] = df.groupby('benchmark')['best_pck'].transform(zscore)
    df['mmd_z'] = df.groupby('benchmark')['mmd2'].transform(zscore)
    
    # Remove any NaN values
    df_clean = df.dropna(subset=['pck_z', 'mmd_z'])
    
    if len(df_clean) >= 3:
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
    print(f"   (Standardizes within each benchmark to remove difficulty differences)")
    r_z, p_z, df_z = compute_zscore_correlation(df)
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
    Create scatter plot using z-score normalized values.
    
    Args:
        df: DataFrame with analysis data
        output_path: Path object for output directory
        analysis_name: Name for plot title and filename
        dataset_color_map: Dictionary mapping training_dataset -> color
    """
    r_z, p_z, df_z = compute_zscore_correlation(df)
    
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
    
    ax.set_xlabel(f'{analysis_name}² (z-scored within benchmark)', fontsize=12)
    ax.set_ylabel('Best PCK (z-scored within benchmark)', fontsize=12)
    ax.set_title(f'Z-Score Normalized {analysis_name}² vs PCK\n(Controls for benchmark difficulty)', 
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


def load_coverage_lookup(csv_path='coverage_results.csv'):
    """
    Load coverage data from CSV file and create a bidirectional lookup.
    
    Args:
        csv_path: Path to coverage_results.csv file
        
    Returns:
        Dictionary mapping (dataset1_split1, dataset2_split2) -> dict of coverage metrics
        Works for both orderings
    """
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found. Cannot create coverage lookup.")
        return {}
    
    coverage_lookup = {}
    
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            dataset1 = str(row['dataset1']).lower()
            dataset2 = str(row['dataset2']).lower()
            split1 = str(row['split1']).lower()
            split2 = str(row['split2']).lower()
            
            # Skip identical comparisons (same dataset AND same split)
            if dataset1 == dataset2 and split1 == split2:
                continue
            
            # Create unique identifiers with splits
            dataset1_id = f"{dataset1}_{split1}"
            dataset2_id = f"{dataset2}_{split2}"
            
            # Store coverage metrics
            metrics = {
                'coverage_abs': float(row['coverage_abs']),
                'coverage_rel': float(row['coverage_rel']),
                'rho_95': float(row['rho_95']),
                'rho_median': float(row['rho_median']),
                'rho_mean': float(row['rho_mean']),
                'epsilon': float(row['epsilon']),
            }
            
            # Store both orderings with split identifiers
            coverage_lookup[(dataset1_id, dataset2_id)] = metrics
            coverage_lookup[(dataset2_id, dataset1_id)] = metrics
            
            # Also store without explicit split in key for backward compatibility
            coverage_lookup[(dataset1, dataset2)] = metrics
            coverage_lookup[(dataset2, dataset1)] = metrics
            
    except Exception as e:
        print(f"Warning: Could not load coverage lookup from {csv_path}: {e}")
        return {}
    
    return coverage_lookup


def create_coverage_vs_pck_scatter_plot(snapshots_data, coverage_lookup, output_dir, dataset_color_map, coverage_type='abs'):
    """
    Create scatter plot showing training dataset coverage vs best PCK.
    
    Args:
        snapshots_data: List of (training_dataset, validation_data_dict, metrics_list, snapshot_path) tuples
        coverage_lookup: Dictionary mapping (dataset1_split1, dataset2_split2) -> coverage metrics dict
        output_dir: Output directory path
        dataset_color_map: Dictionary mapping training_dataset -> color
        coverage_type: 'abs' for absolute coverage or 'rel' for relative coverage
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which coverage metric to use
    coverage_key = 'coverage_abs' if coverage_type == 'abs' else 'coverage_rel'
    coverage_label = 'Absolute Coverage' if coverage_type == 'abs' else 'Relative Coverage'
    
    # Collect all data points: (coverage, best_pck, training_dataset, benchmark)
    data_points = []
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Get base training dataset name from summary (for coverage lookup)
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            print(f"  Warning: Could not parse training dataset from {summary_path}")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            print(f"  Warning: No best performance data found in {summary_path}")
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
            
            if coverage_metrics is not None:
                data_points.append({
                    'coverage': coverage_metrics[coverage_key],
                    'coverage_abs': coverage_metrics['coverage_abs'],
                    'coverage_rel': coverage_metrics['coverage_rel'],
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
    coverage_suffix = 'abs' if coverage_type == 'abs' else 'rel'
    output_file = output_path / f'training_coverage_{coverage_suffix}_vs_best_pck.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved {coverage_label} vs PCK scatter plot: {output_file}")
    plt.close()
    
    # Create DataFrame for statistical analysis
    df = pd.DataFrame(data_points)
    # Rename for compatibility with existing analysis functions
    df['mmd2'] = df['coverage']
    
    # Run and print statistical analysis
    analysis_name = f"Coverage ({coverage_type.upper()})"
    print_statistical_analysis(df, analysis_name)
    
    # Save statistical analysis to file
    save_statistical_analysis_to_file(df, output_path, analysis_name)
    
    # Create faceted plot (one panel per benchmark)
    create_faceted_scatter_plot(df, output_path, analysis_name, dataset_color_map)
    
    # Create z-score normalized plot
    create_zscore_scatter_plot(df, output_path, analysis_name, dataset_color_map)


def create_coverage_vs_pck_errorbar_plot(snapshots_data, coverage_lookup, output_dir, dataset_color_map, coverage_type='abs'):
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
        coverage_type: 'abs' for absolute coverage or 'rel' for relative coverage
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine which coverage metric to use
    coverage_key = 'coverage_abs' if coverage_type == 'abs' else 'coverage_rel'
    coverage_label = 'Absolute Coverage' if coverage_type == 'abs' else 'Relative Coverage'
    
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
            
            if coverage_metrics is not None:
                key = (training_dataset_label, benchmark)
                grouped_data[key].append({
                    'coverage': coverage_metrics[coverage_key],
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
    coverage_suffix = 'abs' if coverage_type == 'abs' else 'rel'
    output_file = output_path / f'training_coverage_{coverage_suffix}_vs_best_pck_errorbars.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved {coverage_label} error bar plot: {output_file}")
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
        # Create error bar plot version
        print("\nCreating Flow MMD vs PCK error bar plot (grouped by training dataset)...")
        create_mmd_vs_pck_errorbar_plot(
            snapshots_data,
            mmd_lookup,
            args.output_dir,
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
            args.output_dir,
            dataset_color_map
        )
        # Create error bar plot version
        print("\nCreating Feature MMD vs PCK error bar plot (grouped by training dataset)...")
        create_mmd_vs_pck_errorbar_plot(
            snapshots_data,
            feature_mmd_lookup,
            args.output_dir,
            dataset_color_map,
            mmd_type="Feature"
        )
    else:
        print("  Skipping Feature MMD vs PCK scatter plot (Feature MMD lookup not available)")
    
    # Load coverage lookup and create scatter plots
    print("\nCreating Coverage vs PCK scatter plots...")
    coverage_lookup = load_coverage_lookup('coverage_results.csv')
    if coverage_lookup:
        # Create plots for absolute coverage
        print("\nCreating Absolute Coverage vs PCK scatter plot...")
        create_coverage_vs_pck_scatter_plot(
            snapshots_data,
            coverage_lookup,
            args.output_dir,
            dataset_color_map,
            coverage_type='abs'
        )
        # Create error bar plot version
        print("\nCreating Absolute Coverage vs PCK error bar plot (grouped by training dataset)...")
        create_coverage_vs_pck_errorbar_plot(
            snapshots_data,
            coverage_lookup,
            args.output_dir,
            dataset_color_map,
            coverage_type='abs'
        )
        
        # Create plots for relative coverage
        print("\nCreating Relative Coverage vs PCK scatter plot...")
        create_coverage_vs_pck_scatter_plot(
            snapshots_data,
            coverage_lookup,
            args.output_dir,
            dataset_color_map,
            coverage_type='rel'
        )
        # Create error bar plot version
        print("\nCreating Relative Coverage vs PCK error bar plot (grouped by training dataset)...")
        create_coverage_vs_pck_errorbar_plot(
            snapshots_data,
            coverage_lookup,
            args.output_dir,
            dataset_color_map,
            coverage_type='rel'
        )
    else:
        print("  Skipping Coverage vs PCK scatter plots (coverage lookup not available)")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
