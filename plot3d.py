#!/usr/bin/env python3
"""
3D visualization script for coverage metrics vs PCK performance.

Creates 3D scatter plots showing:
- x-axis: Feature train->eval coverage (ResNet or DINO)
- y-axis: Flow train->eval coverage
- z-axis: PCK performance (raw and z-scored)

Also creates 2D color-mapped versions for easier interpretation.

Usage:
    python plot3d.py --snapshots_dir snapshots/ \
      --coverage-csv coverage_results.csv \
      --coverage-resnet-csv coverage_resnet_results.csv \
      --output-dir plots3d/
"""

import argparse
import csv
import os
import sys
from typing import List, Optional, Tuple
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

MODEL_FAMILY_DEFAULT = "catspp"
MODEL_FAMILY_ALIASES = {
    "mixed": MODEL_FAMILY_DEFAULT,
    "mixed_plots": MODEL_FAMILY_DEFAULT,
    "cats": MODEL_FAMILY_DEFAULT,
    "catspp": MODEL_FAMILY_DEFAULT,
}

STANDARDIZE_MODES = ("global", "none", "benchmark", "encoder", "model_family")
PREDICTOR_SETS = ("all", "trimmed", "asymmetric", "mmd")

# NOTE ON MEAN_DIST NORMALIZATION:
# - All *_mean_dist metrics are computed on L2-normalized feature/flow vectors
# - The distances themselves are RAW L2 distances (not normalized by radius/median)
# - train_to_eval_mean_dist: mean distance from train samples to nearest eval neighbors
# - eval_to_train_mean_dist: mean distance from eval samples to nearest train neighbors
# - These asymmetric metrics capture directional distribution mismatch
# - MMD metrics are symmetric and may collapse distinct failure modes

ALL_PREDICTORS = [
    "flow_mmd",
    "feature_mmd",
    "dino_mmd",
    "flow_train_to_eval_coverage",
    "flow_eval_to_train_coverage",
    "resnet_train_to_eval_coverage",
    "resnet_eval_to_train_coverage",
    "dino_train_to_eval_coverage",
    "dino_eval_to_train_coverage",
    "flow_train_to_eval_mean_dist",
    "flow_eval_to_train_mean_dist",
    "resnet_train_to_eval_mean_dist",
    "resnet_eval_to_train_mean_dist",
    "dino_train_to_eval_mean_dist",
    "dino_eval_to_train_mean_dist",
]

TRIMMED_PREDICTORS = [
    "flow_train_to_eval_mean_dist",
    "flow_eval_to_train_mean_dist",
    "resnet_train_to_eval_mean_dist",
    "resnet_eval_to_train_mean_dist",
    "dino_train_to_eval_mean_dist",
    "dino_eval_to_train_mean_dist",
]

# Asymmetric directional metrics only (flow + DINO mean distances)
# Tests hypothesis: asymmetric metrics capture distinct directional failure modes
ASYMMETRIC_PREDICTORS = [
    "flow_train_to_eval_mean_dist",
    "flow_eval_to_train_mean_dist",
    "dino_train_to_eval_mean_dist",
    "dino_eval_to_train_mean_dist",
]

# Symmetric MMD metrics only (flow + DINO)
# Tests hypothesis: symmetric metrics collapse directional information
MMD_PREDICTORS = [
    "flow_mmd",
    "dino_mmd",
]


def derive_model_family(snapshot_path: Path) -> str:
    for part in snapshot_path.parts:
        if part == "snapshots":
            return MODEL_FAMILY_DEFAULT
        if part.startswith("snapshots_"):
            suffix = part.split("snapshots_", 1)[1].strip().lower()
            if not suffix:
                return MODEL_FAMILY_DEFAULT
            return MODEL_FAMILY_ALIASES.get(suffix, suffix)
    name = snapshot_path.name.lower()
    if "raft" in name:
        return "raft"
    if "flowformer" in name:
        return "flowformer"
    return MODEL_FAMILY_DEFAULT


def select_predictor_candidates(predictor_set: str) -> List[str]:
    """
    Select predictor variables based on the specified set.
    
    Args:
        predictor_set: One of 'all', 'trimmed', 'asymmetric', 'mmd'
        
    Returns:
        List of predictor column names
    """
    if predictor_set == "trimmed":
        return list(TRIMMED_PREDICTORS)
    elif predictor_set == "asymmetric":
        return list(ASYMMETRIC_PREDICTORS)
    elif predictor_set == "mmd":
        return list(MMD_PREDICTORS)
    else:  # 'all' or default
        return list(ALL_PREDICTORS)


def _standardize_predictors_insample(
    df: pd.DataFrame,
    predictors: List[str],
    mode: str,
    min_std: float = 0.0,
    group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    if not predictors:
        return df, []

    if mode == "none":
        standardized_cols = []
        for col in predictors:
            z_col = f"{col}_std"
            df[z_col] = df[col]
            standardized_cols.append(z_col)
        return df, standardized_cols

    means = df[predictors].mean()
    stds = df[predictors].std(ddof=0).replace(0, 1.0)
    if min_std > 0:
        stds = stds.where(stds >= float(min_std), float(min_std))

    use_group = mode in ("benchmark", "encoder", "model_family") and group_col
    if use_group and group_col not in df.columns:
        print(
            f"Warning: standardize_mode={mode} requested but '{group_col}' column is missing. "
            "Falling back to global standardization."
        )
        use_group = False

    if use_group:
        group_means = df.groupby(group_col)[predictors].transform("mean")
        group_stds = (
            df.groupby(group_col)[predictors].transform(lambda x: x.std(ddof=0)).replace(0, 1.0)
        )
        group_means = group_means.fillna(means)
        group_stds = group_stds.fillna(stds).replace(0, 1.0)

    standardized_cols = []
    for col in predictors:
        z_col = f"{col}_std"
        if use_group:
            df[z_col] = (df[col] - group_means[col]) / group_stds[col]
        else:
            df[z_col] = (df[col] - means[col]) / stds[col]
        standardized_cols.append(z_col)
    return df, standardized_cols


def load_coverage_lookup(csv_path='coverage_results.csv'):
    """
    Load coverage metrics from CSV file.

    Supports both single datasets (e.g., 'spair', 'synthetic') and mixed datasets
    (e.g., 'spair_synthetic_50_50', 'spair_synthetic_70_30').

    Args:
        csv_path: Path to coverage CSV file

    Returns:
        Dictionary mapping (train_dataset_split, eval_dataset_split) -> coverage_metrics dict
        Also includes mappings without explicit splits for backward compatibility
    """
    coverage_lookup = {}

    if not os.path.exists(csv_path):
        print(f"Warning: Coverage CSV not found: {csv_path}")
        return coverage_lookup

    def _parse_float(value):
        if value is None:
            return np.nan
        if isinstance(value, float):
            return value
        try:
            value = str(value).strip()
        except Exception:
            return np.nan
        if not value:
            return np.nan
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse dataset names and splits (normalize to lowercase for consistent matching)
                # Handles both single datasets (e.g., 'spair') and mixed datasets (e.g., 'spair_synthetic_50_50')
                train_dataset = str(row.get('dataset1', '')).strip().lower()
                train_split = str(row.get('split1', '')).strip().lower()
                eval_dataset = str(row.get('dataset2', '')).strip().lower()
                eval_split = str(row.get('split2', '')).strip().lower()

                # Skip rows with empty dataset names
                if not train_dataset or not eval_dataset:
                    continue

                train_id = f"{train_dataset}_{train_split}" if train_split else train_dataset
                eval_id = f"{eval_dataset}_{eval_split}" if eval_split else eval_dataset

                train_to_eval = _parse_float(row.get('train_to_eval_coverage'))
                eval_to_train = _parse_float(row.get('eval_to_train_coverage'))
                recall_val = _parse_float(row.get('recall'))
                precision_val = _parse_float(row.get('precision'))
                outside_val = _parse_float(row.get('outside'))

                if pd.isna(train_to_eval):
                    train_to_eval = recall_val
                if pd.isna(eval_to_train):
                    eval_to_train = precision_val

                train_to_eval_mean = _parse_float(row.get('mean_nn_train_to_eval'))
                eval_to_train_mean = _parse_float(row.get('mean_nn_eval_to_train'))
                train_to_eval_median = _parse_float(row.get('median_nn_train_to_eval'))
                eval_to_train_median = _parse_float(row.get('median_nn_eval_to_train'))
                train_to_eval_p90 = _parse_float(row.get('p90_nn_train_to_eval'))
                eval_to_train_p90 = _parse_float(row.get('p90_nn_eval_to_train'))

                if pd.isna(train_to_eval) and pd.isna(eval_to_train):
                    continue

                metrics = {
                    'train_to_eval_coverage': train_to_eval,
                    'eval_to_train_coverage': eval_to_train,
                    'train_to_eval_mean_dist': train_to_eval_mean,
                    'eval_to_train_mean_dist': eval_to_train_mean,
                    'train_to_eval_median_dist': train_to_eval_median,
                    'eval_to_train_median_dist': eval_to_train_median,
                    'train_to_eval_p90_dist': train_to_eval_p90,
                    'eval_to_train_p90_dist': eval_to_train_p90,
                    'recall': train_to_eval if not pd.isna(train_to_eval) else recall_val,
                    'precision': eval_to_train if not pd.isna(eval_to_train) else precision_val,
                    'outside': outside_val,
                }

                coverage_lookup[(train_id, eval_id)] = metrics
                # Also store without explicit split for backward compatibility
                coverage_lookup[(train_dataset, eval_dataset)] = metrics
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




def collect_3d_data_points(
    snapshots_data,
    flow_coverage_lookup,
    feature_coverage_lookup,
    feature_label="resnet",
    debug=False,
):
    """
    Collect data points for 3D plotting.
    
    Returns:
        List of dicts with keys: 'feature_train_to_eval_coverage', 'flow_train_to_eval_coverage',
        'pck', 'training_dataset', 'benchmark', 'snapshot_path'
    """
    data_points = []
    missing_flow = defaultdict(int)
    missing_feature = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Skip if summary file doesn't exist (shouldn't happen since we filter above, but be safe)
        if not summary_path.exists():
            if debug:
                print(f"  Skipping {snapshot_path}: training_summary.txt not found")
            continue
        
        # Get base training dataset name
        # parse_training_dataset_from_summary already handles mixed datasets by converting '+' to '_'
        # (e.g., "spair+synthetic" -> "spair_synthetic")
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Skipping {snapshot_path}: Could not parse training dataset from summary")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            if debug:
                print(f"  Skipping {snapshot_path}: No best performance data found in summary")
            continue
        
        # For each benchmark, look up both flow and feature coverage
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up coverage with splits
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Get flow train->eval coverage
            flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
            flow_key_used = (training_dataset_train, benchmark_test)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
                flow_key_used = (training_dataset_train, benchmark_val)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((base_training_dataset, benchmark_lower))
                flow_key_used = (base_training_dataset, benchmark_lower)
            
            # Get feature train->eval coverage
            feature_metrics = feature_coverage_lookup.get((training_dataset_train, benchmark_test))
            feature_key_used = (training_dataset_train, benchmark_test)
            if feature_metrics is None:
                feature_metrics = feature_coverage_lookup.get((training_dataset_train, benchmark_val))
                feature_key_used = (training_dataset_train, benchmark_val)
            if feature_metrics is None:
                feature_metrics = feature_coverage_lookup.get((base_training_dataset, benchmark_lower))
                feature_key_used = (base_training_dataset, benchmark_lower)
            
            # Track missing metrics for summary (no per-item warnings)
            if debug:
                if not flow_metrics:
                    missing_flow[flow_key_used] += 1
                if not feature_metrics:
                    missing_feature[feature_key_used] += 1
            
            flow_train_to_eval = (
                flow_metrics.get('train_to_eval_coverage', flow_metrics.get('recall'))
                if flow_metrics
                else np.nan
            )
            feature_train_to_eval = (
                feature_metrics.get('train_to_eval_coverage', feature_metrics.get('recall'))
                if feature_metrics
                else np.nan
            )

            # Only add if we have both coverage values
            if (not pd.isna(flow_train_to_eval) and not pd.isna(feature_train_to_eval)):
                data_points.append({
                    'feature_train_to_eval_coverage': feature_train_to_eval,
                    'flow_train_to_eval_coverage': flow_train_to_eval,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
    
    if debug and (missing_flow or missing_feature):
        print(f"\nDebug: Missing flow train->eval coverage keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print(f"\nDebug: Missing {feature_label} train->eval coverage keys (top 10):")
        for key, count in sorted(missing_feature.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        
        # Show sample of available keys
        print(f"\nDebug: Sample of available flow coverage keys (first 5):")
        for key in list(flow_coverage_lookup.keys())[:5]:
            print(f"  {key}")
        print(f"\nDebug: Sample of available {feature_label} coverage keys (first 5):")
        for key in list(feature_coverage_lookup.keys())[:5]:
            print(f"  {key}")
    
    return data_points


def create_3d_scatter_plot(data_points, output_path, dataset_color_map, feature_label="feature", zscore=False):
    """Create 3D scatter plot"""
    if not data_points:
        print("Warning: No data points for 3D plot")
        return
    
    # Extract data
    feature_cov = [p['feature_train_to_eval_coverage'] for p in data_points]
    flow_cov = [p['flow_train_to_eval_coverage'] for p in data_points]
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
        feature_vals = [p['feature_train_to_eval_coverage'] for p in points]
        flow_vals = [p['flow_train_to_eval_coverage'] for p in points]
        pck_vals = [p['pck'] for p in points]
        
        if zscore:
            # Recompute z-score for this dataset's points
            pck_vals = stats.zscore(pck_vals)
        
        color = dataset_color_map.get(training_dataset, 'black')
        ax.scatter(feature_vals, flow_vals, pck_vals,
                  color=color, label=training_dataset,
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Feature Train->Eval Coverage', fontsize=12, labelpad=10)
    ax.set_ylabel('Flow Train->Eval Coverage', fontsize=12, labelpad=10)
    ax.set_zlabel(z_label, fontsize=12, labelpad=10)
    ax.set_title(
        f'3D: {feature_label} Coverage vs Flow Coverage vs PCK',
        fontsize=14,
        fontweight='bold',
        pad=20,
    )
    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(0, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    suffix = '_zscore' if zscore else '_raw'
    safe_label = feature_label.lower().replace(' ', '_')
    output_file = output_path / f'3d_{safe_label}_flow_pck{suffix}.png'
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


def create_faceted_by_benchmark_plot(
    data_points,
    output_path,
    feature_label="feature",
    zscore_by_benchmark=False,
):
    """
    Create faceted 2D colormap plots with one panel per benchmark.
    Each panel shows feature vs flow train->eval coverage colored by PCK.
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
        scatter = ax.scatter(subset['feature_train_to_eval_coverage'], subset['flow_train_to_eval_coverage'],
                           c=subset['pck_plot'], s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Add training dataset labels
        for _, row in subset.iterrows():
            ax.annotate(row['training_dataset'],
                       (row['feature_train_to_eval_coverage'], row['flow_train_to_eval_coverage']),
                       fontsize=6, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
        
        ax.set_xlabel('Feature Train->Eval Coverage', fontsize=10)
        ax.set_ylabel('Flow Train->Eval Coverage', fontsize=10)
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
    fig.suptitle(f'{feature_label} Coverage vs Flow Coverage{title_suffix} - By Benchmark',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    
    # Save
    safe_label = feature_label.lower().replace(' ', '_')
    output_file = output_path / f'2d_{safe_label}_flow_pck_by_benchmark{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved faceted by benchmark plot: {output_file}")
    plt.close()


def create_faceted_by_training_set_plot(
    data_points,
    output_path,
    dataset_color_map,
    feature_label="feature",
    zscore_by_benchmark=False,
):
    """
    Create faceted 2D colormap plots with one panel per training dataset.
    Each panel shows feature vs flow train->eval coverage colored by PCK.
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
        scatter = ax.scatter(subset['feature_train_to_eval_coverage'], subset['flow_train_to_eval_coverage'],
                           c=subset['pck_plot'], s=100, alpha=0.7,
                           edgecolors='black', linewidth=0.5,
                           cmap='viridis', vmin=vmin, vmax=vmax)
        
        # Add benchmark labels
        for _, row in subset.iterrows():
            ax.annotate(row['benchmark'],
                       (row['feature_train_to_eval_coverage'], row['flow_train_to_eval_coverage']),
                       fontsize=6, alpha=0.7,
                       xytext=(3, 3), textcoords='offset points')
        
        ax.set_xlabel('Feature Train->Eval Coverage', fontsize=10)
        ax.set_ylabel('Flow Train->Eval Coverage', fontsize=10)
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
    fig.suptitle(f'{feature_label} Coverage vs Flow Coverage{title_suffix} - By Training Dataset',
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.98])
    
    # Save
    safe_label = feature_label.lower().replace(' ', '_')
    output_file = output_path / f'2d_{safe_label}_flow_pck_by_training_set{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved faceted by training set plot: {output_file}")
    plt.close()


def create_2d_colormap_plot(
    data_points,
    output_path,
    dataset_color_map,
    feature_label="feature",
    zscore=False,
    zscore_by_benchmark=False,
):
    """Create 2D scatter plot with PCK as color"""
    if not data_points:
        print("Warning: No data points for 2D colormap plot")
        return
    
    # Extract data
    feature_cov = [p['feature_train_to_eval_coverage'] for p in data_points]
    flow_cov = [p['flow_train_to_eval_coverage'] for p in data_points]
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
    scatter = ax.scatter(feature_cov, flow_cov, c=pck_values,
                        s=150, alpha=0.7, edgecolors='black', linewidth=1,
                        cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12)
    
    # Add benchmark labels
    for point in data_points:
        ax.annotate(point['benchmark'],
                   (point['feature_train_to_eval_coverage'], point['flow_train_to_eval_coverage']),
                   fontsize=7, alpha=0.6,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Feature Train->Eval Coverage', fontsize=12)
    ax.set_ylabel('Flow Train->Eval Coverage', fontsize=12)
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else (' (PCK z-scored)' if zscore else '')
    ax.set_title(f'{feature_label} Coverage vs Flow Coverage{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    safe_label = feature_label.lower().replace(' ', '_')
    output_file = output_path / f'2d_{safe_label}_flow_pck_colormap{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D colormap plot: {output_file}")
    plt.close()


def collect_precision_data_points(
    snapshots_data,
    flow_coverage_lookup,
    feature_coverage_lookup,
    feature_label="resnet",
    debug=False,
):
    """
    Collect data points for precision plotting.
    
    Returns:
        List of dicts with keys: 'feature_eval_to_train_coverage', 'flow_eval_to_train_coverage',
        'pck', 'training_dataset', 'benchmark', 'snapshot_path'
    """
    data_points = []
    missing_flow = defaultdict(int)
    missing_feature = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        summary_path = Path(snapshot_path) / 'training_summary.txt'
        
        # Skip if summary file doesn't exist (shouldn't happen since we filter above, but be safe)
        if not summary_path.exists():
            if debug:
                print(f"  Skipping {snapshot_path}: training_summary.txt not found")
            continue
        
        # Get base training dataset name
        # parse_training_dataset_from_summary already handles mixed datasets by converting '+' to '_'
        # (e.g., "spair+synthetic" -> "spair_synthetic")
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Skipping {snapshot_path}: Could not parse training dataset from summary")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            if debug:
                print(f"  Skipping {snapshot_path}: No best performance data found in summary")
            continue
        
        # For each benchmark, look up both flow and feature eval->train coverage
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try to look up coverage with splits
            training_dataset_train = f"{base_training_dataset}_train"
            benchmark_test = f"{benchmark_lower}_test"
            benchmark_val = f"{benchmark_lower}_val"
            
            # Get flow eval->train coverage
            flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
            flow_key_used = (training_dataset_train, benchmark_test)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
                flow_key_used = (training_dataset_train, benchmark_val)
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((base_training_dataset, benchmark_lower))
                flow_key_used = (base_training_dataset, benchmark_lower)
            
            # Get feature eval->train coverage
            feature_metrics = feature_coverage_lookup.get((training_dataset_train, benchmark_test))
            feature_key_used = (training_dataset_train, benchmark_test)
            if feature_metrics is None:
                feature_metrics = feature_coverage_lookup.get((training_dataset_train, benchmark_val))
                feature_key_used = (training_dataset_train, benchmark_val)
            if feature_metrics is None:
                feature_metrics = feature_coverage_lookup.get((base_training_dataset, benchmark_lower))
                feature_key_used = (base_training_dataset, benchmark_lower)
            
            # Debug output
            if debug:
                flow_eval_to_train = (
                    flow_metrics.get('eval_to_train_coverage', flow_metrics.get('precision'))
                    if flow_metrics
                    else np.nan
                )
                feature_eval_to_train = (
                    feature_metrics.get('eval_to_train_coverage', feature_metrics.get('precision'))
                    if feature_metrics
                    else np.nan
                )
                if pd.isna(flow_eval_to_train):
                    missing_flow[flow_key_used] += 1
                if pd.isna(feature_eval_to_train):
                    missing_feature[feature_key_used] += 1
            
            # Only add if we have both eval->train coverage values
            flow_eval_to_train = (
                flow_metrics.get('eval_to_train_coverage', flow_metrics.get('precision'))
                if flow_metrics
                else np.nan
            )
            feature_eval_to_train = (
                feature_metrics.get('eval_to_train_coverage', feature_metrics.get('precision'))
                if feature_metrics
                else np.nan
            )

            if (not pd.isna(flow_eval_to_train) and not pd.isna(feature_eval_to_train)):
                data_points.append({
                    'feature_eval_to_train_coverage': feature_eval_to_train,
                    'flow_eval_to_train_coverage': flow_eval_to_train,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'snapshot_path': str(snapshot_path)
                })
    
    if debug and (missing_flow or missing_feature):
        print(f"\nDebug: Missing flow eval->train coverage keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print(f"\nDebug: Missing {feature_label} eval->train coverage keys (top 10):")
        for key, count in sorted(missing_feature.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
    
    return data_points


def create_2d_precision_colormap_plot(
    data_points,
    output_path,
    dataset_color_map,
    feature_label="feature",
    zscore=False,
    zscore_by_benchmark=False,
):
    """Create 2D scatter plot with feature vs flow eval->train coverage, colored by PCK"""
    if not data_points:
        print("Warning: No data points for 2D precision colormap plot")
        return
    
    # Extract data
    feature_cov = [p['feature_eval_to_train_coverage'] for p in data_points]
    flow_cov = [p['flow_eval_to_train_coverage'] for p in data_points]
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
    scatter = ax.scatter(feature_cov, flow_cov, c=pck_values,
                        s=150, alpha=0.7, edgecolors='black', linewidth=1,
                        cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(color_label, fontsize=12)
    
    # Add benchmark labels
    for point in data_points:
        ax.annotate(point['benchmark'],
                   (point['feature_eval_to_train_coverage'], point['flow_eval_to_train_coverage']),
                   fontsize=7, alpha=0.6,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Feature Eval->Train Coverage', fontsize=12)
    ax.set_ylabel('Flow Eval->Train Coverage', fontsize=12)
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else (' (PCK z-scored)' if zscore else '')
    ax.set_title(f'{feature_label} Eval->Train vs Flow Eval->Train{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    safe_label = feature_label.lower().replace(' ', '_')
    output_file = output_path / f'2d_{safe_label}_flow_eval_to_train_pck_colormap{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D precision colormap plot: {output_file}")
    plt.close()


def collect_mmd_data_points(
    snapshots_data,
    flow_mmd_lookup,
    feature_mmd_lookup,
    feature_label="feature",
    debug=False,
):
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
        
        # Skip if summary file doesn't exist (shouldn't happen since we filter above, but be safe)
        if not summary_path.exists():
            if debug:
                print(f"  Skipping {snapshot_path}: training_summary.txt not found")
            continue
        
        # Get base training dataset name
        # parse_training_dataset_from_summary already handles mixed datasets by converting '+' to '_'
        # (e.g., "spair+synthetic" -> "spair_synthetic")
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Skipping {snapshot_path}: Could not parse training dataset from summary")
            continue
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            if debug:
                print(f"  Skipping {snapshot_path}: No best performance data found in summary")
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
                    print(
                        f"  Missing {feature_label} MMD for: {feature_key_used} "
                        f"(train={base_training_dataset}, bench={benchmark_lower})"
                    )
            
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
        print(f"\nDebug: Missing {feature_label} MMD keys (top 10):")
        for key, count in sorted(missing_feature.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        
        # Show sample of available keys
        print(f"\nDebug: Sample of available flow MMD keys (first 5):")
        for key in list(flow_mmd_lookup.keys())[:5]:
            print(f"  {key}")
        print(f"\nDebug: Sample of available {feature_label} MMD keys (first 5):")
        for key in list(feature_mmd_lookup.keys())[:5]:
            print(f"  {key}")
    
    return data_points


def create_2d_mmd_colormap_plot(
    data_points,
    output_path,
    dataset_color_map,
    feature_key="feature_mmd",
    feature_label="Feature",
    zscore=False,
    zscore_by_benchmark=False,
):
    """Create 2D scatter plot with Flow MMD vs Feature MMD, colored by PCK"""
    if not data_points:
        print("Warning: No data points for 2D MMD colormap plot")
        return
    
    # Extract data
    flow_mmd = [p['flow_mmd'] for p in data_points]
    feature_mmd = [p[feature_key] for p in data_points]
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
                   (point['flow_mmd'], point[feature_key]),
                   fontsize=7, alpha=0.6,
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Flow MMD²', fontsize=12)
    ax.set_ylabel(f'{feature_label} MMD²', fontsize=12)
    title_suffix = ' (PCK z-scored by benchmark)' if zscore_by_benchmark else (' (PCK z-scored)' if zscore else '')
    ax.set_title(f'Flow MMD² vs {feature_label} MMD²{title_suffix}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    safe_label = feature_label.lower().replace(' ', '_')
    output_file = output_path / f'2d_flow_{safe_label}_mmd_pck_colormap{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved 2D MMD colormap plot: {output_file}")
    plt.close()


def collect_all_predictors_data_points(
    snapshots_data,
    flow_mmd_lookup,
    feature_mmd_lookup,
    flow_coverage_lookup,
    resnet_coverage_lookup,
    dino_mmd_lookup=None,
    dino_coverage_lookup=None,
    debug=False,
):
    """
    Collect data points with all predictors: flow/feature MMD, directed coverage, and NN distances.
    
    Returns:
        List of dicts with keys including:
        flow_mmd, feature_mmd, dino_mmd,
        flow_train_to_eval_coverage, flow_eval_to_train_coverage,
        resnet_train_to_eval_coverage, resnet_eval_to_train_coverage,
        dino_train_to_eval_coverage, dino_eval_to_train_coverage,
        flow_train_to_eval_mean_dist, flow_eval_to_train_mean_dist,
        resnet_train_to_eval_mean_dist, resnet_eval_to_train_mean_dist,
        dino_train_to_eval_mean_dist, dino_eval_to_train_mean_dist,
        pck, training_dataset, benchmark, snapshot_path
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

        encoder_info = {}
        summary_info = parse_training_summary(summary_path) or {}
        dir_info = parse_directory_name(Path(snapshot_path).name) or {}
        for key in ("pretrained", "freeze"):
            if key in summary_info:
                encoder_info[key] = summary_info[key]
            elif key in dir_info:
                encoder_info[key] = dir_info[key]
        pretrained_val = encoder_info.get("pretrained")
        freeze_val = encoder_info.get("freeze")
        if pretrained_val is None and freeze_val is None:
            encoder_regime = "unknown"
        else:
            pretrained_tag = pretrained_val if pretrained_val is not None else "U"
            freeze_tag = freeze_val if freeze_val is not None else "U"
            encoder_regime = f"pretrained{pretrained_tag}_freeze{freeze_tag}"
        
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
            
            # Get feature MMD (ResNet)
            feature_mmd = feature_mmd_lookup.get((training_dataset_train, benchmark_test))
            if feature_mmd is None:
                feature_mmd = feature_mmd_lookup.get((training_dataset_train, benchmark_val))
            if feature_mmd is None:
                feature_mmd = feature_mmd_lookup.get((base_training_dataset, benchmark_lower))

            # Get DINO MMD (optional)
            dino_mmd = None
            if dino_mmd_lookup:
                dino_mmd = dino_mmd_lookup.get((training_dataset_train, benchmark_test))
                if dino_mmd is None:
                    dino_mmd = dino_mmd_lookup.get((training_dataset_train, benchmark_val))
                if dino_mmd is None:
                    dino_mmd = dino_mmd_lookup.get((base_training_dataset, benchmark_lower))
            
            # Get flow directed coverage + NN distance stats
            flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
            if flow_metrics is None:
                flow_metrics = flow_coverage_lookup.get((base_training_dataset, benchmark_lower))
            flow_train_to_eval = (
                flow_metrics.get('train_to_eval_coverage', flow_metrics.get('recall'))
                if flow_metrics
                else np.nan
            )
            flow_eval_to_train = (
                flow_metrics.get('eval_to_train_coverage', flow_metrics.get('precision'))
                if flow_metrics
                else np.nan
            )
            flow_train_to_eval_mean = (
                flow_metrics.get('train_to_eval_mean_dist', np.nan) if flow_metrics else np.nan
            )
            flow_eval_to_train_mean = (
                flow_metrics.get('eval_to_train_mean_dist', np.nan) if flow_metrics else np.nan
            )
            
            # Get resnet directed coverage + NN distance stats
            resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_test))
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_val))
            if resnet_metrics is None:
                resnet_metrics = resnet_coverage_lookup.get((base_training_dataset, benchmark_lower))
            resnet_train_to_eval = (
                resnet_metrics.get('train_to_eval_coverage', resnet_metrics.get('recall'))
                if resnet_metrics
                else np.nan
            )
            resnet_eval_to_train = (
                resnet_metrics.get('eval_to_train_coverage', resnet_metrics.get('precision'))
                if resnet_metrics
                else np.nan
            )
            resnet_train_to_eval_mean = (
                resnet_metrics.get('train_to_eval_mean_dist', np.nan) if resnet_metrics else np.nan
            )
            resnet_eval_to_train_mean = (
                resnet_metrics.get('eval_to_train_mean_dist', np.nan) if resnet_metrics else np.nan
            )

            dino_train_to_eval = np.nan
            dino_eval_to_train = np.nan
            dino_train_to_eval_mean = np.nan
            dino_eval_to_train_mean = np.nan
            if dino_coverage_lookup:
                dino_metrics = dino_coverage_lookup.get((training_dataset_train, benchmark_test))
                if dino_metrics is None:
                    dino_metrics = dino_coverage_lookup.get((training_dataset_train, benchmark_val))
                if dino_metrics is None:
                    dino_metrics = dino_coverage_lookup.get((base_training_dataset, benchmark_lower))
                if dino_metrics:
                    dino_train_to_eval = dino_metrics.get(
                        'train_to_eval_coverage', dino_metrics.get('recall')
                    )
                    dino_eval_to_train = dino_metrics.get(
                        'eval_to_train_coverage', dino_metrics.get('precision')
                    )
                    dino_train_to_eval_mean = dino_metrics.get('train_to_eval_mean_dist', np.nan)
                    dino_eval_to_train_mean = dino_metrics.get('eval_to_train_mean_dist', np.nan)
            
            has_resnet = feature_mmd is not None and not pd.isna(resnet_train_to_eval)
            has_dino = dino_mmd is not None and not pd.isna(dino_train_to_eval)

            # Only add if we have flow metrics and at least one feature set (ResNet or DINO)
            if (
                flow_mmd is not None
                and not pd.isna(flow_train_to_eval)
                and (has_resnet or has_dino)
            ):
                model_family = derive_model_family(Path(snapshot_path))
                data_points.append({
                    'flow_mmd': flow_mmd,
                    'feature_mmd': feature_mmd,
                    'dino_mmd': dino_mmd,
                    'flow_train_to_eval_coverage': flow_train_to_eval,
                    'flow_eval_to_train_coverage': flow_eval_to_train,
                    'resnet_train_to_eval_coverage': resnet_train_to_eval,
                    'resnet_eval_to_train_coverage': resnet_eval_to_train,
                    'dino_train_to_eval_coverage': dino_train_to_eval,
                    'dino_eval_to_train_coverage': dino_eval_to_train,
                    'flow_train_to_eval_mean_dist': flow_train_to_eval_mean,
                    'flow_eval_to_train_mean_dist': flow_eval_to_train_mean,
                    'resnet_train_to_eval_mean_dist': resnet_train_to_eval_mean,
                    'resnet_eval_to_train_mean_dist': resnet_eval_to_train_mean,
                    'dino_train_to_eval_mean_dist': dino_train_to_eval_mean,
                    'dino_eval_to_train_mean_dist': dino_eval_to_train_mean,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'benchmark': benchmark,
                    'pretrained': pretrained_val,
                    'freeze': freeze_val,
                    'encoder_regime': encoder_regime,
                    'model_family': model_family,
                    'snapshot_path': str(snapshot_path)
                })
            elif debug:
                missing_key = []
                if flow_mmd is None:
                    missing_key.append('flow_mmd')
                if feature_mmd is None:
                    missing_key.append('feature_mmd')
                if pd.isna(flow_train_to_eval):
                    missing_key.append('flow_train_to_eval_coverage')
                if not has_resnet and not has_dino:
                    if pd.isna(resnet_train_to_eval):
                        missing_key.append('resnet_train_to_eval_coverage')
                    if pd.isna(dino_train_to_eval):
                        missing_key.append('dino_train_to_eval_coverage')
                    if dino_mmd is None:
                        missing_key.append('dino_mmd')
                missing[tuple(missing_key)] += 1
    
    if debug and missing:
        print(f"\nDebug: Missing metrics (top 10):")
        for key, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
    
    return data_points


def compare_predictors_with_mixed_effects(
    df,
    output_path=None,
    create_plots=True,
    encoder_offsets=False,
    encoder_column="encoder_regime",
    model_family_offsets=False,
    model_family_column="model_family",
    encoder_interactions=False,
    model_family_interactions=False,
    predictor_set="all",
    standardize_mode="global",
    output_suffix="",
):
    """
    Compare all predictors using multiple mixed-effects regression models.
    
    This function:
    1. Runs models with each predictor individually
    2. Runs a full model with all predictors
    3. Compares models using AIC/BIC
    4. Shows standardized coefficients for fair comparison
    
    Args:
        df: DataFrame with columns including coverage, MMD, NN-distance metrics, plus 'pck' and 'benchmark'
        output_path: Optional path to save results to file
    """
    if not HAS_STATSMODELS:
        print("Error: statsmodels not installed. Cannot run mixed-effects regression.")
        print("Install with: pip install statsmodels")
        return None
    
    df = df.copy()
    # Require target + grouping
    required_cols = ['pck', 'benchmark']
    df = df.dropna(subset=required_cols)
    
    if len(df) < 10:
        print(f"Error: Insufficient data ({len(df)} points). Need at least 10 points for reliable analysis.")
        return None
    
    print(f"\n{'='*80}")
    print(f"PREDICTOR COMPARISON: Which metric is most predictive of PCK?")
    print(f"{'='*80}")
    print(f"\nData: {len(df)} observations across {df['benchmark'].nunique()} benchmarks")
    print(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}")
    print(f"\nPredictor set: {predictor_set}")
    print(f"Standardize mode: {standardize_mode}")
    print("\nNOTE: *_mean_dist = RAW L2 distances (in L2-normalized feature space)")
    encoder_term = ""
    use_encoder_offsets = False
    model_family_term = ""
    use_model_family_offsets = False
    encoder_interaction_term = ""
    model_family_interaction_term = ""
    use_encoder_interactions = False
    use_model_family_interactions = False
    if encoder_offsets:
        if encoder_column not in df.columns:
            print(f"Warning: Encoder offsets requested but '{encoder_column}' column is missing. Skipping offsets.")
        else:
            df[encoder_column] = df[encoder_column].fillna("unknown")
            if df[encoder_column].nunique() < 2:
                print(
                    f"Warning: Encoder offsets requested but '{encoder_column}' has only one category. "
                    "Skipping offsets."
                )
            else:
                encoder_term = f" + C({encoder_column})"
                use_encoder_offsets = True
                print(f"Encoder offsets enabled: C({encoder_column})")
    if model_family_offsets:
        if model_family_column not in df.columns:
            print(
                "Warning: Model family offsets requested but "
                f"'{model_family_column}' column is missing. Skipping offsets."
            )
        else:
            df[model_family_column] = df[model_family_column].fillna("unknown")
            if df[model_family_column].nunique() < 2:
                print(
                    "Warning: Model family offsets requested but "
                    f"'{model_family_column}' has only one category. Skipping offsets."
                )
            else:
                model_family_term = f" + C({model_family_column})"
                use_model_family_offsets = True
                print(f"Model family offsets enabled: C({model_family_column})")

    if encoder_interactions:
        if encoder_column not in df.columns:
            print(
                "Warning: Encoder interactions requested but "
                f"'{encoder_column}' column is missing. Skipping interactions."
            )
        else:
            df[encoder_column] = df[encoder_column].fillna("unknown")
            if df[encoder_column].nunique() < 2:
                print(
                    "Warning: Encoder interactions requested but "
                    f"'{encoder_column}' has only one category. Skipping interactions."
                )
            else:
                encoder_interaction_term = f":C({encoder_column})"
                use_encoder_interactions = True
                print(f"Encoder interactions enabled: {encoder_column}")

    if model_family_interactions:
        if model_family_column not in df.columns:
            print(
                "Warning: Model family interactions requested but "
                f"'{model_family_column}' column is missing. Skipping interactions."
            )
        else:
            df[model_family_column] = df[model_family_column].fillna("unknown")
            if df[model_family_column].nunique() < 2:
                print(
                    "Warning: Model family interactions requested but "
                    f"'{model_family_column}' has only one category. Skipping interactions."
                )
            else:
                model_family_interaction_term = f":C({model_family_column})"
                use_model_family_interactions = True
                print(f"Model family interactions enabled: {model_family_column}")
    
    min_required = 10
    candidate_predictors = select_predictor_candidates(predictor_set)
    predictors = []
    for name in candidate_predictors:
        if name not in df.columns:
            continue
        available = df[name].notna().sum()
        if available >= min_required:
            predictors.append(name)
        else:
            print(f"  Skipping {name} (only {available} observations)")

    if not predictors:
        print("Error: No predictors with sufficient data for mixed-effects analysis.")
        return None
    
    df_scaled = df.copy()
    predictors_to_scale = [p for p in predictors if p in df.columns]
    if standardize_mode not in STANDARDIZE_MODES:
        print(
            f"Warning: Unknown standardize_mode '{standardize_mode}', "
            "falling back to global."
        )
        standardize_mode = "global"
    group_col = None
    if standardize_mode == "benchmark":
        group_col = "benchmark"
    elif standardize_mode == "encoder":
        group_col = encoder_column
    elif standardize_mode == "model_family":
        group_col = model_family_column
    df_scaled, standardized_cols = _standardize_predictors_insample(
        df_scaled,
        predictors_to_scale,
        standardize_mode,
        min_std=0.0,
        group_col=group_col,
    )
    
    # 0. Collinearity analysis (correlation matrix)
    print(f"\n{'='*80}")
    print("0. COLLINEARITY ANALYSIS (Predictor Correlations)")
    print(f"{'='*80}")
    print("Computing pairwise correlations between predictors...")
    print("(Using original scale, not standardized)")
    print("\n⚠️  High correlations (|r| > 0.7) may indicate collinearity issues")
    print("   This can make coefficients unstable in the full model.\n")
    
    # Initialize correlation matrix variable
    corr_matrix_for_save = None
    high_corr_pairs = []
    
    # Compute correlation matrix for predictors (using original scale)
    predictor_data = df[predictors_to_scale].dropna()
    if len(predictor_data) > 0:
        corr_matrix = predictor_data.corr()
        
        # Display correlation matrix
        print(f"{'Predictor':<20}", end="")
        for p in predictors_to_scale:
            # Truncate long names for display
            display_name = p.replace('_', ' ')[:12]
            print(f"{display_name:>12}", end="")
        print()
        print(f"{'-'*80}")
        
        for i, p1 in enumerate(predictors_to_scale):
            display_name1 = p1.replace('_', ' ')[:18]
            print(f"{display_name1:<20}", end="")
            for j, p2 in enumerate(predictors_to_scale):
                if i <= j:
                    corr_val = corr_matrix.loc[p1, p2]
                    if i == j:
                        print(f"{'1.0000':>12}", end="")
                    else:
                        print(f"{corr_val:>12.4f}", end="")
                        # Flag high correlations
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append((p1, p2, corr_val))
                else:
                    print(f"{'':>12}", end="")
            print()
        
        # Report high correlations
        if high_corr_pairs:
            print(f"\n⚠️  HIGH CORRELATIONS DETECTED (|r| > 0.7):")
            for p1, p2, corr_val in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"   {p1} <-> {p2}: r = {corr_val:.4f}")
            print(f"\n   Interpretation:")
            print(f"   - These predictors share substantial variance")
            print(f"   - Coefficients in full model may be less stable")
            print(f"   - Consider if both are needed or if one is redundant")
        else:
            print(f"\n✓ No high correlations detected (all |r| ≤ 0.7)")
            print(f"   Collinearity is not a major concern")
        
        # Store correlation matrix for saving to file
        corr_matrix_for_save = corr_matrix
    else:
        print("Warning: Could not compute correlations (insufficient data)")
    
    results = {}
    
    # 1. Individual predictor models
    print(f"\n{'='*80}")
    print("1. INDIVIDUAL PREDICTOR MODELS")
    print(f"{'='*80}")
    print(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'AIC':>10} {'BIC':>10}")
    print(f"{'-'*70}")
    
    mixed_fallback_count = 0
    for predictor in predictors:
        try:
            predictor_std = f"{predictor}_std"
            interaction_terms = []
            if use_encoder_interactions:
                interaction_terms.append(f"{predictor_std}{encoder_interaction_term}")
            if use_model_family_interactions:
                interaction_terms.append(f"{predictor_std}{model_family_interaction_term}")
            interaction_clause = ""
            if interaction_terms:
                interaction_clause = " + " + " + ".join(interaction_terms)
            formula = f"pck ~ {predictor_std}{encoder_term}{model_family_term}{interaction_clause}"
            model = smf.mixedlm(
                formula,
                data=df_scaled,
                groups=df_scaled["benchmark"],
            )
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
            try:
                fe_formula = f"{formula} + C(benchmark)"
                fallback = smf.ols(fe_formula, data=df_scaled).fit()
                coef = fallback.params.get(predictor_std, np.nan)
                pval = fallback.pvalues.get(predictor_std, np.nan)
                aic = fallback.aic if hasattr(fallback, "aic") else np.nan
                bic = fallback.bic if hasattr(fallback, "bic") else np.nan
                results[predictor] = {
                    'std_coef': coef,
                    'pvalue': pval,
                    'aic': aic,
                    'bic': bic,
                    'converged': True,
                    'significant': pval < 0.05 if not np.isnan(pval) else False,
                    'model': fallback,
                    'model_type': 'ols_fe',
                }
                mixed_fallback_count += 1
                aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
                bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
                print(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f} {aic_str:>10} {bic_str:>10}")
            except Exception as fallback_error:
                print(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>10} {'ERROR':>10}")
                print(f"  MixedLM error: {e}")
                print(f"  OLS fallback error: {fallback_error}")
                results[predictor] = None

    if mixed_fallback_count:
        print(
            f"Note: MixedLM failed for {mixed_fallback_count} predictor(s); "
            "used OLS with benchmark fixed effects instead."
        )
    
    # 2. Full model with all predictors
    print(f"\n{'='*80}")
    print("2. FULL MODEL (All predictors together)")
    print(f"{'='*80}")
    
    full_model_type = "mixedlm"
    try:
        # Build formula with available predictors
        formula_parts = [f"{p}_std" for p in predictors_to_scale]
        interaction_terms = []
        if use_encoder_interactions:
            interaction_terms.extend(
                [f"{p}_std{encoder_interaction_term}" for p in predictors_to_scale]
            )
        if use_model_family_interactions:
            interaction_terms.extend(
                [f"{p}_std{model_family_interaction_term}" for p in predictors_to_scale]
            )
        formula = "pck ~ " + " + ".join(formula_parts)
        if encoder_term:
            formula += encoder_term
        if model_family_term:
            formula += model_family_term
        if interaction_terms:
            formula += " + " + " + ".join(interaction_terms)
        model_full = smf.mixedlm(formula, data=df_scaled, groups=df_scaled["benchmark"])
        result_full = model_full.fit(method='lbfgs', reml=False)  # Use ML instead of REML for AIC/BIC
        
        # Check if model converged
        if not result_full.converged:
            print(f"Warning: Full model did not converge!")
            results['full_model'] = None
        else:
            predictor_names = {
                'flow_mmd': 'Flow MMD',
                'feature_mmd': 'ResNet MMD',
                'dino_mmd': 'DINO MMD',
                'flow_train_to_eval_coverage': 'Flow Train->Eval Coverage',
                'flow_eval_to_train_coverage': 'Flow Eval->Train Coverage',
                'resnet_train_to_eval_coverage': 'ResNet Train->Eval Coverage',
                'resnet_eval_to_train_coverage': 'ResNet Eval->Train Coverage',
                'dino_train_to_eval_coverage': 'DINO Train->Eval Coverage',
                'dino_eval_to_train_coverage': 'DINO Eval->Train Coverage',
                'flow_train_to_eval_mean_dist': 'Flow Train->Eval Mean Dist',
                'flow_eval_to_train_mean_dist': 'Flow Eval->Train Mean Dist',
                'resnet_train_to_eval_mean_dist': 'ResNet Train->Eval Mean Dist',
                'resnet_eval_to_train_mean_dist': 'ResNet Eval->Train Mean Dist',
                'dino_train_to_eval_mean_dist': 'DINO Train->Eval Mean Dist',
                'dino_eval_to_train_mean_dist': 'DINO Eval->Train Mean Dist',
            }
            formula_display = " + ".join([predictor_names.get(p, p) for p in predictors_to_scale])
            encoder_display = f" + C({encoder_column})" if use_encoder_offsets else ""
            model_family_display = f" + C({model_family_column})" if use_model_family_offsets else ""
            interaction_display = ""
            if use_encoder_interactions:
                interaction_display += f" + predictors:C({encoder_column})"
            if use_model_family_interactions:
                interaction_display += f" + predictors:C({model_family_column})"
            print(
                f"Model: PCK ~ {formula_display}{encoder_display}{model_family_display}"
                f"{interaction_display} + (1|benchmark)"
            )
            print(f"\n{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'Significant':>12}")
            print(f"{'-'*60}")
            
            full_results = {}
            params = result_full.fe_params
            pvalues = result_full.pvalues
            for predictor in predictors_to_scale:
                coef = params.get(f'{predictor}_std', np.nan)
                pval = pvalues.get(f'{predictor}_std', np.nan)
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

            # MixedLM diagnostics: variance components + LR test vs random-intercept only
            mixed_diag = {
                "re_var": np.nan,
                "resid_var": np.nan,
                "icc": np.nan,
                "lr_stat": np.nan,
                "lr_df": np.nan,
                "lr_pvalue": np.nan,
            }
            try:
                re_var = result_full.cov_re.iloc[0, 0] if hasattr(result_full, "cov_re") else np.nan
                resid_var = result_full.scale if hasattr(result_full, "scale") else np.nan
                icc = np.nan
                if np.isfinite(re_var) and np.isfinite(resid_var) and (re_var + resid_var) > 0:
                    icc = re_var / (re_var + resid_var)
                mixed_diag.update({"re_var": re_var, "resid_var": resid_var, "icc": icc})
            except Exception:
                pass

            try:
                null_model = smf.mixedlm("pck ~ 1", data=df_scaled, groups=df_scaled["benchmark"])
                null_result = null_model.fit(method='lbfgs', reml=False)
                if hasattr(result_full, "llf") and hasattr(null_result, "llf"):
                    lr_stat = 2.0 * (result_full.llf - null_result.llf)
                    lr_df = len(result_full.fe_params) - len(null_result.fe_params)
                    lr_pval = stats.chi2.sf(lr_stat, max(lr_df, 1)) if np.isfinite(lr_stat) else np.nan
                    mixed_diag.update({"lr_stat": lr_stat, "lr_df": lr_df, "lr_pvalue": lr_pval})
                    print(f"  Residual variance: {mixed_diag['resid_var']:.4f}")
                    if np.isfinite(mixed_diag["icc"]):
                        print(f"  ICC (benchmark): {mixed_diag['icc']:.3f}")
                    print(f"  LR test vs random-intercept only: LR={lr_stat:.2f}, df={lr_df}, p={lr_pval:.4f}")
            except Exception as e:
                print(f"  MixedLM diagnostics skipped: {e}")

            results['full_model'] = {
                'result': result_full,
                'predictors': full_results,
                'aic': aic,
                'bic': bic,
                'converged': result_full.converged,
                'df_scaled': df_scaled,
                'predictors_to_scale': predictors_to_scale,
                'model_type': full_model_type,
            }
            results['mixedlm_diagnostics'] = mixed_diag
            
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
        try:
            full_model_type = "ols_fe"
            fe_formula = f"{formula} + C(benchmark)"
            result_full = smf.ols(fe_formula, data=df_scaled).fit()
            predictor_names = {
                'flow_mmd': 'Flow MMD',
                'feature_mmd': 'ResNet MMD',
                'dino_mmd': 'DINO MMD',
                'flow_train_to_eval_coverage': 'Flow Train->Eval Coverage',
                'flow_eval_to_train_coverage': 'Flow Eval->Train Coverage',
                'resnet_train_to_eval_coverage': 'ResNet Train->Eval Coverage',
                'resnet_eval_to_train_coverage': 'ResNet Eval->Train Coverage',
                'dino_train_to_eval_coverage': 'DINO Train->Eval Coverage',
                'dino_eval_to_train_coverage': 'DINO Eval->Train Coverage',
                'flow_train_to_eval_mean_dist': 'Flow Train->Eval Mean Dist',
                'flow_eval_to_train_mean_dist': 'Flow Eval->Train Mean Dist',
                'resnet_train_to_eval_mean_dist': 'ResNet Train->Eval Mean Dist',
                'resnet_eval_to_train_mean_dist': 'ResNet Eval->Train Mean Dist',
                'dino_train_to_eval_mean_dist': 'DINO Train->Eval Mean Dist',
                'dino_eval_to_train_mean_dist': 'DINO Eval->Train Mean Dist',
            }
            formula_display = " + ".join([predictor_names.get(p, p) for p in predictors_to_scale])
            encoder_display = f" + C({encoder_column})" if use_encoder_offsets else ""
            model_family_display = f" + C({model_family_column})" if use_model_family_offsets else ""
            interaction_display = ""
            if use_encoder_interactions:
                interaction_display += f" + predictors:C({encoder_column})"
            if use_model_family_interactions:
                interaction_display += f" + predictors:C({model_family_column})"
            print(
                "MixedLM full model failed; falling back to OLS with benchmark fixed effects."
            )
            print(
                f"Model: PCK ~ {formula_display}{encoder_display}{model_family_display}"
                f"{interaction_display} + C(benchmark)"
            )
            print(f"\n{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'Significant':>12}")
            print(f"{'-'*60}")

            full_results = {}
            params = result_full.params
            pvalues = result_full.pvalues
            for predictor in predictors_to_scale:
                coef = params.get(f'{predictor}_std', np.nan)
                pval = pvalues.get(f'{predictor}_std', np.nan)
                sig = pval < 0.05 if not np.isnan(pval) else False
                sig_marker = '*' if sig else ''
                full_results[predictor] = {
                    'std_coef': coef,
                    'pvalue': pval,
                    'significant': sig
                }
                print(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {'Yes' if sig else 'No':>12}")

            aic = result_full.aic if hasattr(result_full, 'aic') else np.nan
            bic = result_full.bic if hasattr(result_full, 'bic') else np.nan
            llf_str = f"{result_full.llf:.2f}" if hasattr(result_full, 'llf') else "N/A"
            aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
            print(f"\nModel fit:")
            print(f"  AIC: {aic_str}")
            print(f"  BIC: {bic_str}")
            print(f"  Log-likelihood: {llf_str}")

            results['full_model'] = {
                'result': result_full,
                'predictors': full_results,
                'aic': aic,
                'bic': bic,
                'converged': True,
                'df_scaled': df_scaled,
                'predictors_to_scale': predictors_to_scale,
                'model_type': full_model_type,
            }
        except Exception as fallback_error:
            print(f"Error fitting full model: {e}")
            print(f"OLS fallback error: {fallback_error}")
            import traceback
            traceback.print_exc()
            results['full_model'] = None

    # 2b. Non-mixed effects (OLS) models
    print(f"\n{'='*80}")
    print("2b. NON-MIXED EFFECTS (OLS, no benchmark effects)")
    print(f"{'='*80}")
    print(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'R2':>8} {'AIC':>10} {'BIC':>10}")
    print(f"{'-'*80}")

    ols_results = {}
    for predictor in predictors_to_scale:
        try:
            model = smf.ols(f"pck ~ {predictor}_std{encoder_term}", data=df_scaled)
            result = model.fit()
            coef = result.params.get(f"{predictor}_std", np.nan)
            pval = result.pvalues.get(f"{predictor}_std", np.nan)
            aic = result.aic if hasattr(result, "aic") else np.nan
            bic = result.bic if hasattr(result, "bic") else np.nan
            r2 = result.rsquared if hasattr(result, "rsquared") else np.nan
            sig_marker = '*' if (not np.isnan(pval) and pval < 0.05) else ''
            aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
            r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
            print(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {r2_str:>8} {aic_str:>10} {bic_str:>10}")
            ols_results[predictor] = {
                "std_coef": coef,
                "pvalue": pval,
                "aic": aic,
                "bic": bic,
                "r2": r2,
                "significant": pval < 0.05 if not np.isnan(pval) else False,
            }
        except Exception as e:
            print(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>8} {'ERROR':>10} {'ERROR':>10}")
            print(f"  Error: {e}")
            ols_results[predictor] = None

    ols_full = None
    try:
        formula_parts = [f"{p}_std" for p in predictors_to_scale]
        formula = "pck ~ " + " + ".join(formula_parts) + encoder_term
        model_full = smf.ols(formula, data=df_scaled)
        result_full = model_full.fit()
        aic = result_full.aic if hasattr(result_full, "aic") else np.nan
        bic = result_full.bic if hasattr(result_full, "bic") else np.nan
        r2 = result_full.rsquared if hasattr(result_full, "rsquared") else np.nan
        aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
        bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
        print(f"\nFull OLS model: R2={r2_str}, AIC={aic_str}, BIC={bic_str}")
        ols_full = {
            "result": result_full,
            "aic": aic,
            "bic": bic,
            "r2": r2,
        }
    except Exception as e:
        print(f"Full OLS model error: {e}")

    results["ols_models"] = ols_results
    results["ols_full_model"] = ols_full

    # 2c. Benchmark-mean OLS (offset sniff test)
    print(f"\n{'='*80}")
    print("2c. BENCHMARK-MEAN OLS (Offset Sniff Test)")
    print(f"{'='*80}")
    if use_encoder_offsets:
        print("Skipping benchmark-mean OLS because encoder offsets are enabled.")
        bench_ols = None
    else:
        bench_df = df_scaled.groupby("benchmark").mean(numeric_only=True)
        if len(bench_df) < 3:
            print("Not enough benchmarks for benchmark-mean OLS (need at least 3).")
            bench_ols = None
        else:
            print(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'R2':>8}")
            print(f"{'-'*60}")
            bench_ols = {}
            for predictor in predictors_to_scale:
                col = f"{predictor}_std"
                if col not in bench_df.columns:
                    continue
                try:
                    model = smf.ols(f"pck ~ {col}", data=bench_df)
                    result = model.fit()
                    coef = result.params.get(col, np.nan)
                    pval = result.pvalues.get(col, np.nan)
                    r2 = result.rsquared if hasattr(result, "rsquared") else np.nan
                    sig_marker = '*' if (not np.isnan(pval) and pval < 0.05) else ''
                    r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                    print(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {r2_str:>8}")
                    bench_ols[predictor] = {
                        "std_coef": coef,
                        "pvalue": pval,
                        "r2": r2,
                        "significant": pval < 0.05 if not np.isnan(pval) else False,
                    }
                except Exception as e:
                    print(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>8}")
                    print(f"  Error: {e}")
                    bench_ols[predictor] = None
    results["benchmark_ols"] = bench_ols
    
    # 3. Model comparison
    print(f"\n{'='*80}")
    print("3. MODEL COMPARISON (Lower AIC/BIC is better)")
    print(f"{'='*80}")
    
    # Sort individual models by AIC (filter out NaN AIC values)
    valid_models = [
        (k, v)
        for k, v in results.items()
        if v is not None
        and k != "full_model"
        and not np.isnan(v.get("aic", np.nan))
    ]
    valid_models.sort(key=lambda x: x[1]['aic'])
    valid_predictor_models = [(k, v) for k, v in valid_models if "std_coef" in v]
    
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
    if valid_predictor_models:
        best_predictor = valid_predictor_models[0][0]
        best_model = valid_predictor_models[0][1]
        print(f"\nBest individual predictor: {best_predictor}")
        print(f"  Standardized coefficient: {best_model['std_coef']:.4f}")
        print(f"  p-value: {best_model['pvalue']:.4f}")
        print(
            f"  {'✓ Statistically significant' if best_model['significant'] else '✗ Not statistically significant'}"
        )
    
    # Check if full model is better
    if results.get('full_model') and not np.isnan(results['full_model'].get('aic', np.nan)):
        full_aic = results['full_model']['aic']
        if valid_models and not np.isnan(best_aic) and full_aic < best_aic:
            print(f"\n✓ Full model (AIC={full_aic:.1f}) is better than best individual model (AIC={best_aic:.1f})")
            print("  -> Multiple predictors together improve prediction")
            
            # Show which predictors remain significant in full model
            sig_in_full = [p for p, data in results['full_model']['predictors'].items() if data['significant']]
            if sig_in_full:
                print(f"  Significant predictors in full model: {', '.join(sig_in_full)}")
        elif valid_models and not np.isnan(best_aic):
            print(f"\n✗ Full model (AIC={full_aic:.1f}) is NOT better than best individual model (AIC={best_aic:.1f})")
            print("  -> Single predictor is sufficient")
    
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
        suffix = output_suffix or ""
        output_file = output_path / f"predictor_comparison_analysis{suffix}.txt"
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PREDICTOR COMPARISON ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Data: {len(df)} observations across {df['benchmark'].nunique()} benchmarks\n")
            f.write(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}\n")
            f.write(f"Predictor set: {predictor_set}\n")
            f.write(f"Standardize mode: {standardize_mode}\n")
            if use_encoder_offsets:
                f.write(f"Encoder offsets: C({encoder_column})\n")
            if use_model_family_offsets:
                f.write(f"Model family offsets: C({model_family_column})\n")
            f.write("\nNOTE ON METRIC NORMALIZATION:\n")
            f.write("- *_mean_dist metrics: RAW L2 distances in L2-normalized feature/flow space\n")
            f.write("  (NOT normalized by radius or median; directional asymmetric metrics)\n")
            f.write("- *_mmd metrics: Maximum Mean Discrepancy (symmetric)\n")
            f.write("- *_coverage metrics: Fraction within learned radius threshold\n")
            if use_encoder_interactions:
                f.write(f"Encoder interactions: predictors:C({encoder_column})\n")
            if use_model_family_interactions:
                f.write(f"Model family interactions: predictors:C({model_family_column})\n")
            
            # Write collinearity analysis section
            f.write(f"\n{'='*80}\n")
            f.write("0. COLLINEARITY ANALYSIS (Predictor Correlations)\n")
            f.write(f"{'='*80}\n")
            f.write("Pairwise correlations between predictors (using original scale):\n")
            f.write("⚠️  High correlations (|r| > 0.7) may indicate collinearity issues\n\n")
            
            if corr_matrix_for_save is not None:
                # Write correlation matrix
                f.write(f"{'Predictor':<20}")
                for p in predictors_to_scale:
                    display_name = p.replace('_', ' ')[:12]
                    f.write(f"{display_name:>12}")
                f.write("\n")
                f.write(f"{'-'*80}\n")
                
                high_corr_pairs_file = []
                for i, p1 in enumerate(predictors_to_scale):
                    display_name1 = p1.replace('_', ' ')[:18]
                    f.write(f"{display_name1:<20}")
                    for j, p2 in enumerate(predictors_to_scale):
                        if i <= j:
                            corr_val = corr_matrix_for_save.loc[p1, p2]
                            if i == j:
                                f.write(f"{'1.0000':>12}")
                            else:
                                f.write(f"{corr_val:>12.4f}")
                                if abs(corr_val) > 0.7:
                                    high_corr_pairs_file.append((p1, p2, corr_val))
                        else:
                            f.write(f"{'':>12}")
                    f.write("\n")
                
                # Report high correlations
                if high_corr_pairs_file:
                    f.write(f"\n⚠️  HIGH CORRELATIONS DETECTED (|r| > 0.7):\n")
                    for p1, p2, corr_val in sorted(high_corr_pairs_file, key=lambda x: abs(x[2]), reverse=True):
                        f.write(f"   {p1} <-> {p2}: r = {corr_val:.4f}\n")
                    f.write(f"\n   Interpretation:\n")
                    f.write(f"   - These predictors share substantial variance\n")
                    f.write(f"   - Coefficients in full model may be less stable\n")
                    f.write(f"   - Consider if both are needed or if one is redundant\n")
                else:
                    f.write(f"\n✓ No high correlations detected (all |r| ≤ 0.7)\n")
                    f.write(f"   Collinearity is not a major concern\n")
            else:
                f.write("Warning: Could not compute correlations (insufficient data)\n")
            
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
            if mixed_fallback_count:
                f.write(
                    "\nNote: MixedLM failed for "
                    f"{mixed_fallback_count} predictor(s); used OLS with benchmark fixed effects instead.\n"
                )
            
            # Write full model section
            f.write(f"\n{'='*80}\n")
            f.write("2. FULL MODEL (All predictors together)\n")
            f.write(f"{'='*80}\n")
            
            if results.get('full_model') and results['full_model'] is not None:
                predictor_names = {
                    'flow_mmd': 'Flow MMD',
                    'feature_mmd': 'ResNet MMD',
                    'dino_mmd': 'DINO MMD',
                    'flow_train_to_eval_coverage': 'Flow Train->Eval Coverage',
                    'flow_eval_to_train_coverage': 'Flow Eval->Train Coverage',
                    'resnet_train_to_eval_coverage': 'ResNet Train->Eval Coverage',
                    'resnet_eval_to_train_coverage': 'ResNet Eval->Train Coverage',
                    'dino_train_to_eval_coverage': 'DINO Train->Eval Coverage',
                    'dino_eval_to_train_coverage': 'DINO Eval->Train Coverage',
                    'flow_train_to_eval_mean_dist': 'Flow Train->Eval Mean Dist',
                    'flow_eval_to_train_mean_dist': 'Flow Eval->Train Mean Dist',
                    'resnet_train_to_eval_mean_dist': 'ResNet Train->Eval Mean Dist',
                    'resnet_eval_to_train_mean_dist': 'ResNet Eval->Train Mean Dist',
                    'dino_train_to_eval_mean_dist': 'DINO Train->Eval Mean Dist',
                    'dino_eval_to_train_mean_dist': 'DINO Eval->Train Mean Dist',
                }
                formula_display = " + ".join([predictor_names.get(p, p) for p in predictors_to_scale])
                encoder_display = f" + C({encoder_column})" if use_encoder_offsets else ""
                f.write(f"Model: PCK ~ {formula_display}{encoder_display} + (1|benchmark)\n")
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
                mixed_diag = results.get("mixedlm_diagnostics")
                if mixed_diag:
                    resid_var = mixed_diag.get("resid_var", np.nan)
                    icc = mixed_diag.get("icc", np.nan)
                    lr_stat = mixed_diag.get("lr_stat", np.nan)
                    lr_df = mixed_diag.get("lr_df", np.nan)
                    lr_pval = mixed_diag.get("lr_pvalue", np.nan)
                    if not np.isnan(resid_var):
                        f.write(f"  Residual variance: {resid_var:.4f}\n")
                    if not np.isnan(icc):
                        f.write(f"  ICC (benchmark): {icc:.3f}\n")
                    if not np.isnan(lr_stat):
                        f.write(f"  LR test vs random-intercept only: LR={lr_stat:.2f}, df={lr_df}, p={lr_pval:.4f}\n")
            else:
                f.write("Full model did not converge or encountered an error.\n")

            # Write non-mixed effects OLS section
            f.write(f"\n{'='*80}\n")
            f.write("2b. NON-MIXED EFFECTS (OLS, no benchmark effects)\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'R2':>8} {'AIC':>10} {'BIC':>10}\n")
            f.write(f"{'-'*80}\n")
            ols_models = results.get("ols_models") or {}
            for predictor in predictors_to_scale:
                model_data = ols_models.get(predictor)
                if model_data and model_data is not None:
                    coef = model_data.get("std_coef", np.nan)
                    pval = model_data.get("pvalue", np.nan)
                    aic = model_data.get("aic", np.nan)
                    bic = model_data.get("bic", np.nan)
                    r2 = model_data.get("r2", np.nan)
                    sig_marker = '*' if model_data.get("significant", False) else ''
                    aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
                    bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
                    r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                    f.write(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {r2_str:>8} {aic_str:>10} {bic_str:>10}\n")
                else:
                    f.write(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>8} {'ERROR':>10} {'ERROR':>10}\n")

            ols_full = results.get("ols_full_model")
            if ols_full and ols_full is not None:
                aic = ols_full.get("aic", np.nan)
                bic = ols_full.get("bic", np.nan)
                r2 = ols_full.get("r2", np.nan)
                aic_str = f"{aic:.1f}" if not np.isnan(aic) else "N/A"
                bic_str = f"{bic:.1f}" if not np.isnan(bic) else "N/A"
                r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                f.write(f"\nFull OLS model: R2={r2_str}, AIC={aic_str}, BIC={bic_str}\n")

            # Write benchmark-mean OLS section
            f.write(f"\n{'='*80}\n")
            f.write("2c. BENCHMARK-MEAN OLS (Offset Sniff Test)\n")
            f.write(f"{'='*80}\n")
            bench_ols = results.get("benchmark_ols")
            if use_encoder_offsets:
                f.write("Skipping benchmark-mean OLS because encoder offsets are enabled.\n")
            elif not bench_ols:
                f.write("Not enough benchmarks for benchmark-mean OLS (need at least 3).\n")
            else:
                f.write(f"{'Predictor':<20} {'Std Coef':>12} {'p-value':>12} {'R2':>8}\n")
                f.write(f"{'-'*60}\n")
                for predictor in predictors_to_scale:
                    model_data = bench_ols.get(predictor)
                    if model_data and model_data is not None:
                        coef = model_data.get("std_coef", np.nan)
                        pval = model_data.get("pvalue", np.nan)
                        r2 = model_data.get("r2", np.nan)
                        sig_marker = '*' if model_data.get("significant", False) else ''
                        r2_str = f"{r2:.3f}" if not np.isnan(r2) else "N/A"
                        f.write(f"{predictor:<20} {coef:>12.4f} {pval:>12.4f}{sig_marker:>1} {r2_str:>8}\n")
                    else:
                        f.write(f"{predictor:<20} {'ERROR':>12} {'ERROR':>12} {'ERROR':>8}\n")
            
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
            
            if valid_predictor_models:
                best_predictor = valid_predictor_models[0][0]
                best_model = valid_predictor_models[0][1]
                f.write(f"\nBest individual predictor: {best_predictor}\n")
                f.write(f"  Standardized coefficient: {best_model['std_coef']:.4f}\n")
                f.write(f"  p-value: {best_model['pvalue']:.4f}\n")
                f.write(
                    f"  {'✓ Statistically significant' if best_model['significant'] else '✗ Not statistically significant'}\n"
                )
            
            # Check if full model is better
            if results.get('full_model') and not np.isnan(results['full_model'].get('aic', np.nan)):
                full_aic = results['full_model']['aic']
                if valid_models and not np.isnan(best_aic) and full_aic < best_aic:
                    f.write(f"\n✓ Full model (AIC={full_aic:.1f}) is better than best individual model (AIC={best_aic:.1f})\n")
                    f.write("  -> Multiple predictors together improve prediction\n")
                    
                    # Show which predictors remain significant in full model
                    sig_in_full = [p for p, data in results['full_model']['predictors'].items() if data['significant']]
                    if sig_in_full:
                        f.write(f"  Significant predictors in full model: {', '.join(sig_in_full)}\n")
                elif valid_models and not np.isnan(best_aic):
                    f.write(f"\n✗ Full model (AIC={full_aic:.1f}) is NOT better than best individual model (AIC={best_aic:.1f})\n")
                    f.write("  -> Single predictor is sufficient\n")
            
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
        '--snapshots-dirs', type=str, default=None,
        help='Comma-separated list of snapshot directories (overrides --snapshots-dir)'
    )
    parser.add_argument(
        '--coverage-csv', type=str, default='coverage_results.csv',
        help='Path to flow coverage CSV (default: coverage_results.csv)'
    )
    parser.add_argument(
        '--coverage-resnet-csv', type=str, default='coverage_resnet_results.csv',
        help='Path to resnet feature coverage CSV (default: coverage_resnet_results.csv)'
    )
    parser.add_argument(
        '--coverage-dino-csv', type=str, default=None,
        help='Optional path to DINO feature coverage CSV'
    )
    parser.add_argument(
        '--flow-mmd-csv', type=str, default='flow_mmd_results.csv',
        help='Path to flow MMD CSV (default: flow_mmd_results.csv)'
    )
    parser.add_argument(
        '--feature-mmd-csv', type=str, default='feature_mmd_results.csv',
        help='Path to resnet feature MMD CSV (default: feature_mmd_results.csv)'
    )
    parser.add_argument(
        '--dino-mmd-csv', type=str, default=None,
        help='Optional path to DINO feature MMD CSV'
    )
    parser.add_argument(
        '--output-dir', type=str, default='plots3d/',
        help='Output directory for plots (default: plots3d/)'
    )
    parser.add_argument(
        '--zscore', action='store_true',
        help='Also create z-scored versions of PCK'
    )
    parser.add_argument(
        '--encoder-offsets', action='store_true',
        help='Add encoder regime offsets (pretrained/freeze) in mixed-effects models'
    )
    parser.add_argument(
        '--encoder-interactions', action='store_true',
        help='Add encoder interactions (predictors x encoder regime) in mixed-effects models'
    )
    parser.add_argument(
        '--model-family-offsets', action='store_true',
        help='Add model family offsets (catspp/raft/flowformer) in mixed-effects models'
    )
    parser.add_argument(
        '--model-family-interactions', action='store_true',
        help='Add model family interactions (predictors x model family) in mixed-effects models'
    )
    parser.add_argument(
        '--predictor-set',
        choices=PREDICTOR_SETS,
        default='all',
        help='Predictor set for mixed-effects analysis (all or trimmed).'
    )
    parser.add_argument(
        '--standardize-mode',
        choices=STANDARDIZE_MODES,
        default='global',
        help='Predictor standardization mode for mixed-effects analysis.'
    )
    parser.add_argument(
        '--analysis-suffix',
        default='',
        help='Suffix appended to predictor comparison analysis output filename.'
    )
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load snapshots
    print("Loading snapshots...")
    snapshots_root_list = []
    if args.snapshots_dirs:
        snapshots_root_list.extend(
            [p.strip() for p in args.snapshots_dirs.split(",") if p.strip()]
        )
    else:
        snapshots_root_list.extend(
            [p.strip() for p in str(args.snapshots_dir).split(",") if p.strip()]
        )

    snapshots_root_paths = [Path(p).expanduser() for p in snapshots_root_list]
    snapshots_root_paths = [p for p in snapshots_root_paths if p.exists()]
    if not snapshots_root_paths:
        missing = args.snapshots_dirs or args.snapshots_dir
        print(f"Error: No snapshot directories exist: {missing}")
        return

    # Collect snapshot directories (recursively search nested directories)
    snapshot_dirs = []

    def find_snapshot_directories(root_path, max_depth=3, current_depth=0):
        """Recursively find directories containing training_summary.txt."""
        found = []
        if current_depth >= max_depth:
            return found
        
        try:
            for item in root_path.iterdir():
                if item.is_dir():
                    # Check if this directory is a snapshot (has training_summary.txt)
                    if (item / 'training_summary.txt').exists():
                        found.append(str(item))
                    else:
                        # Recursively search deeper
                        found.extend(find_snapshot_directories(item, max_depth, current_depth + 1))
        except PermissionError:
            pass  # Skip directories we can't access
        
        return found
    
    for root_path in snapshots_root_paths:
        snapshot_dirs.extend(find_snapshot_directories(root_path))
    snapshot_dirs = sorted(set(snapshot_dirs))
    
    if not snapshot_dirs:
        roots_display = ", ".join(str(p) for p in snapshots_root_paths)
        print(f"Error: No snapshot directories found in {roots_display}")
        print("  (Looking for directories containing training_summary.txt, searched recursively)")
        return
    
    print(f"Found {len(snapshot_dirs)} snapshot directories")
    
    snapshots_data = []
    skipped_count = 0
    for snapshot_dir in snapshot_dirs:
        # Check if required files exist before parsing
        snapshot_path = Path(snapshot_dir)
        summary_path = snapshot_path / 'training_summary.txt'
        csv_path = snapshot_path / 'validation_results.csv'
        
        if not summary_path.exists():
            print(f"  Skipping {snapshot_path.name}: training_summary.txt not found")
            skipped_count += 1
            continue
        
        if not csv_path.exists():
            print(f"  Skipping {snapshot_path.name}: validation_results.csv not found")
            skipped_count += 1
            continue
        
        training_dataset, validation_data, metrics = parse_snapshot_directory(snapshot_dir)
        if validation_data:
            snapshots_data.append((training_dataset, validation_data, metrics, snapshot_dir))
        else:
            print(f"  Skipping {snapshot_path.name}: No validation data found")
            skipped_count += 1
    
    print(f"Loaded {len(snapshots_data)} snapshots")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} snapshots (missing files or no validation data)")
    
    # Load coverage data
    print("\nLoading coverage data...")
    flow_coverage_lookup = load_coverage_lookup(args.coverage_csv)
    resnet_coverage_lookup = {}
    if args.coverage_resnet_csv and Path(args.coverage_resnet_csv).exists():
        resnet_coverage_lookup = load_coverage_lookup(args.coverage_resnet_csv)
    elif args.coverage_resnet_csv:
        print(f"Skipping ResNet coverage (file not found): {args.coverage_resnet_csv}")
    dino_coverage_lookup = load_coverage_lookup(args.coverage_dino_csv) if args.coverage_dino_csv else {}

    print(f"Loaded {len(flow_coverage_lookup)} flow coverage entries from {args.coverage_csv}")
    if resnet_coverage_lookup:
        print(f"Loaded {len(resnet_coverage_lookup)} resnet coverage entries from {args.coverage_resnet_csv}")
    if args.coverage_dino_csv:
        print(f"Loaded {len(dino_coverage_lookup)} DINO coverage entries from {args.coverage_dino_csv}")

    all_datasets = {item[0] for item in snapshots_data}
    if not all_datasets:
        print("Error: No training datasets found in snapshots.")
        return
    num_datasets = len(all_datasets)
    if num_datasets <= 10:
        colors = plt.cm.tab10(np.linspace(0, 1, num_datasets))
    else:
        colors = plt.cm.tab20(np.linspace(0, 1, min(num_datasets, 20)))
        if num_datasets > 20:
            colors = list(colors) * ((num_datasets // 20) + 1)
            colors = colors[:num_datasets]
    dataset_color_map = {dataset: colors[i] for i, dataset in enumerate(sorted(all_datasets))}

    feature_sets = []
    if resnet_coverage_lookup:
        feature_sets.append(("ResNet", resnet_coverage_lookup))
    if dino_coverage_lookup:
        feature_sets.append(("DINO", dino_coverage_lookup))
    if not feature_sets:
        print("Error: No feature coverage entries found (ResNet/DINO).")
        return

    for feature_label, feature_lookup in feature_sets:
        print(f"\nCollecting data points for {feature_label} coverage...")
        data_points = collect_3d_data_points(
            snapshots_data,
            flow_coverage_lookup,
            feature_lookup,
            feature_label=feature_label,
            debug=True,
        )
        print(f"Collected {len(data_points)} data points with both flow and {feature_label} coverage")

        if not data_points:
            print(f"Warning: No data points found for {feature_label} coverage.")
            continue

        print(f"\nCreating 3D scatter plots ({feature_label})...")
        create_3d_scatter_plot(
            data_points, output_path, dataset_color_map, feature_label=feature_label, zscore=False
        )
        if args.zscore:
            create_3d_scatter_plot(
                data_points, output_path, dataset_color_map, feature_label=feature_label, zscore=True
            )

        print(f"\nCreating 2D colormap plots ({feature_label})...")
        create_2d_colormap_plot(
            data_points, output_path, dataset_color_map, feature_label=feature_label, zscore=False
        )
        create_2d_colormap_plot(
            data_points, output_path, dataset_color_map, feature_label=feature_label, zscore_by_benchmark=True
        )
        if args.zscore:
            create_2d_colormap_plot(
                data_points, output_path, dataset_color_map, feature_label=feature_label, zscore=True
            )

        print(f"\nCreating faceted by benchmark plots ({feature_label})...")
        create_faceted_by_benchmark_plot(
            data_points, output_path, feature_label=feature_label, zscore_by_benchmark=False
        )
        create_faceted_by_benchmark_plot(
            data_points, output_path, feature_label=feature_label, zscore_by_benchmark=True
        )

        print(f"\nCreating faceted by training set plots ({feature_label})...")
        create_faceted_by_training_set_plot(
            data_points, output_path, dataset_color_map, feature_label=feature_label, zscore_by_benchmark=False
        )
        create_faceted_by_training_set_plot(
            data_points, output_path, dataset_color_map, feature_label=feature_label, zscore_by_benchmark=True
        )

        print(f"\nCollecting eval->train coverage data points ({feature_label})...")
        precision_data_points = collect_precision_data_points(
            snapshots_data,
            flow_coverage_lookup,
            feature_lookup,
            feature_label=feature_label,
            debug=True,
        )
        print(
            f"Collected {len(precision_data_points)} data points with both flow and "
            f"{feature_label} eval->train coverage"
        )

        if precision_data_points:
            print(f"\nCreating 2D eval->train colormap plots ({feature_label})...")
            create_2d_precision_colormap_plot(
                precision_data_points, output_path, dataset_color_map, feature_label=feature_label, zscore=False
            )
            create_2d_precision_colormap_plot(
                precision_data_points, output_path, dataset_color_map,
                feature_label=feature_label, zscore_by_benchmark=True
            )
            if args.zscore:
                create_2d_precision_colormap_plot(
                    precision_data_points, output_path, dataset_color_map,
                    feature_label=feature_label, zscore=True
                )
        else:
            print(f"  Warning: No eval->train data points found for {feature_label}.")
    
    # Load MMD data and create MMD vs PCK plots
    print("\nLoading MMD data...")
    flow_mmd_lookup = load_mmd_lookup(args.flow_mmd_csv)
    feature_mmd_lookup = {}
    if args.feature_mmd_csv and Path(args.feature_mmd_csv).exists():
        feature_mmd_lookup = load_mmd_lookup(args.feature_mmd_csv)
    elif args.feature_mmd_csv:
        print(f"Skipping ResNet MMD (file not found): {args.feature_mmd_csv}")
    dino_mmd_lookup = load_mmd_lookup(args.dino_mmd_csv) if args.dino_mmd_csv else {}

    print(f"Loaded {len(flow_mmd_lookup)} flow MMD entries from {args.flow_mmd_csv}")
    if feature_mmd_lookup:
        print(f"Loaded {len(feature_mmd_lookup)} resnet MMD entries from {args.feature_mmd_csv}")
    if args.dino_mmd_csv:
        print(f"Loaded {len(dino_mmd_lookup)} DINO MMD entries from {args.dino_mmd_csv}")

    mmd_feature_sets = []
    if feature_mmd_lookup:
        mmd_feature_sets.append(("ResNet", feature_mmd_lookup))
    if dino_mmd_lookup:
        mmd_feature_sets.append(("DINO", dino_mmd_lookup))

    for feature_label, feature_lookup in mmd_feature_sets:
        if not flow_mmd_lookup or not feature_lookup:
            print(f"  Skipping {feature_label} MMD plots (missing MMD lookup files).")
            continue

        print(f"\nCollecting MMD data points ({feature_label})...")
        mmd_data_points = collect_mmd_data_points(
            snapshots_data, flow_mmd_lookup, feature_lookup, feature_label=feature_label, debug=True
        )
        print(f"Collected {len(mmd_data_points)} data points with both flow and {feature_label} MMD")

        if not mmd_data_points:
            print(f"  Warning: No MMD data points found for {feature_label}.")
            continue

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
        print(f"\nCreating 2D MMD colormap plots ({feature_label})...")
        create_2d_mmd_colormap_plot(
            mmd_data_points, output_path, dataset_color_map_mmd,
            feature_label=feature_label, zscore=False
        )
        create_2d_mmd_colormap_plot(
            mmd_data_points, output_path, dataset_color_map_mmd,
            feature_label=feature_label, zscore_by_benchmark=True
        )
        if args.zscore:
            create_2d_mmd_colormap_plot(
                mmd_data_points, output_path, dataset_color_map_mmd,
                feature_label=feature_label, zscore=True
            )
    
    # Compare all predictors using mixed-effects regression
    has_resnet = bool(feature_mmd_lookup) and bool(resnet_coverage_lookup)
    has_dino = bool(dino_mmd_lookup) and bool(dino_coverage_lookup)
    if flow_mmd_lookup and flow_coverage_lookup and (has_resnet or has_dino):
        print("\n" + "="*80)
        print("COMPARING ALL PREDICTORS (MMD, Coverage, NN Distance)")
        print("="*80)
        all_predictors_data = collect_all_predictors_data_points(
            snapshots_data, flow_mmd_lookup, feature_mmd_lookup,
            flow_coverage_lookup, resnet_coverage_lookup,
            dino_mmd_lookup=dino_mmd_lookup if dino_mmd_lookup else None,
            dino_coverage_lookup=dino_coverage_lookup if dino_coverage_lookup else None,
            debug=True
        )
        
        if len(all_predictors_data) >= 10:
            df_all = pd.DataFrame(all_predictors_data)
            compare_predictors_with_mixed_effects(
                df_all,
                output_path=output_path,
                create_plots=True,
                encoder_offsets=args.encoder_offsets,
                model_family_offsets=args.model_family_offsets,
                encoder_interactions=args.encoder_interactions,
                model_family_interactions=args.model_family_interactions,
                predictor_set=args.predictor_set,
                standardize_mode=args.standardize_mode,
                output_suffix=args.analysis_suffix,
            )
        else:
            print(f"\nWarning: Only {len(all_predictors_data)} data points with core predictors.")
            print("  Need at least 10 points for reliable comparison.")
            print("  Make sure all CSV files (flow MMD, feature MMD,")
            print("  flow coverage, feature coverage) exist and have matching entries.")
    else:
        print("\nSkipping predictor comparison (missing required data files)")
        print("  Required: flow MMD + flow coverage, and either")
        print("    - ResNet: feature MMD + feature coverage")
        print("    - DINO: dino MMD + dino coverage")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
