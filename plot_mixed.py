#!/usr/bin/env python3
"""
Analyze and visualize performance differences between models trained with 
synthetic, spair, and mixed datasets (spair_synthetic variants).

Tests the hypothesis that mixing data with better flow distributions improves 
model performance, and shows feature coverage trade-offs.

Usage:
    python plot_mixed.py --snapshots-dir snapshots/ --coverage-csv coverage_results.csv --coverage-resnet-csv coverage_resnet_results.csv --output-dir plots_mixed/
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
)

# Import coverage loading from plot3d.py
from plot3d import load_coverage_lookup


def categorize_training_dataset(dataset_name):
    """
    Categorize training dataset into type.
    
    Args:
        dataset_name: Dataset name string (e.g., 'spair', 'synthetic', 'spair_synthetic_50_50')
        
    Returns:
        Tuple of (category, display_name):
        - category: 'synthetic', 'spair', 'mixed', or 'other'
        - display_name: Formatted name for display (e.g., 'spair_synthetic_50_50' or 'Mixed (50/50)')
    """
    if not dataset_name:
        return 'other', 'unknown'
    
    dataset_lower = dataset_name.lower()
    
    # Check for pure synthetic
    if dataset_lower == 'synthetic':
        return 'synthetic', 'Synthetic'
    
    # Check for pure spair
    if dataset_lower == 'spair':
        return 'spair', 'SPair'
    
    # Check for mixed datasets: spair_synthetic_X_Y pattern
    mixed_match = re.match(r'^spair_synthetic_(\d+)_(\d+)$', dataset_lower)
    if mixed_match:
        pct1 = int(mixed_match.group(1))
        pct2 = int(mixed_match.group(2))
        return 'mixed', f'Mixed ({pct1}/{pct2})'
    
    # Check for mixed datasets without percentages (assumed 50/50)
    if dataset_lower == 'spair_synthetic' or dataset_lower.startswith('spair_synthetic_'):
        return 'mixed', 'Mixed (50/50)'
    
    # Fallback
    return 'other', dataset_name


def extract_freeze_status(snapshot_path):
    """
    Extract freeze status from snapshot directory name.
    
    Args:
        snapshot_path: Path to snapshot directory
        
    Returns:
        'frozen' if freezeTrue, 'unfrozen' if freezeFalse, None if cannot determine
    """
    path_str = str(snapshot_path)
    path_lower = path_str.lower()
    
    if 'freezetrue' in path_lower:
        return 'frozen'
    elif 'freezefalse' in path_lower:
        return 'unfrozen'
    
    return None


def extract_model_config(snapshot_path):
    """
    Extract model configuration from snapshot directory name.
    
    Args:
        snapshot_path: Path to snapshot directory
        
    Returns:
        Tuple of (pretrained_status, freeze_status) or (None, None) if cannot determine
        pretrained_status: 'pretrained' or 'not_pretrained'
        freeze_status: 'frozen' or 'unfrozen'
    """
    path_str = str(snapshot_path)
    path_lower = path_str.lower()
    
    # Extract pretrained status
    if 'pretrainedtrue' in path_lower:
        pretrained = 'pretrained'
    elif 'pretrainedfalse' in path_lower:
        pretrained = 'not_pretrained'
    else:
        pretrained = None
    
    # Extract freeze status
    if 'freezetrue' in path_lower:
        freeze = 'frozen'
    elif 'freezefalse' in path_lower:
        freeze = 'unfrozen'
    else:
        freeze = None
    
    return pretrained, freeze


def get_model_config_label(pretrained, freeze):
    """
    Get a label for the model configuration.
    
    Args:
        pretrained: 'pretrained' or 'not_pretrained'
        freeze: 'frozen' or 'unfrozen'
        
    Returns:
        String label like 'pretrained_frozen', 'pretrained_unfrozen', etc.
    """
    if pretrained and freeze:
        return f"{pretrained}_{freeze}"
    elif pretrained:
        return pretrained
    elif freeze:
        return freeze
    return 'unknown'


def extract_pretrained_status(snapshot_path):
    """
    Extract pretrained status from snapshot directory name.
    
    Args:
        snapshot_path: Path to snapshot directory
        
    Returns:
        'pretrained' if pretrainedTrue, 'not_pretrained' if pretrainedFalse, None if cannot determine
    """
    path_str = str(snapshot_path)
    path_lower = path_str.lower()
    
    if 'pretrainedtrue' in path_lower:
        return 'pretrained'
    elif 'pretrainedfalse' in path_lower:
        return 'not_pretrained'
    
    return None


def extract_mixed_dataset_percentages(snapshot_path):
    """
    Extract mixed dataset percentages from snapshot directory name.
    
    Args:
        snapshot_path: Path to snapshot directory
        
    Returns:
        Tuple of (pct1, pct2) if found, or None if not found
        Example: ("50", "50") for spair_synthetic_50_50
    """
    path_str = str(snapshot_path)
    path_lower = path_str.lower()
    
    # Look for pattern: spair_synthetic_X_Y
    match = re.search(r'spair_synthetic_(\d+)_(\d+)', path_lower)
    if match:
        return (match.group(1), match.group(2))
    
    return None


def collect_mixed_analysis_data(snapshots_data, flow_coverage_lookup, resnet_coverage_lookup, debug=False):
    """
    Collect data points for mixed dataset analysis.
    
    Args:
        snapshots_data: List of (training_dataset_label, validation_data_dict, metrics_list, snapshot_path) tuples
        flow_coverage_lookup: Dictionary mapping (train_dataset_split, eval_dataset_split) -> flow coverage metrics
        resnet_coverage_lookup: Dictionary mapping (train_dataset_split, eval_dataset_split) -> resnet coverage metrics
        debug: If True, print debug information
        
    Returns:
        List of dicts with keys: flow_recall, resnet_recall, pck, training_dataset, 
        training_dataset_type, training_dataset_category, benchmark, freeze_status, snapshot_path
    """
    data_points = []
    missing_flow = defaultdict(int)
    missing_resnet = defaultdict(int)
    
    for training_dataset_label, _, _, snapshot_path in snapshots_data:
        # Ensure snapshot_path is a Path object
        snapshot_path_obj = Path(snapshot_path) if isinstance(snapshot_path, str) else snapshot_path
        summary_path = snapshot_path_obj / 'training_summary.txt'
        
        # Skip if summary file doesn't exist
        if not summary_path.exists():
            if debug:
                print(f"  Skipping {snapshot_path_obj.name}: training_summary.txt not found")
            continue
        
        # Get base training dataset name
        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"  Skipping {snapshot_path_obj.name}: Could not parse training dataset from summary")
            continue
        
        # For mixed datasets, try to extract percentages from directory name
        # Summary file has "spair+synthetic" but coverage CSV has "spair_synthetic_50_50"
        mixed_percentages = None
        if base_training_dataset == 'spair_synthetic':
            mixed_percentages = extract_mixed_dataset_percentages(str(snapshot_path))
            if mixed_percentages:
                # Update base_training_dataset to include percentages for lookup
                base_training_dataset = f"spair_synthetic_{mixed_percentages[0]}_{mixed_percentages[1]}"
                if debug:
                    print(f"  Found mixed dataset with percentages: {base_training_dataset} (from {snapshot_path_obj.name})")
        
        # Categorize dataset (use the updated name with percentages if available)
        dataset_category, dataset_display = categorize_training_dataset(base_training_dataset)
        
        if debug and dataset_category == 'mixed':
            print(f"  Categorized as mixed: {base_training_dataset} -> {dataset_display}")
        
        # Extract freeze status
        freeze_status = extract_freeze_status(str(snapshot_path))
        
        # Extract pretrained status
        pretrained_status = extract_pretrained_status(str(snapshot_path))
        
        # Extract model config (both pretrained and freeze)
        model_pretrained, model_freeze = extract_model_config(str(snapshot_path))
        model_config = get_model_config_label(model_pretrained, model_freeze)
        
        # Get best performance per benchmark
        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            if debug:
                print(f"  Skipping {snapshot_path_obj.name}: No best performance data found in summary")
            continue
        
        # For each benchmark, look up both flow and resnet coverage
        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()
            
            # Try multiple lookup patterns for mixed datasets
            lookup_names = [base_training_dataset]
            # If we have percentages, also try without them as fallback
            if mixed_percentages:
                lookup_names.append('spair_synthetic')
            
            flow_metrics = None
            resnet_metrics = None
            flow_key_used = None
            resnet_key_used = None
            
            # Try each lookup name
            for lookup_name in lookup_names:
                training_dataset_train = f"{lookup_name}_train"
                benchmark_test = f"{benchmark_lower}_test"
                benchmark_val = f"{benchmark_lower}_val"
                
                # Get flow recall
                if flow_metrics is None:
                    flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_test))
                    flow_key_used = (training_dataset_train, benchmark_test)
                    if flow_metrics is None:
                        flow_metrics = flow_coverage_lookup.get((training_dataset_train, benchmark_val))
                        flow_key_used = (training_dataset_train, benchmark_val)
                    if flow_metrics is None:
                        flow_metrics = flow_coverage_lookup.get((lookup_name, benchmark_lower))
                        flow_key_used = (lookup_name, benchmark_lower)
                
                # Get resnet recall
                if resnet_metrics is None:
                    resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_test))
                    resnet_key_used = (training_dataset_train, benchmark_test)
                    if resnet_metrics is None:
                        resnet_metrics = resnet_coverage_lookup.get((training_dataset_train, benchmark_val))
                        resnet_key_used = (training_dataset_train, benchmark_val)
                    if resnet_metrics is None:
                        resnet_metrics = resnet_coverage_lookup.get((lookup_name, benchmark_lower))
                        resnet_key_used = (lookup_name, benchmark_lower)
                
                # If we found both, break
                if flow_metrics and resnet_metrics:
                    break
            
            # Track missing metrics
            if debug:
                if not flow_metrics or not flow_metrics.get('recall'):
                    missing_flow[flow_key_used] += 1
                if not resnet_metrics or not resnet_metrics.get('recall'):
                    missing_resnet[resnet_key_used] += 1
            
            # Only add if we have flow coverage (required)
            if flow_metrics and 'recall' in flow_metrics and not pd.isna(flow_metrics['recall']):
                data_point = {
                    'flow_recall': flow_metrics['recall'],
                    'resnet_recall': resnet_metrics['recall'] if resnet_metrics and 'recall' in resnet_metrics else np.nan,
                    'pck': best_pck,
                    'training_dataset': training_dataset_label,
                    'base_training_dataset': base_training_dataset,
                    'training_dataset_type': dataset_display,
                    'training_dataset_category': dataset_category,
                    'benchmark': benchmark,
                    'freeze_status': freeze_status,
                    'pretrained_status': pretrained_status,
                    'model_config': model_config,
                    'model_pretrained': model_pretrained,
                    'model_freeze': model_freeze,
                    'snapshot_path': str(snapshot_path)
                }
                data_points.append(data_point)
    
    if debug and (missing_flow or missing_resnet):
        print(f"\nDebug: Missing flow recall keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print(f"\nDebug: Missing resnet recall keys (top 10):")
        for key, count in sorted(missing_resnet.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
    
    # Summary of collected data
    if data_points:
        df_summary = pd.DataFrame(data_points)
        print(f"\nData collection summary:")
        print(f"  Total data points: {len(data_points)}")
        print(f"  Training dataset types found:")
        type_counts = df_summary['training_dataset_type'].value_counts()
        for dtype, count in type_counts.items():
            print(f"    {dtype}: {count} points")
        print(f"  Categories found:")
        cat_counts = df_summary['training_dataset_category'].value_counts()
        for cat, count in cat_counts.items():
            print(f"    {cat}: {count} points")
    
    return data_points


def plot_flow_coverage_vs_pck(data_points, output_path, freeze_status=None, model_config=None):
    """
    Create scatter plot of flow coverage vs PCK.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label (e.g., 'pretrained_frozen')
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Create scatter plot of flow coverage vs PCK with regression lines per dataset type.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Optional filter for freeze status ('frozen', 'unfrozen', or None for all)
    """
    if not data_points:
        print("Warning: No data points for flow coverage vs PCK plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by model config if specified
    if model_config:
        df = df[df['model_config'] == model_config]
    elif freeze_status:
        # Backward compatibility
        df = df[df['freeze_status'] == freeze_status]
    
    if len(df) == 0:
        config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
        print(f"Warning: No data points for config={config_label}")
        return
    
    # Get unique dataset types
    dataset_types = sorted(df['training_dataset_type'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for dataset types
    colors = plt.cm.Set2(np.linspace(0, 1, len(dataset_types)))
    dataset_color_map = {dt: colors[i] for i, dt in enumerate(dataset_types)}
    
    # Plot each dataset type
    for dataset_type in dataset_types:
        subset = df[df['training_dataset_type'] == dataset_type]
        if len(subset) == 0:
            continue
        
        color = dataset_color_map[dataset_type]
        ax.scatter(subset['flow_recall'], subset['pck'],
                  color=color, label=dataset_type,
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add regression line
        if len(subset) >= 2:
            z = np.polyfit(subset['flow_recall'], subset['pck'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['flow_recall'].min(), subset['flow_recall'].max(), 100)
            ax.plot(x_line, p(x_line), '--', color=color, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Flow Recall Coverage', fontsize=12)
    ax.set_ylabel('PCK (%)', fontsize=12)
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    ax.set_title(f'Flow Coverage vs PCK Performance{config_suffix}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'flow_coverage_vs_pck{config_str}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved flow coverage vs PCK plot: {output_file}")
    plt.close()


def plot_feature_coverage_vs_pck(data_points, output_path, freeze_status='unfrozen', model_config=None):
    """
    Create scatter plot of feature coverage vs PCK.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label (e.g., 'pretrained_unfrozen')
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Create scatter plot of feature coverage vs PCK for unfrozen models.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Freeze status to filter (default 'unfrozen' where feature metrics are correlated)
    """
    if not data_points:
        print("Warning: No data points for feature coverage vs PCK plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by model config if specified
    if model_config:
        df = df[df['model_config'] == model_config]
    elif freeze_status:
        # Backward compatibility
        df = df[df['freeze_status'] == freeze_status]
    
    # Remove NaN resnet_recall
    df = df[df['resnet_recall'].notna()]
    if len(df) == 0:
        print(f"Warning: No data points for feature coverage plot with freeze_status={freeze_status}")
        return
    
    # Get unique dataset types
    dataset_types = sorted(df['training_dataset_type'].unique())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for dataset types
    colors = plt.cm.Set2(np.linspace(0, 1, len(dataset_types)))
    dataset_color_map = {dt: colors[i] for i, dt in enumerate(dataset_types)}
    
    # Plot each dataset type
    for dataset_type in dataset_types:
        subset = df[df['training_dataset_type'] == dataset_type]
        if len(subset) == 0:
            continue
        
        color = dataset_color_map[dataset_type]
        ax.scatter(subset['resnet_recall'], subset['pck'],
                  color=color, label=dataset_type,
                  s=100, alpha=0.7, edgecolors='black', linewidth=1)
        
        # Add regression line
        if len(subset) >= 2:
            z = np.polyfit(subset['resnet_recall'], subset['pck'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['resnet_recall'].min(), subset['resnet_recall'].max(), 100)
            ax.plot(x_line, p(x_line), '--', color=color, alpha=0.7, linewidth=2)
    
    ax.set_xlabel('ResNet Recall Coverage', fontsize=12)
    ax.set_ylabel('PCK (%)', fontsize=12)
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    ax.set_title(f'Feature Coverage vs PCK Performance{config_suffix}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    # Save
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'feature_coverage_vs_pck{config_str}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved feature coverage vs PCK plot: {output_file}")
    plt.close()


def plot_feature_coverage_change(data_points, output_path, freeze_status=None, model_config=None):
    """
    Create bar chart showing feature coverage by training dataset type.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Show how feature coverage changes when mixing datasets.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Optional filter for freeze status
    """
    if not data_points:
        print("Warning: No data points for feature coverage change plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by model config if specified
    if model_config:
        df = df[df['model_config'] == model_config]
    elif freeze_status:
        # Backward compatibility
        df = df[df['freeze_status'] == freeze_status]
    
    # Remove NaN resnet_recall
    df = df[df['resnet_recall'].notna()]
    
    if len(df) == 0:
        print(f"Warning: No data points for feature coverage change plot")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Group by dataset type and compute statistics
    dataset_types = sorted(df['training_dataset_type'].unique())
    means = []
    stds = []
    labels = []
    
    for dataset_type in dataset_types:
        subset = df[df['training_dataset_type'] == dataset_type]
        if len(subset) > 0:
            means.append(subset['resnet_recall'].mean())
            stds.append(subset['resnet_recall'].std())
            labels.append(dataset_type)
    
    # Create bar chart
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Color bars by category
    category_colors = {'synthetic': 'blue', 'spair': 'green', 'mixed': 'orange', 'other': 'gray'}
    for i, label in enumerate(labels):
        category = df[df['training_dataset_type'] == label]['training_dataset_category'].iloc[0]
        bars[i].set_color(category_colors.get(category, 'gray'))
    
    ax.set_xlabel('Training Dataset Type', fontsize=12)
    ax.set_ylabel('ResNet Recall Coverage (Mean ± Std)', fontsize=12)
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    ax.set_title(f'Feature Coverage by Training Dataset Type{config_suffix}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'feature_coverage_comparison{config_str}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved feature coverage comparison plot: {output_file}")
    plt.close()


def plot_flow_vs_feature_coverage(data_points, output_path, freeze_status=None, model_config=None, debug=False):
    """
    Create scatter plot showing flow coverage vs feature coverage with PCK as color.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label
        debug: If True, print debug information
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Create scatter plot showing flow coverage vs feature coverage with PCK as marker size
    to tell the story of how mixing datasets combines strengths and improves performance.
    
    Story: SPair has good feature coverage but low flow coverage (top-left).
           Synthetic has good flow coverage but low feature coverage (bottom-right).
           Mixed datasets combine both strengths (top-right) AND show better performance (larger markers).
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Optional filter for freeze status
    """
    if not data_points:
        print("Warning: No data points for flow vs feature coverage plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by freeze status if specified
    if freeze_status:
        df = df[df['freeze_status'] == freeze_status]
    
    # Remove rows with NaN values for coverage metrics or PCK
    df = df[df['flow_recall'].notna() & df['resnet_recall'].notna() & df['pck'].notna()]
    
    if len(df) == 0:
        print(f"Warning: No data points with both flow and feature coverage for freeze_status={freeze_status}")
        return
    
    # Focus on SPair, Synthetic, and Mixed datasets
    df = df[df['training_dataset_category'].isin(['spair', 'synthetic', 'mixed'])]
    
    if len(df) == 0:
        print(f"Warning: No SPair/Synthetic/Mixed data points for flow vs feature coverage plot")
        return
    
    # Rank PCK per benchmark: rank only the three dataset types (SPair, Synthetic, Mixed)
    # First, get median PCK per (benchmark, training_dataset_category) to handle multiple runs
    # Group by model_config to handle different model configurations
    df_median = df[df['training_dataset_category'].isin(['spair', 'synthetic', 'mixed'])].copy()
    df_median = df_median.groupby(['benchmark', 'training_dataset_category', 'model_config'])['pck'].median().reset_index()
    df_median.columns = ['benchmark', 'training_dataset_category', 'model_config', 'pck_median']
    
    # Now rank the three categories per benchmark based on median PCK
    df_median['pck_rank'] = df_median.groupby(['benchmark', 'model_config'])['pck_median'].rank(method='min', ascending=True).astype(int) - 1
    # Convert to 0-indexed: rank 1 -> 0 (worst), rank 2 -> 1 (middle), rank 3 -> 2 (best)
    
    # Merge the ranks back to the original dataframe (matching on benchmark, category, and model_config)
    df = df.merge(df_median[['benchmark', 'training_dataset_category', 'model_config', 'pck_rank']], 
                  on=['benchmark', 'training_dataset_category', 'model_config'], 
                  how='left')
    
    # Fill NaN with middle (1) for rows that aren't SPair/Synthetic/Mixed
    df['pck_rank'] = df['pck_rank'].fillna(1).astype(int)
    
    # Debug: print some rank values to verify
    if debug:
        print(f"\nDebug: PCK ranking (median per benchmark, category, and model_config):")
        print(df_median.sort_values(['benchmark', 'model_config', 'pck_rank']))
        print(f"\nRank value counts:")
        print(df['pck_rank'].value_counts().sort_index())
        print(f"\nSample by benchmark and model_config (showing median PCK and ranking):")
        for bench in df_median['benchmark'].unique()[:3]:
            for config in df_median['model_config'].unique():
                bench_df = df_median[(df_median['benchmark'] == bench) & (df_median['model_config'] == config)]
                if len(bench_df) > 0:
                    print(f"\n{bench} ({config}):")
                    print(bench_df[['training_dataset_category', 'pck_median', 'pck_rank']].sort_values('pck_median'))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Use discrete colors for worst, middle, best
    # Colors: worst = red/orange, middle = yellow, best = green
    performance_colors = {
        0: '#e74c3c',  # Red - worst
        1: '#f39c12',  # Orange - middle
        2: '#27ae60'   # Green - best
    }
    
    performance_labels = {
        0: 'Worst',
        1: 'Middle',
        2: 'Best'
    }
    
    # Marker shapes for different dataset categories
    category_markers = {
        'spair': 'o',        # Circle
        'synthetic': 's',    # Square
        'mixed': '^'         # Triangle
    }
    
    # Store points by benchmark for connecting lines
    benchmark_points = {}  # benchmark -> list of (x, y, category, pck) tuples
    
    # Plot each category with quantized PCK as color
    for category in ['spair', 'synthetic', 'mixed']:
        subset = df[df['training_dataset_category'] == category]
        if len(subset) == 0:
            continue
        
        marker = category_markers[category]
        
        # Plot each point with its rank color
        for idx, row in subset.iterrows():
            rank_val = row['pck_rank']
            # Convert to int if it's a float (from pandas)
            if isinstance(rank_val, (float, np.floating)):
                rank_val = int(rank_val)
            elif pd.isna(rank_val):
                rank_val = 1  # Default to middle if NaN
            else:
                rank_val = int(rank_val)
            
            # Clamp to valid range [0, 2]
            rank_val = max(0, min(2, rank_val))
            
            color = performance_colors.get(rank_val, '#95a5a6')  # Gray default
            
            ax.scatter(row['flow_recall'], row['resnet_recall'],
                      s=200,  # Fixed size
                      color=color,  # Use color parameter instead of c
                      marker=marker,
                      alpha=0.7, edgecolors='black', linewidth=1.5,
                      zorder=3)
        
        # Store points for line drawing
        for idx, row in subset.iterrows():
            benchmark = row['benchmark']
            if benchmark not in benchmark_points:
                benchmark_points[benchmark] = []
            benchmark_points[benchmark].append((
                row['flow_recall'],
                row['resnet_recall'],
                category,
                row['pck'],
                row['pck_rank']
            ))
    
    # Draw dotted lines connecting points from the same benchmark
    for benchmark, points in benchmark_points.items():
        if len(points) < 2:
            continue  # Need at least 2 points to draw a line
        
        # Sort points by category order (spair -> synthetic -> mixed)
        category_order = {'spair': 0, 'synthetic': 1, 'mixed': 2}
        sorted_points = sorted(points, key=lambda p: category_order.get(p[2], 99))
        
        # Draw lines between consecutive points
        for i in range(len(sorted_points) - 1):
            x1, y1, cat1, pck1, rank1 = sorted_points[i]
            x2, y2, cat2, pck2, rank2 = sorted_points[i + 1]
            
            ax.plot([x1, x2], [y1, y2], 
                   linestyle='--', color='gray', alpha=0.4, linewidth=1.5, zorder=1)
    
    # Add quadrant lines at median values to show the "best of both worlds" region
    median_flow = df['flow_recall'].median()
    median_feature = df['resnet_recall'].median()
    
    ax.axvline(median_flow, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
    ax.axhline(median_feature, color='gray', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
    
    # Add text annotations for quadrants (repositioned to avoid legend overlap)
    ax.text(0.02, 0.75, 'High Feature\nLow Flow\n(SPair region)', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
    ax.text(0.98, 0.02, 'High Flow\nLow Feature\n(Synthetic region)', 
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
    ax.text(0.98, 0.75, 'High Flow\nHigh Feature\n+ Better Performance\n(Best of Both Worlds)', 
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.6))
    
    ax.set_xlabel('Flow Recall Coverage', fontsize=14, fontweight='bold')
    ax.set_ylabel('ResNet Feature Recall Coverage', fontsize=14, fontweight='bold')
    
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    ax.set_title(f'Flow vs Feature Coverage: Best of Both Worlds + Performance{config_suffix}\n(PCK ranked: Red=Rank 1/Worst, Orange=Rank 2/Middle, Green=Rank 3/Best per benchmark)', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, zorder=0)
    
    # Create custom legend showing marker shapes for dataset types
    legend_elements = []
    for category in ['spair', 'synthetic', 'mixed']:
        subset = df[df['training_dataset_category'] == category]
        if len(subset) > 0:
            marker = category_markers[category]
            label = subset['training_dataset_type'].mode()[0] if len(subset) > 0 else category.capitalize()
            # Use a neutral color for the shape legend
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor='gray', markersize=14,
                                            markeredgecolor='black', markeredgewidth=2,
                                            label=label, linestyle='None'))
    
    # Add line style to legend
    legend_elements.append(plt.Line2D([0], [0], color='gray', linestyle='--', linewidth=1.5,
                                     label='Same benchmark', alpha=0.6))
    
    # Place dataset type legend in upper left
    dataset_legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
                               framealpha=0.9, title='Dataset Type (Shape)', title_fontsize=12)
    ax.add_artist(dataset_legend)  # Keep this legend when adding the next one
    
    # Add legend for performance colors
    performance_legend_elements = []
    for rank_val in [0, 1, 2]:
        color = performance_colors[rank_val]
        label = performance_labels[rank_val]
        performance_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                     markerfacecolor=color, markersize=14,
                                                     markeredgecolor='black', markeredgewidth=2,
                                                     label=label, linestyle='None'))
    
    # Add performance legend in upper right (below title)
    perf_legend = ax.legend(handles=performance_legend_elements, loc='upper right', 
                           fontsize=11, framealpha=0.9, title='PCK Rank\n(per benchmark)', title_fontsize=12)
    # Don't add_artist here - this will be the active legend
    
    # Set equal aspect ratio for better visualization
    ax.set_aspect('auto')
    
    plt.tight_layout()
    
    # Save
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'flow_vs_feature_coverage{config_str}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved flow vs feature coverage plot: {output_file}")
    plt.close()


def plot_performance_comparison(data_points, output_path, freeze_status=None, model_config=None):
    """
    Create grouped bar chart comparing average PCK by dataset type, grouped by benchmark.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Create grouped bar chart comparing average PCK by dataset type, grouped by benchmark.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Optional filter for freeze status
    """
    if not data_points:
        print("Warning: No data points for performance comparison plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by model config if specified
    if model_config:
        df = df[df['model_config'] == model_config]
    elif freeze_status:
        # Backward compatibility
        df = df[df['freeze_status'] == freeze_status]
    
    if len(df) == 0:
        print(f"Warning: No data points for performance comparison plot")
        return
    
    # Group by benchmark and dataset type
    benchmarks = sorted(df['benchmark'].unique())
    dataset_types = sorted(df['training_dataset_type'].unique())
    
    # Compute means and stds
    means = {}
    stds = {}
    for benchmark in benchmarks:
        means[benchmark] = {}
        stds[benchmark] = {}
        for dataset_type in dataset_types:
            subset = df[(df['benchmark'] == benchmark) & (df['training_dataset_type'] == dataset_type)]
            if len(subset) > 0:
                means[benchmark][dataset_type] = subset['pck'].mean()
                stds[benchmark][dataset_type] = subset['pck'].std()
            else:
                means[benchmark][dataset_type] = np.nan
                stds[benchmark][dataset_type] = np.nan
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Bar positions
    x = np.arange(len(benchmarks))
    width = 0.8 / len(dataset_types)
    
    # Color map
    colors = plt.cm.Set2(np.linspace(0, 1, len(dataset_types)))
    dataset_color_map = {dt: colors[i] for i, dt in enumerate(dataset_types)}
    
    # Plot bars
    for i, dataset_type in enumerate(dataset_types):
        offsets = x + (i - len(dataset_types)/2 + 0.5) * width
        y_vals = [means[b][dataset_type] if not np.isnan(means[b][dataset_type]) else 0 for b in benchmarks]
        y_errs = [stds[b][dataset_type] if not np.isnan(stds[b][dataset_type]) else 0 for b in benchmarks]
        
        bars = ax.bar(offsets, y_vals, width, yerr=y_errs, label=dataset_type,
                     color=dataset_color_map[dataset_type], alpha=0.7, 
                     edgecolor='black', linewidth=1, capsize=3)
    
    ax.set_xlabel('Benchmark', fontsize=12)
    ax.set_ylabel('PCK (%) - Mean ± Std', fontsize=12)
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    ax.set_title(f'PCK Performance Comparison by Training Dataset Type{config_suffix}', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=45, ha='right')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'performance_comparison{config_str}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved performance comparison plot: {output_file}")
    plt.close()


def run_mixed_effects_analysis(data_points, output_path, freeze_status=None, model_config=None):
    """
    Run mixed-effects regression analysis.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Run mixed-effects regression analysis to test if flow coverage and dataset type predict PCK.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Optional filter for freeze status
        
    Returns:
        Dictionary with analysis results
    """
    if not HAS_STATSMODELS:
        print("Warning: statsmodels not installed. Skipping mixed-effects regression.")
        return None
    
    if not data_points:
        print("Warning: No data points for mixed-effects analysis")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by freeze status if specified
    if freeze_status:
        df = df[df['freeze_status'] == freeze_status]
    
    # Remove NaN values
    df = df.dropna(subset=['flow_recall', 'pck', 'benchmark', 'training_dataset_type'])
    
    if len(df) < 10:
        print(f"Warning: Insufficient data ({len(df)} points) for mixed-effects regression")
        return None
    
    results = {}
    output_lines = []
    
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    output_lines.append("="*80)
    output_lines.append(f"MIXED-EFFECTS REGRESSION ANALYSIS{config_suffix}")
    output_lines.append("="*80)
    output_lines.append(f"\nData: {len(df)} observations across {df['benchmark'].nunique()} benchmarks")
    output_lines.append(f"Benchmarks: {', '.join(sorted(df['benchmark'].unique()))}")
    output_lines.append(f"Training dataset types: {', '.join(sorted(df['training_dataset_type'].unique()))}")
    output_lines.append("")
    
    # Model 1: Flow coverage only
    output_lines.append("-"*80)
    output_lines.append("Model 1: PCK ~ flow_recall + (1|benchmark)")
    output_lines.append("-"*80)
    try:
        model1 = smf.mixedlm("pck ~ flow_recall", data=df, groups=df["benchmark"])
        result1 = model1.fit(method='lbfgs')
        
        if result1.converged:
            flow_coef = result1.fe_params.get('flow_recall', np.nan)
            flow_pval = result1.pvalues.get('flow_recall', np.nan)
            intercept = result1.fe_params.get('Intercept', np.nan)
            
            # Handle AIC/BIC NaN values
            aic = result1.aic if hasattr(result1, 'aic') and not np.isnan(result1.aic) else np.nan
            bic = result1.bic if hasattr(result1, 'bic') and not np.isnan(result1.bic) else np.nan
            if np.isnan(aic) or np.isnan(bic):
                llf = result1.llf if hasattr(result1, 'llf') and not np.isnan(result1.llf) else np.nan
                if not np.isnan(llf):
                    n_params = len(result1.fe_params) + 1
                    n_obs = len(df)
                    aic = -2 * llf + 2 * n_params
                    bic = -2 * llf + np.log(n_obs) * n_params
            
            output_lines.append(f"Flow recall coefficient: {flow_coef:.4f}")
            output_lines.append(f"Flow recall p-value: {flow_pval:.4f}")
            output_lines.append(f"Intercept: {intercept:.4f}")
            aic_str = f"{aic:.2f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.2f}" if not np.isnan(bic) else "N/A"
            output_lines.append(f"AIC: {aic_str}, BIC: {bic_str}")
            output_lines.append(f"Significant: {'Yes' if flow_pval < 0.05 else 'No'}")
            
            results['model1'] = {
                'flow_coef': flow_coef,
                'flow_pval': flow_pval,
                'intercept': intercept,
                'aic': result1.aic,
                'bic': result1.bic,
                'converged': True
            }
        else:
            output_lines.append("Model did not converge")
            results['model1'] = {'converged': False}
    except Exception as e:
        output_lines.append(f"Error fitting model: {e}")
        results['model1'] = None
    
    output_lines.append("")
    
    # Model 2: Dataset type only
    output_lines.append("-"*80)
    output_lines.append("Model 2: PCK ~ training_dataset_type + (1|benchmark)")
    output_lines.append("-"*80)
    try:
        model2 = smf.mixedlm("pck ~ C(training_dataset_type)", data=df, groups=df["benchmark"])
        result2 = model2.fit(method='lbfgs')
        
        if result2.converged:
            output_lines.append("Dataset type coefficients (relative to reference):")
            for param in result2.fe_params.index:
                if param != 'Intercept':
                    coef = result2.fe_params[param]
                    pval = result2.pvalues[param]
                    output_lines.append(f"  {param}: {coef:.4f} (p={pval:.4f})")
            # Handle AIC/BIC NaN values
            aic = result2.aic if hasattr(result2, 'aic') and not np.isnan(result2.aic) else np.nan
            bic = result2.bic if hasattr(result2, 'bic') and not np.isnan(result2.bic) else np.nan
            if np.isnan(aic) or np.isnan(bic):
                llf = result2.llf if hasattr(result2, 'llf') and not np.isnan(result2.llf) else np.nan
                if not np.isnan(llf):
                    n_params = len(result2.fe_params) + 1
                    n_obs = len(df)
                    aic = -2 * llf + 2 * n_params
                    bic = -2 * llf + np.log(n_obs) * n_params
            
            output_lines.append(f"Intercept: {result2.fe_params.get('Intercept', np.nan):.4f}")
            aic_str = f"{aic:.2f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.2f}" if not np.isnan(bic) else "N/A"
            output_lines.append(f"AIC: {aic_str}, BIC: {bic_str}")
            
            results['model2'] = {
                'coefficients': dict(result2.fe_params),
                'pvalues': dict(result2.pvalues),
                'aic': result2.aic,
                'bic': result2.bic,
                'converged': True
            }
        else:
            output_lines.append("Model did not converge")
            results['model2'] = {'converged': False}
    except Exception as e:
        output_lines.append(f"Error fitting model: {e}")
        results['model2'] = None
    
    output_lines.append("")
    
    # Model 3: Flow coverage + Dataset type + Interaction
    output_lines.append("-"*80)
    output_lines.append("Model 3: PCK ~ flow_recall + training_dataset_type + flow_recall:training_dataset_type + (1|benchmark)")
    output_lines.append("-"*80)
    try:
        model3 = smf.mixedlm("pck ~ flow_recall * C(training_dataset_type)", data=df, groups=df["benchmark"])
        result3 = model3.fit(method='lbfgs')
        
        if result3.converged:
            output_lines.append("Coefficients:")
            for param in result3.fe_params.index:
                coef = result3.fe_params[param]
                pval = result3.pvalues[param]
                sig = '*' if pval < 0.05 else ''
                output_lines.append(f"  {param}: {coef:.4f} (p={pval:.4f}){sig}")
            
            # Handle AIC/BIC NaN values
            aic = result3.aic if hasattr(result3, 'aic') and not np.isnan(result3.aic) else np.nan
            bic = result3.bic if hasattr(result3, 'bic') and not np.isnan(result3.bic) else np.nan
            if np.isnan(aic) or np.isnan(bic):
                llf = result3.llf if hasattr(result3, 'llf') and not np.isnan(result3.llf) else np.nan
                if not np.isnan(llf):
                    n_params = len(result3.fe_params) + 1
                    n_obs = len(df)
                    aic = -2 * llf + 2 * n_params
                    bic = -2 * llf + np.log(n_obs) * n_params
            
            aic_str = f"{aic:.2f}" if not np.isnan(aic) else "N/A"
            bic_str = f"{bic:.2f}" if not np.isnan(bic) else "N/A"
            output_lines.append(f"AIC: {aic_str}, BIC: {bic_str}")
            
            results['model3'] = {
                'coefficients': dict(result3.fe_params),
                'pvalues': dict(result3.pvalues),
                'aic': result3.aic,
                'bic': result3.bic,
                'converged': True
            }
        else:
            output_lines.append("Model did not converge")
            results['model3'] = {'converged': False}
    except Exception as e:
        output_lines.append(f"Error fitting model: {e}")
        results['model3'] = None
    
    output_lines.append("")
    
    # Model 4: ResNet recall (for unfrozen models only)
    if freeze_status == 'unfrozen' or freeze_status is None:
        output_lines.append("-"*80)
        output_lines.append("Model 4: PCK ~ resnet_recall + (1|benchmark) [unfrozen models only]")
        output_lines.append("-"*80)
        
        df_resnet = df.dropna(subset=['resnet_recall'])
        if len(df_resnet) >= 10:
            try:
                model4 = smf.mixedlm("pck ~ resnet_recall", data=df_resnet, groups=df_resnet["benchmark"])
                result4 = model4.fit(method='lbfgs')
                
                if result4.converged:
                    resnet_coef = result4.fe_params.get('resnet_recall', np.nan)
                    resnet_pval = result4.pvalues.get('resnet_recall', np.nan)
                    
                    # Handle AIC/BIC NaN values
                    aic = result4.aic if hasattr(result4, 'aic') and not np.isnan(result4.aic) else np.nan
                    bic = result4.bic if hasattr(result4, 'bic') and not np.isnan(result4.bic) else np.nan
                    if np.isnan(aic) or np.isnan(bic):
                        llf = result4.llf if hasattr(result4, 'llf') and not np.isnan(result4.llf) else np.nan
                        if not np.isnan(llf):
                            n_params = len(result4.fe_params) + 1
                            n_obs = len(df_resnet)
                            aic = -2 * llf + 2 * n_params
                            bic = -2 * llf + np.log(n_obs) * n_params
                    
                    output_lines.append(f"ResNet recall coefficient: {resnet_coef:.4f}")
                    output_lines.append(f"ResNet recall p-value: {resnet_pval:.4f}")
                    aic_str = f"{aic:.2f}" if not np.isnan(aic) else "N/A"
                    bic_str = f"{bic:.2f}" if not np.isnan(bic) else "N/A"
                    output_lines.append(f"AIC: {aic_str}, BIC: {bic_str}")
                    output_lines.append(f"Significant: {'Yes' if resnet_pval < 0.05 else 'No'}")
                    
                    results['model4'] = {
                        'resnet_coef': resnet_coef,
                        'resnet_pval': resnet_pval,
                        'aic': result4.aic,
                        'bic': result4.bic,
                        'converged': True
                    }
                else:
                    output_lines.append("Model did not converge")
                    results['model4'] = {'converged': False}
            except Exception as e:
                output_lines.append(f"Error fitting model: {e}")
                results['model4'] = None
        else:
            output_lines.append(f"Insufficient data ({len(df_resnet)} points with ResNet recall)")
            results['model4'] = None
        
        output_lines.append("")
    
    # Print results
    for line in output_lines:
        print(line)
    
    # Save to file
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'mixed_dataset_analysis{config_str}.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved analysis results to: {output_file}")
    
    return results


def compare_training_dataset_types(data_points, output_path, freeze_status=None, model_config=None):
    """
    Perform pairwise comparisons between training dataset types.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: DEPRECATED - use model_config instead
        model_config: Model configuration label
    """
    # Use model_config if provided, otherwise fall back to freeze_status for backward compatibility
    config_label = model_config if model_config else (freeze_status if freeze_status else 'all')
    """
    Perform pairwise comparisons between training dataset types.
    
    Args:
        data_points: List of data point dicts
        output_path: Path object for output directory
        freeze_status: Optional filter for freeze status
        
    Returns:
        Dictionary with comparison results
    """
    if not HAS_STATSMODELS:
        print("Warning: statsmodels not installed. Skipping pairwise comparisons.")
        return None
    
    if not data_points:
        print("Warning: No data points for pairwise comparisons")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(data_points)
    
    # Filter by freeze status if specified
    if freeze_status:
        df = df[df['freeze_status'] == freeze_status]
    
    # Remove NaN values
    df = df.dropna(subset=['pck', 'benchmark', 'training_dataset_type'])
    
    if len(df) < 10:
        print(f"Warning: Insufficient data ({len(df)} points) for pairwise comparisons")
        return None
    
    # Get unique dataset types
    dataset_types = sorted(df['training_dataset_type'].unique())
    
    if len(dataset_types) < 2:
        print("Warning: Need at least 2 dataset types for comparisons")
        return None
    
    results = {}
    output_lines = []
    
    config_suffix = f" ({config_label})" if config_label != 'all' else ""
    output_lines.append("="*80)
    output_lines.append(f"PAIRWISE COMPARISONS: Training Dataset Types{config_suffix}")
    output_lines.append("="*80)
    output_lines.append(f"\nData: {len(df)} observations across {df['benchmark'].nunique()} benchmarks")
    output_lines.append("")
    
    # Run mixed-effects model with dataset type as predictor
    try:
        model = smf.mixedlm("pck ~ C(training_dataset_type)", data=df, groups=df["benchmark"])
        result = model.fit(method='lbfgs')
        
        if not result.converged:
            output_lines.append("Error: Model did not converge")
            return None
        
        # Get reference level (first dataset type)
        reference = dataset_types[0]
        output_lines.append(f"Reference level: {reference}")
        output_lines.append("")
        output_lines.append(f"{'Comparison':<40} {'Coefficient':>15} {'p-value':>15} {'Significant':>12}")
        output_lines.append("-"*82)
        
        # Extract comparisons
        for param in result.fe_params.index:
            if param != 'Intercept':
                # Extract dataset type from parameter name
                # Format: C(training_dataset_type)[T.{type}]
                match = re.search(r'\[T\.(.+)\]', param)
                if match:
                    compared_type = match.group(1)
                    coef = result.fe_params[param]
                    pval = result.pvalues[param]
                    sig = 'Yes' if pval < 0.05 else 'No'
                    
                    comparison = f"{reference} vs {compared_type}"
                    output_lines.append(f"{comparison:<40} {coef:>15.4f} {pval:>15.4f} {sig:>12}")
                    
                    results[comparison] = {
                        'coefficient': coef,
                        'pvalue': pval,
                        'significant': pval < 0.05
                    }
        
        # Handle AIC/BIC NaN values
        aic = result.aic if hasattr(result, 'aic') and not np.isnan(result.aic) else np.nan
        bic = result.bic if hasattr(result, 'bic') and not np.isnan(result.bic) else np.nan
        if np.isnan(aic) or np.isnan(bic):
            llf = result.llf if hasattr(result, 'llf') and not np.isnan(result.llf) else np.nan
            if not np.isnan(llf):
                n_params = len(result.fe_params) + 1
                n_obs = len(df)
                aic = -2 * llf + 2 * n_params
                bic = -2 * llf + np.log(n_obs) * n_params
        
        output_lines.append("")
        aic_str = f"{aic:.2f}" if not np.isnan(aic) else "N/A"
        bic_str = f"{bic:.2f}" if not np.isnan(bic) else "N/A"
        output_lines.append(f"Model AIC: {aic_str}, BIC: {bic_str}")
        
    except Exception as e:
        output_lines.append(f"Error fitting model: {e}")
        import traceback
        output_lines.append(traceback.format_exc())
        return None
    
    # Print results
    for line in output_lines:
        print(line)
    
    # Save to file
    config_str = f"_{config_label}" if config_label != 'all' else ""
    output_file = output_path / f'pairwise_comparisons{config_str}.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nSaved pairwise comparisons to: {output_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize performance differences between models trained with synthetic, spair, and mixed datasets'
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
        '--output-dir', type=str, default='plots_mixed/',
        help='Output directory for plots (default: plots_mixed/)'
    )
    parser.add_argument(
        '--filter-freeze', type=str, default=None,
        choices=['frozen', 'unfrozen'],
        help='Filter by freeze status (frozen/unfrozen). If not specified, analyzes both separately.'
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
    
    snapshot_dirs = find_snapshot_directories(snapshots_dir_path)
    
    if not snapshot_dirs:
        print(f"Error: No snapshot directories found in {args.snapshots_dir}")
        print("  (Looking for directories containing training_summary.txt, searched recursively)")
        return
    
    print(f"Found {len(snapshot_dirs)} snapshot directories")
    
    # Parse snapshots
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
    
    if not snapshots_data:
        print("Error: No valid snapshots found!")
        return
    
    # Load coverage data
    print("\nLoading coverage data...")
    flow_coverage_lookup = load_coverage_lookup(args.coverage_csv)
    resnet_coverage_lookup = load_coverage_lookup(args.coverage_resnet_csv)
    
    print(f"Loaded {len(flow_coverage_lookup)} flow coverage entries from {args.coverage_csv}")
    print(f"Loaded {len(resnet_coverage_lookup)} resnet coverage entries from {args.coverage_resnet_csv}")
    
    # Collect data points
    print("\nCollecting data points...")
    data_points = collect_mixed_analysis_data(
        snapshots_data, flow_coverage_lookup, resnet_coverage_lookup, debug=True
    )
    print(f"Collected {len(data_points)} data points")
    
    if not data_points:
        print("Error: No data points found. Make sure coverage CSV files exist and have matching entries.")
        return
    
    # Determine which model configs to analyze
    # Four configs: pretrained_frozen, pretrained_unfrozen, not_pretrained_frozen, not_pretrained_unfrozen
    model_configs = [
        ('pretrained', 'frozen'),
        ('pretrained', 'unfrozen'),
        ('not_pretrained', 'frozen'),
        ('not_pretrained', 'unfrozen')
    ]
    
    # Process each model config
    for model_pretrained, model_freeze in model_configs:
        config_label = f"{model_pretrained}_{model_freeze}"
        print("\n" + "="*80)
        print(f"ANALYZING: {config_label.upper().replace('_', ' ')} MODELS")
        print("="*80)
        
        # Filter data points by model config
        filtered_points = [
            p for p in data_points 
            if p.get('model_pretrained') == model_pretrained and p.get('model_freeze') == model_freeze
        ]
        
        if not filtered_points:
            print(f"No data points for {config_label}, skipping...")
            continue
        
        print(f"Data points: {len(filtered_points)}")
        
        # Create visualizations
        print("\nCreating visualizations...")
        plot_flow_coverage_vs_pck(filtered_points, output_path, model_config=config_label)
        plot_performance_comparison(filtered_points, output_path, model_config=config_label)
        plot_feature_coverage_change(filtered_points, output_path, model_config=config_label)
        plot_flow_vs_feature_coverage(filtered_points, output_path, model_config=config_label, debug=True)
        
        # Feature coverage plot only for unfrozen (where feature metrics are correlated)
        if model_freeze == 'unfrozen':
            plot_feature_coverage_vs_pck(filtered_points, output_path, model_config=config_label)
        
        # Run statistical analysis
        print("\nRunning statistical analysis...")
        run_mixed_effects_analysis(filtered_points, output_path, model_config=config_label)
        compare_training_dataset_types(filtered_points, output_path, model_config=config_label)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_path}")


if __name__ == "__main__":
    main()
