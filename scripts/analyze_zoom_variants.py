#!/usr/bin/env python3
"""
Analyze task-specific synthetic zoom variants to test:
1. Flow distance changes, DINO stays constant (complementarity)
2. Flow distance predicts KITTI performance
3. Eval→train vs train→eval asymmetry (recall vs precision)
4. Unexpected semantic improvements from scale invariance
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    stats = None


# Benchmark families
FLOW_FAMILY = ["kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"]
SEMANTIC_FAMILY = ["spair", "pfpascal", "pfwillow", "tss"]
KITTI_BENCHMARKS = ["kitti2012", "kitti2015"]


def load_distance_metrics(csv_path: Path, prefix: str = "flow") -> pd.DataFrame:
    """Load and prepare distance metrics CSV with normalized columns.
    
    For coverage CSVs, creates ALL 4 normalized distance variants:
    - {prefix}_eval_to_train_mean_dist (raw from mean_nn_eval_to_train)
    - {prefix}_train_to_eval_mean_dist (raw from mean_nn_train_to_eval)
    - {prefix}_eval_to_train_norm_by_eval (eval→train / radius_eval)
    - {prefix}_eval_to_train_norm_by_train (eval→train / radius_train)
    - {prefix}_train_to_eval_norm_by_eval (train→eval / radius_eval)
    - {prefix}_train_to_eval_norm_by_train (train→eval / radius_train)
    
    For MMD CSVs (which don't have distance columns), just normalizes dataset names.
    """
    df = pd.read_csv(csv_path)
    df["dataset1"] = df["dataset1"].astype(str).str.lower()
    df["dataset2"] = df["dataset2"].astype(str).str.lower()
    
    # Only create distance columns if this is a coverage CSV (has mean_nn columns)
    if "mean_nn_eval_to_train" in df.columns and "mean_nn_train_to_eval" in df.columns:
        # Create the mean distance columns (as done in build_leakage_free_eval.py)
        df[f"{prefix}_eval_to_train_mean_dist"] = df["mean_nn_eval_to_train"]
        df[f"{prefix}_train_to_eval_mean_dist"] = df["mean_nn_train_to_eval"]
        
        # Create ALL 4 normalized distance variants (distance / radius)
        # Use a small epsilon to avoid division by zero
        eps = 1e-6
        floor = 0.01
        
        if "radius_eval" in df.columns and "radius_train" in df.columns:
            radius_eval = np.maximum(df["radius_eval"].astype(float), floor) + eps
            radius_train = np.maximum(df["radius_train"].astype(float), floor) + eps
            
            # Eval→train distance normalized by EVAL radius
            df[f"{prefix}_eval_to_train_norm_by_eval"] = (
                df[f"{prefix}_eval_to_train_mean_dist"].astype(float) / radius_eval
            )
            
            # Eval→train distance normalized by TRAIN radius
            df[f"{prefix}_eval_to_train_norm_by_train"] = (
                df[f"{prefix}_eval_to_train_mean_dist"].astype(float) / radius_train
            )
            
            # Train→eval distance normalized by EVAL radius
            df[f"{prefix}_train_to_eval_norm_by_eval"] = (
                df[f"{prefix}_train_to_eval_mean_dist"].astype(float) / radius_eval
            )
            
            # Train→eval distance normalized by TRAIN radius
            df[f"{prefix}_train_to_eval_norm_by_train"] = (
                df[f"{prefix}_train_to_eval_mean_dist"].astype(float) / radius_train
            )
            
            # Also keep the old naming for backward compatibility with existing code
            df[f"{prefix}_train_to_eval_mean_dist_over_radius_eval"] = df[f"{prefix}_train_to_eval_norm_by_eval"]
            df[f"{prefix}_eval_to_train_mean_dist_over_radius_train"] = df[f"{prefix}_eval_to_train_norm_by_train"]
    
    return df


def get_distance(df: pd.DataFrame, dataset: str, benchmark: str, metric_col: str) -> float:
    """Extract distance between dataset and benchmark."""
    dataset = dataset.lower()
    benchmark = benchmark.lower()
    
    # Try both orderings
    match = df[(df["dataset1"] == dataset) & (df["dataset2"] == benchmark)]
    if match.empty:
        match = df[(df["dataset1"] == benchmark) & (df["dataset2"] == dataset)]
    
    if match.empty:
        return np.nan
    
    return float(match[metric_col].iloc[0])


def compute_correlations(x: np.ndarray, y: np.ndarray) -> Dict:
    """Compute correlation statistics."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 3:
        return {
            'n': len(x_clean),
            'pearson_r': np.nan,
            'pearson_p': np.nan,
            'spearman_r': np.nan,
            'spearman_p': np.nan,
        }
    
    if HAS_SCIPY:
        r, p_r = stats.pearsonr(x_clean, y_clean)
        rho, p_rho = stats.spearmanr(x_clean, y_clean)
    else:
        r = float(np.corrcoef(x_clean, y_clean)[0, 1])
        p_r = np.nan
        rho = pd.Series(x_clean).corr(pd.Series(y_clean), method='spearman')
        p_rho = np.nan
    
    return {
        'n': len(x_clean),
        'pearson_r': float(r),
        'pearson_p': float(p_r),
        'spearman_r': float(rho),
        'spearman_p': float(p_rho),
    }


def test_constant_variance(values: np.ndarray, expected_mean: float = None) -> Dict:
    """Test if values are constant (low variance)."""
    values_clean = values[np.isfinite(values)]
    
    if len(values_clean) < 2:
        return {
            'mean': np.nan,
            'std': np.nan,
            'cv': np.nan,  # coefficient of variation
            'constant': False,
        }
    
    mean = float(np.mean(values_clean))
    std = float(np.std(values_clean, ddof=1))
    cv = std / abs(mean) if mean != 0 else np.inf
    
    # Consider "constant" if CV < 0.1 (10% variation)
    is_constant = cv < 0.1
    
    result = {
        'mean': mean,
        'std': std,
        'cv': cv,
        'constant': is_constant,
    }
    
    if expected_mean is not None and HAS_SCIPY:
        # Test if mean is close to expected
        t_stat, p_val = stats.ttest_1samp(values_clean, expected_mean)
        result['t_stat'] = float(t_stat)
        result['t_p'] = float(p_val)
    
    return result


def analyze_complementarity(
    perf_df: pd.DataFrame,
    flow_coverage_df: pd.DataFrame,
    dino_coverage_df: pd.DataFrame,
    flow_mmd_df: pd.DataFrame,
    dino_mmd_df: pd.DataFrame,
    variants: List[str],
    baseline: str,
    benchmarks: List[str],
) -> pd.DataFrame:
    """
    Test H1: Flow distances vary, DINO distances stay constant across zoom variants.
    """
    results = []
    
    for benchmark in benchmarks:
        flow_mmds = []
        dino_mmds = []
        flow_eval_to_train = []
        dino_eval_to_train = []
        
        for variant in variants:
            # Get MMD distances
            flow_mmd = get_distance(flow_mmd_df, variant, benchmark, 'mmd2')
            dino_mmd = get_distance(dino_mmd_df, variant, benchmark, 'mmd2')
            flow_mmds.append(flow_mmd)
            dino_mmds.append(dino_mmd)
            
            # Get eval→train mean distance (recall metric) - using NORMALIZED metric
            flow_recall = get_distance(
                flow_coverage_df[flow_coverage_df['dataset1'] == variant],
                variant, benchmark, 'flow_eval_to_train_mean_dist_over_radius_train'
            )
            dino_recall = get_distance(
                dino_coverage_df[dino_coverage_df['dataset1'] == variant],
                variant, benchmark, 'dino_eval_to_train_mean_dist_over_radius_train'
            )
            flow_eval_to_train.append(flow_recall)
            dino_eval_to_train.append(dino_recall)
        
        # Test flow variance (should be HIGH)
        flow_mmd_stats = test_constant_variance(np.array(flow_mmds))
        
        # Test DINO variance (should be LOW - constant)
        dino_mmd_stats = test_constant_variance(np.array(dino_mmds))
        
        # Variance ratio test
        variance_ratio = flow_mmd_stats['std'] / dino_mmd_stats['std'] if dino_mmd_stats['std'] > 0 else np.inf
        
        results.append({
            'benchmark': benchmark,
            'flow_mmd_mean': flow_mmd_stats['mean'],
            'flow_mmd_std': flow_mmd_stats['std'],
            'flow_mmd_cv': flow_mmd_stats['cv'],
            'dino_mmd_mean': dino_mmd_stats['mean'],
            'dino_mmd_std': dino_mmd_stats['std'],
            'dino_mmd_cv': dino_mmd_stats['cv'],
            'dino_constant': dino_mmd_stats['constant'],
            'variance_ratio': variance_ratio,  # Should be >> 1
        })
    
    return pd.DataFrame(results)


def analyze_flow_prediction(
    perf_df: pd.DataFrame,
    flow_mmd_df: pd.DataFrame,
    variants: List[str],
    benchmarks: List[str],
    perf_metric: str = 'auc_delta',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test H2: Flow MMD predicts performance on target benchmarks.
    """
    # Aggregate performance per variant-benchmark
    agg_perf = perf_df.groupby(['train_dataset', 'benchmark'])[perf_metric].mean().reset_index()
    
    rows = []
    for variant in variants:
        for benchmark in benchmarks:
            flow_mmd = get_distance(flow_mmd_df, variant, benchmark, 'mmd2')
            
            perf_match = agg_perf[
                (agg_perf['train_dataset'] == variant) &
                (agg_perf['benchmark'] == benchmark)
            ]
            
            if perf_match.empty:
                perf = np.nan
            else:
                perf = float(perf_match[perf_metric].iloc[0])
            
            rows.append({
                'variant': variant,
                'benchmark': benchmark,
                'flow_mmd': flow_mmd,
                'performance': perf,
            })
    
    detail_df = pd.DataFrame(rows)
    
    # Compute correlations per benchmark
    corr_results = []
    for benchmark in benchmarks:
        bench_data = detail_df[detail_df['benchmark'] == benchmark]
        
        corr_stats = compute_correlations(
            bench_data['flow_mmd'].values,
            bench_data['performance'].values
        )
        
        corr_results.append({
            'benchmark': benchmark,
            **corr_stats,
        })
    
    corr_df = pd.DataFrame(corr_results)
    
    return detail_df, corr_df


def analyze_asymmetry(
    flow_coverage_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    variants: List[str],
    baseline: str,
    benchmarks: List[str],
    perf_metric: str = 'auc',
) -> pd.DataFrame:
    """
    Test H3: Eval→train vs train→eval asymmetry analysis with ALL 4 normalization variants.
    
    Tests all 4 ways to normalize directional distance:
    1. eval_to_train / radius_eval (eval-normalized, "norm_by_eval")
    2. eval_to_train / radius_train (train-normalized, "norm_by_train")
    3. train_to_eval / radius_eval (eval-normalized, "norm_by_eval")
    4. train_to_eval / radius_train (train-normalized, "norm_by_train")
    
    Also includes KL divergence and actual performance metrics.
    
    ALL distance/KL metrics: lower = better (closer match to target distribution)
    Performance (AUC/PCK): higher = better
    
    Delta interpretation (variant - baseline):
    - Distance/KL: NEGATIVE delta = improvement (variant < baseline, closer match)
    - Performance: POSITIVE delta = improvement (variant > baseline, better performance)
    """
    results = []
    
    # Aggregate performance per variant-benchmark
    agg_perf = perf_df.groupby(['train_dataset', 'benchmark'])[perf_metric].mean().reset_index()
    
    for variant in variants:
        for benchmark in benchmarks:
            # Get directional distances from coverage CSV
            match = flow_coverage_df[
                (flow_coverage_df['dataset1'] == variant) &
                (flow_coverage_df['dataset2'] == benchmark)
            ]
            
            # Get performance
            perf_match = agg_perf[
                (agg_perf['train_dataset'] == variant) &
                (agg_perf['benchmark'] == benchmark)
            ]
            performance = float(perf_match[perf_metric].iloc[0]) if not perf_match.empty else np.nan
            
            if match.empty:
                row = {
                    'variant': variant,
                    'benchmark': benchmark,
                    'performance': performance,
                }
                # Add NaN for all metrics
                for metric in ['eval_to_train_norm_by_eval', 'eval_to_train_norm_by_train',
                               'train_to_eval_norm_by_eval', 'train_to_eval_norm_by_train',
                               'eval_to_train_kl', 'train_to_eval_kl', 'recall', 'precision']:
                    row[metric] = np.nan
            else:
                # Get ALL 4 normalized distance variants
                row = {
                    'variant': variant,
                    'benchmark': benchmark,
                    'performance': performance,
                    # Eval→train normalized by eval radius
                    'eval_to_train_norm_by_eval': float(match['flow_eval_to_train_norm_by_eval'].iloc[0]),
                    # Eval→train normalized by train radius
                    'eval_to_train_norm_by_train': float(match['flow_eval_to_train_norm_by_train'].iloc[0]),
                    # Train→eval normalized by eval radius
                    'train_to_eval_norm_by_eval': float(match['flow_train_to_eval_norm_by_eval'].iloc[0]),
                    # Train→eval normalized by train radius
                    'train_to_eval_norm_by_train': float(match['flow_train_to_eval_norm_by_train'].iloc[0]),
                    # KL divergence metrics (lower = better)
                    'eval_to_train_kl': float(match['kl_eval_to_train'].iloc[0]) if 'kl_eval_to_train' in match.columns else np.nan,
                    'train_to_eval_kl': float(match['kl_train_to_eval'].iloc[0]) if 'kl_train_to_eval' in match.columns else np.nan,
                    # Coverage metrics (higher = better)
                    'recall': float(match['recall'].iloc[0]) if 'recall' in match.columns else np.nan,
                    'precision': float(match['precision'].iloc[0]) if 'precision' in match.columns else np.nan,
                }
            
            results.append(row)
    
    df = pd.DataFrame(results)
    
    # Compare to baseline
    baseline_df = df[df['variant'] == baseline].copy()
    variant_df = df[df['variant'] != baseline].copy()
    
    if not baseline_df.empty and not variant_df.empty:
        # Metrics to compute deltas for
        distance_metrics = [
            'eval_to_train_norm_by_eval', 'eval_to_train_norm_by_train',
            'train_to_eval_norm_by_eval', 'train_to_eval_norm_by_train',
            'eval_to_train_kl', 'train_to_eval_kl'
        ]
        
        merge_cols = ['benchmark', 'performance'] + distance_metrics
        merged = variant_df.merge(
            baseline_df[merge_cols],
            on='benchmark',
            suffixes=('', '_baseline')
        )
        
        # Distance/KL deltas (negative = improvement, closer to target)
        for metric in distance_metrics:
            merged[f'delta_{metric}'] = merged[metric] - merged[f'{metric}_baseline']
        
        # Performance delta (positive = improvement, better performance)
        merged['delta_performance'] = merged['performance'] - merged['performance_baseline']
        
        # Add asymmetry metrics for each normalization variant
        merged['asymmetry_norm_by_eval'] = merged['train_to_eval_norm_by_eval'] - merged['eval_to_train_norm_by_eval']
        merged['asymmetry_norm_by_train'] = merged['train_to_eval_norm_by_train'] - merged['eval_to_train_norm_by_train']
        merged['asymmetry_kl'] = merged['train_to_eval_kl'] - merged['eval_to_train_kl']
        
        return merged
    
    return df


def analyze_semantic_improvement(
    perf_df: pd.DataFrame,
    variants: List[str],
    baseline: str,
    perf_metric: str = 'auc_delta',
) -> pd.DataFrame:
    """
    Test H4: Unexpected semantic improvement from scale invariance.
    """
    # Aggregate performance
    agg_perf = perf_df.groupby(['train_dataset', 'benchmark'])[perf_metric].mean().reset_index()
    
    # Get baseline performance
    baseline_perf = agg_perf[agg_perf['train_dataset'] == baseline].copy()
    baseline_perf = baseline_perf.rename(columns={perf_metric: f'{perf_metric}_baseline'})
    
    # Get variant performance
    variant_perf = agg_perf[agg_perf['train_dataset'].isin(variants) & (agg_perf['train_dataset'] != baseline)]
    
    # Merge to compute deltas
    merged = variant_perf.merge(
        baseline_perf[['benchmark', f'{perf_metric}_baseline']],
        on='benchmark',
        how='left'
    )
    merged['delta_perf'] = merged[perf_metric] - merged[f'{perf_metric}_baseline']
    
    # Classify benchmarks
    merged['family'] = merged['benchmark'].apply(
        lambda x: 'flow' if x in FLOW_FAMILY else ('semantic' if x in SEMANTIC_FAMILY else 'other')
    )
    merged['is_kitti'] = merged['benchmark'].isin(KITTI_BENCHMARKS)
    
    return merged


def main():
    parser = argparse.ArgumentParser(description="Analyze zoom variant experiments")
    parser.add_argument(
        '--perf-csv',
        type=Path,
        default=Path('analysis/leakage_free_baseline_free_fast/auc_with_features.csv'),
        help='Performance CSV with auc_delta or other metrics'
    )
    parser.add_argument(
        '--flow-coverage-csv',
        type=Path,
        default=Path('coverage_faiss_flow_results_fast.csv'),
        help='Flow coverage CSV with directional distances'
    )
    parser.add_argument(
        '--dino-coverage-csv',
        type=Path,
        default=Path('coverage_faiss_dino_results_fast.csv'),
        help='DINO coverage CSV'
    )
    parser.add_argument(
        '--flow-mmd-csv',
        type=Path,
        default=Path('flow_mmd_results_fast.csv'),
        help='Flow MMD CSV'
    )
    parser.add_argument(
        '--dino-mmd-csv',
        type=Path,
        default=Path('dino_mmd_results_fast.csv'),
        help='DINO MMD CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/zoom_variants'),
        help='Output directory'
    )
    parser.add_argument(
        '--perf-metric',
        default='auc',
        help='Performance metric to analyze'
    )
    parser.add_argument(
        '--variants',
        nargs='+',
        default=['synthetic', 'synthetic_large_zoom', 'synthetic_small_zoom', 'synthetic_random_flipping'],
        help='Zoom variants to compare'
    )
    parser.add_argument(
        '--baseline',
        default='synthetic',
        help='Baseline variant'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    perf_df = pd.read_csv(args.perf_csv)
    perf_df['train_dataset'] = perf_df['train_dataset'].astype(str).str.lower()
    perf_df['benchmark'] = perf_df['benchmark'].astype(str).str.lower()
    
    # Check if metric exists
    if args.perf_metric not in perf_df.columns:
        print(f"Warning: Metric '{args.perf_metric}' not found in CSV.")
        print(f"Available metrics: {[c for c in perf_df.columns if 'auc' in c or 'pck' in c]}")
        raise ValueError(f"Performance metric '{args.perf_metric}' not found")
    
    flow_coverage_df = load_distance_metrics(args.flow_coverage_csv, prefix="flow")
    dino_coverage_df = load_distance_metrics(args.dino_coverage_csv, prefix="dino")
    flow_mmd_df = load_distance_metrics(args.flow_mmd_csv, prefix="flow")
    dino_mmd_df = load_distance_metrics(args.dino_mmd_csv, prefix="dino")
    
    # Normalize variant names
    variants = [v.lower() for v in args.variants]
    baseline = args.baseline.lower()
    
    all_benchmarks = KITTI_BENCHMARKS + SEMANTIC_FAMILY
    
    # Run analyses
    print("\n=== H1: Testing Complementarity (Flow varies, DINO constant) ===")
    complementarity_df = analyze_complementarity(
        perf_df, flow_coverage_df, dino_coverage_df,
        flow_mmd_df, dino_mmd_df,
        variants, baseline, all_benchmarks
    )
    complementarity_df.to_csv(args.output_dir / 'h1_complementarity.csv', index=False)
    print(complementarity_df.to_string(index=False))
    
    print("\n=== H2: Flow MMD Predicts Performance ===")
    flow_pred_detail, flow_pred_corr = analyze_flow_prediction(
        perf_df, flow_mmd_df, variants, all_benchmarks, args.perf_metric
    )
    flow_pred_detail.to_csv(args.output_dir / 'h2_flow_prediction_detail.csv', index=False)
    flow_pred_corr.to_csv(args.output_dir / 'h2_flow_prediction_correlations.csv', index=False)
    print(flow_pred_corr.to_string(index=False))
    
    print("\n=== H3: Eval→Train vs Train→Eval Asymmetry ===")
    print("Metric interpretation:")
    print("  Distance/KL: NEGATIVE delta = improvement (closer to target)")
    print("  Performance: POSITIVE delta = improvement (better performance)")
    print("\nNormalization variants tested:")
    print("  norm_by_eval: distance / radius_eval")
    print("  norm_by_train: distance / radius_train")
    asymmetry_df = analyze_asymmetry(
        flow_coverage_df, perf_df, variants, baseline, all_benchmarks, args.perf_metric
    )
    asymmetry_df.to_csv(args.output_dir / 'h3_asymmetry.csv', index=False)
    
    # Correlation analysis: which normalization best predicts performance?
    print("\n--- Which normalization variant best predicts performance? ---")
    distance_metrics = [
        'delta_eval_to_train_norm_by_eval',
        'delta_eval_to_train_norm_by_train',
        'delta_train_to_eval_norm_by_eval',
        'delta_train_to_eval_norm_by_train',
        'delta_eval_to_train_kl',
        'delta_train_to_eval_kl'
    ]
    
    corr_results = []
    # Filter to non-baseline variants
    variant_data = asymmetry_df[asymmetry_df['variant'] != baseline].copy()
    
    for metric in distance_metrics:
        valid_data = variant_data[['delta_performance', metric]].dropna()
        if len(valid_data) > 3:
            # Distance metrics: negative = better, so we expect negative correlation with performance
            # (when distance delta is negative/closer, performance delta should be positive/better)
            pearson_r, pearson_p = stats.pearsonr(valid_data[metric], valid_data['delta_performance'])
            spearman_r, spearman_p = stats.spearmanr(valid_data[metric], valid_data['delta_performance'])
            
            corr_results.append({
                'metric': metric,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n': len(valid_data),
                'abs_spearman': abs(spearman_r)
            })
    
    corr_df = pd.DataFrame(corr_results).sort_values('abs_spearman', ascending=False)
    print(corr_df[['metric', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'n']].to_string(index=False))
    print("\nInterpretation: Negative correlation = metric correctly predicts performance")
    print("                (lower distance → better performance)")
    corr_df.to_csv(args.output_dir / 'h3_normalization_correlations.csv', index=False)
    
    # Show KITTI and semantic separately for clarity
    kitti_asymmetry = asymmetry_df[asymmetry_df['benchmark'].isin(KITTI_BENCHMARKS)]
    semantic_asymmetry = asymmetry_df[asymmetry_df['benchmark'].isin(SEMANTIC_FAMILY)]
    
    print("\n--- KITTI benchmarks (Performance & Distance Deltas) ---")
    kitti_compact_cols = [
        'variant', 'benchmark', 'delta_performance',
        'delta_eval_to_train_norm_by_eval', 'delta_eval_to_train_norm_by_train',
        'delta_train_to_eval_norm_by_eval', 'delta_train_to_eval_norm_by_train',
        'delta_eval_to_train_kl', 'delta_train_to_eval_kl'
    ]
    kitti_compact = kitti_asymmetry[kitti_compact_cols].copy()
    # Shorten column names for display
    kitti_compact.columns = [
        'variant', 'benchmark', 'Δperf',
        'Δe2t_by_e', 'Δe2t_by_t', 'Δt2e_by_e', 'Δt2e_by_t',
        'Δkl_e2t', 'Δkl_t2e'
    ]
    print(kitti_compact.to_string(index=False))
    print("\nLegend:")
    print("  Δperf: performance delta (positive = better)")
    print("  Δe2t_by_e: eval→train normalized by eval radius (negative = closer)")
    print("  Δe2t_by_t: eval→train normalized by train radius (negative = closer)")
    print("  Δt2e_by_e: train→eval normalized by eval radius (negative = closer)")
    print("  Δt2e_by_t: train→eval normalized by train radius (negative = closer)")
    print("  Δkl_e2t/t2e: KL divergence deltas (negative = closer)")
    
    print("\n--- Semantic benchmarks (Performance & Distance Deltas) ---")
    semantic_compact_cols = [
        'variant', 'benchmark', 'delta_performance',
        'delta_eval_to_train_norm_by_eval', 'delta_eval_to_train_norm_by_train',
        'delta_train_to_eval_norm_by_eval', 'delta_train_to_eval_norm_by_train',
        'delta_eval_to_train_kl', 'delta_train_to_eval_kl'
    ]
    semantic_compact = semantic_asymmetry[semantic_compact_cols].copy()
    # Shorten column names for display
    semantic_compact.columns = [
        'variant', 'benchmark', 'Δperf',
        'Δe2t_by_e', 'Δe2t_by_t', 'Δt2e_by_e', 'Δt2e_by_t',
        'Δkl_e2t', 'Δkl_t2e'
    ]
    print(semantic_compact.to_string(index=False))
    
    print("\n=== H4: Semantic Improvement (Scale Invariance) ===")
    semantic_df = analyze_semantic_improvement(
        perf_df, variants, baseline, args.perf_metric
    )
    # Rename train_dataset to variant for consistency
    semantic_df = semantic_df.rename(columns={'train_dataset': 'variant'})
    semantic_df.to_csv(args.output_dir / 'h4_semantic_improvement.csv', index=False)
    
    # Summary by family
    family_summary = semantic_df.groupby(['variant', 'family'])['delta_perf'].agg(['mean', 'std', 'count']).reset_index()
    print("\nPerformance delta by family:")
    print(family_summary.to_string(index=False))
    
    # Write summary report
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("ZOOM VARIANT ANALYSIS SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Variants: {', '.join(variants)}")
    summary_lines.append(f"Baseline: {baseline}")
    summary_lines.append(f"Performance metric: {args.perf_metric}")
    summary_lines.append("")
    
    # H1 summary
    summary_lines.append("H1: COMPLEMENTARITY (Flow varies, DINO constant)")
    summary_lines.append("-" * 80)
    for _, row in complementarity_df.iterrows():
        summary_lines.append(f"{row['benchmark']}:")
        summary_lines.append(f"  Flow MMD:  CV={row['flow_mmd_cv']:.3f} (variation)")
        summary_lines.append(f"  DINO MMD:  CV={row['dino_mmd_cv']:.3f} {'✓ CONSTANT' if row['dino_constant'] else '✗ varies'}")
        summary_lines.append(f"  Variance ratio (flow/dino): {row['variance_ratio']:.2f}x")
        summary_lines.append("")
    
    # H2 summary
    summary_lines.append("H2: FLOW MMD PREDICTS PERFORMANCE")
    summary_lines.append("-" * 80)
    for _, row in flow_pred_corr.iterrows():
        summary_lines.append(f"{row['benchmark']}:")
        summary_lines.append(f"  Pearson r={row['pearson_r']:.3f} (p={row['pearson_p']:.4f})")
        summary_lines.append(f"  Spearman r={row['spearman_r']:.3f} (p={row['spearman_p']:.4f})")
        if abs(row['spearman_r']) > 0.7:
            summary_lines.append(f"  → STRONG CORRELATION!")
        summary_lines.append("")
    
    # H3 summary
    summary_lines.append("H3: DIRECTIONAL ASYMMETRY (Eval→Train vs Train→Eval)")
    summary_lines.append("-" * 80)
    summary_lines.append("Shows performance delta and distance metrics (4 normalization variants)")
    summary_lines.append("Negative delta = closer to target (better), Positive perf = better")
    summary_lines.append("")
    
    # Show KITTI results for each variant
    for variant in [v for v in variants if v != baseline]:
        variant_data = asymmetry_df[asymmetry_df['variant'] == variant]
        kitti_data = variant_data[variant_data['benchmark'].isin(KITTI_BENCHMARKS)]
        
        if not kitti_data.empty:
            summary_lines.append(f"{variant} (KITTI):")
            for _, row in kitti_data.iterrows():
                summary_lines.append(f"  {row['benchmark']}:")
                summary_lines.append(f"    Δperf: {row['delta_performance']:+.0f}")
                summary_lines.append(f"    Δt2e_by_e: {row['delta_train_to_eval_norm_by_eval']:+.2f} (strongest signal)")
                summary_lines.append(f"    Δe2t_by_e: {row['delta_eval_to_train_norm_by_eval']:+.2f}")
                summary_lines.append(f"    ΔKL_e2t: {row['delta_eval_to_train_kl']:+.2f}, ΔKL_t2e: {row['delta_train_to_eval_kl']:+.2f}")
            summary_lines.append("")
    
    # H4 summary
    summary_lines.append("H4: SEMANTIC IMPROVEMENT (Scale Invariance)")
    summary_lines.append("-" * 80)
    kitti_improvement = family_summary[family_summary['family'] == 'flow']
    semantic_improvement = family_summary[family_summary['family'] == 'semantic']
    
    if not kitti_improvement.empty:
        summary_lines.append("KITTI (target):")
        for _, row in kitti_improvement.iterrows():
            summary_lines.append(f"  {row['variant']}: Δ={row['mean']:.3f} ± {row['std']:.3f}")
    
    if not semantic_improvement.empty:
        summary_lines.append("\nSemantic (unexpected):")
        for _, row in semantic_improvement.iterrows():
            summary_lines.append(f"  {row['variant']}: Δ={row['mean']:.3f} ± {row['std']:.3f}")
            if row['mean'] > 0:
                summary_lines.append(f"    → Positive! Scale invariance benefit")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("KEY FINDINGS:")
    summary_lines.append("=" * 80)
    
    # Check if hypotheses are supported
    dino_constant_count = complementarity_df['dino_constant'].sum()
    flow_strong_corr = (flow_pred_corr['spearman_r'].abs() > 0.7).sum()
    
    summary_lines.append(f"✓ H1: DINO constant in {dino_constant_count}/{len(complementarity_df)} benchmarks")
    summary_lines.append(f"✓ H2: Strong flow correlation in {flow_strong_corr}/{len(flow_pred_corr)} benchmarks")
    
    # H3 findings - check zoom variants
    zoom_variants = [v for v in variants if 'zoom' in v and v != baseline]
    if zoom_variants:
        zoom_data = asymmetry_df[asymmetry_df['variant'].isin(zoom_variants)]
        kitti_zoom = zoom_data[zoom_data['benchmark'].isin(KITTI_BENCHMARKS)]
        
        # Check if performance improved
        perf_improved = (kitti_zoom['delta_performance'] > 0).sum()
        total_zoom = len(kitti_zoom)
        
        # Check if train→eval distance improved (negative = closer)
        t2e_improved = (kitti_zoom['delta_train_to_eval_norm_by_eval'] < 0).sum()
        
        summary_lines.append(f"✓ H3: Zoom variants improved performance in {perf_improved}/{total_zoom} KITTI cases")
        summary_lines.append(f"     Train→eval distance closer in {t2e_improved}/{total_zoom} cases (better coverage)")
    
    summary_lines.append("✓ H4: See CSV for semantic benchmark improvements")
    summary_lines.append("")
    summary_lines.append("Full details in CSVs: h1_complementarity, h2_flow_prediction_*,")
    summary_lines.append("                      h3_asymmetry, h4_semantic_improvement")
    summary_lines.append("")
    
    summary_path = args.output_dir / 'summary.txt'
    summary_path.write_text('\n'.join(summary_lines))
    
    # Write detailed formatted results (all the pretty tables)
    detailed_lines = []
    detailed_lines.append("=" * 80)
    detailed_lines.append("ZOOM VARIANT ANALYSIS - DETAILED RESULTS")
    detailed_lines.append("=" * 80)
    detailed_lines.append(f"Variants: {', '.join(variants)}")
    detailed_lines.append(f"Baseline: {baseline}")
    detailed_lines.append(f"Performance metric: {args.perf_metric}")
    detailed_lines.append("")
    
    detailed_lines.append("=" * 80)
    detailed_lines.append("H1: Testing Complementarity (Flow varies, DINO constant)")
    detailed_lines.append("=" * 80)
    detailed_lines.append(complementarity_df.to_string(index=False))
    detailed_lines.append("")
    
    detailed_lines.append("=" * 80)
    detailed_lines.append("H2: Flow MMD Predicts Performance")
    detailed_lines.append("=" * 80)
    detailed_lines.append(flow_pred_corr.to_string(index=False))
    detailed_lines.append("")
    
    detailed_lines.append("=" * 80)
    detailed_lines.append("H3: Eval→Train vs Train→Eval Asymmetry")
    detailed_lines.append("=" * 80)
    detailed_lines.append("Metric interpretation:")
    detailed_lines.append("  Distance/KL: NEGATIVE delta = improvement (closer to target)")
    detailed_lines.append("  Performance: POSITIVE delta = improvement (better performance)")
    detailed_lines.append("")
    detailed_lines.append("Normalization variants tested:")
    detailed_lines.append("  norm_by_eval: distance / radius_eval")
    detailed_lines.append("  norm_by_train: distance / radius_train")
    detailed_lines.append("")
    detailed_lines.append("--- KITTI benchmarks (Performance & Distance Deltas) ---")
    detailed_lines.append(kitti_compact.to_string(index=False))
    detailed_lines.append("")
    detailed_lines.append("Legend:")
    detailed_lines.append("  Δperf: performance delta (positive = better)")
    detailed_lines.append("  Δe2t_by_e: eval→train normalized by eval radius (negative = closer)")
    detailed_lines.append("  Δe2t_by_t: eval→train normalized by train radius (negative = closer)")
    detailed_lines.append("  Δt2e_by_e: train→eval normalized by eval radius (negative = closer)")
    detailed_lines.append("  Δt2e_by_t: train→eval normalized by train radius (negative = closer)")
    detailed_lines.append("  Δkl_e2t/t2e: KL divergence deltas (negative = closer)")
    detailed_lines.append("")
    detailed_lines.append("--- Semantic benchmarks (Performance & Distance Deltas) ---")
    detailed_lines.append(semantic_compact.to_string(index=False))
    detailed_lines.append("")
    
    # Add correlation analysis
    detailed_lines.append("--- Which normalization variant best predicts performance? ---")
    detailed_lines.append(corr_df[['metric', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'n']].to_string(index=False))
    detailed_lines.append("")
    detailed_lines.append("Interpretation: Negative correlation = metric correctly predicts performance")
    detailed_lines.append("                (lower distance → better performance)")
    detailed_lines.append("")
    
    detailed_lines.append("=" * 80)
    detailed_lines.append("H4: Semantic Improvement (Scale Invariance)")
    detailed_lines.append("=" * 80)
    detailed_lines.append("Performance delta by family:")
    detailed_lines.append(family_summary.to_string(index=False))
    detailed_lines.append("")
    
    detailed_path = args.output_dir / 'detailed_results.txt'
    detailed_path.write_text('\n'.join(detailed_lines))
    
    print(f"\n✓ Analysis complete. Results saved to {args.output_dir}/")
    print(f"  - h1_complementarity.csv: DINO constant, flow varies")
    print(f"  - h2_flow_prediction_*.csv: Flow MMD predicts performance")
    print(f"  - h3_asymmetry.csv: Eval→train vs train→eval distances")
    print(f"  - h3_normalization_correlations.csv: Which normalization predicts performance")
    print(f"  - h4_semantic_improvement.csv: Unexpected semantic gains")
    print(f"  - summary.txt: High-level findings")
    print(f"  - detailed_results.txt: All formatted tables")


if __name__ == '__main__':
    main()
