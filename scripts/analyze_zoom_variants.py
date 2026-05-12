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

# HOF predictor presets (for zoom intervention analysis)
HOF_PREDICTOR_PRESETS = {
    "motion": [
        "hof_eval_to_train_mean_dist",
        "hof_train_to_eval_mean_dist",
    ],
    "density": [
        "hof_density_l2",
    ],
    "combined": [
        "hof_eval_to_train_mean_dist",
        "hof_train_to_eval_mean_dist",
        "hof_density_l2",
    ],
}


def _parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset1" in df.columns and "dataset2" in df.columns:
        df["dataset1"] = df["dataset1"].astype(str).str.lower()
        df["dataset2"] = df["dataset2"].astype(str).str.lower()
        return df
    if "train_dataset" in df.columns and "eval_dataset" in df.columns:
        df["dataset1"] = df["train_dataset"].astype(str).str.lower()
        df["dataset2"] = df["eval_dataset"].astype(str).str.lower()
        return df
    raise ValueError("CSV must include dataset1/dataset2 or train_dataset/eval_dataset columns.")


def _select_k_column(df: pd.DataFrame, base: str, preferred: Tuple[int, ...]) -> str | None:
    if base in df.columns:
        return base
    for k in preferred:
        col = f"{base}_k{k}"
        if col in df.columns:
            return col
    candidates = [c for c in df.columns if c.startswith(f"{base}_k")]
    if not candidates:
        return None
    def _kval(name: str) -> int:
        try:
            return int(name.split("_k")[-1])
        except (ValueError, IndexError):
            return 1_000_000
    return sorted(candidates, key=_kval)[0]


def _resolve_hof_predictors(arg: str) -> List[str]:
    if not arg:
        return HOF_PREDICTOR_PRESETS["combined"]
    tokens = _parse_csv_list(arg)
    if not tokens:
        return HOF_PREDICTOR_PRESETS["combined"]
    resolved: List[str] = []
    for token in tokens:
        key = token.lower()
        if key in HOF_PREDICTOR_PRESETS:
            for pred in HOF_PREDICTOR_PRESETS[key]:
                if pred not in resolved:
                    resolved.append(pred)
        else:
            if token not in resolved:
                resolved.append(token)
    return resolved


def load_distance_metrics(
    csv_path: Path,
    prefix: str = "flow",
    distance_transform: str = "none",
) -> pd.DataFrame:
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
    df = _normalize_dataset_columns(df)

    # Normalize radius columns if needed (v2 coverage uses train_radius/eval_radius)
    if "radius_eval" not in df.columns and "eval_radius" in df.columns:
        df["radius_eval"] = df["eval_radius"]
    if "radius_train" not in df.columns and "train_radius" in df.columns:
        df["radius_train"] = df["train_radius"]
    
    # Only create distance columns if this is a coverage CSV (has mean_nn columns)
    mean_eval_col = _select_k_column(
        df, "mean_nn_eval_to_train", preferred=(1, 5, 10, 20, 40)
    )
    mean_train_col = _select_k_column(
        df, "mean_nn_train_to_eval", preferred=(1, 5, 10, 20, 40)
    )
    if mean_eval_col and mean_train_col:
        # Create the mean distance columns (as done in build_leakage_free_eval.py)
        df[f"{prefix}_eval_to_train_mean_dist"] = df[mean_eval_col]
        df[f"{prefix}_train_to_eval_mean_dist"] = df[mean_train_col]
        
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

        if distance_transform == "log1p":
            # Log1p transform normalized distance ratios to reduce outlier leverage.
            ratio_cols = [
                f"{prefix}_eval_to_train_norm_by_eval",
                f"{prefix}_eval_to_train_norm_by_train",
                f"{prefix}_train_to_eval_norm_by_eval",
                f"{prefix}_train_to_eval_norm_by_train",
                f"{prefix}_train_to_eval_mean_dist_over_radius_eval",
                f"{prefix}_eval_to_train_mean_dist_over_radius_train",
            ]
            for col in ratio_cols:
                if col not in df.columns:
                    continue
                values = pd.to_numeric(df[col], errors="coerce")
                values = values.where(values >= 0)
                df[f"{col}_log1p"] = np.log1p(values)
    else:
        has_mean_nn = any(
            col.startswith("mean_nn_eval_to_train") or col.startswith("mean_nn_train_to_eval")
            for col in df.columns
        )
        if has_mean_nn:
            warnings.warn(
                f"{csv_path}: missing mean_nn_eval_to_train/mean_nn_train_to_eval columns; "
                "distance-derived metrics will be unavailable."
            )
    
    return df


def load_kl_metrics(csv_path: Path, prefix: str = "hof") -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(csv_path)
    df = _normalize_dataset_columns(df)
    eval_col = _select_k_column(df, "kl_eval_to_train", preferred=(5, 10, 20, 40, 1))
    train_col = _select_k_column(df, "kl_train_to_eval", preferred=(5, 10, 20, 40, 1))
    info: Dict[str, str] = {}
    if eval_col:
        df[f"{prefix}_eval_to_train_kl"] = df[eval_col]
        info["eval_to_train_col"] = eval_col
    if train_col:
        df[f"{prefix}_train_to_eval_kl"] = df[train_col]
        info["train_to_eval_col"] = train_col
    if not eval_col or not train_col:
        warnings.warn(
            f"{csv_path}: missing KL columns (expected kl_eval_to_train/kl_train_to_eval or *_k*)."
        )
    return df, info


def _match_pair(
    df: pd.DataFrame,
    dataset: str,
    benchmark: str,
) -> Tuple[pd.Series | None, bool]:
    dataset = dataset.lower()
    benchmark = benchmark.lower()
    match = df[(df["dataset1"] == dataset) & (df["dataset2"] == benchmark)]
    if not match.empty:
        return match.iloc[0], False
    match = df[(df["dataset1"] == benchmark) & (df["dataset2"] == dataset)]
    if not match.empty:
        return match.iloc[0], True
    return None, False


def _extract_directional(
    row: pd.Series | None,
    flipped: bool,
    eval_to_train_col: str,
    train_to_eval_col: str,
) -> Tuple[float, float]:
    if row is None:
        return np.nan, np.nan
    eval_to_train = float(row[eval_to_train_col]) if eval_to_train_col in row else np.nan
    train_to_eval = float(row[train_to_eval_col]) if train_to_eval_col in row else np.nan
    if flipped:
        eval_to_train, train_to_eval = train_to_eval, eval_to_train
    return eval_to_train, train_to_eval


def _safe_match_value(match: pd.DataFrame, column: str) -> float:
    if column not in match.columns:
        return np.nan
    return float(match[column].iloc[0])


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
    if metric_col not in match.columns:
        return np.nan
    
    return float(match[metric_col].iloc[0])


def _distance_col(name: str, transform: str) -> str:
    return f"{name}_log1p" if transform == "log1p" else name


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
    distance_transform: str,
) -> pd.DataFrame:
    """
    Test H1: Flow distances vary, DINO distances stay constant across zoom variants.
    """
    results = []
    suffix = "_log1p" if distance_transform == "log1p" else ""
    
    for benchmark in benchmarks:
        flow_metric = _first_existing_col(
            flow_coverage_df,
            [
                _distance_col("flow_eval_to_train_mean_dist_over_radius_train", distance_transform),
                _distance_col("flow_eval_to_train_norm_by_train", distance_transform),
                "flow_eval_to_train_mean_dist",
            ],
        )
        dino_metric = _first_existing_col(
            dino_coverage_df,
            [
                _distance_col("dino_eval_to_train_mean_dist_over_radius_train", distance_transform),
                _distance_col("dino_eval_to_train_norm_by_train", distance_transform),
                "dino_eval_to_train_mean_dist",
            ],
        )
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
            
            flow_recall = get_distance(
                flow_coverage_df[flow_coverage_df['dataset1'] == variant],
                variant, benchmark, flow_metric or "flow_eval_to_train_mean_dist"
            )
            dino_recall = get_distance(
                dino_coverage_df[dino_coverage_df['dataset1'] == variant],
                variant, benchmark, dino_metric or "dino_eval_to_train_mean_dist"
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
    group_cols: List[str],
    perf_metric: str = 'auc_delta',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test H2: Flow MMD predicts performance on target benchmarks.
    """
    # Aggregate performance per variant-benchmark (optionally stratified)
    agg_cols = group_cols + ['train_dataset', 'benchmark']
    agg_perf = perf_df.groupby(agg_cols)[perf_metric].mean().reset_index()
    if group_cols:
        group_records = agg_perf[group_cols].drop_duplicates().to_dict('records')
    else:
        group_records = [{}]
    
    rows = []
    for group in group_records:
        if group_cols:
            mask = np.ones(len(agg_perf), dtype=bool)
            for col, val in group.items():
                mask &= agg_perf[col] == val
            group_perf = agg_perf[mask]
        else:
            group_perf = agg_perf
        for variant in variants:
            for benchmark in benchmarks:
                flow_mmd = get_distance(flow_mmd_df, variant, benchmark, 'mmd2')
                
                perf_match = group_perf[
                    (group_perf['train_dataset'] == variant) &
                    (group_perf['benchmark'] == benchmark)
                ]
                
                if perf_match.empty:
                    perf = np.nan
                else:
                    perf = float(perf_match[perf_metric].iloc[0])
                
                rows.append({
                    **group,
                    'variant': variant,
                    'benchmark': benchmark,
                    'flow_mmd': flow_mmd,
                    'performance': perf,
                })
    
    detail_df = pd.DataFrame(rows)
    
    # Compute correlations per benchmark
    corr_results = []
    for benchmark in benchmarks:
        for group in group_records:
            if group_cols:
                mask = np.ones(len(detail_df), dtype=bool)
                for col, val in group.items():
                    mask &= detail_df[col] == val
                bench_data = detail_df[mask & (detail_df['benchmark'] == benchmark)]
            else:
                bench_data = detail_df[detail_df['benchmark'] == benchmark]
            
            corr_stats = compute_correlations(
                bench_data['flow_mmd'].values,
                bench_data['performance'].values
            )
            
            corr_results.append({
                **group,
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
    group_cols: List[str],
    perf_metric: str = 'auc',
    distance_transform: str = "none",
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
    suffix = "_log1p" if distance_transform == "log1p" else ""
    
    # Aggregate performance per variant-benchmark (optionally stratified)
    agg_cols = group_cols + ['train_dataset', 'benchmark']
    agg_perf = perf_df.groupby(agg_cols)[perf_metric].mean().reset_index()
    if group_cols:
        group_records = agg_perf[group_cols].drop_duplicates().to_dict('records')
    else:
        group_records = [{}]
    
    for group in group_records:
        if group_cols:
            mask = np.ones(len(agg_perf), dtype=bool)
            for col, val in group.items():
                mask &= agg_perf[col] == val
            group_perf = agg_perf[mask]
        else:
            group_perf = agg_perf
        for variant in variants:
            for benchmark in benchmarks:
                # Get directional distances from coverage CSV
                match = flow_coverage_df[
                    (flow_coverage_df['dataset1'] == variant) &
                    (flow_coverage_df['dataset2'] == benchmark)
                ]
                
                # Get performance
                perf_match = group_perf[
                    (group_perf['train_dataset'] == variant) &
                    (group_perf['benchmark'] == benchmark)
                ]
                performance = float(perf_match[perf_metric].iloc[0]) if not perf_match.empty else np.nan
                if match.empty:
                    row = {
                        **group,
                        'variant': variant,
                        'benchmark': benchmark,
                        'performance': performance,
                    }
                    # Add NaN for all metrics
                    for metric in [
                        'eval_to_train_mean_dist', 'train_to_eval_mean_dist',
                        f'eval_to_train_norm_by_eval{suffix}', f'eval_to_train_norm_by_train{suffix}',
                        f'train_to_eval_norm_by_eval{suffix}', f'train_to_eval_norm_by_train{suffix}',
                        'eval_to_train_kl', 'train_to_eval_kl', 'recall', 'precision',
                    ]:
                        row[metric] = np.nan
                else:
                    eval_norm_by_eval_col = _distance_col('flow_eval_to_train_norm_by_eval', distance_transform)
                    eval_norm_by_train_col = _distance_col('flow_eval_to_train_norm_by_train', distance_transform)
                    train_norm_by_eval_col = _distance_col('flow_train_to_eval_norm_by_eval', distance_transform)
                    train_norm_by_train_col = _distance_col('flow_train_to_eval_norm_by_train', distance_transform)
                    # Get ALL 4 normalized distance variants + raw mean distances
                    row = {
                        **group,
                        'variant': variant,
                        'benchmark': benchmark,
                        'performance': performance,
                        # Raw mean distances (no normalization)
                        'eval_to_train_mean_dist': _safe_match_value(match, 'flow_eval_to_train_mean_dist'),
                        'train_to_eval_mean_dist': _safe_match_value(match, 'flow_train_to_eval_mean_dist'),
                        # Eval→train normalized by eval radius
                        f'eval_to_train_norm_by_eval{suffix}': _safe_match_value(
                            match, eval_norm_by_eval_col
                        ),
                        # Eval→train normalized by train radius
                        f'eval_to_train_norm_by_train{suffix}': _safe_match_value(
                            match, eval_norm_by_train_col
                        ),
                        # Train→eval normalized by eval radius
                        f'train_to_eval_norm_by_eval{suffix}': _safe_match_value(
                            match, train_norm_by_eval_col
                        ),
                        # Train→eval normalized by train radius
                        f'train_to_eval_norm_by_train{suffix}': _safe_match_value(
                            match, train_norm_by_train_col
                        ),
                        # KL divergence metrics (lower = better)
                        'eval_to_train_kl': _safe_match_value(match, 'kl_eval_to_train'),
                        'train_to_eval_kl': _safe_match_value(match, 'kl_train_to_eval'),
                        # Coverage metrics (higher = better)
                        'recall': _safe_match_value(match, 'recall'),
                        'precision': _safe_match_value(match, 'precision'),
                    }

                results.append(row)
    
    df = pd.DataFrame(results)
    
    # Compare to baseline
    baseline_df = df[df['variant'] == baseline].copy()
    variant_df = df[df['variant'] != baseline].copy()
    
    if not baseline_df.empty and not variant_df.empty:
        # Metrics to compute deltas for
        distance_metrics = [
            'eval_to_train_mean_dist', 'train_to_eval_mean_dist',
            f'eval_to_train_norm_by_eval{suffix}', f'eval_to_train_norm_by_train{suffix}',
            f'train_to_eval_norm_by_eval{suffix}', f'train_to_eval_norm_by_train{suffix}',
            'eval_to_train_kl', 'train_to_eval_kl',
        ]
        
        merge_cols = group_cols + ['benchmark', 'performance'] + distance_metrics
        merged = variant_df.merge(
            baseline_df[merge_cols],
            on=group_cols + ['benchmark'],
            suffixes=('', '_baseline')
        )
        
        # Distance/KL deltas (negative = improvement, closer to target)
        for metric in distance_metrics:
            merged[f'delta_{metric}'] = merged[metric] - merged[f'{metric}_baseline']
        
        # Performance delta (positive = improvement, better performance)
        merged['delta_performance'] = merged['performance'] - merged['performance_baseline']
        
        # Add asymmetry metrics for each normalization variant
        merged[f'asymmetry_norm_by_eval{suffix}'] = (
            merged[f'train_to_eval_norm_by_eval{suffix}']
            - merged[f'eval_to_train_norm_by_eval{suffix}']
        )
        merged[f'asymmetry_norm_by_train{suffix}'] = (
            merged[f'train_to_eval_norm_by_train{suffix}']
            - merged[f'eval_to_train_norm_by_train{suffix}']
        )
        merged['asymmetry_kl'] = merged['train_to_eval_kl'] - merged['eval_to_train_kl']
        
        return merged
    
    return df


def analyze_hof_zoom(
    hof_coverage_df: pd.DataFrame,
    perf_df: pd.DataFrame,
    variants: List[str],
    baseline: str,
    benchmarks: List[str],
    group_cols: List[str],
    predictors: List[str],
    perf_metric: str = "auc",
    hof_kl_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Zoom intervention analysis using HOF distances + density metrics.
    Avoids HOF coverage (binary) and keeps motion/density separate.
    """
    results = []
    agg_cols = group_cols + ["train_dataset", "benchmark"]
    agg_perf = perf_df.groupby(agg_cols)[perf_metric].mean().reset_index()
    if group_cols:
        group_records = agg_perf[group_cols].drop_duplicates().to_dict("records")
    else:
        group_records = [{}]

    for group in group_records:
        if group_cols:
            mask = np.ones(len(agg_perf), dtype=bool)
            for col, val in group.items():
                mask &= agg_perf[col] == val
            group_perf = agg_perf[mask]
        else:
            group_perf = agg_perf
        for variant in variants:
            for benchmark in benchmarks:
                perf_match = group_perf[
                    (group_perf["train_dataset"] == variant)
                    & (group_perf["benchmark"] == benchmark)
                ]
                performance = (
                    float(perf_match[perf_metric].iloc[0]) if not perf_match.empty else np.nan
                )

                row = {
                    **group,
                    "variant": variant,
                    "benchmark": benchmark,
                    "performance": performance,
                }

                match, flipped = _match_pair(hof_coverage_df, variant, benchmark)
                eval_to_train, train_to_eval = _extract_directional(
                    match,
                    flipped,
                    "hof_eval_to_train_mean_dist",
                    "hof_train_to_eval_mean_dist",
                )
                row["hof_eval_to_train_mean_dist"] = eval_to_train
                row["hof_train_to_eval_mean_dist"] = train_to_eval

                for density_col in ("hof_density_l2", "hof_density_l1", "hof_density_cosine"):
                    if density_col in hof_coverage_df.columns:
                        row[density_col] = float(match[density_col]) if match is not None else np.nan

                if hof_kl_df is not None:
                    kl_match, kl_flipped = _match_pair(hof_kl_df, variant, benchmark)
                    kl_eval_to_train, kl_train_to_eval = _extract_directional(
                        kl_match,
                        kl_flipped,
                        "hof_eval_to_train_kl",
                        "hof_train_to_eval_kl",
                    )
                    row["hof_eval_to_train_kl"] = kl_eval_to_train
                    row["hof_train_to_eval_kl"] = kl_train_to_eval

                results.append(row)

    df = pd.DataFrame(results)

    # Compare to baseline
    baseline_df = df[df["variant"] == baseline].copy()
    variant_df = df[df["variant"] != baseline].copy()

    if not baseline_df.empty and not variant_df.empty:
        metric_cols = [p for p in predictors if p in df.columns]
        merge_cols = group_cols + ["benchmark", "performance"] + metric_cols
        merged = variant_df.merge(
            baseline_df[merge_cols],
            on=group_cols + ["benchmark"],
            suffixes=("", "_baseline"),
        )
        for metric in metric_cols:
            merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_baseline"]
        merged["delta_performance"] = merged["performance"] - merged["performance_baseline"]
        return merged

    return df


def _bool_rate(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        return np.nan
    return float(vals.mean())


def build_intervention_consistency(
    asymmetry_df: pd.DataFrame,
    hof_zoom_df: pd.DataFrame | None,
    group_cols: List[str],
    eps: float = 1e-9,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build paired intervention-style consistency tables.
    - perf improvement: delta_performance > 0
    - distance improvement (for mean distances): delta < 0
    """
    base_cols = group_cols + ["variant", "benchmark"]
    required = ["delta_performance", "delta_train_to_eval_mean_dist", "delta_eval_to_train_mean_dist"]
    missing = [c for c in required if c not in asymmetry_df.columns]
    if missing:
        raise ValueError(f"Asymmetry table missing required columns: {missing}")

    pair = asymmetry_df[base_cols + required].copy()
    pair["perf_improved"] = np.where(
        np.isfinite(pair["delta_performance"]),
        pair["delta_performance"] > eps,
        np.nan,
    )
    pair["flow_t2e_improved"] = np.where(
        np.isfinite(pair["delta_train_to_eval_mean_dist"]),
        pair["delta_train_to_eval_mean_dist"] < -eps,
        np.nan,
    )
    pair["flow_e2t_improved"] = np.where(
        np.isfinite(pair["delta_eval_to_train_mean_dist"]),
        pair["delta_eval_to_train_mean_dist"] < -eps,
        np.nan,
    )
    pair["flow_t2e_concordant"] = np.where(
        np.isfinite(pair["perf_improved"]) & np.isfinite(pair["flow_t2e_improved"]),
        pair["perf_improved"] == pair["flow_t2e_improved"],
        np.nan,
    )
    pair["flow_e2t_concordant"] = np.where(
        np.isfinite(pair["perf_improved"]) & np.isfinite(pair["flow_e2t_improved"]),
        pair["perf_improved"] == pair["flow_e2t_improved"],
        np.nan,
    )

    if hof_zoom_df is not None and "delta_performance" in hof_zoom_df.columns:
        hof_keys = base_cols.copy()
        hof_delta_cols = [
            c
            for c in [
                "delta_hof_train_to_eval_mean_dist",
                "delta_hof_eval_to_train_mean_dist",
                "delta_hof_density_l2",
                "delta_hof_density_l1",
                "delta_hof_density_cosine",
            ]
            if c in hof_zoom_df.columns
        ]
        if hof_delta_cols:
            hof_part = hof_zoom_df[hof_keys + hof_delta_cols].copy().drop_duplicates()
            pair = pair.merge(hof_part, on=hof_keys, how="left")
            if "delta_hof_train_to_eval_mean_dist" in pair.columns:
                pair["hof_t2e_improved"] = np.where(
                    np.isfinite(pair["delta_hof_train_to_eval_mean_dist"]),
                    pair["delta_hof_train_to_eval_mean_dist"] < -eps,
                    np.nan,
                )
                pair["hof_t2e_concordant"] = np.where(
                    np.isfinite(pair["perf_improved"]) & np.isfinite(pair["hof_t2e_improved"]),
                    pair["perf_improved"] == pair["hof_t2e_improved"],
                    np.nan,
                )
            if "delta_hof_eval_to_train_mean_dist" in pair.columns:
                pair["hof_e2t_improved"] = np.where(
                    np.isfinite(pair["delta_hof_eval_to_train_mean_dist"]),
                    pair["delta_hof_eval_to_train_mean_dist"] < -eps,
                    np.nan,
                )
                pair["hof_e2t_concordant"] = np.where(
                    np.isfinite(pair["perf_improved"]) & np.isfinite(pair["hof_e2t_improved"]),
                    pair["perf_improved"] == pair["hof_e2t_improved"],
                    np.nan,
                )
            if "delta_hof_density_l2" in pair.columns:
                pair["hof_density_l2_improved"] = np.where(
                    np.isfinite(pair["delta_hof_density_l2"]),
                    pair["delta_hof_density_l2"] < -eps,
                    np.nan,
                )
                pair["hof_density_l2_concordant"] = np.where(
                    np.isfinite(pair["perf_improved"]) & np.isfinite(pair["hof_density_l2_improved"]),
                    pair["perf_improved"] == pair["hof_density_l2_improved"],
                    np.nan,
                )

    summary_cols = [
        "perf_improved",
        "flow_t2e_concordant",
        "flow_e2t_concordant",
        "hof_t2e_concordant",
        "hof_e2t_concordant",
        "hof_density_l2_concordant",
    ]
    summary_input_cols = [c for c in summary_cols if c in pair.columns]
    summary = (
        pair.groupby(["variant", "benchmark"], dropna=False)[summary_input_cols]
        .apply(
            lambda g: pd.Series(
                {
                    "n": int(len(g)),
                    "perf_gain_rate": _bool_rate(g["perf_improved"]),
                    "flow_t2e_concordance": _bool_rate(g["flow_t2e_concordant"]),
                    "flow_e2t_concordance": _bool_rate(g["flow_e2t_concordant"]),
                    "asymmetry_advantage_flow_t2e_minus_e2t": _bool_rate(g["flow_t2e_concordant"])
                    - _bool_rate(g["flow_e2t_concordant"]),
                    "hof_t2e_concordance": _bool_rate(g["hof_t2e_concordant"])
                    if "hof_t2e_concordant" in g.columns
                    else np.nan,
                    "hof_e2t_concordance": _bool_rate(g["hof_e2t_concordant"])
                    if "hof_e2t_concordant" in g.columns
                    else np.nan,
                    "hof_density_l2_concordance": _bool_rate(g["hof_density_l2_concordant"])
                    if "hof_density_l2_concordant" in g.columns
                    else np.nan,
                }
            )
        )
        .reset_index()
    )

    return pair, summary


def build_motion_vs_appearance_intervention(
    perf_df: pd.DataFrame,
    dino_coverage_df: pd.DataFrame | None,
    variants: List[str],
    baseline: str,
    group_cols: List[str],
    perf_metric: str,
    flow_motion_col: str = "flow_train_to_eval_eps1px",
    hof_motion_col: str = "hof_train_to_eval_mean_dist",
    appearance_col: str = "dino_train_to_eval_mean_dist",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build intervention tables focused on the core claim:
    motion proxies (flow/HOF) vs appearance proxy (DINO).
    """
    value_cols = [perf_metric]
    for col in (flow_motion_col, hof_motion_col):
        if col in perf_df.columns:
            value_cols.append(col)
    has_appearance_in_perf = appearance_col in perf_df.columns
    if has_appearance_in_perf:
        value_cols.append(appearance_col)

    agg_cols = group_cols + ["train_dataset", "benchmark"]
    agg = (
        perf_df[agg_cols + value_cols]
        .groupby(agg_cols, dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )

    if appearance_col not in agg.columns:
        agg[appearance_col] = np.nan

    # Fallback: fill appearance proxy from DINO coverage if perf CSV has empty DINO columns.
    if agg[appearance_col].notna().sum() == 0 and dino_coverage_df is not None:
        if {
            "dino_eval_to_train_mean_dist",
            "dino_train_to_eval_mean_dist",
            "dataset1",
            "dataset2",
        }.issubset(dino_coverage_df.columns):
            def _lookup_appearance(train_ds: str, bench: str) -> float:
                match, flipped = _match_pair(dino_coverage_df, str(train_ds), str(bench))
                if appearance_col == "dino_eval_to_train_mean_dist":
                    eval_to_train, _ = _extract_directional(
                        match, flipped, "dino_eval_to_train_mean_dist", "dino_train_to_eval_mean_dist"
                    )
                    return eval_to_train
                _, train_to_eval = _extract_directional(
                    match, flipped, "dino_eval_to_train_mean_dist", "dino_train_to_eval_mean_dist"
                )
                return train_to_eval

            agg[appearance_col] = agg.apply(
                lambda r: _lookup_appearance(r["train_dataset"], r["benchmark"]),
                axis=1,
            )

    if appearance_col not in value_cols:
        value_cols.append(appearance_col)

    base = agg[agg["train_dataset"] == baseline].copy()
    base = base.rename(
        columns={col: f"{col}_baseline" for col in value_cols}
    )
    var = agg[(agg["train_dataset"].isin(variants)) & (agg["train_dataset"] != baseline)].copy()

    merge_keys = group_cols + ["benchmark"]
    merged = var.merge(base[merge_keys + [f"{col}_baseline" for col in value_cols]], on=merge_keys, how="left")

    merged["delta_performance"] = merged[perf_metric] - merged[f"{perf_metric}_baseline"]
    merged["perf_improved"] = np.where(
        np.isfinite(merged["delta_performance"]),
        merged["delta_performance"] > 0,
        np.nan,
    )

    # Flow eps recall-style predictor: higher is better.
    if flow_motion_col in value_cols:
        merged["delta_flow_motion"] = merged[flow_motion_col] - merged[f"{flow_motion_col}_baseline"]
        merged["flow_motion_improved"] = np.where(
            np.isfinite(merged["delta_flow_motion"]),
            merged["delta_flow_motion"] > 0,
            np.nan,
        )
        merged["flow_motion_concordance"] = np.where(
            np.isfinite(merged["perf_improved"]) & np.isfinite(merged["flow_motion_improved"]),
            merged["perf_improved"] == merged["flow_motion_improved"],
            np.nan,
        )

    # HOF distance predictor: lower is better.
    if hof_motion_col in value_cols:
        merged["delta_hof_motion"] = merged[hof_motion_col] - merged[f"{hof_motion_col}_baseline"]
        merged["hof_motion_improved"] = np.where(
            np.isfinite(merged["delta_hof_motion"]),
            merged["delta_hof_motion"] < 0,
            np.nan,
        )
        merged["hof_motion_concordance"] = np.where(
            np.isfinite(merged["perf_improved"]) & np.isfinite(merged["hof_motion_improved"]),
            merged["perf_improved"] == merged["hof_motion_improved"],
            np.nan,
        )

    # Appearance proxy shift magnitude: lower means appearance is better held constant.
    if appearance_col in value_cols:
        merged["delta_appearance"] = merged[appearance_col] - merged[f"{appearance_col}_baseline"]
        merged["appearance_shift_abs"] = np.abs(merged["delta_appearance"])

    merged["variant"] = merged["train_dataset"]
    merged["family"] = np.where(
        merged["benchmark"].isin(KITTI_BENCHMARKS),
        "kitti",
        np.where(merged["benchmark"].isin(SEMANTIC_FAMILY), "semantic", "other"),
    )

    def _summary(df: pd.DataFrame) -> pd.DataFrame:
        agg_map = {
            "n": ("benchmark", "size"),
            "delta_perf_mean": ("delta_performance", "mean"),
            "perf_gain_rate": ("perf_improved", "mean"),
        }
        if "delta_flow_motion" in df.columns:
            agg_map["flow_motion_delta_mean"] = ("delta_flow_motion", "mean")
        if "flow_motion_concordance" in df.columns:
            agg_map["flow_motion_concordance"] = ("flow_motion_concordance", "mean")
        if "delta_hof_motion" in df.columns:
            agg_map["hof_motion_delta_mean"] = ("delta_hof_motion", "mean")
        if "hof_motion_concordance" in df.columns:
            agg_map["hof_motion_concordance"] = ("hof_motion_concordance", "mean")
        if "appearance_shift_abs" in df.columns:
            agg_map["appearance_shift_abs_mean"] = ("appearance_shift_abs", "mean")
        return df.groupby("variant", dropna=False).agg(**agg_map).reset_index()

    all_non_self = merged[merged["benchmark"] != baseline].copy()
    kitti_only = all_non_self[all_non_self["benchmark"].isin(KITTI_BENCHMARKS)].copy()

    return merged, _summary(all_non_self), _summary(kitti_only)


def build_asymmetry_vs_mmd_intervention(
    perf_df: pd.DataFrame,
    flow_mmd_df: pd.DataFrame,
    variants: List[str],
    baseline: str,
    group_cols: List[str],
    perf_metric: str,
    flow_t2e_col: str = "flow_train_to_eval_eps1px",
    flow_e2t_col: str = "flow_eval_to_train_eps1p5px",
    hof_t2e_col: str = "hof_train_to_eval_mean_dist",
    hof_e2t_col: str = "hof_eval_to_train_mean_dist",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build predictor-native directional asymmetry vs symmetric-MMD intervention tables.
    """
    agg_cols = group_cols + ["train_dataset", "benchmark"]
    value_cols = [perf_metric]
    for col in (flow_t2e_col, flow_e2t_col, hof_t2e_col, hof_e2t_col):
        if col in perf_df.columns:
            value_cols.append(col)

    perf_agg = (
        perf_df[agg_cols + value_cols]
        .groupby(agg_cols, dropna=False)
        .mean(numeric_only=True)
        .reset_index()
    )
    base_perf = perf_agg[perf_agg["train_dataset"] == baseline].copy()
    base_perf = base_perf.rename(columns={col: f"{col}_baseline" for col in value_cols})
    var_perf = perf_agg[(perf_agg["train_dataset"].isin(variants)) & (perf_agg["train_dataset"] != baseline)].copy()
    merge_keys = group_cols + ["benchmark"]
    merged = var_perf.merge(
        base_perf[merge_keys + [f"{col}_baseline" for col in value_cols]],
        on=merge_keys,
        how="left",
    )
    merged["delta_performance"] = merged[perf_metric] - merged[f"{perf_metric}_baseline"]
    merged["perf_improved"] = np.where(
        np.isfinite(merged["delta_performance"]),
        merged["delta_performance"] > 0,
        np.nan,
    )

    mmd_col = _first_existing_col(flow_mmd_df, ["mmd2", "mmd", "flow_mmd"])
    if mmd_col is None:
        merged["delta_flow_mmd"] = np.nan
    else:
        merged["delta_flow_mmd"] = merged.apply(
            lambda r: get_distance(flow_mmd_df, str(r["train_dataset"]), str(r["benchmark"]), mmd_col)
            - get_distance(flow_mmd_df, baseline, str(r["benchmark"]), mmd_col),
            axis=1,
        )

    if flow_t2e_col in perf_agg.columns:
        merged[f"delta_{flow_t2e_col}"] = merged[flow_t2e_col] - merged[f"{flow_t2e_col}_baseline"]
    else:
        merged[f"delta_{flow_t2e_col}"] = np.nan
    if flow_e2t_col in perf_agg.columns:
        merged[f"delta_{flow_e2t_col}"] = merged[flow_e2t_col] - merged[f"{flow_e2t_col}_baseline"]
    else:
        merged[f"delta_{flow_e2t_col}"] = np.nan
    if hof_t2e_col in perf_agg.columns:
        merged[f"delta_{hof_t2e_col}"] = merged[hof_t2e_col] - merged[f"{hof_t2e_col}_baseline"]
    else:
        merged[f"delta_{hof_t2e_col}"] = np.nan
    if hof_e2t_col in perf_agg.columns:
        merged[f"delta_{hof_e2t_col}"] = merged[hof_e2t_col] - merged[f"{hof_e2t_col}_baseline"]
    else:
        merged[f"delta_{hof_e2t_col}"] = np.nan

    for col in (
        f"delta_{flow_t2e_col}",
        f"delta_{flow_e2t_col}",
        f"delta_{hof_t2e_col}",
        f"delta_{hof_e2t_col}",
        "delta_flow_mmd",
        "delta_performance",
    ):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    # Predictor-native directionality:
    # - flow eps coverage metrics: higher is better
    # - HOF mean distance metrics: lower is better
    merged[f"{flow_t2e_col}_improved"] = np.where(
        np.isfinite(merged[f"delta_{flow_t2e_col}"]),
        merged[f"delta_{flow_t2e_col}"] > 0,
        np.nan,
    )
    merged[f"{flow_e2t_col}_improved"] = np.where(
        np.isfinite(merged[f"delta_{flow_e2t_col}"]),
        merged[f"delta_{flow_e2t_col}"] > 0,
        np.nan,
    )
    merged[f"{hof_t2e_col}_improved"] = np.where(
        np.isfinite(merged[f"delta_{hof_t2e_col}"]),
        merged[f"delta_{hof_t2e_col}"] < 0,
        np.nan,
    )
    merged[f"{hof_e2t_col}_improved"] = np.where(
        np.isfinite(merged[f"delta_{hof_e2t_col}"]),
        merged[f"delta_{hof_e2t_col}"] < 0,
        np.nan,
    )
    merged["flow_mmd_improved"] = np.where(
        np.isfinite(merged["delta_flow_mmd"]),
        merged["delta_flow_mmd"] < 0,
        np.nan,
    )
    merged[f"{flow_t2e_col}_concordance"] = np.where(
        np.isfinite(merged["perf_improved"]) & np.isfinite(merged[f"{flow_t2e_col}_improved"]),
        merged["perf_improved"] == merged[f"{flow_t2e_col}_improved"],
        np.nan,
    )
    merged[f"{flow_e2t_col}_concordance"] = np.where(
        np.isfinite(merged["perf_improved"]) & np.isfinite(merged[f"{flow_e2t_col}_improved"]),
        merged["perf_improved"] == merged[f"{flow_e2t_col}_improved"],
        np.nan,
    )
    merged[f"{hof_t2e_col}_concordance"] = np.where(
        np.isfinite(merged["perf_improved"]) & np.isfinite(merged[f"{hof_t2e_col}_improved"]),
        merged["perf_improved"] == merged[f"{hof_t2e_col}_improved"],
        np.nan,
    )
    merged[f"{hof_e2t_col}_concordance"] = np.where(
        np.isfinite(merged["perf_improved"]) & np.isfinite(merged[f"{hof_e2t_col}_improved"]),
        merged["perf_improved"] == merged[f"{hof_e2t_col}_improved"],
        np.nan,
    )
    merged["flow_mmd_concordance"] = np.where(
        np.isfinite(merged["perf_improved"]) & np.isfinite(merged["flow_mmd_improved"]),
        merged["perf_improved"] == merged["flow_mmd_improved"],
        np.nan,
    )
    directional_predictors = [flow_t2e_col, flow_e2t_col, hof_t2e_col, hof_e2t_col]
    for predictor_col in directional_predictors:
        conc_col = f"{predictor_col}_concordance"
        valid = np.isfinite(merged[conc_col]) & np.isfinite(merged["flow_mmd_concordance"])
        merged[f"mmd_failure_exposed_by_{predictor_col}"] = np.where(
            valid,
            (merged[conc_col] == 1) & (merged["flow_mmd_concordance"] == 0),
            np.nan,
        )
        merged[f"mmd_only_correct_vs_{predictor_col}"] = np.where(
            valid,
            (merged[conc_col] == 0) & (merged["flow_mmd_concordance"] == 1),
            np.nan,
        )
        merged[f"both_correct_with_mmd_vs_{predictor_col}"] = np.where(
            valid,
            (merged[conc_col] == 1) & (merged["flow_mmd_concordance"] == 1),
            np.nan,
        )
        merged[f"both_wrong_with_mmd_vs_{predictor_col}"] = np.where(
            valid,
            (merged[conc_col] == 0) & (merged["flow_mmd_concordance"] == 0),
            np.nan,
        )
        merged[f"predictor_mmd_disagree_{predictor_col}"] = np.where(
            valid,
            merged[conc_col] != merged["flow_mmd_concordance"],
            np.nan,
        )
    merged["variant"] = merged["train_dataset"]
    merged["family"] = np.where(
        merged["benchmark"].isin(KITTI_BENCHMARKS),
        "kitti",
        np.where(merged["benchmark"].isin(SEMANTIC_FAMILY), "semantic", "other"),
    )

    def _summary(df: pd.DataFrame) -> pd.DataFrame:
        agg_map = {
            "n": ("benchmark", "size"),
            "delta_perf_mean": ("delta_performance", "mean"),
            "flow_mmd_delta_mean": ("delta_flow_mmd", "mean"),
            "flow_mmd_concordance": ("flow_mmd_concordance", "mean"),
        }
        for predictor_col in directional_predictors:
            agg_map[f"delta_{predictor_col}_mean"] = (f"delta_{predictor_col}", "mean")
            agg_map[f"{predictor_col}_concordance"] = (f"{predictor_col}_concordance", "mean")
            agg_map[f"mmd_failure_exposed_by_{predictor_col}_rate"] = (
                f"mmd_failure_exposed_by_{predictor_col}",
                "mean",
            )
            agg_map[f"mmd_only_correct_vs_{predictor_col}_rate"] = (
                f"mmd_only_correct_vs_{predictor_col}",
                "mean",
            )
            agg_map[f"both_correct_with_mmd_vs_{predictor_col}_rate"] = (
                f"both_correct_with_mmd_vs_{predictor_col}",
                "mean",
            )
            agg_map[f"both_wrong_with_mmd_vs_{predictor_col}_rate"] = (
                f"both_wrong_with_mmd_vs_{predictor_col}",
                "mean",
            )
            agg_map[f"predictor_mmd_disagree_{predictor_col}_rate"] = (
                f"predictor_mmd_disagree_{predictor_col}",
                "mean",
            )

        out = df.groupby("variant", dropna=False).agg(**agg_map).reset_index()
        out["mmd_failure_rate"] = 1.0 - out["flow_mmd_concordance"]
        for predictor_col in directional_predictors:
            out[f"asymmetry_adv_{predictor_col}_minus_mmd"] = (
                out[f"{predictor_col}_concordance"] - out["flow_mmd_concordance"]
            )
        return out

    non_self = merged[merged["benchmark"] != baseline].copy()
    kitti = non_self[non_self["benchmark"].isin(KITTI_BENCHMARKS)].copy()
    return merged, _summary(non_self), _summary(kitti)


def analyze_semantic_improvement(
    perf_df: pd.DataFrame,
    variants: List[str],
    baseline: str,
    group_cols: List[str],
    perf_metric: str = 'auc_delta',
) -> pd.DataFrame:
    """
    Test H4: Unexpected semantic improvement from scale invariance.
    """
    # Aggregate performance (optionally stratified)
    agg_cols = group_cols + ['train_dataset', 'benchmark']
    agg_perf = perf_df.groupby(agg_cols)[perf_metric].mean().reset_index()
    
    # Get baseline performance
    baseline_perf = agg_perf[agg_perf['train_dataset'] == baseline].copy()
    baseline_perf = baseline_perf.rename(columns={perf_metric: f'{perf_metric}_baseline'})
    
    # Get variant performance
    variant_perf = agg_perf[agg_perf['train_dataset'].isin(variants) & (agg_perf['train_dataset'] != baseline)]
    
    # Merge to compute deltas
    merge_cols = group_cols + ['benchmark']
    merged = variant_perf.merge(
        baseline_perf[merge_cols + [f'{perf_metric}_baseline']],
        on=merge_cols,
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
        '--hof-coverage-csv',
        type=Path,
        default=Path('analysis/coverage_v2_hof_full_occ.csv'),
        help='HOF coverage CSV (optional; v2 coverage format)'
    )
    parser.add_argument(
        '--hof-kl-csv',
        type=Path,
        default=Path('analysis/kl_v2_hof_full_occ.csv'),
        help='HOF KL CSV (optional; v2 KL format)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('analysis/zoom_variants'),
        help='Output directory'
    )
    parser.add_argument(
        '--perf-metric',
        default='auc_normalized_observed',
        help='Performance metric to analyze'
    )
    parser.add_argument(
        '--group-cols',
        default='',
        help='Comma-separated perf CSV columns to stratify (e.g., encoder_config,model_family).',
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
    parser.add_argument(
        '--distance-transform',
        choices=['none', 'log1p'],
        default='none',
        help='Transform normalized distance ratios before analysis.'
    )
    parser.add_argument(
        '--predictors',
        default='',
        help=(
            "HOF predictor override (comma-separated). Presets: motion, density, combined. "
            "Examples: 'motion' or 'hof_density_l2' or 'motion,hof_density_l2'."
        ),
    )
    parser.add_argument(
        '--intervention-only',
        action='store_true',
        help='Run only intervention-style consistency outputs (skip H1/H2/H4 summaries).',
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    perf_df = pd.read_csv(args.perf_csv)
    perf_df['train_dataset'] = perf_df['train_dataset'].astype(str).str.lower()
    perf_df['benchmark'] = perf_df['benchmark'].astype(str).str.lower()
    group_cols = _parse_csv_list(args.group_cols)
    if group_cols:
        missing = [col for col in group_cols if col not in perf_df.columns]
        if missing:
            print(f"Warning: Missing group columns in perf CSV: {missing}")
            group_cols = [col for col in group_cols if col in perf_df.columns]
    
    # Check if metric exists
    if args.perf_metric not in perf_df.columns:
        print(f"Warning: Metric '{args.perf_metric}' not found in CSV.")
        print(f"Available metrics: {[c for c in perf_df.columns if 'auc' in c or 'pck' in c]}")
        raise ValueError(f"Performance metric '{args.perf_metric}' not found")
    
    flow_coverage_df = load_distance_metrics(
        args.flow_coverage_csv,
        prefix="flow",
        distance_transform=args.distance_transform,
    )
    dino_coverage_df = load_distance_metrics(
        args.dino_coverage_csv,
        prefix="dino",
        distance_transform=args.distance_transform,
    )
    flow_mmd_df = load_distance_metrics(args.flow_mmd_csv, prefix="flow")
    dino_mmd_df = load_distance_metrics(args.dino_mmd_csv, prefix="dino")

    hof_coverage_df = None
    hof_kl_df = None
    hof_kl_info: Dict[str, str] = {}
    if args.hof_coverage_csv and args.hof_coverage_csv.exists():
        hof_coverage_df = load_distance_metrics(
            args.hof_coverage_csv,
            prefix="hof",
            distance_transform="none",
        )
    else:
        print(f"Note: HOF coverage CSV not found; skipping HOF analysis ({args.hof_coverage_csv}).")

    if args.hof_kl_csv and args.hof_kl_csv.exists():
        hof_kl_df, hof_kl_info = load_kl_metrics(args.hof_kl_csv, prefix="hof")
    elif args.hof_kl_csv:
        print(f"Note: HOF KL CSV not found; continuing without KL ({args.hof_kl_csv}).")
    
    # Normalize variant names
    variants = [v.lower() for v in args.variants]
    baseline = args.baseline.lower()
    
    all_benchmarks = sorted(perf_df["benchmark"].dropna().unique().tolist())

    if args.intervention_only:
        print("\n=== Intervention Mode: Motion vs Appearance ===")
        print("Using core proxies:")
        print("  flow motion: flow_train_to_eval_eps1px (higher is better)")
        print("  hof motion:  hof_train_to_eval_mean_dist (lower is better)")
        print("  appearance:  dino_train_to_eval_mean_dist (|delta| should stay small)")

        pairs_df, summary_all_df, summary_kitti_df = build_motion_vs_appearance_intervention(
            perf_df=perf_df,
            dino_coverage_df=dino_coverage_df,
            variants=variants,
            baseline=baseline,
            group_cols=group_cols,
            perf_metric=args.perf_metric,
        )
        asym_pairs_df, asym_summary_all_df, asym_summary_kitti_df = build_asymmetry_vs_mmd_intervention(
            perf_df=perf_df,
            flow_mmd_df=flow_mmd_df,
            variants=variants,
            baseline=baseline,
            group_cols=group_cols,
            perf_metric=args.perf_metric,
        )

        pairs_path = args.output_dir / "intervention_motion_vs_appearance_pairs.csv"
        summary_all_path = args.output_dir / "intervention_motion_vs_appearance_summary_all.csv"
        summary_kitti_path = args.output_dir / "intervention_motion_vs_appearance_summary_kitti.csv"
        asym_pairs_path = args.output_dir / "intervention_asymmetry_vs_mmd_pairs.csv"
        asym_summary_all_path = args.output_dir / "intervention_asymmetry_vs_mmd_summary_all.csv"
        asym_summary_kitti_path = args.output_dir / "intervention_asymmetry_vs_mmd_summary_kitti.csv"
        pairs_df.to_csv(pairs_path, index=False)
        summary_all_df.to_csv(summary_all_path, index=False)
        summary_kitti_df.to_csv(summary_kitti_path, index=False)
        asym_pairs_df.to_csv(asym_pairs_path, index=False)
        asym_summary_all_df.to_csv(asym_summary_all_path, index=False)
        asym_summary_kitti_df.to_csv(asym_summary_kitti_path, index=False)

        def _compact_asymmetry_table(df: pd.DataFrame) -> pd.DataFrame:
            wanted = [
                "variant",
                "n",
                "delta_perf_mean",
                "flow_mmd_concordance",
                "mmd_failure_rate",
                "flow_train_to_eval_eps1px_concordance",
                "asymmetry_adv_flow_train_to_eval_eps1px_minus_mmd",
                "mmd_failure_exposed_by_flow_train_to_eval_eps1px_rate",
                "mmd_only_correct_vs_flow_train_to_eval_eps1px_rate",
                "both_correct_with_mmd_vs_flow_train_to_eval_eps1px_rate",
                "both_wrong_with_mmd_vs_flow_train_to_eval_eps1px_rate",
                "flow_eval_to_train_eps1p5px_concordance",
                "asymmetry_adv_flow_eval_to_train_eps1p5px_minus_mmd",
                "mmd_failure_exposed_by_flow_eval_to_train_eps1p5px_rate",
                "mmd_only_correct_vs_flow_eval_to_train_eps1p5px_rate",
                "hof_train_to_eval_mean_dist_concordance",
                "asymmetry_adv_hof_train_to_eval_mean_dist_minus_mmd",
                "hof_eval_to_train_mean_dist_concordance",
                "asymmetry_adv_hof_eval_to_train_mean_dist_minus_mmd",
            ]
            keep = [c for c in wanted if c in df.columns]
            return df[keep].copy()

        asym_summary_kitti_compact = _compact_asymmetry_table(asym_summary_kitti_df)
        asym_summary_all_compact = _compact_asymmetry_table(asym_summary_all_df)

        summary_lines = [
            "=" * 80,
            "INTERVENTION SUMMARY (MOTION VS APPEARANCE)",
            "=" * 80,
            f"Baseline: {baseline}",
            f"Variants: {', '.join(variants)}",
            f"Performance metric: {args.perf_metric}",
            "Motion proxies: flow_train_to_eval_eps1px (higher better), hof_train_to_eval_mean_dist (lower better)",
            "Appearance proxy: dino_train_to_eval_mean_dist (smaller |delta| means appearance is more constant)",
            "",
            "KITTI-only summary (primary):",
            summary_kitti_df.to_string(index=False),
            "",
            "All non-self benchmarks summary:",
            summary_all_df.to_string(index=False),
            "",
            "ASYMMETRY VS MMD (KITTI, primary):",
            "  flow asymmetric predictors: flow_train_to_eval_eps1px / flow_eval_to_train_eps1p5px (higher is better)",
            "  hof asymmetric predictors: hof_train_to_eval_mean_dist / hof_eval_to_train_mean_dist (lower is better)",
            "  mmd delta: lower is better",
            "  asymmetry_adv_*_minus_mmd > 0 means asymmetric predictor beats MMD concordance",
            "  mmd_failure_exposed_by_*_rate: asymmetric correct while MMD is wrong",
            "  mmd_only_correct_vs_*_rate: MMD correct while asymmetric is wrong",
            "  both_correct_with_mmd_vs_*_rate / both_wrong_with_mmd_vs_*_rate: tie decomposition",
            asym_summary_kitti_compact.to_string(index=False),
            "",
            "ASYMMETRY VS MMD (all non-self benchmarks):",
            asym_summary_all_compact.to_string(index=False),
            "",
            f"CSV: {summary_kitti_path}",
            f"CSV: {summary_all_path}",
            f"CSV: {pairs_path}",
            f"CSV: {asym_summary_kitti_path}",
            f"CSV: {asym_summary_all_path}",
            f"CSV: {asym_pairs_path}",
        ]
        intervention_summary_path = args.output_dir / "intervention_summary.txt"
        intervention_summary_path.write_text("\n".join(summary_lines))

        print("\nKITTI summary:")
        print(summary_kitti_df.to_string(index=False))
        print("\nAsymmetry vs MMD (KITTI):")
        print(asym_summary_kitti_compact.to_string(index=False))
        print(f"\n✓ Saved intervention outputs to {args.output_dir}/")
        print("  - intervention_motion_vs_appearance_pairs.csv")
        print("  - intervention_motion_vs_appearance_summary_all.csv")
        print("  - intervention_motion_vs_appearance_summary_kitti.csv")
        print("  - intervention_asymmetry_vs_mmd_pairs.csv")
        print("  - intervention_asymmetry_vs_mmd_summary_all.csv")
        print("  - intervention_asymmetry_vs_mmd_summary_kitti.csv")
        print("  - intervention_summary.txt")
        return
    
    # Run analyses
    complementarity_df = pd.DataFrame()
    flow_pred_detail = pd.DataFrame()
    flow_pred_corr = pd.DataFrame()
    if not args.intervention_only:
        print("\n=== H1: Testing Complementarity (Flow varies, DINO constant) ===")
        complementarity_df = analyze_complementarity(
            perf_df, flow_coverage_df, dino_coverage_df,
            flow_mmd_df, dino_mmd_df,
            variants, baseline, all_benchmarks,
            args.distance_transform,
        )
        complementarity_df.to_csv(args.output_dir / 'h1_complementarity.csv', index=False)
        print(complementarity_df.to_string(index=False))
        
        print("\n=== H2: Flow MMD Predicts Performance ===")
        flow_pred_detail, flow_pred_corr = analyze_flow_prediction(
            perf_df, flow_mmd_df, variants, all_benchmarks, group_cols, args.perf_metric
        )
        flow_pred_detail.to_csv(args.output_dir / 'h2_flow_prediction_detail.csv', index=False)
        flow_pred_corr.to_csv(args.output_dir / 'h2_flow_prediction_correlations.csv', index=False)
        print(flow_pred_corr.to_string(index=False))
    else:
        print("\n=== Intervention Mode ===")
        print("Skipping H1/H2 (complementarity + MMD prediction).")
    
    print("\n=== H3: Eval→Train vs Train→Eval Asymmetry ===")
    print("Metric interpretation:")
    print("  Distance/KL: NEGATIVE delta = improvement (closer to target)")
    print("  Performance: POSITIVE delta = improvement (better performance)")
    print("\nNormalization variants tested:")
    print("  norm_by_eval: distance / radius_eval")
    print("  norm_by_train: distance / radius_train")
    if args.distance_transform == "log1p":
        print("  distance_transform: log1p (applied to normalized distances)")
    asymmetry_df = analyze_asymmetry(
        flow_coverage_df,
        perf_df,
        variants,
        baseline,
        all_benchmarks,
        group_cols,
        args.perf_metric,
        args.distance_transform,
    )
    asymmetry_df.to_csv(args.output_dir / 'h3_asymmetry.csv', index=False)
    
    if not args.intervention_only:
        # Correlation analysis: which normalization best predicts performance?
        print("\n--- Which normalization variant best predicts performance? ---")
        suffix = "_log1p" if args.distance_transform == "log1p" else ""
        distance_metrics = [
            f'delta_eval_to_train_norm_by_eval{suffix}',
            f'delta_eval_to_train_norm_by_train{suffix}',
            f'delta_train_to_eval_norm_by_eval{suffix}',
            f'delta_train_to_eval_norm_by_train{suffix}',
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
        
        corr_columns = [
            'metric', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'n',
            'abs_spearman',
        ]
        corr_df = pd.DataFrame(corr_results, columns=corr_columns)
        if not corr_df.empty:
            corr_df = corr_df.sort_values('abs_spearman', ascending=False)
        print(corr_df[['metric', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p', 'n']].to_string(index=False))
        print("\nInterpretation: Negative correlation = metric correctly predicts performance")
        print("                (lower distance → better performance)")
        corr_df.to_csv(args.output_dir / 'h3_normalization_correlations.csv', index=False)
        
        # Show KITTI, semantic, and remaining benchmarks separately for clarity
        kitti_asymmetry = asymmetry_df[asymmetry_df['benchmark'].isin(KITTI_BENCHMARKS)]
        semantic_asymmetry = asymmetry_df[asymmetry_df['benchmark'].isin(SEMANTIC_FAMILY)]
        other_mask = ~asymmetry_df['benchmark'].isin(KITTI_BENCHMARKS + SEMANTIC_FAMILY)
        other_asymmetry = asymmetry_df[other_mask]
        
        print("\n--- KITTI benchmarks (Performance & Distance Deltas) ---")
        base_compact_cols = [
            'variant', 'benchmark', 'delta_performance',
            f'delta_eval_to_train_norm_by_eval{suffix}', f'delta_eval_to_train_norm_by_train{suffix}',
            f'delta_train_to_eval_norm_by_eval{suffix}', f'delta_train_to_eval_norm_by_train{suffix}',
            'delta_eval_to_train_kl', 'delta_train_to_eval_kl'
        ]
        base_compact_names = [
            'variant', 'benchmark', 'Δperf',
            'Δe2t_by_e', 'Δe2t_by_t', 'Δt2e_by_e', 'Δt2e_by_t',
            'Δkl_e2t', 'Δkl_t2e'
        ]
        compact_cols = group_cols + base_compact_cols
        compact_names = group_cols + base_compact_names
        kitti_compact = kitti_asymmetry[compact_cols].copy()
        # Shorten column names for display
        kitti_compact.columns = compact_names
        print(kitti_compact.to_string(index=False))
        print("\nLegend:")
        print("  Δperf: performance delta (positive = better)")
        print("  Δe2t_by_e: eval→train normalized by eval radius (negative = closer)")
        print("  Δe2t_by_t: eval→train normalized by train radius (negative = closer)")
        print("  Δt2e_by_e: train→eval normalized by eval radius (negative = closer)")
        print("  Δt2e_by_t: train→eval normalized by train radius (negative = closer)")
        print("  Δkl_e2t/t2e: KL divergence deltas (negative = closer)")
        
        print("\n--- Semantic benchmarks (Performance & Distance Deltas) ---")
        semantic_compact = semantic_asymmetry[compact_cols].copy()
        # Shorten column names for display
        semantic_compact.columns = compact_names
        print(semantic_compact.to_string(index=False))
        if not other_asymmetry.empty:
            print("\n--- Other benchmarks (Performance & Distance Deltas) ---")
            other_compact = other_asymmetry[compact_cols].copy()
            other_compact.columns = compact_names
            print(other_compact.to_string(index=False))

        abs_cols = group_cols + [
            'variant', 'benchmark', 'delta_performance',
            'delta_eval_to_train_mean_dist', 'delta_train_to_eval_mean_dist',
        ]
        abs_names = group_cols + [
            'variant', 'benchmark', 'Δperf', 'Δe2t_raw', 'Δt2e_raw',
        ]
        abs_compact = asymmetry_df[abs_cols].copy()
        abs_compact.columns = abs_names
        print("\n--- Absolute distance deltas (mean_nn, unnormalized) ---")
        print(abs_compact.to_string(index=False))

    hof_zoom_df = None
    hof_predictors: List[str] = []
    if hof_coverage_df is not None:
        print("\n=== H5: HOF Motion/Density (Zoom Intervention) ===")
        hof_predictors = _resolve_hof_predictors(args.predictors)
        if any("coverage" in p for p in hof_predictors):
            print("Warning: HOF coverage predictors are not recommended (binary zeros).")

        hof_zoom_df = analyze_hof_zoom(
            hof_coverage_df,
            perf_df,
            variants,
            baseline,
            all_benchmarks,
            group_cols,
            hof_predictors,
            args.perf_metric,
            hof_kl_df=hof_kl_df,
        )
        hof_zoom_df.to_csv(args.output_dir / "h5_hof_zoom.csv", index=False)

        if "delta_performance" not in hof_zoom_df.columns:
            print("Note: baseline rows missing; HOF deltas not computed.")
        elif not args.intervention_only:
            delta_cols = [
                f"delta_{p}" for p in hof_predictors if f"delta_{p}" in hof_zoom_df.columns
            ]
            missing = [p for p in hof_predictors if f"delta_{p}" not in hof_zoom_df.columns]
            if missing:
                print(f"Note: missing HOF predictors in data: {missing}")

            base_cols = group_cols + ["variant", "benchmark", "delta_performance"] + delta_cols
            compact = hof_zoom_df[base_cols].copy()

            rename_map = {
                "delta_performance": "Δperf",
                "delta_hof_eval_to_train_mean_dist": "Δmotion_e2t",
                "delta_hof_train_to_eval_mean_dist": "Δmotion_t2e",
                "delta_hof_density_l2": "Δdensity_l2",
                "delta_hof_density_l1": "Δdensity_l1",
                "delta_hof_density_cosine": "Δdensity_cos",
                "delta_hof_eval_to_train_kl": "Δkl_e2t",
                "delta_hof_train_to_eval_kl": "Δkl_t2e",
            }
            compact = compact.rename(columns=rename_map)

            kitti_hof = compact[compact["benchmark"].isin(KITTI_BENCHMARKS)]
            semantic_hof = compact[compact["benchmark"].isin(SEMANTIC_FAMILY)]
            other_mask = ~compact["benchmark"].isin(KITTI_BENCHMARKS + SEMANTIC_FAMILY)
            other_hof = compact[other_mask]

            print("\n--- HOF KITTI benchmarks (Motion & Density Deltas) ---")
            print(kitti_hof.to_string(index=False))
            if not semantic_hof.empty:
                print("\n--- HOF Semantic benchmarks (Motion & Density Deltas) ---")
                print(semantic_hof.to_string(index=False))
            if not other_hof.empty:
                print("\n--- HOF Other benchmarks (Motion & Density Deltas) ---")
                print(other_hof.to_string(index=False))
        else:
            available_hof_delta_cols = [
                c for c in hof_zoom_df.columns if c.startswith("delta_hof_")
            ]
            print(
                "HOF deltas ready for intervention summary: "
                + (", ".join(available_hof_delta_cols) if available_hof_delta_cols else "none")
            )

    print("\n=== H4: Semantic Improvement (Scale Invariance) ===")
    semantic_df = analyze_semantic_improvement(
        perf_df, variants, baseline, group_cols, args.perf_metric
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
    summary_lines.append(
        f"Group columns: {', '.join(group_cols) if group_cols else 'none'}"
    )
    summary_lines.append(f"Distance transform: {args.distance_transform}")
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
                summary_lines.append(
                    f"    Δt2e_by_e: {row[f'delta_train_to_eval_norm_by_eval{suffix}']:+.2f} (strongest signal)"
                )
                summary_lines.append(
                    f"    Δe2t_by_e: {row[f'delta_eval_to_train_norm_by_eval{suffix}']:+.2f}"
                )
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
        t2e_improved = (kitti_zoom[f'delta_train_to_eval_norm_by_eval{suffix}'] < 0).sum()
        
        summary_lines.append(f"✓ H3: Zoom variants improved performance in {perf_improved}/{total_zoom} KITTI cases")
        summary_lines.append(f"     Train→eval distance closer in {t2e_improved}/{total_zoom} cases (better coverage)")
    
    # H5 summary (HOF motion/density)
    if hof_zoom_df is not None:
        summary_lines.append("H5: HOF MOTION/DENSITY (Zoom Intervention)")
        summary_lines.append("-" * 80)
        summary_lines.append("Negative delta = closer/better for distances (L2/L1); cosine similarity is higher=better.")
        summary_lines.append(f"HOF predictors: {', '.join(hof_predictors) if hof_predictors else 'none'}")
        if "delta_performance" not in hof_zoom_df.columns:
            summary_lines.append("Baseline rows missing; HOF deltas not computed.")
        else:
            hof_delta_cols = [
                f"delta_{p}" for p in hof_predictors if f"delta_{p}" in hof_zoom_df.columns
            ]
            for variant in [v for v in variants if v != baseline]:
                variant_data = hof_zoom_df[hof_zoom_df["variant"] == variant]
                kitti_data = variant_data[variant_data["benchmark"].isin(KITTI_BENCHMARKS)]
                if kitti_data.empty:
                    continue
                summary_lines.append(f"{variant} (KITTI):")
                for _, row in kitti_data.iterrows():
                    line = f"  {row['benchmark']}: Δperf={row['delta_performance']:+.0f}"
                    for metric in hof_delta_cols:
                        if metric in row and np.isfinite(row[metric]):
                            line += f", {metric.replace('delta_', 'Δ')}: {row[metric]:+.2f}"
                    summary_lines.append(line)
                summary_lines.append("")

    summary_lines.append("✓ H4: See CSV for semantic benchmark improvements")
    if hof_zoom_df is not None:
        summary_lines.append("✓ H5: HOF motion/density deltas in h5_hof_zoom.csv")
    summary_lines.append("")
    summary_lines.append("Full details in CSVs: h1_complementarity, h2_flow_prediction_*,")
    if hof_zoom_df is not None:
        summary_lines.append("                      h3_asymmetry, h4_semantic_improvement, h5_hof_zoom")
    else:
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
    detailed_lines.append(
        f"Group columns: {', '.join(group_cols) if group_cols else 'none'}"
    )
    detailed_lines.append(f"Distance transform: {args.distance_transform}")
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
    if args.distance_transform == "log1p":
        detailed_lines.append("  distance_transform: log1p (applied to normalized distances)")
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
    if not other_asymmetry.empty:
        detailed_lines.append("--- Other benchmarks (Performance & Distance Deltas) ---")
        detailed_lines.append(other_compact.to_string(index=False))
        detailed_lines.append("")
    detailed_lines.append("--- Absolute distance deltas (mean_nn, unnormalized) ---")
    detailed_lines.append(abs_compact.to_string(index=False))
    detailed_lines.append("")
    
    if hof_zoom_df is not None:
        detailed_lines.append("=" * 80)
        detailed_lines.append("H5: HOF Motion/Density (Zoom Intervention)")
        detailed_lines.append("=" * 80)
        detailed_lines.append(
            f"HOF predictors: {', '.join(hof_predictors) if hof_predictors else 'none'}"
        )
        if "delta_performance" not in hof_zoom_df.columns:
            detailed_lines.append("Baseline rows missing; HOF deltas not computed.")
        else:
            delta_cols = [
                f"delta_{p}" for p in hof_predictors if f"delta_{p}" in hof_zoom_df.columns
            ]
            base_cols = group_cols + ["variant", "benchmark", "delta_performance"] + delta_cols
            hof_compact = hof_zoom_df[base_cols].copy()
            hof_compact = hof_compact.rename(columns={
                "delta_performance": "Δperf",
                "delta_hof_eval_to_train_mean_dist": "Δmotion_e2t",
                "delta_hof_train_to_eval_mean_dist": "Δmotion_t2e",
                "delta_hof_density_l2": "Δdensity_l2",
                "delta_hof_density_l1": "Δdensity_l1",
                "delta_hof_density_cosine": "Δdensity_cos",
                "delta_hof_eval_to_train_kl": "Δkl_e2t",
                "delta_hof_train_to_eval_kl": "Δkl_t2e",
            })
            detailed_lines.append("--- HOF Motion/Density Deltas (All Benchmarks) ---")
            detailed_lines.append(hof_compact.to_string(index=False))
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
    if hof_zoom_df is not None:
        print(f"  - h5_hof_zoom.csv: HOF motion/density deltas")
    print(f"  - summary.txt: High-level findings")
    print(f"  - detailed_results.txt: All formatted tables")


if __name__ == '__main__':
    main()
