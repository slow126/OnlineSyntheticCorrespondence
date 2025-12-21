#!/usr/bin/env python3
"""
Analyze label-quality vs feature-representation quality across sorted snapshots.

Usage:
  python plot_label_importance.py \
    --snapshots-dir sorted_snapshots \
    --coverage-csv coverage_results.csv \
    --coverage-resnet-csv coverage_resnet_results.csv \
    --output-dir plots_label_importance
"""

import argparse
import os
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not installed. Mixed-effects regression will be skipped.")
    print("Install with: pip install statsmodels")

from plot3d import load_coverage_lookup
from plot_benchmark_metrics import (
    parse_training_dataset_from_summary,
    parse_best_performance_from_summary,
    load_mmd_lookup,
)


def categorize_training_dataset(dataset_name):
    if not dataset_name:
        return 'other', 'Unknown'

    name = dataset_name.lower()

    if name.startswith('spair_synthetic') or ('spair' in name and 'synthetic' in name):
        match = re.search(r'spair_synthetic_(\d+)_(\d+)', name)
        if match:
            return 'mixed', f"Mixed ({match.group(1)}/{match.group(2)})"
        return 'mixed', 'Mixed (50/50)'

    if name == 'spair' or name.startswith('spair_'):
        return 'spair', 'SPair'

    if name == 'synthetic' or name.startswith('synthetic_'):
        return 'synthetic', 'Synthetic'

    return 'other', dataset_name


def extract_mixed_dataset_percentages(snapshot_path):
    path_str = str(snapshot_path).lower()
    match = re.search(r'spair_synthetic_(\d+)_(\d+)', path_str)
    if match:
        return match.group(1), match.group(2)
    return None


def _parse_bool(value):
    value = value.strip().lower()
    if value in ['true', '1', 'yes']:
        return True
    if value in ['false', '0', 'no']:
        return False
    return None


def extract_encoder_config(summary_path, snapshot_path=None):
    pretrained = None
    freeze = None

    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            for line in f:
                line = line.strip()
                lower = line.lower()
                if lower.startswith('pretrained backbone:'):
                    parsed = _parse_bool(line.split(':', 1)[1])
                    if parsed is not None:
                        pretrained = 'pretrained' if parsed else 'not_pretrained'
                elif lower.startswith('freeze backbone:'):
                    parsed = _parse_bool(line.split(':', 1)[1])
                    if parsed is not None:
                        freeze = 'frozen' if parsed else 'unfrozen'

    if snapshot_path and (pretrained is None or freeze is None):
        path_lower = str(snapshot_path).lower()
        if pretrained is None:
            if 'pretrainedtrue' in path_lower:
                pretrained = 'pretrained'
            elif 'pretrainedfalse' in path_lower:
                pretrained = 'not_pretrained'
        if freeze is None:
            if 'freezetrue' in path_lower:
                freeze = 'frozen'
            elif 'freezefalse' in path_lower:
                freeze = 'unfrozen'

    if pretrained and freeze:
        config = f"{pretrained}_{freeze}"
    elif pretrained:
        config = pretrained
    elif freeze:
        config = freeze
    else:
        config = 'unknown'

    return pretrained, freeze, config


def find_snapshot_directories(root_path, max_depth=4):
    root = Path(root_path)
    found = []

    def _walk(path, depth):
        if depth > max_depth:
            return
        if (path / 'training_summary.txt').exists() and (path / 'validation_results.csv').exists():
            found.append(path)
            return
        for child in path.iterdir():
            if child.is_dir():
                _walk(child, depth + 1)

    if root.exists():
        _walk(root, 0)
    return found


def _lookup_coverage(coverage_lookup, train_dataset, benchmark):
    candidates = [
        (f"{train_dataset}_train", f"{benchmark}_test"),
        (f"{train_dataset}_train", f"{benchmark}_val"),
        (train_dataset, benchmark),
    ]
    for key in candidates:
        metrics = coverage_lookup.get(key)
        if metrics is not None:
            return metrics, key
    return None, candidates[-1]


def _lookup_mmd(mmd_lookup, train_dataset, benchmark):
    candidates = [
        (f"{train_dataset}_train", f"{benchmark}_test"),
        (f"{train_dataset}_train", f"{benchmark}_val"),
        (train_dataset, benchmark),
    ]
    for key in candidates:
        value = mmd_lookup.get(key)
        if value is not None:
            return value, key
    return None, candidates[-1]


def collect_data_points(snapshot_dirs, flow_coverage_lookup, resnet_coverage_lookup,
                        flow_mmd_lookup, feature_mmd_lookup, debug=False):
    data_points = []
    missing_flow = defaultdict(int)
    missing_feature = defaultdict(int)

    for snapshot_dir in snapshot_dirs:
        snapshot_dir = Path(snapshot_dir)
        summary_path = snapshot_dir / 'training_summary.txt'
        if not summary_path.exists():
            continue

        base_training_dataset = parse_training_dataset_from_summary(summary_path)
        if not base_training_dataset:
            if debug:
                print(f"Skipping {snapshot_dir}: no training dataset in summary")
            continue

        mixed_percentages = None
        if base_training_dataset == 'spair_synthetic':
            mixed_percentages = extract_mixed_dataset_percentages(snapshot_dir)
            if mixed_percentages:
                base_training_dataset = f"spair_synthetic_{mixed_percentages[0]}_{mixed_percentages[1]}"

        dataset_category, dataset_display = categorize_training_dataset(base_training_dataset)

        best_performance = parse_best_performance_from_summary(summary_path)
        if not best_performance:
            if debug:
                print(f"Skipping {snapshot_dir}: no best performance in summary")
            continue

        pretrained, freeze, model_config = extract_encoder_config(summary_path, snapshot_dir)

        for benchmark, best_pck in best_performance.items():
            benchmark_lower = str(benchmark).lower()

            flow_metrics, flow_key = _lookup_coverage(flow_coverage_lookup, base_training_dataset, benchmark_lower)
            resnet_metrics, resnet_key = _lookup_coverage(resnet_coverage_lookup, base_training_dataset, benchmark_lower)

            flow_mmd, flow_mmd_key = _lookup_mmd(flow_mmd_lookup, base_training_dataset, benchmark_lower)
            feature_mmd, feature_mmd_key = _lookup_mmd(feature_mmd_lookup, base_training_dataset, benchmark_lower)

            if debug:
                if flow_metrics is None:
                    missing_flow[flow_key] += 1
                if resnet_metrics is None:
                    missing_feature[resnet_key] += 1

            data_points.append({
                'flow_recall': flow_metrics['recall'] if flow_metrics else np.nan,
                'flow_precision': flow_metrics['precision'] if flow_metrics else np.nan,
                'resnet_recall': resnet_metrics['recall'] if resnet_metrics else np.nan,
                'resnet_precision': resnet_metrics['precision'] if resnet_metrics else np.nan,
                'flow_mmd': flow_mmd if flow_mmd is not None else np.nan,
                'feature_mmd': feature_mmd if feature_mmd is not None else np.nan,
                'pck': best_pck,
                'training_dataset': base_training_dataset,
                'training_dataset_type': dataset_display,
                'training_dataset_category': dataset_category,
                'benchmark': benchmark,
                'pretrained_status': pretrained,
                'freeze_status': freeze,
                'model_config': model_config,
                'snapshot_path': str(snapshot_dir),
            })

    if debug and (missing_flow or missing_feature):
        print("\nDebug: Missing flow coverage keys (top 10):")
        for key, count in sorted(missing_flow.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")
        print("\nDebug: Missing feature coverage keys (top 10):")
        for key, count in sorted(missing_feature.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {key}: {count} times")

    return data_points


def _zscore(series):
    mean = series.mean()
    std = series.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def add_benchmark_centering(df, value_col, prefix):
    if 'benchmark' not in df.columns or value_col not in df.columns:
        return df
    means = df.groupby('benchmark')[value_col].transform('mean')
    stds = df.groupby('benchmark')[value_col].transform('std')
    stds = stds.where(stds != 0, np.nan)
    df[f'{prefix}_centered'] = df[value_col] - means
    df[f'{prefix}_z'] = (df[value_col] - means) / stds
    return df


def _compute_aic_bic(result, n_obs):
    aic = result.aic if hasattr(result, 'aic') and not np.isnan(result.aic) else np.nan
    bic = result.bic if hasattr(result, 'bic') and not np.isnan(result.bic) else np.nan
    if np.isnan(aic) or np.isnan(bic):
        llf = result.llf if hasattr(result, 'llf') and not np.isnan(result.llf) else np.nan
        if not np.isnan(llf):
            n_params = len(result.fe_params) + 1
            aic = -2 * llf + 2 * n_params
            bic = -2 * llf + np.log(n_obs) * n_params
    return aic, bic


def _fit_mixedlm(df, formula, group_col):
    try:
        model = smf.mixedlm(formula, data=df, groups=df[group_col])
        result = model.fit(method='lbfgs', reml=False)
        aic, bic = _compute_aic_bic(result, len(df))
        return result, aic, bic
    except Exception as exc:
        return exc, np.nan, np.nan


def _safe_ci(result, param):
    try:
        ci = result.conf_int()
        if param in ci.index:
            return float(ci.loc[param, 0]), float(ci.loc[param, 1])
    except Exception:
        pass
    return np.nan, np.nan


def run_label_feature_models(df, label_col, feature_col, group_label, output_path,
                            spec_name, label_name, feature_name, quality_note=None):
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append(f"LABEL VS FEATURE ANALYSIS: {spec_name.upper()} [{group_label}]")
    output_lines.append("=" * 80)

    if not HAS_STATSMODELS:
        output_lines.append("Statsmodels not installed. Skipping mixed-effects regression.")
        output_file = output_path / f"analysis_{spec_name}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Saved analysis results to: {output_file}")
        return None, None, None

    df_sub = df.dropna(subset=[label_col, feature_col, 'pck', 'benchmark']).copy()

    if len(df_sub) < 10:
        output_lines.append(f"Insufficient data ({len(df_sub)} points) for analysis")
        output_file = output_path / f"analysis_{spec_name}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Saved analysis results to: {output_file}")
        return None, None, None

    if df_sub['benchmark'].nunique() < 2:
        output_lines.append("Need at least 2 benchmarks for mixed-effects regression")
        output_file = output_path / f"analysis_{spec_name}.txt"
        with open(output_file, 'w') as f:
            f.write('\n'.join(output_lines))
        print(f"Saved analysis results to: {output_file}")
        return None, None, None

    df_sub['label_z'] = _zscore(df_sub[label_col])
    df_sub['feature_z'] = _zscore(df_sub[feature_col])

    output_lines.append(f"Data: {len(df_sub)} observations across {df_sub['benchmark'].nunique()} benchmarks")
    output_lines.append(f"Label metric: {label_name}")
    output_lines.append(f"Feature metric: {feature_name}")
    if quality_note:
        output_lines.append(quality_note)
    output_lines.append("")

    full_result, aic_full, bic_full = _fit_mixedlm(df_sub, "pck ~ label_z + feature_z", "benchmark")
    _, aic_label, bic_label = _fit_mixedlm(df_sub, "pck ~ label_z", "benchmark")
    _, aic_feature, bic_feature = _fit_mixedlm(df_sub, "pck ~ feature_z", "benchmark")

    summary = {
        'group': group_label,
        'n_obs': len(df_sub),
        'n_benchmarks': df_sub['benchmark'].nunique(),
        'label_coef': np.nan,
        'label_pval': np.nan,
        'label_ci_low': np.nan,
        'label_ci_high': np.nan,
        'feature_coef': np.nan,
        'feature_pval': np.nan,
        'feature_ci_low': np.nan,
        'feature_ci_high': np.nan,
        'aic_full': aic_full,
        'bic_full': bic_full,
        'aic_label': aic_label,
        'bic_label': bic_label,
        'aic_feature': aic_feature,
        'bic_feature': bic_feature,
        'delta_aic_label': aic_label - aic_full if not np.isnan(aic_label) and not np.isnan(aic_full) else np.nan,
        'delta_aic_feature': aic_feature - aic_full if not np.isnan(aic_feature) and not np.isnan(aic_full) else np.nan,
    }

    output_lines.append("Model: PCK ~ label_z + feature_z + (1|benchmark)")
    if isinstance(full_result, Exception):
        output_lines.append(f"Full model failed: {full_result}")
    else:
        if not full_result.converged:
            output_lines.append("Full model did not converge")
        label_coef = full_result.fe_params.get('label_z', np.nan)
        label_pval = full_result.pvalues.get('label_z', np.nan)
        feature_coef = full_result.fe_params.get('feature_z', np.nan)
        feature_pval = full_result.pvalues.get('feature_z', np.nan)
        label_ci_low, label_ci_high = _safe_ci(full_result, 'label_z')
        feature_ci_low, feature_ci_high = _safe_ci(full_result, 'feature_z')

        summary.update({
            'label_coef': label_coef,
            'label_pval': label_pval,
            'label_ci_low': label_ci_low,
            'label_ci_high': label_ci_high,
            'feature_coef': feature_coef,
            'feature_pval': feature_pval,
            'feature_ci_low': feature_ci_low,
            'feature_ci_high': feature_ci_high,
        })

        output_lines.append(f"Label coef (std): {label_coef:.4f} (p={label_pval:.4f})")
        output_lines.append(f"Feature coef (std): {feature_coef:.4f} (p={feature_pval:.4f})")
        output_lines.append(f"AIC: {aic_full:.2f}  BIC: {bic_full:.2f}")

    output_lines.append("")
    output_lines.append("Drop-one comparisons (same data):")
    output_lines.append(f"  AIC without label: {aic_label:.2f} (delta={summary['delta_aic_label']:.2f})")
    output_lines.append(f"  AIC without feature: {aic_feature:.2f} (delta={summary['delta_aic_feature']:.2f})")

    output_file = output_path / f"analysis_{spec_name}.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"Saved analysis results to: {output_file}")

    return summary, full_result, df_sub


def plot_scatter_by_category(df, x_col, y_col, title, x_label, y_label, output_file):
    if df.empty:
        return

    category_colors = {
        'synthetic': '#1f77b4',
        'spair': '#2ca02c',
        'mixed': '#ff7f0e',
        'other': '#7f7f7f',
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for category in ['synthetic', 'spair', 'mixed', 'other']:
        subset = df[df['training_dataset_category'] == category]
        if subset.empty:
            continue
        ax.scatter(
            subset[x_col],
            subset[y_col],
            label=category,
            color=category_colors.get(category, '#7f7f7f'),
            edgecolors='black',
            linewidth=0.6,
            alpha=0.7,
            s=60,
        )

    if len(df) >= 2:
        try:
            z = np.polyfit(df[x_col], df[y_col], 1)
            x_line = np.linspace(df[x_col].min(), df[x_col].max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color='black', linestyle='--', linewidth=1.5)
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_flow_feature_interaction(df, output_file, title, value_col, value_label):
    df_plot = df.dropna(subset=['flow_recall', 'resnet_recall', value_col])
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    values = df_plot[value_col].to_numpy()
    max_abs = np.nanmax(np.abs(values)) if len(values) > 0 else np.nan
    if not np.isnan(max_abs) and max_abs > 0:
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = None, None

    scatter = ax.scatter(
        df_plot['flow_recall'],
        df_plot['resnet_recall'],
        c=values,
        cmap='coolwarm',
        vmin=vmin,
        vmax=vmax,
        s=70,
        alpha=0.85,
        edgecolors='black',
        linewidth=0.5,
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(value_label)

    ax.set_xlabel('Flow recall (label coverage)')
    ax.set_ylabel('ResNet recall (feature coverage)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_interaction_heatmap(df, output_file, title, x_col, y_col, value_col, bins=4):
    df_plot = df.dropna(subset=[x_col, y_col, value_col]).copy()
    if df_plot.empty:
        return

    df_plot['x_bin'] = pd.qcut(df_plot[x_col], q=bins, duplicates='drop')
    df_plot['y_bin'] = pd.qcut(df_plot[y_col], q=bins, duplicates='drop')

    pivot = df_plot.pivot_table(values=value_col, index='y_bin', columns='x_bin', aggfunc='mean')
    counts = df_plot.pivot_table(values=value_col, index='y_bin', columns='x_bin', aggfunc='size')

    if pivot.empty:
        return

    values = pivot.to_numpy()
    max_abs = np.nanmax(np.abs(values)) if values.size > 0 else np.nan
    if not np.isnan(max_abs) and max_abs > 0:
        vmin, vmax = -max_abs, max_abs
    else:
        vmin, vmax = None, None

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(values, cmap='coolwarm', aspect='auto', origin='lower', vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=30, ha='right')
    ax.set_yticklabels([str(i) for i in pivot.index])

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            mean_val = values[i, j]
            count_val = counts.iloc[i, j] if not counts.empty else 0
            if np.isnan(mean_val):
                text = "n=0"
            else:
                text = f"{mean_val:.2f}\\n(n={int(count_val)})"
            ax.text(j, i, text, ha='center', va='center', fontsize=8, color='black')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{value_col} (mean)')

    ax.set_xlabel(f'{x_col} (binned)')
    ax.set_ylabel(f'{y_col} (binned)')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_partial_residual(df, x_col, y_col, output_file, title, x_label, y_label):
    df_plot = df.dropna(subset=[x_col, y_col])
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        df_plot[x_col],
        df_plot[y_col],
        color='#1f77b4',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.7,
        s=50,
    )

    if len(df_plot) >= 2:
        try:
            z = np.polyfit(df_plot[x_col], df_plot[y_col], 1)
            x_line = np.linspace(df_plot[x_col].min(), df_plot[x_col].max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color='black', linestyle='--', linewidth=1.5)
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_binned_trend(df, x_col, y_col, output_file, title, x_label, y_label, bins=5):
    df_plot = df.dropna(subset=[x_col, y_col])
    if df_plot.empty:
        return

    try:
        df_plot['x_bin'] = pd.qcut(df_plot[x_col], q=bins, duplicates='drop')
    except ValueError:
        return

    grouped = df_plot.groupby('x_bin')
    if grouped.ngroups == 0:
        return

    bin_centers = grouped[x_col].mean()
    bin_means = grouped[y_col].mean()
    bin_counts = grouped[y_col].count()
    bin_stds = grouped[y_col].std()
    bin_sems = bin_stds / np.sqrt(bin_counts)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        bin_centers,
        bin_means,
        yerr=bin_sems,
        fmt='o-',
        color='#1f77b4',
        ecolor='black',
        capsize=4,
        linewidth=1.5,
        markersize=5,
    )

    for x_val, y_val, n_val in zip(bin_centers, bin_means, bin_counts):
        ax.text(x_val, y_val, f"n={int(n_val)}", fontsize=8, ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_standardized_scatter(df, x_col, y_col, output_file, title, x_label, y_label):
    df_plot = df.dropna(subset=[x_col, y_col])
    if df_plot.empty:
        return

    x_z = _zscore(df_plot[x_col])
    y_z = _zscore(df_plot[y_col])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        x_z,
        y_z,
        color='#ff7f0e',
        edgecolors='black',
        linewidth=0.5,
        alpha=0.7,
        s=50,
    )

    if len(df_plot) >= 2:
        try:
            z = np.polyfit(x_z, y_z, 1)
            x_line = np.linspace(x_z.min(), x_z.max(), 100)
            ax.plot(x_line, np.poly1d(z)(x_line), color='black', linestyle='--', linewidth=1.5)
        except Exception:
            pass

    ax.set_title(title)
    ax.set_xlabel(f"{x_label} (z-score)")
    ax.set_ylabel(f"{y_label} (z-score)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_centered_surface(df_sub, label_col, feature_col, output_file, title, label_name, feature_name):
    if not HAS_STATSMODELS:
        return
    if df_sub is None or df_sub.empty:
        return
    if 'pck_centered' not in df_sub.columns:
        return

    df_plot = df_sub.dropna(subset=[label_col, feature_col, 'pck_centered']).copy()
    if df_plot.empty:
        return

    df_plot['label_z'] = _zscore(df_plot[label_col])
    df_plot['feature_z'] = _zscore(df_plot[feature_col])

    try:
        result = smf.ols("pck_centered ~ label_z + feature_z", data=df_plot).fit()
    except Exception:
        return

    label_vals = df_plot[label_col].dropna()
    feature_vals = df_plot[feature_col].dropna()
    if label_vals.empty or feature_vals.empty:
        return

    label_min = label_vals.quantile(0.05)
    label_max = label_vals.quantile(0.95)
    feature_min = feature_vals.quantile(0.05)
    feature_max = feature_vals.quantile(0.95)

    label_grid = np.linspace(label_min, label_max, 40)
    feature_grid = np.linspace(feature_min, feature_max, 40)
    label_mesh, feature_mesh = np.meshgrid(label_grid, feature_grid)

    label_mean = label_vals.mean()
    label_std = label_vals.std(ddof=0)
    feature_mean = feature_vals.mean()
    feature_std = feature_vals.std(ddof=0)

    if label_std == 0 or feature_std == 0:
        return

    label_z = (label_mesh - label_mean) / label_std
    feature_z = (feature_mesh - feature_mean) / feature_std

    intercept = result.params.get('Intercept', 0.0)
    coef_label = result.params.get('label_z', 0.0)
    coef_feature = result.params.get('feature_z', 0.0)
    pred = intercept + coef_label * label_z + coef_feature * feature_z

    fig, ax = plt.subplots(figsize=(10, 7))
    surface = ax.contourf(label_mesh, feature_mesh, pred, levels=20, cmap='coolwarm')
    cbar = plt.colorbar(surface, ax=ax)
    cbar.set_label('Predicted PCK (benchmark-centered)')

    ax.set_xlabel(label_name)
    ax.set_ylabel(feature_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_mixedlm_surface(df_sub, result, label_col, feature_col,
                         output_file, title, label_name, feature_name):
    if df_sub is None or df_sub.empty:
        return
    if isinstance(result, Exception) or result is None:
        return
    if hasattr(result, 'converged') and not result.converged:
        return

    if label_col not in df_sub.columns or feature_col not in df_sub.columns:
        return

    label_vals = df_sub[label_col].dropna()
    feature_vals = df_sub[feature_col].dropna()
    if label_vals.empty or feature_vals.empty:
        return

    label_min = label_vals.quantile(0.05)
    label_max = label_vals.quantile(0.95)
    feature_min = feature_vals.quantile(0.05)
    feature_max = feature_vals.quantile(0.95)

    label_grid = np.linspace(label_min, label_max, 40)
    feature_grid = np.linspace(feature_min, feature_max, 40)
    label_mesh, feature_mesh = np.meshgrid(label_grid, feature_grid)

    label_mean = label_vals.mean()
    label_std = label_vals.std(ddof=0)
    feature_mean = feature_vals.mean()
    feature_std = feature_vals.std(ddof=0)

    if label_std == 0 or feature_std == 0:
        return

    label_z = (label_mesh - label_mean) / label_std
    feature_z = (feature_mesh - feature_mean) / feature_std

    intercept = result.fe_params.get('Intercept', 0.0)
    coef_label = result.fe_params.get('label_z', 0.0)
    coef_feature = result.fe_params.get('feature_z', 0.0)

    pred = intercept + coef_label * label_z + coef_feature * feature_z

    fig, ax = plt.subplots(figsize=(10, 7))
    surface = ax.contourf(label_mesh, feature_mesh, pred, levels=20, cmap='coolwarm')
    cbar = plt.colorbar(surface, ax=ax)
    cbar.set_label('Predicted PCK (fixed effects)')

    ax.set_xlabel(label_name)
    ax.set_ylabel(feature_name)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_pck_by_category(df, output_file, title):
    if df.empty:
        return

    categories = ['synthetic', 'spair', 'mixed', 'other']
    means = []
    stds = []
    labels = []

    for category in categories:
        subset = df[df['training_dataset_category'] == category]
        if subset.empty:
            continue
        means.append(subset['pck'].mean())
        stds.append(subset['pck'].std())
        labels.append(category)

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.8, edgecolor='black')

    for i, label in enumerate(labels):
        if label == 'synthetic':
            bars[i].set_color('#1f77b4')
        elif label == 'spair':
            bars[i].set_color('#2ca02c')
        elif label == 'mixed':
            bars[i].set_color('#ff7f0e')
        else:
            bars[i].set_color('#7f7f7f')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('PCK (%)')
    ax.set_title(title)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_coefficients(summary_rows, output_path, spec_name, group_by, label_name, feature_name):
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    df = df.dropna(subset=['label_coef', 'feature_coef'])
    if df.empty:
        return

    groups = df['group'].tolist()
    x = np.arange(len(groups))
    width = 0.35

    label_err_low = (df['label_coef'] - df['label_ci_low']).clip(lower=0)
    label_err_high = (df['label_ci_high'] - df['label_coef']).clip(lower=0)
    feature_err_low = (df['feature_coef'] - df['feature_ci_low']).clip(lower=0)
    feature_err_high = (df['feature_ci_high'] - df['feature_coef']).clip(lower=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        df['label_coef'],
        width,
        yerr=np.vstack([label_err_low, label_err_high]),
        label=label_name,
        color='#1f77b4',
        alpha=0.85,
        edgecolor='black',
    )
    ax.bar(
        x + width / 2,
        df['feature_coef'],
        width,
        yerr=np.vstack([feature_err_low, feature_err_high]),
        label=feature_name,
        color='#ff7f0e',
        alpha=0.85,
        edgecolor='black',
    )

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylabel('Standardized coefficient (positive -> higher PCK)')
    ax.set_title(f"Label vs Feature effect ({spec_name}) by {group_by}")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    output_file = output_path / f"coefficients_{spec_name}_by_{group_by}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def plot_delta_aic(summary_rows, output_path, spec_name, group_by, label_name, feature_name):
    if not summary_rows:
        return

    df = pd.DataFrame(summary_rows)
    df = df.dropna(subset=['delta_aic_label', 'delta_aic_feature'])
    if df.empty:
        return

    groups = df['group'].tolist()
    x = np.arange(len(groups))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x - width / 2,
        df['delta_aic_label'],
        width,
        label=f"Drop {label_name}",
        color='#1f77b4',
        alpha=0.85,
        edgecolor='black',
    )
    ax.bar(
        x + width / 2,
        df['delta_aic_feature'],
        width,
        label=f"Drop {feature_name}",
        color='#ff7f0e',
        alpha=0.85,
        edgecolor='black',
    )

    ax.axhline(0, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylabel('Delta AIC (higher means more important)')
    ax.set_title(f"Drop-one AIC comparison ({spec_name}) by {group_by}")
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()

    output_file = output_path / f"delta_aic_{spec_name}_by_{group_by}.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def _group_label(point, group_by):
    if group_by == 'pretrained':
        return point.get('pretrained_status') or 'unknown'
    if group_by == 'freeze':
        return point.get('freeze_status') or 'unknown'
    if group_by == 'config':
        return point.get('model_config') or 'unknown'
    return 'all'


def _sorted_groups(group_labels, group_by):
    if group_by == 'config':
        order = [
            'pretrained_frozen',
            'pretrained_unfrozen',
            'not_pretrained_frozen',
            'not_pretrained_unfrozen',
        ]
        return sorted(group_labels, key=lambda x: order.index(x) if x in order else 999)
    if group_by == 'pretrained':
        order = ['pretrained', 'not_pretrained', 'unknown']
        return sorted(group_labels, key=lambda x: order.index(x) if x in order else 999)
    if group_by == 'freeze':
        order = ['frozen', 'unfrozen', 'unknown']
        return sorted(group_labels, key=lambda x: order.index(x) if x in order else 999)
    return sorted(group_labels)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze label-quality vs feature-quality using sorted snapshots.'
    )
    parser.add_argument(
        '--snapshots-dir', type=str, default='sorted_snapshots',
        help='Root directory containing sorted snapshots (default: sorted_snapshots)'
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
        '--flow-mmd-csv', type=str, default='flow_mmd_results.csv',
        help='Path to flow MMD CSV (default: flow_mmd_results.csv)'
    )
    parser.add_argument(
        '--feature-mmd-csv', type=str, default='feature_mmd_results.csv',
        help='Path to feature MMD CSV (default: feature_mmd_results.csv)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='plots_label_importance',
        help='Output directory for plots and summaries (default: plots_label_importance)'
    )
    parser.add_argument(
        '--group-by', type=str, default='config',
        choices=['config', 'pretrained', 'freeze', 'none'],
        help='How to group results (default: config)'
    )
    parser.add_argument(
        '--max-depth', type=int, default=4,
        help='Maximum directory depth to search for snapshots (default: 4)'
    )
    parser.add_argument(
        '--focus-mixed-family', action='store_true',
        help='Restrict analysis to synthetic/spair/mixed datasets only'
    )
    parser.add_argument(
        '--skip-mmd', action='store_true',
        help='Skip MMD-based analysis'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Print debug info while collecting data'
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snapshot_dirs = find_snapshot_directories(args.snapshots_dir, args.max_depth)
    if not snapshot_dirs:
        print(f"Error: No snapshot directories found in {args.snapshots_dir}")
        return

    print(f"Found {len(snapshot_dirs)} snapshot directories")

    print("Loading coverage data...")
    flow_coverage_lookup = load_coverage_lookup(args.coverage_csv)
    resnet_coverage_lookup = load_coverage_lookup(args.coverage_resnet_csv)

    flow_mmd_lookup = {}
    feature_mmd_lookup = {}
    if not args.skip_mmd:
        flow_mmd_lookup = load_mmd_lookup(args.flow_mmd_csv)
        feature_mmd_lookup = load_mmd_lookup(args.feature_mmd_csv)

    print("Collecting data points...")
    data_points = collect_data_points(
        snapshot_dirs,
        flow_coverage_lookup,
        resnet_coverage_lookup,
        flow_mmd_lookup,
        feature_mmd_lookup,
        debug=args.debug,
    )

    if not data_points:
        print("Error: No data points collected. Check CSV lookups and snapshot summaries.")
        return

    df = pd.DataFrame(data_points)
    df = add_benchmark_centering(df, 'pck', 'pck')

    df['flow_mmd_quality'] = -df['flow_mmd']
    df['feature_mmd_quality'] = -df['feature_mmd']

    analysis_specs = [
        {
            'name': 'coverage',
            'label_col': 'flow_recall',
            'feature_col': 'resnet_recall',
            'label_scatter_col': 'flow_recall',
            'feature_scatter_col': 'resnet_recall',
            'label_name': 'Flow recall (label coverage)',
            'feature_name': 'ResNet recall (feature coverage)',
            'quality_note': None,
        }
    ]

    if not args.skip_mmd and flow_mmd_lookup and feature_mmd_lookup:
        analysis_specs.append({
            'name': 'mmd',
            'label_col': 'flow_mmd_quality',
            'feature_col': 'feature_mmd_quality',
            'label_scatter_col': 'flow_mmd',
            'feature_scatter_col': 'feature_mmd',
            'label_name': 'Flow MMD (quality: -MMD)',
            'feature_name': 'Feature MMD (quality: -MMD)',
            'quality_note': 'Lower MMD is better; analysis uses -MMD as quality.',
        })

    if args.focus_mixed_family:
        scopes = [('mixed_family', df[df['training_dataset_category'].isin(['synthetic', 'spair', 'mixed'])])]
    else:
        scopes = [
            ('all', df),
            ('mixed_family', df[df['training_dataset_category'].isin(['synthetic', 'spair', 'mixed'])])
        ]

    for scope_name, scope_df in scopes:
        if scope_df.empty:
            print(f"Skipping scope with no data: {scope_name}")
            continue

        scope_output = output_path / scope_name
        scope_output.mkdir(parents=True, exist_ok=True)

        group_by = args.group_by
        if group_by == 'none':
            group_labels = ['all']
        else:
            group_labels = sorted(scope_df.apply(lambda row: _group_label(row, group_by), axis=1).unique())
            group_labels = _sorted_groups(group_labels, group_by)

        summary_by_spec = {spec['name']: [] for spec in analysis_specs}

        for group_label in group_labels:
            if group_by == 'none':
                group_df = scope_df.copy()
            else:
                group_df = scope_df[scope_df.apply(lambda row: _group_label(row, group_by) == group_label, axis=1)].copy()

            if group_df.empty:
                print(f"Skipping empty group: {group_label} ({scope_name})")
                continue

            group_slug = re.sub(r'[^a-z0-9_-]+', '_', str(group_label).lower()).strip('_')
            group_dir = scope_output / (group_slug or 'all')
            group_dir.mkdir(parents=True, exist_ok=True)

            plot_pck_by_category(
                group_df,
                group_dir / 'pck_by_category.png',
                title=f"PCK by dataset category ({group_label}, {scope_name})",
            )

            interaction_value_col = 'pck_centered' if 'pck_centered' in group_df.columns else 'pck'
            interaction_value_label = 'PCK (benchmark-centered)' if interaction_value_col == 'pck_centered' else 'PCK (%)'

            plot_flow_feature_interaction(
                group_df,
                group_dir / 'flow_feature_interaction.png',
                title=f"Flow vs Feature interaction ({group_label}, {scope_name})",
                value_col=interaction_value_col,
                value_label=interaction_value_label,
            )
            plot_interaction_heatmap(
                group_df,
                group_dir / 'flow_feature_interaction_heatmap.png',
                title=f"Flow vs Feature interaction heatmap ({group_label}, {scope_name})",
                x_col='flow_recall',
                y_col='resnet_recall',
                value_col=interaction_value_col,
            )

            y_col = 'pck_centered' if 'pck_centered' in group_df.columns else 'pck'
            y_label = 'PCK (benchmark-centered)' if y_col == 'pck_centered' else 'PCK (%)'

            for spec in analysis_specs:
                label_col = spec['label_col']
                feature_col = spec['feature_col']

                if label_col not in group_df.columns or feature_col not in group_df.columns:
                    continue

                summary, full_result, df_sub = run_label_feature_models(
                    group_df,
                    label_col,
                    feature_col,
                    group_label,
                    group_dir,
                    spec['name'],
                    spec['label_name'],
                    spec['feature_name'],
                    quality_note=spec['quality_note'],
                )

                if summary:
                    summary_by_spec[spec['name']].append(summary)

                plot_mixedlm_surface(
                    df_sub,
                    full_result,
                    label_col,
                    feature_col,
                    group_dir / f"mixedlm_surface_{spec['name']}.png",
                    title=f"Mixed-effects fixed surface ({spec['name']}, {group_label}, {scope_name})",
                    label_name=spec['label_name'],
                    feature_name=spec['feature_name'],
                )

                plot_centered_surface(
                    df_sub,
                    label_col,
                    feature_col,
                    group_dir / f"centered_surface_{spec['name']}.png",
                    title=f"Centered surface ({spec['name']}, {group_label}, {scope_name})",
                    label_name=spec['label_name'],
                    feature_name=spec['feature_name'],
                )

                plot_partial_residual(
                    group_df,
                    label_col,
                    y_col,
                    group_dir / f"partial_label_{spec['name']}.png",
                    title=f"{y_label} vs {spec['label_name']} (partial, {group_label}, {scope_name})",
                    x_label=spec['label_name'],
                    y_label=y_label,
                )
                plot_partial_residual(
                    group_df,
                    feature_col,
                    y_col,
                    group_dir / f"partial_feature_{spec['name']}.png",
                    title=f"{y_label} vs {spec['feature_name']} (partial, {group_label}, {scope_name})",
                    x_label=spec['feature_name'],
                    y_label=y_label,
                )
                plot_binned_trend(
                    group_df,
                    label_col,
                    y_col,
                    group_dir / f"binned_label_{spec['name']}.png",
                    title=f"{y_label} vs {spec['label_name']} (binned, {group_label}, {scope_name})",
                    x_label=spec['label_name'],
                    y_label=y_label,
                )
                plot_binned_trend(
                    group_df,
                    feature_col,
                    y_col,
                    group_dir / f"binned_feature_{spec['name']}.png",
                    title=f"{y_label} vs {spec['feature_name']} (binned, {group_label}, {scope_name})",
                    x_label=spec['feature_name'],
                    y_label=y_label,
                )
                plot_standardized_scatter(
                    group_df,
                    label_col,
                    y_col,
                    group_dir / f"scatter_label_{spec['name']}_z.png",
                    title=f"{y_label} vs {spec['label_name']} (z-scored, {group_label}, {scope_name})",
                    x_label=spec['label_name'],
                    y_label=y_label,
                )
                plot_standardized_scatter(
                    group_df,
                    feature_col,
                    y_col,
                    group_dir / f"scatter_feature_{spec['name']}_z.png",
                    title=f"{y_label} vs {spec['feature_name']} (z-scored, {group_label}, {scope_name})",
                    x_label=spec['feature_name'],
                    y_label=y_label,
                )

                scatter_df = group_df.dropna(subset=[spec['label_scatter_col'], y_col])
                plot_scatter_by_category(
                    scatter_df,
                    spec['label_scatter_col'],
                    y_col,
                    title=f"{y_label} vs {spec['label_name']} ({group_label}, {scope_name})",
                    x_label=spec['label_name'],
                    y_label=y_label,
                    output_file=group_dir / f"scatter_label_{spec['name']}.png",
                )

                scatter_df = group_df.dropna(subset=[spec['feature_scatter_col'], y_col])
                plot_scatter_by_category(
                    scatter_df,
                    spec['feature_scatter_col'],
                    y_col,
                    title=f"{y_label} vs {spec['feature_name']} ({group_label}, {scope_name})",
                    x_label=spec['feature_name'],
                    y_label=y_label,
                    output_file=group_dir / f"scatter_feature_{spec['name']}.png",
                )

        for spec in analysis_specs:
            summary_rows = summary_by_spec.get(spec['name'], [])
            if not summary_rows:
                continue

            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(scope_output / f"summary_{spec['name']}_by_{group_by}.csv", index=False)

            plot_coefficients(
                summary_rows,
                scope_output,
                spec['name'],
                group_by,
                spec['label_name'],
                spec['feature_name'],
            )
            plot_delta_aic(
                summary_rows,
                scope_output,
                spec['name'],
                group_by,
                spec['label_name'],
                spec['feature_name'],
            )

    print(f"Done. Outputs saved to: {output_path}")


if __name__ == '__main__':
    main()
