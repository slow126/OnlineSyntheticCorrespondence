#!/usr/bin/env python3
"""
Analysis script for sparse data + synthetic/2dwarp regularization experiments.

This script compares three training conditions:
1. SPAIR only (sparse labels)
2. SPAIR + synthetic data (various mixing ratios)
3. SPAIR + 2D ImageNet warps (various mixing ratios)

The goal is to test the hypothesis that synthetic data provides
dense geometric regularization that improves learning.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def load_all_snapshots(snapshot_dir):
    """Load validation results from all snapshots in a directory"""
    results = []
    
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        raise ValueError(f"Snapshot directory not found: {snapshot_dir}")
    
    for snapshot_name in os.listdir(snapshot_dir):
        snapshot_path = snapshot_dir / snapshot_name
        if not snapshot_path.is_dir():
            continue
            
        csv_path = snapshot_path / 'validation_results.csv'
        if not csv_path.exists():
            continue
        
        # Load CSV
        df = pd.read_csv(csv_path)
        df['snapshot_name'] = snapshot_name
        df['snapshot_path'] = str(snapshot_path)
        
        # Parse training configuration from snapshot name
        config = parse_snapshot_name(snapshot_name)
        for key, value in config.items():
            df[key] = value
        
        results.append(df)
    
    if not results:
        raise ValueError(f"No validation results found in {snapshot_dir}")
    
    return pd.concat(results, ignore_index=True)


def parse_snapshot_name(snapshot_name):
    """Parse training configuration from snapshot directory name"""
    config = {
        'condition': 'unknown',
        'mix_ratio': None,
        'model_type': 'unknown',
        'pretrained': None,
        'freeze': None,
    }
    
    snapshot_lower = snapshot_name.lower()
    
    # Determine condition
    if 'spair_only' in snapshot_lower or (snapshot_lower.startswith('spair_') and 'synthetic' not in snapshot_lower and '2d_warp' not in snapshot_lower and 'imagenet' not in snapshot_lower):
        config['condition'] = 'spair_only'
    elif 'synthetic' in snapshot_lower:
        config['condition'] = 'spair_synthetic'
        # Extract mixing ratio
        if '30_70' in snapshot_lower:
            config['mix_ratio'] = '30_70'
        elif '50_50' in snapshot_lower:
            config['mix_ratio'] = '50_50'
        elif '70_30' in snapshot_lower:
            config['mix_ratio'] = '70_30'
    elif '2d_warp' in snapshot_lower or 'imagenet2dwarp' in snapshot_lower:
        config['condition'] = 'spair_2dwarp'
        # Extract mixing ratio
        if '30_70' in snapshot_lower:
            config['mix_ratio'] = '30_70'
        elif '50_50' in snapshot_lower:
            config['mix_ratio'] = '50_50'
        elif '70_30' in snapshot_lower:
            config['mix_ratio'] = '70_30'
        elif 'imagenet2dwarp_cats' in snapshot_lower or 'imagenet2dwarp_raft' in snapshot_lower:
            config['mix_ratio'] = '100_0'  # Pure 2dwarp
    
    # Determine model type
    if 'raft' in snapshot_lower:
        config['model_type'] = 'raft'
    else:
        config['model_type'] = 'cats'
    
    # Determine pretrained status
    if 'pretrainedtrue' in snapshot_lower:
        config['pretrained'] = True
    elif 'pretrainedfalse' in snapshot_lower:
        config['pretrained'] = False
    
    # Determine freeze status
    if 'freezetrue' in snapshot_lower:
        config['freeze'] = True
    elif 'freezefalse' in snapshot_lower:
        config['freeze'] = False
    
    return config


def get_final_performance(df, benchmark='spair'):
    """Get final epoch performance for each snapshot"""
    # Group by snapshot and get last epoch
    final_perf = []
    
    for snapshot_name in df['snapshot_name'].unique():
        snapshot_df = df[df['snapshot_name'] == snapshot_name]
        bench_df = snapshot_df[snapshot_df['benchmark'] == benchmark]
        
        if len(bench_df) == 0:
            continue
        
        # Get last epoch
        if 'training_steps' in bench_df.columns:
            last_epoch = bench_df.loc[bench_df['training_steps'].idxmax()]
        else:
            last_epoch = bench_df.iloc[-1]
        
        final_perf.append(last_epoch)
    
    return pd.DataFrame(final_perf)


def plot_learning_curves(df, output_dir, benchmark='spair'):
    """Plot learning curves comparing all conditions"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Learning Curves Comparison - {benchmark.upper()}', fontsize=16, fontweight='bold')
    
    # Filter to specific benchmark
    bench_df = df[df['benchmark'] == benchmark].copy()
    
    # Define condition colors and labels
    condition_colors = {
        'spair_only': '#d62728',  # red
        'spair_synthetic': '#2ca02c',  # green
        'spair_2dwarp': '#ff7f0e',  # orange
    }
    
    condition_labels = {
        'spair_only': 'SPAIR Only (sparse)',
        'spair_synthetic': 'SPAIR + Synthetic',
        'spair_2dwarp': 'SPAIR + 2D Warp',
    }
    
    # Plot 1: All conditions, pretrained=False, freeze=False (from scratch)
    ax = axes[0, 0]
    for condition in ['spair_only', 'spair_synthetic', 'spair_2dwarp']:
        cond_df = bench_df[(bench_df['condition'] == condition) & 
                           (bench_df['pretrained'] == False) & 
                           (bench_df['freeze'] == False)]
        
        if len(cond_df) == 0:
            continue
        
        # Group by mix_ratio if applicable
        if condition == 'spair_only':
            # Plot single line
            grouped = cond_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
            ax.plot(grouped['training_steps'], grouped['mean'], 
                   color=condition_colors[condition], label=condition_labels[condition],
                   linewidth=2, marker='o', markersize=4)
            if grouped['count'].max() > 1:
                ax.fill_between(grouped['training_steps'],
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=condition_colors[condition], alpha=0.2)
        else:
            # Plot separate lines for each mix ratio
            for mix_ratio in ['30_70', '50_50', '70_30']:
                mix_df = cond_df[cond_df['mix_ratio'] == mix_ratio]
                if len(mix_df) == 0:
                    continue
                
                grouped = mix_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
                label = f"{condition_labels[condition]} ({mix_ratio})"
                ax.plot(grouped['training_steps'], grouped['mean'],
                       color=condition_colors[condition], label=label,
                       linewidth=2, marker='o', markersize=3,
                       linestyle='-' if mix_ratio == '50_50' else '--' if mix_ratio == '30_70' else ':')
                if grouped['count'].max() > 1:
                    ax.fill_between(grouped['training_steps'],
                                   grouped['mean'] - grouped['std'],
                                   grouped['mean'] + grouped['std'],
                                   color=condition_colors[condition], alpha=0.1)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('PCK (%)', fontsize=12)
    ax.set_title('From Scratch (pretrained=False, freeze=False)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Pretrained encoders (pretrained=True, freeze=False)
    ax = axes[0, 1]
    for condition in ['spair_only', 'spair_synthetic', 'spair_2dwarp']:
        cond_df = bench_df[(bench_df['condition'] == condition) & 
                           (bench_df['pretrained'] == True) & 
                           (bench_df['freeze'] == False)]
        
        if len(cond_df) == 0:
            continue
        
        if condition == 'spair_only':
            grouped = cond_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
            ax.plot(grouped['training_steps'], grouped['mean'],
                   color=condition_colors[condition], label=condition_labels[condition],
                   linewidth=2, marker='o', markersize=4)
            if grouped['count'].max() > 1:
                ax.fill_between(grouped['training_steps'],
                               grouped['mean'] - grouped['std'],
                               grouped['mean'] + grouped['std'],
                               color=condition_colors[condition], alpha=0.2)
        else:
            for mix_ratio in ['30_70', '50_50', '70_30']:
                mix_df = cond_df[cond_df['mix_ratio'] == mix_ratio]
                if len(mix_df) == 0:
                    continue
                
                grouped = mix_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
                label = f"{condition_labels[condition]} ({mix_ratio})"
                ax.plot(grouped['training_steps'], grouped['mean'],
                       color=condition_colors[condition], label=label,
                       linewidth=2, marker='o', markersize=3,
                       linestyle='-' if mix_ratio == '50_50' else '--' if mix_ratio == '30_70' else ':')
                if grouped['count'].max() > 1:
                    ax.fill_between(grouped['training_steps'],
                                   grouped['mean'] - grouped['std'],
                                   grouped['mean'] + grouped['std'],
                                   color=condition_colors[condition], alpha=0.1)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('PCK (%)', fontsize=12)
    ax.set_title('Pretrained Encoders (pretrained=True, freeze=False)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Comparison of best mixing ratios (from scratch)
    ax = axes[1, 0]
    # For each condition, find best mixing ratio at final step
    best_configs = {}
    for condition in ['spair_only', 'spair_synthetic', 'spair_2dwarp']:
        cond_df = bench_df[(bench_df['condition'] == condition) & 
                           (bench_df['pretrained'] == False) & 
                           (bench_df['freeze'] == False)]
        
        if len(cond_df) == 0:
            continue
        
        if condition == 'spair_only':
            best_configs[condition] = None
            grouped = cond_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
            ax.plot(grouped['training_steps'], grouped['mean'],
                   color=condition_colors[condition], label=condition_labels[condition],
                   linewidth=3, marker='o', markersize=5)
        else:
            # Find best mix ratio at final step
            final_step = cond_df['training_steps'].max()
            final_df = cond_df[cond_df['training_steps'] == final_step]
            if len(final_df) > 0:
                best_mix = final_df.groupby('mix_ratio')['pck'].mean().idxmax()
                best_configs[condition] = best_mix
                
                mix_df = cond_df[cond_df['mix_ratio'] == best_mix]
                grouped = mix_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
                label = f"{condition_labels[condition]} ({best_mix})"
                ax.plot(grouped['training_steps'], grouped['mean'],
                       color=condition_colors[condition], label=label,
                       linewidth=3, marker='o', markersize=5)
                if grouped['count'].max() > 1:
                    ax.fill_between(grouped['training_steps'],
                                   grouped['mean'] - grouped['std'],
                                   grouped['mean'] + grouped['std'],
                                   color=condition_colors[condition], alpha=0.2)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('PCK (%)', fontsize=12)
    ax.set_title('Best Configuration per Condition (From Scratch)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: RAFT models comparison
    ax = axes[1, 1]
    raft_df = bench_df[bench_df['model_type'] == 'raft']
    
    for condition in ['spair_only', 'spair_synthetic', 'spair_2dwarp']:
        cond_df = raft_df[raft_df['condition'] == condition]
        
        if len(cond_df) == 0:
            continue
        
        if condition == 'spair_only':
            grouped = cond_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
            ax.plot(grouped['training_steps'], grouped['mean'],
                   color=condition_colors[condition], label=condition_labels[condition],
                   linewidth=2, marker='o', markersize=4)
        else:
            for mix_ratio in ['30_70', '50_50', '70_30']:
                mix_df = cond_df[cond_df['mix_ratio'] == mix_ratio]
                if len(mix_df) == 0:
                    continue
                
                grouped = mix_df.groupby('training_steps')['pck'].agg(['mean', 'std', 'count']).reset_index()
                label = f"{condition_labels[condition]} ({mix_ratio})"
                ax.plot(grouped['training_steps'], grouped['mean'],
                       color=condition_colors[condition], label=label,
                       linewidth=2, marker='o', markersize=3,
                       linestyle='-' if mix_ratio == '50_50' else '--' if mix_ratio == '30_70' else ':')
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('PCK (%)', fontsize=12)
    ax.set_title('RAFT Models', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'learning_curves_{benchmark}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to: {output_path}")
    plt.close()


def plot_final_performance_comparison(df, output_dir, benchmark='spair'):
    """Bar plot comparing final performance across conditions"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Final Performance Comparison - {benchmark.upper()}', fontsize=16, fontweight='bold')
    
    # Get final performance
    final_df = get_final_performance(df, benchmark=benchmark)
    
    # Plot 1: From scratch (pretrained=False, freeze=False)
    ax = axes[0]
    from_scratch = final_df[(final_df['pretrained'] == False) & (final_df['freeze'] == False)]
    
    # Prepare data for plotting
    plot_data = []
    for _, row in from_scratch.iterrows():
        condition = row['condition']
        mix_ratio = row['mix_ratio']
        if condition == 'spair_only':
            label = 'SPAIR Only'
        elif condition == 'spair_synthetic':
            label = f'SPAIR+Synth\n({mix_ratio})'
        elif condition == 'spair_2dwarp':
            label = f'SPAIR+2DWarp\n({mix_ratio})'
        else:
            label = condition
        
        plot_data.append({
            'condition': label,
            'condition_type': condition,
            'pck': row['pck']
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Define colors
    colors = []
    for cond in plot_df['condition_type']:
        if cond == 'spair_only':
            colors.append('#d62728')
        elif cond == 'spair_synthetic':
            colors.append('#2ca02c')
        else:
            colors.append('#ff7f0e')
    
    x_pos = range(len(plot_df))
    ax.bar(x_pos, plot_df['pck'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df['condition'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('PCK (%)', fontsize=12)
    ax.set_title('From Scratch (pretrained=False, freeze=False)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(plot_df['pck']):
        ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Statistical comparison (t-tests)
    ax = axes[1]
    
    # Compare SPAIR only vs best synthetic vs best 2dwarp
    spair_only_pck = from_scratch[from_scratch['condition'] == 'spair_only']['pck'].values
    synthetic_pcks = from_scratch[from_scratch['condition'] == 'spair_synthetic'].groupby('mix_ratio')['pck'].apply(list)
    warp_pcks = from_scratch[from_scratch['condition'] == 'spair_2dwarp'].groupby('mix_ratio')['pck'].apply(list)
    
    comparison_data = []
    if len(spair_only_pck) > 0:
        # Best synthetic
        if len(synthetic_pcks) > 0:
            best_synth_ratio = synthetic_pcks.apply(np.mean).idxmax()
            best_synth_pck = synthetic_pcks[best_synth_ratio]
            if len(best_synth_pck) > 1 and len(spair_only_pck) > 1:
                t_stat, p_val = stats.ttest_ind(best_synth_pck, spair_only_pck)
                comparison_data.append({
                    'Comparison': f'Synth({best_synth_ratio})\nvs SPAIR',
                    'Δ PCK': np.mean(best_synth_pck) - np.mean(spair_only_pck),
                    'p-value': p_val
                })
            else:
                comparison_data.append({
                    'Comparison': f'Synth({best_synth_ratio})\nvs SPAIR',
                    'Δ PCK': np.mean(best_synth_pck) - np.mean(spair_only_pck),
                    'p-value': None
                })
        
        # Best 2dwarp
        if len(warp_pcks) > 0:
            best_warp_ratio = warp_pcks.apply(np.mean).idxmax()
            best_warp_pck = warp_pcks[best_warp_ratio]
            if len(best_warp_pck) > 1 and len(spair_only_pck) > 1:
                t_stat, p_val = stats.ttest_ind(best_warp_pck, spair_only_pck)
                comparison_data.append({
                    'Comparison': f'2DWarp({best_warp_ratio})\nvs SPAIR',
                    'Δ PCK': np.mean(best_warp_pck) - np.mean(spair_only_pck),
                    'p-value': p_val
                })
            else:
                comparison_data.append({
                    'Comparison': f'2DWarp({best_warp_ratio})\nvs SPAIR',
                    'Δ PCK': np.mean(best_warp_pck) - np.mean(spair_only_pck),
                    'p-value': None
                })
        
        # Synthetic vs 2dwarp
        if len(synthetic_pcks) > 0 and len(warp_pcks) > 0:
            if len(best_synth_pck) > 1 and len(best_warp_pck) > 1:
                t_stat, p_val = stats.ttest_ind(best_synth_pck, best_warp_pck)
                comparison_data.append({
                    'Comparison': f'Synth({best_synth_ratio})\nvs 2DWarp({best_warp_ratio})',
                    'Δ PCK': np.mean(best_synth_pck) - np.mean(best_warp_pck),
                    'p-value': p_val
                })
            else:
                comparison_data.append({
                    'Comparison': f'Synth({best_synth_ratio})\nvs 2DWarp({best_warp_ratio})',
                    'Δ PCK': np.mean(best_synth_pck) - np.mean(best_warp_pck),
                    'p-value': None
                })
    
    if comparison_data:
        comp_df = pd.DataFrame(comparison_data)
        colors = ['green' if delta > 0 else 'red' for delta in comp_df['Δ PCK']]
        
        x_pos = range(len(comp_df))
        bars = ax.bar(x_pos, comp_df['Δ PCK'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(comp_df['Comparison'], rotation=0, ha='center', fontsize=10)
        ax.set_ylabel('Δ PCK (%)', fontsize=12)
        ax.set_title('Performance Differences (From Scratch)', fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value and p-value labels
        for i, (delta, p_val) in enumerate(zip(comp_df['Δ PCK'], comp_df['p-value'])):
            y_pos = delta + (0.5 if delta > 0 else -0.5)
            label = f'{delta:+.1f}'
            if p_val is not None:
                if p_val < 0.001:
                    label += '\n***'
                elif p_val < 0.01:
                    label += '\n**'
                elif p_val < 0.05:
                    label += '\n*'
                else:
                    label += f'\np={p_val:.3f}'
            ax.text(i, y_pos, label, ha='center', va='bottom' if delta > 0 else 'top', 
                   fontsize=9, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Insufficient data for statistical comparison',
               transform=ax.transAxes, ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'final_performance_{benchmark}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved final performance comparison to: {output_path}")
    plt.close()


def generate_summary_report(df, output_dir, benchmark='spair'):
    """Generate text summary report"""
    output_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("SPARSE DATA + REGULARIZATION ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("Hypothesis: Synthetic data provides dense geometric regularization\n")
        f.write("that improves learning from sparse labels (SPAIR dataset).\n\n")
        
        # Get final performance
        final_df = get_final_performance(df, benchmark=benchmark)
        
        # From scratch analysis
        f.write("-"*80 + "\n")
        f.write("FROM SCRATCH TRAINING (pretrained=False, freeze=False)\n")
        f.write("-"*80 + "\n\n")
        
        from_scratch = final_df[(final_df['pretrained'] == False) & (final_df['freeze'] == False)]
        
        # SPAIR only
        spair_only = from_scratch[from_scratch['condition'] == 'spair_only']
        if len(spair_only) > 0:
            f.write(f"SPAIR Only (sparse labels):\n")
            f.write(f"  Mean PCK: {spair_only['pck'].mean():.2f} ± {spair_only['pck'].std():.2f}\n")
            f.write(f"  N runs: {len(spair_only)}\n\n")
        
        # SPAIR + Synthetic
        spair_synth = from_scratch[from_scratch['condition'] == 'spair_synthetic']
        if len(spair_synth) > 0:
            f.write(f"SPAIR + Synthetic:\n")
            for mix_ratio in ['30_70', '50_50', '70_30']:
                mix_df = spair_synth[spair_synth['mix_ratio'] == mix_ratio]
                if len(mix_df) > 0:
                    improvement = mix_df['pck'].mean() - spair_only['pck'].mean() if len(spair_only) > 0 else 0
                    f.write(f"  {mix_ratio}: {mix_df['pck'].mean():.2f} ± {mix_df['pck'].std():.2f} ")
                    f.write(f"(Δ={improvement:+.2f}, N={len(mix_df)})\n")
            f.write("\n")
        
        # SPAIR + 2D Warp
        spair_warp = from_scratch[from_scratch['condition'] == 'spair_2dwarp']
        if len(spair_warp) > 0:
            f.write(f"SPAIR + 2D ImageNet Warp:\n")
            for mix_ratio in ['30_70', '50_50', '70_30']:
                mix_df = spair_warp[spair_warp['mix_ratio'] == mix_ratio]
                if len(mix_df) > 0:
                    improvement = mix_df['pck'].mean() - spair_only['pck'].mean() if len(spair_only) > 0 else 0
                    f.write(f"  {mix_ratio}: {mix_df['pck'].mean():.2f} ± {mix_df['pck'].std():.2f} ")
                    f.write(f"(Δ={improvement:+.2f}, N={len(mix_df)})\n")
            f.write("\n")
        
        # Statistical comparisons
        f.write("-"*80 + "\n")
        f.write("STATISTICAL COMPARISONS (t-tests)\n")
        f.write("-"*80 + "\n\n")
        
        if len(spair_only) > 0:
            spair_only_pck = spair_only['pck'].values
            
            # Best synthetic vs SPAIR
            if len(spair_synth) > 0:
                best_synth_ratio = spair_synth.groupby('mix_ratio')['pck'].mean().idxmax()
                best_synth = spair_synth[spair_synth['mix_ratio'] == best_synth_ratio]
                best_synth_pck = best_synth['pck'].values
                
                f.write(f"Best Synthetic ({best_synth_ratio}) vs SPAIR Only:\n")
                f.write(f"  Mean difference: {best_synth_pck.mean() - spair_only_pck.mean():+.2f} PCK\n")
                if len(best_synth_pck) > 1 and len(spair_only_pck) > 1:
                    t_stat, p_val = stats.ttest_ind(best_synth_pck, spair_only_pck)
                    f.write(f"  t-statistic: {t_stat:.3f}\n")
                    f.write(f"  p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}\n")
                f.write("\n")
            
            # Best 2dwarp vs SPAIR
            if len(spair_warp) > 0:
                best_warp_ratio = spair_warp.groupby('mix_ratio')['pck'].mean().idxmax()
                best_warp = spair_warp[spair_warp['mix_ratio'] == best_warp_ratio]
                best_warp_pck = best_warp['pck'].values
                
                f.write(f"Best 2D Warp ({best_warp_ratio}) vs SPAIR Only:\n")
                f.write(f"  Mean difference: {best_warp_pck.mean() - spair_only_pck.mean():+.2f} PCK\n")
                if len(best_warp_pck) > 1 and len(spair_only_pck) > 1:
                    t_stat, p_val = stats.ttest_ind(best_warp_pck, spair_only_pck)
                    f.write(f"  t-statistic: {t_stat:.3f}\n")
                    f.write(f"  p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}\n")
                f.write("\n")
            
            # Synthetic vs 2dwarp
            if len(spair_synth) > 0 and len(spair_warp) > 0:
                f.write(f"Best Synthetic ({best_synth_ratio}) vs Best 2D Warp ({best_warp_ratio}):\n")
                f.write(f"  Mean difference: {best_synth_pck.mean() - best_warp_pck.mean():+.2f} PCK\n")
                if len(best_synth_pck) > 1 and len(best_warp_pck) > 1:
                    t_stat, p_val = stats.ttest_ind(best_synth_pck, best_warp_pck)
                    f.write(f"  t-statistic: {t_stat:.3f}\n")
                    f.write(f"  p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}\n")
                f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("="*80 + "\n\n")
        
        f.write("Based on the results above:\n\n")
        
        if len(spair_only) > 0 and len(spair_synth) > 0:
            if best_synth_pck.mean() > spair_only_pck.mean():
                f.write("✓ Synthetic data improves performance over SPAIR-only baseline\n")
            else:
                f.write("✗ Synthetic data does not improve performance over SPAIR-only baseline\n")
        
        if len(spair_synth) > 0 and len(spair_warp) > 0:
            if best_synth_pck.mean() > best_warp_pck.mean():
                f.write("✓ Synthetic data outperforms 2D ImageNet warps\n")
            else:
                f.write("✗ Synthetic data does not outperform 2D ImageNet warps\n")
        
        f.write("\nThis supports/refutes the hypothesis that synthetic data provides\n")
        f.write("dense geometric regularization that improves learning from sparse labels.\n")
    
    print(f"Saved summary report to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze sparse data + regularization experiments'
    )
    parser.add_argument('--spair-only-dir', type=str, required=True,
                       help='Directory containing SPAIR+synthetic snapshots', dest='spair_only_dir')
    parser.add_argument('--2dwarp-dir', type=str, required=True,
                       help='Directory containing SPAIR+2dwarp snapshots', dest='dwarp_dir')
    parser.add_argument('--output-dir', type=str, default='analysis/sparse_regularization',
                       help='Output directory for analysis results')
    parser.add_argument('--benchmark', type=str, default='spair',
                       help='Benchmark to analyze (default: spair)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("SPARSE DATA + REGULARIZATION ANALYSIS")
    print("="*80)
    print()
    
    # Load all snapshots
    print("Loading snapshots...")
    df_spair_synth = load_all_snapshots(args.spair_only_dir)
    print(f"  Loaded {len(df_spair_synth['snapshot_name'].unique())} snapshots from {args.spair_only_dir}")
    
    df_2dwarp = load_all_snapshots(args.dwarp_dir)
    print(f"  Loaded {len(df_2dwarp['snapshot_name'].unique())} snapshots from {args.dwarp_dir}")
    
    # Combine
    df_all = pd.concat([df_spair_synth, df_2dwarp], ignore_index=True)
    print(f"  Total: {len(df_all['snapshot_name'].unique())} snapshots")
    print()
    
    # Generate plots
    print("Generating plots...")
    plot_learning_curves(df_all, args.output_dir, benchmark=args.benchmark)
    plot_final_performance_comparison(df_all, args.output_dir, benchmark=args.benchmark)
    
    # Generate summary report
    print("Generating summary report...")
    generate_summary_report(df_all, args.output_dir, benchmark=args.benchmark)
    
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
