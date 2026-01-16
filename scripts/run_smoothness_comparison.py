#!/usr/bin/env python3
"""
Wrapper script to automate smoothness analysis across multiple training conditions.

This script:
1. Finds best checkpoints from each training condition
2. Runs smoothness calculation on SPAIR test set
3. Aggregates and compares results across conditions
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150


def find_best_checkpoint(snapshot_dir, benchmark=None):
    """
    Find the best checkpoint in a snapshot directory.
    
    Args:
        snapshot_dir: Path to snapshot directory
        benchmark: Optional benchmark name to find benchmark-specific checkpoint
        
    Returns:
        Path to best checkpoint
    """
    snapshot_dir = Path(snapshot_dir)
    
    if not snapshot_dir.exists():
        return None
    
    # Priority order:
    # 1. {benchmark}_best.pth (benchmark-specific best)
    # 2. model_best.pth (best validation performance)
    # 3. Latest checkpoint by training steps
    
    # Check for benchmark-specific checkpoint
    if benchmark:
        benchmark_best = snapshot_dir / f'{benchmark}_best.pth'
        if benchmark_best.exists() and benchmark_best.stat().st_size > 0:
            print(f"    Using benchmark-specific checkpoint: {benchmark}_best.pth")
            return str(benchmark_best)
    
    # Check for model_best.pth
    model_best = snapshot_dir / 'model_best.pth'
    if model_best.exists() and model_best.stat().st_size > 0:
        return str(model_best)
    
    # Find latest checkpoint
    checkpoints = list(snapshot_dir.glob('*.pth'))
    if not checkpoints:
        # Check snapshots subdirectory
        snapshots_subdir = snapshot_dir / 'snapshots'
        if snapshots_subdir.exists():
            checkpoints = list(snapshots_subdir.glob('*.pth'))
    
    # Filter out empty files
    checkpoints = [c for c in checkpoints if c.stat().st_size > 0]
    
    if not checkpoints:
        return None
    
    # Sort by modification time and return latest
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return str(checkpoints[0])


def parse_snapshot_name(snapshot_name):
    """Parse training configuration from snapshot directory name."""
    config = {
        'condition': 'unknown',
        'mix_ratio': None,
        'model_type': 'unknown',
        'pretrained': None,
        'freeze': None,
    }

    snapshot_lower = snapshot_name.lower()

    # Base datasets (no SPAIR mixing).
    if snapshot_lower.startswith('synthetic'):
        config['condition'] = 'synthetic_only'
    elif snapshot_lower.startswith('imagenet2dwarp') or snapshot_lower.startswith('2dwarp'):
        config['condition'] = '2dwarp_only'
        config['mix_ratio'] = '100_0'
    # Mixed datasets.
    elif 'spair_only' in snapshot_lower or (
        snapshot_lower.startswith('spair_')
        and 'synthetic' not in snapshot_lower
        and '2d_warp' not in snapshot_lower
        and '2dwarp' not in snapshot_lower
        and 'imagenet' not in snapshot_lower
    ):
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
    elif '2d_warp' in snapshot_lower or '2dwarp' in snapshot_lower or 'imagenet2dwarp' in snapshot_lower:
        config['condition'] = 'spair_2dwarp'
        # Extract mixing ratio
        if '30_70' in snapshot_lower:
            config['mix_ratio'] = '30_70'
        elif '50_50' in snapshot_lower:
            config['mix_ratio'] = '50_50'
        elif '70_30' in snapshot_lower:
            config['mix_ratio'] = '70_30'
        elif 'imagenet2dwarp_cats' in snapshot_lower or 'imagenet2dwarp_raft' in snapshot_lower:
            config['mix_ratio'] = '100_0'

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


def collect_checkpoints(snapshot_dirs, allowed_prefixes=None):
    """Collect all checkpoints from snapshot directories."""
    checkpoint_info = []
    allowed = [p.lower() for p in allowed_prefixes] if allowed_prefixes else None
    
    for snapshot_dir in snapshot_dirs:
        snapshot_dir = Path(snapshot_dir)
        if not snapshot_dir.exists():
            print(f"Warning: Directory not found: {snapshot_dir}")
            continue
        
        print(f"  Scanning {snapshot_dir.name}...")
        subdirs = [d for d in snapshot_dir.iterdir() if d.is_dir()]
        
        # Iterate through all snapshot subdirectories
        for snapshot_subdir in tqdm(subdirs, desc=f"    Checking snapshots", unit="dir"):
            if not snapshot_subdir.is_dir():
                continue

            if allowed:
                name_lower = snapshot_subdir.name.lower()
                if not any(name_lower.startswith(prefix) for prefix in allowed):
                    continue
            
            # Check if this is a valid snapshot (has validation_results.csv)
            validation_csv = snapshot_subdir / 'validation_results.csv'
            if not validation_csv.exists():
                # Skip empty/incomplete snapshots
                continue
            
            # Store snapshot directory info (checkpoint will be selected per-benchmark)
            # Parse configuration
            config = parse_snapshot_name(snapshot_subdir.name)
            
            checkpoint_info.append({
                'snapshot_dir': str(snapshot_subdir),
                'snapshot_name': snapshot_subdir.name,
                'condition': config['condition'],
                'mix_ratio': config['mix_ratio'],
                'model_type': config['model_type'],
                'pretrained': config['pretrained'],
                'freeze': config['freeze'],
            })
    
    return pd.DataFrame(checkpoint_info)


def run_smoothness_calculation(
    checkpoints_df,
    benchmarks,
    output_dir,
    batch_size=16,
    num_workers=4,
    device='cuda',
    include_gt=False,
    mask_by_gt=False,
    tss_root=None,
):
    """Run smoothness calculation using calculate_flow_smoothness.py"""
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create temporary snapshot list file (with snapshot_dir instead of checkpoint_path)
    temp_checkpoint_file = Path(output_dir) / 'temp_checkpoints.csv'
    checkpoints_df[['snapshot_dir', 'snapshot_name']].to_csv(temp_checkpoint_file, index=False)
    
    # Run calculate_flow_smoothness.py
    cmd = [
        'python', 'scripts/calculate_flow_smoothness.py',
        '--checkpoints', str(temp_checkpoint_file),
        '--benchmarks'] + benchmarks + [
        '--output', str(Path(output_dir) / 'smoothness_raw_results.csv'),
        '--batch-size', str(batch_size),
        '--num-workers', str(num_workers),
        '--device', device
    ]
    if include_gt:
        cmd.append('--include-gt')
    if mask_by_gt:
        cmd.append('--mask-by-gt')
    if tss_root:
        cmd += ['--tss-root', str(tss_root)]
    
    print(f"\nRunning smoothness calculation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Stream output in real-time
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
            
    except subprocess.CalledProcessError as e:
        print(f"\nError running smoothness calculation: {e}")
        raise
    finally:
        # Clean up temp file
        if temp_checkpoint_file.exists():
            temp_checkpoint_file.unlink()
    
    # Load and return results
    results_path = Path(output_dir) / 'smoothness_raw_results.csv'
    if results_path.exists():
        return pd.read_csv(results_path)
    else:
        raise FileNotFoundError(f"Smoothness results not found at {results_path}")


def aggregate_smoothness_results(smoothness_df, checkpoint_info_df):
    """Aggregate smoothness results with checkpoint metadata"""
    # Merge smoothness results with checkpoint info
    # Match on checkpoint_name (from smoothness) and snapshot_name (from checkpoint_info)
    merged = smoothness_df.merge(
        checkpoint_info_df,
        left_on='checkpoint_name',
        right_on='snapshot_name',
        how='left'
    )
    
    return merged


def plot_smoothness_comparison(df, output_dir, benchmark='spair'):
    """Create comparison plots for smoothness metrics"""
    
    # Filter to specific benchmark
    bench_df = df[df['benchmark'] == benchmark].copy()
    
    # Focus on from-scratch training (most relevant)
    from_scratch = bench_df[(bench_df['pretrained'] == False) & (bench_df['freeze'] == False)]
    
    if len(from_scratch) == 0:
        print(f"Warning: No from-scratch data for benchmark {benchmark}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Flow Prediction Smoothness - {benchmark.upper()} (From Scratch)', 
                 fontsize=16, fontweight='bold')
    
    # Define colors
    condition_colors = {
        'spair_only': '#d62728',
        'spair_synthetic': '#2ca02c',
        'spair_2dwarp': '#ff7f0e',
    }
    
    # Plot 1: Total Variation
    ax = axes[0]
    
    # Prepare data for plotting
    plot_data_tv = []
    for _, row in from_scratch.iterrows():
        condition = row['condition']
        mix_ratio = row['mix_ratio']
        
        if condition == 'spair_only':
            label = 'SPAIR Only'
        elif condition == 'spair_synthetic':
            label = f'Synth ({mix_ratio})'
        elif condition == 'spair_2dwarp':
            label = f'2DWarp ({mix_ratio})'
        else:
            label = condition
        
        plot_data_tv.append({
            'label': label,
            'condition': condition,
            'tv': row['mean_tv']
        })
    
    plot_df_tv = pd.DataFrame(plot_data_tv)
    
    # Sort by condition for better visualization
    plot_df_tv = plot_df_tv.sort_values(['condition', 'tv'])
    
    colors = [condition_colors.get(c, 'gray') for c in plot_df_tv['condition']]
    x_pos = range(len(plot_df_tv))
    
    ax.bar(x_pos, plot_df_tv['tv'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df_tv['label'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Total Variation (lower = smoother)', fontsize=12)
    ax.set_title('Total Variation of Flow Predictions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(plot_df_tv['tv']):
        ax.text(i, v + v*0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Laplacian Smoothness
    ax = axes[1]
    
    plot_data_lap = []
    for _, row in from_scratch.iterrows():
        condition = row['condition']
        mix_ratio = row['mix_ratio']
        
        if condition == 'spair_only':
            label = 'SPAIR Only'
        elif condition == 'spair_synthetic':
            label = f'Synth ({mix_ratio})'
        elif condition == 'spair_2dwarp':
            label = f'2DWarp ({mix_ratio})'
        else:
            label = condition
        
        plot_data_lap.append({
            'label': label,
            'condition': condition,
            'laplacian': row['mean_laplacian']
        })
    
    plot_df_lap = pd.DataFrame(plot_data_lap)
    plot_df_lap = plot_df_lap.sort_values(['condition', 'laplacian'])
    
    colors = [condition_colors.get(c, 'gray') for c in plot_df_lap['condition']]
    x_pos = range(len(plot_df_lap))
    
    ax.bar(x_pos, plot_df_lap['laplacian'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(plot_df_lap['label'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Laplacian Magnitude (lower = smoother)', fontsize=12)
    ax.set_title('Laplacian Smoothness of Flow Predictions', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(plot_df_lap['laplacian']):
        ax.text(i, v + v*0.02, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / f'smoothness_comparison_{benchmark}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved smoothness comparison to: {output_path}")
    plt.close()


def load_pck_from_snapshots(df):
    """Load PCK metrics from validation_results.csv for each snapshot + benchmark combo"""
    pck_data = []
    df = df.copy()

    def derive_snapshot_dir(checkpoint_path):
        if not isinstance(checkpoint_path, str):
            return None
        path = Path(checkpoint_path)
        snapshot_dir = path.parent
        if snapshot_dir.name in {'checkpoints', 'snapshots'}:
            snapshot_dir = snapshot_dir.parent
        return str(snapshot_dir)

    if 'snapshot_dir' not in df.columns:
        if 'checkpoint_path' in df.columns:
            df['snapshot_dir'] = df['checkpoint_path'].apply(derive_snapshot_dir)
            if 'snapshot_name' not in df.columns:
                df['snapshot_name'] = df['snapshot_dir'].apply(
                    lambda p: Path(p).name if isinstance(p, str) else None
                )
        else:
            print("Warning: snapshot_dir missing and checkpoint_path unavailable; skipping PCK merge.")
            return pd.DataFrame()
    
    # Get unique snapshot_dir + benchmark combinations
    unique_combos = df[['snapshot_dir', 'benchmark', 'snapshot_name']].drop_duplicates()
    
    for _, row in unique_combos.iterrows():
        snapshot_dir = Path(row['snapshot_dir'])
        benchmark = row['benchmark']
        val_results_path = snapshot_dir / 'validation_results.csv'
        
        if val_results_path.exists():
            try:
                val_df = pd.read_csv(val_results_path)
                # Filter to this benchmark and get best PCK
                benchmark_rows = val_df[val_df['benchmark'] == benchmark]
                if len(benchmark_rows) > 0 and 'pck' in benchmark_rows.columns:
                    best_pck = benchmark_rows['pck'].max()
                    pck_data.append({
                        'snapshot_name': row['snapshot_name'],
                        'benchmark': benchmark,
                        'best_pck': best_pck
                    })
            except Exception as e:
                print(f"Warning: Could not load PCK from {val_results_path}: {e}")
    
    return pd.DataFrame(pck_data)


def generate_smoothness_summary(df, output_dir):
    """Generate summary statistics for smoothness metrics with PCK data"""
    output_path = Path(output_dir) / 'smoothness_summary.txt'
    
    # Load PCK data from validation_results.csv files
    print("Loading PCK data from validation_results.csv files...")
    pck_df = load_pck_from_snapshots(df)
    print(f"Loaded PCK data for {len(pck_df)} snapshot+benchmark combinations")
    
    # Merge PCK data with smoothness data
    if len(pck_df) > 0:
        df = df.merge(pck_df, on=['snapshot_name', 'benchmark'], how='left')
        print(f"Merged PCK data: {df['best_pck'].notna().sum()} rows have PCK values")
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FLOW PREDICTION SMOOTHNESS & PERFORMANCE ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Lower Total Variation (TV) and Laplacian values indicate smoother predictions.\n")
        f.write("Higher PCK indicates better correspondence accuracy.\n")
        f.write("This tests the hypothesis that synthetic data provides dense geometric regularization\n")
        f.write("WITHOUT sacrificing accuracy (i.e., not just blurring everything).\n\n")
        
        # From scratch analysis
        from_scratch = df[(df['pretrained'] == False) & (df['freeze'] == False)]
        
        if len(from_scratch) == 0:
            f.write("No from-scratch training data available.\n")
            return
        
        f.write("-"*80 + "\n")
        f.write("FROM SCRATCH TRAINING (pretrained=False, freeze=False)\n")
        f.write("-"*80 + "\n\n")
        
        # Group by condition
        for condition in ['spair_only', 'spair_synthetic', 'spair_2dwarp']:
            cond_df = from_scratch[from_scratch['condition'] == condition]
            
            if len(cond_df) == 0:
                continue
            
            f.write(f"{condition.upper()}:\n")
            
            if condition == 'spair_only':
                f.write(f"  Mean TV: {cond_df['mean_tv'].mean():.6f} ± {cond_df['mean_tv'].std():.6f}\n")
                f.write(f"  Mean Laplacian: {cond_df['mean_laplacian'].mean():.6f} ± {cond_df['mean_laplacian'].std():.6f}\n")
                
                # Add PCK if available
                if 'best_pck' in cond_df.columns:
                    f.write(f"  Mean PCK: {cond_df['best_pck'].mean():.2f}%\n")
                
                f.write(f"  N: {len(cond_df)}\n\n")
            else:
                # Show per mix ratio
                for mix_ratio in ['30_70', '50_50', '70_30']:
                    mix_df = cond_df[cond_df['mix_ratio'] == mix_ratio]
                    if len(mix_df) > 0:
                        f.write(f"  {mix_ratio}:\n")
                        f.write(f"    Mean TV: {mix_df['mean_tv'].mean():.6f}\n")
                        f.write(f"    Mean Laplacian: {mix_df['mean_laplacian'].mean():.6f}\n")
                        
                        # Add PCK if available
                        if 'best_pck' in mix_df.columns:
                            f.write(f"    Mean PCK: {mix_df['best_pck'].mean():.2f}%\n")
                        
                        f.write(f"    N: {len(mix_df)}\n")
                f.write("\n")
        
        # Find best (lowest smoothness metrics)
        spair_only = from_scratch[from_scratch['condition'] == 'spair_only']
        spair_synth = from_scratch[from_scratch['condition'] == 'spair_synthetic']
        spair_warp = from_scratch[from_scratch['condition'] == 'spair_2dwarp']
        
        f.write("-"*80 + "\n")
        f.write("CONCLUSION\n")
        f.write("-"*80 + "\n\n")
        
        if len(spair_only) > 0:
            baseline_tv = spair_only['mean_tv'].mean()
            baseline_lap = spair_only['mean_laplacian'].mean()
            baseline_pck = spair_only['best_pck'].mean() if 'best_pck' in spair_only.columns else None
            
            if len(spair_synth) > 0:
                # Find best synthetic ratio
                best_synth_tv = spair_synth.groupby('mix_ratio')['mean_tv'].mean()
                best_synth_ratio_tv = best_synth_tv.idxmin()
                synth_tv = best_synth_tv[best_synth_ratio_tv]
                
                tv_improvement = ((baseline_tv - synth_tv) / baseline_tv) * 100
                
                f.write(f"Best Synthetic ({best_synth_ratio_tv}):\n")
                f.write(f"  TV improvement over SPAIR-only: {tv_improvement:+.1f}%\n")
                
                # Add PCK comparison
                if baseline_pck is not None and 'best_pck' in spair_synth.columns:
                    synth_pck_by_ratio = spair_synth.groupby('mix_ratio')['best_pck'].mean()
                    if best_synth_ratio_tv in synth_pck_by_ratio:
                        synth_pck = synth_pck_by_ratio[best_synth_ratio_tv]
                        pck_change = synth_pck - baseline_pck
                        f.write(f"  PCK change: {pck_change:+.2f}% (from {baseline_pck:.2f}% to {synth_pck:.2f}%)\n")
                
                if synth_tv < baseline_tv:
                    f.write("  ✓ Synthetic produces SMOOTHER predictions (supports hypothesis)\n")
                else:
                    f.write("  ✗ Synthetic does NOT produce smoother predictions\n")
                f.write("\n")
            
            if len(spair_warp) > 0:
                # Find best warp ratio
                best_warp_tv = spair_warp.groupby('mix_ratio')['mean_tv'].mean()
                best_warp_ratio_tv = best_warp_tv.idxmin()
                warp_tv = best_warp_tv[best_warp_ratio_tv]
                
                tv_improvement_warp = ((baseline_tv - warp_tv) / baseline_tv) * 100
                
                f.write(f"Best 2D Warp ({best_warp_ratio_tv}):\n")
                f.write(f"  TV improvement over SPAIR-only: {tv_improvement_warp:+.1f}%\n\n")
            
            if len(spair_synth) > 0 and len(spair_warp) > 0:
                if synth_tv < warp_tv:
                    f.write("✓ Synthetic is SMOOTHER than 2D warps\n")
                    f.write("  This supports the claim that 3D geometric consistency matters.\n\n")
                else:
                    f.write("✗ Synthetic is NOT smoother than 2D warps\n\n")
            
            # Final key finding
            if len(spair_synth) > 0 and baseline_pck is not None and 'best_pck' in spair_synth.columns:
                synth_pck_by_ratio = spair_synth.groupby('mix_ratio')['best_pck'].mean()
                if best_synth_ratio_tv in synth_pck_by_ratio:
                    synth_pck = synth_pck_by_ratio[best_synth_ratio_tv]
                    if synth_tv < baseline_tv and synth_pck > baseline_pck:
                        f.write("="*80 + "\n")
                        f.write("KEY FINDING:\n")
                        f.write("="*80 + "\n")
                        f.write("✓✓ Synthetic data produces predictions that are BOTH:\n")
                        f.write(f"   1. SMOOTHER (TV: {baseline_tv:.4f} → {synth_tv:.4f}, {tv_improvement:+.1f}%)\n")
                        f.write(f"   2. MORE ACCURATE (PCK: {baseline_pck:.2f}% → {synth_pck:.2f}%, {synth_pck-baseline_pck:+.2f}%)\n\n")
                        f.write("This demonstrates dense geometric regularization WITHOUT loss of detail.\n")
                        f.write("The model learns better correspondences, not just blurred predictions.\n")
    
    print(f"Saved smoothness summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run smoothness comparison across training conditions'
    )
    parser.add_argument('--snapshot-dirs', type=str, nargs='+', required=True,
                       help='Directories containing snapshot subdirectories')
    parser.add_argument('--output-dir', type=str, default='analysis/sparse_regularization',
                       help='Output directory for results')
    parser.add_argument('--benchmarks', type=str, nargs='+', default=['spair'],
                       help='Benchmarks to evaluate (default: spair)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference (default: 16, can go higher in eval mode)')
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--skip-calculation', action='store_true',
                       help='Skip smoothness calculation (use existing results)')
    parser.add_argument('--include-base-datasets', action='store_true',
                       help='Also include base synthetic/2dwarp datasets (no mixing).')
    parser.add_argument('--base-snapshot-dirs', type=str, nargs='*', default=None,
                       help='Directories to scan for base datasets (optional).')
    parser.add_argument('--base-prefixes', type=str, default='synthetic,imagenet2dwarp',
                       help='Comma-separated snapshot name prefixes to treat as base datasets.')
    parser.add_argument('--include-gt', action='store_true',
                       help='Also compute smoothness on ground-truth flow when available.')
    parser.add_argument('--mask-by-gt', action='store_true',
                       help='Compute smoothness only over valid GT pixels (mask invalid regions).')
    parser.add_argument('--tss-root', type=str, default=None,
                       help='Path to TSS dataset root (overrides config).')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("SMOOTHNESS COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # Collect all checkpoints
    print("Collecting checkpoints...", flush=True)
    checkpoint_df = collect_checkpoints(args.snapshot_dirs)
    print(f"  Found {len(checkpoint_df)} checkpoints", flush=True)
    if args.include_base_datasets:
        base_dirs = args.base_snapshot_dirs
        if base_dirs is None or len(base_dirs) == 0:
            base_dirs = ['snapshots', 'snapshots_mixed', 'snapshots_raft', 'snapshots_2d_warps']
        base_prefixes = [p.strip().lower() for p in args.base_prefixes.split(',') if p.strip()]
        print("Collecting base dataset checkpoints...", flush=True)
        base_df = collect_checkpoints(base_dirs, allowed_prefixes=base_prefixes)
        if not base_df.empty:
            checkpoint_df = pd.concat([checkpoint_df, base_df], ignore_index=True)
            checkpoint_df = checkpoint_df.drop_duplicates(
                subset=['snapshot_dir', 'snapshot_name']
            ).reset_index(drop=True)
        print(f"  Added {len(base_df)} base checkpoints (total {len(checkpoint_df)})", flush=True)
    print(flush=True)
    
    # Run smoothness calculation
    if not args.skip_calculation:
        smoothness_df = run_smoothness_calculation(
            checkpoint_df,
            args.benchmarks,
            args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
            include_gt=args.include_gt,
            mask_by_gt=args.mask_by_gt,
            tss_root=args.tss_root,
        )
    else:
        print("Skipping smoothness calculation, loading existing results...")
        results_path = Path(args.output_dir) / 'smoothness_raw_results.csv'
        if not results_path.exists():
            raise FileNotFoundError(f"No existing results found at {results_path}")
        smoothness_df = pd.read_csv(results_path)
    
    # Aggregate results
    print("Aggregating results...")
    merged_df = aggregate_smoothness_results(smoothness_df, checkpoint_df)
    
    # Save aggregated results
    output_csv = Path(args.output_dir) / 'smoothness_results_aggregated.csv'
    merged_df.to_csv(output_csv, index=False)
    print(f"  Saved aggregated results to: {output_csv}")
    
    # Generate plots
    print("\nGenerating plots...")
    for benchmark in args.benchmarks:
        bench_data = merged_df[merged_df['benchmark'] == benchmark]
        if len(bench_data) > 0:
            plot_smoothness_comparison(merged_df, args.output_dir, benchmark=benchmark)
    
    # Generate summary
    print("\nGenerating summary...")
    generate_smoothness_summary(merged_df, args.output_dir)
    
    print()
    print("="*80)
    print("SMOOTHNESS ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
