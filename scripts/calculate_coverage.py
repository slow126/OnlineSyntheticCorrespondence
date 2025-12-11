"""
Calculate pairwise coverage metrics between datasets.

This script computes coverage metrics between all pairs of precomputed coresets,
similar to calculate_mmd.py but for coverage metrics.

Usage:
    python scripts/calculate_coverage.py --coresets-dir coresets/ --output coverage_results.csv
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from itertools import combinations

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset import WeightedCoreset, coverage_by_train, extraneous_mass_fraction


def parse_coreset_filename(filepath):
    """
    Parse coreset filename to extract dataset name and split.
    
    Expected format: {dataset}_{split}_flow.pt
    Example: synthetic_train_flow.pt -> (synthetic, train)
    """
    stem = Path(filepath).stem  # Remove .pt extension
    
    # Remove _flow suffix if present
    if stem.endswith('_flow'):
        stem = stem[:-5]
    
    # Split by last underscore to separate dataset from split
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        dataset, split = parts
        return dataset, split
    else:
        # Fallback: treat entire name as dataset, split unknown
        return stem, 'unknown'


def load_coreset_files(coresets_dir):
    """
    Load all coreset files from directory.
    
    Returns:
        list of dict with keys: 'path', 'dataset', 'split', 'coreset'
    """
    coresets_dir = Path(coresets_dir)
    coreset_files = list(coresets_dir.glob('*_flow.pt'))
    
    if not coreset_files:
        raise ValueError(f"No coreset files (*_flow.pt) found in {coresets_dir}")
    
    print(f"Found {len(coreset_files)} coreset files")
    
    coresets = []
    for filepath in coreset_files:
        dataset, split = parse_coreset_filename(filepath)
        print(f"  Loading: {filepath.name} -> dataset={dataset}, split={split}")
        
        coreset = WeightedCoreset.load(str(filepath))
        
        coresets.append({
            'path': str(filepath),
            'filename': filepath.name,
            'dataset': dataset,
            'split': split,
            'coreset': coreset
        })
        
        print(f"    Centers: {len(coreset.get_centers())}, Total samples: {coreset.total_samples}")
    
    return coresets


def compute_pairwise_coverage(coresets, epsilon_choice='eps_base', min_count=0):
    """
    Compute coverage metrics for all pairs of coresets.
    
    For each pair (train_coreset, eval_coreset), compute:
    - coverage_rel: relative coverage
    - coverage_abs: absolute coverage
    - rho_95, rho_median, rho_mean: distance quantiles
    - extraneous_mass_frac: fraction of train mass not needed for eval
    
    Args:
        coresets: List of coreset dicts
        epsilon_choice: Which epsilon to use ('eps_base', 'eps_2x', 'eps_4x', or float)
        min_count: Minimum count for absolute coverage
    
    Returns:
        List of result dicts with keys: dataset1, split1, dataset2, split2, metrics...
    """
    results = []
    
    # Compute for all pairs (including self-pairs for reference)
    total_pairs = len(coresets) * len(coresets)
    
    print(f"\nComputing coverage for {total_pairs} pairs...")
    print(f"Using epsilon choice: {epsilon_choice}")
    print("="*60)
    
    for i, train_info in enumerate(coresets):
        for j, eval_info in enumerate(coresets):
            train_coreset = train_info['coreset']
            eval_coreset = eval_info['coreset']
            
            train_centers = train_coreset.get_centers()
            train_counts = train_coreset.get_counts()
            eval_centers = eval_coreset.get_centers()
            eval_counts = eval_coreset.get_counts()
            
            # Get epsilon
            if isinstance(epsilon_choice, (int, float)):
                epsilon = epsilon_choice
            else:
                # Use epsilon from eval coreset
                epsilon_scales = eval_coreset.get_epsilon_scales()
                if epsilon_scales is None:
                    print(f"Warning: {eval_info['filename']} missing epsilon scales, skipping")
                    continue
                
                if epsilon_choice not in epsilon_scales:
                    print(f"Warning: {epsilon_choice} not in {eval_info['filename']}, skipping")
                    continue
                
                epsilon = epsilon_scales[epsilon_choice]
            
            print(f"[{i*len(coresets) + j + 1}/{total_pairs}] "
                  f"{train_info['dataset']}_{train_info['split']} -> "
                  f"{eval_info['dataset']}_{eval_info['split']} (ε={epsilon:.4f})")
            
            # Compute coverage
            coverage = coverage_by_train(
                train_centers, train_counts, eval_centers,
                epsilon=epsilon, min_count=min_count
            )
            
            # Compute extraneous mass
            extran = extraneous_mass_fraction(
                train_centers, train_counts, eval_centers,
                epsilon=epsilon
            )
            
            # Store result
            result = {
                'dataset1': train_info['dataset'],
                'split1': train_info['split'],
                'dataset2': eval_info['dataset'],
                'split2': eval_info['split'],
                'epsilon': epsilon,
                'coverage_rel': coverage['coverage_rel'],
                'coverage_abs': coverage['coverage_abs'],
                'rho_95': coverage['rho_95'],
                'rho_median': coverage['rho_median'],
                'rho_mean': coverage['rho_mean'],
                'extraneous_mass_frac': extran['extraneous_mass_frac'],
                'extraneous_centers_frac': extran['extraneous_centers_frac'],
            }
            
            results.append(result)
            
            print(f"  Coverage (rel): {coverage['coverage_rel']:.2%}, "
                  f"Coverage (abs): {coverage['coverage_abs']:.2%}")
    
    return results


def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    if not results:
        print("No results to save!")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'dataset1', 'split1', 'dataset2', 'split2', 'epsilon',
        'coverage_rel', 'coverage_abs', 
        'rho_95', 'rho_median', 'rho_mean',
        'extraneous_mass_frac', 'extraneous_centers_frac'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSaved {len(results)} results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate pairwise coverage metrics between coresets'
    )
    parser.add_argument(
        '--coresets-dir', type=str, default='coresets/',
        help='Directory containing coreset files (default: coresets/)'
    )
    parser.add_argument(
        '--output', type=str, default='coverage_results.csv',
        help='Output CSV file (default: coverage_results.csv)'
    )
    parser.add_argument(
        '--epsilon', type=str, default='eps_base',
        help='Epsilon choice: eps_base, eps_2x, eps_4x, or a float value (default: eps_base)'
    )
    parser.add_argument(
        '--min-count', type=int, default=0,
        help='Minimum count for absolute coverage (default: 0)'
    )
    args = parser.parse_args()
    
    # Parse epsilon
    try:
        epsilon_choice = float(args.epsilon)
    except ValueError:
        epsilon_choice = args.epsilon
    
    # Load coresets
    print("="*60)
    print("LOADING CORESETS")
    print("="*60)
    coresets = load_coreset_files(args.coresets_dir)
    
    # Compute pairwise coverage
    results = compute_pairwise_coverage(coresets, epsilon_choice, args.min_count)
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    save_results_to_csv(results, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
