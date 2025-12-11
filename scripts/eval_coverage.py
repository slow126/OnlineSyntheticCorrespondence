"""
Evaluate coverage metrics between train and eval coresets.

This script loads precomputed coresets and computes coverage metrics,
using epsilon scales saved in the eval coreset.

Usage:
    # Use saved epsilon from eval coreset
    python scripts/eval_coverage.py \
        --train coresets/synthetic_train_flow.pt \
        --eval coresets/kitti_val_flow.pt \
        --output coverage_results.json
    
    # Override epsilon
    python scripts/eval_coverage.py \
        --train coresets/synthetic_train_flow.pt \
        --eval coresets/kitti_val_flow.pt \
        --epsilon 5.0 \
        --output coverage_results.json
"""

import argparse
import json
import numpy as np
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset import WeightedCoreset, coverage_by_train, extraneous_mass_fraction


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate coverage between train and eval coresets'
    )
    parser.add_argument(
        '--train', type=str, required=True,
        help='Path to train coreset (.pt file)'
    )
    parser.add_argument(
        '--eval', type=str, required=True,
        help='Path to eval coreset (.pt file)'
    )
    parser.add_argument(
        '--epsilon', type=float, default=None,
        help='Override epsilon (default: use saved epsilon from eval coreset)'
    )
    parser.add_argument(
        '--min-count', type=int, default=0,
        help='Minimum count for absolute coverage (default: 0)'
    )
    parser.add_argument(
        '--output', type=str, default='coverage_results.json',
        help='Output JSON file for results'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print detailed results'
    )
    args = parser.parse_args()
    
    # Load coresets
    print("="*60)
    print("LOADING CORESETS")
    print("="*60)
    print(f"Train: {args.train}")
    train_coreset = WeightedCoreset.load(args.train)
    print(f"  Centers: {len(train_coreset.get_centers())}")
    print(f"  Total samples: {train_coreset.total_samples}")
    
    print(f"\nEval: {args.eval}")
    eval_coreset = WeightedCoreset.load(args.eval)
    print(f"  Centers: {len(eval_coreset.get_centers())}")
    print(f"  Total samples: {eval_coreset.total_samples}")
    
    train_centers = train_coreset.get_centers()
    train_counts = train_coreset.get_counts()
    eval_centers = eval_coreset.get_centers()
    eval_counts = eval_coreset.get_counts()
    
    # Get epsilon scales
    if args.epsilon is not None:
        # User override
        epsilon_scales = {
            'eps_manual': args.epsilon,
        }
        print(f"\nUsing manual epsilon: {args.epsilon:.4f}")
    else:
        # Use saved epsilon from eval coreset
        epsilon_scales = eval_coreset.get_epsilon_scales()
        if epsilon_scales is None:
            raise ValueError(
                "Eval coreset does not have saved epsilon scales. "
                "Either rebuild with is_eval=True or use --epsilon flag."
            )
        print(f"\nUsing epsilon scales from eval coreset:")
        print(f"  eps_base: {epsilon_scales.get('eps_base', 'N/A')}")
        print(f"  eps_2x: {epsilon_scales.get('eps_2x', 'N/A')}")
        print(f"  eps_4x: {epsilon_scales.get('eps_4x', 'N/A')}")
    
    # Compute metrics for each epsilon scale
    print("\n" + "="*60)
    print("COMPUTING COVERAGE METRICS")
    print("="*60)
    
    all_results = {
        'train_file': args.train,
        'eval_file': args.eval,
        'train_centers': len(train_centers),
        'eval_centers': len(eval_centers),
        'train_total_samples': int(train_coreset.total_samples),
        'eval_total_samples': int(eval_coreset.total_samples),
        'min_count': args.min_count,
        'metrics_by_epsilon': {}
    }
    
    for eps_name, eps_value in epsilon_scales.items():
        if not isinstance(eps_value, (int, float)):
            continue
        
        print(f"\n{eps_name}: epsilon = {eps_value:.4f}")
        print("-" * 40)
        
        # Train → Eval coverage
        coverage = coverage_by_train(
            train_centers, train_counts, eval_centers,
            epsilon=eps_value, min_count=args.min_count
        )
        
        print(f"  Coverage (train → eval):")
        print(f"    Relative:  {coverage['coverage_rel']:.2%}")
        print(f"    Absolute:  {coverage['coverage_abs']:.2%}")
        print(f"    ρ_95:      {coverage['rho_95']:.4f}")
        print(f"    ρ_median:  {coverage['rho_median']:.4f}")
        print(f"    ρ_mean:    {coverage['rho_mean']:.4f}")
        
        # Train extraneous mass w.r.t. eval
        extran = extraneous_mass_fraction(
            train_centers, train_counts, eval_centers,
            epsilon=eps_value
        )
        
        print(f"  Extraneous mass (train w.r.t. eval):")
        print(f"    Mass fraction:    {extran['extraneous_mass_frac']:.2%}")
        print(f"    Centers fraction: {extran['extraneous_centers_frac']:.2%}")
        
        # Store results
        all_results['metrics_by_epsilon'][eps_name] = {
            'epsilon': eps_value,
            'coverage': coverage,
            'extraneous_mass': extran,
        }
    
    # Save results
    print("\n" + "="*60)
    print(f"SAVING RESULTS TO: {args.output}")
    print("="*60)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("Done!")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if 'eps_base' in all_results['metrics_by_epsilon']:
        base_metrics = all_results['metrics_by_epsilon']['eps_base']
        print(f"At eps_base = {base_metrics['epsilon']:.4f}:")
        print(f"  Coverage (rel): {base_metrics['coverage']['coverage_rel']:.2%}")
        print(f"  Coverage (abs): {base_metrics['coverage']['coverage_abs']:.2%}")
        print(f"  Extraneous mass: {base_metrics['extraneous_mass']['extraneous_mass_frac']:.2%}")


if __name__ == "__main__":
    main()
