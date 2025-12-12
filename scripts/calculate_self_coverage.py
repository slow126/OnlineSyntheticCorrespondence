"""
Quick script to calculate self-coverage of datasets for debugging.

This script computes recall/precision/outside metrics for each coreset
compared against itself (self-coverage).

Usage:
    python scripts/calculate_self_coverage.py --coresets-dir coresets/ --representation flow
"""

import argparse
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset import (
    WeightedCoreset,
    codebook_from_coreset,
    recall_train_covers_eval_soft,
    precision_train_wrt_eval_soft,
)


def parse_coreset_filename(filepath, suffix: str):
    """
    Parse coreset filename to extract dataset name and split.
    
    Expected format: {dataset}_{split}_{suffix}.pt
    Example: synthetic_train_flow.pt -> (synthetic, train)
    """
    stem = Path(filepath).stem  # Remove .pt extension
    
    # Remove suffix if present
    suffix_token = f"_{suffix}"
    if stem.endswith(suffix_token):
        stem = stem[: -len(suffix_token)]
    
    # Split by last underscore to separate dataset from split
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        dataset, split = parts
        return dataset, split
    else:
        # Fallback: treat entire name as dataset, split unknown
        return stem, 'unknown'


def main():
    parser = argparse.ArgumentParser(
        description='Calculate self-coverage of datasets (for debugging)'
    )
    parser.add_argument(
        '--coresets-dir', type=str, default='coresets/',
        help='Directory containing coreset files (default: coresets/)'
    )
    parser.add_argument(
        '--representation', type=str, default='flow',
        help='Representation suffix to load (e.g., flow, resnet)'
    )
    parser.add_argument(
        '--k', type=int, default=5,
        help='Number of neighbors for soft k-NN (default: 5)'
    )
    parser.add_argument(
        '--bandwidth', type=float, default=None,
        help='Bandwidth for kernel (default: inferred from distances)'
    )
    parser.add_argument(
        '--bandwidth-scale', type=float, default=1.0,
        help='Scale factor applied to inferred bandwidth (default: 1.0)'
    )
    parser.add_argument(
        '--M-train', type=float, default=100.0,
        help='Saturation threshold for recall (default: 100.0)'
    )
    parser.add_argument(
        '--M-eval', type=float, default=20.0,
        help='Saturation threshold for precision (default: 20.0)'
    )
    parser.add_argument(
        '--kernel', type=str, default='gaussian',
        choices=['gaussian', 'inverse'],
        help='Kernel type for weighting neighbors (default: gaussian)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=1024,
        help='Batch size for distance computation (default: 1024)'
    )
    parser.add_argument(
        '--adaptive-bandwidth', action='store_true', default=False,
        help='Enable per-pair adaptive bandwidth floor from k-NN distances'
    )
    parser.add_argument(
        '--min-bandwidth-quantile', type=float, default=0.3,
        help='Quantile of k-NN distances to use as minimum bandwidth when adaptive bandwidth is enabled (default: 0.3)'
    )
    parser.add_argument(
        '--adaptive-mass', action='store_true', default=False,
        help='Scale M_train/M_eval per pair using a quantile of the source counts to avoid saturation on dense codebooks'
    )
    parser.add_argument(
        '--mass-quantile', type=float, default=0.75,
        help='Quantile of source counts used when adaptive mass is enabled (default: 0.75)'
    )
    parser.add_argument(
        '--mass-floor', type=float, default=1.0,
        help='Minimum effective mass when adaptive mass is enabled (default: 1.0)'
    )
    args = parser.parse_args()
    
    # Load coresets
    coresets_dir = Path(args.coresets_dir)
    pattern = f"*_{args.representation}.pt"
    coreset_files = list(coresets_dir.glob(pattern))
    
    if not coreset_files:
        raise ValueError(f"No coreset files matching {pattern} found in {coresets_dir}")
    
    print("="*60)
    print("CALCULATING SELF-COVERAGE")
    print("="*60)
    print(f"Found {len(coreset_files)} coreset files (representation={args.representation})")
    print(f"Using k={args.k}, bandwidth={args.bandwidth} (scale={args.bandwidth_scale}), kernel={args.kernel}")
    print("="*60)
    
    results = []
    
    for i, filepath in enumerate(coreset_files):
        dataset, split = parse_coreset_filename(filepath, args.representation)
        print(f"\n[{i+1}/{len(coreset_files)}] {dataset}_{split}")
        print(f"  Loading: {filepath.name}")
        
        coreset = WeightedCoreset.load(str(filepath))
        centers = coreset.get_centers()
        counts = coreset.get_counts()
        
        # Validate coreset
        if len(centers) == 0:
            print(f"  ⚠️  WARNING: Empty coreset! Skipping...")
            continue
        
        if counts.sum() == 0:
            print(f"  ⚠️  WARNING: Coreset has zero total count! Skipping...")
            continue
        
        print(f"  Centers: {len(centers)}, Total samples: {coreset.total_samples}, "
              f"Counts sum: {counts.sum():.1f}, Mean count: {counts.mean():.2f}")
        
        # Create codebook
        codebook = codebook_from_coreset(coreset)
        
        # Calculate self-coverage (compare against itself)
        print(f"  Computing self-coverage...")
        
        # Debug: Check distances for self-coverage
        import torch
        centroids = codebook.centroids.double()
        if len(centroids) > 1:
            # Sample a few points to check distances
            sample_size = min(10, len(centroids))
            sample_centroids = centroids[:sample_size]
            dists = torch.cdist(sample_centroids, centroids)
            # Get k+1 nearest (including self)
            k_check = min(args.k + 1, len(centroids))
            topk_dists, _ = torch.topk(dists, k=k_check, dim=1, largest=False)
            self_dists = topk_dists[:, 0]  # Should be 0 for self-matches
            other_dists = topk_dists[:, 1:args.k+1] if k_check > 1 else torch.zeros(sample_size, 0)
            mean_other_dists = other_dists.mean() if other_dists.numel() > 0 else torch.tensor(0.0)
            print(f"    Debug: Self-distances (should be ~0): min={self_dists.min():.6f}, max={self_dists.max():.6f}")
            if other_dists.numel() > 0:
                print(f"    Debug: Other neighbor distances: mean={mean_other_dists:.6f}, "
                      f"min={other_dists.min():.6f}, max={other_dists.max():.6f}")
            
            # Check train_scale
            train_pairwise_dists = torch.cdist(centroids[:min(100, len(centroids))], centroids)
            train_pairwise_dists = train_pairwise_dists[train_pairwise_dists > 1e-6]  # Exclude self
            if len(train_pairwise_dists) > 0:
                train_scale = torch.quantile(train_pairwise_dists, 0.5).item()
                print(f"    Debug: Train scale (median pairwise dist): {train_scale:.6f}")
                print(f"    Debug: Mean neighbor dist / train_scale ratio: {mean_other_dists.item() / train_scale:.3f}")
        
        recall = recall_train_covers_eval_soft(
            codebook,
            codebook,
            k=args.k,
            bandwidth=args.bandwidth,
            bandwidth_scale=args.bandwidth_scale,
            M_train=args.M_train,
            kernel=args.kernel,
            batch_size=args.batch_size,
            adaptive_bandwidth=args.adaptive_bandwidth,
            min_bandwidth_quantile=args.min_bandwidth_quantile,
            adaptive_mass=args.adaptive_mass,
            mass_quantile=args.mass_quantile,
            mass_floor=args.mass_floor,
            use_simple=True,
        )
        precision = precision_train_wrt_eval_soft(
            codebook,
            codebook,
            k=args.k,
            bandwidth=args.bandwidth,
            bandwidth_scale=args.bandwidth_scale,
            M_eval=args.M_eval,
            kernel=args.kernel,
            batch_size=args.batch_size,
            adaptive_bandwidth=args.adaptive_bandwidth,
            min_bandwidth_quantile=args.min_bandwidth_quantile,
            adaptive_mass=args.adaptive_mass,
            mass_quantile=args.mass_quantile,
            mass_floor=args.mass_floor,
            use_simple=True,
        )
        outside = 1.0 - precision
        
        print(f"  ✅ Recall: {recall:.2%}, Precision: {precision:.2%}, Outside: {outside:.2%}")
        
        results.append({
            'dataset': dataset,
            'split': split,
            'representation': args.representation,
            'centers': len(centers),
            'total_samples': coreset.total_samples,
            'recall': recall,
            'precision': precision,
            'outside': outside,
        })
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in results:
        print(f"{result['dataset']}_{result['split']:10s} | "
              f"Recall: {result['recall']:6.2%} | "
              f"Precision: {result['precision']:6.2%} | "
              f"Outside: {result['outside']:6.2%}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
