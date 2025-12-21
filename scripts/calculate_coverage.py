"""
Calculate pairwise soft k-NN precision/recall metrics between coreset codebooks.

This script computes asymmetric recall/precision/outside metrics for all pairs
of precomputed coresets using the soft k-NN codebook formulation.

Usage:
    python scripts/calculate_coverage.py --coresets-dir coresets/ --representation flow --output coverage_results.csv
"""

import argparse
import csv
from pathlib import Path
import numpy as np


# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coreset import (
    WeightedCoreset,
    codebook_from_coreset,
    recall_train_covers_eval_soft,
    precision_train_wrt_eval_soft,
)


def _effective_num_clusters(counts: np.ndarray, eps: float = 1e-12) -> float:
    """
    Compute effective number of clusters (exp of entropy of normalized counts).
    Higher = mass spread across many clusters; lower = few dominant clusters.
    """
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    entropy = -np.sum(p * np.log(p + eps))
    return float(np.exp(entropy))


def parse_coreset_filename(filepath, suffix: str):
    """
    Parse coreset filename to extract dataset name and split.
    
    Expected format: {dataset}_{split}_{suffix}.pt
    Examples:
        - synthetic_train_flow.pt -> (synthetic, train)
        - spair_synthetic_50_50_train_flow.pt -> (spair_synthetic_50_50, train)
        - spair_synthetic_70_30_train_flow.pt -> (spair_synthetic_70_30, train)
    
    Supports mixed dataset names with underscores (e.g., spair_synthetic_50_50).
    The parser splits by the last underscore to separate dataset from split.
    """
    stem = Path(filepath).stem  # Remove .pt extension
    
    # Remove suffix if present (e.g., "_flow" or "_resnet")
    suffix_token = f"_{suffix}"
    if stem.endswith(suffix_token):
        stem = stem[: -len(suffix_token)]
    
    # Split by last underscore to separate dataset from split
    # This works for both single datasets (e.g., "synthetic_train") 
    # and mixed datasets (e.g., "spair_synthetic_50_50_train")
    parts = stem.rsplit('_', 1)
    if len(parts) == 2:
        dataset, split = parts
        return dataset, split
    else:
        # Fallback: treat entire name as dataset, split unknown
        return stem, 'unknown'


def load_coreset_files(coresets_dir, representation: str):
    """
    Load all coreset files from directory.
    
    Returns:
        list of dict with keys: 'path', 'dataset', 'split', 'coreset'
    """
    coresets_dir = Path(coresets_dir)
    pattern = f"*_{representation}.pt"
    coreset_files = list(coresets_dir.glob(pattern))
    
    if not coreset_files:
        raise ValueError(f"No coreset files matching {pattern} found in {coresets_dir}")
    
    print(f"Found {len(coreset_files)} coreset files (representation={representation})")
    
    coresets = []
    for filepath in coreset_files:
        dataset, split = parse_coreset_filename(filepath, representation)
        print(f"  Loading: {filepath.name} -> dataset={dataset}, split={split}")
        
        coreset = WeightedCoreset.load(str(filepath))
        centers = coreset.get_centers()
        counts = coreset.get_counts()
        
        # Validate coreset
        if len(centers) == 0:
            print(f"    ⚠️  WARNING: Empty coreset! Skipping...")
            continue
        
        if counts.sum() == 0:
            print(f"    ⚠️  WARNING: Coreset has zero total count! Skipping...")
            continue

        # Diagnostic: effective number of clusters (exp of entropy of weights)
        n_eff = _effective_num_clusters(counts)
        print(f"    N_eff: {n_eff:.1f} (of {len(centers)} centers)")
        
        coresets.append({
            'path': str(filepath),
            'filename': filepath.name,
            'dataset': dataset,
            'split': split,
            'representation': representation,
            'coreset': coreset,
            'codebook': codebook_from_coreset(coreset),
        })
        
        print(f"    Centers: {len(centers)}, Total samples: {coreset.total_samples}, "
              f"Counts sum: {counts.sum():.1f}, Mean count: {counts.mean():.2f}")
    
    return coresets


def compute_pairwise_metrics(
    coresets,
    k: int,
    bandwidth: float,
    bandwidth_scale: float,
    M_train: float,
    M_eval: float,
    kernel: str,
    batch_size: int,
    adaptive_bandwidth: bool,
    min_bandwidth_quantile: float,
    adaptive_mass: bool,
    mass_quantile: float,
    mass_floor: float,
    emit_direction: str,
):
    """
    Compute soft k-NN recall/precision/outside metrics for all pairs of coresets.
    """
    results = []
    
    # Compute for all pairs (including self-pairs for reference)
    total_pairs = len(coresets) * len(coresets)
    
    print(f"\nComputing coverage for {total_pairs} pairs...")
    print(f"Using k={k}, bandwidth={bandwidth} (scale={bandwidth_scale}), kernel={kernel}")
    print("="*60)
    
    train_splits = {"train", "training"}
    eval_splits = {"val", "test", "validation"}

    for i, train_info in enumerate(coresets):
        for j, eval_info in enumerate(coresets):
            # Directional filtering to avoid using reversed pairs
            if emit_direction == "train_to_eval":
                if train_info["split"] not in train_splits or eval_info["split"] not in eval_splits:
                    continue
            elif emit_direction == "eval_to_train":
                if train_info["split"] not in eval_splits or eval_info["split"] not in train_splits:
                    continue
            train_cb = train_info['codebook']
            eval_cb = eval_info['codebook']
            
            print(f"[{i*len(coresets) + j + 1}/{total_pairs}] "
                  f"{train_info['dataset']}_{train_info['split']} -> "
                  f"{eval_info['dataset']}_{eval_info['split']}")
            
            # Debug: Print coreset sizes
            train_coreset = train_info['coreset']
            eval_coreset = eval_info['coreset']
            train_centers = train_coreset.get_centers()
            eval_centers = eval_coreset.get_centers()
            train_counts = train_coreset.get_counts()
            eval_counts = eval_coreset.get_counts()
            
            print(f"  Train coreset: {len(train_centers)} centers, {train_coreset.total_samples} total samples, "
                  f"counts sum={train_counts.sum():.1f}, mean={train_counts.mean():.2f}")
            print(f"  Eval coreset: {len(eval_centers)} centers, {eval_coreset.total_samples} total samples, "
                  f"counts sum={eval_counts.sum():.1f}, mean={eval_counts.mean():.2f}")
            
            # Use simple metric by default (more robust to dataset size and hyperparameters)
            # Set use_simple=False to use the original complex metric
            recall = recall_train_covers_eval_soft(
                train_cb,
                eval_cb,
                k=k,
                bandwidth=bandwidth,
                bandwidth_scale=bandwidth_scale,
                M_train=M_train,
                kernel=kernel,
                batch_size=batch_size,
                adaptive_bandwidth=adaptive_bandwidth,
                min_bandwidth_quantile=min_bandwidth_quantile,
                adaptive_mass=adaptive_mass,
                mass_quantile=mass_quantile,
                mass_floor=mass_floor,
                use_simple=True,  # Use simpler, more robust metric
            )
            precision = precision_train_wrt_eval_soft(
                train_cb,
                eval_cb,
                k=k,
                bandwidth=bandwidth,
                bandwidth_scale=bandwidth_scale,
                M_eval=M_eval,
                kernel=kernel,
                batch_size=batch_size,
                adaptive_bandwidth=adaptive_bandwidth,
                min_bandwidth_quantile=min_bandwidth_quantile,
                adaptive_mass=adaptive_mass,
                mass_quantile=mass_quantile,
                mass_floor=mass_floor,
                use_simple=True,  # Use simpler, more robust metric
            )
            outside = 1.0 - precision
            
            result = {
                'dataset1': train_info['dataset'],
                'split1': train_info['split'],
                'dataset2': eval_info['dataset'],
                'split2': eval_info['split'],
                'representation': train_info.get('representation', ''),
                'k': k,
                'bandwidth': bandwidth,
                'bandwidth_scale': bandwidth_scale,
                'kernel': kernel,
                'M_train': M_train,
                'M_eval': M_eval,
                'recall': recall,
                'precision': precision,
                'outside': outside,
            }
            
            results.append(result)
            
            print(f"  Recall: {recall:.2%}, Precision: {precision:.2%}, Outside: {outside:.2%}")
            
            # Additional debug for zero recall
            # Check if dataset name contains 'spair' (handles both single and mixed datasets)
            dataset_lower = train_info['dataset'].lower()
            if recall < 0.01 and 'spair' in dataset_lower:
                print(f"  ⚠️  WARNING: Very low recall for {train_info['dataset']}! This might indicate:")
                print(f"     - Coreset might be too small or empty")
                print(f"     - Distances between training and eval datasets might be very large")
                print(f"     - Bandwidth might be too small (try increasing --bandwidth-scale)")
                print(f"     - Adaptive mass might be setting M_train too high")
    
    return results


def save_results_to_csv(results, output_file):
    """Save results to CSV file."""
    if not results:
        print("No results to save!")
        return
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = [
        'dataset1', 'split1', 'dataset2', 'split2', 'representation',
        'k', 'bandwidth', 'bandwidth_scale', 'kernel', 'M_train', 'M_eval',
        'recall', 'precision', 'outside'
    ]
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nSaved {len(results)} results to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate pairwise soft k-NN coverage metrics between coresets'
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
    parser.add_argument(
        '--emit-direction',
        type=str,
        default='train_to_eval',
        choices=['train_to_eval', 'eval_to_train', 'both'],
        help='Which pair directions to emit into the CSV. Default: train_to_eval'
    )
    args = parser.parse_args()
    
    # Load coresets
    print("="*60)
    print("LOADING CORESETS")
    print("="*60)
    coresets = load_coreset_files(args.coresets_dir, args.representation)
    
    # Compute pairwise coverage
    results = compute_pairwise_metrics(
        coresets,
        k=args.k,
        bandwidth=args.bandwidth,
        bandwidth_scale=args.bandwidth_scale,
        M_train=args.M_train,
        M_eval=args.M_eval,
        kernel=args.kernel,
        batch_size=args.batch_size,
        adaptive_bandwidth=args.adaptive_bandwidth,
        min_bandwidth_quantile=args.min_bandwidth_quantile,
        adaptive_mass=args.adaptive_mass,
        mass_quantile=args.mass_quantile,
        mass_floor=args.mass_floor,
        emit_direction=args.emit_direction,
    )
    
    # Save results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    save_results_to_csv(results, args.output)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
