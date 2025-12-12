"""
Evaluate soft k-NN recall/precision metrics between train and eval coresets.

Usage:
    python scripts/eval_coverage.py \
        --train coresets/synthetic_train_flow.pt \
        --eval coresets/kitti_val_flow.pt \
        --output coverage_results.json \
        --k 5 --bandwidth-scale 1.0 --M-train 100 --M-eval 20
"""

import argparse
import json
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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate soft k-NN coverage between train and eval coresets'
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

    train_cb = codebook_from_coreset(train_coreset)
    eval_cb = codebook_from_coreset(eval_coreset)

    # Compute metrics
    print("\n" + "="*60)
    print("COMPUTING COVERAGE METRICS")
    print("="*60)

    recall = recall_train_covers_eval_soft(
        train_cb,
        eval_cb,
        k=args.k,
        bandwidth=args.bandwidth,
        bandwidth_scale=args.bandwidth_scale,
        M_train=args.M_train,
        kernel=args.kernel,
        batch_size=args.batch_size,
    )
    precision = precision_train_wrt_eval_soft(
        train_cb,
        eval_cb,
        k=args.k,
        bandwidth=args.bandwidth,
        bandwidth_scale=args.bandwidth_scale,
        M_eval=args.M_eval,
        kernel=args.kernel,
        batch_size=args.batch_size,
    )
    outside = 1.0 - precision

    if args.verbose:
        print(f"  Recall (train → eval):    {recall:.2%}")
        print(f"  Precision (train w.r.t.): {precision:.2%}")
        print(f"  Outside mass:             {outside:.2%}")

    all_results = {
        'train_file': args.train,
        'eval_file': args.eval,
        'train_centers': len(train_cb.centroids),
        'eval_centers': len(eval_cb.centroids),
        'train_total_samples': int(train_coreset.total_samples),
        'eval_total_samples': int(eval_coreset.total_samples),
        'k': args.k,
        'bandwidth': args.bandwidth,
        'bandwidth_scale': args.bandwidth_scale,
        'M_train': args.M_train,
        'M_eval': args.M_eval,
        'kernel': args.kernel,
        'metrics': {
            'recall': recall,
            'precision': precision,
            'outside': outside,
        },
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
    print(f"Recall: {recall:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Outside: {outside:.2%}")


if __name__ == "__main__":
    main()
