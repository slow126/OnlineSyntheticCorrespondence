"""
Quick sanity script to exercise the refactored CorrespondenceDataset, collate pipeline,
and visualizers. It uses the same config file as training and saves a couple of
flow visualizations for each dataset.
"""

import argparse
from train_cats_unified import load_config, inspect_datasets


def main():
    parser = argparse.ArgumentParser(description="Test CorrespondenceDataset and collate pipeline")
    parser.add_argument("--config", required=True, help="Path to training YAML config")
    parser.add_argument("--output-dir", default="debug_collate", help="Where to save visualizations")
    parser.add_argument("--no-visuals", action="store_true", help="Skip saving visualizations")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional list of dataset names to inspect (defaults to train + eval benchmarks from config)",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for inspection")
    args = parser.parse_args()

    config = load_config(args.config)

    # If datasets list is not provided, inspect training dataset and all eval benchmarks
    datasets_to_check = args.datasets
    inspect_datasets(
        config,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        datasets_to_check=datasets_to_check,
        save_visuals=not args.no_visuals,
    )


if __name__ == "__main__":
    main()
