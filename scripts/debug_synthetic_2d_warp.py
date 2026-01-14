#!/usr/bin/env python3
"""
Quick sanity visualization for the synthetic_2d_warp dataset.
Outputs debug images under debug/{dataset_name}/train/.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train_cats_unified import load_config, create_training_dataset, visualize_batch_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize synthetic_2d_warp batch")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/CorrespondenceConfigs/synthetic_2d_warps.yaml",
        help="Path to a CorrespondenceConfigs YAML file",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feature-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = create_training_dataset(config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    batch = next(iter(loader))
    dataset_name = config["dataset"].get("dataset_name", "synthetic_2d_warp")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Downsampled flow visualization (default path)
    visualize_batch_flow(
        model=None,
        batch=batch,
        device=device,
        train_dataset_name=dataset_name,
        val_dataset_name=None,
        split_name="train",
        flow_source="gt",
        feature_size=args.feature_size,
        epoch=None,
    )

    # Full-resolution flow visualization (force flow/full flow path)
    batch_full = dict(batch)
    batch_full.pop("flow_downsampled", None)
    visualize_batch_flow(
        model=None,
        batch=batch_full,
        device=device,
        train_dataset_name=dataset_name,
        val_dataset_name=None,
        split_name="train",
        flow_source="gt",
        feature_size=args.feature_size,
        epoch=None,
    )


if __name__ == "__main__":
    main()
