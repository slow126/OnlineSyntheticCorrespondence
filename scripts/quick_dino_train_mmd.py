#!/usr/bin/env python3
"""
Quick DINO MMD sanity check for training datasets.

Loads DINO features for each train dataset in a config and reports MMD vs a baseline
dataset (default: synthetic). This is intentionally minimal and intended for quick checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.mmd import load_config_from_yaml, StreamingMMD, StreamingMMDTorch, DinoV3Encoder


def _extract_features_from_batch(batch: dict, encoder: DinoV3Encoder) -> torch.Tensor:
    if "src_img" in batch:
        img = batch["src_img"]
    elif "source" in batch:
        img = batch["source"]
    elif "image0" in batch:
        img = batch["image0"]
    else:
        raise ValueError(f"Could not find source image in batch. Available keys: {batch.keys()}")

    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    return encoder.extract_features(img)


def _create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    entry_overrides: Optional[dict] = None,
) -> CorrespondenceDataset:
    dataset_config = common_params.copy()
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    if entry_overrides:
        dataset_config.update(entry_overrides)

    if "size" in dataset_config and isinstance(dataset_config["size"], list):
        dataset_config["size"] = tuple(dataset_config["size"])
    dataset_config["split"] = split

    if "max_kps" in dataset_config and dataset_config["max_kps"] is None:
        dataset_config["max_kps"] = None

    if dataset_name == "tss":
        dataset_config["thres"] = dataset_config.get("thres", "img")
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
    elif dataset_name == "middlebury":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
    elif dataset_name == "pointodyssey":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", True)
        dataset_config["thres"] = dataset_config.get("thres", "img")
    elif dataset_name in ["kitti2012", "kitti2015"]:
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", False)
        dataset_config["thres"] = dataset_config.get("thres", "img")
        if dataset_config.get("kitti_val_use_full_training", False) and dataset_config.get("split") == "val":
            dataset_config["split"] = "training"
    elif dataset_name == "flyingthings":
        dataset_config["reverse_flow"] = dataset_config.get("reverse_flow", True)

    return CorrespondenceDataset(dataset_name, **dataset_config)


def _stream_features_to_mmd(
    dataloader: DataLoader,
    num_batches: int,
    dataset_id: str,
    encoder: DinoV3Encoder,
    streaming_mmd,
    backend: str,
    device: Optional[torch.device],
) -> int:
    processed = 0
    total_vectors = 0
    for batch_idx, batch in enumerate(dataloader):
        if processed >= num_batches:
            break
        try:
            with torch.no_grad():
                features = _extract_features_from_batch(batch, encoder)
            if features.shape[0] == 0:
                processed += 1
                continue
            if features.dtype != torch.float32:
                features = features.float()
            if backend == "torch":
                features = features.to(device)
                streaming_mmd.update(dataset_id, features)
            else:
                features_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
                streaming_mmd.update(dataset_id, features_np)
            total_vectors += features.shape[0]
        except Exception as exc:
            print(f"Warning: error in batch {batch_idx} ({dataset_id}): {exc}")
        processed += 1
    return total_vectors


def _load_train_entries(config: dict) -> List[dict]:
    entries = []
    for ds_config in config["datasets"]:
        if ds_config.get("split") != "train":
            continue
        if ds_config.get("mixed", False) or "datasets" in ds_config:
            continue
        entries.append(ds_config)
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick DINO MMD sanity check (train datasets only).")
    parser.add_argument(
        "--config",
        default="src/configs/mmd_configs/feature_mmd_config_dino_fast.yaml",
        help="Feature MMD config with dataset params.",
    )
    parser.add_argument("--baseline", default="synthetic", help="Baseline dataset name.")
    parser.add_argument(
        "--num-batches",
        type=int,
        default=None,
        help="Override num_batches for each dataset (quick mode).",
    )
    parser.add_argument(
        "--output",
        default="analysis/comprehensive/dino_mmd_quick.csv",
        help="CSV output path.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    datasets_config = _load_train_entries(config)
    if not datasets_config:
        raise SystemExit("No train datasets found in config.")

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]
    common_params = config["dataset_params"]
    dataset_overrides = config.get("dataset_overrides", {})
    mmd_preset = config.get("mmd_preset", "dino_features")

    mmd_config = load_config_from_yaml("src/configs/mmd_configs/mmd_config.yaml", preset=mmd_preset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DinoV3Encoder(device=device)

    if mmd_config.input_dim != encoder.feature_dim:
        mmd_config.input_dim = encoder.feature_dim

    if mmd_config.backend == "torch":
        streaming_mmd = StreamingMMDTorch(mmd_config.create_rff_map())
        device = streaming_mmd.rff.device
    else:
        streaming_mmd = StreamingMMD(config=mmd_config)
        device = None

    vector_counts: Dict[str, int] = {}
    id_to_name: Dict[str, str] = {}

    for ds_config in datasets_config:
        label = ds_config["name"]
        dataset_name = ds_config.get("dataset_name", label)
        entry_overrides = ds_config.get("overrides", None)
        num_batches = args.num_batches if args.num_batches is not None else ds_config["num_batches"]

        dataset_id = f"{label}_train"
        id_to_name[dataset_id] = label

        dataset = _create_dataset_from_config(
            dataset_name, "train", common_params, dataset_overrides, entry_overrides
        )
        workers = 0 if dataset_name == "synthetic" else num_workers
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            pin_memory=False,
        )
        print(f"Streaming {dataset_id} (num_batches={num_batches})...")
        vector_counts[dataset_id] = _stream_features_to_mmd(
            dataloader, num_batches, dataset_id, encoder, streaming_mmd, mmd_config.backend, device
        )

    baseline_id = f"{args.baseline}_train"
    if baseline_id not in vector_counts:
        raise SystemExit(f"Baseline '{args.baseline}' not found in train datasets.")

    rows = []
    print("\nMMD vs baseline:")
    for dataset_id, count in vector_counts.items():
        if dataset_id == baseline_id:
            continue
        if count == 0 or vector_counts[baseline_id] == 0:
            print(f"  {dataset_id}: skipped (no vectors)")
            continue
        mmd2_val = streaming_mmd.mmd2(baseline_id, dataset_id)
        mmd_val = streaming_mmd.mmd(baseline_id, dataset_id)
        print(f"  {args.baseline} vs {id_to_name[dataset_id]}: MMD={mmd_val:.6f} (MMD2={mmd2_val:.6f})")
        rows.append(
            {
                "baseline": args.baseline,
                "dataset": id_to_name[dataset_id],
                "mmd2": mmd2_val,
                "mmd": mmd_val,
                "baseline_vectors": vector_counts[baseline_id],
                "dataset_vectors": count,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
