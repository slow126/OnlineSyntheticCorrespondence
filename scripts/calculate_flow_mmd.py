"""
Calculate MMD between flow features from different datasets.

This script:
1. Loads datasets using CorrespondenceDataset
2. Extracts flow_full from batches (dense grid format)
3. Converts flows to [x, y, dx, dy] format
4. Filters out invalid flows (inf/nan and zero flows (0,0))
5. Calculates MMD between datasets using the MMD library
"""

import argparse
import sys
import os
import json
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.datasets.MixedCorrespondenceDataset import MixedCorrespondenceDataset
from src.mmd import load_config_from_yaml, StreamingMMD, StreamingMMDTorch, mmd2_rff


def _is_synthetic_dataset(name: Optional[str]) -> bool:
    return isinstance(name, str) and name.startswith("synthetic")


def _state_dataset_ids(streaming_mmd) -> set[str]:
    return set(getattr(streaming_mmd, "state", {}).keys())


def _state_count(streaming_mmd, dataset_id: str) -> int:
    state = getattr(streaming_mmd, "state", {}).get(dataset_id, {})
    count = state.get("count", 0)
    if hasattr(count, "item"):
        count = count.item()
    return int(count)


def _save_mmd_state(streaming_mmd, state_file: str, counts_file: str, counts: dict[str, int]) -> None:
    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    streaming_mmd.save_state(state_file)
    with open(counts_file, "w") as f:
        json.dump(counts, f, indent=2, sort_keys=True)
    print(f"  Saved MMD state cache: {state_file}")


def _load_counts(counts_file: str) -> dict[str, int]:
    if not os.path.exists(counts_file):
        return {}
    with open(counts_file, "r") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def _dataset_id_from_config(ds_config: dict) -> str:
    split = ds_config["split"]
    is_mixed = ds_config.get("mixed", False) or "datasets" in ds_config
    if is_mixed:
        label = ds_config.get("name")
        if not label:
            datasets_list = ds_config.get("datasets", [])
            percentages = ds_config.get("percentages", [])
            if len(percentages) == 2 and len(datasets_list) == 2:
                pct1 = int(percentages[0] * 100)
                pct2 = int(percentages[1] * 100)
                label = f"{datasets_list[0]}_{datasets_list[1]}_{pct1}_{pct2}"
            else:
                label = "+".join(datasets_list)
    else:
        label = ds_config["name"]
    return f"{label}_{split}"


def _split_dataset_id(dataset_id: str) -> tuple[str, str]:
    return dataset_id.rsplit("_", 1) if "_" in dataset_id else (dataset_id, "unknown")


def _load_done_pairs(results_file: str) -> set[tuple[str, str, str, str]]:
    if not os.path.exists(results_file):
        return set()
    import csv
    done_pairs = set()
    with open(results_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done_pairs.add((row["dataset1"], row["split1"], row["dataset2"], row["split2"]))
    return done_pairs


def _reverse_pair(pair: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    dataset1, split1, dataset2, split2 = pair
    return (dataset2, split2, dataset1, split1)


def _pair_is_done(pair: tuple[str, str, str, str], done_pairs: set[tuple[str, str, str, str]]) -> bool:
    return pair in done_pairs or _reverse_pair(pair) in done_pairs


def _load_required_pairs(required_pairs_file: str | None) -> list[tuple[str, str, str, str]]:
    if not required_pairs_file:
        return []
    path = Path(required_pairs_file)
    if not path.exists():
        print(f"WARNING: required_pairs_file does not exist: {path}; falling back to all configured pairs")
        return []
    import csv
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        if {"train_dataset", "train_split", "eval_dataset", "eval_split"}.issubset(cols):
            pairs = [
                (row["train_dataset"], row["train_split"], row["eval_dataset"], row["eval_split"])
                for row in reader
            ]
        elif {"dataset1", "split1", "dataset2", "split2"}.issubset(cols):
            pairs = [
                (row["dataset1"], row["split1"], row["dataset2"], row["split2"])
                for row in reader
            ]
        else:
            raise ValueError(
                f"{path} must contain either train/eval pair columns or dataset1/split1/dataset2/split2"
            )
    return list(dict.fromkeys(pairs))


def _all_configured_pairs(dataset_ids: list[str]) -> list[tuple[str, str, str, str]]:
    pairs = []
    for i, name1 in enumerate(dataset_ids):
        for name2 in dataset_ids[i + 1:]:
            dataset1_name, split1 = _split_dataset_id(name1)
            dataset2_name, split2 = _split_dataset_id(name2)
            pairs.append((dataset1_name, split1, dataset2_name, split2))
    return pairs


def _required_dataset_ids_for_missing_pairs(
    required_pairs: list[tuple[str, str, str, str]],
    done_pairs: set[tuple[str, str, str, str]],
) -> set[str]:
    required: set[str] = set()
    for pair in required_pairs:
        dataset1_name, split1, dataset2_name, split2 = pair
        if not _pair_is_done(pair, done_pairs):
            required.add(f"{dataset1_name}_{split1}")
            required.add(f"{dataset2_name}_{split2}")
    return required


def extract_flow_vectors(flow_full: torch.Tensor) -> np.ndarray:
    """
    Extract flow vectors as [x, y, dx, dy] from flow_full dense grid tensor.
    
    Args:
        flow_full: [2, H, W] tensor where:
            - flow_full[0] = dx (horizontal flow)
            - flow_full[1] = dy (vertical flow)
            - Invalid flows are marked with inf/nan or (0,0) vectors
    
    Returns:
        [N, 4] array where N is number of valid flows, columns are [x, y, dx, dy]
        - x, y: pixel coordinates (0-indexed)
        - dx, dy: flow displacement values
        - Filters out: inf/nan values and (0,0) flow vectors (invalid/no-flow regions)
    """
    if flow_full is None:
        return np.empty((0, 4), dtype=np.float32)
    
    # flow_full is [2, H, W]
    _, H, W = flow_full.shape
    dx = flow_full[0].cpu().numpy()  # [H, W]
    dy = flow_full[1].cpu().numpy()  # [H, W]
    
    # Create coordinate grid
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    # y_coords: [H, W], x_coords: [H, W]
    
    # Flatten to vectors
    x_flat = x_coords.flatten()  # [H*W]
    y_flat = y_coords.flatten()  # [H*W]
    dx_flat = dx.flatten()  # [H*W]
    dy_flat = dy.flatten()  # [H*W]
    
    # Filter invalid flows (inf/nan and zero flows)
    # Zero flows (0,0) often represent invalid/no-flow regions
    valid_mask = (
        np.isfinite(dx_flat) & 
        np.isfinite(dy_flat) & 
        ~((dx_flat == 0) & (dy_flat == 0))
    )
    
    # Stack to [N, 4] format: [x, y, dx, dy]
    flow_vectors = np.stack([
        x_flat[valid_mask],
        y_flat[valid_mask],
        dx_flat[valid_mask],
        dy_flat[valid_mask]
    ], axis=1).astype(np.float32)
    
    return flow_vectors


def stream_flows_to_mmd(
    dataloader: DataLoader,
    num_batches: int,
    dataset_name: str,
    streaming_mmd,
    backend: str,
    device: torch.device = None
) -> int:
    """
    Stream flow vectors directly to StreamingMMD without accumulating in memory.
    
    Args:
        dataloader: DataLoader for the dataset
        num_batches: Number of batches to process
        dataset_name: Name of dataset (for logging)
        streaming_mmd: StreamingMMD or StreamingMMDTorch instance to update
        backend: 'numpy' or 'torch'
        device: Device for PyTorch tensors (if backend is 'torch')
    
    Returns:
        Total number of flow vectors processed
    """
    batches_processed = 0
    total_vectors = 0
    
    print(f"  Streaming flows from {dataset_name} to MMD...")
    
    for batch_idx, batch in enumerate(dataloader):
        if num_batches is not None and batches_processed >= num_batches:
            break
        
        # Get flow_full from batch
        if 'flow_full' in batch:
            flow_full_batch = batch['flow_full']
        elif 'flow' in batch:
            flow_full_batch = batch['flow']
        else:
            print(f"    Warning: No flow_full or flow in batch {batch_idx}")
            continue
        
        if flow_full_batch is None:
            print(f"    Warning: flow_full is None in batch {batch_idx}")
            continue
        
        # flow_full_batch is [B, 2, H, W]
        batch_size = flow_full_batch.shape[0]
        
        # Process each sample in batch and update StreamingMMD immediately
        for sample_idx in range(batch_size):
            flow_full = flow_full_batch[sample_idx]  # [2, H, W]
            flow_vectors = extract_flow_vectors(flow_full)  # [N, 4]
            
            if len(flow_vectors) > 0:
                # Update StreamingMMD immediately - don't accumulate!
                if backend == 'torch':
                    flows_tensor = torch.from_numpy(flow_vectors).float().to(device)
                    streaming_mmd.update(dataset_name, flows_tensor)
                else:
                    streaming_mmd.update(dataset_name, flow_vectors)
                
                total_vectors += len(flow_vectors)
        
        batches_processed += 1
        
        if (batch_idx + 1) % 5 == 0:
            print(f"    Processed {batch_idx + 1} batches, {total_vectors} flow vectors streamed...")
    
    print(f"    Streamed {total_vectors} flow vectors from {dataset_name}")
    return total_vectors


def create_dataset_from_config(
    dataset_name: str,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    entry_overrides: dict = None
) -> CorrespondenceDataset:
    """
    Create a CorrespondenceDataset from config parameters.
    
    Args:
        dataset_name: Name of dataset
        split: Split to use ('train', 'val', 'test')
        common_params: Common parameters for all datasets
        dataset_overrides: Dataset-specific overrides
    
    Returns:
        CorrespondenceDataset instance
    """
    # Start with common parameters
    dataset_config = common_params.copy()
    
    # Apply dataset-specific overrides
    if dataset_name in dataset_overrides:
        dataset_config.update(dataset_overrides[dataset_name])
    if entry_overrides:
        dataset_config.update(entry_overrides)
    
    # Handle size tuple
    if 'size' in dataset_config and isinstance(dataset_config['size'], list):
        dataset_config['size'] = tuple(dataset_config['size'])
    
    # Set split
    dataset_config['split'] = split
    
    # Handle max_kps: null -> None
    if 'max_kps' in dataset_config and dataset_config['max_kps'] is None:
        dataset_config['max_kps'] = None
    
    # Add dataset-specific parameters based on dataset type
    if dataset_name == 'synthetic':
        # Synthetic-specific params are already in overrides
        pass
    elif dataset_name == 'tss':
        dataset_config['thres'] = dataset_config.get('thres', 'img')
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
    elif dataset_name == 'middlebury':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
    elif dataset_name == 'pointodyssey':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
        dataset_config['thres'] = dataset_config.get('thres', 'img')
    elif dataset_name in ['kitti2012', 'kitti2015']:
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', False)
        dataset_config['thres'] = dataset_config.get('thres', 'img')
        # Handle special case for kitti_val_use_full_training
        kitti_val_use_full_training = dataset_config.get('kitti_val_use_full_training', False)
        if kitti_val_use_full_training and dataset_config.get('split') == 'val':
            dataset_config['split'] = 'training'  # Use full training set for validation
    elif dataset_name == 'flyingthings':
        dataset_config['reverse_flow'] = dataset_config.get('reverse_flow', True)
    # For spair, pfpascal, pfwillow: datapath is already set in overrides
    
    print(f"Creating dataset: {dataset_name} (split: {split})")
    dataset = CorrespondenceDataset(dataset_name, **dataset_config)
    return dataset


def create_mixed_dataset_from_config(
    datasets_list: list,
    percentages: list,
    split: str,
    common_params: dict,
    dataset_overrides: dict,
    epoch_size: int = None,
    seed: int = None
) -> MixedCorrespondenceDataset:
    if len(datasets_list) != len(percentages):
        raise ValueError(f"Number of datasets ({len(datasets_list)}) must match number of percentages ({len(percentages)})")
    created_datasets = []
    for dataset_name in datasets_list:
        ds_config = common_params.copy()
        if dataset_name in dataset_overrides:
            ds_config.update(dataset_overrides[dataset_name])
        if 'size' in ds_config and isinstance(ds_config['size'], list):
            ds_config['size'] = tuple(ds_config['size'])
        ds_config['split'] = split
        if 'max_kps' in ds_config and ds_config['max_kps'] is None:
            ds_config['max_kps'] = None
        if dataset_name == 'tss':
            ds_config['thres'] = ds_config.get('thres', 'img')
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', False)
        elif dataset_name == 'middlebury':
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', False)
        elif dataset_name == 'pointodyssey':
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', True)
            ds_config['thres'] = ds_config.get('thres', 'img')
        elif dataset_name in ['kitti2012', 'kitti2015']:
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', False)
            ds_config['thres'] = ds_config.get('thres', 'img')
            kitti_val_use_full_training = ds_config.get('kitti_val_use_full_training', False)
            if kitti_val_use_full_training and ds_config.get('split') == 'val':
                ds_config['split'] = 'training'
        elif dataset_name == 'flyingthings':
            ds_config['reverse_flow'] = ds_config.get('reverse_flow', True)
        print(f"Creating sub-dataset: {dataset_name} (split: {split})")
        sub_dataset = CorrespondenceDataset(dataset_name, **ds_config)
        created_datasets.append(sub_dataset)
    print(f"Creating mixed dataset with {len(created_datasets)} datasets")
    mixed_dataset = MixedCorrespondenceDataset(
        datasets=created_datasets,
        percentages=percentages,
        epoch_size=epoch_size,
        seed=seed,
    )
    return mixed_dataset


def main():
    parser = argparse.ArgumentParser(description='Calculate MMD between flow features from datasets')
    parser.add_argument('--config', type=str, 
                       default='src/configs/mmd_configs/flow_mmd_config.yaml',
                       help='Path to flow MMD config YAML file')
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config sections
    datasets_config = config['datasets']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    mmd_preset = config['mmd_preset']
    common_params = config['dataset_params']
    dataset_overrides = config.get('dataset_overrides', {})
    sampling_cfg = config.get('sampling', {})
    batch_limit_default = sampling_cfg.get('batch_limit', 500)
    shuffle_default = bool(sampling_cfg.get('shuffle', True))
    output_config = config.get('output', {})
    
    # Load MMD config
    mmd_config_path = 'src/configs/mmd_configs/mmd_config.yaml'
    print(f"Loading MMD config preset: {mmd_preset}")
    mmd_config = load_config_from_yaml(mmd_config_path, preset=mmd_preset)
    
    # Verify input_dim matches flow vector dimension (4: x, y, dx, dy)
    # Only warn if using a preset that's not designed for flow vectors
    if mmd_config.input_dim != 4:
        if mmd_preset != 'flow_vectors':
            print(f"Warning: MMD config input_dim={mmd_config.input_dim}, but flow vectors are 4D [x, y, dx, dy]")
            print(f"Updating MMD config input_dim to 4")
        mmd_config.input_dim = 4
    
    # Create RFF map and StreamingMMD BEFORE processing datasets
    print("\n" + "="*60)
    print("INITIALIZING MMD COMPUTATION")
    print("="*60)
    
    rff_map = mmd_config.create_rff_map()
    
    # Create streaming MMD instance
    if mmd_config.backend == 'torch':
        from src.mmd import StreamingMMDTorch
        streaming_mmd = StreamingMMDTorch(rff_map)
        device = rff_map.device
    else:
        streaming_mmd = StreamingMMD(config=mmd_config)
        device = None
    
    print(f"Created StreamingMMD with backend: {mmd_config.backend}")
    if device is not None:
        print(f"Using device: {device}")

    results_file = output_config.get('results_file', 'flow_mmd_results.csv')
    save_results = bool(output_config.get('save_results', False))
    done_pairs = _load_done_pairs(results_file) if save_results else set()
    if done_pairs:
        print(f"Resuming MMD CSV: {len(done_pairs)} pairs already in {results_file}")
    planned_dataset_ids = [_dataset_id_from_config(ds_config) for ds_config in datasets_config]
    required_pairs_file = output_config.get("required_pairs_file")
    required_pairs = _load_required_pairs(required_pairs_file)
    if required_pairs:
        print(f"Restricting MMD pair computation to {len(required_pairs)} required pairs from {required_pairs_file}")
    else:
        required_pairs = _all_configured_pairs(planned_dataset_ids)
    required_dataset_ids = _required_dataset_ids_for_missing_pairs(required_pairs, done_pairs)
    if save_results and done_pairs:
        print(
            f"Missing pair endpoints to stream: {len(required_dataset_ids)}/"
            f"{len(planned_dataset_ids)} configured datasets"
        )

    state_file = output_config.get(
        'state_file',
        str(Path(results_file).with_suffix('.state.pt' if mmd_config.backend == 'torch' else '.state.npz')),
    )
    counts_file = output_config.get(
        'counts_file',
        str(Path(results_file).with_suffix('.counts.json')),
    )
    save_state = bool(output_config.get('save_state', True))
    if os.path.exists(state_file):
        print(f"Loading cached MMD streaming state: {state_file}")
        streaming_mmd.load_state(state_file)
    dataset_vector_counts = _load_counts(counts_file)
    for ds_id in _state_dataset_ids(streaming_mmd):
        dataset_vector_counts.setdefault(ds_id, _state_count(streaming_mmd, ds_id))
    if dataset_vector_counts:
        print(f"Cached MMD datasets: {len(dataset_vector_counts)}")
    
    # Process datasets and stream flows directly to MMD (no accumulation!)
    print("\n" + "="*60)
    print("STREAMING FLOWS TO MMD (NO MEMORY ACCUMULATION)")
    print("="*60)

    for ds_config in datasets_config:
        is_mixed = ds_config.get('mixed', False) or 'datasets' in ds_config
        split = ds_config['split']
        num_batches = ds_config.get('num_batches', batch_limit_default)
        entry_overrides = ds_config.get('overrides', None)
        dataset_id = _dataset_id_from_config(ds_config)
        if save_results and done_pairs and dataset_id not in required_dataset_ids:
            print(f"Skipping {dataset_id}: all pairwise MMD rows involving this dataset are cached")
            continue

        if is_mixed:
            datasets_list = ds_config.get('datasets', [])
            percentages = ds_config.get('percentages', [])
            label = ds_config.get('name')
            if not label:
                if len(percentages) == 2 and len(datasets_list) == 2:
                    pct1 = int(percentages[0] * 100)
                    pct2 = int(percentages[1] * 100)
                    label = f"{datasets_list[0]}_{datasets_list[1]}_{pct1}_{pct2}"
                else:
                    label = "+".join(datasets_list)
            if dataset_id in _state_dataset_ids(streaming_mmd):
                dataset_vector_counts[dataset_id] = _state_count(streaming_mmd, dataset_id)
                print(f"Skipping {dataset_id}: cached MMD state with {dataset_vector_counts[dataset_id]} vectors")
                continue
            dataset = create_mixed_dataset_from_config(
                datasets_list,
                percentages,
                split,
                common_params,
                dataset_overrides,
                epoch_size=ds_config.get('epoch_size', None),
                seed=ds_config.get('seed', None),
            )
            has_synthetic = any(_is_synthetic_dataset(ds_name) for ds_name in datasets_list)
            workers = 0 if has_synthetic else num_workers
        else:
            label = ds_config['name']
            dataset_name = ds_config.get('dataset_name', label)
            if dataset_id in _state_dataset_ids(streaming_mmd):
                dataset_vector_counts[dataset_id] = _state_count(streaming_mmd, dataset_id)
                print(f"Skipping {dataset_id}: cached MMD state with {dataset_vector_counts[dataset_id]} vectors")
                continue
            dataset = create_dataset_from_config(
                dataset_name, split, common_params, dataset_overrides, entry_overrides
            )
            workers = 0 if _is_synthetic_dataset(dataset_name) else num_workers
        
        shuffle = bool(ds_config.get('shuffle', shuffle_default))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            pin_memory=False
        )
        
        # Stream flows directly to StreamingMMD (no accumulation!)
        # Use dataset_id instead of dataset_name to treat splits as unique
        vector_count = stream_flows_to_mmd(
            dataloader, num_batches, dataset_id,
            streaming_mmd, mmd_config.backend, device
        )
        dataset_vector_counts[dataset_id] = vector_count
        if save_state:
            _save_mmd_state(streaming_mmd, state_file, counts_file, dataset_vector_counts)
    
    # Now calculate pairwise MMD (all data already streamed)
    print("\n" + "="*60)
    print("CALCULATING MMD BETWEEN DATASETS")
    print("="*60)
    
    # Calculate pairwise MMD and checkpoint each pair as soon as it is computed.
    if save_results and not os.path.exists(results_file):
        Path(results_file).parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset1', 'split1', 'dataset2', 'split2', 'mmd2', 'mmd', 'num_vectors1', 'num_vectors2'])

    print("\nPairwise MMD² results:")
    print("-" * 60)

    for pair_key in required_pairs:
        dataset1_name, split1, dataset2_name, split2 = pair_key
        name1 = f"{dataset1_name}_{split1}"
        name2 = f"{dataset2_name}_{split2}"
        if _pair_is_done(pair_key, done_pairs):
            print(f"  {name1:15} vs {name2:15}: SKIPPED (cached)")
            continue
        if dataset_vector_counts.get(name1, 0) == 0 or dataset_vector_counts.get(name2, 0) == 0:
            print(f"  {name1} vs {name2}: SKIPPED (missing streamed vectors)")
            continue

        mmd2_val = streaming_mmd.mmd2(name1, name2)
        mmd_val = streaming_mmd.mmd(name1, name2)

        print(f"  {name1:15} vs {name2:15}: MMD² = {mmd2_val:.6f}, MMD = {mmd_val:.6f}")

        if save_results:
            import csv
            with open(results_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dataset1_name,
                    split1,
                    dataset2_name,
                    split2,
                    mmd2_val,
                    mmd_val,
                    dataset_vector_counts[name1],
                    dataset_vector_counts[name2]
                ])
            done_pairs.add(pair_key)
            print(f"    Checkpoint saved to {results_file}")

    if save_results:
        print(f"\nResults saved to: {results_file}")
    
    print("="*60)
    print("Done!")


if __name__ == "__main__":
    main()
