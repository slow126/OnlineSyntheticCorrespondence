#!/usr/bin/env python3
"""
Batch UMAP analysis for all datasets with hybrid visualization approach.
- Extracts DinoV3 features if needed (using config-based dataset creation)
- Only extracts from src_img to avoid double counting
- Distinguishes train/eval versions with suffixes (e.g., spair_train, spair_eval)
- Option A: Unsupervised UMAP colored by dataset name (primary)
- Option B: Same UMAP colored by class labels (if available) or dataset name (optional)
- Equal sampling per dataset for fair comparison
- Uses GPU-accelerated cuML UMAP when available
- Computes pairwise metrics (11x11 matrix) with asymmetric coverage metrics
"""

import argparse
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm

# Add project root to path
import sys
project_root = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(project_root))

# Import DinoV3
from models.DinoV3.DinoV3 import DinoV3
from torch.utils.data import DataLoader

# Import dataset creation functions (same as run_vector_analysis.py)
from src.fingerprints.vector_representations.generate_vectors import (
    create_training_dataset_from_exp_config,
    create_eval_dataset_from_exp_config,
    build_geometry_config_overrides,
)
from slurm.generate_jobs import load_machine_config

# Import functions from dino_umap.py
sys.path.insert(0, str(Path(__file__).parent))
from dino_umap import (
    load_extracted_features, calculate_umap_parameters,
    run_umap_analysis, USE_CUML, UMAP
)

# Dataset lists with train/eval suffixes
TRAIN_DATASETS = ['synthetic', 'spair_train', 'flyingthings_train', 'pointodyssey_train']
EVAL_DATASETS = ['kitti2012', 'kitti2015', 'pointodyssey_eval', 'tss', 'pfpascal', 'pfwillow', 'spair_eval', 'flyingthings_eval']
ALL_DATASETS = list(set(TRAIN_DATASETS + EVAL_DATASETS))

# Base dataset names (without suffixes)
BASE_TRAIN_DATASETS = ['synthetic', 'spair', 'flyingthings', 'pointodyssey']
BASE_EVAL_DATASETS = ['kitti2012', 'kitti2015', 'pointodyssey', 'tss', 'pfpascal', 'pfwillow', 'spair', 'flyingthings']


def extract_features_from_batch(dino_model, batch, device='cuda', use_src_only=True):
    """
    Extract spatial features from a batch of images using DinoV3.
    Only extracts from src_img to avoid double counting.
    """
    features = {}
    
    # Only extract from src_img to avoid double counting
    key = 'src_img' if use_src_only else 'trg_img'
    if key in batch:
        images = batch[key]
        if images.device != device:
            images = images.to(device)
        spatial_features = dino_model.get_spatial_features(images)
        features[key] = spatial_features.cpu()
    
    return features


def save_features(features_dict, save_path, batch_idx, metadata=None):
    """Save extracted features to disk."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    features_file = save_path / f"features_batch_{batch_idx:04d}.pkl"
    with open(features_file, 'wb') as f:
        pickle.dump(features_dict, f)
    
    if metadata is not None:
        metadata_file = save_path / f"metadata_batch_{batch_idx:04d}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def get_base_dataset_name(dataset_name):
    """Extract base dataset name from suffixed name."""
    if dataset_name.endswith('_train') or dataset_name.endswith('_eval'):
        return dataset_name.rsplit('_', 1)[0]
    return dataset_name


def create_simple_exp_config(dataset_name, machine_config, split='train'):
    """
    Create a simple experiment config for feature extraction.
    Since we're just extracting features, the specific training parameters don't matter.
    """
    # Get base dataset name
    base_name = get_base_dataset_name(dataset_name)
    
    # Get dataset paths from machine config
    datasets = machine_config.get('datasets', {})
    
    exp_config = {
        'train_dataset': base_name if base_name in BASE_TRAIN_DATASETS else None,
        'split': split,
        'normalize': False,  # Disable normalization for UMAP analysis
    }
    
    # Add dataset-specific paths from machine config (no hardcoded defaults)
    if base_name == 'synthetic':
        exp_config['geometry_config_path'] = 'src/configs/online_synth_configs/OnlineGeometryConfig_UMAP.yaml'
        exp_config['processor_config_path'] = 'src/configs/online_synth_configs/OnlineProcessorConfig_UMAP.yaml'
        exp_config['size'] = 256
    elif base_name == 'flyingthings':
        exp_config['flyingthings_root'] = datasets.get('flyingthings_root')
        exp_config['size'] = 256
        exp_config['feature_size'] = 32
    elif base_name == 'pointodyssey':
        exp_config['pointodyssey_root'] = datasets.get('pointodyssey_root')
        exp_config['size'] = 256
        exp_config['feature_size'] = 32
        exp_config['num_pts_to_track_pointodyssey'] = 32
    elif base_name in ['kitti2012', 'kitti2015']:
        exp_config['kitti_root'] = datasets.get('kitti_root')
        exp_config['size'] = 256
        exp_config['feature_size'] = 32
    elif base_name == 'tss':
        exp_config['tss_root'] = datasets.get('tss_root')
        exp_config['size'] = 256
        exp_config['feature_size'] = 32
    elif base_name in ['spair', 'pfpascal', 'pfwillow']:
        exp_config['datapath'] = datasets.get('datapath')
        exp_config['thres'] = 'img'
        exp_config['feature_size'] = 32
        exp_config['size'] = 256
    
    # Validate that required paths are present
    required_keys = {
        'flyingthings': ['flyingthings_root'],
        'pointodyssey': ['pointodyssey_root'],
        'kitti2012': ['kitti_root'],
        'kitti2015': ['kitti_root'],
        'tss': ['tss_root'],
        'spair': ['datapath'],
        'pfpascal': ['datapath'],
        'pfwillow': ['datapath'],
    }
    
    if base_name in required_keys:
        for key in required_keys[base_name]:
            if exp_config.get(key) is None:
                raise ValueError(f"Missing required config key '{key}' for dataset '{base_name}' in machine config. "
                               f"Please add it to the 'datasets' section of your machine config file.")
    
    return exp_config


def extract_features_for_dataset(
    dataset_name,
    features_dir,
    dino_model,
    device,
    machine_config,
    split='train',
    batch_size=8,
    num_batches=None,
    max_samples=None,
    n_threads=0,
    model_name=None
):
    """
    Extract features for a single dataset using config-based dataset creation.
    Handles train/eval suffixes properly.
    """
    print(f"\n{'='*60}")
    print(f"Extracting features for: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_output_dir = Path(features_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if features already exist
    existing_files = list(dataset_output_dir.glob('features_batch_*.pkl'))
    if existing_files:
        print(f"  ⚠️  Features already exist ({len(existing_files)} batches). Skipping extraction.")
        print(f"  To re-extract, delete: {dataset_output_dir}")
        return dataset_output_dir
    
    # Get base dataset name and determine if train or eval
    base_name = get_base_dataset_name(dataset_name)
    is_train = dataset_name.endswith('_train') or (base_name in BASE_TRAIN_DATASETS and not dataset_name.endswith('_eval'))
    
    # Create simple exp config for this dataset
    exp_config = create_simple_exp_config(dataset_name, machine_config, split)
    
    # Create dataset using the same functions as run_vector_analysis.py
    try:
        if is_train and base_name in BASE_TRAIN_DATASETS:
            dataset = create_training_dataset_from_exp_config(exp_config, split=split)
        else:
            # For eval datasets, determine the split
            if base_name == 'tss':
                eval_split = None
            elif base_name == 'flyingthings':
                eval_split = 'test'
            else:
                eval_split = 'val'
            dataset = create_eval_dataset_from_exp_config(exp_config, base_name, split=eval_split)
        
        # Disable normalization for synthetic dataset
        if base_name == 'synthetic' and hasattr(dataset, 'processor'):
            dataset.processor.normalize = None
        
        print(f"  ✓ Dataset created successfully")
    except Exception as e:
        print(f"  ✗ Failed to create dataset: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Check if dataset is empty
    try:
        dataset_len = len(dataset)
    except (TypeError, AttributeError):
        # Some datasets might not have __len__, try to check differently
        dataset_len = None
    
    if dataset_len == 0:
        print(f"  ⚠️  Dataset is empty (0 samples). Skipping extraction.")
        return None
    
    if dataset_len is not None:
        print(f"  Dataset has {dataset_len} samples")
    
    # Create dataloader
    if hasattr(dataset, 'collate_fn'):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_threads,
            shuffle=True,
            collate_fn=dataset.collate_fn
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=n_threads,
            shuffle=True,
            persistent_workers=True if n_threads > 0 else False,
            prefetch_factor=8 if n_threads > 0 else None,
        )
    
    print(f"  Dataset size: {len(dataloader)} batches")
    
    # Calculate max batches based on num_batches and/or max_samples
    if max_samples is not None:
        max_batches_from_samples = (max_samples + batch_size - 1) // batch_size  # Ceiling division
        if num_batches is not None:
            max_batches = min(num_batches, max_batches_from_samples)
        else:
            max_batches = max_batches_from_samples
        print(f"  Limiting to {max_samples} samples ({max_batches} batches)")
    else:
        max_batches = num_batches if num_batches is not None else len(dataloader)
        if num_batches is not None:
            print(f"  Limiting to {num_batches} batches")
    
    all_features = []
    batch_metadata = []
    total_samples_processed = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting {dataset_name}")):
        if batch_idx >= max_batches:
            break
        
        try:
            # Extract features (only from src_img to avoid double counting)
            features = extract_features_from_batch(dino_model, batch, device, use_src_only=True)
            
            if not features:  # Skip if no features extracted
                continue
            
            # Create metadata
            metadata = {
                'batch_idx': batch_idx,
                'batch_size': batch['src_img'].shape[0] if 'src_img' in batch else 0,
                'image_shape': batch['src_img'].shape[1:] if 'src_img' in batch else None,
                'feature_shapes': {k: v.shape for k, v in features.items()},
                'device': str(device),
                'model_name': model_name,
                'dataset': dataset_name,
                'split': split
            }
            
            # Add category information
            if 'category' in batch:
                metadata['category'] = batch['category'].tolist() if hasattr(batch['category'], 'tolist') else batch['category']
            else:
                if base_name != 'synthetic':
                    metadata['category'] = [base_name] * batch['src_img'].shape[0]
                else:
                    batch_size = batch['src_img'].shape[0]
                    metadata['category'] = ['synthetic'] * batch_size
            
            # Save features
            save_features(features, dataset_output_dir, batch_idx, metadata)
            
            all_features.append(features)
            batch_metadata.append(metadata)
            
            # Update sample count
            total_samples_processed += metadata['batch_size']
            
            # Check if we've reached max_samples
            if max_samples is not None and total_samples_processed >= max_samples:
                print(f"  Reached max_samples limit ({max_samples})")
                break
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary
    summary = {
        'total_batches_processed': len(all_features),
        'model_name': model_name,
        'dataset': dataset_name,
        'split': split,
    }
    
    summary_file = dataset_output_dir / 'extraction_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Extracted {len(all_features)} batches")
    print(f"  ✓ Saved to: {dataset_output_dir}")
    
    return dataset_output_dir


def extract_class_labels_from_metadata(all_features, all_metadata, dataset_name):
    """Extract class labels from metadata."""
    base_name = get_base_dataset_name(dataset_name)
    class_labels = []
    
    for batch_idx, (features, metadata) in enumerate(zip(all_features, all_metadata)):
        batch_size = metadata['batch_size']
        categories = metadata.get('category', [None] * batch_size)
        
        # Only process src_img (since we only extracted from src_img)
        if 'src_img' in features:
            img_features = features['src_img']
            batch_size_actual, num_patches, dim = img_features.shape
            
            for sample_idx in range(batch_size_actual):
                sample_category = categories[sample_idx] if sample_idx < len(categories) else None
                
                if sample_category and sample_category != base_name and sample_category != 'synthetic':
                    label = sample_category
                else:
                    label = dataset_name  # Use full name with suffix
                
                for patch_idx in range(num_patches):
                    class_labels.append(label)
    
    return class_labels


def prepare_umap_data_src_only(all_features, all_metadata):
    """
    Prepare data for UMAP by flattening features (only src_img since we only extracted from src).
    """
    print("Preparing data for UMAP...")
    
    all_feature_vectors = []
    all_categories = []
    feature_info = []
    
    for batch_idx, (features, metadata) in enumerate(zip(all_features, all_metadata)):
        batch_size = metadata['batch_size']
        categories = metadata.get('category', [None] * batch_size)
        
        # Only process src_img (since we only extracted from src_img)
        if 'src_img' in features:
            img_features = features['src_img']
            batch_size_actual, num_patches, dim = img_features.shape
            
            for sample_idx in range(batch_size_actual):
                sample_features = img_features[sample_idx]
                sample_category = categories[sample_idx] if sample_idx < len(categories) else None
                
                for patch_idx in range(num_patches):
                    patch_feature = sample_features[patch_idx]
                    all_feature_vectors.append(patch_feature.cpu().numpy())
                    all_categories.append(sample_category)
                    feature_info.append({
                        'batch_idx': batch_idx,
                        'sample_idx': sample_idx,
                        'patch_idx': patch_idx,
                        'img_type': 'src_img'
                    })
    
    feature_matrix = np.array(all_feature_vectors)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Total feature vectors: {len(all_feature_vectors)}")
    
    return feature_matrix, all_categories, feature_info


def subsample_equal_per_dataset(dataset_results, num_samples_umap, random_state=42):
    """Subsample each dataset to have the same number of feature vectors."""
    rng = np.random.default_rng(random_state)
    
    if num_samples_umap is None:
        min_samples = min(len(r['features']) for r in dataset_results if r is not None)
        num_samples_umap = min_samples
        print(f"  Using minimum samples per dataset: {num_samples_umap:,}")
    else:
        print(f"  Using specified samples per dataset: {num_samples_umap:,}")
    
    subsampled_results = []
    
    for result in dataset_results:
        if result is None:
            subsampled_results.append(None)
            continue
        
        n_samples = len(result['features'])
        dataset_name = result['name']
        
        if n_samples > num_samples_umap:
            indices = rng.choice(n_samples, num_samples_umap, replace=False)
            
            result['features'] = result['features'][indices]
            result['categories'] = [result['categories'][i] for i in indices]
            result['class_labels'] = [result['class_labels'][i] for i in indices] if 'class_labels' in result else None
            
            print(f"    {dataset_name}: {n_samples:,} -> {num_samples_umap:,} samples")
        elif n_samples < num_samples_umap:
            print(f"    ⚠️  {dataset_name}: Only {n_samples:,} samples (less than {num_samples_umap:,})")
        else:
            print(f"    {dataset_name}: {n_samples:,} samples (no subsampling needed)")
        
        subsampled_results.append(result)
    
    return subsampled_results


def process_single_dataset(features_dir, dataset_name, output_base_dir):
    """Process a single dataset: load features, extract class labels."""
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'='*60}")
    
    dataset_dir = Path(features_dir) / dataset_name
    if not dataset_dir.exists():
        print(f"  ⚠️  Dataset directory not found: {dataset_dir}")
        return None
    
    # Load features
    all_features, all_metadata, summary = load_extracted_features(dataset_dir)
    if not all_features:
        print(f"  ⚠️  No features found for {dataset_name}")
        return None
    
    # Prepare data (using src-only version since we only extracted from src_img)
    feature_matrix, categories, feature_info = prepare_umap_data_src_only(all_features, all_metadata)
    
    # Extract class labels
    class_labels = extract_class_labels_from_metadata(all_features, all_metadata, dataset_name)
    
    # Count class labels
    class_label_counts = Counter(class_labels)
    print(f"  Class label distribution:")
    for label, count in class_label_counts.most_common(10):
        print(f"    {label}: {count:,}")
    
    # Save individual dataset info
    dataset_output_dir = Path(output_base_dir) / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(dataset_output_dir / 'feature_matrix.npy', feature_matrix)
    
    with open(dataset_output_dir / 'categories.json', 'w') as f:
        json.dump(categories, f)
    
    with open(dataset_output_dir / 'class_labels.json', 'w') as f:
        json.dump(class_labels, f)
    
    with open(dataset_output_dir / 'metadata.json', 'w') as f:
        json.dump({
            'dataset_name': dataset_name,
            'num_features': len(feature_matrix),
            'feature_dim': feature_matrix.shape[1],
            'category_counts': dict(Counter(categories)),
            'class_label_counts': dict(class_label_counts),
        }, f, indent=2)
    
    print(f"  ✓ Saved results to {dataset_output_dir}")
    
    return {
        'name': dataset_name,
        'features': feature_matrix,
        'categories': categories,
        'class_labels': class_labels,
        'class_label_counts': class_label_counts,
        'metadata': {
            'num_features': len(feature_matrix),
            'feature_dim': feature_matrix.shape[1]
        }
    }


def compute_dataset_metrics(feature_matrix_1, feature_matrix_2, dataset_name_1, dataset_name_2, epsilon=1e-6):
    """
    Compute simple metrics to quantify how different two datasets are.
    Uses reversed coverage ratio: d_E_to_T / (d_T_to_E + ε) for consistency.
    """
    print(f"  Computing metrics between {dataset_name_1} and {dataset_name_2}...")
    
    # Subsample if datasets are too large (for efficiency)
    max_samples = 10000
    if len(feature_matrix_1) > max_samples:
        indices_1 = np.random.choice(len(feature_matrix_1), max_samples, replace=False)
        feature_matrix_1 = feature_matrix_1[indices_1]
    if len(feature_matrix_2) > max_samples:
        indices_2 = np.random.choice(len(feature_matrix_2), max_samples, replace=False)
        feature_matrix_2 = feature_matrix_2[indices_2]
    
    metrics = {}
    
    # 1. Centroid distance (L2)
    centroid_1 = feature_matrix_1.mean(axis=0)
    centroid_2 = feature_matrix_2.mean(axis=0)
    centroid_distance = np.linalg.norm(centroid_1 - centroid_2)
    metrics['centroid_distance'] = float(centroid_distance)
    
    # 2. Mean pairwise distance (sample a subset for efficiency)
    n_samples = min(1000, len(feature_matrix_1), len(feature_matrix_2))
    sample_1 = feature_matrix_1[np.random.choice(len(feature_matrix_1), n_samples, replace=False)]
    sample_2 = feature_matrix_2[np.random.choice(len(feature_matrix_2), n_samples, replace=False)]
    pairwise_distances = cdist(sample_1, sample_2, metric='euclidean')
    metrics['mean_pairwise_distance'] = float(pairwise_distances.mean())
    metrics['min_pairwise_distance'] = float(pairwise_distances.min())
    metrics['max_pairwise_distance'] = float(pairwise_distances.max())
    
    # 3. Mean cosine similarity
    cosine_sim = cosine_similarity(sample_1, sample_2)
    metrics['mean_cosine_similarity'] = float(cosine_sim.mean())
    metrics['min_cosine_similarity'] = float(cosine_sim.min())
    metrics['max_cosine_similarity'] = float(cosine_sim.max())
    
    # 4. Coverage: mean distance from dataset 2 to nearest neighbor in dataset 1
    tree_1 = cKDTree(feature_matrix_1)
    distances_2_to_1, _ = tree_1.query(feature_matrix_2, k=1)
    distances_2_to_1 = distances_2_to_1.flatten()
    metrics['coverage_2_to_1'] = float(distances_2_to_1.mean())
    metrics['coverage_2_to_1_median'] = float(np.median(distances_2_to_1))
    metrics['coverage_2_to_1_p95'] = float(np.percentile(distances_2_to_1, 95))
    
    # 5. Reverse coverage: mean distance from dataset 1 to nearest neighbor in dataset 2
    tree_2 = cKDTree(feature_matrix_2)
    distances_1_to_2, _ = tree_2.query(feature_matrix_1, k=1)
    distances_1_to_2 = distances_1_to_2.flatten()
    metrics['coverage_1_to_2'] = float(distances_1_to_2.mean())
    metrics['coverage_1_to_2_median'] = float(np.median(distances_1_to_2))
    metrics['coverage_1_to_2_p95'] = float(np.percentile(distances_1_to_2, 95))
    
    # 6. Coverage score: d_1_to_2 / (d_2_to_1 + ε)
    # Higher is better: > 1 means dataset 1 covers dataset 2 well (dataset 1 extends beyond dataset 2)
    coverage_score = distances_1_to_2.mean() / (distances_2_to_1.mean() + epsilon)
    metrics['coverage_score'] = float(coverage_score)
    
    # 7. Standard deviation ratio (measure of spread)
    std_1 = feature_matrix_1.std(axis=0).mean()
    std_2 = feature_matrix_2.std(axis=0).mean()
    metrics['std_ratio'] = float(std_1 / (std_2 + 1e-10))
    
    return metrics


def compute_all_pairwise_metrics(dataset_results, output_dir):
    """
    Compute pairwise metrics between all datasets (using equal-sampled data).
    Creates 11x11 matrix for all dataset pairs.
    """
    print(f"\n{'='*60}")
    print("Computing pairwise dataset metrics...")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter out None results
    valid_results = [r for r in dataset_results if r is not None]
    n_datasets = len(valid_results)
    
    if n_datasets < 2:
        print("  ⚠️  Need at least 2 datasets for pairwise comparison")
        return
    
    print(f"  Computing metrics for {n_datasets} datasets ({n_datasets * (n_datasets - 1) // 2} pairs)...")
    
    # Compute all pairwise metrics
    metrics_dict = {}
    metrics_matrix = {}
    
    for i, result1 in enumerate(valid_results):
        for j, result2 in enumerate(valid_results):
            if i >= j:
                continue  # Only compute upper triangle
            
            name1 = result1['name']
            name2 = result2['name']
            pair_key = f"{name1}_vs_{name2}"
            
            print(f"  [{len(metrics_dict) + 1}/{n_datasets*(n_datasets-1)//2}] {pair_key}")
            
            metrics = compute_dataset_metrics(
                result1['features'], result2['features'],
                name1, name2
            )
            metrics_dict[pair_key] = metrics
    
    # Create distance matrices for key metrics
    dataset_names = [r['name'] for r in valid_results]
    
    for metric_name in ['centroid_distance', 'mean_pairwise_distance', 'mean_cosine_similarity', 
                        'coverage_score', 'coverage_2_to_1', 'coverage_1_to_2']:
        matrix = np.zeros((n_datasets, n_datasets))
        
        for i, name1 in enumerate(dataset_names):
            for j, name2 in enumerate(dataset_names):
                if i == j:
                    matrix[i, j] = 0.0
                elif i < j:
                    pair_key = f"{name1}_vs_{name2}"
                    matrix[i, j] = metrics_dict[pair_key].get(metric_name, np.nan)
                else:
                    # For reverse pairs, use appropriate metric
                    pair_key = f"{name2}_vs_{name1}"
                    if metric_name == 'coverage_2_to_1':
                        matrix[i, j] = metrics_dict[pair_key].get('coverage_1_to_2', np.nan)
                    elif metric_name == 'coverage_1_to_2':
                        matrix[i, j] = metrics_dict[pair_key].get('coverage_2_to_1', np.nan)
                    else:
                        matrix[i, j] = metrics_dict[pair_key].get(metric_name, np.nan)
        
        # Make symmetric for distance metrics
        if metric_name in ['centroid_distance', 'mean_pairwise_distance', 'coverage_2_to_1', 'coverage_1_to_2']:
            matrix = (matrix + matrix.T) / 2
        
        metrics_matrix[metric_name] = matrix
    
    # Save metrics
    with open(output_dir / 'pairwise_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    # Create visualization of distance matrices
    n_metrics = len(metrics_matrix)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    metric_display_names = {
        'centroid_distance': 'Centroid Distance',
        'mean_pairwise_distance': 'Mean Pairwise Distance',
        'mean_cosine_similarity': 'Mean Cosine Similarity',
        'coverage_score': 'Coverage Score (higher=better)',
        'coverage_2_to_1': 'Coverage: Eval→Train',
        'coverage_1_to_2': 'Coverage: Train→Eval',
    }
    
    for idx, (metric_name, matrix) in enumerate(list(metrics_matrix.items())[:n_metrics]):
        ax = axes[idx]
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(metric_display_names.get(metric_name, metric_name), fontsize=12)
        ax.set_xticks(range(n_datasets))
        ax.set_yticks(range(n_datasets))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(dataset_names, fontsize=8)
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_metrics_heatmaps.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved metrics heatmaps to {output_dir / 'pairwise_metrics_heatmaps.png'}")
    plt.close()
    
    # Create summary table
    summary_data = []
    for pair_key, metrics in metrics_dict.items():
        summary_data.append({
            'Dataset 1': pair_key.split('_vs_')[0],
            'Dataset 2': pair_key.split('_vs_')[1],
            'Centroid Distance': metrics.get('centroid_distance', np.nan),
            'Mean Pairwise Distance': metrics.get('mean_pairwise_distance', np.nan),
            'Mean Cosine Similarity': metrics.get('mean_cosine_similarity', np.nan),
            'Coverage Score': metrics.get('coverage_score', np.nan),
            'Coverage 1→2': metrics.get('coverage_1_to_2', np.nan),
            'Coverage 2→1': metrics.get('coverage_2_to_1', np.nan),
        })
    
    df = pd.DataFrame(summary_data)
    df = df.sort_values(['Dataset 1', 'Dataset 2'])
    df.to_csv(output_dir / 'pairwise_metrics_summary.csv', index=False)
    print(f"  ✓ Saved summary table to {output_dir / 'pairwise_metrics_summary.csv'}")
    
    print(f"\n  Summary statistics:")
    print(f"    Total pairs compared: {len(metrics_dict)}")
    print(f"    Mean centroid distance: {df['Centroid Distance'].mean():.4f}")
    print(f"    Mean pairwise distance: {df['Mean Pairwise Distance'].mean():.4f}")
    print(f"    Mean cosine similarity: {df['Mean Cosine Similarity'].mean():.4f}")
    print(f"    Mean coverage score: {df['Coverage Score'].mean():.4f}")


def create_combined_umap(dataset_results, output_dir, num_samples_umap=None, skip_class_labels=False):
    """
    Create combined UMAP from all datasets with equal sampling.
    Uses GPU-accelerated cuML UMAP when available.
    """
    print(f"\n{'='*60}")
    print("Creating Combined UMAP Analysis")
    print(f"{'='*60}")
    print(f"Using {'GPU-accelerated cuML' if USE_CUML else 'CPU'} UMAP")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    valid_results = [r for r in dataset_results if r is not None]
    if not valid_results:
        print("  ⚠️  No valid datasets to process")
        return
    
    print(f"\nStep 1: Equal subsampling per dataset...")
    valid_results = subsample_equal_per_dataset(valid_results, num_samples_umap)
    
    print(f"\nStep 2: Combining all datasets...")
    all_features = []
    all_dataset_labels = []
    all_class_labels = []
    
    for result in valid_results:
        if result is None:
            continue
        all_features.append(result['features'])
        all_dataset_labels.extend([result['name']] * len(result['features']))
        all_class_labels.extend(result['class_labels'])
    
    combined_features = np.vstack(all_features)
    all_dataset_labels = np.array(all_dataset_labels)
    all_class_labels = np.array(all_class_labels)
    
    print(f"  Combined feature matrix shape: {combined_features.shape}")
    print(f"  Total samples: {len(combined_features):,}")
    print(f"  Unique datasets: {len(set(all_dataset_labels))}")
    print(f"  Unique class labels: {len(set(all_class_labels))}")
    
    print(f"\nStep 3: Computing UMAP parameters...")
    params, _ = calculate_umap_parameters(all_dataset_labels.tolist(), combined_features)
    
    print(f"\nStep 4: Running UMAP (unsupervised on all data)...")
    print(f"  Using {'GPU-accelerated cuML' if USE_CUML else 'CPU'} UMAP")
    
    temp_output = output_dir / 'temp_umap'
    temp_output.mkdir(exist_ok=True)
    
    combined_embeddings = run_umap_analysis(combined_features, params, temp_output)
    
    print(f"  UMAP embeddings shape: {combined_embeddings.shape}")
    
    # Save combined embeddings
    np.save(output_dir / 'combined_umap_embeddings.npy', combined_embeddings)
    np.save(output_dir / 'combined_features.npy', combined_features)
    np.save(output_dir / 'dataset_labels.npy', all_dataset_labels)
    np.save(output_dir / 'class_labels.npy', all_class_labels)
    
    print(f"\nStep 5: Creating visualizations...")
    
    # Option A: Colored by dataset name (primary)
    create_dataset_colored_visualization(
        combined_embeddings, all_dataset_labels, valid_results, output_dir
    )
    
    # Option B: Colored by class labels (optional)
    if not skip_class_labels:
        create_class_label_colored_visualization(
            combined_embeddings, all_dataset_labels, all_class_labels, valid_results, output_dir
        )
    else:
        print("  Skipping class-label-colored visualization (--skip_class_labels)")


def create_dataset_colored_visualization(embeddings, dataset_labels, dataset_results, output_dir):
    """Create visualization colored by dataset name (Option A - primary)."""
    print("  Creating dataset-colored visualization (Option A)...")
    
    unique_datasets = sorted(set(dataset_labels))
    n_datasets = len(unique_datasets)
    
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, n_datasets)))
    dataset_to_color = {ds: colors[i % len(colors)] for i, ds in enumerate(unique_datasets)}
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.suptitle('UMAP Analysis: All Datasets (Colored by Dataset Name)', fontsize=16, fontweight='bold')
    
    # 1. All datasets together
    ax1 = axes[0, 0]
    for dataset in unique_datasets:
        mask = dataset_labels == dataset
        ax1.scatter(embeddings[mask, 0], embeddings[mask, 1],
                   c=[dataset_to_color[dataset]], label=dataset, alpha=0.6, s=10)
    ax1.set_title('All Datasets Combined', fontsize=12)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    
    # 2. Training vs Eval split
    ax2 = axes[0, 1]
    train_mask = np.array([ds in TRAIN_DATASETS or ds.endswith('_train') for ds in dataset_labels])
    eval_mask = ~train_mask
    ax2.scatter(embeddings[train_mask, 0], embeddings[train_mask, 1],
               c='blue', label='Training', alpha=0.5, s=10)
    ax2.scatter(embeddings[eval_mask, 0], embeddings[eval_mask, 1],
               c='orange', label='Eval', alpha=0.5, s=10)
    ax2.set_title('Training vs Eval Split', fontsize=12)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    ax2.legend()
    
    # 3. Density plot
    ax3 = axes[1, 0]
    ax3.hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Blues')
    ax3.set_title('Feature Density', fontsize=12)
    ax3.set_xlabel('UMAP 1')
    ax3.set_ylabel('UMAP 2')
    
    # 4. Dataset distribution
    ax4 = axes[1, 1]
    dataset_counts = Counter(dataset_labels)
    datasets_list = list(dataset_counts.keys())
    counts_list = [dataset_counts[d] for d in datasets_list]
    ax4.barh(range(len(datasets_list)), counts_list)
    ax4.set_yticks(range(len(datasets_list)))
    ax4.set_yticklabels(datasets_list, fontsize=8)
    ax4.set_xlabel('Number of Features')
    ax4.set_title('Samples per Dataset', fontsize=12)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_dataset_colored.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_dir / 'umap_dataset_colored.png'}")


def create_class_label_colored_visualization(embeddings, dataset_labels, class_labels, dataset_results, output_dir):
    """Create visualization colored by class labels (Option B - interpretation)."""
    print("  Creating class-label-colored visualization (Option B)...")
    
    datasets_with_class_labels = set()
    for result in dataset_results:
        if result is None:
            continue
        unique_labels = set(result['class_labels'])
        # Check if there are class labels beyond just the dataset name
        base_name = get_base_dataset_name(result['name'])
        if len(unique_labels) > 1 or (len(unique_labels) == 1 and list(unique_labels)[0] not in [result['name'], base_name, 'synthetic']):
            datasets_with_class_labels.add(result['name'])
    
    print(f"    Datasets with class labels: {sorted(datasets_with_class_labels)}")
    
    unique_class_labels = sorted(set(class_labels))
    n_labels = len(unique_class_labels)
    
    if n_labels <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
    elif n_labels <= 40:
        colors = list(plt.cm.tab20(np.linspace(0, 1, 20))) + list(plt.cm.tab20b(np.linspace(0, 1, 20)))
    else:
        colors = plt.cm.turbo(np.linspace(0, 1, n_labels))
    
    class_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_class_labels)}
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('UMAP Analysis: Colored by Class Labels (or Dataset if no labels)', fontsize=16, fontweight='bold')
    
    # 1. All points colored by class/dataset label
    ax1 = axes[0, 0]
    # Plot datasets without class labels first (as background)
    for dataset in set(dataset_labels):
        if dataset not in datasets_with_class_labels:
            mask = dataset_labels == dataset
            ax1.scatter(embeddings[mask, 0], embeddings[mask, 1],
                       c=[class_to_color[dataset]], label=f"{dataset} (dataset)", 
                       alpha=0.3, s=3, edgecolors='none')
    
    # Plot datasets with class labels (more prominent)
    for label in unique_class_labels:
        if label in datasets_with_class_labels:
            continue  # Skip dataset names that are in the class labels set
        
        mask = class_labels == label
        if mask.sum() > 0:
            # Check if this label belongs to a dataset with class labels
            dataset_for_label = dataset_labels[mask][0] if len(dataset_labels[mask]) > 0 else None
            if dataset_for_label in datasets_with_class_labels:
                ax1.scatter(embeddings[mask, 0], embeddings[mask, 1],
                           c=[class_to_color[label]], label=label, alpha=0.6, s=5)
    
    ax1.set_title('All Features (Class Labels where Available)', fontsize=12)
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1, 
              framealpha=0.9)
    
    # 2. Show only datasets WITH class labels
    ax2 = axes[0, 1]
    if datasets_with_class_labels:
        mask_with_labels = np.array([ds in datasets_with_class_labels for ds in dataset_labels])
        filtered_embeddings = embeddings[mask_with_labels]
        filtered_class_labels = class_labels[mask_with_labels]
        
        for label in sorted(set(filtered_class_labels)):
            mask = filtered_class_labels == label
            if mask.sum() > 0:
                ax2.scatter(filtered_embeddings[mask, 0], filtered_embeddings[mask, 1],
                           c=[class_to_color[label]], label=label, alpha=0.6, s=5)
        ax2.set_title('Datasets with Class Labels Only', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'No datasets with class labels', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Datasets with Class Labels Only', fontsize=12)
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    if datasets_with_class_labels:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=1)
    
    # 3. Individual dataset views (first 2 with class labels)
    datasets_to_show = sorted(datasets_with_class_labels)[:2]
    for idx, dataset in enumerate(datasets_to_show):
        if idx >= 2:
            break
        ax = axes[1, idx]
        mask = dataset_labels == dataset
        dataset_embeddings = embeddings[mask]
        dataset_class_labels = class_labels[mask]
        
        for label in sorted(set(dataset_class_labels)):
            label_mask = dataset_class_labels == label
            ax.scatter(dataset_embeddings[label_mask, 0], dataset_embeddings[label_mask, 1],
                      c=[class_to_color[label]], label=label, alpha=0.6, s=5)
        ax.set_title(f'{dataset}\n({mask.sum():,} samples)', fontsize=10)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(fontsize=6, ncol=1)
    
    # 4. Class label distribution
    ax4 = axes[1, 1] if len(datasets_to_show) < 2 else axes[1, 1]
    class_label_counts = Counter(class_labels)
    top_labels = class_label_counts.most_common(15)
    labels_list = [l[0] for l in top_labels]
    counts_list = [l[1] for l in top_labels]
    ax4.barh(range(len(labels_list)), counts_list)
    ax4.set_yticks(range(len(labels_list)))
    ax4.set_yticklabels(labels_list, fontsize=7)
    ax4.set_xlabel('Number of Features')
    ax4.set_title('Top 15 Class Labels', fontsize=12)
    ax4.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'umap_class_label_colored.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✓ Saved: {output_dir / 'umap_class_label_colored.png'}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch UMAP analysis with config-based feature extraction'
    )
    
    # Config arguments
    parser.add_argument('--machine_config', type=str,
                       default='slurm/machine_configs/local.yaml',
                       help='Path to machine config YAML file')
    
    # Feature extraction arguments
    parser.add_argument('--extract_features', action='store_true',
                       help='Extract features before running UMAP analysis')
    parser.add_argument('--features_dir', type=str, default='extracted_features',
                       help='Directory to save/load extracted features')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_batches', type=int, default=None,
                       help='Maximum number of batches to extract (default: all)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to extract per dataset (default: all)')
    parser.add_argument('--n_threads', type=int, default=0,
                       help='Number of parallel threads for dataloaders')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--model_name', type=str, 
                       default='facebook/dinov3-vit7b16-pretrain-lvd1689m',
                       help='DinoV3 model name')
    
    # UMAP analysis arguments
    parser.add_argument('--output_dir', type=str, default='umap_batch_results',
                       help='Output directory for UMAP results')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                       help='Specific datasets to process (default: all)')
    parser.add_argument('--num_samples_umap', type=int, default=None,
                       help='Number of samples per dataset for UMAP analysis (default: use minimum)')
    parser.add_argument('--skip_class_labels', action='store_true',
                       help='Skip class-label-colored visualization (Option B)')
    parser.add_argument('--skip_metrics', action='store_true',
                       help='Skip pairwise metrics computation')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Batch UMAP Analysis: Config-Based Feature Extraction")
    print("="*60)
    print(f"Features directory: {args.features_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"UMAP backend: {'GPU-accelerated cuML' if USE_CUML else 'CPU (standard UMAP)'}")
    
    # Load machine config
    machine_config_path = Path(args.machine_config)
    if not machine_config_path.is_absolute():
        machine_config_path = project_root / machine_config_path
    
    if not machine_config_path.exists():
        raise FileNotFoundError(f"Machine config not found: {machine_config_path}")
    
    machine_config = load_machine_config(str(machine_config_path))
    
    # Determine which datasets to process
    if args.datasets:
        datasets_to_process = args.datasets
    else:
        datasets_to_process = ALL_DATASETS
    
    print(f"\nDatasets to process: {datasets_to_process}")
    
    # Step 1: Extract features if requested
    dino_model = None
    if args.extract_features:
        print(f"\n{'='*60}")
        print("Step 1: Extracting Features")
        print(f"{'='*60}")
        
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        print("\nInitializing DinoV3 model...")
        dino_model = DinoV3(pretrained_model_name=args.model_name)
        print("DinoV3 model loaded successfully!")
        
        for dataset_name in datasets_to_process:
            extract_features_for_dataset(
                dataset_name=dataset_name,
                features_dir=args.features_dir,
                dino_model=dino_model,
                device=device,
                machine_config=machine_config,
                split=args.split,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                max_samples=args.max_samples,
                n_threads=args.n_threads,
                model_name=args.model_name
            )
    
    # Step 2: Process datasets and create UMAP
    print(f"\n{'='*60}")
    print("Step 2: Processing Datasets for UMAP")
    print(f"{'='*60}")
    
    dataset_results = []
    for dataset_name in datasets_to_process:
        result = process_single_dataset(
            args.features_dir, dataset_name, args.output_dir
        )
        dataset_results.append(result)
    
    # Step 3: Create combined UMAP with visualizations
    create_combined_umap(
        dataset_results, 
        args.output_dir, 
        args.num_samples_umap,
        skip_class_labels=args.skip_class_labels
    )
    
    # Step 4: Compute pairwise metrics
    if not args.skip_metrics:
        compute_all_pairwise_metrics(dataset_results, args.output_dir)
    
    print("\n" + "="*60)
    print("Batch Analysis Complete!")
    print("="*60)
    print(f"Results saved to: {args.output_dir}")
    print(f"  - Individual dataset data: {args.output_dir}/<dataset_name>/")
    print(f"  - Combined UMAP (dataset-colored): {args.output_dir}/umap_dataset_colored.png")
    if not args.skip_class_labels:
        print(f"  - Combined UMAP (class-label-colored): {args.output_dir}/umap_class_label_colored.png")
    if not args.skip_metrics:
        print(f"  - Pairwise metrics: {args.output_dir}/pairwise_metrics.json")
        print(f"  - Metrics heatmaps: {args.output_dir}/pairwise_metrics_heatmaps.png")


if __name__ == "__main__":
    main()

