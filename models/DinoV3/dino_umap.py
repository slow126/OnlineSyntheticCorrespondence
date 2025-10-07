#!/usr/bin/env python3
"""
UMAP analysis script for extracted DinoV3 features.
Loads extracted features and metadata, then runs UMAP dimensionality reduction.
"""

import argparse
import pickle
import json
import numpy as np
import torch
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import sys
import os

# Import cuML UMAP for GPU acceleration
print("Attempting to import UMAP libraries...")
print(f"Python executable: {sys.executable}")
print(f"Current working directory: {os.getcwd()}")

# Check key environment variables
print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH', 'Not set')}")

# Check CUDA availability first
try:
    import cupy as cp
    print(f"CuPy available: {cp.cuda.runtime.getDeviceCount()} GPU(s) detected")
    # Print CUDA version that Python is using (from cupy, if available)
    try:
        cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cuda_major = cuda_version // 1000
        cuda_minor = (cuda_version % 1000) // 10
        print(f"CUDA Runtime Version (used by Python/cuPy): {cuda_major}.{cuda_minor}")
    except Exception as e:
        print(f"Could not determine CUDA runtime version from cuPy: {e}")
    CUDA_AVAILABLE = True
except ImportError:
    print("CuPy not available - CUDA may not be properly configured")
    CUDA_AVAILABLE = False

# Try cuML first
try:
    from cuml.manifold.umap import UMAP
    print("✅ Successfully imported cuML UMAP (GPU accelerated)")
    USE_CUML = True
except ImportError as e:
    print(f"❌ Failed to import cuML UMAP: {e}")
    print("This might be due to:")
    print("  - cuML not installed")
    print("  - CUDA environment issues")
    print("  - Debug environment not having access to CUDA libraries")
    print("  - CUDA library version mismatch (common with conda cuML)")
    
    # Check for specific CUDA library issues
    if "undefined symbol" in str(e) or "cublas" in str(e):
        print("\n🔧 CUDA Library Version Mismatch Detected!")
        print("This is likely a cuML/CUDA version compatibility issue.")
        print("Solutions:")
        print("  1. Reinstall cuML: conda install -c rapidsai -c conda-forge -c nvidia cuml")
        print("  2. Or use pip: pip install cuml-cu12")
        print("  3. Or use standard UMAP (fallback will be used)")
        print("  4. Check CUDA version: nvidia-smi")
        print("  5. Ensure cuML matches your CUDA version")
    
    # Try standard UMAP as fallback
    try:
        from umap import UMAP
        print("✅ Successfully imported standard UMAP (CPU only)")
        USE_CUML = False
    except ImportError as e2:
        print(f"❌ Failed to import standard UMAP: {e2}")
        raise ImportError("Neither cuml.manifold.umap nor umap-learn could be imported")

# Print environment info for debugging
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"CUDA available: {CUDA_AVAILABLE}")
print(f"Using cuML: {USE_CUML}")

def load_extracted_features(features_dir):
    """
    Load all extracted features and metadata from the features directory.
    Supports both single directory and multiple dataset directories.
    
    Args:
        features_dir: Path to directory containing extracted features (or parent directory with subdirectories)
        
    Returns:
        tuple: (all_features, all_metadata, summary)
    """
    features_dir = Path(features_dir)
    
    # Check if this is a single dataset directory or parent directory with multiple datasets
    feature_files = list(features_dir.glob('features_batch_*.pkl'))
    
    if feature_files:
        # Single dataset directory
        print(f"Loading from single dataset directory: {features_dir}")
        return _load_single_dataset(features_dir)
    else:
        # Multiple dataset directories
        print(f"Loading from multiple dataset directories in: {features_dir}")
        return _load_multiple_datasets(features_dir)

def _load_single_dataset(dataset_dir):
    """Load features from a single dataset directory."""
    # Load summary first
    summary_file = dataset_dir / 'extraction_summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        print(f"Loaded summary: {summary['total_batches_processed']} batches processed")
    else:
        summary = None
        print("No summary file found")
    
    # Load all feature files
    feature_files = sorted(dataset_dir.glob('features_batch_*.pkl'))
    metadata_files = sorted(dataset_dir.glob('metadata_batch_*.json'))
    
    print(f"Found {len(feature_files)} feature files and {len(metadata_files)} metadata files")
    
    all_features = []
    all_metadata = []
    
    for feat_file, meta_file in zip(feature_files, metadata_files):
        # Load features
        with open(feat_file, 'rb') as f:
            features = pickle.load(f)
        all_features.append(features)
        
        # Load metadata
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        all_metadata.append(metadata)
    
    return all_features, all_metadata, summary

def _load_multiple_datasets(parent_dir):
    """Load features from multiple dataset directories."""
    # Find all subdirectories that contain feature files
    dataset_dirs = []
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and list(subdir.glob('features_batch_*.pkl')):
            dataset_dirs.append(subdir)
    
    if not dataset_dirs:
        raise ValueError(f"No dataset directories found in {parent_dir}")
    
    print(f"Found {len(dataset_dirs)} dataset directories: {[d.name for d in dataset_dirs]}")
    
    all_features = []
    all_metadata = []
    all_summaries = []
    
    for dataset_dir in sorted(dataset_dirs):
        print(f"\nLoading dataset: {dataset_dir.name}")
        features, metadata, summary = _load_single_dataset(dataset_dir)
        
        # Add dataset name to metadata for tracking
        for meta in metadata:
            meta['dataset_name'] = dataset_dir.name
        
        all_features.extend(features)
        all_metadata.extend(metadata)
        if summary:
            summary['dataset_name'] = dataset_dir.name
            all_summaries.append(summary)
    
    # Combine summaries
    combined_summary = {
        'total_datasets': len(dataset_dirs),
        'dataset_names': [d.name for d in dataset_dirs],
        'total_batches_processed': len(all_features),
        'individual_summaries': all_summaries
    }
    
    return all_features, all_metadata, combined_summary

def prepare_umap_data(all_features, all_metadata):
    """
    Prepare data for UMAP by flattening features and collecting categories.
    
    Args:
        all_features: List of feature dictionaries
        all_metadata: List of metadata dictionaries
        
    Returns:
        tuple: (flattened_features, categories, feature_info)
    """
    print("Preparing data for UMAP...")
    
    # Collect all features and categories
    all_feature_vectors = []
    all_categories = []
    feature_info = []  # Track which batch/sample each feature came from
    
    for batch_idx, (features, metadata) in enumerate(zip(all_features, all_metadata)):
        batch_size = metadata['batch_size']
        categories = metadata.get('category', [None] * batch_size)
        
        # Process each image type (src_img, trg_img)
        for img_type in ['src_img', 'trg_img']:
            if img_type in features:
                img_features = features[img_type]  # Shape: (batch_size, num_patches, dim)
                batch_size_actual, num_patches, dim = img_features.shape
                
                # Flatten to individual patch features
                for sample_idx in range(batch_size_actual):
                    sample_features = img_features[sample_idx]  # (num_patches, dim)
                    sample_category = categories[sample_idx] if sample_idx < len(categories) else None
                    
                    # Add each patch as a separate point
                    for patch_idx in range(num_patches):
                        patch_feature = sample_features[patch_idx]  # (dim,)
                        all_feature_vectors.append(patch_feature.cpu().numpy())
                        all_categories.append(sample_category)
                        feature_info.append({
                            'batch_idx': batch_idx,
                            'sample_idx': sample_idx,
                            'patch_idx': patch_idx,
                            'img_type': img_type
                        })
    
    # Convert to numpy arrays
    feature_matrix = np.array(all_feature_vectors)
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Total feature vectors: {len(all_feature_vectors)}")
    
    return feature_matrix, all_categories, feature_info

def calculate_umap_parameters(categories, feature_matrix, build_algo='auto', n_clusters=None):
    """
    Calculate UMAP parameters based on category statistics.
    
    Args:
        categories: List of categories for each feature
        feature_matrix: Feature matrix
        build_algo: Build algorithm for cuML UMAP
        n_clusters: Number of clusters for batching (auto-calculated if None)
        
    Returns:
        dict: UMAP parameters
    """
    # Count categories (excluding None)
    valid_categories = [cat for cat in categories if cat is not None]
    category_counts = Counter(valid_categories)
    
    print(f"\nCategory Statistics:")
    print(f"  Total features: {len(categories)}")
    print(f"  Features with categories: {len(valid_categories)}")
    print(f"  Unique categories: {len(category_counts)}")
    
    # Show synthetic vs real breakdown
    synthetic_count = category_counts.get('synthetic', 0)
    real_count = len(valid_categories) - synthetic_count
    print(f"  Synthetic features: {synthetic_count}")
    print(f"  Real dataset features: {real_count}")
    print(f"  Category distribution: {dict(category_counts.most_common(10))}")
    
    # Calculate parameters
    n_unique_categories = len(category_counts)
    n_samples = len(feature_matrix)
    
    # n_neighbors: only use default if ONLY synthetic category exists
    if valid_categories:
        # Check if only synthetic category exists
        unique_cats = set(valid_categories)
        if len(unique_cats) == 1 and 'synthetic' in unique_cats:
            n_neighbors = 15  # Default for synthetic-only data
        else:
            # Calculate based on category frequency for mixed real/synthetic data
            avg_category_freq = len(valid_categories) / n_unique_categories
            n_neighbors = max(5, min(50, int(avg_category_freq * 0.5)))
    else:
        n_neighbors = 15  # Default fallback
    
    # n_components: 2D for visualization
    n_components = 2
    
    # min_dist: smaller for tighter clusters
    min_dist = 0.1
    
    # metric: cosine often works well for high-dimensional features
    metric = 'cosine'
    
    # Base parameters
    params = {
        'n_neighbors': n_neighbors,
        'n_components': n_components,
        'min_dist': min_dist,
        'metric': metric,
        'random_state': 42
    }
    
    # Add cuML-specific parameters
    if USE_CUML:
        params['build_algo'] = build_algo
        
        # Calculate n_clusters for batching if not provided
        if n_clusters is None:
            # Use number of unique categories as base, with some scaling
            n_clusters = max(2, min(8, n_unique_categories))
        
        # Build keywords for cuML
        build_kwds = {
            'nnd_graph_degree': 32,  # Smaller for faster computation
            'nnd_intermediate_graph_degree': 64,
            'nnd_max_iterations': 20,
            'nnd_termination_threshold': 0.0001,
            'nnd_return_distances': True,
            'nnd_n_clusters': n_clusters,
            'nnd_do_batch': True  # Enable batching for large datasets
        }
        
        params['build_kwds'] = build_kwds
        
        print(f"\ncuML UMAP Parameters:")
        print(f"  n_neighbors: {n_neighbors}")
        print(f"  n_components: {n_components}")
        print(f"  min_dist: {min_dist}")
        print(f"  metric: {metric}")
        print(f"  build_algo: {build_algo}")
        print(f"  n_clusters: {n_clusters}")
        print(f"  data_on_host: True (for batching)")
    else:
        print(f"\nStandard UMAP Parameters:")
        print(f"  n_neighbors: {n_neighbors}")
        print(f"  n_components: {n_components}")
        print(f"  min_dist: {min_dist}")
        print(f"  metric: {metric}")
    
    return params, category_counts

def run_umap_analysis(feature_matrix, params, output_dir):
    """
    Run UMAP dimensionality reduction (UNSUPERVISED - category labels only used for visualization).
    
    Args:
        feature_matrix: Feature matrix
        params: UMAP parameters
        output_dir: Output directory for results
        
    Returns:
        np.array: UMAP embeddings
    """
    print(f"\nRunning UMAP analysis...")
    print(f"Input shape: {feature_matrix.shape}")
    print("Note: This is UNSUPERVISED UMAP - category labels are only used for visualization coloring")
    
    # Create UMAP instance
    umap_reducer = UMAP(**params)
    
    # Fit and transform with cuML-specific options
    if USE_CUML:
        # Use data_on_host=True for batching with large datasets
        embeddings = umap_reducer.fit_transform(feature_matrix)
        print("Used cuML UMAP with GPU acceleration and batching")
    else:
        embeddings = umap_reducer.fit_transform(feature_matrix)
        print("Used standard UMAP (CPU)")
    
    print(f"UMAP embeddings shape: {embeddings.shape}")
    
    return embeddings

def visualize_umap_results(embeddings, categories, category_counts, params, output_dir):
    """
    Create visualizations of UMAP results.
    
    Args:
        embeddings: UMAP embeddings
        categories: Category labels
        category_counts: Category count statistics
        params: UMAP parameters
        output_dir: Output directory
    """
    print(f"\nCreating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('UMAP Analysis of DinoV3 Features', fontsize=16)
    
    # 1. All points colored by category
    ax1 = axes[0, 0]
    valid_mask = np.array([cat is not None for cat in categories])
    valid_embeddings = embeddings[valid_mask]
    valid_categories = np.array(categories)[valid_mask]
    
    if len(valid_categories) > 0:
        # Get unique categories and assign colors
        unique_categories = list(set(valid_categories))
        
        # Special handling: synthetic gets red, real data gets other colors
        category_to_color = {}
        if 'synthetic' in unique_categories:
            category_to_color['synthetic'] = 'red'
            # Remove synthetic from list for other color assignment
            real_categories = [cat for cat in unique_categories if cat != 'synthetic']
            if real_categories:
                # Use tab20 colormap but skip red-ish colors
                real_colors = plt.cm.tab20(np.linspace(0, 1, len(real_categories) + 5))  # Extra colors to skip red-ish ones
                # Skip colors that are too red-ish (indices 0, 1, 6, 7, 12, 13, 18, 19)
                red_indices = [0, 1, 6, 7, 12, 13, 18, 19]
                real_colors_filtered = [color for i, color in enumerate(real_colors) if i not in red_indices]
                for i, cat in enumerate(real_categories):
                    category_to_color[cat] = real_colors_filtered[i % len(real_colors_filtered)]
        else:
            # No synthetic data, use normal color assignment
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_categories)))
            category_to_color = dict(zip(unique_categories, colors))
        
        for category in unique_categories:
            mask = valid_categories == category
            ax1.scatter(valid_embeddings[mask, 0], valid_embeddings[mask, 1], 
                       c=[category_to_color[category]], label=category, alpha=0.6, s=1)
        
        ax1.set_title('Features Colored by Category')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    else:
        ax1.scatter(embeddings[:, 0], embeddings[:, 1], alpha=0.6, s=1)
        ax1.set_title('All Features (No Categories)')
    
    ax1.set_xlabel('UMAP 1')
    ax1.set_ylabel('UMAP 2')
    
    # 2. Density plot
    ax2 = axes[0, 1]
    ax2.hexbin(embeddings[:, 0], embeddings[:, 1], gridsize=50, cmap='Blues')
    ax2.set_title('Feature Density')
    ax2.set_xlabel('UMAP 1')
    ax2.set_ylabel('UMAP 2')
    
    # 3. Category distribution
    ax3 = axes[1, 0]
    if category_counts:
        categories_list = list(category_counts.keys())
        counts_list = list(category_counts.values())
        ax3.bar(range(len(categories_list)), counts_list)
        ax3.set_xticks(range(len(categories_list)))
        ax3.set_xticklabels(categories_list, rotation=45, ha='right')
        ax3.set_title('Category Distribution')
        ax3.set_ylabel('Number of Features')
    else:
        ax3.text(0.5, 0.5, 'No Categories Found', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Category Distribution')
    
    # 4. UMAP parameters info
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    if USE_CUML and 'build_algo' in params:
        info_text = f"""cuML UMAP Parameters:
n_neighbors: {params['n_neighbors']}
n_components: {params['n_components']}
min_dist: {params['min_dist']}
metric: {params['metric']}
build_algo: {params['build_algo']}
n_clusters: {params['build_kwds']['nnd_n_clusters']}
graph_degree: {params['build_kwds']['nnd_graph_degree']}
max_iterations: {params['build_kwds']['nnd_max_iterations']}

Data Statistics:
Total features: {len(embeddings):,}
Unique categories: {len(category_counts)}
Features with categories: {sum(1 for cat in categories if cat is not None):,}
"""
    else:
        info_text = f"""Standard UMAP Parameters:
n_neighbors: {params['n_neighbors']}
n_components: {params['n_components']}
min_dist: {params['min_dist']}
metric: {params['metric']}

Data Statistics:
Total features: {len(embeddings):,}
Unique categories: {len(category_counts)}
Features with categories: {sum(1 for cat in categories if cat is not None):,}
"""
    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'umap_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved UMAP visualization to: {plot_path}")
    
    # Save embeddings and metadata
    np.save(output_dir / 'umap_embeddings.npy', embeddings)
    
    # Save category information
    category_info = {
        'categories': categories,
        'category_counts': dict(category_counts),
        'umap_params': params
    }
    with open(output_dir / 'umap_metadata.json', 'w') as f:
        json.dump(category_info, f, indent=2)
    
    print(f"Saved embeddings to: {output_dir / 'umap_embeddings.npy'}")
    print(f"Saved metadata to: {output_dir / 'umap_metadata.json'}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='UMAP analysis of extracted DinoV3 features')
    parser.add_argument('--features_dir', type=str, required=True,
                       help='Path to directory containing extracted features')
    parser.add_argument('--output_dir', type=str, default='umap_results',
                       help='Output directory for UMAP results')
    parser.add_argument('--n_neighbors', type=int, default=None,
                       help='Override n_neighbors parameter (auto-calculated if not provided)')
    parser.add_argument('--min_dist', type=float, default=0.1,
                       help='UMAP min_dist parameter')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'manhattan', 'hamming'],
                       help='UMAP distance metric')
    
    # cuML-specific parameters
    parser.add_argument('--build_algo', type=str, default='auto',
                       choices=['auto', 'brute_force_knn', 'nn_descent'],
                       help='Build algorithm for cuML UMAP (auto, brute_force_knn, nn_descent)')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='Number of clusters for batching (auto-calculated if not provided)')
    parser.add_argument('--graph_degree', type=int, default=32,
                       help='Graph degree for nn-descent algorithm')
    parser.add_argument('--max_iterations', type=int, default=20,
                       help='Maximum iterations for nn-descent')
    
    args = parser.parse_args()
    
    print("=== DinoV3 UMAP Analysis ===")
    print(f"Features directory: {args.features_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Load extracted features and metadata
    all_features, all_metadata, summary = load_extracted_features(args.features_dir)
    
    if not all_features:
        print("No features found! Exiting.")
        return
    
    # Prepare data for UMAP
    feature_matrix, categories, feature_info = prepare_umap_data(all_features, all_metadata)
    
    # Calculate UMAP parameters
    params, category_counts = calculate_umap_parameters(
        categories, feature_matrix, 
        build_algo=args.build_algo, 
        n_clusters=args.n_clusters
    )
    
    # Override parameters if provided
    if args.n_neighbors is not None:
        params['n_neighbors'] = args.n_neighbors
    if args.min_dist is not None:
        params['min_dist'] = args.min_dist
    if args.metric is not None:
        params['metric'] = args.metric
    
    # Override cuML-specific parameters if provided
    if USE_CUML and 'build_kwds' in params:
        if args.graph_degree is not None:
            params['build_kwds']['nnd_graph_degree'] = args.graph_degree
        if args.max_iterations is not None:
            params['build_kwds']['nnd_max_iterations'] = args.max_iterations
    
    # Run UMAP analysis
    embeddings = run_umap_analysis(feature_matrix, params, args.output_dir)
    
    # Create visualizations
    visualize_umap_results(embeddings, categories, category_counts, params, args.output_dir)
    
    print(f"\n=== UMAP Analysis Complete ===")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()