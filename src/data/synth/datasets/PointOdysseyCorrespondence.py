"""
PointOdyssey dataset wrapper for correspondence and optical flow.

This module provides a wrapper around the PointOdyssey dataset that returns
data in a format suitable for correspondence learning: src, trg, and flow.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as _random
import json
import hashlib
import threading
import time

# Add the project root to sys.path so models.CATs_PlusPlus can be imported
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add the pips2 path to sys.path so the utils can be found
pips2_path = Path(__file__).parent / "pips2"
sys.path.insert(0, str(pips2_path))

# Import the base dataset and utils
from src.data.synth.datasets.pips2.datasets.pointodysseydataset import PointOdysseyDataset as BasePointOdysseyDataset


class PointOdysseyFlowDataset(torch.utils.data.Dataset):
    """
    Wrapper for PointOdyssey dataset that returns correspondence data.
    
    Returns:
        - src: Source image (first frame)
        - trg: Target image (second frame) 
        - flow: Optical flow from trg to src (dx, dy)
    """
    
    def __init__(self, 
                 dataset_location: str = '/home/spencer/Data/sample',
                 dset: str = 'train',
                 use_augs: bool = False,
                 S: int = 8,
                 N: int = 32,
                 strides: list = [1, 2, 4,],
                 clip_step: int = 2,
                 resize_size: tuple = (368+64, 496+64),
                 crop_size: tuple = (368, 496),
                 req_full: bool = False,
                 quick: bool = False,
                 verbose: bool = False,
                 filter_instances: bool = False,
                 reverse_flow: bool = True,
                 downsample_for_cats: bool = False,
                 cats_feat_size: int = 32,
                 all_points: bool = False,
                 max_sequences: Optional[int] = None,
                 max_pts: int = 40,
                 thres: str = 'img',
                 normalize_images: bool = False,
                 normalize: bool = True,
                 val_sequence_fraction: Optional[float] = None):
        """
        Initialize the PointOdyssey flow dataset.
        
        Args:
            dataset_location: Path to PointOdyssey dataset
            dset: Dataset split ('train', 'val', 'test')
            use_augs: Whether to use data augmentations
            S: Number of frames per sequence
            N: Number of points to track
            strides: Frame strides for sampling
            clip_step: Step size for clip sampling
            resize_size: Size to resize images to
            crop_size: Size to crop images to
            req_full: Whether to require full sequences
            quick: Whether to use quick mode (fewer samples)
            verbose: Whether to print verbose information
            filter_instances: Whether to filter instances
            reverse_flow: Whether to reverse flow direction
            downsample_for_cats: Whether to downsample flow for CATs (training mode)
            cats_feat_size: Feature size for downsampled flow
            all_points: Whether to use all points
            max_sequences: Maximum number of sequences to use (None = all, deterministic sampling)
            max_pts: Maximum number of keypoints (default: 40). Padded keypoints use (0, 0) so flow is (0, 0) and doesn't affect metrics.
            thres: PCK threshold type ('img' or 'bbox')
            normalize_images: If True, enables validation mode and returns keypoints-based format for evaluation
            normalize: If True, applies ImageNet normalization to images (default: True, model expects normalized images)
        """
        # Check if the dataset has the expected structure (with train/val/test subdirs)
        expected_dset_path = os.path.join(dataset_location, dset)
        if not os.path.exists(expected_dset_path):
            # If no train/val/test subdirs, assume sequences are directly in dataset_location
            print(f"Warning: No '{dset}' subdirectory found in {dataset_location}")
            print("Assuming sequences are directly in the dataset location")
            # Create a temporary structure by pointing to the parent directory
            actual_dataset_location = os.path.dirname(dataset_location)
            actual_dset = os.path.basename(dataset_location)
        else:
            actual_dataset_location = dataset_location
            actual_dset = dset
            
        self.base_dataset = BasePointOdysseyDataset(
            dataset_location=actual_dataset_location,
            dset=actual_dset,
            use_augs=use_augs,
            S=S,
            N=N,
            strides=strides,
            clip_step=clip_step,
            resize_size=resize_size,
            crop_size=crop_size,
            req_full=req_full,
            quick=quick,
            max_sequences=max_sequences,
            verbose=verbose,
            all_points=all_points,
            val_sequence_fraction=val_sequence_fraction
        )
        
        self.S = S
        self.N = N
        self.filter_instances = filter_instances
        self.downsample_for_cats = downsample_for_cats
        self.cats_feat_size = cats_feat_size
        self.verbose = verbose
        self.reverse_flow = reverse_flow
        self.max_pts = max_pts
        self.thres = thres
        self.normalize_images = normalize_images
        self.normalize = normalize
        # Device management - defaults to CPU
        self._device = torch.device('cpu')
        
        # Initialize KeypointToFlow converter only when downsample_for_cats is True
        # This replaces manual flow calculation for consistency with other datasets
        self.kps_to_flow = None
        if downsample_for_cats:
            try:
                from models.CATs_PlusPlus.data.keypoint_to_flow import KeypointToFlow
                # Get image size from crop_size (final size after processing)
                img_size = crop_size[0] if isinstance(crop_size, tuple) else crop_size
                self.kps_to_flow = KeypointToFlow(
                    receptive_field_size=35,
                    jsz=img_size // cats_feat_size,
                    feat_size=cats_feat_size,
                    img_size=img_size
                )
            except ImportError:
                self.kps_to_flow = None
        
        # Cache management for valid/invalid indices
        self.cache_dir = os.path.join(actual_dataset_location, '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create a unique hash for this dataset configuration
        config_str = json.dumps({
            'dataset_location': actual_dataset_location,
            'dset': actual_dset,
            'S': S,
            'N': N,
            'strides': sorted(strides),
            'clip_step': clip_step,
            'resize_size': resize_size,
            'crop_size': crop_size,
            'req_full': req_full,
            'all_points': all_points,
            'max_sequences': max_sequences,
            'val_sequence_fraction': val_sequence_fraction,
        }, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]  # Use first 8 chars for brevity
        
        # Create a more readable filename with key parameters
        strides_str = '_'.join(map(str, sorted(strides)))
        cache_name = f'valid_indices_{actual_dset}_S{S}_N{N}_strides{strides_str}_{config_hash}.json'
        self.cache_file = os.path.join(self.cache_dir, cache_name)
        
        # Thread-safe data structures
        self._cache_lock = threading.Lock()
        self._valid_indices = set()
        self._use_worker_temp_files = False  # If True, workers save to persistent temp files instead of merging to main cache
        self._worker_id = None  # Will be set by worker_init_fn for parallel processing
        self._invalid_indices = set()
        self._cache_save_interval = 1000  # Save interval for precomputation (per-worker)
        self._base_dataset_len = None  # Cache base dataset length to avoid repeated calls
        self._worker_caches = None  # Will be initialized by initialize_worker_caches() before threading
        
        # Load existing cache
        self._load_cache()
        
        # Check if cache exists and enable read-only mode if it does
        self._cache_readonly = False
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    existing_valid = set(cache_data.get('valid', []))
                    existing_invalid = set(cache_data.get('invalid', []))
                    total_cached = len(existing_valid) + len(existing_invalid)
                    total_samples = len(self.base_dataset)
                    
                    # If cache exists and covers >50% of dataset, make it read-only
                    # (allows some discovery but prevents most writes)
                    if total_cached > 0.5 * total_samples:
                        self._cache_readonly = True
                        print(f"Cache exists ({total_cached}/{total_samples} indices), enabling read-only mode")
            except Exception as e:
                if self.verbose:
                    print(f"Could not check cache completeness: {e}")
        else:
            print(f"No cache found at {self.cache_file}, starting fresh")
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        If cache exists and has valid indices, return count of valid indices only.
        Otherwise, return full dataset length for discovery.
        """
        with self._cache_lock:
            if self._valid_indices and not self._use_worker_temp_files:
                # Cache exists and we're not in precomputation mode - only use valid indices
                return len(self._valid_indices)
        return len(self.base_dataset)
    
    def _load_cache(self):
        """Load validation cache from disk (called before threading starts)."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    self._valid_indices = set(cache_data.get('valid', []))
                    self._invalid_indices = set(cache_data.get('invalid', []))
                if self.verbose:
                    print(f"Loaded cache from {self.cache_file}: {len(self._valid_indices)} valid, {len(self._invalid_indices)} invalid indices")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load cache: {e}")
                self._valid_indices = set()
                self._invalid_indices = set()
        else:
            if self.verbose:
                print(f"No existing cache found at {self.cache_file}, starting fresh")
    
    def _is_index_valid(self, index):
        """Thread-safe check if index is not in invalid set."""
        with self._cache_lock:
            return index not in self._invalid_indices
    
    def _is_index_known_valid(self, index):
        """Thread-safe check if index is in valid set."""
        with self._cache_lock:
            return index in self._valid_indices
    
    
    def _get_random_valid_index(self):
        """Thread-safe get random valid index."""
        with self._cache_lock:
            if self._valid_indices:
                return _random.choice(list(self._valid_indices))
            return None
    
    def _get_first_valid_index(self):
        """Thread-safe get first valid index (faster than random)."""
        with self._cache_lock:
            if self._valid_indices:
                return min(self._valid_indices)  # Get first (smallest) valid index
            return None
    
    def _get_valid_index_at_position(self, position):
        """Get valid index at given position in sorted valid indices list."""
        with self._cache_lock:
            if self._valid_indices:
                sorted_valid = sorted(self._valid_indices)
                if position < len(sorted_valid):
                    return sorted_valid[position]
            return None
    
    def initialize_worker_caches(self, num_workers: int):
        """Initialize per-worker cache storage before starting threads (call this once before threading)."""
        if self._worker_caches is None:
            self._worker_caches = {}
        for worker_id in range(num_workers):
            if worker_id not in self._worker_caches:
                self._worker_caches[worker_id] = {
                    'valid': set(),
                    'invalid': set(),
                    'updates': 0
                }
        # Cache base dataset length once to avoid repeated calls
        if self._base_dataset_len is None:
            self._base_dataset_len = len(self.base_dataset)
    
    def _save_worker_cache(self, worker_id, worker_cache):
        """Save a worker's local cache to its own file (async, no locking needed - each worker has its own file)."""
        if self._cache_readonly:
            return
        
        try:
            cache_data = {
                'valid': sorted(list(worker_cache['valid'])),
                'invalid': sorted(list(worker_cache['invalid'])),
                'timestamp': time.time(),
                'total_samples': self._base_dataset_len if self._base_dataset_len is not None else len(self.base_dataset)
            }
            worker_file = self.cache_file + f'.worker_{worker_id}.json'
            with open(worker_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                f.flush()
            # Disable verbose printing during precomputation to avoid stdout contention
            # if self.verbose:
            #     print(f"Worker {worker_id} saved cache: {len(worker_cache['valid'])} valid, {len(worker_cache['invalid'])} invalid indices")
        except Exception as e:
            # Disable verbose printing during precomputation to avoid stdout contention
            # if self.verbose:
            #     print(f"Worker {worker_id} failed to save cache: {e}")
            pass
    
    def merge_worker_temp_files(self):
        """Merge all worker temp files into the main cache file (blocking - waits for all workers to finish)."""
        import glob
        import os
        
        # First, save any remaining in-memory worker caches to their files
        if self._worker_caches is not None:
            for worker_id, worker_cache in self._worker_caches.items():
                if worker_cache['updates'] > 0:
                    # Save any remaining updates
                    self._save_worker_cache(worker_id, worker_cache)
        
        # Find all worker files
        worker_pattern = self.cache_file + '.worker_*.json'
        temp_pattern = self.cache_file + '.tmp.*'
        worker_files = glob.glob(worker_pattern)
        temp_files = glob.glob(temp_pattern)
        all_files = worker_files + temp_files
        
        if not all_files:
            if self.verbose:
                print(f"No worker files found to merge (patterns: {worker_pattern}, {temp_pattern})")
            # Still merge in-memory caches if they exist
            if self._worker_caches is not None:
                all_valid = set()
                all_invalid = set()
                for worker_id, worker_cache in self._worker_caches.items():
                    all_valid.update(worker_cache['valid'])
                    all_invalid.update(worker_cache['invalid'])
                
                # Save merged cache
                self._save_merged_cache(all_valid, all_invalid)
            return
        
        if self.verbose:
            print(f"Found {len(all_files)} worker files to merge ({len(worker_files)} worker files, {len(temp_files)} temp files)")
        
        # Collect all valid/invalid indices from all worker files
        all_valid = set()
        all_invalid = set()
        total_samples = None
        
        for temp_file in all_files:
            try:
                with open(temp_file, 'r') as f:
                    cache_data = json.load(f)
                    all_valid.update(cache_data.get('valid', []))
                    all_invalid.update(cache_data.get('invalid', []))
                    if total_samples is None:
                        total_samples = cache_data.get('total_samples')
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load temp file {temp_file}: {e}")
        
        # Also merge in-memory worker caches
        if self._worker_caches is not None:
            for worker_id, worker_cache in self._worker_caches.items():
                all_valid.update(worker_cache['valid'])
                all_invalid.update(worker_cache['invalid'])
        
        # Merge with existing cache file if it exists
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    existing_cache = json.load(f)
                    all_valid.update(existing_cache.get('valid', []))
                    all_invalid.update(existing_cache.get('invalid', []))
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load existing cache for merge: {e}")
        
        # Save merged cache
        self._save_merged_cache(all_valid, all_invalid, total_samples)
        
        # Clean up worker files
        for temp_file in all_files:
            try:
                os.remove(temp_file)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to remove temp file {temp_file}: {e}")
    
    def _save_merged_cache(self, all_valid, all_invalid, total_samples=None):
        """Save the merged cache to the main cache file."""
        valid_list = sorted(list(all_valid))
        invalid_list = sorted(list(all_invalid))
        timestamp = time.time()
        
        try:
            cache_data = {
                'valid': valid_list,
                'invalid': invalid_list,
                'timestamp': timestamp,
                'total_samples': total_samples if total_samples is not None else (self._base_dataset_len if self._base_dataset_len is not None else len(self.base_dataset))
            }
            
            # Write to final cache file
            final_temp = self.cache_file + '.final_merge.tmp'
            with open(final_temp, 'w') as f:
                json.dump(cache_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            # Atomic rename
            os.replace(final_temp, self.cache_file)
            
            # Update in-memory state (only after successful save)
            with self._cache_lock:
                self._valid_indices = all_valid
                self._invalid_indices = all_invalid
            
            if self.verbose:
                print(f"Merged cache saved to {self.cache_file}: {len(valid_list)} valid, {len(invalid_list)} invalid indices")
            
        except Exception as e:
            if self.verbose:
                print(f"Failed to save merged cache: {e}")
            # Clean up final temp if it exists
            try:
                final_temp = self.cache_file + '.final_merge.tmp'
                if os.path.exists(final_temp):
                    os.remove(final_temp)
            except:
                pass
    
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset (training/validation mode - READ ONLY).
        Uses cache if available, otherwise does resampling with retries.
        NEVER writes to cache - this is for training/validation only.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing:
                - 'src_img': Source image tensor (C, H, W)
                - 'trg_img': Target image tensor (C, H, W) 
                - 'flow': Flow tensor (2, H, W) from trg to src
        """
        sample = None
        gotit = False
        
        # Check if cache exists and map index to valid index
        with self._cache_lock:
            has_valid_cache = len(self._valid_indices) > 0
        
        if has_valid_cache:
            # Cache exists - map requested index to actual valid index
            actual_index = self._get_valid_index_at_position(index)
            if actual_index is not None:
                # Use the mapped valid index
                sample, gotit = self.base_dataset[actual_index]
                if not gotit:
                    # Unexpected - but don't modify cache, just try resampling
                    gotit = False
        
        if not has_valid_cache or not gotit:
            # No cache yet or cache lookup failed - use resampling logic (but don't write to cache)
            # Try the requested index first if not known to be invalid
            if self._is_index_valid(index):
                # Check if it's known-valid - if so, use it directly (fast path)
                if self._is_index_known_valid(index):
                    # Already know it's valid, use it directly
                    sample, gotit = self.base_dataset[index]
                else:
                    # Not known yet, try it (but don't write to cache)
                    sample, gotit = self.base_dataset[index]
            else:
                # Known-invalid index requested - use a known-valid one if available
                valid_idx = self._get_random_valid_index()
                if valid_idx is not None:
                    index = valid_idx
                    sample, gotit = self.base_dataset[index]
            
            # If invalid, try to find a valid sample (resampling loop, but don't write to cache)
            attempts = 0
            start_time = time.time()
            max_attempts = 100
            resample_timeout = 5.0
            
            while not gotit and attempts < max_attempts:
                # Timeout check - use cached valid index if available
                if time.time() - start_time > resample_timeout:
                    valid_idx = self._get_random_valid_index()
                    if valid_idx is not None:
                        if self.verbose:
                            print(f"Timeout after {resample_timeout}s, using random known-valid index")
                        index = valid_idx
                        sample, gotit = self.base_dataset[index]
                        if gotit:
                            break
                
                # Prefer known-valid indices
                valid_idx = self._get_random_valid_index()
                if valid_idx is not None:
                    index = valid_idx
                else:
                    # Try sequential indices near the original first
                    if attempts < 10:
                        offset = (attempts // 2 + 1) * (1 if attempts % 2 == 0 else -1)
                        index = (index + offset) % len(self.base_dataset)
                    else:
                        # Fall back to random
                        index = _random.randint(0, len(self.base_dataset) - 1)
                
                # Skip known-invalid indices
                if not self._is_index_valid(index):
                    attempts += 1
                    continue
                    
                sample, gotit = self.base_dataset[index]
                attempts += 1
        
        if not gotit:
            raise RuntimeError(f"Failed to get valid sample after {attempts} attempts")
        
        # Keep everything on CPU - DataLoader will handle GPU transfer
        # Extract the data we need (keep on CPU)
        rgbs = sample['rgbs']  # (S, C, H, W) - keep on CPU
        trajs = sample['trajs']  # (S, N, 2) - keep on CPU
        visibs = sample['visibs']  # (S, N) - keep on CPU
        valids = sample['valids']  # (S, N) - keep on CPU
        masks = sample['masks']  # (S, 1, H, W) - keep on CPU

        # Sample two non-repeating frames (order doesn't matter)
        i, j = 0, self.S - 1

        src_img = rgbs[i]
        trg_img = rgbs[j]
        
        # Convert images to float32 in [0, 1] range (on CPU)
        src_img = src_img.to(torch.float32) / 255.0
        trg_img = trg_img.to(torch.float32) / 255.0
        
        # Clamp to ensure valid [0, 1] range before normalization
        src_img = torch.clamp(src_img, 0.0, 1.0)
        trg_img = torch.clamp(trg_img, 0.0, 1.0)

        # Trajectories and flags for those two frames
        src_trajs = trajs[i]  # (N, 2)
        trg_trajs = trajs[j]  # (N, 2)
        src_vis = visibs[i]
        trg_vis = visibs[j]
        src_valid = valids[i]
        trg_valid = valids[j]
        src_mask = masks[i]
        trg_mask = masks[j]

        # Normalize images if requested (model expects ImageNet normalization)
        # ImageNet normalization: (img - mean) / std produces "dark and crunchy" appearance
        if self.normalize:
            from torchvision.transforms.functional import normalize
            src_img = normalize(src_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            trg_img = normalize(trg_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Extract valid keypoints from trajectories
        valid_points = (src_vis > 0) & (trg_vis > 0) & (src_valid > 0) & (trg_valid > 0)
        
        if valid_points.any():
            valid_src_trajs = src_trajs[valid_points]  # (M, 2)
            valid_trg_trajs = trg_trajs[valid_points]  # (M, 2)
            n_valid = valid_src_trajs.shape[0]
            
            # Convert to [2, M] format (x, y coordinates)
            src_kps = valid_src_trajs.t()  # [2, M]
            trg_kps = valid_trg_trajs.t()  # [2, M]
        else:
            # No valid points - use dummy keypoints
            n_valid = 0
            src_kps = torch.zeros((2, 0), dtype=torch.float32)
            trg_kps = torch.zeros((2, 0), dtype=torch.float32)

        # Pad/truncate keypoints to max_pts
        # Use (0, 0) for padding so flow is (0, 0) and doesn't affect loss/metrics
        if n_valid < self.max_pts:
            pad_size = self.max_pts - n_valid
            # Pad with (0, 0) so flow will be (0, 0) for padded points
            src_kps = torch.cat([src_kps, torch.zeros(2, pad_size, dtype=torch.float32)], dim=1)
            trg_kps = torch.cat([trg_kps, torch.zeros(2, pad_size, dtype=torch.float32)], dim=1)
        elif n_valid > self.max_pts:
            # Truncate to max_pts (use first max_pts keypoints)
            src_kps = src_kps[:, :self.max_pts]
            trg_kps = trg_kps[:, :self.max_pts]
            n_valid = self.max_pts

        # Calculate flow based on downsample_for_cats flag
        if not self.downsample_for_cats:
            # Use full resolution flow (manual calculation only)
            flow_full = self._create_flow_field(
                src_trajs, trg_trajs, src_vis, trg_vis, src_valid, trg_valid,
                src_img.shape, src_mask, trg_mask, self.filter_instances, torch.device('cpu')
            )
            flow_downsampled = flow_full
        else:
            # downsample_for_cats is True: try kps_to_flow first, fallback to manual downsampling
            if self.kps_to_flow is not None and n_valid > 0:
                try:
                    # Use KeypointToFlow for downsampled flow (matches other datasets like SPair, PFPascal)
                    batch_for_flow = {
                        'src_kps': src_kps,  # [2, max_pts]
                        'trg_kps': trg_kps,  # [2, max_pts]
                        'n_pts': torch.tensor(n_valid)
                    }
                    flow_downsampled = self.kps_to_flow(batch_for_flow)  # [2, feature_size, feature_size]
                except Exception as e:
                    # If kps_to_flow fails, fall back to manual downsampling
                    if self.verbose:
                        print(f"Warning: kps_to_flow failed ({e}), falling back to manual downsampling")
                    flow_full = self._create_flow_field(
                        src_trajs, trg_trajs, src_vis, trg_vis, src_valid, trg_valid,
                        src_img.shape, src_mask, trg_mask, self.filter_instances, torch.device('cpu')
                    )
                    flow_downsampled = self._downsample_flow_for_cats(flow_full, self.cats_feat_size)
            else:
                # Fallback: create full flow and downsample (when kps_to_flow is not available or no valid points)
                flow_full = self._create_flow_field(
                    src_trajs, trg_trajs, src_vis, trg_vis, src_valid, trg_valid,
                    src_img.shape, src_mask, trg_mask, self.filter_instances, torch.device('cpu')
                )
                flow_downsampled = self._downsample_flow_for_cats(flow_full, self.cats_feat_size)

        # Get image size (images are in CHW format)
        if src_img.ndim == 3:
            C, H, W = src_img.shape
            img_size_tuple = (H, W)
        else:
            H, W = src_img.shape[-2:]
            img_size_tuple = (H, W)

        # Get PCK threshold
        if self.thres == 'img':
            pckthres = torch.tensor(max(H, W), dtype=torch.float32)
        else:
            # Default to image size
            pckthres = torch.tensor(max(H, W), dtype=torch.float32)

        # Build output dictionary
        if self.normalize_images:
            # Validation format (matching TSSDataset and other evaluation datasets)
            out = {
                'src_img': src_img,
                'trg_img': trg_img,
                'flow': flow_downsampled,  # Downsampled flow [2, feature_size, feature_size]
                'src_kps': src_kps,  # [2, max_pts]
                'trg_kps': trg_kps,  # [2, max_pts]
                'n_pts': torch.tensor(n_valid),
                'pckthres': pckthres,
                'src_imsize': img_size_tuple,
                'trg_imsize': img_size_tuple,
                'datalen': len(self),
            }
        else:
            # Training format (original)
            out = {
                'src_img': src_img,
                'trg_img': trg_img,
                'flow': flow_downsampled,
                'masks': masks
            }

        # All tensors are already on CPU, no need to move them
        return out
    
    def __getitem_precompute__(self, index: int, worker_id: int = None):
        """
        Simple precomputation version - just try index, mark valid/invalid, raise if invalid.
        This is the ONLY function that writes to cache.
        Each worker maintains its own local valid/invalid sets (no locking needed).
        
        Args:
            index: Sample index to check
            worker_id: Optional worker ID for thread-safe cache saving (used for worker temp files)
            
        Returns:
            Sample dict if valid
            
        Raises:
            RuntimeError: If index is invalid (expected during precomputation)
        """
        # Get this worker's local cache (assumes _worker_caches was initialized before threading)
        if worker_id is not None:
            if self._worker_caches is None:
                raise RuntimeError(f"Worker caches not initialized. Call dataset.initialize_worker_caches(num_workers) before starting threads.")
            worker_cache = self._worker_caches.get(worker_id)
            if worker_cache is None:
                raise RuntimeError(f"Worker {worker_id} cache not initialized. Call dataset.initialize_worker_caches(num_workers) with num_workers > {worker_id}.")
        else:
            # Fallback: use shared cache (shouldn't happen in precompute mode)
            worker_cache = None
        
        # Try the requested index
        if worker_cache is None or index not in worker_cache['invalid']:
            sample, gotit = self.base_dataset[index]
            if gotit:
                # Valid - add to worker's local cache (no lock needed!)
                if worker_cache is not None:
                    if index not in worker_cache['valid']:
                        worker_cache['valid'].add(index)
                        worker_cache['updates'] += 1
                        
                        # Periodically save worker's local cache to its own file (async, no lock)
                        if worker_cache['updates'] >= self._cache_save_interval:
                            self._save_worker_cache(worker_id, worker_cache)
                            worker_cache['updates'] = 0
                
                return sample
            else:
                # Invalid - add to worker's local cache (no lock needed!)
                if worker_cache is not None:
                    if index not in worker_cache['invalid']:
                        worker_cache['invalid'].add(index)
                        worker_cache['updates'] += 1
                        
                        # Periodically save worker's local cache to its own file (async, no lock)
                        if worker_cache['updates'] >= self._cache_save_interval:
                            self._save_worker_cache(worker_id, worker_cache)
                            worker_cache['updates'] = 0
                
                raise RuntimeError(f"Index {index} is invalid (precomputation mode)")
        else:
            # Already known invalid in this worker's cache
            raise RuntimeError(f"Index {index} is known invalid (precomputation mode)")
    
    def _create_flow_field(self, 
                          src_trajs: torch.Tensor, 
                          trg_trajs: torch.Tensor,
                          src_vis: torch.Tensor,
                          trg_vis: torch.Tensor, 
                          src_valid: torch.Tensor,
                          trg_valid: torch.Tensor,
                          img_shape: Tuple[int, int, int],
                          src_mask: torch.Tensor,
                          trg_mask: torch.Tensor,
                          filter_instances: bool,
                          device: torch.device) -> torch.Tensor:
        """
        Create a flow field from target to source.
        
        Args:
            src_trajs: Source frame trajectories (N, 2)
            trg_trajs: Target frame trajectories (N, 2)
            src_vis: Source frame visibility (N,)
            trg_vis: Target frame visibility (N,)
            src_valid: Source frame validity (N,)
            trg_valid: Target frame validity (N,)
            img_shape: Image shape (C, H, W)
            src_mask: Source instance mask (C, H, W)
            trg_mask: Target instance mask (C, H, W)
        Returns:
            Flow field tensor (2, H, W) where flow[0] = dx, flow[1] = dy from trg to src
        """
        C, H, W = img_shape
        
        # Initialize flow field with inf (invalid) on the specified device
        # Flow format: (2, H, W) where flow[0] = dx, flow[1] = dy
        # Invalid pixels start as inf, which will be converted to 0 by downsampler if needed
        # This allows proper handling of sparse flow fields
        flow = torch.full((2, H, W), float('inf'), dtype=torch.float32, device=device)
        
        # Ensure masks are (H, W) instance id maps
        if src_mask.ndim == 3:
            src_mask = src_mask.squeeze(0)
        if trg_mask.ndim == 3:
            trg_mask = trg_mask.squeeze(0)
        
        # Find points that are visible and valid in both frames
        valid_points = (src_vis > 0) & (trg_vis > 0) & (src_valid > 0) & (trg_valid > 0)
        
        if not valid_points.any():
            return flow
        
        # Get valid trajectories
        valid_src_trajs = src_trajs[valid_points]  # (M, 2) in pixel coords
        valid_trg_trajs = trg_trajs[valid_points]  # (M, 2)

        # Displacement from trg to src: [dx, dy]
        flow_vectors = valid_src_trajs - valid_trg_trajs  # (M, 2)

        # Round to nearest integer pixel positions for placement
        x_t = torch.round(valid_trg_trajs[:, 0]).long()
        y_t = torch.round(valid_trg_trajs[:, 1]).long()

        # In-bounds mask
        in_bounds = (x_t >= 0) & (x_t < W) & (y_t >= 0) & (y_t < H)

        if in_bounds.any():
            x_ib = x_t[in_bounds]
            y_ib = y_t[in_bounds]
            flow_ib = flow_vectors[in_bounds]  # (M, 2)

            if filter_instances:
                # Compute background (0) and floor/max id filtering
                max_id = torch.max(torch.max(src_mask), torch.max(trg_mask)).item()
                src_ok = (src_mask[y_ib, x_ib] != 0) & (src_mask[y_ib, x_ib] != max_id)
                trg_ok = (trg_mask[y_ib, x_ib] != 0) & (trg_mask[y_ib, x_ib] != max_id)
                keep = src_ok & trg_ok
                if keep.any():
                    # Assign flow vectors: flow[:, y, x] = [dx, dy] for (2, H, W) format
                    flow[:, y_ib[keep], x_ib[keep]] = flow_ib[keep].T  # flow_ib[keep] is (K, 2), .T makes (2, K)
            else:
                # Assign flow vectors: flow[:, y, x] = [dx, dy] for (2, H, W) format
                flow[:, y_ib, x_ib] = flow_ib.T  # flow_ib is (M, 2), .T makes (2, M)
        
        return flow

    def _downsample_flow_for_cats(self, flow: torch.Tensor, feat_size: int) -> torch.Tensor:
        """Downsample (2, H, W) flow to (2, feat_size, feat_size) normalized to feature grid units.
        Only invalid (inf) values are excluded from averaging. Zero vectors are valid and included.
        Flow values are normalized to feature grid units to match CATS convention:
        - A flow of 1.0 = one feature grid cell = (H // feat_size) pixels
        - This matches how CATS stores flow from keypoint annotations.
        """
        if flow is None:
            return flow
        _, H, W = flow.shape  # flow is (2, H, W)
        flow_batch = flow.unsqueeze(0)  # (1, 2, H, W)

        # Valid mask: only exclude infs, zeros are valid (represent zero motion)
        # Check for finite values (includes zeros, excludes infs)
        valid_mask = torch.isfinite(flow_batch).all(dim=1, keepdim=True)  # (1, 1, H, W)

        # Set invalid values to zero for pooling (temporary, won't affect masked average)
        flow_for_pool = flow_batch.clone()
        flow_for_pool[~valid_mask.expand_as(flow_for_pool)] = 0
        
        # Calculate scale factors for converting averages to sums
        scale_factor_h = H / feat_size
        scale_factor_w = W / feat_size
        
        # Sum of valid flow values in each pooling region
        flow_sum = torch.nn.functional.adaptive_avg_pool2d(
            flow_for_pool, (feat_size, feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get sum
        
        # Count of valid pixels in each pooling region
        valid_count = torch.nn.functional.adaptive_avg_pool2d(
            valid_mask.float(), (feat_size, feat_size)
        ) * (scale_factor_h * scale_factor_w)  # Multiply back to get count
        
        # Compute masked average: divide sum by count of valid pixels (not total pixels)
        # Avoid division by zero for regions with no valid pixels
        valid_count_safe = torch.clamp(valid_count, min=1e-8)
        flow_ds = flow_sum / valid_count_safe

        # Normalize flow to feature grid units to match CATS convention
        # CATS expects flow in feature grid units, not pixel space
        # A flow of 1.0 = one feature grid cell = (H // feat_size) pixels
        # Use the same dimension for both x and y to match other datasets (FlyingThings, SPair, etc.)
        # For square images (which PointOdyssey uses), H == W, so this is consistent
        downsampling_factor = H // feat_size
        flow_ds = flow_ds / downsampling_factor

        # Mark regions with no valid pixels as invalid (set to 0)
        # For sparse flow (like PointOdyssey keypoints), use a very low threshold
        # to preserve patches that have any valid flow, even if sparse
        # Threshold of > 0 means we keep any patch with at least some valid pixels
        valid_mask_downsampled = valid_count > 1e-6  # Keep patches with any valid pixels
        flow_ds[~valid_mask_downsampled.expand_as(flow_ds)] = 0

        return flow_ds.squeeze(0)  # (2, feat_size, feat_size)


    # ---- PyTorch-friendly device API ----
    @property
    def device(self):
        return self._device

    def to(self, device):
        new_device = torch.device(device)
        if self._device != new_device:
            self._device = new_device
        return self

    def cuda(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                new_device = torch.device(f'cuda:{idx}')
            else:
                new_device = torch.device('cuda')  # will error on use if unavailable
        else:
            if isinstance(device, int):
                new_device = torch.device(f'cuda:{device}')
            else:
                new_device = torch.device(device)
        if self._device != new_device:
            self._device = new_device
        return self

    def cpu(self):
        new_device = torch.device('cpu')
        if self._device != new_device:
            self._device = new_device
        return self
    
    def _resize_sample(self, src_img: torch.Tensor, trg_img: torch.Tensor, 
                      flow: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Resize images and flow to target size.
        
        Args:
            src_img: Source image (C, H, W)
            trg_img: Target image (C, H, W)
            flow: Flow field (H, W, 2)
            target_size: Target size (H, W)
            
        Returns:
            Resized src_img, trg_img, flow
        """
        target_h, target_w = size, size
        
        # Resize images
        src_img_resized = torch.nn.functional.interpolate(
            src_img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        trg_img_resized = torch.nn.functional.interpolate(
            trg_img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        # Resize flow
        # Flow needs special handling - we need to scale the flow vectors by the resize factor
        orig_h, orig_w = flow.shape[:2]
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Resize flow field
        flow_resized = torch.nn.functional.interpolate(
            flow.permute(2, 0, 1).unsqueeze(0),  # (1, 2, H, W)
            size=(target_h, target_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0).permute(1, 2, 0)  # (H, W, 2)
        
        # Scale flow vectors by resize factors
        flow_resized[:, :, 0] *= scale_x  # x component
        flow_resized[:, :, 1] *= scale_y  # y component
        
        return src_img_resized, trg_img_resized, flow_resized
    
    def _create_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Create a dummy sample when the base dataset fails."""
        print("Creating dummy sample")
        if self.size is not None:
            H, W = self.size, self.size
        else:
            H, W = 368, 496
            
        return {
            'src_img': torch.zeros((3, H, W), dtype=torch.float32, device=self._device),
            'trg_img': torch.zeros((3, H, W), dtype=torch.float32, device=self._device),
            'flow': torch.zeros((2, H, W), dtype=torch.float32, device=self._device),
            'masks': torch.zeros((self.S, 1, H, W), dtype=torch.int64, device=self._device)
        }
    
    def visualize_masks(self, masks: torch.Tensor, save_path: str = "./debug/class_masks_visualization.png"):
        """
        Visualize instance masks with distinct colors for each instance ID.
        
        Args:
            masks: Instance masks tensor (S, 1, H, W) where values are instance IDs 0-k
            save_path: Path to save the visualization
        """
        S, C, H, W = masks.shape
        
        # Create a figure with subplots for each frame
        fig, axes = plt.subplots(2, (S + 1) // 2, figsize=(4 * ((S + 1) // 2), 8))
        if S == 1:
            axes = [axes]
        elif S <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # Get unique instance IDs across all frames
        all_instance_ids = torch.unique(masks).cpu().numpy()
        num_instances = len(all_instance_ids)
        
        # Create a colormap with distinct colors
        # Class 0 (background) gets bright red
        colors = ['red']  # Class 0 = bright red
        if num_instances > 1:
            # Generate distinct colors for other classes
            other_colors = plt.cm.tab20(np.linspace(0, 1, max(1, num_instances - 1)))
            colors.extend([other_colors[i] for i in range(num_instances - 1)])
        
        # Create a custom colormap
        cmap = mcolors.ListedColormap(colors)
        
        # Normalize instance IDs to [0, num_instances-1] for colormap
        id_to_index = {instance_id: i for i, instance_id in enumerate(all_instance_ids)}
        
        for s in range(S):
            mask_frame = masks[s, 0].cpu().numpy()  # (H, W)
            
            # Convert instance IDs to colormap indices
            mask_colored = np.zeros_like(mask_frame, dtype=float)
            for instance_id in all_instance_ids:
                mask_colored[mask_frame == instance_id] = id_to_index[instance_id]
            
            # Plot the mask
            im = axes[s].imshow(mask_colored, cmap=cmap, vmin=0, vmax=num_instances-1)
            axes[s].set_title(f'Frame {s} - Instance Masks')
            axes[s].axis('off')
        
        # Hide unused subplots
        for s in range(S, len(axes)):
            axes[s].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
        cbar.set_ticks(range(num_instances))
        cbar.set_ticklabels([f'ID {int(id)}' for id in all_instance_ids])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Mask visualization saved to: {save_path}")
        print(f"Found {num_instances} unique instance IDs: {all_instance_ids}")
        print(f"Class 0 (background) is colored bright red")
    
    def visualize_masks_batch(self, batch_masks: torch.Tensor, save_path: str = "./debug/class_masks_batch_visualization.png"):
        """
        Visualize instance masks for a batch of samples.
        
        Args:
            batch_masks: Batch of instance masks tensor (B, S, 1, H, W)
            save_path: Path to save the visualization
        """
        B, S, C, H, W = batch_masks.shape
        
        # Create a figure with subplots for each sample and frame
        fig, axes = plt.subplots(B, S, figsize=(4 * S, 4 * B))
        if B == 1:
            axes = axes.reshape(1, -1)
        if S == 1:
            axes = axes.reshape(-1, 1)
        
        # Get unique instance IDs across all samples and frames
        all_instance_ids = torch.unique(batch_masks).cpu().numpy()
        num_instances = len(all_instance_ids)
        
        # Create a colormap with distinct colors
        # Class 0 (background) gets bright red
        colors = ['red']  # Class 0 = bright red
        if num_instances > 1:
            # Generate distinct colors for other classes
            other_colors = plt.cm.tab20(np.linspace(0, 1, max(1, num_instances - 1)))
            colors.extend([other_colors[i] for i in range(num_instances - 1)])
        
        # Create a custom colormap
        cmap = mcolors.ListedColormap(colors)
        
        # Normalize instance IDs to [0, num_instances-1] for colormap
        id_to_index = {instance_id: i for i, instance_id in enumerate(all_instance_ids)}
        
        for b in range(B):
            for s in range(S):
                mask_frame = batch_masks[b, s, 0].cpu().numpy()  # (H, W)
                
                # Convert instance IDs to colormap indices
                mask_colored = np.zeros_like(mask_frame, dtype=float)
                for instance_id in all_instance_ids:
                    mask_colored[mask_frame == instance_id] = id_to_index[instance_id]
                
                # Plot the mask
                im = axes[b, s].imshow(mask_colored, cmap=cmap, vmin=0, vmax=num_instances-1)
                axes[b, s].set_title(f'Sample {b}, Frame {s}')
                axes[b, s].axis('off')
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=axes, shrink=0.8, aspect=20)
        cbar.set_ticks(range(num_instances))
        cbar.set_ticklabels([f'ID {int(id)}' for id in all_instance_ids])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Batch mask visualization saved to: {save_path}")
        print(f"Found {num_instances} unique instance IDs: {all_instance_ids}")
        print(f"Class 0 (background) is colored bright red")


def test_dataset_with_visualization(dataset_path: str = None, size: Optional[int] = None, downsample_for_cats: bool = False):
    from torch.utils.data import DataLoader
    """Test the dataset wrapper with visualization."""
    print("Testing PointOdyssey Flow Dataset with Visualization...")
    
    # Default dataset path if not provided
    if dataset_path is None:
        dataset_path = '/home/spencer/Data/sample'
        print(f"Using default dataset path: {dataset_path}")
        print("To specify a different path, use: python PointOdysseyCorrespondence.py --dataset_path /path/to/pointodyssey")
    else:
        print(f"Using dataset path: {dataset_path}")
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        print("Please provide a valid path to the PointOdyssey dataset using --dataset_path")
        return
    
    # The PointOdyssey dataset expects a 'train' subdirectory
    # If the dataset_path points directly to sequences, we need to adjust
    train_path = os.path.join(dataset_path, 'train')
    if not os.path.exists(train_path):
        print(f"Note: No 'train' subdirectory found. The dataset expects sequences to be in {train_path}")
        print("Creating a temporary train directory structure...")
        # We'll modify the dataset to look directly in the provided path
        actual_dataset_path = dataset_path
    else:
        actual_dataset_path = dataset_path
    
    # Create dataset
    dataset = PointOdysseyFlowDataset(
        dataset_location=dataset_path,
        dset='train',
        use_augs=False,
        S=4,
        N=64,
        quick=False,  # Use quick mode for testing
        verbose=True,
        filter_instances=True,
        resize_size=(size+64, size+64),
        crop_size=(size, size),
        all_points=downsample_for_cats,
        downsample_for_cats=downsample_for_cats,
        cats_feat_size=32,
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No samples in dataset")
        return
    
    # Collect samples for batch visualization
    batch_data = []
    batch_masks = []
    max_samples = 4
    
    for i in range(min(len(dataset), max_samples)):
        try:
            sample = dataset[i]
            
            src_img = sample['src_img']
            trg_img = sample['trg_img']
            
            batch_data.append({
                'src_img': src_img.unsqueeze(0),  # Add batch dimension
                'trg_img': trg_img.unsqueeze(0),  # Add batch dimension
                'flow': sample['flow'].unsqueeze(0)     # Add batch dimension
            })
            
            # Collect sequence masks for visualization (expects B x S x 1 x H x W)
            batch_masks.append(sample['masks'].unsqueeze(0))  # Add batch dimension
            
            print(f"Sample {i}:")
            print(f"  Source shape: {src_img.shape}")
            print(f"  Target shape: {trg_img.shape}")
            print(f"  Flow shape: {sample['flow'].shape}")
            
            # Debug image format
            print(f"  Source dtype: {src_img.dtype}")
            print(f"  Source range: [{src_img.min():.2f}, {src_img.max():.2f}]")
            
            # Check flow statistics
            flow = sample['flow']  # (2, H, W)
            # Valid flow is where both components are finite
            valid_mask = torch.isfinite(flow).all(dim=0)  # (H, W)
            if valid_mask.any():
                print(f"  Valid flow points: {valid_mask.sum().item()}")
                print(f"  Flow range: x=[{flow[0][valid_mask].min():.2f}, {flow[0][valid_mask].max():.2f}], y=[{flow[1][valid_mask].min():.2f}, {flow[1][valid_mask].max():.2f}]")
                # Sample a few valid flow vectors
                valid_indices = torch.nonzero(valid_mask, as_tuple=False)
                if len(valid_indices) > 0:
                    sample_indices = valid_indices[:min(5, len(valid_indices))]
                    sample_flows = flow[:, sample_indices[:, 0], sample_indices[:, 1]].T  # (N, 2)
                    print(f"  Sample flow vectors: {sample_flows[:5]}")
            else:
                print("  No valid flow found")
                
        except Exception as e:
            print(f"Error loading sample {i}: {e}")
            continue
    
    if not batch_data:
        print("No valid samples loaded")
        return
    

    dataloader = DataLoader[Any](dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))

    
    # # Visualize masks
    # print("\nVisualizing instance masks...")
    # dataset_instance = PointOdysseyFlowDataset(
    #     dataset_location=dataset_path,
    #     dset='train',
    #     use_augs=False,
    #     S=8,
    #     N=32,
    #     quick=False,
    #     verbose=False,
    #     resize_size=(size+64, size+64),
    #     crop_size=(size, size),
    #     all_points=False,
    # )
    # dataset_instance.visualize_masks_batch(masks_batch, "./debug/class_masks_batch_visualization.png")
    
    
    try:
        from src.data.synth.datasets.visualizers import CorrespondenceVisualizer
        from src.data.synth.datasets.cats_flow_visualizers import CATSFlowVisualizer
        
        # Create visualizer
        visualizer = CorrespondenceVisualizer(
            figsize=(20, 15),
            dpi=150,
            arrow_scale=1.0,
            arrow_density=20
        )

        cats_visualizer = CATSFlowVisualizer(
            feat_size=32,
            figsize=(20, 15),
            dpi=150,
            show_patch_boundaries=True
        )

        if downsample_for_cats:
            batch_downsampled = {
                'src_img': batch['src_img'],
                'trg_img': batch['trg_img'],
                'flow_downsampled': batch['flow']
            }
            cats_visualizer.visualize_downsampled_flow_batch(
                batch_downsampled,
                save_path="./debug/pointodyssey_flow_downsampled_side_by_side.png",
                max_samples=len(batch_data),
                visualization_mode='side_by_side'
            )
            cats_visualizer.visualize_downsampled_flow_batch(
                batch_downsampled,
                save_path="./debug/pointodyssey_flow_downsampled_overlay.png",
                max_samples=len(batch_data),
                visualization_mode='overlay'
            )
            
            # Create mask-based visualization: src_img and trg_img from masks
            if 'masks' in batch:
                masks_batch = batch['masks']  # (batch_size, S, 1, H, W)
                batch_size = masks_batch.shape[0]
                num_instances = masks_batch.shape[1]
                H, W = masks_batch.shape[3], masks_batch.shape[4]
                
                # Convert to numpy if needed
                if torch.is_tensor(masks_batch):
                    masks_np = masks_batch.cpu().numpy()
                else:
                    masks_np = masks_batch
                
                # Create mask-based images
                src_mask_imgs = []
                trg_mask_imgs = []
                
                for i in range(batch_size):
                    # Get src mask (first frame) and trg mask (last frame)
                    src_mask = masks_np[i, 0, 0, :, :]  # (H, W)
                    trg_mask = masks_np[i, num_instances-1, 0, :, :]  # (H, W)
                    
                    # Identify background: mask value 0 (sky) or max (landscape/background)
                    max_mask_value = np.max(masks_np[i])
                    src_background = (src_mask == 0) | (src_mask == max_mask_value)
                    trg_background = (trg_mask == 0) | (trg_mask == max_mask_value)
                    
                    # Create object masks: NOT background
                    src_object_mask = ~src_background  # (H, W) - True where objects
                    trg_object_mask = ~trg_background  # (H, W) - True where objects
                    
                    # Create RGB images: src as red, trg as green (matching overlay_background_aware pattern)
                    src_mask_img = np.zeros((3, H, W), dtype=np.float32)
                    src_mask_img[0] = src_object_mask.astype(np.float32)  # Red channel = src objects
                    
                    trg_mask_img = np.zeros((3, H, W), dtype=np.float32)
                    trg_mask_img[1] = trg_object_mask.astype(np.float32)  # Green channel = trg objects
                    
                    src_mask_imgs.append(torch.from_numpy(src_mask_img))
                    trg_mask_imgs.append(torch.from_numpy(trg_mask_img))
                
                # Stack into batch tensors
                src_mask_batch = torch.stack(src_mask_imgs, dim=0)  # (batch_size, 3, H, W)
                trg_mask_batch = torch.stack(trg_mask_imgs, dim=0)  # (batch_size, 3, H, W)
                
                batch_mask_overlay = {
                    'src_img': src_mask_batch,
                    'trg_img': trg_mask_batch,
                    'flow_downsampled': batch['flow']
                }
                
                cats_visualizer.visualize_downsampled_flow_batch(
                    batch_mask_overlay,
                    save_path="./debug/pointodyssey_flow_downsampled_overlay_mask.png",
                    max_samples=len(batch_data),
                    visualization_mode='overlay'
                )
            
        else:
            # Visualize with side-by-side layout
            print("\nCreating side-by-side visualization...")
            visualizer.visualize_rendered_batch(
                batch,
                save_path="./debug/pointodyssey_flow_side_by_side.png",
                max_samples=len(batch_data),
                visualization_mode='side_by_side',
                sampling_mode='all_valid'
            )

            # Visualize with overlay_background_aware layout
            print("Creating overlay_background_aware visualization...")
            visualizer.visualize_rendered_batch(
                batch,
                save_path="./debug/pointodyssey_flow_overlay_background_aware.png",
                max_samples=len(batch_data),
                visualization_mode='overlay_background_aware',
                sampling_mode='all_valid'
            )


        
        print("Visualization complete! Check the generated PNG files.")
        
    except ImportError as e:
        print(f"Could not import visualizer: {e}")
        print("Skipping visualization, but dataset test completed successfully.")
    
    return batch


def test_dataset():
    """Test the dataset wrapper without visualization."""
    print("Testing PointOdyssey Flow Dataset...")
    
    # Create dataset
    dataset = PointOdysseyFlowDataset(
        dataset_location='/home/spencer/Data/sample',
        dset='train',
        use_augs=False,
        S=8,
        N=32,
        quick=False,  # Use quick mode for testing
        verbose=True,
        size=256  # Resize to 256x256 (square)
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        
        print(f"Sample keys: {sample.keys()}")
        print(f"Source image shape: {sample['src_img'].shape}")
        print(f"Target image shape: {sample['trg_img'].shape}")
        print(f"Flow shape: {sample['flow'].shape}")
        
        # Check flow statistics
        flow = sample['flow']  # (2, H, W)
        # Valid flow is where both components are finite
        valid_mask = torch.isfinite(flow).all(dim=0)  # (H, W)
        if valid_mask.any():
            print(f"Valid flow points: {valid_mask.sum().item()}")
            print(f"Flow range: x=[{flow[0][valid_mask].min():.2f}, {flow[0][valid_mask].max():.2f}], y=[{flow[1][valid_mask].min():.2f}, {flow[1][valid_mask].max():.2f}]")
        else:
            print("No valid flow found")
        
        return sample
    else:
        print("No samples in dataset")
        return None


def test_mask_visualization():
    """Test mask visualization specifically."""
    print("Testing Mask Visualization...")
    
    # Create dataset
    dataset = PointOdysseyFlowDataset(
        dataset_location='/home/spencer/Data/sample',
        dset='train',
        use_augs=False,
        S=8,
        N=32,
        quick=False,
        verbose=True
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get a sample
        sample = dataset[0]
        masks = sample['masks']  # Shape: (S, 1, H, W)
        
        print(f"Masks shape: {masks.shape}")
        print(f"Unique instance IDs: {torch.unique(masks)}")
        
        # Visualize masks
        dataset.visualize_masks(masks, "./debug/class_masks_visualization.png")
        
        return sample
    else:
        print("No samples in dataset")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PointOdyssey flow dataset')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to PointOdyssey dataset directory')
    parser.add_argument('--size', type=int, default=None,
                        help='Target square size for resizing (size x size)')
    parser.add_argument('--dset', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to use')
    parser.add_argument('--visualize', action='store_true',
                        help='Run with visualization')
    parser.add_argument('--masks', action='store_true',
                        help='Test mask visualization only')
    parser.add_argument('--downsample_for_cats', type=bool, default=False,
                        help='Downsample flow for CATs')
    
    args = parser.parse_args()
    
    # Use size directly if provided
    size = args.size if args.size else None
    
    if args.masks:
        # Test mask visualization only
        sample = test_mask_visualization()
    elif args.visualize:
        # Test with visualization
        batch_dict = test_dataset_with_visualization(args.dataset_path, size, args.downsample_for_cats)
    else:
        # Test without visualization
        sample = test_dataset()

