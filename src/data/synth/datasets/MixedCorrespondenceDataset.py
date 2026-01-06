from typing import List, Optional, Tuple
import random
import torch
from torch.utils.data import Dataset

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset, _strip_leading_batch
from src.data.synth.adapters import SyntheticAdapter
from src.data.synth.common.common_sample import CommonSample
from src.data.synth.collate_pipeline import (
    resize_sample,
    ensure_flow_and_kps,
    normalize_images,
    collate_common_samples,
)


def _move_sample_to_cpu(sample: CommonSample) -> CommonSample:
    """Move all tensors in a CommonSample to CPU."""
    if sample.src_img is not None and isinstance(sample.src_img, torch.Tensor) and sample.src_img.is_cuda:
        sample.src_img = sample.src_img.cpu()
    if sample.trg_img is not None and isinstance(sample.trg_img, torch.Tensor) and sample.trg_img.is_cuda:
        sample.trg_img = sample.trg_img.cpu()
    if sample.flow_full is not None and isinstance(sample.flow_full, torch.Tensor) and sample.flow_full.is_cuda:
        sample.flow_full = sample.flow_full.cpu()
    if sample.flow_feat is not None and isinstance(sample.flow_feat, torch.Tensor) and sample.flow_feat.is_cuda:
        sample.flow_feat = sample.flow_feat.cpu()
    if sample.src_kps is not None and isinstance(sample.src_kps, torch.Tensor) and sample.src_kps.is_cuda:
        sample.src_kps = sample.src_kps.cpu()
    if sample.trg_kps is not None and isinstance(sample.trg_kps, torch.Tensor) and sample.trg_kps.is_cuda:
        sample.trg_kps = sample.trg_kps.cpu()
    if sample.pckthres is not None and isinstance(sample.pckthres, torch.Tensor) and sample.pckthres.is_cuda:
        sample.pckthres = sample.pckthres.cpu()
    return sample


class MixedCorrespondenceDataset(Dataset):
    """
    Mixes multiple CorrespondenceDataset instances with specified percentages.
    
    Critical requirement: Synthetic samples must be processed as isolated batches
    due to CUDA kernel parallelization in apply_texture() from multi_texturing.py.
    
    Usage:
        dataset1 = CorrespondenceDataset("spair", ...)
        dataset2 = CorrespondenceDataset("synthetic", ...)
        mixed = MixedCorrespondenceDataset(
            datasets=[dataset1, dataset2],
            percentages=[0.5, 0.5]
        )
    """
    
    def __init__(
        self,
        datasets: List[CorrespondenceDataset],
        percentages: List[float],
        epoch_size: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize mixed dataset.
        
        Args:
            datasets: List of CorrespondenceDataset instances to mix
            percentages: List of percentages for each dataset (will be normalized to sum to 1.0)
            epoch_size: Optional fixed epoch size. If None, uses sum of all dataset lengths
            seed: Optional random seed for reproducibility
        """
        super().__init__()
        
        if len(datasets) != len(percentages):
            raise ValueError(f"Number of datasets ({len(datasets)}) must match number of percentages ({len(percentages)})")
        
        if len(datasets) == 0:
            raise ValueError("Must provide at least one dataset")
        
        self.datasets = datasets
        self.dataset_names = [ds.dataset_name for ds in datasets]
        
        # Normalize percentages
        total = sum(percentages)
        if total <= 0:
            raise ValueError("Percentages must sum to a positive value")
        self.percentages = [p / total for p in percentages]
        
        # Build cumulative distribution for sampling
        self.cumulative_percentages = []
        cumsum = 0.0
        for p in self.percentages:
            cumsum += p
            self.cumulative_percentages.append(cumsum)
        # Ensure last value is exactly 1.0 to avoid floating point issues
        self.cumulative_percentages[-1] = 1.0
        
        # Track which datasets are synthetic
        self.is_synthetic = [isinstance(ds.adapter, SyntheticAdapter) for ds in datasets]
        self.synthetic_dataset_idx = None
        for i, is_synth in enumerate(self.is_synthetic):
            if is_synth:
                if self.synthetic_dataset_idx is not None:
                    raise ValueError("Currently only one synthetic dataset is supported")
                self.synthetic_dataset_idx = i
        
        # Store unified config from first dataset (or allow override)
        first_ds = datasets[0]
        self.size = first_ds.size
        self.max_kps = first_ds.max_kps
        self.downsample_feat_size = first_ds.downsample_feat_size
        self.prefer_all_dense = first_ds.prefer_all_dense
        # For mixed datasets, use CPU as target_device to match non-synthetic samples
        # (Training loop will move to GPU as needed)
        self.target_device = torch.device("cpu")
        
        # Normalization: track per-dataset flags so mixed batches normalize correctly
        self.normalize_images_flags = [ds.normalize_images_flag for ds in datasets]
        self.normalize_images_flag = self.normalize_images_flags[0]
        
        # Epoch size
        if epoch_size is not None:
            self.epoch_size = epoch_size
        else:
            # Default: sum of all dataset lengths
            self.epoch_size = sum(len(ds) for ds in datasets)
        
        # Random state
        self.rng = random.Random(seed) if seed is not None else random.Random()
        
        # Store dataset name for compatibility
        self.dataset_name = "mixed"
        
        print(f"Created MixedCorrespondenceDataset with {len(datasets)} datasets:")
        for i, (ds_name, pct, is_synth) in enumerate(zip(self.dataset_names, self.percentages, self.is_synthetic)):
            synth_str = " (synthetic)" if is_synth else ""
            print(f"  {i}: {ds_name} - {pct*100:.1f}%{synth_str}")
        print(f"Total epoch size: {self.epoch_size}")
    
    def __len__(self):
        return self.epoch_size
    
    def __getitem__(self, idx):
        """
        Sample from datasets according to percentages.
        
        Note: idx is ignored for random sampling. For deterministic sampling,
        we could use idx as a seed, but for now we use pure random sampling.
        """
        # Sample random value to select dataset
        r = self.rng.random()
        
        # Find which dataset to sample from using cumulative distribution
        dataset_idx = 0
        for i, cum_pct in enumerate(self.cumulative_percentages):
            if r <= cum_pct:
                dataset_idx = i
                break
        
        # Sample random index from selected dataset
        selected_dataset = self.datasets[dataset_idx]
        dataset_len = len(selected_dataset)
        if dataset_len == 0:
            raise ValueError(f"Dataset {self.dataset_names[dataset_idx]} is empty")
        
        sample_idx = self.rng.randint(0, dataset_len - 1)
        
        # Get raw sample from adapter (bypassing CorrespondenceDataset.__getitem__)
        # This is important for synthetic datasets which need raw [src_dict, trg_dict] format
        raw_sample = selected_dataset.adapter[sample_idx]
        
        # For synthetic datasets, raw_sample is [src_dict, trg_dict]
        # For non-synthetic, it's typically a CommonSample or dict
        if self.is_synthetic[dataset_idx]:
            # Synthetic: store raw sample and create a placeholder CommonSample for tracking
            sample = CommonSample(meta={})
            sample.meta['raw_synthetic_sample'] = raw_sample
            sample.meta['sample_idx'] = sample_idx
        else:
            # Non-synthetic: convert to CommonSample if needed
            if not isinstance(raw_sample, CommonSample):
                sample = CommonSample(
                    src_img=raw_sample.get("src_img") if isinstance(raw_sample, dict) else None,
                    trg_img=raw_sample.get("trg_img") if isinstance(raw_sample, dict) else None,
                    flow_full=raw_sample.get("flow_full") or (raw_sample.get("flow") if isinstance(raw_sample, dict) else None),
                    flow_feat=raw_sample.get("flow_downsampled") if isinstance(raw_sample, dict) else None,
                    src_kps=raw_sample.get("src_kps") if isinstance(raw_sample, dict) else None,
                    trg_kps=raw_sample.get("trg_kps") if isinstance(raw_sample, dict) else None,
                    n_pts=raw_sample.get("n_pts") if isinstance(raw_sample, dict) else None,
                    pckthres=raw_sample.get("pckthres") if isinstance(raw_sample, dict) else None,
                )
            else:
                sample = raw_sample
        
        # Add metadata to track source dataset
        if sample.meta is None:
            sample.meta = {}
        sample.meta['source_dataset'] = self.dataset_names[dataset_idx]
        sample.meta['source_dataset_idx'] = dataset_idx
        if 'sample_idx' not in sample.meta:
            sample.meta['sample_idx'] = sample_idx
        
        return sample
    
    def collate_fn(self, batch):
        """
        Smart batch processing that:
        1. Groups synthetic samples for isolated batch processing (CUDA kernel requirement)
        2. Processes non-synthetic samples separately
        3. Merges all processed samples
        4. Shuffles merged samples
        5. Applies unified post-processing
        6. Final collation
        """
        if len(batch) == 0:
            return {}
        
        # Step 1: Group samples by dataset type
        synthetic_samples = []
        non_synthetic_samples = []
        synthetic_raw_batch = []  # Raw samples before processing (for synthetic batch processing)
        
        for sample in batch:
            source_dataset_idx = sample.meta.get('source_dataset_idx', None)
            if source_dataset_idx is not None and self.is_synthetic[source_dataset_idx]:
                synthetic_samples.append(sample)
                # Get raw synthetic sample from metadata (stored in __getitem__)
                raw_sample = sample.meta.get('raw_synthetic_sample', None)
                if raw_sample is None:
                    # Fallback: re-fetch from adapter using stored index
                    sample_idx = sample.meta.get('sample_idx', None)
                    if sample_idx is not None:
                        dataset = self.datasets[source_dataset_idx]
                        raw_sample = dataset.adapter[sample_idx]
                    else:
                        raise ValueError("Synthetic samples need raw adapter output for batch processing")
                synthetic_raw_batch.append(raw_sample)
            else:
                non_synthetic_samples.append(sample)
        
        # Step 2: Process synthetic samples as isolated batch
        synthetic_processed_samples = []
        if len(synthetic_samples) > 0:
            if self.synthetic_dataset_idx is None:
                raise ValueError("Found synthetic samples but no synthetic dataset configured")
            
            synthetic_dataset = self.datasets[self.synthetic_dataset_idx]
            # Process synthetic batch using the dataset's _process_synthetic_batch method
            synthetic_processed_samples = synthetic_dataset._process_synthetic_batch(synthetic_raw_batch)
            if len(synthetic_processed_samples) != len(synthetic_samples):
                raise ValueError(
                    "Synthetic batch processing returned mismatched sample count "
                    f"({len(synthetic_processed_samples)} != {len(synthetic_samples)})"
                )
            for processed, original in zip(synthetic_processed_samples, synthetic_samples):
                original_meta = original.meta or {}
                processed.meta = {
                    "source_dataset": original_meta.get(
                        "source_dataset",
                        self.dataset_names[self.synthetic_dataset_idx],
                    ),
                    "source_dataset_idx": original_meta.get(
                        "source_dataset_idx",
                        self.synthetic_dataset_idx,
                    ),
                    "sample_idx": original_meta.get("sample_idx"),
                }
            
            # Move synthetic samples to CPU to match non-synthetic samples
            # (They'll be moved to target_device later in collate_common_samples)
            for sample in synthetic_processed_samples:
                _move_sample_to_cpu(sample)
        
        # Step 3: Process non-synthetic samples separately
        non_synthetic_processed_samples = []
        for sample in non_synthetic_samples:
            # Non-synthetic samples are already CommonSample objects from their adapters
            # We just need to ensure they're properly formatted
            if not isinstance(sample, CommonSample):
                sample = CommonSample(
                    src_img=_strip_leading_batch(sample.get("src_img") if isinstance(sample, dict) else None),
                    trg_img=_strip_leading_batch(sample.get("trg_img") if isinstance(sample, dict) else None),
                    flow_full=_strip_leading_batch(sample.get("flow_full") or (sample.get("flow") if isinstance(sample, dict) else None)),
                    flow_feat=_strip_leading_batch(sample.get("flow_downsampled") if isinstance(sample, dict) else None),
                    src_kps=sample.get("src_kps") if isinstance(sample, dict) else None,
                    trg_kps=sample.get("trg_kps") if isinstance(sample, dict) else None,
                    n_pts=sample.get("n_pts") if isinstance(sample, dict) else None,
                    pckthres=sample.get("pckthres") if isinstance(sample, dict) else None,
                )
            
            # Remove accidental leading batch dims
            sample.src_img = _strip_leading_batch(sample.src_img)
            sample.trg_img = _strip_leading_batch(sample.trg_img)
            sample.flow_full = _strip_leading_batch(sample.flow_full)
            sample.flow_feat = _strip_leading_batch(sample.flow_feat)
            
            non_synthetic_processed_samples.append(sample)
        
        # Step 4: Merge all processed samples
        all_processed_samples = synthetic_processed_samples + non_synthetic_processed_samples
        
        # Step 5: Shuffle merged samples
        # Use random.shuffle to shuffle the list in-place
        random.shuffle(all_processed_samples)
        
        # Step 6: Apply unified post-processing
        processed_samples = []
        for sample in all_processed_samples:
            # Resize
            sample = resize_sample(sample, self.size)
            
            # Ensure flow and keypoints
            # Use dataset_name from source if available, otherwise use "mixed"
            source_dataset_name = sample.meta.get('source_dataset', 'mixed')
            sample = ensure_flow_and_kps(
                sample,
                dataset_name=source_dataset_name,
                max_kps=self.max_kps,
                downsample_feat_size=self.downsample_feat_size,
                prefer_all_dense=self.prefer_all_dense,
            )
            
            # Normalize images
            source_dataset_idx = sample.meta.get("source_dataset_idx")
            if (
                source_dataset_idx is not None
                and 0 <= source_dataset_idx < len(self.normalize_images_flags)
            ):
                normalize_flag = self.normalize_images_flags[source_dataset_idx]
            else:
                normalize_flag = self.normalize_images_flag
            sample = normalize_images(sample, normalize_flag)
            
            processed_samples.append(sample)
        
        # Step 7: Final collation
        batch_out = collate_common_samples(
            processed_samples,
            max_kps=self.max_kps,
            target_device=self.target_device,
        )
        return batch_out
