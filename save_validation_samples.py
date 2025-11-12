"""
Script to save validation batches from all benchmarks for easy transfer.
Collects 100 samples (10 batches with batch_size=10) from each validation dataloader.
"""

import os
import yaml
import torch
from torch.utils.data import DataLoader

# Import dataset classes
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from src.data.synth.datasets.TSSDataset import TSSDataset
from src.data.synth.datasets.KittiDataset import KittiDataset
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsDataset
from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseyFlowDataset
import models.CATs_PlusPlus.data.download as download

# Configuration
BATCH_SIZE = 10
NUM_BATCHES = 100  # Total samples = BATCH_SIZE * NUM_BATCHES = 100
CONFIG_PATH = 'slurm/machine_configs/remote.yaml'
DATASET_CONFIG_PATH = 'src/configs/online_synth_configs/OnlineDatasetConfig_FullFlow.yaml'
OUTPUT_PATH = 'validation_samples.pt'  # Use .pt extension for torch.save


def load_dataset_config(config_path):
    """Load geometry and processor config paths from dataset config YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    val_config = config['dataset_configs']['val_dataset_config']['init_args']
    return {
        'geometry_config_path': val_config['geometry_config_path'],
        'processor_config_path': val_config['processor_config_path']
    }


# Default values (from train_cats.py)
# Note: geometry_config_path and processor_config_path will be loaded from DATASET_CONFIG_PATH
DEFAULTS = {
    'size': 512,
    'feature_size': 32,
    'thres': 'img',
    'val_batch_size': 8,
    'val_num_workers': 16,
    'n_threads': 0,
    'split_to_use_for_validation': 'val',
    'datapath': './models/Datasets_CATs',
    'kitti_val_use_full_training': True,
    'subsample_flow': 0.01,
    'subsample_flow_seed': None,
    'val_sequence_fraction_pointodyssey': 1.0,
}

# All benchmarks to collect
EVAL_BENCHMARKS = ['synthetic', 'spair', 'pfpascal', 'pfwillow', 'caltech', 'tss', 
                   'pointodyssey', 'kitti2012', 'kitti2015', 'flyingthings']


def load_config(config_path):
    """Load dataset roots from YAML config file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['datasets']


def collect_batches(dataloader, num_batches):
    """Collect specified number of batches from dataloader.
    
    If the dataloader has fewer batches than requested, collects all available batches
    and returns them (no error is raised).
    """
    dataloader_len = len(dataloader)
    batches_to_collect = min(num_batches, dataloader_len)
    
    if dataloader_len < num_batches:
        print(f"  Warning: Requested {num_batches} batches but dataloader only has {dataloader_len} batches. Collecting all available.")
    
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= batches_to_collect:
            break
        # Move tensors to CPU for saving
        cpu_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                cpu_batch[key] = value.cpu()
            elif isinstance(value, (list, tuple)):
                cpu_batch[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                cpu_batch[key] = value
        batches.append(cpu_batch)
    return batches


def main():
    # Load dataset roots from config
    print(f"Loading config from {CONFIG_PATH}...")
    dataset_config = load_config(CONFIG_PATH)
    
    # Load dataset config for geometry and processor paths
    print(f"Loading dataset config from {DATASET_CONFIG_PATH}...")
    dataset_paths = load_dataset_config(DATASET_CONFIG_PATH)
    print(f"  Geometry config: {dataset_paths['geometry_config_path']}")
    print(f"  Processor config: {dataset_paths['processor_config_path']}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Download standard benchmark datasets if needed
    standard_benchmarks = ['spair', 'pfpascal', 'pfwillow', 'caltech']
    for benchmark in EVAL_BENCHMARKS:
        if benchmark in standard_benchmarks:
            print(f"Downloading {benchmark} dataset if needed...")
            download.download_dataset(DEFAULTS['datapath'], benchmark)
    
    # Dictionary to store all batches
    all_batches = {}
    
    # Geometry config overrides for synthetic dataset (None = use defaults)
    geometry_config_overrides = None
    
    # Setup validation dataloaders and collect batches
    for benchmark in EVAL_BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"Processing benchmark: {benchmark}")
        print(f"{'='*60}")
        
        try:
            if benchmark == 'synthetic':
                # Use paths from dataset config
                val_dataset = OnlineCorrespondenceDataset(
                    geometry_config_path=dataset_paths['geometry_config_path'],
                    processor_config_path=dataset_paths['processor_config_path'],
                    split='val',
                    geometry_config_overrides=geometry_config_overrides
                )
                val_dataset.cuda()
                val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=0, 
                    shuffle=False, 
                    collate_fn=val_dataset.collate_fn
                )
                
            elif benchmark == 'tss':
                val_dataset = TSSDataset(
                    root=dataset_config['tss_root'], 
                    device=device,
                    size=DEFAULTS['size'],
                    feature_size=DEFAULTS['feature_size'],
                    thres=DEFAULTS['thres']
                )
                val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=DEFAULTS['val_num_workers'], 
                    persistent_workers=True, 
                    prefetch_factor=8, 
                    shuffle=False
                )
                
            elif benchmark == 'pointodyssey':
                val_dataset = PointOdysseyFlowDataset(
                    dataset_location=dataset_config['pointodyssey_root'],
                    dset='val',
                    use_augs=False,
                    S=4,
                    N=32,
                    quick=False,
                    verbose=False,
                    resize_size=(DEFAULTS['size']+64, DEFAULTS['size']+64),
                    crop_size=(DEFAULTS['size'], DEFAULTS['size']),
                    filter_instances=True,
                    downsample_for_cats=False,  
                    cats_feat_size=DEFAULTS['feature_size'],
                    max_pts=200,
                    thres=DEFAULTS['thres'],
                    normalize_images=True, 
                    val_sequence_fraction=DEFAULTS['val_sequence_fraction_pointodyssey']
                )
                val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=DEFAULTS['val_num_workers'], 
                    persistent_workers=True, 
                    prefetch_factor=8, 
                    shuffle=False, 
                    pin_memory=True
                )
                
            elif benchmark in ['kitti2012', 'kitti2015']:
                version = '2012' if '2012' in benchmark else '2015'
                kitti_split = 'training' if DEFAULTS['kitti_val_use_full_training'] else 'val'
                val_dataset = KittiDataset(
                    root=os.path.join(dataset_config['kitti_unsplit_root'], f'kitti-{version}'),
                    split=kitti_split,
                    version=version,
                    occ_type='occ',
                    size=(DEFAULTS['size'], DEFAULTS['size']),
                    downsample_flow=DEFAULTS['feature_size'],
                    normalize=True,
                    normalize_images=True,
                    thres=DEFAULTS['thres'],
                    max_pts=200
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=DEFAULTS['val_num_workers'],
                    persistent_workers=True,
                    prefetch_factor=8,
                    shuffle=False,
                    pin_memory=True
                )
                
            elif benchmark == 'flyingthings':
                print(f"Flyingthings val root: {dataset_config['flyingthings_root']}")
                val_dataset = FlyingThingsDataset(
                    root=dataset_config['flyingthings_root'], 
                    split="test", 
                    transforms=None, 
                    size=(DEFAULTS['size'], DEFAULTS['size']), 
                    downsample_flow=DEFAULTS['feature_size'], 
                    subsample_flow=DEFAULTS['subsample_flow'], 
                    subsample_flow_seed=DEFAULTS['subsample_flow_seed'], 
                    use_valid_mask=True, 
                    reverse_flow=True, 
                    filter_out_of_bounds=True
                )
                val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    num_workers=DEFAULTS['val_num_workers'], 
                    persistent_workers=True, 
                    prefetch_factor=8, 
                    shuffle=False
                )
                
            else:  # Standard benchmarks: spair, pfpascal, pfwillow, caltech
                val_dataset = download.load_dataset(
                    benchmark, 
                    DEFAULTS['datapath'], 
                    DEFAULTS['thres'], 
                    device, 
                    DEFAULTS['split_to_use_for_validation'], 
                    False, 
                    DEFAULTS['feature_size']
                )
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=DEFAULTS['val_num_workers'],
                    persistent_workers=True,
                    prefetch_factor=8,
                    shuffle=False,
                    pin_memory=True
                )
            
            print(f"Val dataloader for benchmark '{benchmark}' size: {len(val_dataloader)}")
            
            # Collect batches
            print(f"Collecting {NUM_BATCHES} batches (batch_size={BATCH_SIZE})...")
            batches = collect_batches(val_dataloader, NUM_BATCHES)
            all_batches[benchmark] = batches
            print(f"✓ Collected {len(batches)} batches for {benchmark}")
            
        except Exception as e:
            print(f"✗ Error processing {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save the dictionary using torch.save (more efficient for PyTorch tensors)
    print(f"\n{'='*60}")
    print(f"Saving all batches to {OUTPUT_PATH}...")
    print(f"  This may take a while for large files...")
    
    # torch.save is more efficient for PyTorch tensors and handles large files better
    torch.save(all_batches, OUTPUT_PATH, _use_new_zipfile_serialization=True)
    
    # Check file size
    file_size_gb = os.path.getsize(OUTPUT_PATH) / (1024**3)
    
    print(f"✓ Saved validation samples for {len(all_batches)} benchmarks")
    print(f"  Total size: {sum(len(batches) for batches in all_batches.values())} batches")
    print(f"  Output file: {OUTPUT_PATH}")
    print(f"  File size: {file_size_gb:.2f} GB")


if __name__ == '__main__':
    main()
