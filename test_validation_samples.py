"""
Script to load validation samples from individual dataset files.
"""

import os
import torch
import glob

# Path to the directory containing validation sample files
VALIDATION_SAMPLES_DIR = 'validation_samples'


def load_validation_samples(directory):
    """Load validation samples from directory of individual .pt files.
    
    Returns a dictionary with benchmark names as keys and lists of batches as values.
    """
    validation_samples = {}
    
    # Find all .pt files in the directory
    pattern = os.path.join(directory, '*.pt')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No .pt files found in {directory}")
        return validation_samples
    
    print(f"Loading validation samples from {directory}...")
    for file_path in sorted(files):
        benchmark = os.path.splitext(os.path.basename(file_path))[0]
        print(f"  Loading {benchmark} from {file_path}...")
        batches = torch.load(file_path, map_location='cpu')
        validation_samples[benchmark] = batches
        print(f"    Loaded {len(batches)} batches")
    
    return validation_samples


if __name__ == '__main__':
    validation_samples = load_validation_samples(VALIDATION_SAMPLES_DIR)
    
    print(f"\nLoaded validation samples for {len(validation_samples)} benchmarks:")
    for benchmark, batches in validation_samples.items():
        print(f"  {benchmark}: {len(batches)} batches")
        if len(batches) > 0:
            print(f"    First batch keys: {list(batches[0].keys())}")
    
    # The dictionary is now available as 'validation_samples'
    # Example usage:
    # synthetic_batches = validation_samples['synthetic']
    # first_batch = synthetic_batches[0]

