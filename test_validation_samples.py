"""
Script to load validation samples from pickle file.
"""

import pickle

# Path to the saved validation samples
VALIDATION_SAMPLES_PATH = 'validation_samples.pkl'


def load_validation_samples(file_path):
    """Load validation samples dictionary from pickle file."""
    with open(file_path, 'rb') as f:
        samples = pickle.load(f)
    return samples


if __name__ == '__main__':
    print(f"Loading validation samples from {VALIDATION_SAMPLES_PATH}...")
    validation_samples = load_validation_samples(VALIDATION_SAMPLES_PATH)
    
    print(f"\nLoaded validation samples for {len(validation_samples)} benchmarks:")
    for benchmark, batches in validation_samples.items():
        print(f"  {benchmark}: {len(batches)} batches")
        if len(batches) > 0:
            print(f"    First batch keys: {list(batches[0].keys())}")
    
    # The dictionary is now available as 'validation_samples'
    # Example usage:
    # synthetic_batches = validation_samples['synthetic']
    # first_batch = synthetic_batches[0]

