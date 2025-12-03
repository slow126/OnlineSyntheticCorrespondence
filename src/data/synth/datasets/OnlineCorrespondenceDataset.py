from platform import processor
import torch
import copy
from src.data.synth.datasets.OnlineGeometryDataset import OnlineGeometryDataset
from src.data.synth.datasets.processors.SyntheticCorrespondenceProcessor import SyntheticCorrespondenceProcessor
from src.data.synth.datasets.base import ComponentsBase
from src.data.synth.datasets.visualizers import GeometryVisualizer, CorrespondenceVisualizer
import yaml
from torch.utils.data.dataloader import default_collate
import os


def deep_merge(base_dict, override_dict):
    """Recursively merge override_dict into base_dict."""
    result = copy.deepcopy(base_dict)
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class OnlineCorrespondenceDataset():
    def __init__(
        self, 
        geometry_config_path, 
        processor_config_path, 
        split='train',
        opengl_device_index=None,
        geometry_config_overrides=None
    ):
        super().__init__()
        with open(geometry_config_path, 'r') as f:
            geometry_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(processor_config_path, 'r') as f:
            processor_config = yaml.load(f, Loader=yaml.FullLoader)

        # Apply overrides if provided
        if geometry_config_overrides:
            geometry_config = deep_merge(geometry_config, geometry_config_overrides)

        self._device = torch.device('cpu')   
        self.split = split
        
        # Pass GPU index to geometry dataset (for multi-GPU support)
        # None = auto-detect from torch.cuda.current_device() (works with Lightning DDP)
        geometry_config['opengl_device_index'] = opengl_device_index
        
        self.dataset = OnlineGeometryDataset(**geometry_config)
        self.processor = SyntheticCorrespondenceProcessor(**processor_config)
        self.geometry_visualizer = GeometryVisualizer()
        self.correspondence_visualizer = CorrespondenceVisualizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        raw_sample = self.dataset.__getitem__(idx)
        return raw_sample

    def debug_view(self, idx, batch_size=4, save_path='./debug'):
        """Visualize a batch of samples starting from the given index."""

        os.makedirs(save_path, exist_ok=True)
        geometry_save_path = os.path.join(save_path, 'geometry.png')
        correspondence_save_path_side_by_side = os.path.join(save_path, 'correspondence_side_by_side.png')
        correspondence_save_path_overlay = os.path.join(save_path, 'correspondence_overlay.png')
        
        batch_data = []
        for i in range(batch_size): 
            raw_sample = self.dataset[idx + i]
            batch_data.append(raw_sample)
        self.geometry_visualizer.visualize_batch(batch_data, geometry_save_path)
        batch_data = self.collate_fn(batch_data)
        self.correspondence_visualizer.visualize_rendered_batch(batch_data, correspondence_save_path_side_by_side, visualization_mode='side_by_side')
        self.correspondence_visualizer.visualize_rendered_batch(batch_data, correspondence_save_path_overlay, visualization_mode='overlay')

    def collate_fn(self, batch):
        batch = default_collate(batch)
        batch = self.processor.batch_to_device(batch, self.processor.device)
        processed_batch = self.processor.process_scene(batch)

        # Keep an explicit alias for feature-grid flow so downstream
        # visualizers/debuggers can rely on a stable key during training.
        if (
            "flow_downsampled" not in processed_batch
            and self.processor.downsample_for_cats
            and "flow" in processed_batch
        ):
            processed_batch["flow_downsampled"] = processed_batch["flow"]

        return processed_batch
    
    def process_geometry(self, batch):
        batch = self.processor.batch_to_device(batch, self.processor.device)
        return self.processor.process_scene(batch)
    
    def process_sample(self, raw_sample):
        """
        Process a single raw sample (without batch dimension) through the processor.
        
        Args:
            raw_sample: Single raw sample from __getitem__ (list of two dicts: [src_dict, trg_dict])
        
        Returns:
            Processed sample dict with tensors without batch dimension
        """
        # raw_sample is a list of two dicts [src_dict, trg_dict]
        # Add batch dimension using default_collate
        # default_collate expects a list of samples, so we wrap raw_sample in a list
        batch = default_collate([raw_sample])
        # Convert to list if it's a tuple (default_collate may return tuple)
        if isinstance(batch, tuple):
            batch = list(batch)
        # Move to device
        batch = self.processor.batch_to_device(batch, self.processor.device)
        # Process through processor
        processed_batch = self.processor.process_scene(batch)
        # Remove batch dimension (squeeze dim 0) from all tensors
        processed_sample = {}
        for key, value in processed_batch.items():
            if isinstance(value, torch.Tensor):
                # Remove batch dimension
                if value.dim() > 0:
                    processed_sample[key] = value.squeeze(0)
                else:
                    processed_sample[key] = value
            else:
                processed_sample[key] = value
        return processed_sample


    @property
    def device(self):
        """Get the current device"""
        return self._device

    def to(self, device):
        """Move dataset to specified device (similar to PyTorch's .to() method)"""
        self._device = torch.device(device)
        self.processor.to(device)
        return self

    def cuda(self, device=None):
        """Move dataset to CUDA device"""
        # Get the current CUDA device if not specified
        if device is None:
            current_device = torch.cuda.current_device() if torch.cuda.is_available() else 0
            self._device = torch.device(f'cuda:{current_device}')
        else:
            self._device = torch.device(f'cuda:{device}')
        self.processor.cuda(device)
        return self

    def cpu(self):
        """Move dataset to CPU"""
        self._device = torch.device('cpu')
        self.processor.cpu()
        return self

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--geometry_config_path', type=str, default='src/configs/online_synth_configs/OnlineGeometryConfig.yaml')
    parser.add_argument('--processor_config_path', type=str, default='src/configs/online_synth_configs/OnlineProcessorConfig.yaml')
    args = parser.parse_args()
    dataset = OnlineCorrespondenceDataset(
        geometry_config_path=args.geometry_config_path,
        processor_config_path=args.processor_config_path
    )
    
    print(f"Dataset length: {len(dataset)}")
    print(f"Default device: {dataset.device}")
    print(f"Processor device: {dataset.processor.device}")
    
    # # Example usage with different devices
    if torch.cuda.is_available():
        print("\n=== GPU Usage ===")
        dataset.cuda()
        print(f"Dataset device: {dataset.device}")
        print(f"Processor device: {dataset.processor.device}")
        sample_gpu0 = dataset.__getitem__(0)
        sample_gpu1 = dataset.__getitem__(1)
        sample_gpu2 = dataset.__getitem__(2)
        print(f"GPU sample device: {sample_gpu0[0]['geometry'].device}")
        batch = [sample_gpu0, sample_gpu1, sample_gpu2]
        batch = dataset.collate_fn(batch)
        print(f"GPU batch device: {batch['src_img'].device}")
        
        print("\n=== Specific GPU Usage ===")
        dataset.to('cuda:0')
        print(f"Dataset device: {dataset.device}")
        print(f"Processor device: {dataset.processor.device}")
        sample_specific = dataset[0]
        print(f"Specific GPU sample device: {sample_specific[0]['geometry'].device}")
    
    # print("\n=== CPU Usage ===")
    # dataset.cpu()
    # print(f"Dataset device: {dataset.device}")
    # print(f"Processor device: {dataset.processor.device}")
    # sample_cpu = dataset[0]
    # print(f"CPU sample device: {sample_cpu[0]['geometry'].device}")
    
    # Debug visualization
    dataset.debug_view(5, save_path='./debug')
