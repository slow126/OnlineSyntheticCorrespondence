from platform import processor
import torch
from src.data.synth.datasets.OnlineGeometryDataset import OnlineGeometryDataset
from src.data.synth.datasets.processors.SyntheticCorrespondenceProcessor import SyntheticCorrespondenceProcessor
from src.data.synth.datasets.base import ComponentsBase
from src.data.synth.datasets.visualizers import GeometryVisualizer, CorrespondenceVisualizer
import yaml
from torch.utils.data.dataloader import default_collate
import os


class OnlineCorrespondenceDataset():
    def __init__(self, geometry_config_path, processor_config_path):
        super().__init__()
        with open(geometry_config_path, 'r') as f:
            geometry_config = yaml.load(f, Loader=yaml.FullLoader)
        with open(processor_config_path, 'r') as f:
            processor_config = yaml.load(f, Loader=yaml.FullLoader)

        self._device = torch.device('cpu')   

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
        return self.processor.process_scene(batch)

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
        print(f"GPU batch device: {batch['src'].device}")
        
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
