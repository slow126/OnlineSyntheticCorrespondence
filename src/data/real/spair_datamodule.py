import os
import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from ..base import BaseDatamodule


class SPairDataset(torch.utils.data.Dataset):
    """
    SPair-71k dataset that works with the actual dataset structure.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = 'trn',
        layout: str = 'large',
        feature_size: int = 256,
        normalize: Union[bool, str] = True,
        augmentation: bool = False,
    ):
        """
        Initialize SPair dataset.
        
        Args:
            root: Path to SPair-71k dataset directory
            split: 'trn', 'val', or 'tst'
            layout: 'large' or 'small'
            feature_size: Size to resize images to
            normalize: Normalization method
            augmentation: Whether to apply augmentation
        """
        self.root = Path(root)
        self.split = split
        self.layout = layout
        self.feature_size = feature_size
        self.augmentation = augmentation
        
        # Setup paths
        self.layout_path = self.root / 'Layout' / layout
        self.img_path = self.root / 'JPEGImages'
        self.ann_path = self.root / 'PairAnnotation' / split
        
        # Load split file
        split_file = self.layout_path / f'{split}.txt'
        with open(split_file, 'r') as f:
            self.pair_list = [line.strip() for line in f.readlines() if line.strip()]
        
        # Setup transforms
        self.setup_transforms(normalize)
        
    def setup_transforms(self, normalize):
        """Setup image transforms."""
        if normalize == 'imagenet':
            from src import imagenet_stats
            normalize = imagenet_stats
        elif normalize == True:
            normalize = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        self.transform = transforms.Compose([
            transforms.Resize((self.feature_size, self.feature_size)),
            transforms.ToTensor(),
            transforms.Normalize(*normalize) if normalize else transforms.Lambda(lambda x: x),
        ])
    
    def __len__(self):
        return len(self.pair_list)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        pair_line = self.pair_list[idx]
        
        # Parse pair line: "000001-2008_000585-2008_002221:aeroplane"
        parts = pair_line.split('-')
        pair_id = parts[0]
        src_name = parts[1]
        trg_name = parts[2].split(':')[0]
        category = parts[2].split(':')[1]
        
        # Load annotation
        ann_file = self.ann_path / f"{pair_line}.json"
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        
        # Load images
        src_img = self.load_image(category, src_name)
        trg_img = self.load_image(category, trg_name)
        
        # Apply transforms
        src_img = self.transform(src_img)
        trg_img = self.transform(trg_img)
        
        # Get keypoints and convert to flow
        src_kps = torch.tensor(ann['src_kps'], dtype=torch.float32).T  # (2, N)
        trg_kps = torch.tensor(ann['trg_kps'], dtype=torch.float32).T  # (2, N)
        
        # Create flow field from keypoints
        flow = self.keypoints_to_flow(src_kps, trg_kps, ann['src_imsize'][:2])
        
        return {
            'src_img': src_img,
            'trg_img': trg_img,
            'flow': flow,
            'src_kps': src_kps,
            'trg_kps': trg_kps,
            'n_pts': torch.tensor(len(src_kps[0])),
            'category': category,
            'src_imname': src_name + '.jpg',
            'trg_imname': trg_name + '.jpg',
            'src_imsize': torch.tensor(ann['src_imsize'][:2]),
            'trg_imsize': torch.tensor(ann['trg_imsize'][:2]),
            'pckthres': torch.tensor(self.get_pck_threshold(ann['src_bndbox'], ann['src_imsize'][:2])),
            'vpvar': torch.tensor(ann['viewpoint_variation']),
            'scvar': torch.tensor(ann['scale_variation']),
            'trncn': torch.tensor(ann['truncation']),
            'occln': torch.tensor(ann['occlusion']),
        }
    
    def load_image(self, category: str, img_name: str) -> Image.Image:
        """Load image from category directory."""
        img_path = self.img_path / category / f"{img_name}.jpg"
        return Image.open(img_path).convert('RGB')
    
    def keypoints_to_flow(self, src_kps, trg_kps, src_size):
        """Convert keypoints to flow field."""
        h, w = self.feature_size, self.feature_size
        flow = torch.zeros(2, h, w)
        
        # Scale keypoints to feature size
        scale_x = w / src_size[0]
        scale_y = h / src_size[1]
        
        src_kps_scaled = src_kps.clone()
        src_kps_scaled[0] *= scale_x
        src_kps_scaled[1] *= scale_y
        
        trg_kps_scaled = trg_kps.clone()
        trg_kps_scaled[0] *= scale_x
        trg_kps_scaled[1] *= scale_y
        
        # Create sparse flow field
        for i in range(len(src_kps_scaled[0])):
            x, y = int(src_kps_scaled[0, i]), int(src_kps_scaled[1, i])
            if 0 <= x < w and 0 <= y < h:
                flow[0, y, x] = trg_kps_scaled[0, i] - src_kps_scaled[0, i]
                flow[1, y, x] = trg_kps_scaled[1, i] - src_kps_scaled[1, i]
        
        return flow
    
    def get_pck_threshold(self, bbox, imsize):
        """Get PCK threshold based on bounding box."""
        if self.thres == 'bbox':
            return max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 0.1
        else:  # img
            return max(imsize) * 0.1


class SPairDatamodule(BaseDatamodule):
    """
    Datamodule for SPair-71k correspondence dataset.
    
    Returns source image, target image, and flow field for correspondence learning.
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        layout: str = 'large',
        thres: str = 'bbox',
        feature_size: int = 256,
        normalize: Union[bool, str] = True,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True,
        copy_data_local: Optional[str] = None,
    ):
        """
        Initialize SPair datamodule.
        
        Args:
            root: Path to SPair-71k dataset directory
            layout: 'large' or 'small' layout
            thres: Threshold type for PCK evaluation ('bbox' or 'img')
            feature_size: Size to resize images to
            normalize: Normalization method ('imagenet', True, or False)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            shuffle: Whether to shuffle training data
            copy_data_local: Optional local copy path
        """
        self.layout = layout
        self.thres = thres
        self.feature_size = feature_size
        self.normalize = normalize
        super().__init__(root, batch_size, num_workers, shuffle, copy_data_local)

    def setup(self, stage: str = 'fit'):
        """
        Setup train and validation datasets.
        
        Args:
            stage: 'fit' for training, 'test' for testing, 'validate' for validation
        """
        if stage == 'fit':
            # Load training dataset
            self.train_data = SPairDataset(
                self.root,
                split='trn',
                layout=self.layout,
                feature_size=self.feature_size,
                normalize=self.normalize,
                augmentation=True
            )
            
        # Load validation dataset (used for both validation and test)
        self.val_data = SPairDataset(
            self.root,
            split='val',
            layout=self.layout,
            feature_size=self.feature_size,
            normalize=self.normalize,
            augmentation=False
        )

    def collate(self, batch):
        """
        Custom collate function to handle SPair dataset format.
        
        Returns:
            Dictionary with keys: 'src_img', 'trg_img', 'flow', and other metadata
        """
        # The SPair dataset already returns a dictionary with the required fields
        # We just need to ensure proper batching
        return torch.utils.data.default_collate(batch)
