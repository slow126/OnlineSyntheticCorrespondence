"""
Wrapper class to make OnlineGenerationDataset compatible with CATs++ model.
This converts geometry/normal maps to RGB images and generates corresponding keypoints.
"""

import os
import random
from typing import Dict, Literal, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from .online_dataset import OnlineGenerationDataset
from .base import ComponentsBase
from ..datamodule import OnlineComponentsDatamodule

# Import CATs++ flow generation
import sys
sys.path.append('models/CATs-PlusPlus')
from data.keypoint_to_flow import KeypointToFlow


class CATsSyntheticDataset(Dataset):
    """
    Wrapper class that makes OnlineGenerationDataset compatible with CATs++ training.
    
    This class:
    1. Uses OnlineGenerationDataset to generate geometry/normal pairs
    2. Converts geometry/normal maps to RGB images
    3. Generates corresponding keypoints based on geometry correspondence
    4. Returns data in the format expected by CATs++ model
    """
    
    def __init__(
        self,
        shader_code_path: str = None,
        antialias: int = 3,
        num_samples: int = 1000,
        random_flip: float = 0.0,
        random_swap: bool = True,
        julia_sampler: Dict = None,
        angle_sampler: Dict = None,
        scale_sampler: Dict = None,
        mandelbulb_sampler: Dict = None,
        size: int = 256,
        crop: Literal['center', 'none', 'random'] = 'none',
        single_process: bool = True,
        seed: int = 987654321,
        shaders: Dict = None,
        # CATs++ specific parameters
        benchmark: str = 'synthetic',
        datapath: str = '',
        thres: str = 'img',
        device: str = 'cuda',
        split: str = 'trn',
        augmentation: bool = True,
        feature_size: int = 16,
        max_pts: int = 40,
        imside: int = 512,
    ):
        super(CATsSyntheticDataset, self).__init__()
        
        # Set up default shaders if not provided
        if shaders is None:
            # Use the working configuration from test_datamodule.py
            shaders = {
                'program0': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/terrain_sdf.glsl',
                'program1': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c',
                'program2': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/mandelbulb.glsl',
                'program3': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/deformed_sphere.glsl',
                'merge': '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/merge.glsl',
            }
        
        # Set default shader_code_path if not provided
        if shader_code_path is None:
            shader_code_path = '/home/spencer/Deployments/synthetic-correspondence/rendering/shaders/quatJulia.c'
        
        # Initialize the underlying online dataset with proper rendering
        self.online_dataset = OnlineGenerationDataset(
            shader_code_path=shader_code_path,
            antialias=antialias,
            num_samples=num_samples,
            random_flip=random_flip,
            random_swap=random_swap,
            julia_sampler=julia_sampler,
            angle_sampler=angle_sampler,
            scale_sampler=scale_sampler,
            mandelbulb_sampler=mandelbulb_sampler,
            size=size,
            crop=crop,
            single_process=single_process,
            seed=seed,
            shaders=shaders,
        )
        
        # CATs++ specific attributes
        self.benchmark = benchmark
        self.datapath = datapath
        self.thres = thres
        self.device = device
        self.split = split
        self.augmentation = augmentation
        self.feature_size = feature_size
        self.max_pts = max_pts
        self.imside = imside
        self.range_ts = torch.arange(self.max_pts)
        
        # Set up transforms for CATs++ compatibility
        if split == 'trn' and augmentation:
            self.transform = A.Compose([
                A.Resize(self.imside, self.imside),  # Ensure consistent resizing
                A.ToGray(p=0.2),
                A.Posterize(p=0.2),
                A.Equalize(p=0.2),
                A.augmentations.transforms.Sharpen(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Solarize(p=0.2),
                A.ColorJitter(p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                A.pytorch.transforms.ToTensorV2(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.imside, self.imside)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # For compatibility with CATs++ training script
        self.train_data = list(range(num_samples))
        self.src_imnames = [f'synthetic_src_{i}' for i in range(num_samples)]
        self.trg_imnames = [f'synthetic_trg_{i}' for i in range(num_samples)]
        self.cls = ['synthetic']
        self.cls_ids = [0] * num_samples
        
        # Initialize keypoint lists (will be generated on-the-fly)
        self.src_kps = [None] * num_samples
        self.trg_kps = [None] * num_samples
        
        # Set up random number generator for keypoint generation
        self.rng = torch.Generator().manual_seed(seed)
        
        # Initialize keypoint to flow converter (same as in CATs++ datasets)
        self.kps_to_flow = KeypointToFlow(
            receptive_field_size=35, 
            jsz=imside//feature_size,  # Use the provided feature size
            feat_size=feature_size,    # Use the provided feature size
            img_size=imside
        )
        
    def __len__(self):
        return len(self.online_dataset)
    
    def geometry_to_rgb(self, geometry) -> np.ndarray:
        """
        Convert geometry map to RGB image.
        Uses the geometry coordinates as RGB values.
        """
        # Convert to numpy if it's a tensor
        if torch.is_tensor(geometry):
            geometry = geometry.numpy()
        
        # Normalize geometry to [0, 1] range
        geometry_norm = (geometry - geometry.min()) / (geometry.max() - geometry.min())
        
        # Convert to uint8
        geometry_uint8 = (geometry_norm * 255).astype(np.uint8)
        
        return geometry_uint8
    
    def normals_to_rgb(self, normals) -> np.ndarray:
        """
        Convert normal map to RGB image.
        Maps normal vectors to RGB colors.
        """
        # Convert to numpy if it's a tensor
        if torch.is_tensor(normals):
            normals = normals.numpy()
        
        # Normalize normals to [0, 1] range (normals are typically in [-1, 1])
        normals_norm = (normals + 1.0) / 2.0
        normals_norm = np.clip(normals_norm, 0, 1)
        
        # Convert to uint8
        normals_uint8 = (normals_norm * 255).astype(np.uint8)
        
        return normals_uint8
    
    def generate_keypoints(self, geometry, num_points: int = None) -> torch.Tensor:
        """
        Generate keypoints based on geometry map.
        Samples points from areas with significant geometry variation.
        """
        if num_points is None:
            num_points = min(self.max_pts, 20)  # Default to 20 points
        
        # Convert to numpy if it's a tensor
        if torch.is_tensor(geometry):
            geometry = geometry.numpy()
        
        h, w = geometry.shape[:2]
        
        # Compute gradient magnitude to find interesting points
        if len(geometry.shape) == 3:
            # Use depth channel if available
            depth = geometry[:, :, 2] if geometry.shape[2] >= 3 else geometry[:, :, 0]
        else:
            depth = geometry
        
        # Compute gradients
        grad_x = np.gradient(depth, axis=1)
        grad_y = np.gradient(depth, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find points with high gradient (edges/features)
        threshold = np.percentile(gradient_magnitude, 80)  # Top 20% of gradient values
        interesting_points = np.where(gradient_magnitude > threshold)
        
        if len(interesting_points[0]) == 0:
            # Fallback: sample random points
            y_coords = torch.randint(0, h, (num_points,), generator=self.rng)
            x_coords = torch.randint(0, w, (num_points,), generator=self.rng)
        else:
            # Sample from interesting points
            num_interesting = len(interesting_points[0])
            if num_interesting >= num_points:
                # Randomly sample from interesting points
                indices = torch.randperm(num_interesting, generator=self.rng)[:num_points]
                y_coords = torch.tensor(interesting_points[0][indices])
                x_coords = torch.tensor(interesting_points[1][indices])
            else:
                # Use all interesting points and fill with random ones
                y_coords = torch.tensor(interesting_points[0])
                x_coords = torch.tensor(interesting_points[1])
                
                # Fill remaining with random points
                remaining = num_points - num_interesting
                y_rand = torch.randint(0, h, (remaining,), generator=self.rng)
                x_rand = torch.randint(0, w, (remaining,), generator=self.rng)
                
                y_coords = torch.cat([y_coords, y_rand])
                x_coords = torch.cat([x_coords, x_rand])
        
        # Stack coordinates
        keypoints = torch.stack([x_coords.float(), y_coords.float()])
        
        return keypoints
    
    def generate_keypoints_from_image(self, image_tensor, num_points: int = None) -> torch.Tensor:
        """
        Generate keypoints based on rendered image tensor.
        Samples points from areas with significant intensity variation.
        """
        if num_points is None:
            num_points = min(self.max_pts, 20)  # Default to 20 points
        
        # Convert tensor to numpy
        if torch.is_tensor(image_tensor):
            image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image_tensor
        
        h, w = image_np.shape[:2]
        
        # Convert to grayscale for gradient computation
        if len(image_np.shape) == 3:
            gray = np.mean(image_np, axis=2)
        else:
            gray = image_np
        
        # Compute gradient magnitude to find interesting points
        grad_x = np.gradient(gray, axis=1)
        grad_y = np.gradient(gray, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find points with high gradient (edges/features)
        threshold = np.percentile(gradient_magnitude, 80)  # Top 20% of gradient values
        interesting_points = np.where(gradient_magnitude > threshold)
        
        if len(interesting_points[0]) == 0:
            # Fallback: sample random points
            y_coords = torch.randint(0, h, (num_points,), generator=self.rng)
            x_coords = torch.randint(0, w, (num_points,), generator=self.rng)
        else:
            # Sample from interesting points
            num_interesting = len(interesting_points[0])
            if num_interesting >= num_points:
                # Randomly sample from interesting points
                indices = torch.randperm(num_interesting, generator=self.rng)[:num_points]
                y_coords = torch.tensor(interesting_points[0][indices])
                x_coords = torch.tensor(interesting_points[1][indices])
            else:
                # Use all interesting points and fill with random ones
                y_coords = torch.tensor(interesting_points[0])
                x_coords = torch.tensor(interesting_points[1])
                
                # Fill remaining with random points
                remaining = num_points - num_interesting
                y_rand = torch.randint(0, h, (remaining,), generator=self.rng)
                x_rand = torch.randint(0, w, (remaining,), generator=self.rng)
                
                y_coords = torch.cat([y_coords, y_rand])
                x_coords = torch.cat([x_coords, x_rand])
        
        # Stack coordinates
        keypoints = torch.stack([x_coords.float(), y_coords.float()])
        
        return keypoints
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset in CATs++ format using the proper rendering pipeline.
        """
        # Get data from online dataset (this should have proper rendering)
        data = self.online_dataset[idx]
        
        # Extract source and target data
        src_data = data[0]
        trg_data = data[1]
        
        # The online dataset should provide rendered images, not just geometry
        # Check if we have rendered images or need to convert geometry
        if 'rendered' in src_data and 'rendered' in trg_data:
            # Use the rendered images directly
            src_img = src_data['rendered']
            trg_img = trg_data['rendered']
        else:
            # Fallback: convert geometry to RGB (this is what we had before)
            src_img = self.geometry_to_rgb(src_data['geometry'])
            trg_img = self.geometry_to_rgb(trg_data['geometry'])
        
        # Convert to PIL Images if needed
        if isinstance(src_img, np.ndarray):
            src_pil = Image.fromarray(src_img)
            trg_pil = Image.fromarray(trg_img)
        else:
            src_pil = src_img
            trg_pil = trg_img
        
        # Apply transforms
        if self.split == 'trn' and self.augmentation:
            if isinstance(src_img, np.ndarray):
                src_tensor = self.transform(image=src_img)['image']
                trg_tensor = self.transform(image=trg_img)['image']
            else:
                src_tensor = self.transform(image=np.array(src_pil))['image']
                trg_tensor = self.transform(image=np.array(trg_pil))['image']
        else:
            src_tensor = self.transform(src_pil)
            trg_tensor = self.transform(trg_pil)
        
        # Generate keypoints based on the images
        if isinstance(src_img, np.ndarray):
            src_kps = self.generate_keypoints_from_image(torch.from_numpy(src_img).permute(2, 0, 1))
            trg_kps = self.generate_keypoints_from_image(torch.from_numpy(trg_img).permute(2, 0, 1))
        else:
            src_kps = self.generate_keypoints_from_image(torch.from_numpy(np.array(src_pil)).permute(2, 0, 1))
            trg_kps = self.generate_keypoints_from_image(torch.from_numpy(np.array(trg_pil)).permute(2, 0, 1))
        
        # Ensure we have the right number of points
        num_pts = min(src_kps.shape[1], self.max_pts)
        src_kps = src_kps[:, :num_pts]
        trg_kps = trg_kps[:, :num_pts]
        
        # Pad keypoints to max_pts if necessary
        if num_pts < self.max_pts:
            pad_size = self.max_pts - num_pts
            src_pad = torch.zeros(2, pad_size) - 1
            trg_pad = torch.zeros(2, pad_size) - 1
            src_kps = torch.cat([src_kps, src_pad], dim=1)
            trg_kps = torch.cat([trg_kps, trg_pad], dim=1)
        
        # Scale keypoints to match resized image size (imside x imside)
        if isinstance(src_img, np.ndarray):
            original_h, original_w = src_img.shape[:2]
        else:
            original_h, original_w = src_pil.size[1], src_pil.size[0]
        
        src_kps[0] = src_kps[0] * (self.imside / original_w)
        src_kps[1] = src_kps[1] * (self.imside / original_h)
        trg_kps[0] = trg_kps[0] * (self.imside / original_w)
        trg_kps[1] = trg_kps[1] * (self.imside / original_h)
        
        # Create batch dictionary in CATs++ format
        batch = {
            'src_imname': self.src_imnames[idx],
            'trg_imname': self.trg_imnames[idx],
            'category_id': self.cls_ids[idx],
            'category': self.cls[0],
            'src_imsize': (self.imside, self.imside),  # Use resized dimensions
            'trg_imsize': (self.imside, self.imside),  # Use resized dimensions
            'src_img': src_tensor,
            'trg_img': trg_tensor,
            'src_kps': src_kps,
            'trg_kps': trg_kps,
            'n_pts': torch.tensor(num_pts),
            'datalen': len(self.train_data),
        }
        
        # Generate flow from keypoints (required by CATs++ training)
        batch['flow'] = self.kps_to_flow(batch)
        
        # Add PCK threshold for evaluation
        batch['pckthres'] = self.get_pckthres(batch, batch['src_imsize'])
        
        return batch
    
    def get_pckthres(self, batch, imsize):
        """
        Compute PCK threshold for evaluation.
        """
        if self.thres == 'bbox':
            # For synthetic data, use image size as threshold
            pckthres = torch.tensor(max(imsize[0], imsize[1]))
        elif self.thres == 'img':
            imsize_t = batch['src_img'].size()
            pckthres = torch.tensor(max(imsize_t[1], imsize_t[2]))
        else:
            raise Exception('Invalid pck threshold type: %s' % self.thres)
        return pckthres.float()
