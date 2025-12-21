"""
Feature encoders for extracting deep features from images for MMD calculation.

This module provides abstract and concrete encoder implementations for extracting
features from images using various backbones (ResNet101, Dino, etc.).
"""

import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod
from functools import reduce
from operator import add

from torchvision.models import resnet
from models.CATs_PlusPlus.models.base.feature import extract_feat_res


class BaseFeatureEncoder(ABC):
    """Abstract base class for feature encoders."""
    
    @abstractmethod
    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image batch.
        
        Args:
            img: Input image tensor [B, 3, H, W]
        
        Returns:
            Flattened features [B*H'*W', C] where C is feature dimension
        """
        pass
    
    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the feature dimension (C)."""
        pass


class ResNet101Encoder(BaseFeatureEncoder):
    """ResNet101 encoder extracting pretrained features (no CATs-trained components)."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize ResNet101 encoder with pretrained backbone.
        
        Args:
            device: Device to run encoder on (default: cuda if available)
        """
        super().__init__()
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Load pretrained ResNet101 backbone (same as CATsImproved)
        self.backbone = resnet.resnet101(pretrained=True)
        self.backbone.eval()
        self.backbone.to(device)
        
        # Feature extraction parameters (same as CATsImproved for resnet101)
        self.feat_ids = list(range(4, 34))
        nbottlenecks = [3, 4, 23, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        
        # Feature dimension: 512 channels from ResNet layer 2
        self._feature_dim = 512
        
        print(f"Initialized ResNet101Encoder on device: {device}")
        print(f"Feature dimension: {self._feature_dim}")
    
    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """
        Extract features from image batch.
        
        Args:
            img: Input image tensor [B, 3, H, W]
        
        Returns:
            Flattened features [B*H'*W', 512]
        """
        # Move image to device
        img = img.to(self.device)
        
        with torch.no_grad():
            # Extract intermediate features using same method as CATsImproved
            feats = extract_feat_res(img, self.backbone, self.feat_ids, 
                                    self.bottleneck_ids, self.lids)
            
            # Stack features into l2 level (same as CATsImproved.stack_feats)
            # feats_l2 contains features from layer 2
            feats_l2 = torch.stack(feats[-self.stack_ids[2]:-self.stack_ids[1]]).transpose(0, 1)
            
            # Take last layer: [B, 512, H', W']
            feats_l2_last = feats_l2[:, -1]  # [B, 512, H', W']
            
            # Flatten spatial dimensions: [B, 512, H', W'] -> [B*H'*W', 512]
            B, C, H, W = feats_l2_last.shape
            features_flat = feats_l2_last.permute(0, 2, 3, 1).contiguous()  # [B, H', W', 512]
            features_flat = features_flat.view(B * H * W, C)  # [B*H'*W', 512]
            
            return features_flat
    
    @property
    def feature_dim(self) -> int:
        """Return the feature dimension (512 for ResNet101 layer 2)."""
        return self._feature_dim


class DinoV3Encoder(BaseFeatureEncoder):
    """DINOv3 encoder extracting spatial features."""

    def __init__(
        self,
        device: Optional[torch.device] = None,
        model_name: str = "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        resize_size: int = 512,
    ):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model_name = model_name
        self.resize_size = resize_size

        # Import locally to avoid transformers dependency for non-DINO runs.
        from models.DinoV3.DinoV3 import DinoV3

        self.dino = DinoV3(pretrained_model_name=model_name, resize_size=resize_size)
        try:
            self.dino.model.to(self.device)
        except Exception:
            pass

        if hasattr(self.dino.model, "config") and hasattr(self.dino.model.config, "hidden_size"):
            self._feature_dim = int(self.dino.model.config.hidden_size)
        else:
            self._feature_dim = 768

        print(f"Initialized DinoV3Encoder on device: {device}")
        print(f"Feature dimension: {self._feature_dim}")

    def extract_features(self, img: torch.Tensor) -> torch.Tensor:
        img = img.to(self.device)
        with torch.no_grad():
            feats = self.dino.get_spatial_features(img)  # [B, N, C]
        if feats.dim() == 3:
            bsz, n_tokens, dim = feats.shape
            feats = feats.reshape(bsz * n_tokens, dim)
        return feats

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
