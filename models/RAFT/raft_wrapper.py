"""
RAFT model wrapper for dense correspondence training.

This wrapper handles:
- Image normalization conversion (ImageNet-normalized -> 0-255 range)
- Extracting final prediction from sequence of predictions
- Interface compatibility with existing training pipeline
"""

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_RAFT_CLASS = None


def _is_within_root(path: Optional[str], root: Path) -> bool:
    if not path:
        return False
    try:
        path_obj = Path(path).resolve()
        root_obj = root.resolve()
    except OSError:
        return False
    return root_obj == path_obj or root_obj in path_obj.parents


def _purge_modules(module_names, root: Path) -> None:
    for name in module_names:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        module_path = getattr(mod, "__file__", None)
        if module_path and not _is_within_root(module_path, root):
            del sys.modules[name]


def _load_raft_class():
    global _RAFT_CLASS

    raft_dir = Path(__file__).parent.resolve()
    core_dir = raft_dir / "core"

    if _RAFT_CLASS is not None:
        module_name = getattr(_RAFT_CLASS, "__module__", "")
        module = sys.modules.get(module_name)
        module_path = getattr(module, "__file__", None)
        if module_path and _is_within_root(module_path, core_dir):
            return _RAFT_CLASS
        _RAFT_CLASS = None

    # Ensure FlowFormer copies don't shadow RAFT's local modules.
    _purge_modules(["raft", "update", "extractor", "corr"], core_dir)

    core_path = str(core_dir)
    if core_path not in sys.path:
        sys.path.insert(0, core_path)

    from raft import RAFT  # pylint: disable=import-error

    _RAFT_CLASS = RAFT
    return _RAFT_CLASS


class RAFTWrapper(nn.Module):
    """
    Wrapper for RAFT model that adapts it to the training pipeline.
    
    Converts ImageNet-normalized images to 0-255 range expected by RAFT,
    and extracts the final prediction from the sequence of predictions.
    """
    
    def __init__(
        self,
        small: bool = False,
        iters: int = 12,
        alternate_corr: bool = False,
        mixed_precision: bool = False,
        dropout: float = 0.0,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize RAFT wrapper.
        
        Args:
            small: If True, use RAFT-small architecture
            iters: Number of refinement iterations
            alternate_corr: Use alternate correlation implementation
            mixed_precision: Enable mixed precision training
            dropout: Dropout rate
            pretrained_path: Optional path to pretrained RAFT checkpoint
        """
        super().__init__()
        raft_class = _load_raft_class()
        
        # Create args object for RAFT
        # RAFT expects an object that supports both attribute access and 'in' operator
        class Args:
            def __init__(self):
                self.small = small
                self.alternate_corr = alternate_corr
                self.mixed_precision = mixed_precision
                self.dropout = dropout
            
            def __contains__(self, key):
                """Support 'in' operator for checking if attribute exists"""
                return hasattr(self, key)
        
        args = Args()
        
        # Initialize RAFT model
        self.raft = raft_class(args)
        self.iters = iters
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            print(f"Loading pretrained RAFT weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle DataParallel wrapper if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
            self.raft.load_state_dict(state_dict, strict=False)
            print("Pretrained RAFT weights loaded")
        
        # ImageNet normalization constants (for conversion)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, trg_img: torch.Tensor, src_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through RAFT model.
        
        Args:
            trg_img: Target image tensor (B, 3, H, W) - ImageNet normalized
            src_img: Source image tensor (B, 3, H, W) - ImageNet normalized
            
        Returns:
            Flow tensor (B, 2, H, W) - final prediction from sequence
        """
        # Convert ImageNet-normalized images to 0-255 range
        # Reverse normalization: img = (img * std) + mean, then scale to 0-255
        trg_img_255 = ((trg_img * self.std + self.mean) * 255.0).clamp(0, 255)
        src_img_255 = ((src_img * self.std + self.mean) * 255.0).clamp(0, 255)
        
        # Forward pass through RAFT - returns list of predictions
        # Note: RAFT expects (image1, image2) where image1 is source, image2 is target
        flow_predictions = self.raft(src_img_255, trg_img_255, iters=self.iters, upsample=True, test_mode=False)
        
        # Extract final prediction for dense correspondence
        # flow_predictions is a list, last element is the final refined prediction
        final_flow = flow_predictions[-1]  # (B, 2, H, W)
        
        return final_flow
