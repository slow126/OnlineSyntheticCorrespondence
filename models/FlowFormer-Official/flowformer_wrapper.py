"""
FlowFormer model wrapper for dense correspondence training.

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

_FLOWFORMER_BUILD = None
_FLOWFORMER_GET_CFG = None


def _is_within_root(path: Optional[str], root: Path) -> bool:
    if not path:
        return False
    try:
        path_obj = Path(path).resolve()
        root_obj = root.resolve()
    except OSError:
        return False
    return root_obj == path_obj or root_obj in path_obj.parents


def _purge_package(package: str, root: Path) -> None:
    for name in list(sys.modules.keys()):
        if name == package or name.startswith(f"{package}."):
            mod = sys.modules.get(name)
            module_path = getattr(mod, "__file__", None)
            if module_path and not _is_within_root(module_path, root):
                del sys.modules[name]


def _load_flowformer_modules():
    global _FLOWFORMER_BUILD
    global _FLOWFORMER_GET_CFG

    if _FLOWFORMER_BUILD is not None and _FLOWFORMER_GET_CFG is not None:
        return _FLOWFORMER_BUILD, _FLOWFORMER_GET_CFG

    flowformer_dir = Path(__file__).parent.resolve()
    # Avoid RAFT's core package shadowing FlowFormer's core (both use "core").
    _purge_package("core", flowformer_dir)
    _purge_package("configs", flowformer_dir)

    flowformer_path = str(flowformer_dir)
    if flowformer_path not in sys.path:
        sys.path.insert(0, flowformer_path)

    from core.FlowFormer import build_flowformer  # pylint: disable=import-error
    from configs.default import get_cfg  # pylint: disable=import-error

    _FLOWFORMER_BUILD = build_flowformer
    _FLOWFORMER_GET_CFG = get_cfg
    return build_flowformer, get_cfg


class FlowFormerWrapper(nn.Module):
    """
    Wrapper for FlowFormer model that adapts it to the training pipeline.
    
    Converts ImageNet-normalized images to 0-255 range expected by FlowFormer,
    and extracts the final prediction from the sequence of predictions.
    """
    
    def __init__(
        self,
        pretrain: bool = True,
        iters: int = 12,
        pretrained_path: Optional[str] = None,
    ):
        """
        Initialize FlowFormer wrapper.
        
        Args:
            pretrain: If True, use pretrained Twins-SVT encoder
            iters: Number of refinement iterations (used by decoder)
            pretrained_path: Optional path to pretrained FlowFormer checkpoint
        """
        super().__init__()

        build_flowformer, get_cfg = _load_flowformer_modules()
        
        # Get default config
        cfg = get_cfg()
        
        # Override config parameters
        cfg.latentcostformer.pretrain = pretrain
        # Note: decoder_depth controls number of iterations (default is 12)
        # If iters is different, we could override cfg.latentcostformer.decoder_depth = iters
        # But for now, we'll use the default decoder_depth from config
        
        # Build FlowFormer model
        self.flowformer = build_flowformer(cfg)
        self.iters = iters  # Store for reference (decoder_depth in config controls actual iterations)
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            print(f"Loading pretrained FlowFormer weights from: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location='cpu')
            # Handle DataParallel wrapper if present
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items() if k.startswith('module.')}
            self.flowformer.load_state_dict(state_dict, strict=False)
            print("Pretrained FlowFormer weights loaded")
        
        # ImageNet normalization constants (for conversion)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, trg_img: torch.Tensor, src_img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FlowFormer model.
        
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
        
        # Forward pass through FlowFormer - returns list of predictions
        # Note: FlowFormer expects (image1, image2) where image1 is source, image2 is target
        flow_predictions = self.flowformer(src_img_255, trg_img_255, output={}, flow_init=None)
        
        # Extract final prediction for dense correspondence
        # flow_predictions is a list, last element is the final refined prediction
        final_flow = flow_predictions[-1]  # (B, 2, H, W)
        
        return final_flow
