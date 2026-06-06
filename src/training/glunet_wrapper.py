"""
GLU-Net model wrapper for the unified correspondence training pipeline.

This adapts ``src.model.glunet.sem_glunet.SemanticGLUNet`` to the interface
expected by ``CorrespondenceLightningModule`` / ``train_lightning.py``:

    forward(trg_img, src_img) -> flow  [B, 2, H, W]   (used for eval)

Unlike the RAFT / FlowFormer wrappers, GLU-Net expects **ImageNet-normalized**
inputs (its ``FeatureExtractor`` uses a torchvision ResNet with ImageNet
weights, and ``standardize=True`` instance-normalizes the *features*, not the
images). So no 0-255 conversion is performed here -- the same normalization
convention as CATs++ applies.

For training, GLU-Net keeps its native multi-scale supervision: ``forward_train``
returns the full pyramid of predicted flows and ``loss_fn`` is the
``MultiscaleEndpointError`` used by the standalone GLU-Net lightning module.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.model.glunet import sem_glunet
from src.objectives import MultiscaleEndpointError


class GLUNetWrapper(nn.Module):
    """Adapter exposing GLU-Net through the ``forward(trg_img, src_img)`` interface.

    Args:
        pretrained_backbone: If True, initialize the ResNet feature extractor with
            ImageNet weights (``IMAGENET1K_V2``); otherwise train from scratch.
        freeze: If True, freeze the feature-extractor backbone (handled natively by
            ``SemanticGLUNet``'s ``FeatureExtractor``).
        model_name: torchvision backbone name for the feature extractor.
        local_window_size: GLU-Net local correlation window size.
        decoder_dense_connect: Whether decoders use dense connections.
        weights: Optional explicit weights spec/path forwarded to SemanticGLUNet.
            If None, derived from ``pretrained_backbone``.
        **model_kwargs: Additional kwargs forwarded to ``SemanticGLUNet``.
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        freeze: bool = False,
        model_name: str = 'resnet50',
        local_window_size: int = 9,
        decoder_dense_connect: bool = False,
        weights: Optional[str] = None,
        **model_kwargs,
    ):
        super().__init__()

        if weights is None:
            weights = 'IMAGENET1K_V2' if pretrained_backbone else None

        self.glunet = sem_glunet.SemanticGLUNet(
            model_name=model_name,
            weights=weights,
            freeze=freeze,
            local_window_size=local_window_size,
            decoder_dense_connect=decoder_dense_connect,
            **model_kwargs,
        )

        # Native GLU-Net multi-scale endpoint-error loss (matches GLUNet lightning module).
        self.loss_fn = MultiscaleEndpointError(reduction='batch_sum')

    def _predict_pyramid(self, trg_img: torch.Tensor, src_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run GLU-Net and append the full-resolution prediction.

        Note the argument flip: the unified pipeline passes ``(trg_img, src_img)``,
        but GLU-Net's native order is ``(src_img, trg_img)`` (same convention the
        RAFT/FlowFormer wrappers use when they call ``model(src, trg)``).
        """
        preds = self.glunet(src_img, trg_img)
        preds['full'] = self.glunet.resize(preds['level3'], size=src_img.shape[-2:])
        return preds

    def forward(self, trg_img: torch.Tensor, src_img: torch.Tensor) -> torch.Tensor:
        """Return the full-resolution predicted flow ``[B, 2, H, W]`` (for eval)."""
        return self._predict_pyramid(trg_img, src_img)['full']

    def forward_train(self, trg_img: torch.Tensor, src_img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return the full flow pyramid (coarsest->finest) for the multi-scale loss."""
        return self._predict_pyramid(trg_img, src_img)
