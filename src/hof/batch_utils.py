"""Batch utilities for extracting flow + masks."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import torch


FLOW_KEYS = ("flow_full", "flow")
MASK_KEYS = ("valid_flow_mask", "valid_mask", "mask")


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def extract_flow_from_batch(batch: Dict, index: int = 0, prefer_full: bool = True) -> Optional[np.ndarray]:
    """Extract single flow [H, W, 2] from a batch dict."""
    flow = None
    if prefer_full and "flow_full" in batch:
        flow = batch.get("flow_full")
    if flow is None:
        flow = batch.get("flow")
    if flow is None:
        return None

    flow = _to_numpy(flow)
    if flow.ndim == 4:
        flow = flow[index]
    if flow.ndim == 3 and flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Unexpected flow shape: {flow.shape}")
    return flow.astype(np.float32, copy=False)


def extract_valid_mask_from_batch(batch: Dict, index: int = 0) -> Optional[np.ndarray]:
    """Extract [H, W] valid mask if present."""
    for key in MASK_KEYS:
        if key in batch:
            mask = _to_numpy(batch[key])
            if mask.ndim == 3:
                mask = mask[index]
            if mask.ndim != 2:
                return None
            if mask.dtype != bool:
                mask = mask > 0
            return mask.astype(bool, copy=False)
    return None
