"""
Reusable flow filtering utilities for filtering downsampled flow vectors by length.

This module provides utilities to filter flow vectors based on their magnitude (L2 norm).
Designed to work with downsampled flow where invalid regions are represented as (0, 0).

Usage:
    # Filter downsampled flow in batches
    filter = FlowLengthFilter(min_flow_length=10.0, max_flow_length=200.0)
    batch['flow'] = filter.filter_batch_flow(batch['flow'])  # flow: (B, 2, H, W)
"""

import torch
from typing import Optional


class FlowLengthFilter:
    """
    Filter downsampled flow vectors by their length (L2 norm).
    
    Invalid flow regions (marked as (0, 0)) are preserved. Only valid flow vectors
    are filtered by length. Vectors that don't meet the length criteria are set to (0, 0).
    
    This is designed to work with downsampled flow from kps_to_flow or similar operations.
    """
    
    def __init__(self, min_flow_length: Optional[float] = None, max_flow_length: Optional[float] = None):
        """
        Initialize flow length filter.
        
        Args:
            min_flow_length: Minimum flow vector length. 
                For SPair and other keypoint-based datasets using kps_to_flow: 
                flow is in feature grid units (normalized by downsampling factor).
                For feature_size=32 and img_size=512: 1.0 feature unit = 16 pixels.
                Flow vectors shorter than this will be set to (0, 0). If None, no minimum filter.
            max_flow_length: Maximum flow vector length (same units as min_flow_length).
                Flow vectors longer than this will be set to (0, 0). If None, no maximum filter.
        """
        self.min_flow_length = min_flow_length
        self.max_flow_length = max_flow_length
    
    def filter_batch_flow(self, flow: torch.Tensor) -> torch.Tensor:
        """
        Filter batched downsampled flow tensor by vector length.
        
        Invalid flow regions (marked as (0, 0)) are preserved. Only valid flow vectors
        are filtered by length. Vectors that don't meet the length criteria are set to (0, 0).
        
        Args:
            flow: Batched downsampled flow tensor of shape (B, 2, H, W) in feature grid units.
                Invalid regions should be marked as (0, 0).
        
        Returns:
            Filtered flow tensor with same shape. Invalid vectors (either originally
            invalid or filtered out) are set to (0, 0).
        """
        if self.min_flow_length is None and self.max_flow_length is None:
            return flow
        
        # Compute flow lengths for each vector in the batch
        flow_lengths = torch.norm(flow, dim=1)  # (B, H, W)
        
        # Create mask for originally valid flow values (not (0, 0))
        # Consider a flow vector valid if its length is greater than a small threshold
        # This distinguishes (0, 0) from very small but valid flow
        valid_mask = flow_lengths > 1e-6  # (B, H, W)
        
        # Create length-based filter mask
        length_mask = torch.ones_like(valid_mask, dtype=torch.bool)
        if self.min_flow_length is not None:
            length_mask = length_mask & (flow_lengths >= self.min_flow_length)
        if self.max_flow_length is not None:
            length_mask = length_mask & (flow_lengths <= self.max_flow_length)
        
        # Combined mask: must be originally valid AND pass length filter
        final_valid_mask = valid_mask & length_mask  # (B, H, W)
        
        # Set invalid flow to (0, 0)
        flow_filtered = flow.clone()
        # Expand mask to match flow shape (B, 2, H, W)
        final_valid_mask_expanded = final_valid_mask.unsqueeze(1).expand_as(flow_filtered)
        flow_filtered[~final_valid_mask_expanded] = 0.0
        
        return flow_filtered

