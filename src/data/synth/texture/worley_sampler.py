from typing import Dict, List, Optional, Tuple, Union

import torch


class WorleyParamSampler(object):
    def __init__(
        self,
        texture_type: str = 'worley',
        matching_prob: float = 1.0,
        terrain_matching_prob: float = 1.0,
        texture_scale0: Optional[Dict] = None, # Sample the power/frequency for the worley
        texture_scale1: Optional[Dict] = None,
        mixing_param0: Optional[Dict] = None, # Sample the power/frequency for the worley
        mixing_param1: Optional[Dict] = None,
        terrain_texture_scale0: Optional[Dict] = None,
        terrain_texture_scale1: Optional[Dict] = None,
        terrain_mixing_param0: Optional[Dict] = None,
        terrain_mixing_param1: Optional[Dict] = None,
    ):
        self.texture_type = texture_type
        self.matching_prob = matching_prob
        self.terrain_matching_prob = terrain_matching_prob
        self.texture_scale0 = texture_scale0
        self.texture_scale1 = texture_scale1
        self.mixing_param0 = mixing_param0
        self.mixing_param1 = mixing_param1

        self.terrain_texture_scale0 = terrain_texture_scale0
        self.terrain_texture_scale1 = terrain_texture_scale1
        self.terrain_mixing_param0 = terrain_mixing_param0
        self.terrain_mixing_param1 = terrain_mixing_param1

    def set_probs(self, probs: Dict=None, **kwargs):
        if probs is None: probs = {}
        probs.update(**kwargs)
        for k in ('matching_prob', 'texture_scale0', 'texture_scale1', 'mixing_param0', 'mixing_param1'):
            if k in probs:
                setattr(self, k, probs[k])

    def sample(self, n: int, num_objects: int, rng: torch.Generator):
        """Sample Worley noise parameters for src and target images
        
        Args:
            n: Number of samples
            num_objects: Number of objects
            rng: PyTorch random number generator
            
        Returns:
            Tuple of (src_params, tgt_params) where each is a tensor of shape:
            (batch_size, num_objects, 2) containing scale and texture parameters
        """
        device = rng.device

        # Sample matching pattern - now per object
        is_terrain_matching = bernoulli((n, 1), self.terrain_matching_prob, rng)
        is_matching = bernoulli((n, num_objects-1), self.matching_prob, rng)
        is_matching = torch.cat([is_terrain_matching, is_matching], dim=1)
        # Sample source parameters - we need 4 parameters now (2 texture scales, 2 mixing params)
        src_params = torch.empty(n, num_objects, 4, device=device)
        src_params = uniform(src_params.shape, 0, 1, rng)

        # Create target parameters
        tgt_params = torch.empty_like(src_params)
        
        # Copy matching parameters
        match_idx = is_matching.nonzero()
        tgt_params[match_idx[:,0], match_idx[:,1]] = src_params[match_idx[:,0], match_idx[:,1]]

        # Sample new parameters for non-matching
        non_match_idx = (~is_matching).nonzero()
        if len(non_match_idx) > 0:
            new_params = uniform((len(non_match_idx), 4), 0, 1, rng)
            tgt_params[non_match_idx[:,0], non_match_idx[:,1]] = new_params

        if self.terrain_texture_scale0 is not None:
            src_params[:, 0, 0] = src_params[:, 0, 0] * self.terrain_texture_scale0['alpha']['scale'] + self.terrain_texture_scale0['alpha']['offset']
            tgt_params[:, 0, 0] = tgt_params[:, 0, 0] * self.terrain_texture_scale0['alpha']['scale'] + self.terrain_texture_scale0['alpha']['offset']
        if self.terrain_texture_scale1 is not None:
            src_params[:, 0, 1] = src_params[:, 0, 1] * self.terrain_texture_scale1['alpha']['scale'] + self.terrain_texture_scale1['alpha']['offset']
            tgt_params[:, 0, 1] = tgt_params[:, 0, 1] * self.terrain_texture_scale1['alpha']['scale'] + self.terrain_texture_scale1['alpha']['offset']
        if self.terrain_mixing_param0 is not None:
            src_params[:, 0, 2] = src_params[:, 0, 2] * self.terrain_mixing_param0['alpha']['scale'] + self.terrain_mixing_param0['alpha']['offset']
            tgt_params[:, 0, 2] = tgt_params[:, 0, 2] * self.terrain_mixing_param0['alpha']['scale'] + self.terrain_mixing_param0['alpha']['offset']
        if self.terrain_mixing_param1 is not None:
            src_params[:, 0, 3] = src_params[:, 0, 3] * self.terrain_mixing_param1['alpha']['scale'] + self.terrain_mixing_param1['alpha']['offset']
            tgt_params[:, 0, 3] = tgt_params[:, 0, 3] * self.terrain_mixing_param1['alpha']['scale'] + self.terrain_mixing_param1['alpha']['offset']
        
        
        # Apply scaling to texture_scale0 and texture_scale1
        if self.terrain_texture_scale0 is not None or self.terrain_texture_scale1 is not None or self.terrain_mixing_param0 is not None or self.terrain_mixing_param1 is not None:
            object_indices = slice(1, None) # Apply to objects from index 1 onwards
        else:
            object_indices = slice(None) # Apply to all objects

        if self.texture_scale0 is not None:
            src_params[:, object_indices, 0] = src_params[:, object_indices, 0] * self.texture_scale0['alpha']['scale'] + self.texture_scale0['alpha']['offset']
            tgt_params[:, object_indices, 0] = tgt_params[:, object_indices, 0] * self.texture_scale0['alpha']['scale'] + self.texture_scale0['alpha']['offset']
        
        if self.texture_scale1 is not None:
            src_params[:, object_indices, 1] = src_params[:, object_indices, 1] * self.texture_scale1['alpha']['scale'] + self.texture_scale1['alpha']['offset']
            tgt_params[:, object_indices, 1] = tgt_params[:, object_indices, 1] * self.texture_scale1['alpha']['scale'] + self.texture_scale1['alpha']['offset']
        
        # Apply scaling to mixing_param0 and mixing_param1
        if self.mixing_param0 is not None:
            src_params[:, object_indices, 2] = src_params[:, object_indices, 2] * self.mixing_param0['alpha']['scale'] + self.mixing_param0['alpha']['offset']
            tgt_params[:, object_indices, 2] = tgt_params[:, object_indices, 2] * self.mixing_param0['alpha']['scale'] + self.mixing_param0['alpha']['offset']
        
        if self.mixing_param1 is not None:
            src_params[:, object_indices, 3] = src_params[:, object_indices, 3] * self.mixing_param1['alpha']['scale'] + self.mixing_param1['alpha']['offset']
            tgt_params[:, object_indices, 3] = tgt_params[:, object_indices, 3] * self.mixing_param1['alpha']['scale'] + self.mixing_param1['alpha']['offset']


        

        # Sample in HSV for more vibrant colors
        # H: sample full range [0,1]
        # S: sample high saturation [0.5,1]
        # V: sample high value [0.7,1]
        src_hsv = torch.empty(n, num_objects, 3, device=device)
        src_hsv[:, :, 0] = uniform((n, num_objects), 0, 1, rng)  # Hue
        src_hsv[:, :, 1] = uniform((n, num_objects), 0.5, 1, rng)  # Saturation
        src_hsv[:, :, 2] = uniform((n, num_objects), 0.7, 1, rng)  # Value
        
        # Convert HSV to RGB
        src_base_colors = torch.empty_like(src_hsv)
        h, s, v = src_hsv[:, :, 0], src_hsv[:, :, 1], src_hsv[:, :, 2]
        
        # HSV to RGB conversion
        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c
        
        # Calculate RGB based on hue segment
        zeros = torch.zeros_like(h)
        ones = torch.ones_like(h)
        
        mask0 = (h < 1/6)
        mask1 = (h >= 1/6) & (h < 2/6)
        mask2 = (h >= 2/6) & (h < 3/6)
        mask3 = (h >= 3/6) & (h < 4/6)
        mask4 = (h >= 4/6) & (h < 5/6)
        mask5 = (h >= 5/6)
        
        r = torch.where(mask0, c, torch.where(mask1, x, torch.where(mask2, zeros, torch.where(mask3, zeros, torch.where(mask4, x, c)))))
        g = torch.where(mask0, x, torch.where(mask1, c, torch.where(mask2, c, torch.where(mask3, x, torch.where(mask4, zeros, zeros)))))
        b = torch.where(mask0, zeros, torch.where(mask1, zeros, torch.where(mask2, x, torch.where(mask3, c, torch.where(mask4, c, x)))))
        
        src_base_colors[:, :, 0] = r + m
        src_base_colors[:, :, 1] = g + m
        src_base_colors[:, :, 2] = b + m
        
        # Create target base colors
        tgt_base_colors = torch.empty_like(src_base_colors)
        
        # Copy matching base colors
        tgt_base_colors[match_idx[:,0], match_idx[:,1]] = src_base_colors[match_idx[:,0], match_idx[:,1]]
        
        # Sample new base colors for non-matching
        if len(non_match_idx) > 0:
            new_base_colors = uniform((len(non_match_idx), 3), 0, 1, rng)
            tgt_base_colors[non_match_idx[:,0], non_match_idx[:,1]] = new_base_colors
        
        # Combine parameters and base colors
        src_params = torch.cat([src_params, src_base_colors], dim=2)
        tgt_params = torch.cat([tgt_params, tgt_base_colors], dim=2)

        return src_params, tgt_params

    def __repr__(self):
        return '\n'.join((
            'WorleyParamSampler()',
            f'  matching_prob: {self.matching_prob}',
            f'  texture_scale0: {self.texture_scale0}',
            f'  texture_scale1: {self.texture_scale1}',
            f'  mixing_param0: {self.mixing_param0}',
            f'  mixing_param1: {self.mixing_param1}'
        ))


def bernoulli(n, p, rng):
    return torch.empty(n, dtype=torch.bool, device=rng.device).bernoulli_(p, generator=rng)


def uniform(shape, vmin, vmax, rng):
    return torch.empty(shape, device=rng.device).uniform_(vmin, vmax, generator=rng)


def normal(shape, loc, scale, rng):
    return torch.empty(shape, device=rng.device).normal_(loc, scale, generator=rng)