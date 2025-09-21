# import torch
# import colorsys


# class BGColorSampler(object):
#     def __init__(self, matching_prob: float = 0.5, 
#                  hue_offset: float = 0.0, hue_scale: float = 1.0, 
#                  saturation_offset: float = 0.0, saturation_scale: 
#                  float = 1.0, value_offset: float = 0.0, value_scale: float = 1.0):
#         self.matching_prob = matching_prob
#         self.hue_offset = hue_offset
#         self.hue_scale = hue_scale
#         self.saturation_offset = saturation_offset
#         self.saturation_scale = saturation_scale
#         self.value_offset = value_offset
#         self.value_scale = value_scale

#     def sample(self, n: int, rng: torch.Generator):
#         """Sample solid RGB colors for source and target with a probability of matching.

#         Args:
#             n: Number of samples (batch size).
#             rng: PyTorch random number generator.

#         Returns:
#             Tuple[Tensor, Tensor]: A tuple containing two tensors of shape (batch_size, 3)
#                                  representing source and target RGB colors.
#         """
#         device = rng.device

#         # Sample base HSV colors (for source and potentially target)
#         base_hsv_colors = torch.empty(n, 3, device=device)
#         base_hsv_colors[:, 0] = torch.rand(n, generator=rng, device=device) * self.hue_scale + self.hue_offset  # Hue
#         base_hsv_colors[:, 1] = torch.rand(n, generator=rng, device=device) * self.saturation_scale + self.saturation_offset  # Saturation [0.5, 1]
#         base_hsv_colors[:, 2] = torch.rand(n, generator=rng, device=device) * self.value_scale + self.value_offset  # Value [0.7, 1]

#         # Convert base HSV to source RGB
#         src_rgb_colors = torch.empty_like(base_hsv_colors)
#         for i in range(n):
#             h, s, v = base_hsv_colors[i, 0].item(), base_hsv_colors[i, 1].item(), base_hsv_colors[i, 2].item()
#             r, g, b = colorsys.hsv_to_rgb(h, s, v)
#             src_rgb_colors[i, 0] = r
#             src_rgb_colors[i, 1] = g
#             src_rgb_colors[i, 2] = b

#         # Determine which samples should have matching target colors
#         match_mask = torch.rand(n, generator=rng, device=device) < self.matching_prob

#         # Initialize target RGB colors, initially same as source
#         trg_rgb_colors = src_rgb_colors.clone()

#         # For non-matching samples, generate new target colors
#         non_match_indices = ~match_mask
#         if non_match_indices.any():
#             non_match_n = non_match_indices.sum()
#             non_match_hsv_colors = torch.empty(non_match_n, 3, device=device)
#             non_match_hsv_colors[:, 0] = torch.rand(non_match_n, generator=rng, device=device)  # Hue
#             non_match_hsv_colors[:, 1] = torch.rand(non_match_n, generator=rng, device=device) * 0.5 + 0.5  # Saturation [0.5, 1]
#             non_match_hsv_colors[:, 2] = torch.rand(non_match_n, generator=rng, device=device) * 0.3 + 0.7  # Value [0.7, 1]

#             for i, idx in enumerate(torch.where(non_match_indices)[0]):
#                 h, s, v = non_match_hsv_colors[i, 0].item(), non_match_hsv_colors[i, 1].item(), non_match_hsv_colors[i, 2].item()
#                 r, g, b = colorsys.hsv_to_rgb(h, s, v)
#                 trg_rgb_colors[idx, 0] = r
#                 trg_rgb_colors[idx, 1] = g
#                 trg_rgb_colors[idx, 2] = b

#         return src_rgb_colors, trg_rgb_colors

import torch
import colorsys

class BGColorSampler(object):
    def __init__(self, matching_prob: float = 0.5,
                 hue_offset: float = 0.0, hue_scale: float = 1.0,
                 saturation_offset: float = 0.0, saturation_scale:
                 float = 1.0, value_offset: float = 0.0, value_scale: float = 1.0):
        self.matching_prob = matching_prob
        self.hue_offset = hue_offset
        self.hue_scale = hue_scale
        self.saturation_offset = saturation_offset
        self.saturation_scale = saturation_scale
        self.value_offset = value_offset
        self.value_scale = value_scale

    @staticmethod
    def hsv_to_rgb_batch(hsv: torch.Tensor) -> torch.Tensor:
        """Convert a batch of HSV colors to RGB.

        Args:
            hsv: A tensor of shape (n, 3) where the last dimension represents HSV.

        Returns:
            A tensor of shape (n, 3) representing RGB.
        """
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        i = torch.floor(h * 6)
        f = (h * 6) - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i.long() % 6

        rgb = torch.zeros_like(hsv)
        rgb[i == 0] = torch.stack((v[i == 0], t[i == 0], p[i == 0]), dim=1)
        rgb[i == 1] = torch.stack((q[i == 1], v[i == 1], p[i == 1]), dim=1)
        rgb[i == 2] = torch.stack((p[i == 2], v[i == 2], t[i == 2]), dim=1)
        rgb[i == 3] = torch.stack((p[i == 3], q[i == 3], v[i == 3]), dim=1)
        rgb[i == 4] = torch.stack((t[i == 4], p[i == 4], v[i == 4]), dim=1)
        rgb[i == 5] = torch.stack((v[i == 5], p[i == 5], q[i == 5]), dim=1)

        return rgb

    def sample(self, n: int, rng: torch.Generator):
        """Sample solid RGB colors for source and target with a probability of matching.

        Args:
            n: Number of samples (batch size).
            rng: PyTorch random number generator.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing two tensors of shape (batch_size, 3)
                                 representing source and target RGB colors.
        """
        device = rng.device

        # Sample base HSV colors for source
        src_hsv_colors = torch.empty(n, 3, device=device)
        src_hsv_colors[:, 0] = torch.rand(n, generator=rng, device=device) * self.hue_scale + self.hue_offset  # Hue
        src_hsv_colors[:, 1] = torch.rand(n, generator=rng, device=device) * self.saturation_scale + self.saturation_offset  # Saturation
        src_hsv_colors[:, 2] = torch.rand(n, generator=rng, device=device) * self.value_scale + self.value_offset  # Value

        # Convert base HSV to source RGB
        src_rgb_colors = self.hsv_to_rgb_batch(src_hsv_colors)

        # Determine which samples should have matching target colors
        match_mask = torch.rand(n, generator=rng, device=device) < self.matching_prob

        # Initialize target RGB colors
        trg_rgb_colors = torch.empty(n, 3, device=device)

        # For matching samples, target is the same as source
        trg_rgb_colors[match_mask] = src_rgb_colors[match_mask]

        # For non-matching samples, generate new target colors
        non_match_indices = ~match_mask
        if non_match_indices.any():
            non_match_n = non_match_indices.sum()
            non_match_hsv_colors = torch.empty(non_match_n, 3, device=device)
            non_match_hsv_colors[:, 0] = torch.rand(non_match_n, generator=rng, device=device)  # Hue
            non_match_hsv_colors[:, 1] = torch.rand(non_match_n, generator=rng, device=device) * 0.5 + 0.5  # Saturation [0.5, 1]
            non_match_hsv_colors[:, 2] = torch.rand(non_match_n, generator=rng, device=device) * 0.3 + 0.7  # Value [0.7, 1]
            trg_rgb_colors[non_match_indices] = self.hsv_to_rgb_batch(non_match_hsv_colors)

        return src_rgb_colors, trg_rgb_colors

if __name__ == '__main__':
    # Example usage and benchmarking
    batch_size = 1024
    rng = torch.Generator().manual_seed(42)
    sampler_original = BGColorSampler(matching_prob=0.7)
    sampler_optimized = BGColorSamplerOptimized(matching_prob=0.7)

    import time

    start_time = time.time()
    src_orig, trg_orig = sampler_original.sample(batch_size, rng)
    end_time = time.time()
    print(f"Original implementation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    src_opt, trg_opt = sampler_optimized.sample(batch_size, rng)
    end_time = time.time()
    print(f"Optimized implementation time: {end_time - start_time:.4f} seconds")

    # Verify that the outputs are the same (within floating point precision)
    assert torch.allclose(src_orig, src_opt), "Source colors do not match"
    assert torch.allclose(trg_orig, trg_opt), "Target colors do not match"
    print("Outputs match!")
