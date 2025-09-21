from typing import Optional

import torch


__all__ = [
    'KernelSoftArgmax',
]


class KernelSoftArgmax(torch.nn.Module):
    '''SFNet: Learning Object-aware Semantic Flow (Lee et al.)
    '''
    no_decay: tuple = ('beta', 'sigma')

    def __init__(
        self,
        feature_size: Optional[int] = None,
        beta: float = 0.05,
        apply_kernel: bool = False,
        learn_sigma: bool = False,
        learn_beta: bool = False,
        normalized: bool = True,
    ):
        super().__init__()
        self.feature_size = feature_size
        self.normalized = normalized

        self.register_buffer('coords', self.get_coords(feature_size or 8, normalized))

        self.apply_kernel = apply_kernel
        if apply_kernel:
            self.sigma = torch.nn.Parameter(torch.tensor(5.)).requires_grad_(learn_sigma)

        self.beta = torch.nn.Parameter(torch.tensor([beta]).log()).requires_grad_(learn_beta)

    @staticmethod
    def get_coords(n, normalized, device='cpu'):
        if normalized:
            coords = torch.linspace(-1, 1, n, device=device)
        else:
            coords = torch.arange(n, dtype=torch.float32, device=device)
        return coords

    def apply_gaussian_kernel(self, corr: torch.Tensor):
        b, h, w = corr.shape[:3]

        coords = self.get_coords(max(h, w), False, corr.device)

        # best match for each source location
        idx = corr.flatten(-2).argmax(-1)
        idx_y = idx.div(w, rounding_mode='floor').float().view(b, h, w, 1, 1)
        idx_x = idx.remainder(w).float().view(b, h, w, 1, 1)
        
        # kernel weights
        x = coords[:w].view(1, w).sub(idx_x).square()
        y = coords[:h].view(h, 1).sub(idx_y).square()
        gauss_kernel = x.add(y).div(2 * self.sigma**2).neg().exp()

        filtered = corr * gauss_kernel

        return filtered
    
    def forward(self, corr: torch.Tensor, as_tuple: bool=True, dim: int=-1):
        b, h1, w1, h2, w2 = corr.shape
        
        if self.apply_kernel:
            corr = self.apply_gaussian_kernel(corr)

        # softmax with temperature
        corr = corr.flatten(-2).div(self.beta.exp()).softmax(dim=-1).view(b, h1, w1, h2, w2)

        # spatial coordinates
        coords = self.coords
        if coords.shape[0] != w1:
            coords = self.get_coords(w1, self.normalized, corr.device)
            self.coords.data = coords

        # marginalize over y and get x as weighted sum
        grid_x = corr.sum(3).mul(coords).sum(-1)
        
        if coords.shape[0] != h1:
            coords = self.get_coords(h1, self.normalized, corr.device)

        # marginalize over x and get y as weighted sum
        grid_y = corr.sum(4).mul(coords).sum(-1)

        if as_tuple:
            return grid_x, grid_y

        return torch.stack((grid_x, grid_y), dim)