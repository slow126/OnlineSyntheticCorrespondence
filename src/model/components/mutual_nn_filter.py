import torch


__all__ = [
    'MutualNNFilter',
    'mutual_nn',
]


class MutualNNFilter(torch.nn.Module):
    '''Mutual nearest neighbor filtering (Rocco et al. NeurIPS'18)
    '''
    def __init__(self, eps=1e-7):
        super().__init__()
        self.register_buffer('eps', torch.tensor([eps], dtype=float))

    def forward(self, corr: torch.Tensor):
        return mutual_nn(corr, self.eps)


def mutual_nn(corr: torch.Tensor, eps=None):
    in_shape = corr.shape
    # (b, h1, w1, h2, w2) -> (b, h1*w1, h2*w2)
    corr = corr.flatten(3, 4).flatten(1, 2)

    if eps is None:
        eps = torch.full((1, ), 1e-7, dtype=corr.dtype, device=corr.device)
    corr_src_max = corr.max(dim=2, keepdim=True)[0]
    corr_src_max = torch.where(corr_src_max == 0, eps, corr_src_max)

    corr_trg_max = corr.max(dim=1, keepdim=True)[0]
    corr_trg_max = torch.where(corr_trg_max == 0, eps, corr_trg_max)

    corr_src = corr / corr_src_max
    corr_trg = corr / corr_trg_max

    corr = corr * (corr_src * corr_trg)
    corr = corr.view(in_shape)
    return corr