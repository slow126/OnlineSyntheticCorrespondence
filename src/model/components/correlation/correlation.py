import torch
from torch.nn.functional import interpolate, normalize, unfold


__all__= [
    'correlate',
    'local_window_correlate',
    'upsample_correlation',
    'local_correlation_from_matches',
]


class LocalWindowCorrelate(torch.autograd.Function):
    def forward(ctx, x1: torch.Tensor, x2: torch.Tensor, k: int):
        ctx.save_for_backward(x1, x2)
        ctx.k = k

        b, c, h, w = x1.shape
        windows = unfold(x1, k, padding=k // 2)
        windows = windows.reshape(b, c, -1, h, w)
        # in the einsum: n = k^2, m = 1
        corr = torch.einsum('bcnhw,bcmhw->bnmhw', windows, x2.unsqueeze(2))
        return corr.view(b, k, k, h, w)

    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        k = ctx.k
        kk = k // 2
        b, c, h, w = x1.shape

        dc = grad_output.unsqueeze(1)
        dx2 = unfold(x1, k, padding=kk).reshape(b, c, k, k, h, w).mul(dc).sum((2, 3))
        
        dc = grad_output.permute(0, 3, 1, 4, 2).reshape(b, 1, h * k, w * k)[..., kk:-kk, kk:-kk]
        dc = unfold(dc, k, dilation=k - 1, padding=kk * (k - 1), stride=k).reshape(b, 1, k, k, h, w)
        dx1 = unfold(x2, k, padding=kk).reshape(b, c, k, k, h, w).mul(dc).sum((2, 3))

        return dx1, dx2, None


def correlate(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    '''Compute global correlation volume between two feature tensors.

    The correlation volume contains the correlation between every pair of features from x1 and x2.

    Args:
        x1 (Tensor): feature tensor of shape (B, C, H1, W1).
        x2 (Tensor): feature tensor of shape (B, C, H2, W2).

    Returns:
        (Tensor): global correlation volume of shape (B, H1, W1, H2, W2).
    '''
    h1, w1 = x1.shape[-2:]
    h2, w2 = x2.shape[-2:]
    x1 = normalize(x1, dim=1)
    x2 = normalize(x2, dim=1)
    corr = x1.flatten(2).transpose(-1, -2) @ x2.flatten(2)
    return corr.view(-1, h1, w1, h2, w2)


def local_window_correlate(x1: torch.Tensor, x2: torch.Tensor, k: int) -> torch.Tensor:
    '''Compute correlations between two feature tensors within a local spatial window.

    For each feature at location (i, j) in x2, we compute the correlation with k^2 features from
    x1 in a (k, k) window centered at (i, j). For x1 and x2 with shape (B, C, H, W), the result
    is a correlation volume V with shape (B, k, k, H, W). V[b, y, x, i, j] is the correlation
    between x2[b, :, i, j] and x1[b, :, i + y - k//2, j + x - k//2]; out-of-bounds locations are
    treated as zero vectors.

    NOTE: this implementation gives the correct results, but it might use a prohibitively large
    amount of memory.

    Args:
        x1 (Tensor): feature tensor with shape (B, C, H, W).
        x2 (Tensor): feature tensor with shape (B, C, H, W).
        k (int): window size.

    Returns:
        (Tensor): local correlation volume with shape (B, k, k, H, W).
    '''
    b, c, h, w = x1.shape
    x1 = normalize(x1, dim=1)
    x2 = normalize(x2, dim=1)
    corr = LocalWindowCorrelate.apply(x1, x2, k)
    return corr


def upsample_correlation(corr: torch.Tensor, size: int, align_corners: bool=False):
    '''Rescale a 4D correlation volume using bilinear interpolation.

    Produces a correlation volume of shape (B, S, S, S, S) from a volume of shape
    (B, H1, W1, H2, W2) using bilinear interpolation.

    Args:
        corr (Tensor): correlation volume of shape (B, Hs, Ws, Ht, Wt).
        size (int): desired output size, used for all spatial dimensions.
        align_corners (bool): alignment method passed to torch.nn.functional.interpolate.

    Returns:
        (Tensor): resampled correlation volume of shape (B, size, size, size, size).
    '''
    s = size
    ht, wt = corr.shape[-2:]

    # interpolate on the source side
    corr = corr.flatten(3).moveaxis(-1, 1) # (B, Ht*Wt, Hs, Ws)
    corr = interpolate(corr, s, mode='bilinear', align_corners=align_corners) # (B, Ht*Wt, s, s)
    
    # interpolate on the target side
    corr = corr.moveaxis(1, -1).view(-1, s * s, ht, wt)
    corr = interpolate(corr, s, mode='bilinear', align_corners=align_corners) # (B, s*s, s, s)

    corr = corr.view(-1, s, s, s, s)

    return corr


def local_correlation_from_matches(coords: torch.Tensor, src_feat: torch.Tensor, trg_feat: torch.Tensor):
    '''
    coords: (B, 2, Hc, Wc) long tensor, xy locations in src for each location in trg (at coarse scale (Hc, Wc))
    src_feat: (B, C, Hf, Wf)
    trg_feat: (B, C, Hf, Wf)

    output: (B, Hf, Wf, Hf / Hc, Wf / Wc)
    '''
    d = coords.device
    hc, wc = coords.shape[-2:]
    b, c, hf, wf = src_feat.shape

    hh, ww = hf // hc, wf // wc

    src_feat = normalize(src_feat, dim=1)
    trg_feat = normalize(trg_feat, dim=1)

    k = torch.arange(max(hh, ww), device=d)
    neighbors = k[None, :ww].add(k[:hh, None] * wf).view(1, 1, -1) # (1, 1, hh * ww)

    _scale = torch.tensor([ww, hh * wf], device=d).view(1, 2, 1, 1)
    idx = coords.mul(_scale).sum(1).view(b, hc * wc, 1) # (b, hc * wc)
    idx = idx.add(neighbors) # (b, hc * wc, hh * ww)
    idx = idx.view(b, 1, -1).expand(-1, c, -1) # (b, c, hc * wc * hh * ww)

    src_patches = src_feat.view(b, c, -1).gather(2, idx) # (b, c, hc * wc * hh * ww)
    src_patches = src_patches.view(b, c, hc * wc, hh * ww)
    trg_patches = trg_feat.view(b, c, hc, hh, wc, ww).permute(0, 1, 2, 4, 3, 5).reshape(b, c, -1, hh * ww)

    local_corr = src_patches.permute(0, 2, 3, 1) @ trg_patches.permute(0, 2, 1, 3)
    local_corr = local_corr.reshape(b, hc, wc, hh, ww, hh, ww)
    return local_corr