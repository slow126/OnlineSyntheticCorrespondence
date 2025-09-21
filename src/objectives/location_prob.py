from typing import Tuple

import torch
from torch.nn.functional import normalize, pairwise_distance


class LocationProbabilityMatchingLoss(torch.nn.Module):
    '''
    '''
    def __init__(self, smoothing_kernel_size=0):
        super().__init__()
        self._init_kernel(smoothing_kernel_size)

    def _init_kernel(self, size):
        self.smoothing_kernel_size = size
        if size > 0:
            if size % 2 == 0:
                raise RuntimeError(f'Kernel size must be an odd number, got {size}')
            n = size // 2
            sigma = n / 3.0
            k = torch.exp(torch.arange(-n, n + 1, dtype=torch.float32).pow_(2).div_(-2 * sigma**2))
            k = k.div_(k.sum())
            k = k[:, None] * k
            self.register_buffer('kernel', k.reshape(1, 1, size, size))

    def create_probability_map(self, pts: torch.Tensor, size: Tuple[int, int]):
        '''
        Args:
            pts (Tensor): yx keypoints of shape (B, N, 2).
        '''
        # get the four nearest pixels to a subpixel location
        nidx = [[[0, 1]], [[0, 0], [0, 1], [1, 0], [1, 1]]]
        neighbors = pts.unsqueeze(-1).repeat(1, 1, 1, 2)
        neighbors[..., 0].floor_()
        neighbors[..., 1].ceil_()
        neighbors = neighbors[:, :, nidx[0], nidx[1]]

        # calculate the weight or "probability" for each of the pixels based on distance from subpixel location
        t = normalize(pairwise_distance(neighbors, pts.unsqueeze(-2), p=2), p=1, dim=-1)
        # TODO: another way to do this would be to calculate the bilinear interpolants
        # tt = pts.frac()
        # tt = torch.stack((1 - tt, tt), -1)
        # # outer product giving the bilinear interpolation weights
        # tt = torch.einsum('bny,bnx->bnyx', tt[:, :, 0], tt[:, :, 1]).flatten(-2)

        # create probability map
        b, n = pts.shape[:2]
        nbors = neighbors.long()
        seq = torch.arange(max(b, n), device=pts.device)
        idx = (seq[:b].view(-1, 1, 1), seq[:n].view(1, -1, 1), nbors[..., 0], nbors[..., 1])
        prob_map = torch.zeros(b, n, *size, device=pts.device).index_put_(idx, t, accumulate=True)
        if self.smoothing_kernel_size > 0:
            prob_map = torch.nn.functional.conv2d(
                prob_map.view(b * n, 1, *size), self.kernel, padding=self.smoothing_kernel_size // 2,
            ).view(b, n, *size)
        prob_map = prob_map.flatten(2)  # (B, N, HW)
        prob_map = normalize(prob_map, p=2, dim=-1)

        return prob_map

    def get_correlation_rows(self, corr_flat: torch.Tensor, pts: torch.Tensor, size: Tuple[int, int], dim: int=1):
        '''
        Args:
            corr_flat (Tensor): flattened correlations of shape (B, HW, HW).
            pts (Tensor): keypoints of shape (B, N, 2).
            dim (int): which dimension to select from, should be either 1 or 2, corresponding to the direction
                of correlation: img1 -> img2 or img2 -> img1
        '''
        idx = pts.long()  # rounds down (floor) and converts to integer
        # flatten index
        idx[:, :, 0] *= size[1]
        idx = idx.sum(-1, keepdim=True).expand(-1, -1, corr_flat.shape[-1])

        mat = corr_flat.gather(dim, idx)

        if dim == 1:
            corr_flat = corr_flat.transpose(1, 2)
        corr_flat = corr_flat.view(*corr_flat.shape[:2], *size)

        return mat
    
    def get_interpolated_correlation_rows(self, corr: torch.Tensor, pts: torch.Tensor):
        '''Gets interpolated correlation rows based on subpixel keypoint locations.
        '''
        corr = corr.flatten(1, 2)
        # normalize point locations to range [-1, 1] and flip yx to xy for grid_sample
        hw = torch.tensor(tuple(corr.shape[-2:]), device=corr.device)
        p = pts.div(0.5 * (hw - 1)).sub(1).flip(-1).unsqueeze(2)  # xy points (B, N, 1, 2)
        mat = torch.nn.functional.grid_sample(corr, p, align_corners=True).squeeze(-1).transpose(-1, -2)
        # should be equivalent to this
        # w = pts.frac()
        # w = torch.stack((1 - w, w), -1)
        # w = torch.einsum('bny,bnx->bnyx', w[:, :, 0], w[:, :, 1]).flatten(-2).unsqueeze(-1)
        # bidx = torch.arange(corr.shape[0], device=corr.device).view(-1, 1, 1)
        # yx = torch.stack((pts.floor(), pts.ceil()), -1)[..., [[0,1]], [[0,0],[0,1],[1,0],[1,1]]].long()
        # mat2 = corr[bidx, :, yx[..., 0], yx[..., 1]].mul(w).sum(-2).transpose(1, 2)
        # torch.allclose(mat, mat2, atol=1e-7, rtol=1e-3)
        return mat

    def forward(self, corr: torch.Tensor, kp: torch.Tensor):
        '''
        Args:
            corr (Tensor): correlation volume of shape (B, H, W, H, W).
            kp (Tensor): keypoints of shape (B, N, 2, 2) where the last two dimensions are (y, x) and (img1, img2).
        '''
        pts1 = kp[..., 0]
        pts2 = kp[..., 1]

        # breakpoint()
        size = corr.shape[-2:]
        # corr_flat = corr.flatten(3, 4).flatten(1, 2)  # (B, HW, HW)

        target12 = self.create_probability_map(pts2, size)  # (B, N, HW)
        # pred12 = self.get_correlation_rows(corr_flat, pts1, size, dim=1)  # (B, N, HW)
        pred12 = self.get_interpolated_correlation_rows(corr.permute(0, 3, 4, 1, 2), pts1)
        loss1 = pred12.sub(target12).norm(p='fro', dim=(-1, -2)).mean()

        target21 = self.create_probability_map(pts1, size)  # (B, N, HW)
        # pred21 = self.get_correlation_rows(corr_flat, pts2, size, dim=2)  # (B, N, HW)
        pred21 = self.get_interpolated_correlation_rows(corr, pts2)
        loss2 = pred21.sub(target21).norm(p='fro', dim=(-1, -2)).mean()

        loss = loss1 + loss2

        return loss