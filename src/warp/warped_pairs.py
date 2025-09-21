from typing import Optional

import kornia
import torch

from . import warp


class SyntheticPairWarper(torch.nn.Module):
    '''Module for creating paired images by synthetic warps.

    Args:
        affine (dict): keyword arguments for warp.sample_affine (optional). If None, no affine
            warps are sampled.
        homography (dict): keyword arguments for warp.sample_homography (optional). If None, no
            homography warps are sampled.
        tps (dict): keyword arguments for warp.sample_tps (optional). If None, no thin plate
            spline warps are sampled.
        elastic (dict): keyword arguments for warp.multi_local_elastic_transform (optional). If
            None, no elastic transform warps are sampled.
        transform_src (float): probability of applying an affine transform to the source images.

    Returns:
        Tensor of source images.
        Tensor of target images.
        Tensor of correspondences, containing the source image pixel location for each target
            pixel, or 'inf' for pixels without a match.
    '''
    def __init__(
        self,
        affine: Optional[dict] = None,
        homography: Optional[dict] = None,
        tps: Optional[dict] = None,
        elastic: Optional[dict] = None,
        transform_src: float = 0.0,
    ):
        '''
        Affine:
            translation: float = 0.25
            scale: Union[float, Tuple[float, float]]= 0.45
            angle: float = 0.083
            shear: float = 0.2
        Homography:
            sigma: float = 0.3
        TPS:
            grid_size: int = 3
            shrink: float = 0.0
            sigma: float = 0.08
        Elastic:
            num_range: Tuple[int, int] = (2, 6)
            sigma_range: Tuple[float, float] = (5, 8)
            mask_range: Tuple[float, float] = (0.03, 0.09)
            alpha_range: Tuple[float, float] = (1, 1)
        '''
        super().__init__()
        if affine is None:
            self.affine = None
        else:
            self.affine = dict(translation=0.25, scale=0.45, angle=0.083, shear=0.2)
            self.affine.update(affine)
        
        if homography is None:
            self.homography = None
        else:
            self.homography = dict(sigma=0.3)
            self.homography.update(homography)

        if tps is None:
            self.tps = None
        else:
            self.tps = dict(sigma=0.08)
            self.tps.update(tps)

        if elastic is None:
            self.elastic = None
        else:
            self.elastic = dict(alpha_range=(1, 4))

        self.transform_src = transform_src

    def forward(self, imgs: torch.Tensor):
        not_batched = imgs.ndim == 3
        if not_batched:
            imgs = imgs[None]

        b, _, h, w = imgs.shape
        d = {'device': imgs.device}

        aff = hom = tps = ela = None

        if self.affine is not None:
            aff = warp.sample_affine(b, **self.affine, **d)
        if self.homography is not None:
            hom = warp.sample_homography(b, **self.homography, **d)
        if self.tps is not None:
            tps = warp.sample_tps(b, **self.tps, **d)
        if self.elastic is not None:
            ela = warp.local_elastic_transforms(h, w, b, **self.elastic, **d)

        grid = kornia.create_meshgrid(h, w, **d)

        field = warp.get_displacement_field(h, w, grid, aff, hom, tps, ela, **d)
        mask = field.abs().le(1).all(-1)
        trg = warp.warp_imgs(imgs, field)

        src = imgs

        if self.transform_src > 0 and torch.rand(1).item() < self.transform_src:
            aff_src = warp.sample_affine(b, 0.05, (0.1, 0), 0.05, 0.0, **d)
            field2 = warp.get_displacement_field(h, w, grid, affine=aff_src, **d)
            src = warp.warp_imgs(imgs, field2)

            iaff_src = torch.linalg.inv(aff_src)
            field = warp.warp_affine(field.flatten(1, 2), iaff_src).view(b, h, w, 2)
            mask = mask & field.abs().le(1).all(-1)

        s = field.new_tensor([w - 1, h - 1])
        field = field.add(1).mul(s / 2)
        field[~mask] = float('inf')

        field = field.moveaxis(-1, 1)  # (B, 2, H, W)

        if not_batched:
            src = src[0]
            trg = trg[0]
            field = field[0]

        return src, trg, field