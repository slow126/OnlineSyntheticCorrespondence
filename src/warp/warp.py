from typing import Optional, Tuple, Union

import kornia
import torch


def get_displacement_field(
    height: int,
    width: int,
    grid: Optional[torch.Tensor] = None,
    affine: Optional[torch.Tensor] = None,
    homography: Optional[torch.Tensor] = None,
    tps: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    elastic: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu',
):
    '''Create a single displacement field by composing one or more different types of warps.
    '''
    if grid is None:
        grid = kornia.create_meshgrid(height, width, device=device).flatten(1, 2)
    elif grid.ndim == 4:
        grid = grid.flatten(1, 2)

    if affine is not None and homography is not None:
        mat = homography @ affine
        grid = warp_homography(grid, mat)
    elif affine is not None:
        grid = warp_affine(grid, affine)
    elif homography is not None:
        grid = warp_homography(grid, homography)

    if tps is not None:
        grid = warp_tps(grid, *tps)

    grid = grid.view(-1, height, width, 2)

    if elastic is not None:
        grid += elastic

    return grid


def warp_imgs(imgs: torch.Tensor, grid: torch.Tensor, align_corners: bool=False):
    return torch.nn.functional.grid_sample(imgs, grid, align_corners=align_corners)


def warp_homogenous(points: torch.Tensor, matrix: torch.Tensor):
    points = torch.cat((points, points.new_ones(*points.shape[:-1], 1)), -1)
    warped = matrix.matmul(points.transpose(1, 2)).transpose(1, 2)
    return warped[..., :2] / warped[..., 2:]


###################
# Affine Transform
###################

def sample_affine(
    batch_size: int = 1,
    translation: float = 0.25,
    scale: Union[float, Tuple[float, float]] = 0.45,
    angle: float = 15.0,
    shear: float = 0.2,
    device: Union[str, torch.device] = 'cpu',
):
    if isinstance(scale, float):
        scale = (scale, scale)

    c = torch.zeros(batch_size, 2, device=device)
    t = torch.empty_like(c).uniform_(-translation, translation)
    s = torch.empty_like(c).uniform_(1 - scale[0], 1 + scale[1])

    a = torch.empty(batch_size, device=device).uniform_(-angle, angle)
    x = torch.empty_like(a).uniform_(-shear, shear)
    y = torch.empty_like(a).uniform_(-shear, shear)

    A = kornia.geometry.get_affine_matrix2d(t, c, s, a, x, y)

    return A


def warp_affine(
    points: torch.Tensor,
    affmat: torch.Tensor,
):
    '''
    Args:
        points (Tensor): xy source points to be warped, with shape (B, N, 2).
        affmat (Tensor): affine matrix with shape (B, 3, 3).
    
    Returns:
        Tensor containing the warped source points, with shape (B, N, 2).
    '''
    points = points.transpose(1, 2)
    warped = affmat[:, :2, :2].matmul(points).add(affmat[:, :2, 2:3]).transpose(1, 2)
    return warped


#############
# Homography
#############

def sample_homography(
    batch_size: int = 1,
    sigma: float = 0.3,
    device: Union[str, torch.device] = 'cpu',
):
    grid = torch.tensor([[
        [-1, -1],
        [-1,  1],
        [ 1, -1],
        [ 1,  1],
    ]], device=device, dtype=torch.float32)

    warp_grid = grid + grid.new_empty(batch_size, 4, 2).uniform_(-2 * sigma, 2 * sigma)

    A = grid.new_zeros(batch_size, 8, 8)
    A[:, 0::2, :2] = grid
    A[:, 0::2, 2] = 1
    A[:, 1::2, 5] = 1
    A[:, 1::2, 3:5] = warp_grid
    A[:, 0::2, -2:] = -grid[:, :, 0:1] * warp_grid
    A[:, 1::2, -2:] = -grid[:, :, 1:] * warp_grid

    H = torch.linalg.lstsq(A, warp_grid.flatten(1)[..., None]).solution.squeeze(-1)
    H = torch.cat((H, H.new_ones(batch_size, 1)), -1).reshape(-1, 3, 3)

    return H


def warp_homography(
    points: torch.Tensor,
    hmat: torch.Tensor,
):
    return warp_homogenous(points, hmat)


####################
# Thin Plate Spline
####################

def sample_tps(
    batch_size: int = 1,
    grid_size: int = 3,
    shrink: float = 0.0,
    sigma: float = 0.08,
    linear_matrix: Optional[torch.Tensor] = None,
    src: Optional[torch.Tensor] = None,
    trg: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu',
):
    shrink = shrink / grid_size

    if src is None:
        x = torch.linspace(-1 + shrink, 1 - shrink, grid_size, device=device)
        src = torch.stack(torch.meshgrid(x, x, indexing='xy'), -1).flatten(0, 1)[None].expand(batch_size, -1, -1)
    if trg is None:
        trg = src + torch.empty_like(src).uniform_(-2 * sigma, 2 * sigma)
    if linear_matrix is not None:
        trg = warp_homogenous(trg, linear_matrix)

    kernel, affine = kornia.geometry.get_tps_transform(src, trg)

    return trg, kernel, affine


warp_tps = kornia.geometry.warp_points_tps


####################
# Elastic Transform
####################

def sample_elastic_transforms(
    height: int,
    width: int,
    num_transforms: int = 1,
    sigma: Union[int, float, torch.Tensor] = 5.0,
    device: Union[str, torch.device] = 'cpu',
):
    '''Sample a batch of elastic deformation fields.
    '''
    if isinstance(sigma, (float, int)):
        sigma = torch.tensor([float(sigma)], device=device)
        sigma = sigma.view(1, -1).expand(num_transforms, 2)

    device = sigma.device

    k = sigma.mul(6).round().long()
    k.add_(k.remainder(2).eq(0).long())
    k = k.max(0)[0]

    noise = torch.empty(num_transforms, 2, height, width, device=device).uniform_(-1, 1)
    kernel = kornia.filters.get_gaussian_kernel2d(k, sigma, device=device)

    return kornia.filters.filter2d(noise, kernel, border_type='constant').moveaxis(1, -1)


def sample_location_masks(
    height: int,
    width: int,
    num_masks: int = 1,
    sigma_range: Tuple[float, float] = (8, 24),
    device: Union[str, torch.device] = 'cpu',
):
    '''Sample a batch of location masks.

    Each mask is defined by a Guassian distribution centered at a random location.
    '''
    sigma = torch.empty(num_masks, 1, 1, 2, device=device).uniform_(*sigma_range)

    xy = torch.rand(num_masks, 1, 1, 2, device=device)
    xy[..., 0] *= width
    xy[..., 1] *= height

    masks = kornia.create_meshgrid(height, width, normalized_coordinates=False, device=device)
    masks = torch.exp(-masks.sub(xy).pow(2).div(2 * sigma.pow(2)).sum(-1))
    masks = masks.mul(2).clamp_max(1)

    return masks


def local_elastic_transforms(
    height: int,
    width: int,
    batch_size: int = 1,
    num_range: Tuple[int, int] = (2, 6),
    sigma_range: Tuple[float, float] = (5, 8),
    mask_range: Tuple[float, float] = (0.03, 0.09),
    alpha_range: Tuple[float, float] = (1, 1),
    device: Union[str, torch.device] = 'cpu',
):
    '''Sample a deformation field defined by multiple local elastic deformations.
    '''
    mask_range = tuple(a * x for a, x in zip(mask_range, (height, width)))
    n = torch.randint(*num_range, (1,)).item()
    sigma = torch.empty(batch_size, 2, device=device).uniform_(*sigma_range)
    alpha = torch.empty(batch_size, n, 1, 1, 2, device=device).uniform_(*alpha_range)

    # scale the elastic deformation so that the maximum displacement is approximately 1 pixel,
    # so that alpha can be interpreted as a pixel displacement magnitude
    elastic = sample_elastic_transforms(height, width, batch_size, sigma)
    elastic = elastic.div(elastic.amax(dim=(1, 2, 3), keepdim=True) * max(height, width))

    mask = sample_location_masks(height, width, batch_size * n, mask_range, device)
    mask = mask.reshape(batch_size, n, height, width, 1).mul(alpha).sum(1)

    field = elastic.mul(mask)

    return field