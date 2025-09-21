from typing import List, Tuple

import torch


def corr2flow(paired_points: List[torch.Tensor], size: Tuple[int]=(64, 64), relative: bool=True):
    '''Create a flow field (img2 -> img1) from a set of corresponding points.

    The input `paired_points` is a list of tensors with shape (Ni, 2, 2), with Ni the number of
    point correspondences between the images. The second to last dimension indexes the yx
    coordinates, and the last dimension indexes the two images. So paired_points[:, :, 0, 0]
    would give all the y-coordinates for points in img1, or paired_points[:, :, 1, 1] would give
    all the x-coordinates for points in img2, etc.

    Args:
        paired_points (list of Tensor): list of keypoint tensors of shape (Ni, 2, 2), where Ni
            might be different for each tensor.
        size (list of int): spatial dimensions of the output flow field (H, W).
        relative (bool): whether to output relative coordinates (flow field) or absolute
            coordinates (direct mapping).

    Returns:
        (Tensor): tensor of shape (B, 2, H, W) containing the flow from img2 to img1.
    '''
    d = dict(device=paired_points[0].device)
    B = len(paired_points)
    one = torch.ones(1, **d)
    flow = torch.zeros(B, 2, *size, **d)
    count = torch.zeros_like(flow[0, 0])
    for i, pts in enumerate(paired_points):
        yi = pts[:, 0, 1]  # y coords for points in img2
        xi = pts[:, 1, 1]  # x coords for points in img2
        # note: flip offsets from yx to xy to be consistent with other datasets, thats why .flip(1)
        if relative:
            # flow field: img2 -> img1    [ -(pts2 - pts1) ]
            offsets = -pts.diff().squeeze(-1).flip(1).permute(1, 0).float()
        else:
            # absolute coordinates in img1 for each img2 location
            offsets = pts[:, :, 0].flip(1).permute(1, 0).float()
        # flow[i, :, yi, xi] = offsets
        flow[i, 0].index_put_((yi, xi), offsets[0], accumulate=True)
        flow[i, 1].index_put_((yi, xi), offsets[1], accumulate=True)
        count.index_put_((yi, xi), one, accumulate=True)
        mask = count > 0
        flow[i, :, mask] /= count[mask]
        flow[i, :, ~mask] = float('inf')
        count.zero_()
    return flow


def flow_to_keypoints(
    trg_kps: torch.Tensor,
    flow: torch.Tensor,
    n_pts: torch.Tensor,
    upsample_size: Tuple[int]=(256, 256),
):
    '''TODO
    '''
    _, _, h, w = flow.shape
    d = dict(device=flow.device)
    upsize = torch.tensor(upsample_size, **d)
    if (h, w) != upsample_size:
        hw = torch.tensor([h, w], **d)
        flow = torch.nn.functional.interpolate(flow, upsample_size, mode='bilinear')
        flow.mul_(upsize.div(hw).view(1,2,1,1))
    
    kp = trg_kps.permute(0, 2, 1).contiguous()  # (b, 2, n) -> (b, n ,2)
    # adding the .floor() reproduces the implementation from CATs
    # (https://github.com/SunghwanHong/Cost-Aggregation-transformers).
    kp.floor_()  # would rounding be better?
    kpi = kp.long()
    mask = torch.arange(0, kpi.shape[1], **d).view(1,-1)
    mask = mask.lt(n_pts.view(-1, 1)).unsqueeze_(-1).expand_as(kpi)
    b_idx = mask.nonzero(as_tuple=True)[0][::2]
    kpm = kpi[mask].view(-1, 2)
    kpm.clamp_(torch.zeros_like(upsize).view(1, 2), upsize.view(1, 2) - 1)
    kp[mask] += flow[b_idx, :, kpm[:, 1], kpm[:, 0]].view(-1)
    kp[~mask] = -1

    return kp.moveaxis(2, 1)
    
    ### Original implementation from CATs
    # _, _, h, w = flow.size()
    # flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    # src_kps = []
    # for trg_kps, flow, n_pts in zip(trg_kps.long(), flow, n_pts):
    #     size = trg_kps.size(1)
    #     kp = torch.clamp(trg_kps.narrow_copy(1, 0, n_pts), 0, upsample_size[0] - 1)
    #     estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
    #     estimated_kps = torch.cat((estimated_kps, torch.ones(2, size - n_pts, **d) * -1), dim=1)
    #     src_kps.append(estimated_kps)
    # return torch.stack(src_kps)
    ###


def sample_kps(paired_points: List[torch.Tensor], num_points: int):
    '''
    Args:
        paired_points (list of Tensors): tensors of shape (N_i, 2, 2).
        num_points (int): number of points to sample.
    '''
    b = len(paired_points)
    maxn = max(p.shape[0] for p in paired_points)
    num_points = min(num_points, maxn)
    points = torch.zeros(b, num_points, 2, 2)
    for i, p in enumerate(paired_points):
        n = p.shape[0]
        if n > num_points:
            r = n / num_points
            k = torch.randint(0, int(r), (1,)).item()
            subset = torch.linspace(k, k + (num_points - 1) * r, num_points).round().long()
            p = p[subset]
        
        points[i] = p

    return points


def normalize_coords(coords: torch.Tensor, dim: int=1):
    '''Convert coordinates from absolute range [0, (h-1) | (w-1)] to relative range [-1, 1].

    Args:
        coords (Tensor): coordinate tensor with shape (B, 2, H, W) or (B, H, W, 2).
        dim (int): dimension corresponding to the coordinate values; 1 if shape of coords is
            (B, 2, H, W), 3 if shape of coords is (B, H, W, 2).
    '''
    if dim not in (1, 3):
        raise ValueError(f"Argument `dim` must be either 1 or 3, got {dim}")
    shape = [1, 1, 1, 1]
    shape[dim] = 2
    size = coords.shape[-2:] if dim == 1 else coords.shape[1:3]
    s = coords.new_tensor(size).view(shape).sub(1)
    coords = coords.mul(2 / s).sub(1)
    return coords


def unnormalize_coords(coords: torch.Tensor, dim: int=1):
    '''Convert coordinates from relative range [-1, 1] to absolute range [0, (h-1) | (w-1)].

    Args:
        coords (Tensor): coordinate tensor with shape (B, 2, H, W) or (B, H, W, 2).
        dim (int): dimension corresponding to the coordinate values; 1 if shape of coords is
            (B, 2, H, W), 3 if shape of coords is (B, H, W, 2).
    '''
    if dim not in (1, 3):
        raise ValueError(f"Argument `dim` must be either 1 or 3, got {dim}")
    shape = [1, 1, 1, 1]
    shape[dim] = 2
    size = coords.shape[-2:] if dim == 1 else coords.shape[1:3]
    s = coords.new_tensor(size).view(shape).sub(1)
    coords = coords.add(1).mul(s * 0.5)
    return coords


def convert_mapping_to_flow(mapping: torch.Tensor, is_normalized: bool=True, as_normalized: bool=False):
    '''Convert from direct pixel-to-pixel mapping to flow (relative offset) in pixel coordinates.

    Args:
        mapping (Tensor): mapping from each pixel to a new coordinate, with shape (B, 2, H, W).
        is_normalized (bool): if True, indicates that the mapping is expressed in normalized
            coordinates between [-1, 1].

    Returns:
        (Tensor): flow (pixel offsets), with shape (B, 2, H, W).
    '''
    d = dict(device=mapping.device, dtype=torch.float32)

    if as_normalized:
        if not is_normalized:
            # convert from [0, (h-1) | (w-1)] to [-1, 1]
            mapping = normalize_coords(mapping)

        grid = torch.stack(torch.meshgrid(
            torch.linspace(-1, 1, mapping.shape[-1], **d),
            torch.linspace(-1, 1, mapping.shape[-2], **d),
            indexing='xy'
        ), dim=0)
    else:
        if is_normalized:
            # convert from [-1, 1] to [0, (h-1) | (w-1)]
            mapping = unnormalize_coords(mapping)

        grid = torch.stack(torch.meshgrid(
            torch.arange(mapping.shape[-1], **d),
            torch.arange(mapping.shape[-2], **d),
            indexing='xy'
        ), dim=0)

    flow = mapping - grid
    return flow


def convert_flow_to_mapping(flow: torch.Tensor, normalize: bool=True):
    '''Convert from flow (relative offset) to direct pixel-to-pixel mapping in pixel coordinates.

    Args:
        flow (Tensor): offset from each pixel to a new coordinate, with shape (B, 2, H, W).
        normalize (bool): if True, normalize the mapping coordinates from [0, width/height - 1]
            to [-1, 1].

    Returns:
        (Tensor): mapping (pixel locations), with shape (B, 2, H, W).
    '''
    d = dict(device=flow.device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(
        torch.arange(flow.shape[-1], **d),
        torch.arange(flow.shape[-2], **d),
        indexing='xy',
    ), dim=0)

    mapping = grid + flow

    if normalize:
        mapping = normalize_coords(mapping)
    
    return mapping


def sparse_downsample(x: torch.Tensor, size: Tuple[int]):
    '''Downsample input tensor while minimizing impact of "invalid" values. Used for downsampling
    ground-truth flow with invalid flow locations encoded as inf.

    Generic interpolation strategies like bilinear or nearest will either blend valid and invalid
    values or lose isolated points. The strategy used here is based on max and min pooling to
    minimize sparsity.

    Adapted from implementation of `sparse_max_pool` from GLU-Net
    (https://github.com/PruneTruong/GLU-Net).

    Args:
        x (Tensor): input Tensor to be downsampled, with shape (B, C, H, W).
        size (list of int): desired output size [H, W].

    Returns:
        (Tensor): downsampled input, with shape (B, C, size[0], size[1])
    '''
    pool = torch.nn.functional.adaptive_max_pool2d
    # convert inf values ("invalid") to 0
    mask = x.detach().ne(float('inf')).all(dim=1, keepdim=True)
    x = x.clone()
    x[~mask.expand_as(x)] = 0.0

    pooled = pool(x * x.gt(0), size) - pool(-x * x.lt(0), size)
    
    # also pool the mask and then set "invalid" output regions back to inf
    mask = pool(mask.float(), size)
    pooled[(~mask.bool()).expand_as(pooled)] = float('inf')

    return pooled


def flow_by_coordinate_matching(
    coord1: torch.Tensor,
    coord2: torch.Tensor,
    index=None,
    train_index: bool=True,
    threshold: float=5e-5
):
    '''Compute ground-truth flow by finding geometrically matching points between two sets of
    surface coordinates.

    Args:
        coord1 (Tensor): first coordinate Tensor with shape (B, H, W, 3). Locations where the
            coordinate is all zeros are ignored.
        coord2 (Tensor): second coordinate Tensor with shape (B, H, W, 3). Locations where the
            coordinate is all zeros are ignored.
        index (faiss index): a FAISS index for similarity search, or None. If None, the index
            will be constructed internally. Note that index construction adds significant
            overhead, so it will be more efficient to maintain the index externally and pass it
            in if calling this function repeatedly.
        train_index (bool): if True, the index will be trained using valid coordinates from coord1.
            Otherwise, the index is untrained. Generally, indices need to be trained unless they
            are of the brute-force variety.
        threshold (float): threshold on distance that determines whether two points are a match.
    
    Returns:
        (Tensor): the computed flow from coord2 to coord1, measured in pixels.
    '''
    if index is None:
        # NOTE: constructing the index adds significant overhead, it's better to construct it
        # outside and pass it in if calling this function frequently
        import faiss
        import faiss.contrib.torch_utils

        gres = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gres, 0, faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 32))
        index.nprobe = 2
    
    mask1 = coord1.ne(0).any(-1)
    mask2 = coord2.ne(0).any(-1)

    if train_index:
        index.train(coord1[mask1])
    
    b, h, w = coord2.shape[:-1]
    flow = coord2.new_full((b, 2, h, w), torch.inf)
    
    for i in range(coord1.shape[0]):
        m1, m2 = mask1[i], mask2[i]
        c1, c2 = coord1[i][m1], coord2[i][m2] 
        index.reset()
        index.add(c1)

        p1, p2 = _match_points(c2, m1, m2, index, threshold)
        offsets = p1.sub(p2).float()

        # putting coordinates in as (x, y), where offsets has (y, x) ordering
        flow[i, 0].index_put_((p2[:, 0], p2[:, 1]), offsets[:, 1])
        flow[i, 1].index_put_((p2[:, 0], p2[:, 1]), offsets[:, 0])

    return flow


def _match_points(c2: torch.Tensor, m1: torch.Tensor, m2: torch.Tensor, index, threshold: float):
    '''Find matching points by nearest neighbor search.'''
    dist, idx1 = index.search(c2, 1)
    
    mask = dist.ravel() <= threshold
    idx1 = idx1.ravel()[mask]

    p1 = m1.nonzero()[idx1]
    p2 = m2.nonzero()[mask]

    return p1, p2