from typing import List, Optional, Tuple
import torch


def pad_keypoints_batch(
    src_list: List[Optional[torch.Tensor]],
    trg_list: List[Optional[torch.Tensor]],
    max_kps: Optional[int]
) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]], torch.Tensor]:
    """Pad/truncate keypoints to a common size across the batch."""
    lengths = []
    for src in src_list:
        if src is None:
            lengths.append(0)
        else:
            lengths.append(src.shape[1])

    target_size = max(lengths) if max_kps is None else min(max(lengths), max_kps)
    padded_src, padded_trg = [], []

    for src, trg in zip(src_list, trg_list):
        if src is None or trg is None:
            padded_src.append(src)
            padded_trg.append(trg)
            continue

        n = src.shape[1]
        if n > target_size:
            padded_src.append(src[:, :target_size])
            padded_trg.append(trg[:, :target_size])
        elif n < target_size:
            pad = torch.full(
                (2, target_size - n),
                -1,
                dtype=src.dtype,
                device=src.device,
            )
            padded_src.append(torch.cat([src, pad], dim=1))
            padded_trg.append(torch.cat([trg, pad], dim=1))
        else:
            padded_src.append(src)
            padded_trg.append(trg)

    device = None
    for tensor in padded_src:
        if tensor is not None:
            device = tensor.device
            break
    if device is None:
        device = torch.device("cpu")

    n_pts = torch.tensor([min(l, target_size) for l in lengths], dtype=torch.int32, device=device)
    return padded_src, padded_trg, n_pts


def weighted_sample_from_flow(
    flow: torch.Tensor,
    max_kps: int,
    border_frac: float = 0.08,
    center_sigma: float = 0.35,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Sample up to max_kps from dense flow with a mild center bias and border masking."""
    # flow: [2, H, W]
    _, H, W = flow.shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, H, device=flow.device),
        torch.linspace(-1, 1, W, device=flow.device),
        indexing="ij",
    )
    center_weight = torch.exp(-((xs ** 2 + ys ** 2) / (2 * center_sigma ** 2)))
    flow_mag = flow.norm(dim=0)  # [H, W]
    valid = flow_mag.isfinite()
    flow_mag = flow_mag.clamp(min=1e-4)
    mask = (ys.abs() < (1 - border_frac)) & (xs.abs() < (1 - border_frac)) & valid
    weights = (center_weight * flow_mag * mask).reshape(-1)
    valid_mask = weights > 0
    if valid_mask.sum() == 0:
        return (
            torch.empty(2, 0, device=flow.device, dtype=flow.dtype),
            torch.empty(2, 0, device=flow.device, dtype=flow.dtype),
            0,
        )

    weights = weights / (weights.sum() + 1e-8)
    k = min(max_kps, valid_mask.sum().item())
    idx = torch.multinomial(weights, k, replacement=False)
    y = (idx // W).long()
    x = (idx % W).long()
    trg = torch.stack([x.float(), y.float()], dim=0)  # [2, k]
    src = trg + flow[:, y, x]
    return src, trg, k
