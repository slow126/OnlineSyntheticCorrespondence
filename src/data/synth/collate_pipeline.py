from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from src.data.synth.common.common_sample import CommonSample
from src.data.synth.common.sampling import pad_keypoints_batch, weighted_sample_from_flow
from src.data.synth.datasets.flow_utils import (
    flow_from_kps,
    kps_from_flow,
    downsample_flow,
    prepare_invalids,
)


def _flow_make_finite(flow: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Replace non-finite entries with zero to keep model inputs clean."""
    if flow is None:
        return None
    if not torch.is_tensor(flow):
        return flow
    mask = flow.isfinite()
    if mask.all():
        return flow
    flow = flow.clone()
    flow[~mask] = 0.0
    return flow


def _resize_tensor(img: torch.Tensor, size: Tuple[int, int]) -> Tuple[torch.Tensor, float, float]:
    """Resize CHW tensor and return scales."""
    target_h, target_w = size
    _, h, w = img.shape
    if h == target_h and w == target_w:
        return img, 1.0, 1.0
    out = F.interpolate(img.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    return out, target_w / w, target_h / h


def resize_sample(sample: CommonSample, size: Optional[Tuple[int, int]]) -> CommonSample:
    if size is None or sample.src_img is None:
        return sample
    target_h, target_w = size
    sample.src_img, scale_w, scale_h = _resize_tensor(sample.src_img, size)
    if sample.trg_img is not None:
        sample.trg_img, _, _ = _resize_tensor(sample.trg_img, size)

    if sample.flow_full is not None:
        flow = sample.flow_full.unsqueeze(0) if sample.flow_full.dim() == 3 else sample.flow_full
        flow = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
        flow[..., 0, :, :] *= scale_w
        flow[..., 1, :, :] *= scale_h
        sample.flow_full = flow.squeeze(0)

    if sample.src_kps is not None and sample.trg_kps is not None:
        sample.src_kps = sample.src_kps.clone()
        sample.trg_kps = sample.trg_kps.clone()
        sample.src_kps[0] *= scale_w
        sample.src_kps[1] *= scale_h
        sample.trg_kps[0] *= scale_w
        sample.trg_kps[1] *= scale_h
    return sample


def ensure_flow_and_kps(
    sample: CommonSample,
    dataset_name: str,
    max_kps: Optional[int],
    downsample_feat_size: Optional[int],
    prefer_all_dense: bool,
) -> CommonSample:
    # If we only have feature flow and native keypoints, build full flow from kps
    if sample.flow_full is None and sample.src_kps is not None and sample.trg_kps is not None:
        if sample.src_img is not None:
            _, H, W = sample.src_img.shape
        else:
            H = W = 512
        sample.flow_full = flow_from_kps(sample.src_kps, sample.trg_kps, (H, W))

    # If keypoints missing but flow_full exists, sample them
    if sample.src_kps is None and sample.flow_full is not None:
        if max_kps is None and prefer_all_dense:
            trg_kps, src_kps, n_valid = kps_from_flow(sample.flow_full, num_kps=None, use_fast_sampling=False)
        elif max_kps is None:
            trg_kps, src_kps, n_valid = kps_from_flow(sample.flow_full, num_kps=None)
        else:
            src_kps, trg_kps, n_valid = weighted_sample_from_flow(sample.flow_full, max_kps)
        sample.src_kps = src_kps
        sample.trg_kps = trg_kps
        sample.n_pts = int(n_valid) if not isinstance(n_valid, list) else int(n_valid[0])

    # Prepare invalids on full flow
    if sample.flow_full is not None:
        sample.flow_full = prepare_invalids(sample.flow_full, dataset_name)

    # Downsample flow once (only if downsample_feat_size is provided)
    if sample.flow_feat is None and sample.flow_full is not None:
        if downsample_feat_size is not None:
            sample.flow_feat = downsample_flow(sample.flow_full, feat_size=downsample_feat_size)
        else:
            # If no downsampling requested, use full-resolution flow
            sample.flow_feat = sample.flow_full
    if sample.flow_feat is not None:
        sample.flow_feat = _flow_make_finite(sample.flow_feat)

    # pckthres
    if sample.pckthres is None and sample.src_img is not None:
        _, H, W = sample.src_img.shape
        sample.pckthres = torch.tensor(max(H, W), dtype=torch.float32, device=sample.src_img.device)

    return sample


def normalize_images(sample: CommonSample, enable: bool) -> CommonSample:
    if not enable or sample.src_img is None:
        return sample
    mean = torch.tensor([0.485, 0.456, 0.406], device=sample.src_img.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=sample.src_img.device).view(1, 3, 1, 1)
    
    # First, normalize to 0-1 range
    if sample.src_img.max() > 1.0:
        sample.src_img = sample.src_img / 255.0
    sample.src_img = torch.clamp(sample.src_img, 0.0, 1.0)
    
    if sample.trg_img is not None:
        if sample.trg_img.max() > 1.0:
            sample.trg_img = sample.trg_img / 255.0
        sample.trg_img = torch.clamp(sample.trg_img, 0.0, 1.0)
    
    # Then, standardize with ImageNet statistics
    sample.src_img = (sample.src_img - mean) / std
    if sample.trg_img is not None:
        sample.trg_img = (sample.trg_img - mean) / std
    return sample


def collate_common_samples(
    samples: List[CommonSample],
    max_kps: Optional[int],
    target_device: torch.device,
) -> dict:
    def squeeze_leading_one(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.shape[0] == 1:
            return t.squeeze(0)
        return t

    # Pad keypoints
    src_list = [s.src_kps for s in samples]
    trg_list = [s.trg_kps for s in samples]
    padded_src, padded_trg, n_pts = pad_keypoints_batch(src_list, trg_list, max_kps)

    for s, src_kps, trg_kps, n in zip(samples, padded_src, padded_trg, n_pts):
        s.src_kps = src_kps
        s.trg_kps = trg_kps
        s.n_pts = n

    dicts = []
    for s in samples:
        d = s.to_dict()
        d.pop("meta", None)
        d.pop("flow_feat", None)  # keep flow_downsampled alias only
        # Ensure no accidental batch dim snuck in
        for key in ["src_img", "trg_img", "flow", "flow_full", "flow_downsampled"]:
            if key in d:
                d[key] = squeeze_leading_one(d[key])
                if key == "flow_downsampled":
                    d[key] = _flow_make_finite(d[key])  # only zero-out invalids on feature-grid flow
        dicts.append(d)
    # Drop keys that are None for any sample to keep default_collate happy
    keys = list(dicts[0].keys())
    for key in keys:
        if any(d.get(key) is None for d in dicts):
            for d in dicts:
                d.pop(key, None)

    batch = default_collate(dicts)

    # If any image/flow tensors ended up with an extra singleton batch dim, remove it
    for key in ["src_img", "trg_img", "flow", "flow_full", "flow_downsampled"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            t = batch[key]
            if t.dim() == 5 and t.shape[1] == 1:
                batch[key] = t.squeeze(1)

    # Move tensors to target device once
    for key, val in batch.items():
        if isinstance(val, torch.Tensor) and val.device != target_device:
            batch[key] = val.to(target_device, non_blocking=(target_device.type == "cuda"))
    return batch
