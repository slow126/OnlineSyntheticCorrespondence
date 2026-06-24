from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from src.data.synth.adapters import build_adapter, SyntheticAdapter
from src.data.synth.common.common_sample import CommonSample
from src.data.synth.collate_pipeline import (
    resize_sample,
    ensure_flow_and_kps,
    normalize_images,
    collate_common_samples,
)


def _strip_leading_batch(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """If a per-sample tensor already has a batch dim of size 1, squeeze it."""
    if t is None:
        return None
    if isinstance(t, torch.Tensor) and t.dim() >= 4 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


class CorrespondenceDataset(Dataset):
    """
    Thin wrapper that delegates dataset-specific loading to adapters and uses a
    small, deterministic collate pipeline to produce:
      - flow (full res), flow_full (alias), flow_downsampled (feature res)
      - src_kps/trg_kps padded to a common size (with n_pts)
      - pckthres and normalized images
    """

    def __init__(
        self,
        dataset_name: str,
        verbose: bool = False,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.debug = debug
        self.synthetic_flow_warp = kwargs.get("synthetic_flow_warp", dataset_name == "synthetic_2d_warp")
        self.synthetic_flow_warp_swap = kwargs.get(
            "synthetic_flow_warp_swap", dataset_name == "synthetic_2d_warp"
        )
        self._flow_warp_grid_cache = {}

        self.size: Optional[Tuple[int, int]] = kwargs.get("size", (512, 512))
        if self.size is not None:
            if isinstance(self.size, (list, tuple)) and len(self.size) == 2:
                self.size = (int(self.size[0]), int(self.size[1]))
            elif isinstance(self.size, (int, float)):
                side = int(self.size)
                self.size = (side, side)
            else:
                raise ValueError(f"size must be int or (H, W), got {self.size!r}")
        self.max_kps: Optional[int] = kwargs.get("max_kps", None)
        self.downsample_feat_size: int = kwargs.get("downsample_flow", 32)
        self.prefer_all_dense: bool = kwargs.get("dense_kps_use_all", True)
        # occlusion_mask: keep occluded feature cells as inf so the sparse
        # endpoint-error loss skips them (Kubric flow only). Default off.
        self.occlusion_mask: bool = bool(kwargs.get("occlusion_mask", False))

        # Device policy: synthetic prefers GPU, others default to CPU for worker safety
        target_device_str = kwargs.get("target_device", None)
        if target_device_str:
            self.target_device = torch.device(target_device_str)
        else:
            if dataset_name.startswith("synthetic") and torch.cuda.is_available():
                self.target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                self.target_device = torch.device("cpu")

        # Normalization policy
        already_normalized = ["pfpascal", "pfwillow", "spair"]
        normalize_flag = kwargs.get("normalize_images", None)
        if normalize_flag is None:
            self.normalize_images_flag = dataset_name not in already_normalized
        else:
            self.normalize_images_flag = normalize_flag

        # Build adapter (handles dataset-specific loading)
        adapter_excludes = {
            "size",
            "max_kps",
            "downsample_flow",
            "dense_kps_use_all",
            "target_device",
            "normalize_images",
            "debug",
            "verbose",
            "synthetic_flow_warp",
            "synthetic_flow_warp_swap",
        }
        adapter_kwargs = {k: v for k, v in kwargs.items() if k not in adapter_excludes}
        self.adapter = build_adapter(dataset_name, **adapter_kwargs)

    def _get_flow_warp_base_grid(
        self,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (height, width, device, dtype)
        grid = self._flow_warp_grid_cache.get(key)
        if grid is None:
            y_coords, x_coords = torch.meshgrid(
                torch.arange(height, device=device, dtype=dtype),
                torch.arange(width, device=device, dtype=dtype),
                indexing="ij",
            )
            grid = torch.stack((x_coords, y_coords), dim=0).unsqueeze(0)
            self._flow_warp_grid_cache[key] = grid
        return grid

    def _warp_src_with_flow(self, src_img: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        if src_img is None or flow is None:
            return src_img
        squeeze = False
        if src_img.dim() == 3:
            src_img = src_img.unsqueeze(0)
            flow = flow.unsqueeze(0)
            squeeze = True

        _, _, height, width = src_img.shape
        base_grid = self._get_flow_warp_base_grid(height, width, flow.device, flow.dtype)
        mapping = base_grid + flow

        denom_w = max(width - 1, 1)
        denom_h = max(height - 1, 1)
        mapping[:, 0].mul_(2.0 / denom_w).sub_(1.0)
        mapping[:, 1].mul_(2.0 / denom_h).sub_(1.0)
        grid = mapping.permute(0, 2, 3, 1)

        valid = torch.isfinite(flow).all(1)
        if not valid.all():
            grid = grid.clone()
            grid[~valid] = 2.0

        warped = F.grid_sample(
            src_img,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        if squeeze:
            warped = warped.squeeze(0)
        return warped

    def _invert_flow(self, flow: torch.Tensor) -> torch.Tensor:
        if flow is None:
            return flow
        squeeze = False
        if flow.dim() == 3:
            flow = flow.unsqueeze(0)
            squeeze = True

        batch_size, _, height, width = flow.shape
        base_grid = self._get_flow_warp_base_grid(height, width, flow.device, flow.dtype)
        src_x = base_grid[:, 0] + flow[:, 0]
        src_y = base_grid[:, 1] + flow[:, 1]

        valid = torch.isfinite(flow).all(1)
        src_x_round = src_x.round().long()
        src_y_round = src_y.round().long()
        in_bounds = (
            valid
            & (src_x_round >= 0)
            & (src_x_round < width)
            & (src_y_round >= 0)
            & (src_y_round < height)
        )

        inv_sum = flow.new_zeros((batch_size, 2, height * width))
        inv_count = flow.new_zeros((batch_size, 1, height * width))

        for b in range(batch_size):
            mask = in_bounds[b]
            if not mask.any():
                continue
            idx = (src_y_round[b][mask] * width + src_x_round[b][mask]).view(1, -1)
            vals = -flow[b, :, mask]
            inv_sum[b].scatter_add_(1, idx.expand(2, -1), vals)
            inv_count[b].scatter_add_(1, idx, torch.ones_like(idx, dtype=flow.dtype))

        inv_flow = inv_sum / inv_count.clamp(min=1.0)
        inv_flow = inv_flow.view(batch_size, 2, height, width)
        invalid = inv_count.view(batch_size, 1, height, width) == 0
        inv_flow[invalid.expand_as(inv_flow)] = float("inf")

        if squeeze:
            inv_flow = inv_flow.squeeze(0)
        return inv_flow

    def __len__(self):
        return len(self.adapter)

    def __getitem__(self, idx):
        return self.adapter[idx]

    def _process_synthetic_batch(self, batch):
        """Synthetic uses the processor's batch API."""
        batch_size = len(batch)
        # Each item is [src_dict, trg_dict]
        src_dicts = [item[0] for item in batch]
        trg_dicts = [item[1] for item in batch]

        collated_src = default_collate(src_dicts)
        collated_trg = default_collate(trg_dicts)
        collated_batch = [collated_src, collated_trg]

        collated_batch = self.adapter.dataset.processor.batch_to_device(
            collated_batch, self.adapter.dataset.processor.device
        )
        processed_batch = self.adapter.dataset.processor.process_scene(collated_batch)
        if self.synthetic_flow_warp:
            flow = processed_batch.get("flow_full")
            if flow is None:
                flow = processed_batch.get("flow")
            src_img = processed_batch.get("src_img")
            if flow is not None and src_img is not None:
                processed_batch["trg_img"] = self._warp_src_with_flow(src_img, flow)
            if self.synthetic_flow_warp_swap and src_img is not None:
                swap_mask = torch.rand(batch_size, device=src_img.device) < 0.5
                if swap_mask.any():
                    trg_img = processed_batch.get("trg_img")
                    if trg_img is not None:
                        src_tmp = src_img[swap_mask].clone()
                        trg_tmp = trg_img[swap_mask].clone()
                        src_img[swap_mask] = trg_tmp
                        trg_img[swap_mask] = src_tmp
                        processed_batch["src_img"] = src_img
                        processed_batch["trg_img"] = trg_img
                    if processed_batch.get("src_kps") is not None and processed_batch.get("trg_kps") is not None:
                        src_kps = processed_batch["src_kps"]
                        trg_kps = processed_batch["trg_kps"]
                        src_kps_tmp = src_kps[swap_mask].clone()
                        trg_kps_tmp = trg_kps[swap_mask].clone()
                        src_kps[swap_mask] = trg_kps_tmp
                        trg_kps[swap_mask] = src_kps_tmp
                        processed_batch["src_kps"] = src_kps
                        processed_batch["trg_kps"] = trg_kps
                    if processed_batch.get("flow_full") is not None:
                        flow_full = processed_batch["flow_full"]
                        inv_flow_full = self._invert_flow(flow_full)
                        flow_full[swap_mask] = inv_flow_full[swap_mask]
                        processed_batch["flow_full"] = flow_full
                    if processed_batch.get("flow") is not None:
                        flow = processed_batch["flow"]
                        inv_flow = self._invert_flow(flow)
                        flow[swap_mask] = inv_flow[swap_mask]
                        processed_batch["flow"] = flow

        samples = []
        for i in range(batch_size):
            sample_dict = {}
            for key, value in processed_batch.items():
                if isinstance(value, torch.Tensor):
                    sample_dict[key] = value[i]
                elif isinstance(value, (list, tuple)):
                    sample_dict[key] = value[i]
                else:
                    sample_dict[key] = value
            samples.append(
                CommonSample(
                    src_img=sample_dict.get("src_img"),
                    trg_img=sample_dict.get("trg_img"),
                    flow_full=sample_dict.get("flow_full") if sample_dict.get("flow_full") is not None else sample_dict.get("flow"),
                    src_kps=sample_dict.get("src_kps"),
                    trg_kps=sample_dict.get("trg_kps"),
                    n_pts=sample_dict.get("n_pts"),
                    pckthres=sample_dict.get("pckthres"),
                )
            )
        return samples

    def collate_fn(self, batch):
        # Synthetic stays on GPU and uses its processor to collate first
        if isinstance(self.adapter, SyntheticAdapter):
            samples = self._process_synthetic_batch(batch)
        else:
            samples = batch

        processed_samples = []
        for sample in samples:
            # Adapter returns CommonSample for non-synthetic paths
            if not isinstance(sample, CommonSample):
                # Try to coerce dict to CommonSample for safety
                sample = CommonSample(
                    src_img=_strip_leading_batch(sample.get("src_img")),
                    trg_img=_strip_leading_batch(sample.get("trg_img")),
                    flow_full=_strip_leading_batch(sample.get("flow_full") or sample.get("flow")),
                    flow_feat=_strip_leading_batch(sample.get("flow_downsampled")),
                    src_kps=sample.get("src_kps"),
                    trg_kps=sample.get("trg_kps"),
                    n_pts=sample.get("n_pts"),
                    pckthres=sample.get("pckthres"),
                )

            # Remove accidental leading batch dims from sources that already batch internally
            sample.src_img = _strip_leading_batch(sample.src_img)
            sample.trg_img = _strip_leading_batch(sample.trg_img)
            sample.flow_full = _strip_leading_batch(sample.flow_full)
            sample.flow_feat = _strip_leading_batch(sample.flow_feat)

            sample = resize_sample(sample, self.size)
            sample = ensure_flow_and_kps(
                sample,
                dataset_name=self.dataset_name,
                max_kps=self.max_kps,
                downsample_feat_size=self.downsample_feat_size,
                prefer_all_dense=self.prefer_all_dense,
                occlusion_mask=self.occlusion_mask,
            )
            sample = normalize_images(sample, self.normalize_images_flag)
            processed_samples.append(sample)

        batch_out = collate_common_samples(
            processed_samples,
            max_kps=self.max_kps,
            target_device=self.target_device,
            occlusion_mask=self.occlusion_mask,
        )
        return batch_out
