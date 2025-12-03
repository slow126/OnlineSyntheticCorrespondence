from typing import Optional, Tuple
import torch
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

        self.size: Optional[Tuple[int, int]] = kwargs.get("size", (512, 512))
        self.max_kps: Optional[int] = kwargs.get("max_kps", None)
        self.downsample_feat_size: int = kwargs.get("downsample_flow", 32)
        self.prefer_all_dense: bool = kwargs.get("dense_kps_use_all", True)

        # Device policy: synthetic prefers GPU, others default to CPU for worker safety
        target_device_str = kwargs.get("target_device", None)
        if target_device_str:
            self.target_device = torch.device(target_device_str)
        else:
            if dataset_name == "synthetic" and torch.cuda.is_available():
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
        }
        adapter_kwargs = {k: v for k, v in kwargs.items() if k not in adapter_excludes}
        self.adapter = build_adapter(dataset_name, **adapter_kwargs)

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
            )
            sample = normalize_images(sample, self.normalize_images_flag)
            processed_samples.append(sample)

        batch_out = collate_common_samples(
            processed_samples,
            max_kps=self.max_kps,
            target_device=self.target_device,
        )
        return batch_out
