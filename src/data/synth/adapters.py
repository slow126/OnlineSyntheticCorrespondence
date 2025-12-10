from typing import Dict, Optional
import torch
import torch.nn.functional as F

from src.data.synth.common.common_sample import CommonSample
import models.CATs_PlusPlus.data.download as download
from src.data.synth.datasets.FlyingThingsDataset import FlyingThingsSimpleDataset
from src.data.synth.datasets.PointOdysseyCorrespondence import PointOdysseySimpleDataset
from src.data.synth.datasets.KittiDataset import KittiSimpleDataset
from src.data.synth.datasets.OnlineCorrespondenceDataset import OnlineCorrespondenceDataset
from src.data.synth.datasets.TSSDataset import TSSSimpleDataset
from src.data.synth.datasets.MiddleburyDataset import MiddleburySimpleDataset
from src.data.synth.datasets.MonkaaDataset import MonkaaSimpleDataset
from src.data.synth.datasets.DrivingDataset import DrivingSimpleDataset
from src.data.synth.datasets.SintelDataset import SintelSimpleDataset
from src.data.synth.datasets.HD1KDataset import HD1KSimpleDataset
from src.data.synth.datasets.VirtualKitti2Dataset import VirtualKitti2SimpleDataset


class BaseAdapter:
    name: str = "base"
    normalize_images: bool = True
    target_device: torch.device = torch.device("cpu")
    flow_is_feat_res: bool = False  # for pf datasets

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx) -> CommonSample:
        raise NotImplementedError


class FlyingThingsAdapter(BaseAdapter):
    name = "flyingthings"

    def __init__(self, datapath: str, split: str, reverse_flow: bool = False, transforms=None, **_: Dict):
        self.dataset = FlyingThingsSimpleDataset(
            root=datapath,
            split=split,
            transforms=transforms,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class PointOdysseyAdapter(BaseAdapter):
    name = "pointodyssey"

    def __init__(self, dataset_location: str, split: str = "train", dset: Optional[str] = None, **kwargs):
        # accept either split or dset for convenience
        dset_to_use = dset if dset is not None else split
        self.dataset = PointOdysseySimpleDataset(
            dataset_location=dataset_location,
            dset=dset_to_use,
            use_augs=kwargs.get("pointodyssey_use_augs", False),
            S=kwargs.get("pointodyssey_sequence_length", 8),
            N=kwargs.get("pointodyssey_num_pts_to_track", 32),
            strides=kwargs.get("pointodyssey_strides", [1, 2, 4]),
            quick=kwargs.get("pointodyssey_quick", False),
            verbose=kwargs.get("pointodyssey_verbose", False),
            reverse_flow=kwargs.get("reverse_flow", True),
            thres=kwargs.get("thres", "img"),
            use_all_valid=kwargs.get("use_all_valid", False),
            disable_motion_filter=kwargs.get("pointodyssey_disable_motion_filter", False),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
            src_kps=raw.get("src_kps"),
            trg_kps=raw.get("trg_kps"),
            n_pts=raw.get("n_pts"),
        )


class KittiAdapter(BaseAdapter):
    name = "kitti"

    def __init__(self, datapath: str, split: str, version: str = "2015", reverse_flow: bool = False, **kwargs):
        kitti_root = f"{datapath}/kitti-{version}"
        self.dataset = KittiSimpleDataset(
            root=kitti_root,
            split=split,
            version=version,
            occ_type=kwargs.get("kitti_occ_type", "occ"),
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class TSSAdapter(BaseAdapter):
    name = "tss"

    def __init__(self, datapath: str, split: str = "train", reverse_flow: bool = False, **kwargs):
        self.dataset = TSSSimpleDataset(root=datapath, reverse_flow=reverse_flow)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
            src_kps=raw.get("src_kps"),
            trg_kps=raw.get("trg_kps"),
            n_pts=raw.get("n_pts"),
        )


class MiddleburyAdapter(BaseAdapter):
    name = "middlebury"

    def __init__(self, datapath: str, split: str = "train", reverse_flow: bool = False, **kwargs):
        self.dataset = MiddleburySimpleDataset(
            root=datapath,
            split=split,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class SyntheticAdapter(BaseAdapter):
    name = "synthetic"
    normalize_images = False

    def __init__(self, geometry_config_path: str, processor_config_path: str, split: str = "train", **kwargs):
        self.dataset = OnlineCorrespondenceDataset(
            geometry_config_path=geometry_config_path,
            processor_config_path=processor_config_path,
            split=split,
            opengl_device_index=kwargs.get("opengl_device_index", None),
            geometry_config_overrides=kwargs.get("geometry_config_overrides", None),
        )
        self.collate_first = True  # synthetic expects processor-based batching

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MonkaaAdapter(BaseAdapter):
    name = "monkaa"

    def __init__(self, datapath: str, split: str, reverse_flow: bool = False, transforms=None, **_: Dict):
        self.dataset = MonkaaSimpleDataset(
            root=datapath,
            split=split,
            transforms=transforms,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class DrivingAdapter(BaseAdapter):
    name = "driving"

    def __init__(self, datapath: str, split: str, reverse_flow: bool = False, transforms=None, **_: Dict):
        self.dataset = DrivingSimpleDataset(
            root=datapath,
            split=split,
            transforms=transforms,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class SintelAdapter(BaseAdapter):
    name = "sintel"

    def __init__(self, sintel_root: str, split: str, pass_name: str = "clean", size=None, reverse_flow: bool = False, transforms=None, **kwargs):
        self.dataset = SintelSimpleDataset(
            root=sintel_root,
            split=split,
            pass_name=pass_name,
            size=tuple(size) if size is not None else None,
            transforms=transforms,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class HD1KAdapter(BaseAdapter):
    name = "hd1k"

    def __init__(self, datapath: str, split: str, reverse_flow: bool = False, transforms=None, **_: Dict):
        self.dataset = HD1KSimpleDataset(
            root=datapath,
            split=split,
            transforms=transforms,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class VirtualKitti2Adapter(BaseAdapter):
    name = "virtualkitti2"

    def __init__(self, datapath: str, split: str, camera: str = "Camera_0", reverse_flow: bool = False, **kwargs):
        self.dataset = VirtualKitti2SimpleDataset(
            root=datapath,
            split=split,
            camera=camera,
            reverse_flow=reverse_flow,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=raw.get("flow"),
        )


class BenchmarkAdapter(BaseAdapter):
    """PF-Pascal/Willow/SPair; flows are already feature-res in many cases."""

    name = "benchmark"
    flow_is_feat_res = True

    def __init__(self, dataset_name: str, datapath: str, split: str, thres: str = "img", **kwargs):
        split_map = {"train": "trn", "val": "val", "test": "test"}
        split_mapped = split_map.get(split, split)
        device = kwargs.get("device", torch.device("cpu"))
        self.dataset = download.load_dataset(
            dataset_name,
            datapath,
            thres,
            device,
            split_mapped,
            kwargs.get("augmentation", False),
            kwargs.get("downsample_flow", 32),
        )
        self.normalize_images = False  # already normalized in dataset
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        # Most of these datasets expose flow at feature resolution plus keypoints
        flow = raw.get("flow") if isinstance(raw, dict) else None
        src_kps = raw.get("src_kps") if isinstance(raw, dict) else None
        trg_kps = raw.get("trg_kps") if isinstance(raw, dict) else None
        
        # Prefer rebuilding full-res flow from keypoints for benchmark datasets to avoid
        # bias from upsampling feature-resolution flow.
        prefer_kp_flow = (
            self.dataset_name in {"pfpascal", "pfwillow", "spair"}
            and src_kps is not None
            and trg_kps is not None
        )
        
        # Convert feature-space flow to full-resolution flow when not using the keypoint path.
        # Old datasets expose flow at feature resolution (e.g., 32x32) with values divided by
        # (img_size // feat_size). We need to:
        # 1. Multiply by scale factor to get pixel-space values
        # 2. Upsample to full resolution (flow vectors don't need scaling during upsampling)
        flow_full = None
        if flow is not None and not prefer_kp_flow:
            # Get image size to determine scale factor
            if raw.get("src_img") is not None:
                _, H, W = raw["src_img"].shape
            else:
                H = W = 512  # Default assumption
            
            # Scale factor: img_size / feat_size (typically 512/32 = 16)
            feat_size = flow.shape[-1]  # flow is [2, H, W] where H=W=feat_size
            scale_factor = W // feat_size
            
            # Convert to pixel space and upsample to full resolution
            flow_pixel_space = flow * scale_factor
            flow_full = F.interpolate(
                flow_pixel_space.unsqueeze(0), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            # Note: flow vectors don't need scaling during upsampling - they represent displacement
        elif prefer_kp_flow:
            # Drop provided feature-space flow so the collate pipeline rebuilds full-res flow
            # directly from keypoints and then derives the downsampled version from that.
            flow = None
        
        return CommonSample(
            src_img=raw.get("src_img") if isinstance(raw, dict) else None,
            trg_img=raw.get("trg_img") if isinstance(raw, dict) else None,
            flow_full=flow_full,
            flow_feat=flow,
            src_kps=raw.get("src_kps") if isinstance(raw, dict) else None,
            trg_kps=raw.get("trg_kps") if isinstance(raw, dict) else None,
            n_pts=raw.get("n_pts") if isinstance(raw, dict) else None,
            meta={"dataset": self.dataset_name},
        )


ADAPTER_REGISTRY = {
    "flyingthings": FlyingThingsAdapter,
    "pointodyssey": PointOdysseyAdapter,
    "kitti": KittiAdapter,
    "kitti2012": KittiAdapter,
    "kitti2015": KittiAdapter,
    "synthetic": SyntheticAdapter,
    "tss": TSSAdapter,
    "middlebury": MiddleburyAdapter,
    "monkaa": MonkaaAdapter,
    "driving": DrivingAdapter,
    "sintel": SintelAdapter,
    "hd1k": HD1KAdapter,
    "virtualkitti2": VirtualKitti2Adapter,
    "pfpascal": BenchmarkAdapter,
    "pfwillow": BenchmarkAdapter,
    "spair": BenchmarkAdapter,
}


def build_adapter(dataset_name: str, **kwargs) -> BaseAdapter:
    cls = ADAPTER_REGISTRY.get(dataset_name)
    if cls is None:
        raise ValueError(f"Unknown dataset {dataset_name}")
    if dataset_name in ["pfpascal", "pfwillow", "spair"]:
        return cls(dataset_name=dataset_name, **kwargs)
    if dataset_name in ["kitti2012", "kitti2015"]:
        version = "2012" if "2012" in dataset_name else "2015"
        kwargs["version"] = version
        return cls(**kwargs)
    return cls(**kwargs)
