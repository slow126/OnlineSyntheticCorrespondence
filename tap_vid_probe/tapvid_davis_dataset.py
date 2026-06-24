"""TAP-Vid-DAVIS eval dataset + adapter for the correspondence pipeline.

Mirrors `PointOdysseySimpleDataset` / `PointOdysseyAdapter` exactly (the sparse
real-motion benchmark this is modelled on). The dataset reduces each DAVIS video to
image PAIRS (t1, t2=t1+stride) and returns, for the points co-visible in BOTH frames,
the same dict PointOdyssey's loader returns:

    {src_img (3,512,512) float[0,1], trg_img (3,512,512), src_kps (2,M), trg_kps (2,M),
     n_pts}

It deliberately returns NO flow / NO pckthres: the collate pipeline
(`ensure_flow_and_kps` in collate_pipeline.py) builds the dense flow from the kps with
`flow_from_kps` and sets pckthres = max(H,W), identically to PointOdyssey. Images are
left un-normalised in [0,1]; the collate applies ImageNet normalisation (the adapter's
`normalize_images` default is True for non-PF datasets).

Consumes the mmap cache produced by `preprocess_davis.py`.

src/trg direction (mirrors PointOdyssey reverse_flow=True): with reverse_flow=True the
SOURCE is the LATER frame and the TARGET the EARLIER frame, so eval predicts "given a
point in the earlier (trg) frame, where is it in the later (src) frame" — forward
tracking, the natural correspondence direction. Flip it to sanity-check (PCK collapses).

Set env TAPVID_PROBE_DEBUG=1 for verbose per-sample shape/range prints.
"""
import os
import pickle
from typing import Dict, Optional

import numpy as np
import torch

from src.data.synth.common.common_sample import CommonSample

_DEBUG = os.environ.get("TAPVID_PROBE_DEBUG", "0") not in ("0", "", "false", "False")


class TapVidDavisSimpleDataset(torch.utils.data.Dataset):
    """Returns 512x512 image pairs + co-visible query keypoints from TAP-Vid-DAVIS.

    Args:
        cache_dir: dir with frames.npy + index.pkl (from preprocess_davis.py).
        stride: frame gap of each pair (t2 = t1 + stride). Larger => more motion / harder.
        frame_step: sample t1 every `frame_step` frames (controls #pairs / eval cost).
        min_pts: drop pairs with fewer than this many co-visible points.
        reverse_flow: True => src=later frame, trg=earlier frame (PointOdyssey convention).
        max_pairs_per_video: optional cap on pairs per video (None = no cap).
    """

    def __init__(self,
                 cache_dir: str,
                 stride: int = 5,
                 frame_step: int = 5,
                 min_pts: int = 1,
                 reverse_flow: bool = True,
                 max_pairs_per_video: Optional[int] = None,
                 verbose: bool = True,
                 **_ignored):
        self.cache_dir = cache_dir
        self.stride = int(stride)
        self.frame_step = max(1, int(frame_step))
        self.min_pts = int(min_pts)
        self.reverse_flow = bool(reverse_flow)
        self._device = torch.device("cpu")

        frames_path = os.path.join(cache_dir, "frames.npy")
        index_path = os.path.join(cache_dir, "index.pkl")
        if not (os.path.exists(frames_path) and os.path.exists(index_path)):
            raise FileNotFoundError(
                f"[tapvid] cache missing in {cache_dir} (need frames.npy + index.pkl). "
                f"Run: python tap_vid_probe/preprocess_davis.py --out {cache_dir}")

        # Memory-mapped frames are SHARED across processes/workers via the OS page cache.
        self.frames = np.load(frames_path, mmap_mode="r")  # (TotalFrames,512,512,3) uint8
        with open(index_path, "rb") as f:
            self.index = pickle.load(f)
        self.img_size = int(self.frames.shape[1])

        # Build the flat pair list: (offset, src_frame, trg_frame, covis_idx).
        self.pairs = []
        total_pts = 0
        for vid in self.index:
            occ = vid["occluded"]            # (N,S) bool, True=occluded
            S = int(vid["S"])
            n_added = 0
            for t1 in range(0, S - self.stride, self.frame_step):
                t2 = t1 + self.stride
                covis = (~occ[:, t1]) & (~occ[:, t2])   # visible in BOTH
                idx = np.nonzero(covis)[0]
                if idx.size < self.min_pts:
                    continue
                # reverse_flow: src=later (t2), trg=earlier (t1)
                src_f, trg_f = (t2, t1) if self.reverse_flow else (t1, t2)
                self.pairs.append((vid["offset"], src_f, trg_f, idx,
                                   vid["points512"]))
                total_pts += idx.size
                n_added += 1
                if max_pairs_per_video is not None and n_added >= max_pairs_per_video:
                    break

        if verbose:
            print(f"[tapvid] {len(self.index)} videos, {len(self.pairs)} pairs "
                  f"(stride={self.stride}, frame_step={self.frame_step}, "
                  f"min_pts={self.min_pts}, reverse_flow={self.reverse_flow}), "
                  f"avg covis pts/pair={total_pts/max(len(self.pairs),1):.1f}, "
                  f"img={self.img_size}px", flush=True)
        if len(self.pairs) == 0:
            raise RuntimeError(f"[tapvid] no usable pairs (stride={self.stride} too large?)")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_frame(self, row: int) -> torch.Tensor:
        # mmap row (512,512,3) uint8 -> (3,512,512) float [0,1]; .copy() detaches from mmap
        arr = np.asarray(self.frames[row]).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).to(torch.float32) / 255.0

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        offset, src_f, trg_f, idx, points512 = self.pairs[i]
        src_img = self._load_frame(offset + src_f)
        trg_img = self._load_frame(offset + trg_f)

        # points512: (N,S,2) in (x,y) 512-pixel space. (M,2) -> (2,M)
        src_kps = torch.from_numpy(points512[idx, src_f, :].T.copy()).to(torch.float32)
        trg_kps = torch.from_numpy(points512[idx, trg_f, :].T.copy()).to(torch.float32)
        n_pts = src_kps.shape[1]

        if _DEBUG and i < 3:
            print(f"[tapvid][dbg] i={i} src_f={src_f} trg_f={trg_f} n_pts={n_pts} "
                  f"img={tuple(src_img.shape)} kps_x[{src_kps[0].min():.0f},{src_kps[0].max():.0f}] "
                  f"kps_y[{src_kps[1].min():.0f},{src_kps[1].max():.0f}] "
                  f"mean|disp|={(src_kps-trg_kps).abs().mean():.1f}px", flush=True)

        return {
            "src_img": src_img,
            "trg_img": trg_img,
            "src_kps": src_kps,                       # (2,M) pixel space
            "trg_kps": trg_kps,                       # (2,M) pixel space
            "n_pts": torch.tensor(n_pts, dtype=torch.int32),
        }

    # ---- device API parity with other datasets (no-ops; data stays on CPU) ----
    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    def cpu(self):
        self._device = torch.device("cpu")
        return self

    def cuda(self, device=None):
        return self


class TapVidDavisAdapter:
    """Adapter mirroring PointOdysseyAdapter. Registered as 'tapvid_davis'."""

    name = "tapvid_davis"
    normalize_images = True          # collate applies ImageNet norm (images returned in [0,1])
    target_device = torch.device("cpu")
    flow_is_feat_res = False

    def __init__(self,
                 cache_dir: str,
                 split: str = "val",
                 tapvid_stride: int = 5,
                 tapvid_frame_step: int = 5,
                 tapvid_min_pts: int = 1,
                 reverse_flow: bool = True,
                 tapvid_max_pairs_per_video: Optional[int] = None,
                 **kwargs):
        self.dataset = TapVidDavisSimpleDataset(
            cache_dir=cache_dir,
            stride=tapvid_stride,
            frame_step=tapvid_frame_step,
            min_pts=tapvid_min_pts,
            reverse_flow=reverse_flow,
            max_pairs_per_video=tapvid_max_pairs_per_video,
            verbose=kwargs.get("verbose", True),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> CommonSample:
        raw = self.dataset[idx]
        return CommonSample(
            src_img=raw.get("src_img"),
            trg_img=raw.get("trg_img"),
            flow_full=None,                 # collate builds dense flow from kps
            src_kps=raw.get("src_kps"),
            trg_kps=raw.get("trg_kps"),
            n_pts=raw.get("n_pts"),
        )
