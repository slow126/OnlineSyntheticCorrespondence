"""
MOVi-F dataset for optical flow correspondence training.

Reads directly from TFRecord shards — no extraction step needed.
Keeps storage at zero overhead over the raw download.

Flow convention (matches rest of project):
  - src = frame t,  trg = frame t+1  (reverse_flow=False, default)
  - flow[0, y, x] = dx  (horizontal displacement, trg -> src)
  - flow[1, y, x] = dy  (vertical   displacement, trg -> src)
  Kubric backward_flow[t+1] already gives trg->src; we only reorder channels.

  reverse_flow=True swaps src/trg and uses forward_flow[t] instead, which gives
  trg->src from the new pair's perspective (same trick as Sintel/FlyingThings).

Streaming design:
  __getitem__(idx) ignores idx and returns the next pair from an internal TF
  dataset iterator. This works correctly with PyTorch's DataLoader:
    - shuffle=True on the DataLoader generates random idx calls, which we ignore;
      randomness instead comes from the TF shuffle pipeline per worker.
    - Each DataLoader worker gets its own shard subset so workers don't overlap.
    - repeat() in the TF pipeline prevents StopIteration mid-epoch.
  __len__ is estimated from shard count; steps_per_epoch caps actual usage.

Args (passed from adapter / dataset config):
  datapath   : directory containing .tfrecord shard files
               e.g. /home/spencer/Data/kubric_tfds/movi_f/512x512/1.0.0
  split      : 'train', 'validation', or 'test'
  reverse_flow: see above
  kubric_dir : path to kubric repo root (needed to load the movi_f builder)
               default: /home/spencer/Projects/kubric
  num_frames : frames per video (24 for MOVi-F); used to estimate __len__
  shuffle_buffer: TF shuffle buffer in units of videos (default 64)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_DEFAULT_KUBRIC_DIR = "/home/spencer/Projects/kubric"
_NUM_FRAMES = 24          # fixed for MOVi-F
_PAIRS_PER_VIDEO = _NUM_FRAMES - 1   # 23


def _load_movi_builder(kubric_dir: str, config: str = "512x512"):
    """Load and return a MoviF builder instance from the kubric repo.

    The kubric builder imports png/imageio/etils at the top level but only uses
    them in _generate_examples (rendering), not in _info() or deserialization.
    We stub any missing packages so the import succeeds without them installed.
    """
    import types

    for mod_name in ("png", "imageio", "epy",
                     "etils", "etils.epath", "etils.epy",
                     "etils.edc", "etils.epy.pretty_repr"):
        if mod_name not in sys.modules:
            try:
                importlib.import_module(mod_name)
            except ImportError:
                sys.modules[mod_name] = types.ModuleType(mod_name)

    builder_path = Path(kubric_dir).resolve() / "challenges" / "movi" / "movi_f.py"
    if not builder_path.exists():
        raise FileNotFoundError(
            f"movi_f builder not found at {builder_path}. "
            f"Set kubric_dir to your kubric repo root."
        )
    kubric_root = str(Path(kubric_dir).resolve())
    if kubric_root not in sys.path:
        sys.path.insert(0, kubric_root)
    spec = importlib.util.spec_from_file_location("movi_f_builder", builder_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.MoviF(config=config, data_dir="/tmp/_movi_f_dummy_builder")


def _list_shards(shard_dir: Path, split: str) -> list:
    pattern = f"movi_f-{split}.tfrecord-*"
    return sorted(
        p for p in shard_dir.glob(pattern)
        if not p.name.endswith(".gstmp")
    )


def _decode_flow(flow_uint16: np.ndarray, flow_range: np.ndarray) -> np.ndarray:
    flow = flow_uint16.astype(np.float32)
    return flow / 65535.0 * (float(flow_range[1]) - float(flow_range[0])) + float(flow_range[0])


def _to_image_tensor(frame: np.ndarray) -> torch.Tensor:
    """uint8 [H,W,3] -> float32 [3,H,W] in [0,1]."""
    return torch.from_numpy(frame.astype(np.float32)).permute(2, 0, 1) / 255.0


def _to_flow_tensor(flow_hw2: np.ndarray) -> torch.Tensor:
    """Kubric [H,W,2] (dy,dx) -> project [2,H,W] (dx,dy)."""
    return torch.from_numpy(flow_hw2).permute(2, 0, 1)[[1, 0]]


def _make_pair(example: dict, t: int, reverse_flow: bool) -> Dict[str, torch.Tensor]:
    video = example["video"]          # [T, H, W, 3] uint8
    meta  = example["metadata"]

    if not reverse_flow:
        # src=frame[t], trg=frame[t+1], flow=bflow[t+1]  (trg->src) ✓
        src_img = _to_image_tensor(video[t])
        trg_img = _to_image_tensor(video[t + 1])
        bflow   = _decode_flow(example["backward_flow"][t + 1],
                               meta["backward_flow_range"])
        flow    = _to_flow_tensor(bflow)
    else:
        # src=frame[t+1], trg=frame[t], flow=fflow[t]  (trg->src) ✓
        src_img = _to_image_tensor(video[t + 1])
        trg_img = _to_image_tensor(video[t])
        fflow   = _decode_flow(example["forward_flow"][t],
                               meta["forward_flow_range"])
        flow    = _to_flow_tensor(fflow)

    return {"src_img": src_img, "trg_img": trg_img, "flow": flow}


class MoviFSimpleDataset(Dataset):
    """
    Streams adjacent frame pairs from MOVi-F TFRecord shards.

    __getitem__ ignores idx and returns the next pair from a per-worker
    TF iterator.  __len__ is an estimate; cap actual usage with steps_per_epoch.
    """

    def __init__(
        self,
        datapath: str,
        split: str = "train",
        reverse_flow: bool = False,
        kubric_dir: str = _DEFAULT_KUBRIC_DIR,
        config: str = "512x512",
        shuffle_buffer: int = 16,
        **_,
    ):
        super().__init__()
        self.shard_dir    = Path(datapath)
        self.split        = split
        self.reverse_flow = reverse_flow
        self.shuffle_buffer = shuffle_buffer

        self.shards = _list_shards(self.shard_dir, split)
        if not self.shards:
            raise FileNotFoundError(
                f"No complete '{split}' shards found in {self.shard_dir}. "
                f"Expected files: movi_f-{split}.tfrecord-XXXXX-of-NNNNN"
            )

        self.builder = _load_movi_builder(kubric_dir, config=config)

        # Estimate: MOVi-F has ~6000 train videos across 1024 shards ≈ 5.86/shard
        self._estimated_len = len(self.shards) * 6 * _PAIRS_PER_VIDEO

        # Lazy per-worker iterators
        self._iter: Optional[object] = None
        self._iter_worker_id: Optional[int] = None

    def __len__(self) -> int:
        return self._estimated_len

    def _build_iterator(self, my_shards):
        """Build a TF pipeline over a subset of shards and yield pairs."""
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")

        features = self.builder.info.features

        def _project(example):
            # Drop heavy unused fields (segmentations, depth, normal,
            # object_coordinates, instances, camera, events) before the
            # shuffle buffer so we only hold ~42 MB/video instead of ~100 MB.
            return {
                "video": example["video"],
                "forward_flow": example["forward_flow"],
                "backward_flow": example["backward_flow"],
                "metadata": {
                    "video_name": example["metadata"]["video_name"],
                    "forward_flow_range": example["metadata"]["forward_flow_range"],
                    "backward_flow_range": example["metadata"]["backward_flow_range"],
                },
            }

        raw_ds = (
            tf.data.TFRecordDataset([str(s) for s in my_shards])
            .map(features.deserialize_example, num_parallel_calls=1)
            .map(_project, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(buffer_size=self.shuffle_buffer)
            .repeat()
        )
        for example in raw_ds.as_numpy_iterator():
            T = example["video"].shape[0]
            indices = list(range(T - 1))
            np.random.shuffle(indices)
            for t in indices:
                yield _make_pair(example, t, self.reverse_flow)

    def _get_iter(self):
        """Return (or lazily create) the per-worker iterator."""
        worker_info = torch.utils.data.get_worker_info()
        worker_id   = worker_info.id          if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        if self._iter is None or self._iter_worker_id != worker_id:
            my_shards = self.shards[worker_id::num_workers]
            if not my_shards:
                my_shards = self.shards  # fallback: more workers than shards
            self._iter_worker_id = worker_id
            self._iter = self._build_iterator(my_shards)

        return self._iter

    def __getitem__(self, _idx: int) -> Dict[str, torch.Tensor]:
        return next(self._get_iter())
