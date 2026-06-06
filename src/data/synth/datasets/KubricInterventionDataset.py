"""Materialized Kubric intervention scenelet dataset.

Expected layout:
  <datapath>/<split>/scene_000000/
      rgba_00000.png
      rgba_00001.png
      backward_flow_00001.png
      forward_flow_00000.png
      data_ranges.json
      metadata.json

The default flow convention matches MOVi-F and the rest of the project:
src=frame 0, trg=frame 1, and flow is target->source, decoded from
Kubric backward_flow_00001. Kubric stores flow as (delta_row, delta_col); this
dataset returns project flow as [dx, dy].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_image(path: Path) -> torch.Tensor:
    arr = imageio.imread(path)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype(np.float32, copy=False) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _decode_flow_png(path: Path, flow_range: np.ndarray) -> np.ndarray:
    # Flow is stored as a 16-bit PNG. imageio/Pillow silently downcast 16-bit
    # multichannel PNGs to 8-bit, which collapses the decoded flow to ~constant.
    # cv2.IMREAD_UNCHANGED preserves the full 16-bit depth (channels are BGR).
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise FileNotFoundError(path)
    if encoded.ndim < 3 or encoded.shape[-1] < 2:
        raise ValueError(f"expected at least two flow channels in {path}")
    # cv2 returns BGR; take R,G to recover the first two stored flow channels
    # (matches the previous imageio RGB[..., :2] ordering).
    encoded = encoded[..., [2, 1]].astype(np.float32, copy=False)
    lo, hi = float(flow_range[0]), float(flow_range[1])
    if hi == lo:
        return np.zeros(encoded.shape, dtype=np.float32)
    return encoded / 65535.0 * (hi - lo) + lo


def _to_flow_tensor(flow_hw2: np.ndarray) -> torch.Tensor:
    """Kubric [H,W,2] (dy,dx) -> project [2,H,W] (dx,dy)."""
    return torch.from_numpy(flow_hw2.astype(np.float32, copy=False)).permute(2, 0, 1)[[1, 0]].contiguous()


class KubricInterventionDataset(Dataset):
    """Reads compact 2-frame Kubric intervention scenelets."""

    def __init__(
        self,
        datapath: str,
        split: str = "train",
        reverse_flow: bool = False,
        max_pairs: Optional[int] = None,
        seed: Optional[int] = None,
        **_,
    ):
        super().__init__()
        self.root = Path(datapath)
        self.split = split
        self.reverse_flow = reverse_flow

        split_root = self.root / split
        self.scene_root = split_root if split_root.exists() else self.root
        self.scene_dirs = self._find_scene_dirs(self.scene_root)
        if not self.scene_dirs:
            raise FileNotFoundError(
                f"No Kubric intervention scenelets found under {self.scene_root}"
            )

        if seed is not None:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(self.scene_dirs))
            self.scene_dirs = [self.scene_dirs[i] for i in order]
        if max_pairs is not None:
            self.scene_dirs = self.scene_dirs[: int(max_pairs)]

    def __len__(self) -> int:
        return len(self.scene_dirs)

    @staticmethod
    def _is_scene_dir(p: Path) -> bool:
        # rgba is optional: geometry-only ("search") renders skip it and carry
        # only flow + data_ranges, which is all flow-vector extraction needs.
        return (
            (p / "data_ranges.json").exists()
            and (
                (p / "backward_flow_00001.png").exists()
                or (p / "forward_flow_00000.png").exists()
            )
        )

    @classmethod
    def _find_scene_dirs(cls, root: Path) -> list[Path]:
        if not root.exists():
            return []
        if cls._is_scene_dir(root):
            return [root]
        return sorted(p for p in root.iterdir() if p.is_dir() and cls._is_scene_dir(p))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        scene_dir = self.scene_dirs[idx]
        with (scene_dir / "data_ranges.json").open() as f:
            ranges = json.load(f)

        if not self.reverse_flow:
            src_name, trg_name = "rgba_00000.png", "rgba_00001.png"
            flow_name = "backward_flow"
            flow_path = scene_dir / "backward_flow_00001.png"
        else:
            src_name, trg_name = "rgba_00001.png", "rgba_00000.png"
            flow_name = "forward_flow"
            flow_path = scene_dir / "forward_flow_00000.png"

        if not flow_path.exists():
            raise FileNotFoundError(f"missing required flow file: {flow_path}")
        if flow_name not in ranges:
            raise KeyError(f"missing {flow_name!r} range in {scene_dir / 'data_ranges.json'}")

        flow_range = np.array([ranges[flow_name]["min"], ranges[flow_name]["max"]], dtype=np.float32)
        flow = _to_flow_tensor(_decode_flow_png(flow_path, flow_range))

        # rgba is optional for flow-only (geometry-only) renders. When absent,
        # return zero images shaped like the flow so collation/extraction still
        # work -- the extractor only consumes "flow". Training uses full renders
        # (rgba present) and is unaffected.
        h, w = flow.shape[-2], flow.shape[-1]
        src_path, trg_path = scene_dir / src_name, scene_dir / trg_name
        src_img = _read_image(src_path) if src_path.exists() else torch.zeros(3, h, w)
        trg_img = _read_image(trg_path) if trg_path.exists() else torch.zeros(3, h, w)

        return {"src_img": src_img, "trg_img": trg_img, "flow": flow}
