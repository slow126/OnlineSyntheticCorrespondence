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
        mirror_flip: float = 0.0,
        occlusion_mask: bool = False,
        **_,
    ):
        super().__init__()
        self.root = Path(datapath)
        self.split = split
        self.reverse_flow = reverse_flow
        # occlusion_mask: when True, mark occluded target-frame pixels INVALID (inf)
        # in the returned full-res flow, using a forward/backward round-trip check
        # (see _occlusion_mask_from_flows). The downstream occlusion-aware downsample
        # then turns mostly-occluded feature cells into inf so the sparse endpoint
        # loss skips them. Default False keeps the flow byte-identical to before.
        self.occlusion_mask = bool(occlusion_mask)
        # mirror_flip: fraction of pairs replaced by (frame0, horizontal-mirror(frame0))
        # with the corresponding mirror flow (W-1-2x, 0). This injects unnatural,
        # strongly OFF-TARGET motion at FIXED appearance (a mirrored frame has the
        # same texture distribution) and FIXED coverage (the unflipped pairs still
        # span the target's motion). It is the controlled precision/off-target knob
        # (mirror of the camera-dolly coverage ladder). Flipped pairs are the first
        # f-fraction of the (seed-shuffled) scene order -> deterministic & reproducible.
        self.mirror_flip = float(mirror_flip)

        split_root = self.root / split
        self.scene_root = split_root if split_root.exists() else self.root
        self.scene_dirs = self._find_scene_dirs(self.scene_root)
        if not self.scene_dirs:
            raise FileNotFoundError(
                f"No Kubric intervention scenelets found under {self.scene_root}"
            )

        # Drop scenes whose render is incomplete for the flow direction we read
        # (rare generation artifacts: a missing data_ranges flow-range key or flow
        # PNG). Filtering here means one bad scene can't KeyError mid-epoch.
        self.scene_dirs = [d for d in self.scene_dirs
                           if self._scene_has_flow(d, self.reverse_flow)]

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

    @staticmethod
    def _scene_has_flow(scene_dir: Path, reverse_flow: bool) -> bool:
        """True iff the scene has the flow PNG + data_ranges range for the
        direction this dataset reads (backward by default, forward if reversed)."""
        flow_name = "forward_flow" if reverse_flow else "backward_flow"
        png = ("forward_flow_00000.png" if reverse_flow
               else "backward_flow_00001.png")
        if not (scene_dir / png).exists():
            return False
        try:
            ranges = json.load((scene_dir / "data_ranges.json").open())
        except Exception:
            return False
        return flow_name in ranges

    @staticmethod
    def _occlusion_mask_from_flows(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
        """Forward/backward round-trip occlusion test.

        Args:
            primary:   [2, H, W] (dx, dy) flow this dataset returns (target->source),
                       i.e. for each target-frame pixel q its displacement to the
                       other frame.
            secondary: [2, H, W] (dx, dy) flow for the OPPOSITE direction.

        A target pixel q is occluded if, after warping q by primary to q' and
        sampling secondary at q' (nearest), the round-trip is inconsistent. The
        validated sign convention is |primary(q) - secondary(q')| ~ 0 for visible
        pixels. A pixel is occluded if q' leaves the frame OR the squared residual
        exceeds 0.01*(|primary|^2 + |secondary@q'|^2) + 0.5.

        Returns boolean [H, W], True == occluded.
        """
        _, H, W = primary.shape
        pdx, pdy = primary[0], primary[1]
        ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        qx = xs + pdx
        qy = ys + pdy
        ix = np.rint(qx).astype(np.int64)
        iy = np.rint(qy).astype(np.int64)
        out_of_frame = (ix < 0) | (ix >= W) | (iy < 0) | (iy >= H)
        ixc = np.clip(ix, 0, W - 1)
        iyc = np.clip(iy, 0, H - 1)
        sdx = secondary[0][iyc, ixc]
        sdy = secondary[1][iyc, ixc]
        err2 = (pdx - sdx) ** 2 + (pdy - sdy) ** 2
        mag_primary2 = pdx ** 2 + pdy ** 2
        mag_secondary2 = sdx ** 2 + sdy ** 2
        occ = out_of_frame | (err2 > 0.01 * (mag_primary2 + mag_secondary2) + 0.5)
        return occ

    def _compute_occlusion(self, scene_dir: Path, ranges: dict) -> Optional[np.ndarray]:
        """Decode the opposite-direction flow and compute the occlusion mask.

        Returns boolean [H, W] (True == occluded) or None if the secondary flow
        for this scene is unavailable (occlusion left disabled for that scene)."""
        if not self.reverse_flow:
            # primary = backward (frame1->0); secondary = forward (frame0->1)
            sec_name = "forward_flow"
            sec_path = scene_dir / "forward_flow_00000.png"
            prim_name = "backward_flow"
            prim_path = scene_dir / "backward_flow_00001.png"
        else:
            sec_name = "backward_flow"
            sec_path = scene_dir / "backward_flow_00001.png"
            prim_name = "forward_flow"
            prim_path = scene_dir / "forward_flow_00000.png"

        if not sec_path.exists() or sec_name not in ranges or prim_name not in ranges:
            return None

        prim_range = np.array([ranges[prim_name]["min"], ranges[prim_name]["max"]], dtype=np.float32)
        sec_range = np.array([ranges[sec_name]["min"], ranges[sec_name]["max"]], dtype=np.float32)
        primary = _to_flow_tensor(_decode_flow_png(prim_path, prim_range)).numpy()
        secondary = _to_flow_tensor(_decode_flow_png(sec_path, sec_range)).numpy()
        return self._occlusion_mask_from_flows(primary, secondary)

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

        flipped = bool(self.mirror_flip) and idx < int(self.mirror_flip * len(self.scene_dirs))
        if flipped:
            # off-target injection: frame1 := horizontal mirror of frame0, with the
            # exact mirror flow dx=(w-1-2x), dy=0 (consistent in both directions,
            # since a mirror is its own inverse). Appearance is preserved (mirrored
            # texture == same distribution); motion becomes strongly off-target.
            trg_img = src_img.flip(dims=(-1,)).contiguous()
            xs = torch.arange(w, dtype=torch.float32).view(1, w).expand(h, w)
            flow = torch.stack([(w - 1) - 2.0 * xs, torch.zeros(h, w)], dim=0).contiguous()
        elif self.occlusion_mask:
            # Mark occluded target-frame pixels invalid (inf) via fwd/bwd round-trip.
            occ = self._compute_occlusion(scene_dir, ranges)
            if occ is not None:
                flow = flow.clone()
                flow[:, torch.from_numpy(occ)] = float("inf")

        return {"src_img": src_img, "trg_img": trg_img, "flow": flow}
