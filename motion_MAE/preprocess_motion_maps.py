#!/usr/bin/env python3
"""
Preprocess motion maps for motion-only MAE experiments.

Outputs per-sample tensors:
  - dx float16 (H x W), normalized by orig_W
  - dy float16 (H x W), normalized by orig_H
  - mask uint8 (H x W)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.calculate_coverage_faiss import create_dataset_from_config
from torch.utils.data import DataLoader

from src.data.synth.adapters import SyntheticAdapter
from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset


def _ensure_flow_hw(flow: torch.Tensor) -> torch.Tensor:
    if flow.dim() != 3:
        raise ValueError(f"Expected flow [2,H,W], got shape {list(flow.shape)}")
    if flow.shape[0] == 2:
        return flow
    if flow.shape[-1] == 2:
        return flow.permute(2, 0, 1)
    raise ValueError(f"Could not interpret flow shape {list(flow.shape)}")


def _disk_offsets(radius: int) -> np.ndarray:
    offsets = []
    r2 = radius * radius
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))
    return np.array(offsets, dtype=np.int64)


def _rasterize_keypoints(
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    img_size: Tuple[int, int],
    splat_radius: int,
    max_motion_px: Tuple[int, int],
    n_pts: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = img_size
    dx_map = np.zeros((height, width), dtype=np.float32)
    dy_map = np.zeros((height, width), dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)

    if src_kps is None or trg_kps is None:
        return dx_map, dy_map, mask

    if src_kps.dim() != 2 or trg_kps.dim() != 2:
        return dx_map, dy_map, mask

    n_pts_total = min(src_kps.shape[1], trg_kps.shape[1])
    n_pts = n_pts_total if n_pts is None else min(int(n_pts), n_pts_total)
    if n_pts == 0:
        return dx_map, dy_map, mask

    src = src_kps[:, :n_pts].detach().cpu().numpy().astype(np.float32)
    trg = trg_kps[:, :n_pts].detach().cpu().numpy().astype(np.float32)

    offsets = _disk_offsets(splat_radius) if splat_radius > 0 else None

    for i in range(n_pts):
        x = int(round(trg[0, i]))
        y = int(round(trg[1, i]))
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        dx = src[0, i] - trg[0, i]
        dy = src[1, i] - trg[1, i]

        if not np.isfinite(dx) or not np.isfinite(dy):
            continue
        if dx == 0.0 and dy == 0.0:
            continue
        if abs(dx) > max_motion_px[1] or abs(dy) > max_motion_px[0]:
            continue

        if offsets is None:
            dx_map[y, x] = dx
            dy_map[y, x] = dy
            mask[y, x] = 1
            continue

        ys = y + offsets[:, 0]
        xs = x + offsets[:, 1]
        valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        xs = xs[valid]
        ys = ys[valid]
        dx_map[ys, xs] = dx
        dy_map[ys, xs] = dy
        mask[ys, xs] = 1

    return dx_map, dy_map, mask


def _process_dense_flow(
    flow: torch.Tensor,
    img_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flow = _ensure_flow_hw(flow)
    flow_np = flow.detach().cpu().float().numpy()
    dx_raw = flow_np[0]
    dy_raw = flow_np[1]

    height, width = img_size
    finite = np.isfinite(dx_raw) & np.isfinite(dy_raw)
    non_zero = ~((dx_raw == 0.0) & (dy_raw == 0.0))
    within = (np.abs(dx_raw) <= width) & (np.abs(dy_raw) <= height)
    mask = (finite & non_zero & within).astype(np.uint8)

    dx_raw = np.clip(dx_raw, -width, width)
    dy_raw = np.clip(dy_raw, -height, height)
    invalid = mask == 0
    if np.any(invalid):
        dx_raw[invalid] = 0.0
        dy_raw[invalid] = 0.0

    return dx_raw, dy_raw, mask


def _resize_and_normalize(
    dx_raw: np.ndarray,
    dy_raw: np.ndarray,
    mask: np.ndarray,
    img_size: Tuple[int, int],
    train_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    height, width = img_size
    dx_norm = dx_raw / float(width)
    dy_norm = dy_raw / float(height)

    flow = torch.from_numpy(np.stack([dx_norm, dy_norm], axis=0)).unsqueeze(0)
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    flow = F.interpolate(flow, size=(train_size, train_size), mode="bilinear", align_corners=False)
    mask_t = F.interpolate(mask_t, size=(train_size, train_size), mode="nearest")
    mask_resized = mask_t.squeeze(0).squeeze(0) > 0.5

    flow = flow.squeeze(0)
    flow[:, ~mask_resized] = 0.0

    dx = flow[0].to(dtype=torch.float16)
    dy = flow[1].to(dtype=torch.float16)
    mask_out = mask_resized.to(dtype=torch.uint8)

    return dx, dy, mask_out


def _save_viz(
    dx: torch.Tensor,
    dy: torch.Tensor,
    mask: torch.Tensor,
    out_path: Path,
) -> None:
    dx_np = dx.float().cpu().numpy()
    dy_np = dy.float().cpu().numpy()
    mask_np = mask.float().cpu().numpy()

    rgb = np.zeros((dx_np.shape[0], dx_np.shape[1], 3), dtype=np.float32)
    rgb[..., 0] = np.clip(dx_np * 0.5 + 0.5, 0.0, 1.0)
    rgb[..., 1] = np.clip(dy_np * 0.5 + 0.5, 0.0, 1.0)
    rgb[..., 2] = np.clip(mask_np, 0.0, 1.0)

    img = Image.fromarray((rgb * 255.0).astype(np.uint8))
    img.save(out_path)


def _build_dataset(config: dict, entry: dict) -> CorrespondenceDataset:
    return create_dataset_from_config(
        entry["name"],
        entry.get("split", "train"),
        config.get("dataset_params", {}),
        config.get("dataset_overrides", {}),
        entry_overrides=entry.get("overrides"),
    )


def preprocess_dataset(
    config: dict,
    entry: dict,
    output_root: Path,
    train_size: int,
    max_samples: int,
    max_attempts: Optional[int],
    splat_radius: int,
    save_viz: bool,
    viz_every: int,
    viz_max: int,
    batch_size: int,
    num_workers: int,
) -> None:
    dataset_name = entry["name"]
    split = entry.get("split", "train")
    dataset = _build_dataset(config, entry)
    is_synthetic = isinstance(dataset.adapter, SyntheticAdapter)
    if is_synthetic and num_workers > 0:
        num_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not is_synthetic,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )

    output_dir = output_root / dataset_name / split
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = output_dir / "viz"
    if save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)

    index_entries: List[Dict[str, object]] = []
    saved = 0
    attempts = 0
    stop = False

    for batch in dataloader:
        if stop:
            break

        flow = batch.get("flow_full", batch.get("flow"))
        src_kps = batch.get("src_kps")
        trg_kps = batch.get("trg_kps")
        n_pts = batch.get("n_pts")

        if flow is None and (src_kps is None or trg_kps is None):
            continue

        if flow is not None:
            if flow.dim() == 4:
                height, width = int(flow.shape[2]), int(flow.shape[3])
                batch_size_actual = flow.shape[0]
            else:
                flow_hw = _ensure_flow_hw(flow)
                height, width = int(flow_hw.shape[1]), int(flow_hw.shape[2])
                batch_size_actual = 1
        else:
            height = width = None
            batch_size_actual = src_kps.shape[0]

        for b in range(batch_size_actual):
            attempts += 1
            if max_attempts is not None and attempts > max_attempts:
                stop = True
                break
            if saved >= max_samples:
                stop = True
                break

            if dataset_name == "spair":
                if height is None or width is None:
                    continue
                dx_raw, dy_raw, mask = _rasterize_keypoints(
                    src_kps[b],
                    trg_kps[b],
                    (height, width),
                    splat_radius,
                    (height, width),
                    n_pts=int(n_pts[b].item()) if n_pts is not None else None,
                )
            else:
                if flow is None:
                    continue
                sample_flow = flow[b] if flow.dim() == 4 else flow
                flow_hw = _ensure_flow_hw(sample_flow)
                height, width = int(flow_hw.shape[1]), int(flow_hw.shape[2])
                dx_raw, dy_raw, mask = _process_dense_flow(sample_flow, (height, width))

            dx, dy, mask_out = _resize_and_normalize(
                dx_raw,
                dy_raw,
                mask,
                (height, width),
                train_size,
            )

            n_valid = int(mask_out.sum().item())
            if n_valid == 0:
                continue

            max_mag = 0.0
            if n_valid > 0:
                flow_mag = torch.sqrt(dx.float() ** 2 + dy.float() ** 2)
                max_mag = float(flow_mag[mask_out.bool()].max().item())

            sample_id = f"{saved:06d}"
            output_path = output_dir / f"{sample_id}.pt"

            torch.save(
                {
                    "dx": dx,
                    "dy": dy,
                    "mask": mask_out,
                    "orig_h": int(height),
                    "orig_w": int(width),
                    "n_valid": n_valid,
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                },
                output_path,
            )

            if save_viz and saved < viz_max and (saved % max(1, viz_every) == 0):
                viz_path = viz_dir / f"{sample_id}.png"
                _save_viz(dx, dy, mask_out, viz_path)

            index_entries.append(
                {
                    "file": output_path.name,
                    "n_valid": n_valid,
                    "orig_h": int(height),
                    "orig_w": int(width),
                    "dataset": dataset_name,
                    "sample_id": sample_id,
                    "max_mag": max_mag,
                }
            )

            saved += 1
            if saved % 250 == 0:
                print(f"[{dataset_name}/{split}] saved {saved} samples")

    index_path = output_dir / "index.jsonl"
    with open(index_path, "w") as f:
        for entry in index_entries:
            f.write(json.dumps(entry) + "\n")

    print(f"[{dataset_name}/{split}] done: {saved} samples -> {output_dir}")


def _filter_entries(entries: List[dict], allowed: Optional[List[str]]) -> List[dict]:
    if not allowed:
        return entries
    allowed_set = {name.strip() for name in allowed}
    return [entry for entry in entries if entry.get("name") in allowed_set]


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess motion maps for motion MAE")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output-root", type=str, default="motion_MAE/preproc", help="Output directory")
    parser.add_argument("--train-size", type=int, default=256, help="Training resolution (square)")
    parser.add_argument("--max-samples", type=int, default=10000, help="Max valid samples per dataset")
    parser.add_argument("--max-attempts", type=int, default=20000, help="Max attempted samples per dataset")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    parser.add_argument("--splat-radius", type=int, default=3, help="SPair keypoint splat radius")
    parser.add_argument("--save-viz", action="store_true", help="Save visualization PNGs")
    parser.add_argument("--viz-every", type=int, default=50, help="Save every Nth sample visualization")
    parser.add_argument("--viz-max", type=int, default=200, help="Max visualizations per dataset")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--num-workers", type=int, default=None, help="Override num_workers from config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    entries = config.get("datasets", [])
    if not entries:
        raise ValueError("No datasets found in config")

    allowed = [name for name in (args.datasets.split(",") if args.datasets else []) if name]
    entries = _filter_entries(entries, allowed)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    batch_size = int(args.batch_size) if args.batch_size is not None else int(config.get("batch_size", 1))
    num_workers = int(args.num_workers) if args.num_workers is not None else int(config.get("num_workers", 0))

    for entry in entries:
        preprocess_dataset(
            config,
            entry,
            output_root,
            args.train_size,
            args.max_samples,
            args.max_attempts,
            args.splat_radius,
            args.save_viz,
            args.viz_every,
            args.viz_max,
            batch_size,
            num_workers,
        )


if __name__ == "__main__":
    main()
