#!/usr/bin/env python3
"""
Quick visualization utility for dataset configs (single or mixed).

Loads a dataset config YAML with a top-level "dataset" block, fetches one batch,
and saves flow overlay/side-by-side visuals for sanity checking warps.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.datasets.MixedCorrespondenceDataset import MixedCorrespondenceDataset
from src.data.synth.datasets.visualizers import CorrespondenceVisualizer


def _coerce_size(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return value


def _build_dataset(dataset_cfg):
    cfg = dict(dataset_cfg)
    is_mixed = cfg.get("mixed", False) or "datasets" in cfg
    if is_mixed:
        datasets_list = cfg.pop("datasets", [])
        percentages = cfg.pop("percentages", [])
        dataset_overrides = cfg.pop("dataset_overrides", {})
        epoch_size = cfg.pop("epoch_size", None)
        seed = cfg.pop("seed", None)

        if len(datasets_list) != len(percentages):
            raise ValueError(
                f"Number of datasets ({len(datasets_list)}) must match percentages ({len(percentages)})"
            )

        cfg["size"] = _coerce_size(cfg.get("size"))
        if cfg.get("max_kps", "unset") is None:
            cfg["max_kps"] = None

        created = []
        for name in datasets_list:
            ds_cfg = dict(cfg)
            if name in dataset_overrides:
                ds_cfg.update(dataset_overrides[name])
            ds_cfg["size"] = _coerce_size(ds_cfg.get("size"))
            if ds_cfg.get("max_kps", "unset") is None:
                ds_cfg["max_kps"] = None
            created.append(CorrespondenceDataset(name, **ds_cfg))

        return MixedCorrespondenceDataset(
            datasets=created,
            percentages=percentages,
            epoch_size=epoch_size,
            seed=seed,
        )

    dataset_name = cfg.pop("dataset_name", None) or cfg.pop("name", None)
    if not dataset_name:
        raise ValueError("dataset_name is required for non-mixed dataset configs")
    cfg["size"] = _coerce_size(cfg.get("size"))
    if cfg.get("max_kps", "unset") is None:
        cfg["max_kps"] = None
    cfg.pop("mixed", None)
    cfg.pop("datasets", None)
    cfg.pop("percentages", None)
    cfg.pop("dataset_overrides", None)
    return CorrespondenceDataset(dataset_name, **cfg)


def main():
    parser = argparse.ArgumentParser(description="Visualize dataset warps from a dataset config")
    parser.add_argument("--dataset-config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--output-dir", default="debug_warp_viz", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for visualization")
    parser.add_argument("--max-samples", type=int, default=1, help="Max samples to visualize")
    parser.add_argument(
        "--mode",
        choices=["overlay", "side_by_side"],
        default="overlay",
        help="Visualization mode",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=["regular", "all_valid"],
        default="regular",
        help="Flow sampling mode for arrows",
    )
    parser.add_argument(
        "--arrow-density",
        type=int,
        default=12,
        help="Arrow density for regular sampling",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.dataset_config).read_text())
    dataset_cfg = cfg.get("dataset", cfg)
    dataset = _build_dataset(dataset_cfg)

    loader = DataLoader(
        dataset,
        batch_size=max(1, args.batch_size),
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(loader))
    flow = batch.get("flow_full", batch.get("flow"))
    if flow is None:
        raise ValueError("Batch missing flow data (flow or flow_full)")

    visualizer = CorrespondenceVisualizer(
        sampling_mode=args.sampling_mode,
        arrow_density=args.arrow_density,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "warp_overlay.png"
    visualizer.visualize_rendered_batch(
        {
            "src_img": batch["src_img"].cpu(),
            "trg_img": batch["trg_img"].cpu(),
            "flow": flow.cpu(),
        },
        save_path=str(save_path),
        max_samples=args.max_samples,
        visualization_mode=args.mode,
        sampling_mode=args.sampling_mode,
    )
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
