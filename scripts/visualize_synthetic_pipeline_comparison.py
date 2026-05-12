#!/usr/bin/env python3
"""
Render supplementary-friendly image panels comparing the base synthetic 3D
pipeline against the synthetic 2D-warp ablation.

For exact visual comparison, the 2D target is derived from the *same processed
3D sample* by warping the source image with the recovered dense flow. This
avoids nondeterminism in the renderer causing source-view mismatches across two
separately instantiated dataset objects.

The script intentionally produces image-only panels:
- no quiver arrows
- no per-image min/max normalization
- optional ImageNet denormalization for viewable RGB output
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _load_dataset_from_config(config_path: str, extra_dataset_overrides=None):
    from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset

    cfg = yaml.safe_load(Path(config_path).read_text())
    dataset_cfg = dict(cfg.get("dataset", cfg))
    dataset_name = dataset_cfg.pop("dataset_name", None) or dataset_cfg.pop("name", None)
    if not dataset_name:
        raise ValueError(f"Could not find dataset_name in {config_path}")
    if extra_dataset_overrides:
        dataset_cfg.update(extra_dataset_overrides)
    return CorrespondenceDataset(dataset_name, **dataset_cfg)


def _to_display_image(img: torch.Tensor, denormalize: str) -> torch.Tensor:
    img = img.detach().cpu().float()
    if img.dim() == 4 and img.shape[0] == 1:
        img = img.squeeze(0)
    if img.dim() != 3:
        raise ValueError(f"Expected image with shape [C,H,W], got {tuple(img.shape)}")

    if denormalize == "imagenet":
        mean = IMAGENET_MEAN.to(img)
        std = IMAGENET_STD.to(img)
        img = img * std + mean

    return img.clamp(0.0, 1.0).permute(1, 2, 0)


def _build_default_overrides(disable_random_swap: bool, disable_warp_swap: bool):
    overrides = {}
    if disable_random_swap:
        overrides["geometry_config_overrides"] = {"random_swap": False}
    if disable_warp_swap:
        overrides["synthetic_flow_warp_swap"] = False
    return overrides


def main():
    parser = argparse.ArgumentParser(description="Visualize synthetic 3D vs synthetic 2D-warp samples")
    parser.add_argument(
        "--synthetic-config",
        default="src/configs/lightning/datasets/synthetic_rc.yaml",
        help="Dataset config for base synthetic pipeline",
    )
    parser.add_argument(
        "--synthetic-2d-config",
        default="src/configs/lightning/datasets/synthetic_2d_warp_rc.yaml",
        help="Unused legacy argument retained for CLI compatibility",
    )
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0, 1, 2], help="Sample indices to visualize")
    parser.add_argument("--output-path", default="figures/supplementary/synthetic_pipeline_comparison.png", help="Output image path")
    parser.add_argument(
        "--denormalize",
        choices=["none", "imagenet"],
        default="imagenet",
        help="How to convert tensors back to display RGB",
    )
    parser.add_argument(
        "--allow-random-swap",
        action="store_true",
        default=False,
        help="Keep the underlying 3D pipeline's random source/target swap enabled",
    )
    parser.add_argument(
        "--allow-warp-swap",
        action="store_true",
        default=False,
        help="Keep the 2D-warp ablation's random swap enabled",
    )
    args = parser.parse_args()

    synthetic_overrides = _build_default_overrides(
        disable_random_swap=not args.allow_random_swap,
        disable_warp_swap=False,
    )
    synthetic_ds = _load_dataset_from_config(args.synthetic_config, synthetic_overrides)

    nrows = len(args.sample_indices)
    fig, axes = plt.subplots(nrows, 3, figsize=(10.8, 3.5 * nrows), dpi=180)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    column_titles = [
        r"SDF-Fractal $\mathbf{3D/2D}$ Source",
        r"SDF-Fractal $\mathbf{3D}$ Target",
        r"SDF-Fractal $\mathbf{2D}$ Target",
    ]

    for row, sample_idx in enumerate(args.sample_indices):
        base_sample = synthetic_ds.collate_fn([synthetic_ds[sample_idx]])
        flow = base_sample.get("flow_full", base_sample.get("flow"))
        if flow is None:
            raise ValueError("Base synthetic sample is missing flow data")
        warp_target = synthetic_ds._warp_src_with_flow(base_sample["src_img"], flow)

        images = [
            _to_display_image(base_sample["src_img"][0], args.denormalize),
            _to_display_image(base_sample["trg_img"][0], args.denormalize),
            _to_display_image(warp_target[0], args.denormalize),
        ]

        for col, img in enumerate(images):
            ax = axes[row, col]
            ax.imshow(img.numpy())
            ax.axis("off")
            if row == 0:
                ax.set_title(column_titles[col])
            if col == 0:
                ax.set_ylabel(f"sample {sample_idx}", rotation=90, fontsize=10)

    plt.tight_layout()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved comparison figure to {output_path}")


if __name__ == "__main__":
    main()
