#!/usr/bin/env python3
"""
Dataset-level variogram analysis (flow-only).

Computes a variogram curve per dataset (train/eval) using per-image flow vectors,
then reports pairwise curve distances for all train→eval combinations, similar to
coverage configs.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.distance import pdist
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Dataset utilities (reuse existing config-driven loaders)
from scripts.calculate_coverage_faiss import (
    create_dataset_from_config,
    create_mixed_dataset_from_config,
    _is_synthetic_dataset,
)

# Flow extraction (reuse filtering logic)
from src.coreset.validation import extract_flow_vectors_from_batch


def _get_flow_tensor(batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
    if "flow_full" in batch:
        return batch["flow_full"]
    if "flow" in batch:
        return batch["flow"]
    return None


def _scale_flow_vectors(
    vectors: np.ndarray,
    src_hw: Tuple[int, int],
    target_hw: Tuple[int, int],
) -> np.ndarray:
    src_h, src_w = src_hw
    tgt_h, tgt_w = target_hw
    if src_h <= 0 or src_w <= 0:
        return vectors
    if src_h == tgt_h and src_w == tgt_w:
        return vectors
    scale_x = float(tgt_w) / float(src_w)
    scale_y = float(tgt_h) / float(src_h)
    scaled = vectors.copy()
    scaled[:, 0] *= scale_x
    scaled[:, 2] *= scale_x
    scaled[:, 1] *= scale_y
    scaled[:, 3] *= scale_y
    return scaled


class VariogramAccumulator:
    def __init__(
        self,
        image_size: int = 512,
        n_bins: int = 50,
        max_dense_samples: int = 2000,
        min_pairs_per_bin: int = 50,
    ) -> None:
        self.image_size = int(image_size)
        self.bins = np.linspace(0.0, float(self.image_size), n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.max_dense_samples = int(max_dense_samples)
        self.min_pairs_per_bin = int(min_pairs_per_bin)
        self.bin_sums = np.zeros(len(self.bin_centers), dtype=np.float64)
        self.bin_sumsq = np.zeros(len(self.bin_centers), dtype=np.float64)
        self.bin_counts = np.zeros(len(self.bin_centers), dtype=np.int64)

    def add_vectors(self, vectors: np.ndarray, mean_mag: Optional[float] = None) -> int:
        if vectors is None or len(vectors) < 2:
            return 0

        pts = vectors[:, :2]
        flow = vectors[:, 2:]

        if mean_mag is None:
            mean_mag = float(np.mean(np.linalg.norm(flow, axis=1)))
        if mean_mag < 1e-6:
            return 0

        if len(vectors) > self.max_dense_samples:
            indices = np.random.choice(len(vectors), self.max_dense_samples, replace=False)
            pts = pts[indices]
            flow = flow[indices]

        d_spatial = pdist(pts)
        d_flow = pdist(flow) / mean_mag

        bin_idx = np.searchsorted(self.bins, d_spatial, side="right") - 1
        valid = (bin_idx >= 0) & (bin_idx < len(self.bin_centers))
        if not np.any(valid):
            return 0

        counts = np.bincount(bin_idx[valid], minlength=len(self.bin_centers))
        vals = d_flow[valid]
        sums = np.bincount(bin_idx[valid], weights=vals, minlength=len(self.bin_centers))
        sums_sq = np.bincount(bin_idx[valid], weights=vals * vals, minlength=len(self.bin_centers))
        self.bin_counts += counts
        self.bin_sums += sums
        self.bin_sumsq += sums_sq
        return int(np.sum(valid))

    def compute_stats(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.bin_counts.sum() == 0:
            return None, None, None, None
        mean = np.full(len(self.bin_centers), np.nan, dtype=np.float64)
        sem = np.full(len(self.bin_centers), np.nan, dtype=np.float64)
        valid = self.bin_counts >= self.min_pairs_per_bin
        counts = self.bin_counts.astype(np.float64)
        mean[valid] = self.bin_sums[valid] / counts[valid]
        var = np.maximum(self.bin_sumsq[valid] / counts[valid] - mean[valid] ** 2, 0.0)
        sem[valid] = np.sqrt(var) / np.sqrt(counts[valid])
        return self.bin_centers, mean, sem, counts


def _build_dataloader(ds_config: Dict, config: Dict) -> Tuple[object, DataLoader]:
    dataset_name = ds_config.get("name")
    split = ds_config.get("split")
    if ds_config.get("mixed", False):
        dataset = create_mixed_dataset_from_config(
            ds_config["datasets"],
            ds_config["percentages"],
            split,
            config["dataset_params"],
            config["dataset_overrides"],
            seed=config["sampling"]["seed"],
        )
        is_synthetic = any(_is_synthetic_dataset(name) for name in ds_config["datasets"])
    else:
        dataset = create_dataset_from_config(
            dataset_name,
            split,
            config["dataset_params"],
            config["dataset_overrides"],
            entry_overrides=ds_config.get("overrides"),
        )
        is_synthetic = _is_synthetic_dataset(dataset_name)

    num_workers = 0 if is_synthetic else config["num_workers"]
    pin_memory = False if is_synthetic else True
    if is_synthetic and config["num_workers"] > 0:
        print("  WARNING: Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["sampling"].get("shuffle", True),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
    )
    return dataset, dataloader


def _parse_max_batches_overrides(overrides: Optional[List[str]]) -> Dict[str, int]:
    parsed: Dict[str, int] = {}
    if not overrides:
        return parsed
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --max-batches-overrides entry: {item!r}. Expected name=INT.")
        key, value = item.split("=", 1)
        key = key.strip().lower()
        if not key:
            raise ValueError(f"Invalid --max-batches-overrides entry: {item!r}. Empty name.")
        try:
            parsed[key] = int(value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --max-batches-overrides entry: {item!r}. Expected integer value."
            ) from exc
    return parsed


def _resolve_max_batches(
    dataset_name: Optional[str],
    split: Optional[str],
    default_max: int,
    overrides: Dict[str, int],
) -> int:
    if not dataset_name:
        return default_max
    name_key = dataset_name.lower()
    split_key = f"{name_key}:{str(split).lower()}" if split else None
    if split_key and split_key in overrides:
        return overrides[split_key]
    if name_key in overrides:
        return overrides[name_key]
    return default_max


def _compute_auc(centers: np.ndarray, mean: np.ndarray) -> Tuple[float, float]:
    valid = np.isfinite(mean)
    if centers is None or mean is None or np.sum(valid) < 2:
        return float("nan"), float("nan")
    auc = float(np.trapz(mean[valid], centers[valid]))
    span = float(centers[valid][-1] - centers[valid][0])
    auc_norm = auc / span if span > 0 else float("nan")
    return auc, auc_norm


def _compute_curve_distances(
    centers: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
) -> Dict[str, float]:
    if centers is None or mean_a is None or mean_b is None:
        return {"curve_l1": float("nan"), "curve_l2": float("nan"), "curve_corr": float("nan")}
    valid = np.isfinite(mean_a) & np.isfinite(mean_b)
    if np.sum(valid) < 2:
        return {"curve_l1": float("nan"), "curve_l2": float("nan"), "curve_corr": float("nan")}
    diff = mean_a[valid] - mean_b[valid]
    curve_l1 = float(np.mean(np.abs(diff)))
    curve_l2 = float(np.sqrt(np.mean(diff * diff)))
    corr = float(np.corrcoef(mean_a[valid], mean_b[valid])[0, 1])
    return {"curve_l1": curve_l1, "curve_l2": curve_l2, "curve_corr": corr}


def _default_output_paths(config: Dict, config_path: str) -> Tuple[Path, Path]:
    output_cfg = config.get("output", {})
    results_file = output_cfg.get("results_file")
    if results_file:
        results_path = Path(results_file)
        name = results_path.name
        stem = name.replace("coverage_v2_", "variogram_").replace("coverage_", "variogram_")
        if stem == name:
            stem = f"variogram_{results_path.stem}.csv"
        results_path = results_path.with_name(stem)
    else:
        base = Path(config_path).stem
        results_path = Path("analysis") / f"variogram_{base}.csv"

    curves_path = results_path.with_name(results_path.stem.replace("variogram_", "variogram_curves_") + ".csv")
    return results_path, curves_path


def run_variogram_dataset(
    config_path: str,
    max_batches: Optional[int],
    output_results: Optional[str],
    output_curves: Optional[str],
    max_batches_overrides: Optional[List[str]],
    n_bins: int,
    min_pairs_per_bin: int,
    vectors_per_image: Optional[int],
) -> None:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config.get("representation") != "flow":
        raise ValueError("Variogram dataset analysis only supports representation: flow")

    if max_batches is None:
        max_batches = int(config.get("max_batches", 10))
    overrides = _parse_max_batches_overrides(max_batches_overrides)

    if vectors_per_image is None:
        vectors_per_image = config["sampling"].get(
            "vectors_per_image", config["sampling"].get("flow_per_image_max", 2000)
        )

    seed = int(config["sampling"].get("seed", 42))
    np.random.seed(seed)

    image_size = config.get("flow_normalization", {}).get("image_size", [512, 512])
    target_hw = (int(image_size[0]), int(image_size[1]))
    spatial_max = max(target_hw)
    if target_hw[0] != target_hw[1]:
        print(
            f"  WARNING: Non-square target size {target_hw}, using max({target_hw}) for bin range."
        )
    max_flow = float(np.hypot(target_hw[0], target_hw[1]))

    results_path, curves_path = _default_output_paths(config, config_path)
    if output_results:
        results_path = Path(output_results)
    if output_curves:
        curves_path = Path(output_curves)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    curves_path.parent.mkdir(parents=True, exist_ok=True)

    dataset_curves = {}
    dataset_stats = {}
    train_keys = []
    eval_keys = []

    for ds_config in config["datasets"]:
        dataset_name = ds_config.get("name")
        split = ds_config.get("split")
        is_eval = bool(ds_config.get("is_eval", False))
        label = f"{dataset_name}_{split}"

        print(f"\n[{label}] {'(eval)' if is_eval else '(train)'}")
        dataset, dataloader = _build_dataloader(ds_config, config)

        dataset_max_batches = _resolve_max_batches(
            dataset_name=dataset_name,
            split=split,
            default_max=max_batches,
            overrides=overrides,
        )
        print(f"  max_batches={dataset_max_batches}")

        accumulator = VariogramAccumulator(
            image_size=spatial_max,
            n_bins=n_bins,
            max_dense_samples=vectors_per_image,
            min_pairs_per_bin=min_pairs_per_bin,
        )

        batches_seen = 0
        images_seen = 0
        images_used = 0
        total_vectors = 0
        total_pairs = 0

        for batch in tqdm(dataloader, desc="  Processing", leave=False):
            if batches_seen >= dataset_max_batches:
                break

            flow_tensor = _get_flow_tensor(batch)
            if flow_tensor is None:
                batches_seen += 1
                continue

            if flow_tensor.dim() == 3:
                flow_tensor = flow_tensor.unsqueeze(0)
            _, _, src_h, src_w = flow_tensor.shape

            per_image = extract_flow_vectors_from_batch(
                batch,
                return_per_image=True,
                max_flow_magnitude=max_flow,
            )
            if per_image is None:
                batches_seen += 1
                continue

            for img_vectors in per_image:
                images_seen += 1
                if img_vectors is None or len(img_vectors) == 0:
                    continue
                scaled = _scale_flow_vectors(
                    img_vectors,
                    src_hw=(int(src_h), int(src_w)),
                    target_hw=target_hw,
                )
                pairs_added = accumulator.add_vectors(scaled)
                if pairs_added > 0:
                    images_used += 1
                    total_vectors += int(len(scaled))
                    total_pairs += int(pairs_added)

            batches_seen += 1

        centers, mean, sem, counts = accumulator.compute_stats()
        if centers is not None and mean is not None:
            dataset_curves[label] = (centers, mean, sem, counts)
            dataset_stats[label] = {
                "dataset": dataset_name,
                "split": split,
                "is_eval": is_eval,
                "images_seen": images_seen,
                "images_used": images_used,
                "total_vectors": total_vectors,
                "total_pairs": total_pairs,
            }
            if is_eval:
                eval_keys.append(label)
            else:
                train_keys.append(label)
        else:
            print("  WARNING: No variogram curve computed (insufficient pairs).")

        del dataset, dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not dataset_curves:
        print("No variogram results computed.")
        return

    # Save curves (long format)
    curve_rows = []
    for label, (centers, mean, sem, counts) in dataset_curves.items():
        meta = dataset_stats.get(label, {})
        for i in range(len(centers)):
            curve_rows.append(
                {
                    "dataset": meta.get("dataset"),
                    "split": meta.get("split"),
                    "is_eval": meta.get("is_eval"),
                    "label": label,
                    "bin_center": float(centers[i]),
                    "variogram_mean": float(mean[i]) if np.isfinite(mean[i]) else np.nan,
                    "variogram_sem": float(sem[i]) if sem is not None and np.isfinite(sem[i]) else np.nan,
                    "bin_count": int(counts[i]) if counts is not None else 0,
                }
            )
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
    print(f"✓ Curves saved to: {curves_path}")

    # Pairwise train→eval distances
    results = []
    for train_label in sorted(train_keys):
        train_centers, train_mean, _, _ = dataset_curves[train_label]
        train_auc, train_auc_norm = _compute_auc(train_centers, train_mean)
        train_meta = dataset_stats.get(train_label, {})

        for eval_label in sorted(eval_keys):
            eval_centers, eval_mean, _, _ = dataset_curves[eval_label]
            eval_auc, eval_auc_norm = _compute_auc(eval_centers, eval_mean)
            eval_meta = dataset_stats.get(eval_label, {})

            dist = _compute_curve_distances(train_centers, train_mean, eval_mean)

            overlap = np.isfinite(train_mean) & np.isfinite(eval_mean)
            overlap_bins = int(np.sum(overlap))

            results.append(
                {
                    "train_dataset": train_meta.get("dataset"),
                    "train_split": train_meta.get("split"),
                    "eval_dataset": eval_meta.get("dataset"),
                    "eval_split": eval_meta.get("split"),
                    "train_label": train_label,
                    "eval_label": eval_label,
                    "train_images_seen": train_meta.get("images_seen"),
                    "train_images_used": train_meta.get("images_used"),
                    "train_total_vectors": train_meta.get("total_vectors"),
                    "train_total_pairs": train_meta.get("total_pairs"),
                    "eval_images_seen": eval_meta.get("images_seen"),
                    "eval_images_used": eval_meta.get("images_used"),
                    "eval_total_vectors": eval_meta.get("total_vectors"),
                    "eval_total_pairs": eval_meta.get("total_pairs"),
                    "train_auc": train_auc,
                    "eval_auc": eval_auc,
                    "train_auc_norm": train_auc_norm,
                    "eval_auc_norm": eval_auc_norm,
                    "auc_diff": abs(train_auc - eval_auc) if np.isfinite(train_auc) and np.isfinite(eval_auc) else np.nan,
                    "auc_diff_norm": abs(train_auc_norm - eval_auc_norm)
                    if np.isfinite(train_auc_norm) and np.isfinite(eval_auc_norm)
                    else np.nan,
                    "curve_l1": dist["curve_l1"],
                    "curve_l2": dist["curve_l2"],
                    "curve_corr": dist["curve_corr"],
                    "overlap_bins": overlap_bins,
                    "total_bins": int(len(train_centers)) if train_centers is not None else 0,
                }
            )

    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset-level variogram analysis (flow).")
    parser.add_argument("--config", required=True, help="Path to a coverage-style config YAML.")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum batches to process per dataset (default: config max_batches or 10).",
    )
    parser.add_argument(
        "--max-batches-overrides",
        nargs="*",
        default=None,
        help=(
            "Override max batches for specific datasets, e.g. "
            "spair=1000 or spair:train=1000. Keys are case-insensitive."
        ),
    )
    parser.add_argument("--results-file", type=str, default=None, help="CSV output for train/eval distances.")
    parser.add_argument("--curves-file", type=str, default=None, help="CSV output for per-dataset curves.")
    parser.add_argument("--n-bins", type=int, default=50, help="Number of spatial distance bins.")
    parser.add_argument(
        "--min-pairs-per-bin",
        type=int,
        default=50,
        help="Minimum pair count per bin to emit a value.",
    )
    parser.add_argument(
        "--vectors-per-image",
        type=int,
        default=None,
        help="Override vectors per image (default: config sampling vectors_per_image).",
    )
    args = parser.parse_args()

    run_variogram_dataset(
        config_path=args.config,
        max_batches=args.max_batches,
        output_results=args.results_file,
        output_curves=args.curves_file,
        max_batches_overrides=args.max_batches_overrides,
        n_bins=args.n_bins,
        min_pairs_per_bin=args.min_pairs_per_bin,
        vectors_per_image=args.vectors_per_image,
    )


if __name__ == "__main__":
    main()
