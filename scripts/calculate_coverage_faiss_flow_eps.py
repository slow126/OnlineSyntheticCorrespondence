#!/usr/bin/env python3
"""
Coverage Pipeline v2.1 - Raw Flow Epsilon Curves

Computes directed NN distances between train/eval flow vectors and
reports coverage over fixed epsilon thresholds (pixels or normalized).

This avoids self-radius normalization and focuses on asymmetric distance
curves: eval→train and train→eval.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Dataset utilities
from calculate_coverage_faiss import (
    create_dataset_from_config,
    create_mixed_dataset_from_config,
    _is_synthetic_dataset,
)

# Vector extraction (reused from v2 pipeline)
from calculate_coverage_faiss_v2 import extract_vectors_from_dataset

# Modular coverage utilities
from coverage import cache, spaces, faiss_ops


def _format_eps_label(eps_px: float) -> str:
    if float(eps_px).is_integer():
        return str(int(eps_px))
    return f"{eps_px:g}".replace(".", "p")


def _convert_epsilons(
    eps_px: List[float],
    img_w: int,
    img_h: int,
    use_normalized: bool,
) -> List[Dict[str, float]]:
    """
    Convert epsilon values in pixels to the units used by vectors.

    Distances are squared L2 (FAISS default). We return squared thresholds.
    """
    eps_info = []
    if not use_normalized:
        for e in eps_px:
            eps_info.append(
                {
                    "eps_px": float(e),
                    "eps_norm": float(e),
                    "eps_sq": float(e * e),
                }
            )
        return eps_info

    scale_x = 2.0 / float(img_w)
    scale_y = 2.0 / float(img_h)
    if img_w != img_h:
        print(
            f"  ⚠️  Non-square flow normalization ({img_w}x{img_h}). "
            "Using geometric-mean scaling for epsilon conversion."
        )
    scale = scale_x if img_w == img_h else float((scale_x * scale_y) ** 0.5)

    for e in eps_px:
        eps_norm = float(e) * scale
        eps_info.append(
            {
                "eps_px": float(e),
                "eps_norm": eps_norm,
                "eps_sq": float(eps_norm * eps_norm),
            }
        )
    return eps_info


def _distance_stats(dists: np.ndarray) -> Dict[str, float]:
    if dists.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(dists)),
        "median": float(np.median(dists)),
        "p90": float(np.quantile(dists, 0.90)),
        "p95": float(np.quantile(dists, 0.95)),
    }


def run_pipeline(config_path: str):
    print(f"\n{'=' * 80}")
    print("COVERAGE PIPELINE v2.1 - RAW FLOW EPSILON CURVES")
    print(f"{'=' * 80}\n")
    print(f"Config: {config_path}\n")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    representation = config["representation"]
    if representation != "flow":
        raise ValueError("This pipeline only supports representation: flow")

    cache_dir = Path(config["cache"]["dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and config["faiss"]["use_gpu"] else "cpu"

    # ======================
    # STEP 0: Load/Extract Vectors
    # ======================
    print(f"\n{'=' * 80}")
    print("STEP 0: LOAD/EXTRACT VECTORS")
    print(f"{'=' * 80}\n")

    train_vectors = {}
    eval_vectors = {}

    for ds_config in config["datasets"]:
        is_eval = ds_config.get("is_eval", False)
        dataset_name = ds_config.get("name")
        split = ds_config.get("split")

        print(f"\n[{dataset_name}/{split}] {'(eval)' if is_eval else '(train)'}")

        vectors = cache.load_cached_vectors(cache_dir, dataset_name, split, representation)
        dataset = None
        dataloader = None

        if vectors is None:
            if ds_config.get("mixed", False):
                print(
                    "  Mixed dataset: "
                    + " + ".join(
                        [f"{d}({p:.0%})" for d, p in zip(ds_config["datasets"], ds_config["percentages"])]
                    )
                )
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
                print("  ⚠️  Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")

            dataloader = DataLoader(
                dataset,
                batch_size=config["batch_size"],
                shuffle=config["sampling"].get("shuffle", True),
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
            )

            extract_kwargs = {
                "max_vectors": config["sampling"]["max_vectors"],
                "vectors_per_image": config["sampling"].get(
                    "vectors_per_image", config["sampling"].get("flow_per_image_max", 2000)
                ),
                "seed": config["sampling"]["seed"],
                "device": device,
                "verbose": True,
            }
            if representation == "flow":
                img_size = config.get("flow_normalization", {}).get("image_size", [512, 512])
                extract_kwargs["image_size"] = img_size
                extract_kwargs["stats_dir"] = cache_dir / "stats"
                extract_kwargs["dataset_label"] = f"{dataset_name}_{split}_{representation}"

            vectors = extract_vectors_from_dataset(
                dataset,
                dataloader,
                representation,
                encoder=None,
                **extract_kwargs,
            )

            if representation == "flow":
                print("  Pre-cache check:")
                print(f"    Shape: {vectors.shape}")
                print(f"    dx: [{vectors[:, 2].min():.2f}, {vectors[:, 2].max():.2f}]")
                print(f"    dy: [{vectors[:, 3].min():.2f}, {vectors[:, 3].max():.2f}]")

            cache.save_cached_vectors(cache_dir, dataset_name, split, representation, vectors)
        else:
            print(f"  ✓ Loaded {len(vectors):,} cached vectors")

        key = (dataset_name, split)
        if is_eval:
            eval_vectors[key] = vectors
        else:
            train_vectors[key] = vectors

        if dataset is not None:
            del dataset
        if dataloader is not None:
            del dataloader
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\nLoaded {len(train_vectors)} train sets, {len(eval_vectors)} eval sets")

    # ======================
    # PREPROCESSING
    # ======================
    flow_norm = config.get("flow_normalization", {}).get("enabled", False)
    if flow_norm:
        print("\nNormalizing flow vectors to [-1, 1]...")
        img_h, img_w = config["flow_normalization"]["image_size"]
        for key, vectors in train_vectors.items():
            train_vectors[key] = spaces.normalize_flow_vectors(vectors, img_w, img_h)
        for key, vectors in eval_vectors.items():
            eval_vectors[key] = spaces.normalize_flow_vectors(vectors, img_w, img_h)
    else:
        img_h, img_w = config.get("flow_normalization", {}).get("image_size", [512, 512])

    # ======================
    # STEP 1: Define Space
    # ======================
    space_name = config.get("flow_space", "flow")
    if space_name not in ("flow", "xy", "joint"):
        raise ValueError(f"Unsupported flow_space: {space_name}")
    joint_alpha = float(config.get("joint_alpha", 1.0))

    print(f"\n{'=' * 80}")
    if space_name == "flow":
        print("STEP 1: DEFINE FLOW SPACE (dx, dy)")
    elif space_name == "xy":
        print("STEP 1: DEFINE XY SPACE (x, y)")
    else:
        print(f"STEP 1: DEFINE JOINT SPACE (x, y, {joint_alpha:.3f}*dx, {joint_alpha:.3f}*dy)")
    print(f"{'=' * 80}\n")

    space_train_vectors = {
        k: spaces.transform_to_space(v, space_name, alpha=joint_alpha) for k, v in train_vectors.items()
    }
    space_eval_vectors = {
        k: spaces.transform_to_space(v, space_name, alpha=joint_alpha) for k, v in eval_vectors.items()
    }

    # ======================
    # STEP 2: Epsilon Curves
    # ======================
    print(f"\n{'=' * 80}")
    print("STEP 2: EPSILON CURVES")
    print(f"{'=' * 80}\n")

    eps_cfg = config.get("epsilon_curves", {})
    eps_values_px = eps_cfg.get("values_px", [1, 2, 4, 8, 16, 32, 64])
    eps_info = _convert_epsilons(eps_values_px, img_w, img_h, use_normalized=flow_norm)

    print("Epsilon thresholds:")
    for info in eps_info:
        label = _format_eps_label(info["eps_px"])
        if flow_norm:
            print(f"  {label}px → {info['eps_norm']:.6f} (squared={info['eps_sq']:.6f})")
        else:
            print(f"  {label}px (squared={info['eps_sq']:.6f})")

    results = []
    curves = []

    for train_key in space_train_vectors.keys():
        for eval_key in space_eval_vectors.keys():
            train_name, train_split = train_key
            eval_name, eval_split = eval_key

            print(f"\n[{train_name}/{train_split} → {eval_name}/{eval_split}]")

            train_vecs = space_train_vectors[train_key]
            eval_vecs = space_eval_vectors[eval_key]

            directed = faiss_ops.compute_directed_distances(
                train_vecs,
                eval_vecs,
                k=1,
                use_gpu=config["faiss"]["use_gpu"],
                index_factory=config["faiss"]["index_factory"],
                batch_size=config["faiss"].get("batch_size"),
                verbose=True,
            )

            eval_to_train = directed["eval_to_train"][:, 0]
            train_to_eval = directed["train_to_eval"][:, 0]

            row = {
                "space": space_name,
                "train_dataset": train_name,
                "train_split": train_split,
                "eval_dataset": eval_name,
                "eval_split": eval_split,
                "train_n_vectors": int(len(train_vecs)),
                "eval_n_vectors": int(len(eval_vecs)),
                "distance_metric": config["distance_metric"]["name"],
                "flow_normalized": bool(flow_norm),
            }

            eval_stats = _distance_stats(eval_to_train)
            train_stats = _distance_stats(train_to_eval)

            row.update(
                {
                    "mean_nn_eval_to_train_k1": eval_stats["mean"],
                    "median_nn_eval_to_train_k1": eval_stats["median"],
                    "p90_nn_eval_to_train_k1": eval_stats["p90"],
                    "p95_nn_eval_to_train_k1": eval_stats["p95"],
                    "mean_nn_train_to_eval_k1": train_stats["mean"],
                    "median_nn_train_to_eval_k1": train_stats["median"],
                    "p90_nn_train_to_eval_k1": train_stats["p90"],
                    "p95_nn_train_to_eval_k1": train_stats["p95"],
                }
            )

            for info in eps_info:
                eps_px = info["eps_px"]
                eps_label = _format_eps_label(eps_px)
                eps_sq = info["eps_sq"]
                eval_cov = float(np.mean(eval_to_train <= eps_sq))
                train_cov = float(np.mean(train_to_eval <= eps_sq))

                col_eval = f"eval_covered_by_train_eps{eps_label}px"
                col_train = f"train_covered_by_eval_eps{eps_label}px"
                row[col_eval] = eval_cov
                row[col_train] = train_cov

                curves.append(
                    {
                        "space": space_name,
                        "train_dataset": train_name,
                        "train_split": train_split,
                        "eval_dataset": eval_name,
                        "eval_split": eval_split,
                        "direction": "eval_to_train",
                        "epsilon_px": info["eps_px"],
                        "epsilon_norm": info["eps_norm"] if flow_norm else np.nan,
                        "epsilon_sq": info["eps_sq"],
                        "coverage": eval_cov,
                    }
                )
                curves.append(
                    {
                        "space": space_name,
                        "train_dataset": train_name,
                        "train_split": train_split,
                        "eval_dataset": eval_name,
                        "eval_split": eval_split,
                        "direction": "train_to_eval",
                        "epsilon_px": info["eps_px"],
                        "epsilon_norm": info["eps_norm"] if flow_norm else np.nan,
                        "epsilon_sq": info["eps_sq"],
                        "coverage": train_cov,
                    }
                )

            results.append(row)

            for info in eps_info:
                eps_label = _format_eps_label(info["eps_px"])
                col_eval = f"eval_covered_by_train_eps{eps_label}px"
                col_train = f"train_covered_by_eval_eps{eps_label}px"
                print(
                    f"  eps={eps_label}px: "
                    f"eval_covered={row[col_eval]:.3f}, "
                    f"train_covered={row[col_train]:.3f}"
                )

    # ======================
    # Save Results
    # ======================
    print(f"\n{'=' * 80}")
    print("SAVING RESULTS")
    print(f"{'=' * 80}\n")

    results_df = pd.DataFrame(results)
    output_file = Path(config["output"]["results_file"])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"✓ Results saved to: {output_file}")
    print(f"  Total rows: {len(results_df)}")

    curves_file = Path(config["output"]["curves_file"])
    curves_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(curves).to_csv(curves_file, index=False)
    print(f"✓ Curves saved to: {curves_file}")
    print(f"  Total curve rows: {len(curves)}")

    print(f"\n{'=' * 80}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Coverage Pipeline v2.1 (Flow Epsilon Curves)")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    run_pipeline(args.config)


if __name__ == "__main__":
    main()
