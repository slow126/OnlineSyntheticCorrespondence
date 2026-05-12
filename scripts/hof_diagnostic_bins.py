#!/usr/bin/env python3
"""
Diagnostic binning for HOF fingerprints.

Outputs, per train/eval pair:
- train_in_eval.csv
- train_out_eval.csv
- eval_in_train.csv
- eval_out_train.csv
- (optional) train_missing.csv / eval_missing.csv
- summary.json
- (optional) *_hist.npz (log-polar histograms for visualization)

The script is careful with sparse datasets by:
- computing radii on non-empty samples only
- applying a radius floor
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.calculate_coverage_hof import _load_or_build_vectors  # noqa: E402
from scripts.coverage import faiss_ops  # noqa: E402


def _as_int(value, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _as_float(value, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_item(x, default=None):
    if x is None:
        return default
    try:
        return np.asarray(x).item()
    except Exception:
        return default


def _dataset_key(ds_config: Dict) -> Tuple[str, str]:
    return ds_config.get("name"), ds_config.get("split")


def _repr_name_from_config(config: Dict) -> Tuple[str, str, Dict]:
    representation = config.get("representation", "hof")
    hof_cfg = config.get("hof", {})
    l2_normalize = bool(hof_cfg.get("l2_normalize", False))
    occupancy_mode = str(hof_cfg.get("occupancy_mode", "none"))
    occupancy_weight = float(hof_cfg.get("occupancy_weight", 0.3))
    if occupancy_weight < 0.0 or occupancy_weight > 1.0:
        raise ValueError("hof.occupancy_weight must be in [0, 1].")

    occ_tag = ""
    if occupancy_mode and occupancy_mode.lower() != "none":
        weight_tag = f"{occupancy_weight:.2f}".replace(".", "p")
        occ_tag = f"_occ{occupancy_mode.lower()}{weight_tag}"
    repr_name = f"{representation}{occ_tag}"
    if l2_normalize:
        repr_name = f"{repr_name}_l2"

    normalization = "l2" if l2_normalize else "raw"
    if occupancy_mode and occupancy_mode.lower() != "none":
        weight_tag = f"{occupancy_weight:.2f}".replace(".", "p")
        normalization = f"{normalization}_occ{occupancy_mode.lower()}{weight_tag}"

    return repr_name, normalization, hof_cfg


def _load_sample_table(
    hof_cache_dir: Path,
    dataset: str,
    split: str,
    max_samples: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    ds_dir = hof_cache_dir / dataset / split
    if not ds_dir.exists():
        raise FileNotFoundError(f"HOF cache directory not found: {ds_dir}")

    files = sorted(ds_dir.glob("*.npz"))
    if max_samples is not None:
        files = files[: max_samples]
    if not files:
        raise FileNotFoundError(f"No HOF fingerprints found in {ds_dir}")

    indices = np.zeros(len(files), dtype=np.int64)
    valid_counts = np.zeros(len(files), dtype=np.int64)
    sample_ids: List[str] = []
    cache_paths: List[str] = []

    for i, path in enumerate(files):
        with np.load(path) as data:
            idx = _safe_item(data.get("index"), None)
            if idx is None:
                try:
                    idx = int(path.stem)
                except Exception:
                    idx = i
            indices[i] = int(idx)
            valid_counts[i] = _as_int(_safe_item(data.get("valid_count"), 0), 0)
        sample_id = f"{dataset}/{split}/idx_{int(indices[i]):08d}"
        sample_ids.append(sample_id)
        cache_paths.append(str(path))

    return {
        "indices": indices,
        "valid_counts": valid_counts,
        "sample_ids": np.array(sample_ids, dtype=object),
        "cache_paths": np.array(cache_paths, dtype=object),
    }


def _compute_radius_from_vectors(
    vectors: np.ndarray,
    valid_mask: np.ndarray,
    k: int,
    quantile: float,
    neighbor_agg: str,
    use_gpu: bool,
    index_factory: str,
    nprobe: Optional[int],
    batch_size: Optional[int],
    radius_floor: Optional[float],
    verbose: bool = True,
) -> Dict[str, float]:
    result = {
        "radius": float("nan"),
        "radius_raw": float("nan"),
        "radius_floor": float("nan"),
        "median": float("nan"),
        "p90": float("nan"),
        "p95": float("nan"),
        "mean": float("nan"),
        "n_total": int(vectors.shape[0]),
        "n_valid": int(np.count_nonzero(valid_mask)),
    }
    if result["n_valid"] < 2:
        if verbose:
            print("  ⚠️  Not enough valid samples for radius (n_valid < 2)")
        return result

    vec = vectors[valid_mask]
    k_eff = min(k, max(1, vec.shape[0] - 1))
    if k_eff < 1:
        if verbose:
            print("  ⚠️  Not enough neighbors for radius")
        return result

    index = None
    fallback = None
    try:
        index = faiss_ops.build_index(
            vec,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            verbose=verbose,
        )
        if index_factory.lower() != "flat":
            fallback = faiss_ops.build_index(vec, use_gpu=True, index_factory="Flat", verbose=False)
        dists, _ = faiss_ops.compute_knn_distances(
            index,
            vec,
            k=k_eff,
            exclude_self=True,
            filter_duplicates=True,
            fallback_index=fallback,
            batch_size=batch_size,
            verbose=verbose,
        )
    finally:
        faiss_ops.release_index(index)
        faiss_ops.release_index(fallback)

    if dists.size == 0:
        return result

    if neighbor_agg in ("first", "min"):
        sample = dists[:, 0]
    elif neighbor_agg in ("kth", "last", "max"):
        sample = dists[:, k_eff - 1]
    elif neighbor_agg == "mean":
        sample = dists[:, :k_eff].mean(axis=1)
    elif neighbor_agg == "median":
        sample = np.median(dists[:, :k_eff], axis=1)
    else:
        raise ValueError(f"Unknown neighbor_agg: {neighbor_agg}")

    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return result

    radius_raw = float(np.quantile(sample, quantile))
    result["radius_raw"] = radius_raw
    result["median"] = float(np.median(sample))
    result["p90"] = float(np.quantile(sample, 0.90))
    result["p95"] = float(np.quantile(sample, 0.95))
    result["mean"] = float(np.mean(sample))

    if radius_floor is None:
        nonzero = sample[sample > 0]
        if nonzero.size > 0:
            radius_floor = float(np.median(nonzero))
        else:
            radius_floor = 1e-6

    result["radius_floor"] = float(radius_floor)
    result["radius"] = float(max(radius_raw, radius_floor))
    return result


def _split_histogram_from_fingerprint(
    fp: np.ndarray,
    grid_hw: Tuple[int, int],
    angle_bins: int,
    mag_bins: int,
) -> np.ndarray:
    gh, gw = grid_hw
    hist_bins = angle_bins * mag_bins
    per_cell = 1 + hist_bins
    expected_dim = gh * gw * per_cell
    if fp.shape[0] != expected_dim:
        raise ValueError(f"Unexpected fingerprint dim={fp.shape[0]} (expected {expected_dim})")
    cells = fp.reshape(gh * gw, per_cell)
    hist = cells[:, 1:]
    hist = hist.reshape(gh * gw, angle_bins, mag_bins)
    hist_sum = hist.sum(axis=0)
    return hist_sum.astype(np.float32, copy=False)


def _export_histograms(
    out_path: Path,
    sample_ids: Sequence[str],
    indices: Sequence[int],
    cache_paths: Sequence[str],
    grid_hw: Tuple[int, int],
    angle_bins: int,
    mag_bins: int,
    max_samples: Optional[int] = None,
) -> None:
    if max_samples is not None and len(sample_ids) > max_samples:
        sample_ids = sample_ids[:max_samples]
        indices = indices[:max_samples]
        cache_paths = cache_paths[:max_samples]

    hists = np.zeros((len(sample_ids), angle_bins, mag_bins), dtype=np.float32)
    for i, path in enumerate(cache_paths):
        with np.load(path) as data:
            fp = data["fingerprint"]
        hists[i] = _split_histogram_from_fingerprint(fp, grid_hw, angle_bins, mag_bins)

    np.savez_compressed(
        out_path,
        sample_id=np.array(sample_ids, dtype=object),
        index=np.asarray(indices, dtype=np.int64),
        hist=hists,
    )


def _bin_direction(
    samples: Dict[str, np.ndarray],
    distances: np.ndarray,
    indices: np.ndarray,
    radius: float,
    ratio_threshold: float,
    missing_mask: np.ndarray,
    neighbor_samples: Dict[str, np.ndarray],
    k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_mask = np.asarray(missing_mask, dtype=bool)
    k_eff = min(k, distances.shape[1]) if distances.ndim == 2 else 1
    d = distances[:, k_eff - 1]
    nn = indices[:, k_eff - 1]
    radius_ok = np.isfinite(radius) and (radius > 0)
    denom = radius if radius_ok else 1e-12
    ratio = d / denom

    missing = missing_mask | (~np.isfinite(d)) | (~radius_ok)
    covered = (~missing) & (ratio <= ratio_threshold)
    outside = (~missing) & (ratio > ratio_threshold)

    def _df(mask: np.ndarray) -> pd.DataFrame:
        mask = np.asarray(mask, dtype=bool)
        nn_ids = np.full(mask.shape[0], "", dtype=object)
        nn_idx = np.full(mask.shape[0], -1, dtype=np.int64)
        if nn.size > 0:
            nn_idx = nn.astype(np.int64)
            valid_nn = (nn_idx >= 0) & (nn_idx < neighbor_samples["sample_ids"].shape[0])
            nn_ids[valid_nn] = neighbor_samples["sample_ids"][nn_idx[valid_nn]]

        df = pd.DataFrame(
            {
                "sample_id": samples["sample_ids"],
                "index": samples["indices"],
                "cache_path": samples["cache_paths"],
                "valid_count": samples["valid_counts"],
                "distance_k": d,
                "radius": radius,
                "ratio": ratio,
                "neighbor_index": nn_idx,
                "neighbor_sample_id": nn_ids,
            }
        )
        return df[mask].reset_index(drop=True)

    return _df(covered), _df(outside), _df(missing)


def run(config_path: str, output_dir: Optional[str] = None):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    cache_dir = Path(config["cache"]["dir"]).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    hof_cache_dir = Path(config["cache"]["hof_dir"]).expanduser()

    repr_name, normalization, hof_cfg = _repr_name_from_config(config)
    grid_hw = tuple(hof_cfg.get("grid_hw", (32, 32)))
    angle_bins = int(hof_cfg.get("angle_bins", 8))
    mag_edges = hof_cfg.get("mag_edges", (0.0, 0.01, 0.03, 0.08, 0.25))
    mag_bins = max(1, len(mag_edges) - 1)
    hist_bins = angle_bins * mag_bins

    sampling_cfg = config.get("sampling", {})
    max_samples = sampling_cfg.get("max_samples")

    diag_cfg = config.get("diagnostics", {})
    k = _as_int(diag_cfg.get("k", 5), 5)
    radius_quantile = _as_float(diag_cfg.get("radius_quantile", 0.95), 0.95)
    neighbor_agg = str(diag_cfg.get("neighbor_agg", "kth"))
    ratio_threshold = _as_float(diag_cfg.get("ratio_threshold", 1.0), 1.0)
    min_valid_count = _as_int(diag_cfg.get("min_valid_count", 1), 1)
    radius_floor = diag_cfg.get("radius_floor")
    if radius_floor is not None:
        radius_floor = _as_float(radius_floor, None)
    export_hist = bool(diag_cfg.get("export_hist", True))
    max_hist_samples = diag_cfg.get("max_hist_samples")
    if max_hist_samples is not None:
        max_hist_samples = _as_int(max_hist_samples, None)

    faiss_cfg = config.get("faiss", {})
    use_gpu = bool(faiss_cfg.get("use_gpu", True))
    try:
        import torch

        if not torch.cuda.is_available():
            use_gpu = False
    except Exception:
        use_gpu = False

    index_factory = faiss_cfg.get("index_factory", "Flat")
    nprobe = faiss_cfg.get("nprobe")
    batch_size = faiss_cfg.get("batch_size")

    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets in config")

    train_configs = [d for d in datasets if not d.get("is_eval", False)]
    eval_configs = [d for d in datasets if d.get("is_eval", False)]

    out_root = Path(output_dir or diag_cfg.get("output_dir", "analysis/hof_diag"))
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Using HOF repr: {repr_name} (norm={normalization})")
    print(f"k={k}, radius_quantile={radius_quantile}, ratio_threshold={ratio_threshold}")
    print(f"export_hist={export_hist}")

    # Preload sample tables for all datasets
    sample_tables: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for ds in datasets:
        ds_name, split = _dataset_key(ds)
        key = (ds_name, split)
        print(f"\n[{ds_name}/{split}] loading sample table")
        samples = _load_sample_table(hof_cache_dir, ds_name, split, max_samples=max_samples)
        sample_tables[key] = samples

    # Precompute radii per dataset
    radii_cache: Dict[Tuple[str, str], Dict[str, float]] = {}
    for ds in datasets:
        ds_name, split = _dataset_key(ds)
        key = (ds_name, split)
        print(f"\n[{ds_name}/{split}] computing radius")
        vectors = _load_or_build_vectors(
            cache_dir,
            hof_cache_dir,
            ds_name,
            split,
            repr_name,
            max_samples=max_samples,
            l2_normalize=bool(hof_cfg.get("l2_normalize", False)),
            occupancy_mode=str(hof_cfg.get("occupancy_mode", "none")),
            occupancy_weight=float(hof_cfg.get("occupancy_weight", 0.3)),
            grid_hw=grid_hw,
            hist_bins=hist_bins,
            occ_mean_cache=None,
            verbose=True,
        )
        valid_counts = sample_tables[key]["valid_counts"]
        valid_mask = valid_counts >= min_valid_count
        radii = _compute_radius_from_vectors(
            vectors,
            valid_mask,
            k=k,
            quantile=radius_quantile,
            neighbor_agg=neighbor_agg,
            use_gpu=use_gpu,
            index_factory=index_factory,
            nprobe=nprobe,
            batch_size=batch_size,
            radius_floor=radius_floor,
            verbose=True,
        )
        radii_cache[key] = radii
        del vectors

    # Pairwise diagnostics
    for train_cfg in train_configs:
        train_ds, train_split = _dataset_key(train_cfg)
        train_key = (train_ds, train_split)
        print(f"\n=== Train set: {train_ds}/{train_split} ===")
        train_vectors = _load_or_build_vectors(
            cache_dir,
            hof_cache_dir,
            train_ds,
            train_split,
            repr_name,
            max_samples=max_samples,
            l2_normalize=bool(hof_cfg.get("l2_normalize", False)),
            occupancy_mode=str(hof_cfg.get("occupancy_mode", "none")),
            occupancy_weight=float(hof_cfg.get("occupancy_weight", 0.3)),
            grid_hw=grid_hw,
            hist_bins=hist_bins,
            occ_mean_cache=None,
            verbose=True,
        )
        train_samples = sample_tables[train_key]
        train_valid_mask = train_samples["valid_counts"] >= min_valid_count
        train_valid_idx = np.where(train_valid_mask)[0]

        # Build train index once
        print("  Building train index for eval→train queries")
        if train_valid_idx.size == 0:
            train_index = None
            train_fallback = None
            print("  ⚠️  No valid train samples; eval→train distances will be missing.")
        else:
            train_index = faiss_ops.build_index(
                train_vectors[train_valid_mask],
                use_gpu=use_gpu,
                index_factory=index_factory,
                nprobe=nprobe,
                verbose=True,
            )
        train_fallback = None
        if train_index is not None and index_factory.lower() != "flat":
            train_fallback = faiss_ops.build_index(
                train_vectors[train_valid_mask], use_gpu=True, index_factory="Flat", verbose=False
            )

        for eval_cfg in eval_configs:
            eval_ds, eval_split = _dataset_key(eval_cfg)
            eval_key = (eval_ds, eval_split)
            print(f"\n--- Eval set: {eval_ds}/{eval_split} ---")

            eval_vectors = _load_or_build_vectors(
                cache_dir,
                hof_cache_dir,
                eval_ds,
                eval_split,
                repr_name,
                max_samples=max_samples,
                l2_normalize=bool(hof_cfg.get("l2_normalize", False)),
                occupancy_mode=str(hof_cfg.get("occupancy_mode", "none")),
                occupancy_weight=float(hof_cfg.get("occupancy_weight", 0.3)),
                grid_hw=grid_hw,
                hist_bins=hist_bins,
                occ_mean_cache=None,
                verbose=True,
            )
            eval_samples = sample_tables[eval_key]
            eval_valid_mask = eval_samples["valid_counts"] >= min_valid_count
            eval_valid_idx = np.where(eval_valid_mask)[0]

            # Eval → Train distances/indices
            print("  Computing eval→train kNN")
            if train_index is None:
                eval_to_train_dists = np.full((eval_vectors.shape[0], 1), np.nan, dtype=np.float32)
                eval_to_train_idx = np.full((eval_vectors.shape[0], 1), -1, dtype=np.int64)
            else:
                eval_to_train_dists, eval_to_train_idx = faiss_ops.compute_knn_distances(
                    train_index,
                    eval_vectors,
                    k=min(k, train_valid_idx.size),
                    exclude_self=False,
                    filter_duplicates=False,
                    fallback_index=train_fallback,
                    batch_size=batch_size,
                    verbose=True,
                )
                # Map neighbor indices back to original train indices
                eval_to_train_idx = train_valid_idx[eval_to_train_idx]

            # Train → Eval distances/indices (build eval index)
            print("  Computing train→eval kNN")
            if eval_valid_idx.size == 0:
                eval_index = None
                eval_fallback = None
                print("  ⚠️  No valid eval samples; train→eval distances will be missing.")
            else:
                eval_index = faiss_ops.build_index(
                    eval_vectors[eval_valid_mask],
                    use_gpu=use_gpu,
                    index_factory=index_factory,
                    nprobe=nprobe,
                    verbose=True,
                )
            eval_fallback = None
            if eval_index is not None and index_factory.lower() != "flat":
                eval_fallback = faiss_ops.build_index(
                    eval_vectors[eval_valid_mask], use_gpu=True, index_factory="Flat", verbose=False
                )
            if eval_index is None:
                train_to_eval_dists = np.full((train_vectors.shape[0], 1), np.nan, dtype=np.float32)
                train_to_eval_idx = np.full((train_vectors.shape[0], 1), -1, dtype=np.int64)
            else:
                train_to_eval_dists, train_to_eval_idx = faiss_ops.compute_knn_distances(
                    eval_index,
                    train_vectors,
                    k=min(k, eval_valid_idx.size),
                    exclude_self=False,
                    filter_duplicates=False,
                    fallback_index=eval_fallback,
                    batch_size=batch_size,
                    verbose=True,
                )
                # Map neighbor indices back to original eval indices
                train_to_eval_idx = eval_valid_idx[train_to_eval_idx]
            faiss_ops.release_index(eval_index)
            faiss_ops.release_index(eval_fallback)

            train_radius = radii_cache[train_key]["radius"]
            eval_radius = radii_cache[eval_key]["radius"]

            train_missing = train_samples["valid_counts"] < min_valid_count
            eval_missing = eval_samples["valid_counts"] < min_valid_count

            eval_in_train, eval_out_train, eval_missing_df = _bin_direction(
                eval_samples,
                eval_to_train_dists,
                eval_to_train_idx,
                radius=train_radius,
                ratio_threshold=ratio_threshold,
                missing_mask=eval_missing,
                neighbor_samples=train_samples,
                k=k,
            )

            train_in_eval, train_out_eval, train_missing_df = _bin_direction(
                train_samples,
                train_to_eval_dists,
                train_to_eval_idx,
                radius=eval_radius,
                ratio_threshold=ratio_threshold,
                missing_mask=train_missing,
                neighbor_samples=eval_samples,
                k=k,
            )

            out_dir = out_root / f"{train_ds}_{train_split}__{eval_ds}_{eval_split}"
            out_dir.mkdir(parents=True, exist_ok=True)

            eval_in_train.to_csv(out_dir / "eval_in_train.csv", index=False)
            eval_out_train.to_csv(out_dir / "eval_out_train.csv", index=False)
            train_in_eval.to_csv(out_dir / "train_in_eval.csv", index=False)
            train_out_eval.to_csv(out_dir / "train_out_eval.csv", index=False)
            if not eval_missing_df.empty:
                eval_missing_df.to_csv(out_dir / "eval_missing.csv", index=False)
            if not train_missing_df.empty:
                train_missing_df.to_csv(out_dir / "train_missing.csv", index=False)

            summary = {
                "train_dataset": train_ds,
                "train_split": train_split,
                "eval_dataset": eval_ds,
                "eval_split": eval_split,
                "k": k,
                "radius_quantile": radius_quantile,
                "ratio_threshold": ratio_threshold,
                "train_radius": train_radius,
                "eval_radius": eval_radius,
                "train_radius_raw": radii_cache[train_key]["radius_raw"],
                "eval_radius_raw": radii_cache[eval_key]["radius_raw"],
                "train_radius_floor": radii_cache[train_key]["radius_floor"],
                "eval_radius_floor": radii_cache[eval_key]["radius_floor"],
                "n_train": int(train_vectors.shape[0]),
                "n_eval": int(eval_vectors.shape[0]),
                "n_train_missing": int(train_missing_df.shape[0]),
                "n_eval_missing": int(eval_missing_df.shape[0]),
                "n_train_in_eval": int(train_in_eval.shape[0]),
                "n_train_out_eval": int(train_out_eval.shape[0]),
                "n_eval_in_train": int(eval_in_train.shape[0]),
                "n_eval_out_train": int(eval_out_train.shape[0]),
            }
            with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, sort_keys=True)

            if export_hist:
                print("  Exporting histograms for visualization")
                if not eval_out_train.empty:
                    _export_histograms(
                        out_dir / "eval_out_train_hist.npz",
                        eval_out_train["sample_id"].tolist(),
                        eval_out_train["index"].tolist(),
                        eval_out_train["cache_path"].tolist(),
                        grid_hw=grid_hw,
                        angle_bins=angle_bins,
                        mag_bins=mag_bins,
                        max_samples=max_hist_samples,
                    )
                if not eval_in_train.empty:
                    _export_histograms(
                        out_dir / "eval_in_train_hist.npz",
                        eval_in_train["sample_id"].tolist(),
                        eval_in_train["index"].tolist(),
                        eval_in_train["cache_path"].tolist(),
                        grid_hw=grid_hw,
                        angle_bins=angle_bins,
                        mag_bins=mag_bins,
                        max_samples=max_hist_samples,
                    )
                if not train_out_eval.empty:
                    _export_histograms(
                        out_dir / "train_out_eval_hist.npz",
                        train_out_eval["sample_id"].tolist(),
                        train_out_eval["index"].tolist(),
                        train_out_eval["cache_path"].tolist(),
                        grid_hw=grid_hw,
                        angle_bins=angle_bins,
                        mag_bins=mag_bins,
                        max_samples=max_hist_samples,
                    )
                if not train_in_eval.empty:
                    _export_histograms(
                        out_dir / "train_in_eval_hist.npz",
                        train_in_eval["sample_id"].tolist(),
                        train_in_eval["index"].tolist(),
                        train_in_eval["cache_path"].tolist(),
                        grid_hw=grid_hw,
                        angle_bins=angle_bins,
                        mag_bins=mag_bins,
                        max_samples=max_hist_samples,
                    )

            del eval_vectors

        faiss_ops.release_index(train_index)
        faiss_ops.release_index(train_fallback)
        del train_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic binning for HOF fingerprints")
    parser.add_argument("--config", required=True, help="Path to HOF coverage config YAML")
    parser.add_argument("--output-dir", default=None, help="Output directory for diagnostics")
    args = parser.parse_args()

    run(args.config, output_dir=args.output_dir)
