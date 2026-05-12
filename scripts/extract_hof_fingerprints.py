#!/usr/bin/env python3
"""
Extract and cache per-image HOF fingerprints.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional
import json
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calculate_coverage_faiss import (
    create_dataset_from_config,
    create_mixed_dataset_from_config,
    _is_synthetic_dataset,
)

from src.hof.fingerprint import HOFFingerprintConfig, compute_hof_fingerprint, fingerprint_dim
from src.hof.cache import (
    fingerprint_cache_dir,
    fingerprint_cache_path,
    load_manifest,
    write_manifest,
    save_fingerprint,
)
from src.hof.batch_utils import extract_flow_from_batch, extract_valid_mask_from_batch


DEFAULT_HOF_CFG = {
    "grid_hw": [32, 32],
    "angle_bins": 8,
    "mag_edges": [0.0, 0.01, 0.03, 0.08, 0.25],
    "mag_clip": None,
    "occupancy_tau": 5.0,
    "normalize_hist": True,
    "use_sqrt_mag": True,
    "zero_is_invalid": False,
}



def _build_manifest_base(config: Dict, ds_config: Dict, dataset_name: str, split: str, hof_cfg: HOFFingerprintConfig, shuffle: bool, seed: int, num_workers: int, batch_size: int) -> Dict:
    base = {
        "dataset": dataset_name,
        "split": split,
        "hof_config": hof_cfg.as_dict(),
        "fingerprint_dim": fingerprint_dim(hof_cfg),
        "sampling": {
            "shuffle": bool(shuffle),
            "seed": int(seed),
        },
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "dataset_params": config.get("dataset_params", {}),
        "dataset_overrides": config.get("dataset_overrides", {}).get(dataset_name, {}),
        "entry_overrides": ds_config.get("overrides"),
    }
    if ds_config.get("mixed", False):
        base["mixed"] = True
        base["datasets"] = ds_config.get("datasets")
        base["percentages"] = ds_config.get("percentages")
        base["mixed_seed"] = int(config.get("sampling", {}).get("seed", 42))
    return base


def _manifest_matches(existing: Dict, expected: Dict) -> bool:
    for key, val in expected.items():
        if existing.get(key) != val:
            return False
    return True

def _build_hof_config(cfg: Dict) -> HOFFingerprintConfig:
    merged = DEFAULT_HOF_CFG.copy()
    merged.update(cfg or {})
    grid_hw = tuple(merged["grid_hw"])
    mag_edges = tuple(float(x) for x in merged["mag_edges"])
    return HOFFingerprintConfig(
        grid_hw=grid_hw,
        angle_bins=int(merged["angle_bins"]),
        mag_edges=mag_edges,
        mag_clip=merged.get("mag_clip", None),
        occupancy_tau=float(merged["occupancy_tau"]),
        normalize_hist=bool(merged["normalize_hist"]),
        use_sqrt_mag=bool(merged["use_sqrt_mag"]),
        zero_is_invalid=bool(merged["zero_is_invalid"]),
    )


def _dataset_matches(name: str, only: Optional[List[str]]) -> bool:
    if not only:
        return True
    return name in set(only)


def _get_batch_size_from_flow(batch: Dict) -> int:
    flow_ref = batch.get("flow_full")
    if flow_ref is None:
        flow_ref = batch.get("flow")
    if flow_ref is None:
        return 0
    if isinstance(flow_ref, torch.Tensor):
        if flow_ref.ndim == 3:
            return 1
        if flow_ref.ndim == 4:
            return int(flow_ref.shape[0])
    if isinstance(flow_ref, np.ndarray):
        if flow_ref.ndim == 3:
            return 1
        if flow_ref.ndim == 4:
            return int(flow_ref.shape[0])
    return 0


def run(
    config_path: str,
    only: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    overwrite: bool = False,
    log_every: int = 200,
    diagnostic_samples: int = 25,
):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if max_samples is None:
        sampling_cfg = config.get("sampling", {})
        cfg_cap = sampling_cfg.get("max_samples")
        if cfg_cap is None:
            cfg_cap = config.get("max_samples")
        if cfg_cap is not None:
            max_samples = int(cfg_cap)

    cache_root = Path(config["cache"]["dir"]).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    hof_cfg_base = _build_hof_config(config.get("hof", {}))

    sampling_cfg = config.get("sampling", {})
    shuffle = bool(sampling_cfg.get("shuffle", False))
    seed = int(sampling_cfg.get("seed", 42))

    datasets = config.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets specified in config")

    for ds_config in datasets:
        dataset_name = ds_config.get("name")
        split = ds_config.get("split")

        if dataset_name is None or split is None:
            raise ValueError("Each dataset entry must have name and split")

        if not _dataset_matches(dataset_name, only):
            continue

        print(f"\n[{dataset_name}/{split}]")

        hof_cfg = hof_cfg_base
        if ds_config.get("hof_overrides"):
            merged = dict(hof_cfg_base.as_dict())
            merged.update(ds_config["hof_overrides"])
            hof_cfg = _build_hof_config(merged)

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
                seed=seed,
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

        num_workers = 0 if is_synthetic else int(config.get("num_workers", 0))
        pin_memory = False if is_synthetic else True
        if is_synthetic and int(config.get("num_workers", 0)) > 0:
            print("  Synthetic dataset detected - forcing num_workers=0 and pin_memory=False")

        batch_size = int(config.get("batch_size", 1))

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        )

        ds_cache_dir = fingerprint_cache_dir(cache_root, dataset_name, split)
        manifest_base = _build_manifest_base(
            config,
            ds_config,
            dataset_name,
            split,
            hof_cfg,
            shuffle,
            seed,
            num_workers,
            batch_size,
        )

        if shuffle and not overwrite and ds_cache_dir.exists():
            existing_cache = list(ds_cache_dir.glob("*.npz"))
            if existing_cache:
                raise ValueError(
                    f"shuffle=true with existing cache for {dataset_name}/{split} is unsafe. "
                    "Use --overwrite or set sampling.shuffle=false."
                )

        if overwrite:
            write_manifest(ds_cache_dir, manifest_base, overwrite=True)
        else:
            existing = load_manifest(ds_cache_dir)
            if existing is None:
                write_manifest(ds_cache_dir, manifest_base, overwrite=False)
            elif not _manifest_matches(existing, manifest_base):
                raise ValueError(
                    f"Manifest mismatch for {dataset_name}/{split}. "
                    "Use --overwrite to replace."
                )

        total = len(dataset)
        if max_samples is not None:
            total = min(total, max_samples)

        pbar = tqdm(total=total, desc=f"  extracting", leave=False)

        index = 0
        processed = 0
        empty_count = 0
        valid_sum = 0
        diag_seen = 0
        diag_occ_mean = []
        diag_occ_nonzero = []
        diag_hist_mass = []
        for batch in dataloader:
            if max_samples is not None and index >= max_samples:
                break

            batch_size = _get_batch_size_from_flow(batch)
            if batch_size == 0:
                continue

            for i in range(batch_size):
                if max_samples is not None and index >= max_samples:
                    break

                path = fingerprint_cache_path(cache_root, dataset_name, split, index, ext="npz")
                if path.exists() and not overwrite:
                    index += 1
                    pbar.update(1)
                    continue

                flow = extract_flow_from_batch(batch, index=i, prefer_full=True)
                if flow is None:
                    index += 1
                    pbar.update(1)
                    continue

                valid_mask = extract_valid_mask_from_batch(batch, index=i)
                want_diag = diagnostic_samples > 0 and diag_seen < diagnostic_samples
                fingerprint, meta = compute_hof_fingerprint(
                    flow,
                    valid_mask,
                    hof_cfg,
                    return_components=want_diag,
                )

                meta["index"] = index

                save_fingerprint(path, fingerprint, meta=meta, compressed=True)

                processed += 1
                valid_sum += int(meta.get("valid_count", 0))
                if int(meta.get("valid_count", 0)) == 0:
                    empty_count += 1

                if want_diag:
                    occ = meta.get("occupancy")
                    hist = meta.get("histogram")
                    if occ is not None and hist is not None:
                        occ_mean = float(np.mean(occ))
                        occ_nonzero = float(np.mean(occ > 0))
                        hist_mass = float(np.mean(hist.sum(axis=(2, 3))))
                        diag_occ_mean.append(occ_mean)
                        diag_occ_nonzero.append(occ_nonzero)
                        diag_hist_mass.append(hist_mass)
                    diag_seen += 1

                index += 1
                pbar.update(1)

                if log_every > 0 and processed > 0 and processed % log_every == 0:
                    avg_valid = valid_sum / max(1, processed)
                    empty_frac = empty_count / max(1, processed)
                    print(
                        f"  [{dataset_name}/{split}] processed={processed} "
                        f"avg_valid={avg_valid:.1f} empty_frac={empty_frac:.3f}"
                    )

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

        # Update manifest with sample count
        manifest = dict(manifest_base)
        manifest["samples"] = int(min(len(dataset), max_samples) if max_samples is not None else len(dataset))
        write_manifest(ds_cache_dir, manifest, overwrite=True)

        if diagnostic_samples > 0 and diag_seen > 0:
            diag = {
                "dataset": dataset_name,
                "split": split,
                "samples": diag_seen,
                "occupancy_mean_avg": float(np.mean(diag_occ_mean)) if diag_occ_mean else float("nan"),
                "occupancy_nonzero_frac_avg": float(np.mean(diag_occ_nonzero)) if diag_occ_nonzero else float("nan"),
                "hist_mass_avg": float(np.mean(diag_hist_mass)) if diag_hist_mass else float("nan"),
                "avg_valid_count": float(valid_sum / max(1, processed)),
                "empty_frac": float(empty_count / max(1, processed)),
            }
            diag_path = ds_cache_dir / "hof_diagnostics.json"
            with diag_path.open("w", encoding="utf-8") as f:
                json.dump(diag, f, indent=2, sort_keys=True)

        del dataset, dataloader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract HOF fingerprints")
    parser.add_argument("--config", required=True, help="Path to HOF config YAML")
    parser.add_argument("--only", nargs="*", help="Only process these dataset names")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    parser.add_argument("--log-every", type=int, default=200, help="Log progress every N samples")
    parser.add_argument("--diagnostic-samples", type=int, default=25, help="Compute diagnostics on first N samples")
    args = parser.parse_args()

    run(
        config_path=args.config,
        only=args.only,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        log_every=args.log_every,
        diagnostic_samples=args.diagnostic_samples,
    )
