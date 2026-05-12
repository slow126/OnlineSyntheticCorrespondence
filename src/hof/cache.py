"""Caching utilities for HOF fingerprints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json
import re
import numpy as np


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def fingerprint_cache_dir(cache_root: Path, dataset: str, split: str) -> Path:
    safe_ds = sanitize_name(dataset)
    safe_split = sanitize_name(split)
    return cache_root / safe_ds / safe_split


def fingerprint_cache_path(
    cache_root: Path,
    dataset: str,
    split: str,
    index: int,
    ext: str = "npz",
) -> Path:
    cache_dir = fingerprint_cache_dir(cache_root, dataset, split)
    return cache_dir / f"{index:08d}.{ext}"


def load_manifest(cache_dir: Path) -> Optional[Dict[str, Any]]:
    path = cache_dir / "manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_manifest(
    cache_dir: Path,
    manifest: Dict[str, Any],
    overwrite: bool = False,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "manifest.json"
    if path.exists() and not overwrite:
        existing = load_manifest(cache_dir)
        if existing is not None and existing != manifest:
            raise ValueError(
                "Existing manifest does not match requested config. "
                "Use overwrite=True to replace."
            )
        return path

    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return path


def save_fingerprint(
    path: Path,
    fingerprint: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    compressed: bool = True,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"fingerprint": fingerprint.astype(np.float32, copy=False)}
    if meta:
        for k, v in meta.items():
            if isinstance(v, np.ndarray):
                payload[k] = v
            else:
                payload[k] = np.asarray(v)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path
