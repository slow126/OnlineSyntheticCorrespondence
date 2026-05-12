#!/usr/bin/env python3
"""
Visualize joint-flow support bins with raw epsilon thresholds.

Given train/eval flow vectors [x, y, dx, dy], this script:
1) Computes directed 1-NN distances in a chosen flow space (default: joint).
2) Splits vectors into four bins using raw epsilon thresholds:
   - train_in_eval
   - train_out_eval
   - eval_in_train
   - eval_out_train
3) Renders Gaussian-splat figures for each bin using visualize_flow_splats.py.

This mirrors the HOF diagnostic grouping workflow, but operates directly on flow vectors.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

import visualize_flow_splats as vfs  # type: ignore
from scripts.coverage import faiss_ops, spaces  # type: ignore


def _write_placeholder_image(out_path: Path, title: str, subtitle: str, dpi: int = 200) -> None:
    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()
    ax.set_facecolor("#f2f2f2")
    ax.text(0.5, 0.6, title, ha="center", va="center", fontsize=14, weight="bold")
    ax.text(0.5, 0.45, subtitle, ha="center", va="center", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_flows(path: Path, max_vectors: Optional[int], seed: int, label: str) -> np.ndarray:
    flows = vfs.load_any(str(path))
    if flows.ndim != 2 or flows.shape[1] != 4:
        raise ValueError(f"{label}: expected (N, 4) flow vectors, got shape={flows.shape}")
    flows = flows.astype(np.float32, copy=False)
    finite = np.isfinite(flows).all(axis=1)
    if not finite.all():
        n_drop = int((~finite).sum())
        print(f"[WARN] {label}: dropping {n_drop:,} rows with non-finite values")
        flows = flows[finite]
    if max_vectors is not None and max_vectors > 0 and flows.shape[0] > max_vectors:
        rng = np.random.default_rng(seed)
        idx = rng.choice(flows.shape[0], size=max_vectors, replace=False)
        flows = flows[idx]
    return flows


def _dist_stats(dist_sq: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(dist_sq, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean_sq": float("nan"),
            "median_sq": float("nan"),
            "p90_sq": float("nan"),
            "p95_sq": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    arr_lin = np.sqrt(np.maximum(arr, 0.0))
    return {
        "mean_sq": float(np.mean(arr)),
        "median_sq": float(np.median(arr)),
        "p90_sq": float(np.quantile(arr, 0.90)),
        "p95_sq": float(np.quantile(arr, 0.95)),
        "mean": float(np.mean(arr_lin)),
        "median": float(np.median(arr_lin)),
        "p90": float(np.quantile(arr_lin, 0.90)),
        "p95": float(np.quantile(arr_lin, 0.95)),
    }


def _safe_frac(n: int, d: int) -> float:
    if d <= 0:
        return float("nan")
    return float(n) / float(d)


def _eps_to_space_units(
    eps_px: float,
    flow_normalized: bool,
    img_w: int,
    img_h: int,
) -> float:
    if not flow_normalized:
        return float(eps_px)

    scale_x = 2.0 / float(img_w)
    scale_y = 2.0 / float(img_h)
    scale = scale_x if img_w == img_h else float((scale_x * scale_y) ** 0.5)
    return float(eps_px) * scale


def _transform_vectors(
    flows: np.ndarray,
    space_name: str,
    joint_alpha: float,
    flow_normalized: bool,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    vec = flows
    if flow_normalized:
        vec = spaces.normalize_flow_vectors(vec, img_w, img_h)
    vec = spaces.transform_to_space(vec, space_name, alpha=joint_alpha)
    return np.ascontiguousarray(vec.astype(np.float32, copy=False))


def _save_bin_npz(
    out_path: Path,
    flows: np.ndarray,
    dist_sq: np.ndarray,
    threshold: float,
    threshold_sq: float,
    space_name: str,
    joint_alpha: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        flows=flows.astype(np.float32, copy=False),
        distance_sq=dist_sq.astype(np.float32, copy=False),
        threshold=float(threshold),
        threshold_sq=float(threshold_sq),
        space=np.array(space_name),
        joint_alpha=float(joint_alpha),
    )


def _render_bin(
    flows: np.ndarray,
    dataset_name: str,
    out_png: Path,
    args: argparse.Namespace,
) -> None:
    if flows.shape[0] == 0:
        _write_placeholder_image(out_png, dataset_name, "No vectors in this support bin", dpi=args.dpi)
        print(f"[SAVED] {out_png} (placeholder)")
        return

    safe_k = max(1, min(int(args.K), int(flows.shape[0])))
    vfs.make_figure_for_dataset(
        flows=flows,
        dataset_name=dataset_name,
        out_path=str(out_png),
        H=args.height,
        W=args.width,
        K=safe_k,
        subsample=args.subsample,
        seed=args.seed,
        max_radius_px=args.max_radius_px,
        flow_bins=args.flow_bins,
        dpi=args.dpi,
        grid=args.grid,
        k_dir=args.k_dir,
        min_bin=args.min_bin,
        dir_base_sigma=args.dir_base_sigma,
        dir_max_sigma=args.dir_max_sigma,
        dir_mode=args.dir_mode,
        joint_xy_scale=args.joint_xy_scale,
        joint_flow_scale=args.joint_flow_scale,
        soft_edge=args.soft_edge,
        support_sigma=args.support_sigma,
        flow_range=args.flow_range,
        show_endpoint=(not args.no_endpoint),
        legend_side=args.legend_side,
    )


def _make_pair_montage(
    pair_name: str,
    image_dir: Path,
    dpi: int = 200,
) -> Optional[Path]:
    targets = {
        "Train outside eval": f"{pair_name}__train_out_eval_splat.png",
        "Train in eval": f"{pair_name}__train_in_eval_splat.png",
        "Eval in train": f"{pair_name}__eval_in_train_splat.png",
        "Eval outside train": f"{pair_name}__eval_out_train_splat.png",
    }
    imgs = {}
    for title, fname in targets.items():
        p = image_dir / fname
        if not p.exists():
            return None
        imgs[title] = plt.imread(str(p))

    sample = next(iter(imgs.values()))
    h, w = sample.shape[:2]
    aspect = float(w) / max(float(h), 1.0)
    fig_h = 7.0
    fig_w = fig_h * aspect
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(2, 2)
    fig.subplots_adjust(
        left=0.005,
        right=0.995,
        bottom=0.005,
        top=0.965,
        wspace=0.01,
        hspace=0.08,
    )

    titles = list(targets.keys())
    for i, title in enumerate(titles):
        r = i // 2
        c = i % 2
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(imgs[title])
        ax.set_axis_off()
        ax.set_title(title, fontsize=11.0, pad=2.0)

    out_path = image_dir / f"{pair_name}__montage.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def _dataset_tag(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_flow"):
        stem = stem[: -len("_flow")]
    return stem


def _default_pair_name(train_path: Path, eval_path: Path) -> str:
    return f"{_dataset_tag(train_path)}__{_dataset_tag(eval_path)}"


def _discover_pairs(
    vectors_dir: Path,
    train_regex: str,
    eval_regex: str,
    pair_regex: Optional[str],
    recursive: bool,
) -> List[Tuple[Path, Path, str]]:
    globber = vectors_dir.rglob if recursive else vectors_dir.glob
    files = sorted(globber("*_flow.npy"))
    train_re = re.compile(train_regex)
    eval_re = re.compile(eval_regex)
    pair_re = re.compile(pair_regex) if pair_regex else None

    train_files = [p for p in files if train_re.search(p.name)]
    eval_files = [p for p in files if eval_re.search(p.name)]

    pairs: List[Tuple[Path, Path, str]] = []
    for train_path in train_files:
        for eval_path in eval_files:
            pair_name = _default_pair_name(train_path, eval_path)
            if pair_re and not pair_re.search(pair_name):
                continue
            pairs.append((train_path, eval_path, pair_name))

    return pairs


def _process_pair(
    args: argparse.Namespace,
    train_path: Path,
    eval_path: Path,
    pair_name: str,
) -> Dict[str, str]:
    img_out_dir = Path(args.out_dir) / pair_name
    vec_out_dir = Path(args.flows_out_dir) / pair_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    vec_out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = img_out_dir / f"{pair_name}__summary.json"
    if summary_path.exists() and (not args.overwrite):
        print(f"[SKIP] {pair_name}: summary exists ({summary_path})")
        return {
            "pair_name": pair_name,
            "train_file": str(train_path),
            "eval_file": str(eval_path),
            "images_dir": str(img_out_dir),
            "vectors_dir": str(vec_out_dir),
            "summary_json": str(summary_path),
            "montage_png": str(img_out_dir / f"{pair_name}__montage.png"),
            "status": "skipped",
        }

    print(f"[PAIR] {pair_name}")
    print(f"[LOAD] train: {train_path}")
    train_flows = _load_flows(train_path, args.max_train_vectors, args.seed, "train")
    print(f"[LOAD] eval : {eval_path}")
    eval_flows = _load_flows(eval_path, args.max_eval_vectors, args.seed, "eval")
    print(f"[INFO] train vectors: {train_flows.shape[0]:,}")
    print(f"[INFO] eval vectors : {eval_flows.shape[0]:,}")

    if train_flows.shape[0] == 0 or eval_flows.shape[0] == 0:
        raise ValueError("Train/eval vectors must both be non-empty")

    train_space = _transform_vectors(
        train_flows, args.space, args.joint_alpha, args.flow_normalized, args.image_width, args.image_height
    )
    eval_space = _transform_vectors(
        eval_flows, args.space, args.joint_alpha, args.flow_normalized, args.image_width, args.image_height
    )

    train_eps = _eps_to_space_units(
        args.train_to_eval_eps_px, args.flow_normalized, args.image_width, args.image_height
    )
    eval_eps = _eps_to_space_units(
        args.eval_to_train_eps_px, args.flow_normalized, args.image_width, args.image_height
    )
    train_eps_sq = float(train_eps * train_eps)
    eval_eps_sq = float(eval_eps * eval_eps)

    print(f"[DIST] space={args.space}, joint_alpha={args.joint_alpha}")
    print(
        f"[DIST] thresholds: train→eval <= {train_eps:.6f} (sq={train_eps_sq:.6f}), "
        f"eval→train <= {eval_eps:.6f} (sq={eval_eps_sq:.6f})"
    )

    if args.exclude_cross_duplicates:
        print("[DIST] mode: nearest non-duplicate cross-neighbor (filter zero-distance matches)")
        train_index = None
        eval_index = None
        train_fallback = None
        eval_fallback = None
        try:
            train_index = faiss_ops.build_index(
                train_space,
                use_gpu=(not args.cpu),
                index_factory=args.index_factory,
                nprobe=args.nprobe,
                verbose=True,
            )
            if args.index_factory.lower() != "flat":
                train_fallback = faiss_ops.build_index(
                    train_space,
                    use_gpu=(not args.cpu),
                    index_factory="Flat",
                    verbose=False,
                )
            eval_to_train, _ = faiss_ops.compute_knn_distances(
                train_index,
                eval_space,
                k=1,
                exclude_self=True,
                filter_duplicates=True,
                fallback_index=train_fallback,
                batch_size=args.batch_size,
                verbose=True,
            )
            eval_to_train_sq = eval_to_train[:, 0]
        finally:
            faiss_ops.release_index(train_index)
            faiss_ops.release_index(train_fallback)

        try:
            eval_index = faiss_ops.build_index(
                eval_space,
                use_gpu=(not args.cpu),
                index_factory=args.index_factory,
                nprobe=args.nprobe,
                verbose=True,
            )
            if args.index_factory.lower() != "flat":
                eval_fallback = faiss_ops.build_index(
                    eval_space,
                    use_gpu=(not args.cpu),
                    index_factory="Flat",
                    verbose=False,
                )
            train_to_eval, _ = faiss_ops.compute_knn_distances(
                eval_index,
                train_space,
                k=1,
                exclude_self=True,
                filter_duplicates=True,
                fallback_index=eval_fallback,
                batch_size=args.batch_size,
                verbose=True,
            )
            train_to_eval_sq = train_to_eval[:, 0]
        finally:
            faiss_ops.release_index(eval_index)
            faiss_ops.release_index(eval_fallback)
    else:
        print("[DIST] mode: nearest cross-neighbor (duplicates allowed)")
        directed = faiss_ops.compute_directed_distances(
            train_space,
            eval_space,
            k=1,
            use_gpu=(not args.cpu),
            index_factory=args.index_factory,
            nprobe=args.nprobe,
            batch_size=args.batch_size,
            verbose=True,
        )
        eval_to_train_sq = directed["eval_to_train"][:, 0]
        train_to_eval_sq = directed["train_to_eval"][:, 0]

    eval_valid = np.isfinite(eval_to_train_sq)
    train_valid = np.isfinite(train_to_eval_sq)

    eval_in_mask = eval_valid & (eval_to_train_sq <= eval_eps_sq)
    eval_out_mask = eval_valid & (eval_to_train_sq > eval_eps_sq)
    eval_missing_mask = ~eval_valid

    train_in_mask = train_valid & (train_to_eval_sq <= train_eps_sq)
    train_out_mask = train_valid & (train_to_eval_sq > train_eps_sq)
    train_missing_mask = ~train_valid

    bins = {
        "train_in_eval": (train_flows[train_in_mask], train_to_eval_sq[train_in_mask], train_eps, train_eps_sq),
        "train_out_eval": (train_flows[train_out_mask], train_to_eval_sq[train_out_mask], train_eps, train_eps_sq),
        "eval_in_train": (eval_flows[eval_in_mask], eval_to_train_sq[eval_in_mask], eval_eps, eval_eps_sq),
        "eval_out_train": (eval_flows[eval_out_mask], eval_to_train_sq[eval_out_mask], eval_eps, eval_eps_sq),
    }

    for bin_name, (bin_flows, bin_d2, thr, thr_sq) in bins.items():
        tag = f"{pair_name}__{bin_name}"
        out_npz = vec_out_dir / f"{tag}.npz"
        out_png = img_out_dir / f"{tag}_splat.png"
        _save_bin_npz(
            out_npz,
            flows=bin_flows,
            dist_sq=bin_d2,
            threshold=thr,
            threshold_sq=thr_sq,
            space_name=args.space,
            joint_alpha=args.joint_alpha,
        )
        _render_bin(bin_flows, tag, out_png, args)

    summary = {
        "pair_name": pair_name,
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "space": args.space,
        "joint_alpha": float(args.joint_alpha),
        "flow_normalized": bool(args.flow_normalized),
        "image_width": int(args.image_width),
        "image_height": int(args.image_height),
        "thresholds": {
            "train_to_eval_eps_px": float(args.train_to_eval_eps_px),
            "eval_to_train_eps_px": float(args.eval_to_train_eps_px),
            "train_to_eval_eps_space": float(train_eps),
            "eval_to_train_eps_space": float(eval_eps),
            "train_to_eval_eps_sq": float(train_eps_sq),
            "eval_to_train_eps_sq": float(eval_eps_sq),
        },
        "counts": {
            "train_total": int(train_flows.shape[0]),
            "eval_total": int(eval_flows.shape[0]),
            "train_in_eval": int(train_in_mask.sum()),
            "train_out_eval": int(train_out_mask.sum()),
            "train_missing": int(train_missing_mask.sum()),
            "eval_in_train": int(eval_in_mask.sum()),
            "eval_out_train": int(eval_out_mask.sum()),
            "eval_missing": int(eval_missing_mask.sum()),
        },
        "fractions": {
            "train_in_eval": _safe_frac(int(train_in_mask.sum()), int(train_flows.shape[0])),
            "train_out_eval": _safe_frac(int(train_out_mask.sum()), int(train_flows.shape[0])),
            "eval_in_train": _safe_frac(int(eval_in_mask.sum()), int(eval_flows.shape[0])),
            "eval_out_train": _safe_frac(int(eval_out_mask.sum()), int(eval_flows.shape[0])),
        },
        "distance_stats": {
            "train_to_eval": _dist_stats(train_to_eval_sq),
            "eval_to_train": _dist_stats(eval_to_train_sq),
        },
        "exclude_cross_duplicates": bool(args.exclude_cross_duplicates),
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {summary_path}")

    montage_path = None
    if not args.no_montage:
        montage_path = _make_pair_montage(pair_name=pair_name, image_dir=img_out_dir, dpi=args.montage_dpi)
        if montage_path is not None:
            print(f"[SAVED] {montage_path}")

    return {
        "pair_name": pair_name,
        "train_file": str(train_path),
        "eval_file": str(eval_path),
        "images_dir": str(img_out_dir),
        "vectors_dir": str(vec_out_dir),
        "summary_json": str(summary_path),
        "montage_png": str(montage_path) if montage_path is not None else "",
        "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Joint-flow support splats from raw directional nearest-neighbor thresholds"
    )
    ap.add_argument("--train", default=None, help="Train flow vectors file (.npz/.npy/.pt/.pkl)")
    ap.add_argument("--eval", default=None, help="Eval flow vectors file (.npz/.npy/.pt/.pkl)")
    ap.add_argument("--vectors-dir", default=None, help="Directory containing *_flow.npy vectors for batch mode")
    ap.add_argument("--recursive", action="store_true", help="Recursive discovery under --vectors-dir")
    ap.add_argument("--train-regex", default=r".*_train_flow\.npy$", help="Regex for train vector filenames")
    ap.add_argument("--eval-regex", default=r".*_(test|val)_flow\.npy$", help="Regex for eval vector filenames")
    ap.add_argument("--pair-regex", default=None, help="Optional regex filter on discovered pair name")
    ap.add_argument("--max-pairs", type=int, default=None, help="Optional cap for discovered pairs")
    ap.add_argument("--dry-run", action="store_true", help="List discovered pairs without processing")
    ap.add_argument("--overwrite", action="store_true", help="Recompute pairs even if summary exists")
    ap.add_argument("--run-label", default=None, help="Optional label written into run metadata")
    ap.add_argument("--pair-name", default=None, help="Optional pair label (defaults to train_stem__eval_stem)")
    ap.add_argument("--out-dir", default="gaussian_splat/output_joint_flow_splats", help="Output image directory")
    ap.add_argument(
        "--flows-out-dir",
        default="gaussian_splat/joint_flow_support_vectors",
        help="Output support-bin vector NPZ directory",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--max-train-vectors", type=int, default=None, help="Optional cap on train vectors")
    ap.add_argument("--max-eval-vectors", type=int, default=None, help="Optional cap on eval vectors")

    # Threshold semantics (raw distance thresholds in selected space units).
    ap.add_argument("--train-to-eval-eps-px", type=float, default=1.0, help="Threshold for train→eval support")
    ap.add_argument("--eval-to-train-eps-px", type=float, default=1.5, help="Threshold for eval→train support")

    # Distance-space configuration.
    ap.add_argument("--space", choices=["xy", "flow", "joint"], default="joint", help="Distance space")
    ap.add_argument("--joint-alpha", type=float, default=1.0, help="Scale factor on dx,dy in joint space")
    ap.add_argument(
        "--flow-normalized",
        action="store_true",
        help="Treat vectors in normalized flow space and convert eps(px) accordingly",
    )
    ap.add_argument("--image-width", type=int, default=512, help="Image width for flow normalization conversion")
    ap.add_argument("--image-height", type=int, default=512, help="Image height for flow normalization conversion")

    # FAISS options.
    ap.add_argument("--cpu", action="store_true", help="Force CPU FAISS")
    ap.add_argument("--index-factory", default="Flat", help="FAISS index factory string")
    ap.add_argument("--nprobe", type=int, default=None, help="FAISS nprobe for IVF indexes")
    ap.add_argument("--batch-size", type=int, default=None, help="FAISS search batch size")
    ap.add_argument(
        "--exclude-cross-duplicates",
        action="store_true",
        help="Use nearest non-duplicate cross-neighbor (skip zero-distance matches).",
    )
    ap.add_argument(
        "--allow-cross-duplicates",
        action="store_true",
        help="Allow zero-distance cross matches (disables --exclude-cross-duplicates).",
    )

    # Renderer options (forwarded to visualize_flow_splats.make_figure_for_dataset).
    ap.add_argument("--height", type=int, default=None, help="Output height (infer if omitted)")
    ap.add_argument("--width", type=int, default=None, help="Output width (infer if omitted)")
    ap.add_argument("--K", type=int, default=800)
    ap.add_argument("--subsample", type=int, default=2_000_000)
    ap.add_argument("--max_radius_px", type=int, default=64)
    ap.add_argument("--soft_edge", type=float, default=0.15)
    ap.add_argument("--support_sigma", type=float, default=3.0)
    ap.add_argument("--flow_range", type=float, default=None)
    ap.add_argument("--flow_bins", type=int, default=512)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--k_dir", type=int, default=3)
    ap.add_argument("--min_bin", type=int, default=20)
    ap.add_argument("--dir_base_sigma", type=float, default=3.0)
    ap.add_argument("--dir_max_sigma", type=float, default=48.0)
    ap.add_argument("--dir_mode", choices=["grid", "cluster", "joint"], default="joint")
    ap.add_argument("--joint_xy_scale", type=float, default=1.0)
    ap.add_argument("--joint_flow_scale", type=float, default=1.5)
    ap.add_argument("--no-endpoint", action="store_true")
    ap.add_argument("--legend-side", choices=["inside", "left", "right"], default="inside")
    ap.add_argument("--no-montage", action="store_true", help="Disable writing per-pair 2x2 montage PNG")
    ap.add_argument("--montage-dpi", type=int, default=200, help="DPI for montage image")
    args = ap.parse_args()
    if args.allow_cross_duplicates:
        args.exclude_cross_duplicates = False
    elif not args.exclude_cross_duplicates:
        # Default behavior: filter cross-set duplicates.
        args.exclude_cross_duplicates = True

    pairs: List[Tuple[Path, Path, str]]
    if args.vectors_dir:
        vectors_dir = Path(args.vectors_dir)
        if not vectors_dir.exists():
            raise FileNotFoundError(f"Missing vectors directory: {vectors_dir}")
        pairs = _discover_pairs(
            vectors_dir=vectors_dir,
            train_regex=args.train_regex,
            eval_regex=args.eval_regex,
            pair_regex=args.pair_regex,
            recursive=args.recursive,
        )
        if args.max_pairs is not None:
            pairs = pairs[: max(0, int(args.max_pairs))]
        print(f"[DISCOVER] vectors_dir={vectors_dir}")
        print(f"[DISCOVER] train files regex: {args.train_regex}")
        print(f"[DISCOVER] eval files regex : {args.eval_regex}")
        print(f"[DISCOVER] discovered pairs: {len(pairs)}")
        if args.dry_run:
            for i, (_, _, pair_name) in enumerate(pairs, start=1):
                print(f"  {i:03d}: {pair_name}")
            print("[DONE] dry-run")
            return
    else:
        if not args.train or not args.eval:
            raise ValueError("Provide --train and --eval, or use --vectors-dir for batch discovery")
        train_path = Path(args.train)
        eval_path = Path(args.eval)
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train vectors file: {train_path}")
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing eval vectors file: {eval_path}")
        pair_name = args.pair_name or _default_pair_name(train_path, eval_path)
        pairs = [(train_path, eval_path, pair_name)]

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "run_label": args.run_label or out_root.name,
        "out_dir": str(out_root),
        "flows_out_dir": str(Path(args.flows_out_dir)),
        "space": args.space,
        "joint_alpha": float(args.joint_alpha),
        "flow_normalized": bool(args.flow_normalized),
        "train_to_eval_eps_px": float(args.train_to_eval_eps_px),
        "eval_to_train_eps_px": float(args.eval_to_train_eps_px),
        "train_regex": args.train_regex,
        "eval_regex": args.eval_regex,
        "pair_regex": args.pair_regex,
        "n_pairs": int(len(pairs)),
        "exclude_cross_duplicates": bool(args.exclude_cross_duplicates),
    }
    run_meta_path = out_root / "_RUN_INFO.json"
    with run_meta_path.open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    print(f"[SAVED] {run_meta_path}")

    n_ok = 0
    n_fail = 0
    records: List[Dict[str, str]] = []
    for i, (train_path, eval_path, pair_name) in enumerate(pairs, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{i}/{len(pairs)}] {pair_name}")
        print(f"{'=' * 80}")
        try:
            record = _process_pair(args, train_path, eval_path, pair_name)
            records.append(record)
            if record.get("status") in {"ok", "skipped"}:
                n_ok += 1
            else:
                n_fail += 1
        except Exception as exc:
            n_fail += 1
            print(f"[ERROR] {pair_name}: {exc}")
            records.append(
                {
                    "pair_name": pair_name,
                    "train_file": str(train_path),
                    "eval_file": str(eval_path),
                    "images_dir": str(Path(args.out_dir) / pair_name),
                    "vectors_dir": str(Path(args.flows_out_dir) / pair_name),
                    "summary_json": "",
                    "montage_png": "",
                    "status": f"error: {exc}",
                }
            )
            continue

    index_path = out_root / "_PAIR_INDEX.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_name",
                "train_file",
                "eval_file",
                "images_dir",
                "vectors_dir",
                "summary_json",
                "montage_png",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"[SAVED] {index_path}")

    print("\n[DONE] Joint-flow support bin visualizations complete.")
    print(f"  processed: {len(pairs)}")
    print(f"  success  : {n_ok}")
    print(f"  failed   : {n_fail}")


if __name__ == "__main__":
    main()
