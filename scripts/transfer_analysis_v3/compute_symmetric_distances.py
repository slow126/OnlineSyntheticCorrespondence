#!/usr/bin/env python3
"""
Compute symmetric distribution-level distances for all (train_dataset, eval_dataset)
pairs found in the flow coverage CSV.

Metrics:
  flow_fid          - Fréchet distance in 4D flow space
  flow_sliced_w2    - Sliced Wasserstein-2 in 4D flow space
  dino_fid          - Fréchet distance in DINO PCA-256 space
  dino_sliced_w2    - Sliced Wasserstein-2 in DINO PCA-256 space

MEMORY STRATEGY — two passes, controlled peak usage:
  Pass 1 (FID): Load each unique dataset once, compute (mean, cov), store only stats
                (~256 KB per dataset → ~10 MB total for 34 datasets). Vectors freed immediately.
  Pass 2 (SW2): For each pair, load first N vectors from each file via mmap sequentially.
                Peak memory = 2 × N × d × 4 bytes ≈ 100 MB. NVMe page cache makes
                repeated reads of the same file fast after the first load.

Usage:
    python scripts/transfer_analysis_v3/compute_symmetric_distances.py \
        [--flow-csv analysis/coverage_v2_flow_only_raw_joint_full.csv] \
        [--vec-dir /mnt/nvme_1tb_b/coverage_vectors] \
        [--output analysis_v3/symmetric_distances.csv] \
        [--n-proj 200] [--sw-samples 100000] [--fid-samples 200000] \
        [--seed 42] [--skip-dino] [--skip-flow]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from tqdm import tqdm

try:
    import torch
    _TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def fid_from_stats(stats_a: dict, stats_b: dict, eps: float = 1e-6) -> float:
    m1, c1 = stats_a["mean"], stats_a["cov"]
    m2, c2 = stats_b["mean"], stats_b["cov"]
    d = len(m1)
    c1r = c1 + eps * np.eye(d)
    c2r = c2 + eps * np.eye(d)
    covmean, _ = linalg.sqrtm(c1r @ c2r, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = m1 - m2
    return float(diff @ diff + np.trace(c1r + c2r - 2.0 * covmean))


def sliced_w2(v_a: np.ndarray, v_b: np.ndarray, projs: np.ndarray) -> float:
    """projs: (n_proj, d) precomputed unit vectors. Returns W2 (not W2^2).

    GPU path (torch): matmul + sort on CUDA, ~10x faster than CPU for 256-D.
    CPU path: fully vectorized sort across all projections at once.
    Both paths handle unequal sample sizes via quantile interpolation.
    """
    if _TORCH_AVAILABLE:
        return _sliced_w2_gpu(v_a, v_b, projs)
    return _sliced_w2_cpu(v_a, v_b, projs)


def _sliced_w2_gpu(v_a: np.ndarray, v_b: np.ndarray, projs: np.ndarray) -> float:
    device = torch.device("cuda")
    ta = torch.from_numpy(v_a).to(device)          # (n_a, d)
    tb = torch.from_numpy(v_b).to(device)          # (n_b, d)
    tp = torch.from_numpy(projs.T).to(device)      # (d, n_proj)
    pa = torch.sort(ta @ tp, dim=0).values         # (n_a, n_proj)
    pb = torch.sort(tb @ tp, dim=0).values         # (n_b, n_proj)
    na, nb = pa.shape[0], pb.shape[0]
    if na != nb:
        # Interpolate shorter to longer via linspace index mapping
        n = max(na, nb)
        idx_a = torch.linspace(0, na - 1, n, device=device).long().clamp(0, na - 1)
        idx_b = torch.linspace(0, nb - 1, n, device=device).long().clamp(0, nb - 1)
        pa = pa[idx_a]
        pb = pb[idx_b]
    sw_sq = ((pa - pb) ** 2).mean(dim=0).mean()
    return float(sw_sq.sqrt().cpu())


def _sliced_w2_cpu(v_a: np.ndarray, v_b: np.ndarray, projs: np.ndarray) -> float:
    pa = np.sort(v_a @ projs.T, axis=0)   # (n_a, n_proj) — one call sorts all projections
    pb = np.sort(v_b @ projs.T, axis=0)   # (n_b, n_proj)
    na, nb = pa.shape[0], pb.shape[0]
    if na != nb:
        n = max(na, nb)
        idx_a = np.round(np.linspace(0, na - 1, n)).astype(int).clip(0, na - 1)
        idx_b = np.round(np.linspace(0, nb - 1, n)).astype(int).clip(0, nb - 1)
        pa = pa[idx_a]
        pb = pb[idx_b]
    sw_sq = np.mean((pa - pb) ** 2, axis=0).mean()
    return float(np.sqrt(sw_sq))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _vec_path(vec_dir: Path, dataset: str, split: str, rep: str) -> Path:
    suffix = "_flow.npy" if rep == "flow" else "_dino_pca256_l2norm.npy"
    return vec_dir / f"{dataset}_{split}{suffix}"


def load_random_n(path: Path, n: int, seed: int = 0) -> np.ndarray | None:
    """Random subsample of n rows — unbiased regardless of on-disk ordering.
    Indices sorted before mmap access for sequential I/O efficiency."""
    if not path.exists():
        return None
    v = np.load(path, mmap_mode="r")
    if len(v) <= n:
        return np.array(v, dtype=np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(v), n, replace=False)
    idx.sort()
    return np.array(v[idx], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pass 1: compute FID stats (mean + cov) per dataset, free vectors immediately
# ---------------------------------------------------------------------------

def compute_all_fid_stats(
    unique_keys: list[tuple[str, str]],
    vec_dir: Path,
    rep: str,
    fid_samples: int,
    desc: str = "FID stats",
) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    bar = tqdm(unique_keys, desc=desc, unit="dataset", dynamic_ncols=True)
    for dataset, split in bar:
        key = f"{dataset}_{split}"
        bar.set_postfix_str(f"{dataset}/{split}", refresh=False)
        path = _vec_path(vec_dir, dataset, split, rep)
        t0 = time.time()
        seed = int(hash(path.stem) % (2**31))
        v = load_random_n(path, fid_samples, seed=seed)
        if v is None:
            tqdm.write(f"  MISSING: {path.name}")
            continue
        load_ms = (time.time() - t0) * 1000
        t1 = time.time()
        if _TORCH_AVAILABLE:
            t = torch.from_numpy(v).cuda().double()
            mean = t.mean(0)
            t -= mean
            cov = (t.T @ t) / (len(t) - 1)
            stats[key] = {"mean": mean.cpu().numpy(), "cov": cov.cpu().numpy()}
            del t
        else:
            vd = v.astype(np.float64)
            d = vd.shape[1]
            mean = vd.mean(0)
            cov = np.cov(vd.T) if len(vd) > 1 else np.zeros((d, d))
            stats[key] = {"mean": mean, "cov": cov}
            del vd
        compute_ms = (time.time() - t1) * 1000
        del v
        bar.set_postfix(load=f"{load_ms:.0f}ms", cov=f"{compute_ms:.0f}ms", refresh=False)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-csv",    default="analysis/coverage_v2_flow_only_raw_joint_full.csv")
    parser.add_argument("--vec-dir",     default="/mnt/nvme_1tb_b/coverage_vectors")
    parser.add_argument("--output",      default="analysis_v3/symmetric_distances.csv")
    parser.add_argument("--n-proj",      type=int, default=200)
    parser.add_argument("--sw-samples",  type=int, default=100_000,
        help="Vectors per dataset for sliced Wasserstein (loaded pair-by-pair).")
    parser.add_argument("--fid-samples", type=int, default=200_000,
        help="Vectors per dataset for FID covariance estimation.")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--skip-dino",   action="store_true")
    parser.add_argument("--skip-flow",   action="store_true")
    args = parser.parse_args()

    vec_dir  = Path(args.vec_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_df = pd.read_csv(args.flow_csv)[
        ["train_dataset", "train_split", "eval_dataset", "eval_split"]
    ].drop_duplicates().reset_index(drop=True)
    print(f"Total pairs: {len(pairs_df)}")

    # Resumability
    done: set[tuple] = set()
    if out_path.exists():
        for _, r in pd.read_csv(out_path).iterrows():
            done.add((r["train_dataset"], r["train_split"], r["eval_dataset"], r["eval_split"]))
        print(f"Already done: {len(done)}, remaining: {len(pairs_df) - len(done)}")
    todo = pairs_df[~pairs_df.apply(
        lambda r: (r["train_dataset"], r["train_split"], r["eval_dataset"], r["eval_split"]) in done,
        axis=1,
    )].reset_index(drop=True)
    if todo.empty:
        print("All pairs already computed.")
        return

    rng = np.random.default_rng(args.seed)

    unique_keys = list({
        (row["train_dataset"], row["train_split"])
        for _, row in todo.iterrows()
    } | {
        (row["eval_dataset"], row["eval_split"])
        for _, row in todo.iterrows()
    })

    # -----------------------------------------------------------------------
    # Pass 1: FID stats (low memory — vectors freed after each dataset)
    # -----------------------------------------------------------------------
    flow_fid_stats: dict[str, dict] = {}
    dino_fid_stats: dict[str, dict] = {}

    device_str = "GPU" if _TORCH_AVAILABLE else "CPU"
    if not args.skip_flow:
        print(f"\n--- Pass 1: flow FID stats ({len(unique_keys)} datasets, {device_str}) ---")
        flow_fid_stats = compute_all_fid_stats(
            unique_keys, vec_dir, "flow", args.fid_samples, desc="flow cov")

    if not args.skip_dino:
        print(f"\n--- Pass 1: DINO FID stats ({len(unique_keys)} datasets, {device_str}) ---")
        dino_fid_stats = compute_all_fid_stats(
            unique_keys, vec_dir, "dino", args.fid_samples, desc="dino cov")

    # -----------------------------------------------------------------------
    # Pass 2: FID pair computation (from tiny cached stats) + SW2 pair-by-pair
    # Peak memory: ~2 × sw_samples × d × 4 bytes ≈ 100 MB
    # -----------------------------------------------------------------------
    flow_projs = rng.standard_normal((args.n_proj, 4)).astype(np.float32)
    flow_projs /= np.linalg.norm(flow_projs, axis=1, keepdims=True)
    dino_projs = rng.standard_normal((args.n_proj, 256)).astype(np.float32)
    dino_projs /= np.linalg.norm(dino_projs, axis=1, keepdims=True)

    print(f"\n--- Pass 2: pair distances ({len(todo)} pairs, SW2 on {device_str}) ---")
    t_start = time.time()
    rows = []
    bar = tqdm(todo.iterrows(), total=len(todo), desc="pairs", unit="pair", dynamic_ncols=True)
    for _, pair in bar:
        td, ts, ed, es = (pair["train_dataset"], pair["train_split"],
                          pair["eval_dataset"],  pair["eval_split"])
        kt, ke = f"{td}_{ts}", f"{ed}_{es}"
        bar.set_postfix_str(f"{td} → {ed}", refresh=False)
        row: dict = {
            "train_dataset": td, "train_split": ts,
            "eval_dataset":  ed, "eval_split":  es,
        }
        timings: dict[str, str] = {}

        # --- FID (from cached stats, near-zero time) ---
        if not args.skip_flow:
            if kt in flow_fid_stats and ke in flow_fid_stats:
                row["flow_fid"] = fid_from_stats(flow_fid_stats[kt], flow_fid_stats[ke])
            else:
                row["flow_fid"] = float("nan")

        if not args.skip_dino:
            if kt in dino_fid_stats and ke in dino_fid_stats:
                row["dino_fid"] = fid_from_stats(dino_fid_stats[kt], dino_fid_stats[ke])
            else:
                row["dino_fid"] = float("nan")

        # --- SW2 (load pair, compute, discard) ---
        if not args.skip_flow:
            t0 = time.time()
            pt, pe = _vec_path(vec_dir, td, ts, "flow"), _vec_path(vec_dir, ed, es, "flow")
            vt = load_random_n(pt, args.sw_samples, seed=int(hash(pt.stem) % (2**31)))
            ve = load_random_n(pe, args.sw_samples, seed=int(hash(pe.stem) % (2**31)))
            load_ms = (time.time() - t0) * 1000
            t0 = time.time()
            if vt is not None and ve is not None:
                row["flow_sliced_w2"] = sliced_w2(vt, ve, flow_projs)
            else:
                row["flow_sliced_w2"] = float("nan")
            timings["flow"] = f"{load_ms:.0f}+{(time.time()-t0)*1000:.0f}ms"
            del vt, ve

        if not args.skip_dino:
            t0 = time.time()
            pt, pe = _vec_path(vec_dir, td, ts, "dino"), _vec_path(vec_dir, ed, es, "dino")
            vt = load_random_n(pt, args.sw_samples, seed=int(hash(pt.stem) % (2**31)))
            ve = load_random_n(pe, args.sw_samples, seed=int(hash(pe.stem) % (2**31)))
            load_ms = (time.time() - t0) * 1000
            t0 = time.time()
            if vt is not None and ve is not None:
                row["dino_sliced_w2"] = sliced_w2(vt, ve, dino_projs)
            else:
                row["dino_sliced_w2"] = float("nan")
            timings["dino"] = f"{load_ms:.0f}+{(time.time()-t0)*1000:.0f}ms"
            del vt, ve

        bar.set_postfix(timings, refresh=False)
        rows.append(row)

        # Flush every 20 pairs so progress is saved on interrupt
        if len(rows) >= 20:
            batch = pd.DataFrame(rows)
            write_header = not out_path.exists() or out_path.stat().st_size == 0
            batch.to_csv(out_path, mode="a", header=write_header, index=False)
            rows = []

    if rows:
        batch = pd.DataFrame(rows)
        write_header = not out_path.exists() or out_path.stat().st_size == 0
        batch.to_csv(out_path, mode="a", header=write_header, index=False)

    final = pd.read_csv(out_path)
    print(f"\n✓ Done in {time.time()-t_start:.0f}s. {len(final)} total pairs in {out_path}")
    for col in ["flow_fid", "flow_sliced_w2", "dino_fid", "dino_sliced_w2"]:
        if col in final.columns:
            v = final[col].dropna()
            if len(v):
                print(f"  {col}: {len(v)}/{len(final)} valid, "
                      f"range [{v.min():.4f}, {v.max():.4f}], median {v.median():.4f}")


if __name__ == "__main__":
    main()
