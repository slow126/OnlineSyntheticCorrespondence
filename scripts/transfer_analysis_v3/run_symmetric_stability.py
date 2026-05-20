#!/usr/bin/env python3
"""
Subsampling stability for symmetric distributional distance metrics (FID, SW2).

The full-data values in symmetric_distances.csv were computed at:
  FID:  200K samples per dataset
  SW2:  100K samples per dataset

This script recomputes both metrics at smaller caps and measures how stable
the pair-wise rankings are (Spearman ρ vs the full-cap reference).

Usage:
    python scripts/transfer_analysis_v3/run_symmetric_stability.py \
        [--sym-csv analysis_v3/symmetric_distances.csv] \
        [--vec-dir /mnt/nvme_1tb_b/coverage_vectors] \
        [--output-dir scripts/transfer_analysis_v3/results/subsampling_stability] \
        [--caps 10000 25000 50000 100000] \
        [--n-proj 200] [--seed 42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import spearmanr
from tqdm import tqdm

try:
    import torch
    _TORCH = torch.cuda.is_available()
except ImportError:
    _TORCH = False


# ---------------------------------------------------------------------------
# Math — lifted from compute_symmetric_distances.py
# ---------------------------------------------------------------------------

def fid_from_vecs(va: np.ndarray, vb: np.ndarray, eps: float = 1e-6) -> float:
    m1, c1 = va.mean(0), np.cov(va.T)
    m2, c2 = vb.mean(0), np.cov(vb.T)
    d = len(m1)
    c1r, c2r = c1 + eps * np.eye(d), c2 + eps * np.eye(d)
    covmean, _ = linalg.sqrtm(c1r @ c2r, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = m1 - m2
    return float(diff @ diff + np.trace(c1r + c2r - 2.0 * covmean))


def sliced_w2(va: np.ndarray, vb: np.ndarray, projs: np.ndarray) -> float:
    if _TORCH:
        device = torch.device("cuda")
        ta = torch.from_numpy(va).to(device)
        tb = torch.from_numpy(vb).to(device)
        tp = torch.from_numpy(projs.T).to(device)
        pa = torch.sort(ta @ tp, dim=0).values
        pb = torch.sort(tb @ tp, dim=0).values
        na, nb = pa.shape[0], pb.shape[0]
        if na != nb:
            n = max(na, nb)
            ia = torch.linspace(0, na - 1, n, device=device).long().clamp(0, na - 1)
            ib = torch.linspace(0, nb - 1, n, device=device).long().clamp(0, nb - 1)
            pa, pb = pa[ia], pb[ib]
        return float(((pa - pb) ** 2).mean(0).mean().sqrt().cpu())
    pa = np.sort(va @ projs.T, axis=0)
    pb = np.sort(vb @ projs.T, axis=0)
    na, nb = pa.shape[0], pb.shape[0]
    if na != nb:
        n = max(na, nb)
        ia = np.round(np.linspace(0, na - 1, n)).astype(int).clip(0, na - 1)
        ib = np.round(np.linspace(0, nb - 1, n)).astype(int).clip(0, nb - 1)
        pa, pb = pa[ia], pb[ib]
    return float(np.sqrt(np.mean((pa - pb) ** 2, axis=0).mean()))


def load_vecs(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.array(np.load(path, mmap_mode="r"), dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sym-csv",    default="analysis_v3/symmetric_distances.csv")
    parser.add_argument("--vec-dir",    default="/mnt/nvme_1tb_b/coverage_vectors")
    parser.add_argument("--output-dir", default="scripts/transfer_analysis_v3/results/subsampling_stability")
    parser.add_argument("--caps", nargs="+", type=int, default=[10_000, 25_000, 50_000, 100_000],
                        help="Subsample caps to test (compared against full-cap reference).")
    parser.add_argument("--n-proj", type=int, default=200, help="Random projections for SW2.")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    vec_dir = Path(args.vec_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.sym_csv)
    print(f"Reference pairs: {len(ref)}")

    metrics = ["flow_fid", "flow_sliced_w2", "dino_fid", "dino_sliced_w2"]
    for m in metrics:
        if m not in ref.columns:
            print(f"WARNING: {m} not in {args.sym_csv} — skipping")
    ref = ref.dropna(subset=[m for m in metrics if m in ref.columns])
    print(f"Pairs with all reference metrics: {len(ref)}")

    # Precompute projections for SW2 (use flow dim=4, DINO dim=256 — detect from data).
    flow_projs = dino_projs = None

    pair_rows = []
    for _, row in tqdm(ref.iterrows(), total=len(ref), desc="pairs"):
        t_ds, t_sp = row["train_dataset"], row.get("train_split", "train")
        e_ds, e_sp = row["eval_dataset"],  row.get("eval_split",  "test")

        flow_t = load_vecs(vec_dir / f"{t_ds}_{t_sp}_flow.npy")
        flow_e = load_vecs(vec_dir / f"{e_ds}_{e_sp}_flow.npy")
        dino_t = load_vecs(vec_dir / f"{t_ds}_{t_sp}_dino_pca256_l2norm.npy")
        dino_e = load_vecs(vec_dir / f"{e_ds}_{e_sp}_dino_pca256_l2norm.npy")

        if flow_t is None or flow_e is None or dino_t is None or dino_e is None:
            tqdm.write(f"  MISSING vecs for {t_ds}/{e_ds} — skipping")
            continue

        # Lazy-init projections once we know the dims.
        if flow_projs is None:
            flow_projs = rng.standard_normal((args.n_proj, flow_t.shape[1])).astype(np.float32)
            flow_projs /= np.linalg.norm(flow_projs, axis=1, keepdims=True)
        if dino_projs is None:
            dino_projs = rng.standard_normal((args.n_proj, dino_t.shape[1])).astype(np.float32)
            dino_projs /= np.linalg.norm(dino_projs, axis=1, keepdims=True)

        for cap in args.caps:
            n_flow_t = min(cap, len(flow_t))
            n_flow_e = min(cap, len(flow_e))
            n_dino_t = min(cap, len(dino_t))
            n_dino_e = min(cap, len(dino_e))

            ft = flow_t[rng.choice(len(flow_t), n_flow_t, replace=False)]
            fe = flow_e[rng.choice(len(flow_e), n_flow_e, replace=False)]
            dt = dino_t[rng.choice(len(dino_t), n_dino_t, replace=False)]
            de = dino_e[rng.choice(len(dino_e), n_dino_e, replace=False)]

            try:
                pair_rows.append({
                    "train_dataset": t_ds, "eval_dataset": e_ds, "cap": cap,
                    "cap_label": f"{cap//1000}K",
                    "flow_fid":       fid_from_vecs(ft, fe),
                    "flow_sliced_w2": sliced_w2(ft, fe, flow_projs),
                    "dino_fid":       fid_from_vecs(dt, de),
                    "dino_sliced_w2": sliced_w2(dt, de, dino_projs),
                })
            except Exception as exc:
                tqdm.write(f"  ERROR {t_ds}/{e_ds} cap={cap}: {exc}")

    if not pair_rows:
        print("No pairs computed — check vector paths.")
        sys.exit(1)

    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(out_dir / "pair_metrics_symmetric.csv", index=False)
    print(f"Saved {len(pair_df)} pair-cap rows.")

    # Merge with reference and compute Spearman per (metric, cap).
    ref_slim = ref[["train_dataset", "eval_dataset"] + metrics].rename(
        columns={m: f"{m}_ref" for m in metrics}
    )
    merged = pair_df.merge(ref_slim, on=["train_dataset", "eval_dataset"], how="inner")

    stab_rows = []
    caps_tested = sorted(pair_df["cap"].unique())
    for metric in metrics:
        if f"{metric}_ref" not in merged.columns:
            continue
        for cap in caps_tested:
            sub = merged[merged["cap"] == cap].dropna(subset=[metric, f"{metric}_ref"])
            if len(sub) < 3:
                continue
            rho, _ = spearmanr(sub[metric], sub[f"{metric}_ref"])
            stab_rows.append({
                "metric":    metric,
                "cap":       cap,
                "cap_label": f"{cap//1000}K",
                "spearman":  rho,
                "n_pairs":   len(sub),
            })

    stab = pd.DataFrame(stab_rows)
    out_path = out_dir / "stability_symmetric.csv"
    stab.to_csv(out_path, index=False)
    print(f"\nStability summary saved to {out_path}")
    print(stab.pivot_table(index="metric", columns="cap_label", values="spearman").to_string())


if __name__ == "__main__":
    main()
