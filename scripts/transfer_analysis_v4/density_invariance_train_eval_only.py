"""Density invariance for the interventional-study scenario specifically:
test only the (train_dataset → benchmark) cross-distance features, not the
train_train or eval_eval pairs that the predictor's L term needs.

For the interventional study, the candidate dataset's (i → k) features are
the only NEW features we have to compute per candidate. The benchmark-
benchmark distances stay cached. So train_eval stability is the relevant
question for "how many frames does my candidate need?"

This script:
  1. Calls `compute_pairwise_self_distances.py` at the requested N levels.
  2. Filters output to train_eval pairs only.
  3. Reports stability vs the full-size baseline.

Compared to `density_invariance.py`, this is ~3× faster because we
post-filter to train_eval pairs (but the underlying script still computes
all pair types — see TODO below for a true train_eval-only optimization).

Run:
    python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
        --space flow --levels 50000 200000 1000000

    python scripts/transfer_analysis_v4/density_invariance_train_eval_only.py \
        --space dino --levels 10000 50000 200000

TODO: if even faster is needed, patch compute_pairwise_self_distances.py
to accept --pair-types train_eval and skip the train_train / eval_eval
enumeration entirely. That gives the full 3× speedup.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SELFDIST_METRIC_COLS = [
    "mean_nn_a_to_b", "mean_nn_b_to_a", "mean_nn_sym",
    "a_covered_by_b_eps1px", "b_covered_by_a_eps1px",
    "a_covered_by_b_eps4px", "b_covered_by_a_eps4px",
    "a_covered_by_b_eps16px", "b_covered_by_a_eps16px",
    "kl_a_to_b_k5", "kl_b_to_a_k5",
    "kl_a_to_b_k20", "kl_b_to_a_k20",
]


def run_one_level(space: str, level: int, vec_dir: Path,
                  output_csv: Path, gpu: bool) -> None:
    if output_csv.exists():
        print(f"  skip (exists): {output_csv}")
        return
    cmd = [
        sys.executable,
        "scripts/transfer_analysis_v3/compute_pairwise_self_distances.py",
        "--vec-dir", str(vec_dir),
        "--output", str(output_csv),
        "--spaces", space,
        f"--max-{space}", str(level),
    ]
    if gpu:
        cmd.append("--gpu")
    print(f"  [{space}, N={level}] running...")
    subprocess.run(cmd, check=True)
    print(f"  [{space}, N={level}] done → {output_csv}")


def stability(baseline_csv: Path, level_csvs: list[tuple[int, Path]],
              space: str) -> pd.DataFrame:
    """Spearman ρ between each level and the full-size baseline, train_eval
    pairs only."""
    base = pd.read_csv(baseline_csv)
    base = base[(base["space"] == space) & (base["pair_type"] == "train_eval")]
    key = ["dataset_a", "split_a", "dataset_b", "split_b"]
    rows = []
    for level, lvl_csv in level_csvs:
        lvl = pd.read_csv(lvl_csv)
        lvl = lvl[(lvl["space"] == space) & (lvl["pair_type"] == "train_eval")]
        merged = base.merge(lvl, on=key, suffixes=("_base", "_lvl"))
        if len(merged) < 3:
            print(f"  warning: only {len(merged)} train_eval pairs at N={level}")
        for metric in SELFDIST_METRIC_COLS:
            ba = merged.get(f"{metric}_base")
            lv = merged.get(f"{metric}_lvl")
            if ba is None or lv is None:
                rows.append((metric, level, float("nan"), 0))
                continue
            mask = ba.notna() & lv.notna()
            if mask.sum() < 3:
                rows.append((metric, level, float("nan"), int(mask.sum())))
                continue
            rho = spearmanr(ba[mask], lv[mask]).statistic
            rows.append((metric, level, float(rho), int(mask.sum())))
    return pd.DataFrame(rows, columns=["metric", "level", "rho", "n_pairs"])


def make_heatmap(stab: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    pivot = stab.pivot(index="metric", columns="level", values="rho")
    pivot = pivot.reindex(SELFDIST_METRIC_COLS)
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(pivot.columns)),
                                    max(4, 0.45 * len(pivot.index))),
                           constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"N={c:,}" for c in pivot.columns],
                       rotation=30, ha="right", fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if v > 0.7 else "white")
    plt.colorbar(im, label="Spearman ρ vs full-size baseline")
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def recommend(stab: pd.DataFrame, threshold: float = 0.90) -> None:
    print(f"\n  Minimum N for ρ >= {threshold} (train_eval pairs only):")
    all_n = []
    for metric, sub in stab.groupby("metric"):
        sub_sorted = sub.sort_values("level")
        passing = sub_sorted[sub_sorted["rho"] >= threshold]
        if len(passing) == 0:
            print(f"    {metric:30s} : NEVER passes")
        else:
            n = int(passing.iloc[0]["level"])
            print(f"    {metric:30s} : N >= {n:,}")
            all_n.append(n)
    if all_n:
        worst = max(all_n)
        print(f"\n  RECOMMENDED minimum N (all metrics stable): {worst:,}")
        print(f"  -> use this floor when generating candidates for the interventional study")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vec-dir", default="/mnt/nvme_1tb_b/coverage_vectors")
    ap.add_argument("--baseline", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--output-dir", default="analysis_v3/density_invariance")
    ap.add_argument("--space", required=True, choices=["flow", "dino"])
    ap.add_argument("--levels", nargs="+", type=int, required=True,
                    help="Subsample sizes to test (smaller = faster)")
    ap.add_argument("--threshold", type=float, default=0.90)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--compute-only", action="store_true")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = Path(args.vec_dir)
    baseline_csv = Path(args.baseline)

    if not args.analyze_only:
        for level in sorted(args.levels):
            csv = out_dir / f"pairwise_self_{args.space}_N{level}.csv"
            run_one_level(args.space, level, vec_dir, csv, gpu=not args.no_gpu)

    if args.compute_only:
        print("compute-only mode: done")
        return

    level_csvs = [(lv, out_dir / f"pairwise_self_{args.space}_N{lv}.csv")
                  for lv in sorted(args.levels)]
    level_csvs = [(lv, p) for lv, p in level_csvs if p.exists()]
    if not level_csvs:
        print("no level CSVs found, nothing to analyze")
        return
    stab = stability(baseline_csv, level_csvs, args.space)
    stab_path = out_dir / f"stability_{args.space}_train_eval.csv"
    stab.to_csv(stab_path, index=False)
    print(f"  wrote {stab_path}")
    make_heatmap(stab, out_dir / f"stability_heatmap_{args.space}_train_eval.png",
                 f"Feature stability (train_eval only) — {args.space}")
    recommend(stab, threshold=args.threshold)


if __name__ == "__main__":
    main()
