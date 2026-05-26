"""Density invariance check — how stable are the 13 self-distance features
under aggressive downsampling?

For the interventional study (predictor-guided kubric hyperparameter search)
you need to know: "if I generate N frames for a candidate dataset, are the
features stable enough to give a reliable predictor score?"

This script:
  1. Re-runs `compute_pairwise_self_distances.py` at several `--max-flow` and
     `--max-dino` levels (writes to a level-tagged CSV each time).
  2. For each metric, computes Spearman ρ between the downsampled feature
     vector and the FULL baseline (the existing `pairwise_self_distances.csv`).
  3. Outputs a stability heatmap (metric × level → ρ).
  4. Recommends the smallest level where every metric stays above ρ_threshold.

Run:
    python scripts/transfer_analysis_v4/density_invariance.py
    python scripts/transfer_analysis_v4/density_invariance.py --space flow
    python scripts/transfer_analysis_v4/density_invariance.py --levels 100000 1000000 4000000

Outputs land under `analysis_v3/density_invariance/`.
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
    # mean_nn (3)
    "mean_nn_a_to_b", "mean_nn_b_to_a", "mean_nn_sym",
    # eps coverage (6)
    "a_covered_by_b_eps1px", "b_covered_by_a_eps1px",
    "a_covered_by_b_eps4px", "b_covered_by_a_eps4px",
    "a_covered_by_b_eps16px", "b_covered_by_a_eps16px",
    # KL (4)
    "kl_a_to_b_k5", "kl_b_to_a_k5",
    "kl_a_to_b_k20", "kl_b_to_a_k20",
]


def run_compute_at_level(space: str, level: int, vec_dir: Path,
                         output_csv: Path, gpu: bool) -> None:
    """Invoke compute_pairwise_self_distances.py at a given subsample level."""
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
    print(f"  [{space}, N={level}] → {output_csv}")
    print(f"    cmd: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def stability_table(baseline_csv: Path, level_csvs: list[tuple[int, Path]],
                    space: str, pair_type: str = "train_eval") -> pd.DataFrame:
    """For each metric col × level, compute Spearman ρ between the level's
    feature vector and the baseline feature vector across (a, b) pairs."""
    base = pd.read_csv(baseline_csv)
    base = base[(base["space"] == space) & (base["pair_type"] == pair_type)]
    key_cols = ["dataset_a", "split_a", "dataset_b", "split_b"]

    rows = []
    for level, lvl_csv in level_csvs:
        lvl = pd.read_csv(lvl_csv)
        lvl = lvl[(lvl["space"] == space) & (lvl["pair_type"] == pair_type)]
        merged = base.merge(lvl, on=key_cols, suffixes=("_base", "_lvl"))
        for metric in SELFDIST_METRIC_COLS:
            ba = merged.get(f"{metric}_base")
            lv = merged.get(f"{metric}_lvl")
            if ba is None or lv is None or len(merged) < 3:
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
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(pivot.columns)),
                                    max(4, 0.4 * len(pivot.index))),
                           constrained_layout=True)
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"N={c:,}" for c in pivot.columns], rotation=30, ha="right", fontsize=8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if v > 0.7 else "white")
    plt.colorbar(im, label="Spearman ρ vs. full-size baseline")
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


def recommend_min_n(stab: pd.DataFrame, threshold: float = 0.90) -> dict:
    """For each metric, find the smallest N where stability ρ >= threshold.
    For paper recommendation: report the metric-set-wide max of these."""
    min_n_per_metric = {}
    for metric, sub in stab.groupby("metric"):
        sub_sorted = sub.sort_values("level")
        passing = sub_sorted[sub_sorted["rho"] >= threshold]
        if len(passing) == 0:
            min_n_per_metric[metric] = None
        else:
            min_n_per_metric[metric] = int(passing.iloc[0]["level"])
    return min_n_per_metric


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vec-dir", default="/mnt/nvme_1tb_b/coverage_vectors",
                    help="Where the .npy flow / DINO vectors live")
    ap.add_argument("--baseline", default="analysis_v3/pairwise_self_distances.csv",
                    help="Full-size baseline to compare against")
    ap.add_argument("--output-dir", default="analysis_v3/density_invariance")
    ap.add_argument("--space", default="flow", choices=["flow", "dino", "both"],
                    help="Which feature space to sweep. 'both' runs flow then dino.")
    ap.add_argument("--flow-levels", nargs="+", type=int,
                    default=[50_000, 200_000, 1_000_000, 4_000_000, 16_000_000],
                    help="--max-flow values to test")
    ap.add_argument("--dino-levels", nargs="+", type=int,
                    default=[10_000, 50_000, 200_000, 1_000_000, 8_000_000],
                    help="--max-dino values to test")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="ρ threshold for recommended minimum N")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--compute-only", action="store_true",
                    help="Just run the compute step, skip analysis")
    ap.add_argument("--analyze-only", action="store_true",
                    help="Just analyze existing level CSVs, skip compute")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = Path(args.vec_dir)
    baseline_csv = Path(args.baseline)
    spaces = ["flow", "dino"] if args.space == "both" else [args.space]

    # 1. Compute
    if not args.analyze_only:
        for sp in spaces:
            levels = args.flow_levels if sp == "flow" else args.dino_levels
            for level in levels:
                lvl_csv = out_dir / f"pairwise_self_{sp}_N{level}.csv"
                if lvl_csv.exists():
                    print(f"  skip (exists): {lvl_csv}")
                    continue
                run_compute_at_level(sp, level, vec_dir, lvl_csv, gpu=not args.no_gpu)

    if args.compute_only:
        print("compute_only: done.")
        return

    # 2. Analyze
    for sp in spaces:
        levels = args.flow_levels if sp == "flow" else args.dino_levels
        level_csvs = []
        for level in levels:
            lvl_csv = out_dir / f"pairwise_self_{sp}_N{level}.csv"
            if lvl_csv.exists():
                level_csvs.append((level, lvl_csv))
        if not level_csvs:
            print(f"  no level CSVs for {sp}, skip")
            continue
        stab = stability_table(baseline_csv, level_csvs, sp, pair_type="train_eval")
        stab_path = out_dir / f"stability_{sp}_train_eval.csv"
        stab.to_csv(stab_path, index=False)
        print(f"  wrote {stab_path}")

        make_heatmap(stab, out_dir / f"stability_heatmap_{sp}_train_eval.png",
                     f"Feature stability vs subsampled N — {sp} (train_eval pairs)")

        rec = recommend_min_n(stab, threshold=args.threshold)
        print(f"\n  Minimum N for ρ >= {args.threshold} ({sp}, train_eval):")
        all_pass = []
        for metric, n in rec.items():
            if n is None:
                print(f"    {metric:30s} : NEVER passes threshold")
            else:
                print(f"    {metric:30s} : N >= {n:,}")
                all_pass.append(n)
        if all_pass:
            worst = max(all_pass)
            print(f"\n  RECOMMENDED minimum N (all metrics stable): {worst:,}")
            print(f"  (Use this as the sample-size floor for the interventional study.)")


if __name__ == "__main__":
    main()
