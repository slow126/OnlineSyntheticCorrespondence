#!/usr/bin/env python3
"""Regenerate CROSS_ARCHITECTURE_CONSENSUS_ALL_SPLITS.csv in one command.

Closes the Paper-2 §6.5 reproducibility gap: the consensus table was previously
hand-compiled from three separate `ceiling_analysis.py` runs (numbers correct,
but no single script regenerated the file). This wrapper imports
`ceiling_analysis`'s own functions, runs the held-architecture / held-variant /
balanced consensus for each CV split on the 3-architecture (CATs++/RAFT/GLU-Net)
observed-peak rows, and writes the exact CSV.

Usage (from scripts/transfer_analysis_v4/):
    python regenerate_consensus_csv.py
    # custom location / bootstrap settings:
    python regenerate_consensus_csv.py \
        --rows-dir results_glunet_observed_peak_all_splits/predictions/peak_pck \
        --n-boot 5000 --seed 0 \
        --out results_glunet_observed_peak_all_splits/CROSS_ARCHITECTURE_CONSENSUS_ALL_SPLITS.csv

The architecture-level (held_arch) columns are the camera-ready numbers:
motion_rho_held_arch / cross_arch_consensus_rho / fraction (= motion / consensus).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

import ceiling_analysis as ca

SPLITS = ["LOTO", "LOBO", "JOINT"]
FIELDS = [
    "split",
    "motion_rho_held_arch", "cross_arch_consensus_rho", "fraction",
    "ratio_ci_low", "ratio_ci_high",
    "variant_motion_rho", "held_variant_consensus_rho", "variant_fraction",
    "variant_ratio_ci_low", "variant_ratio_ci_high",
    "balanced_consensus_rho", "balanced_fraction",
    "balanced_ratio_ci_low", "balanced_ratio_ci_high",
]


def per_split(rows_path: Path, min_src: int, n_boot: int, seed: int) -> dict:
    data = pd.read_csv(rows_path)
    data["replicate_group"] = data["variant"].astype(str).str.split("|").str[0]

    consensus = ca.held_variant_consensus(data, min_src, balance_groups=False)
    consensus_bal = ca.held_variant_consensus(data, min_src, balance_groups=True)
    consensus_group = ca.held_group_consensus(data, min_src)

    # variant-level
    consensus_mean = float(consensus["consensus_rho"].mean())
    matched_feature = float(consensus["feature_rho"].mean())
    # balanced
    consensus_bal_mean = float(consensus_bal["consensus_rho"].mean())
    matched_feature_bal = float(consensus_bal["feature_rho"].mean())
    # held-architecture (camera-ready)
    consensus_group_mean = float(consensus_group["consensus_rho"].mean())
    feature_group_mean = float(consensus_group["feature_rho"].mean())

    by_benchmark = (
        consensus.groupby("benchmark")["feature_rho"].mean().rename("feature_rho")
        .to_frame()
        .join(consensus.groupby("benchmark")["consensus_rho"].mean()
              .rename("consensus_rho"), how="outer")
        .join(consensus_bal.groupby("benchmark")["consensus_rho"].mean()
              .rename("balanced_consensus_rho"), how="outer")
        .join(consensus_group.groupby("benchmark")["consensus_rho"].mean()
              .rename("held_arch_consensus_rho"), how="outer")
        .join(consensus_group.groupby("benchmark")["feature_rho"].mean()
              .rename("held_arch_feature_rho"), how="outer")
        .reset_index()
    )

    ratio_interval = ca.paired_ratio_bootstrap(
        by_benchmark, "feature_rho", "consensus_rho", n_boot, seed)
    ratio_bal_interval = ca.paired_ratio_bootstrap(
        by_benchmark, "feature_rho", "balanced_consensus_rho", n_boot, seed)
    ratio_group_interval = ca.paired_ratio_bootstrap(
        by_benchmark, "held_arch_feature_rho", "held_arch_consensus_rho",
        n_boot, seed)

    def frac(a, b):
        return a / b if b > 0 else float("nan")

    return {
        "motion_rho_held_arch": feature_group_mean,
        "cross_arch_consensus_rho": consensus_group_mean,
        "fraction": frac(feature_group_mean, consensus_group_mean),
        "ratio_ci_low": ratio_group_interval[0],
        "ratio_ci_high": ratio_group_interval[1],
        "variant_motion_rho": matched_feature,
        "held_variant_consensus_rho": consensus_mean,
        "variant_fraction": frac(matched_feature, consensus_mean),
        "variant_ratio_ci_low": ratio_interval[0],
        "variant_ratio_ci_high": ratio_interval[1],
        "balanced_consensus_rho": consensus_bal_mean,
        "balanced_fraction": frac(matched_feature_bal, consensus_bal_mean),
        "balanced_ratio_ci_low": ratio_bal_interval[0],
        "balanced_ratio_ci_high": ratio_bal_interval[1],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rows-dir",
        default="results_glunet_observed_peak_all_splits/predictions/peak_pck",
        help="dir containing rows_{SPLIT}_motion.csv with catspp/raft/glunet",
    )
    ap.add_argument("--min-src", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default=("results_glunet_observed_peak_all_splits/"
                 "CROSS_ARCHITECTURE_CONSENSUS_ALL_SPLITS.csv"),
    )
    args = ap.parse_args()

    rows_dir = Path(args.rows_dir)
    out = Path(args.out)
    records = []
    for split in SPLITS:
        rows_path = rows_dir / f"rows_{split}_motion.csv"
        if not rows_path.exists():
            raise SystemExit(f"missing rows file: {rows_path}")
        rec = per_split(rows_path, args.min_src, args.n_boot, args.seed)
        rec["split"] = split
        records.append(rec)
        print(f"{split}: motion_held_arch={rec['motion_rho_held_arch']:.6f} "
              f"consensus={rec['cross_arch_consensus_rho']:.6f} "
              f"fraction={rec['fraction']:.6f}")

    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for rec in records:
            # match the committed file's formatting: 6 dp on rhos/fractions,
            # 2 dp on the bootstrap ratio CIs
            for k in FIELDS:
                if k == "split":
                    continue
                if k.endswith("_ci_low") or k.endswith("_ci_high"):
                    rec[k] = f"{rec[k]:.2f}"
                else:
                    rec[k] = f"{rec[k]:.6f}"
            w.writerow({k: rec.get(k) for k in FIELDS})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
