"""
Per-benchmark stratification of L and g predictors.

Reads existing LOTO prediction row CSVs from results_dir — no rerun needed.
For each benchmark, pools all variants and reports:

  g (within-context ranking signal):
    ρ_g  = Spearman(g, actual_demeaned)   actual demeaned within each context
    r_g  = Pearson (g, actual_demeaned)

  L (context-level calibration):
    ρ_L  = Spearman(L, actual)            across all rows for that benchmark
    r_L  = Pearson (L, actual)

  L+g (combined):
    r_Lg = Pearson (L+g, actual)

Reports for motion and appearance families side by side, with the motion−appearance
gap, then a dense/sparse group summary.

Usage:
    python scripts/transfer_analysis_v4/benchmark_stratification.py
    python scripts/transfer_analysis_v4/benchmark_stratification.py \
        --results scripts/transfer_analysis_v4/results_mixed \
        --target peak_pck --split LOTO
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

# ── benchmark groupings ──────────────────────────────────────────────────────
DENSE  = {"flyingthings", "kitti2012", "kitti2015", "synthetic", "pointodyssey"}
SPARSE = {"spair", "pfpascal", "pfwillow", "tss", "middlebury"}

BENCHMARK_ORDER = (
    ["flyingthings", "kitti2012", "kitti2015", "pointodyssey", "synthetic"],
    ["middlebury", "pfpascal", "pfwillow", "spair", "tss"],
)

# ── families to compare ──────────────────────────────────────────────────────
FAMILIES = {
    "motion":     "motion",
    "motion_sym": "motion_sym",
    "appearance": "appearance",
    "app_sym":    "appearance_sym",
}


def safe_corr(fn, x, y):
    """Return correlation or NaN if too few valid rows."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4 or x[mask].std() == 0:
        return np.nan
    return fn(x[mask], y[mask])[0]


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Given all rows for one (benchmark, family) combination,
    return g / L / L+g correlation metrics.

    Demeaning: subtract per-context mean from actual to get actual_demeaned.
    g is already the within-context ridge residual, so it's naturally on the
    same scale as actual_demeaned.
    """
    df = df.copy()
    ctx_means = df.groupby("context_id")["actual"].transform("mean")
    actual_dem = df["actual"] - ctx_means

    g   = df["g"].values
    L   = df["L"].values
    Lg  = (df["L"] + df["g"]).values
    act = df["actual"].values
    act_dem = actual_dem.values

    return {
        "n":    len(df),
        "rho_g": safe_corr(spearmanr, g,  act_dem),
        "r_g":   safe_corr(pearsonr,  g,  act_dem),
        "rho_L": safe_corr(spearmanr, L,  act),
        "r_L":   safe_corr(pearsonr,  L,  act),
        "r_Lg":  safe_corr(pearsonr,  Lg, act),
    }


def load(pred_dir: Path, split: str, family: str) -> pd.DataFrame | None:
    p = pred_dir / f"rows_{split}_{family}.csv"
    return pd.read_csv(p) if p.exists() else None


def fmt(v, width=7):
    return f"{v:>+{width}.3f}" if np.isfinite(v) else f"{'—':>{width}}"


def run(results_dir: str, target: str, split: str) -> None:
    pred_dir = Path(results_dir) / "predictions" / target

    # ── collect metrics per (benchmark, family) ──────────────────────────────
    records = {}  # (benchmark, family_label) -> metrics dict
    for label, family in FAMILIES.items():
        df = load(pred_dir, split, family)
        if df is None:
            print(f"  [skip] {split} {family} — file not found")
            continue
        for bench, grp in df.groupby("benchmark"):
            records[(bench, label)] = compute_metrics(grp)

    if not records:
        print("No prediction files found.")
        return

    # ── print per-benchmark table ─────────────────────────────────────────────
    families = list(FAMILIES.keys())
    col_w = 7

    def section_header():
        hdr = f"{'benchmark':<14} {'n':>4}  "
        for fam in families:
            hdr += f"  {fam:^23}"
        return hdr

    def metric_subheader():
        sub = " " * 20
        for _ in families:
            sub += f"  {'ρ_g':>{col_w}} {'r_g':>{col_w}} {'r_Lg':>{col_w}}"
        return sub

    def L_subheader():
        sub = " " * 20
        for _ in families:
            sub += f"  {'ρ_L':>{col_w}} {'r_L':>{col_w}} {'r_Lg':>{col_w}}"
        return sub

    print()
    print("=" * 90)
    print(f"BENCHMARK STRATIFICATION — target={target}, split={split}, pooled across variants")
    print("=" * 90)

    # ── Table 1: g metrics (within-context ranking) ───────────────────────────
    print()
    print("── Table 1: g  (within-context ranking signal; actual demeaned by context) ──")
    print(f"{'benchmark':<14} {'n':>4}", end="")
    for fam in families:
        print(f"  {'── ' + fam + ' ──':^23}", end="")
    print()
    print(" " * 20, end="")
    for _ in families:
        print(f"  {'ρ_g':>{col_w}} {'r_g':>{col_w}} {'Δmot-app':>{col_w}}", end="")
    print()
    print("-" * 90)

    mot_rho_g, app_rho_g = {}, {}
    for group_idx, group in enumerate(BENCHMARK_ORDER):
        if group_idx > 0:
            print()
        for bench in group:
            if not any((bench, fam) in records for fam in families):
                continue
            n = records.get((bench, families[0]), {}).get("n", 0)
            line = f"{bench:<14} {n:>4}"
            for fam in families:
                m = records.get((bench, fam), {})
                rho = m.get("rho_g", np.nan)
                r   = m.get("r_g",   np.nan)
                line += f"  {fmt(rho)} {fmt(r)}"
                if fam == "motion":
                    mot_rho_g[bench] = rho
                elif fam == "appearance":
                    app_rho_g[bench] = rho
                # gap (motion − appearance) in last slot
                if fam == "app_sym":
                    gap = mot_rho_g.get(bench, np.nan) - app_rho_g.get(bench, np.nan)
                    line += f"  {fmt(gap)}"
                else:
                    line += " " * (col_w + 2)
            grp_tag = "dense" if bench in DENSE else "sparse"
            print(line + f"  ← {grp_tag}")

    # ── Table 2: L metrics (level calibration) ────────────────────────────────
    print()
    print("── Table 2: L  (context-level calibration; ρ_L and r_L vs raw actual) ──")
    print(f"{'benchmark':<14} {'n':>4}", end="")
    for fam in families:
        print(f"  {'── ' + fam + ' ──':^21}", end="")
    print()
    print(" " * 20, end="")
    for _ in families:
        print(f"  {'ρ_L':>{col_w}} {'r_L':>{col_w}} {'r_Lg':>{col_w}}", end="")
    print()
    print("-" * 90)

    for group_idx, group in enumerate(BENCHMARK_ORDER):
        if group_idx > 0:
            print()
        for bench in group:
            if not any((bench, fam) in records for fam in families):
                continue
            n = records.get((bench, families[0]), {}).get("n", 0)
            line = f"{bench:<14} {n:>4}"
            for fam in families:
                m = records.get((bench, fam), {})
                line += f"  {fmt(m.get('rho_L', np.nan))} {fmt(m.get('r_L', np.nan))} {fmt(m.get('r_Lg', np.nan))}"
            grp_tag = "dense" if bench in DENSE else "sparse"
            print(line + f"  ← {grp_tag}")

    # ── Table 3: dense vs sparse group summary ────────────────────────────────
    print()
    print("── Table 3: dense vs sparse group summary (mean across benchmarks) ──")
    print(f"{'group':<10}", end="")
    for fam in families:
        print(f"  {'── ' + fam + ' ──':^23}", end="")
    print()
    print(" " * 10, end="")
    for _ in families:
        print(f"  {'ρ_g':>{col_w}} {'r_g':>{col_w}} {'r_Lg':>{col_w}}", end="")
    print()
    print("-" * 85)

    for group_name, bench_set in [("dense", DENSE), ("sparse", SPARSE)]:
        line = f"{group_name:<10}"
        for fam in families:
            vals_rho, vals_r, vals_rLg = [], [], []
            for bench in bench_set:
                m = records.get((bench, fam), {})
                if np.isfinite(m.get("rho_g", np.nan)):
                    vals_rho.append(m["rho_g"])
                if np.isfinite(m.get("r_g", np.nan)):
                    vals_r.append(m["r_g"])
                if np.isfinite(m.get("r_Lg", np.nan)):
                    vals_rLg.append(m["r_Lg"])
            mean_rho = np.mean(vals_rho) if vals_rho else np.nan
            mean_r   = np.mean(vals_r)   if vals_r   else np.nan
            mean_rLg = np.mean(vals_rLg) if vals_rLg else np.nan
            line += f"  {fmt(mean_rho)} {fmt(mean_r)} {fmt(mean_rLg)}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="scripts/transfer_analysis_v4/results_mixed")
    ap.add_argument("--target",  default="peak_pck",
                    choices=["peak_pck", "auc_normalized"])
    ap.add_argument("--split",   default="LOTO",
                    choices=["LOTO", "LOBO", "JOINT"])
    args = ap.parse_args()
    run(args.results, args.target, args.split)


if __name__ == "__main__":
    main()
