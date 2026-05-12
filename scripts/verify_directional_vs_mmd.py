#!/usr/bin/env python3
"""
Verify directional coverage vs symmetric MMD ranking for each eval benchmark,
and validate both against actual model transfer performance.

Performance can be loaded from raw snapshot directories (recommended) or from a
pre-built tier0 grid CSV.  Snapshot dirs are scanned for training_summary.txt
files, which contain per-benchmark peak PCK already computed.

Coverage defaults to the bag-of-flows Q50 AUC summary
(coverage_v2_flow_only_raw_joint_curve_summary_q50.csv): each flow vector is
treated independently and directional coverage is summarised as the AUC of the
coverage-vs-epsilon curve at the 50th percentile.  This is directly comparable
to flow MMD, which also operates on the raw per-pixel flow distribution.

Usage:
    # From raw snapshots (recommended):
    python scripts/verify_directional_vs_mmd.py \\
        --snapshot-dirs /mnt/nvme_1tb_b/snapshots_ptody_fix \\
                        /mnt/nvme_1tb_b/snapshots_synthetic_long \\
                        ./snapshots_2d_warps \\
                        ./snapshots_mixed ./snapshots_raft ./snapshots_raft_2d_mix \\
                        ./snapshots_spair_only \\
                        /mnt/nvme_1tb_b/snapshots_synth_2d \\
        --latex-out figures/section4

    # From pre-built tier0 CSV (legacy):
    python scripts/verify_directional_vs_mmd.py --perf path/to/tier0.csv
"""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import kendalltau

BASE = (
    "analysis_comprehensive_runs/"
    "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3"
)

# -----------------------------------------------------------------------
# Display-name maps
# -----------------------------------------------------------------------
TRAIN_NAMES = {
    "flyingthings": "FlyingThings3D",
    "sintel": "Sintel",
    "pointodyssey": "PointOdyssey",
    "spair": "SPair-71k",
    "synthetic": "SDF-Fractal3D",
    "synthetic_2d_warp": "SDF-Fractal3D (2D warp)",
    "synthetic_large_zoom": "SDF-Fractal3D (large zoom)",
    "synthetic_small_zoom": "SDF-Fractal3D (small zoom)",
    "synthetic_random_flipping": "SDF-Fractal3D (rand.\\ flip)",
    "imagenet2dwarp": "ImageNet-2D-Warp",
}

# Datasets that are sample-limited outliers — drawn from a separator in the table.
SAMPLE_LIMITED = {"spair"}  # tiny training sets with near-random transfer

BENCH_NAMES = {
    "flyingthings": "FlyingThings",
    "kitti2012": "KITTI-2012",
    "kitti2015": "KITTI-2015",
    "middlebury": "Middlebury",
    "pfpascal": "PF-Pascal",
    "pfwillow": "PF-Willow",
    "pointodyssey": "PointOdyssey",
    "spair": "SPair-71k",
    "tss": "TSS",
    "synthetic": "Synthetic (val)",
}

def _tn(key: str) -> str:
    return TRAIN_NAMES.get(key, key.replace("_", r"\_"))

def _bn(key: str) -> str:
    return BENCH_NAMES.get(key, key.replace("_", r"\_"))

def _fmt(v: float, decimals: int = 3, signed: bool = True) -> str:
    if not math.isfinite(v):
        return "--"
    fmt = f"+.{decimals}f" if signed else f".{decimals}f"
    return format(v, fmt)

def _sig(p: float, math: bool = False) -> str:
    """Return significance stars. math=True for use inside $...$ (no extra delimiters)."""
    if p < 0.001:
        sup = r"^{***}"
    elif p < 0.01:
        sup = r"^{**}"
    elif p < 0.05:
        sup = r"^{*}"
    else:
        return ""
    return sup if math else f"${sup}$"


# -----------------------------------------------------------------------
# LaTeX table writers
# -----------------------------------------------------------------------

def _write_tau_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    caption: str,
    label: str,
) -> None:
    """
    Table: per-benchmark Kendall tau of (t2e, e2t, MMD) vs actual PCK@5%.
    """
    lines: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"\textbf{Benchmark} & $\tau$(t$\to$e) & $\tau$(e$\to$t) & $\tau$(MMD) & Best \\",
        r"\midrule",
    ]

    tau_t2e_vals, tau_e2t_vals, tau_mmd_vals = [], [], []
    e2t_wins, mmd_wins, t2e_wins = 0, 0, 0

    for bench in sorted(df["benchmark"].unique()):
        sub = (
            df[df["benchmark"] == bench]
            .drop_duplicates(subset=["train"])
            .dropna(subset=["t2e", "mmd", "perf_mean"])
        )
        if len(sub) < 3:
            continue
        perf_rank = sub["perf_mean"].rank()
        tau_t2e, p_t2e = kendalltau(sub["t2e"].rank(), perf_rank)
        tau_e2t, p_e2t = kendalltau(sub["e2t"].rank(), perf_rank)
        tau_mmd, p_mmd = kendalltau((-sub["mmd"]).rank(), perf_rank)

        tau_t2e_vals.append(tau_t2e)
        tau_e2t_vals.append(tau_e2t)
        tau_mmd_vals.append(tau_mmd)

        best_name, best_tau = max(
            [("e$\\to$t", tau_e2t), ("MMD", tau_mmd), ("t$\\to$e", tau_t2e)],
            key=lambda x: x[1],
        )
        if best_name == "e$\\to$t":
            e2t_wins += 1
        elif best_name == "MMD":
            mmd_wins += 1
        else:
            t2e_wins += 1

        # bold the best value
        vals = {"t2e": (tau_t2e, p_t2e), "e2t": (tau_e2t, p_e2t), "mmd": (tau_mmd, p_mmd)}
        best_key = max(vals, key=lambda k: vals[k][0])

        def _cell(key: str) -> str:
            v, p = vals[key]
            s = _fmt(v) + _sig(p)
            return rf"\textbf{{{s}}}" if key == best_key else s

        lines.append(
            rf"{_bn(bench)} & {_cell('t2e')} & {_cell('e2t')} & {_cell('mmd')} & {best_name} \\"
        )

    # summary row
    n = len(tau_t2e_vals)
    if n:
        mt = sum(tau_t2e_vals) / n
        me = sum(tau_e2t_vals) / n
        mm = sum(tau_mmd_vals) / n
        lines += [
            r"\midrule",
            rf"\textit{{Mean}} & \textit{{{_fmt(mt)}}} & \textit{{{_fmt(me)}}} & \textit{{{_fmt(mm)}}} & -- \\",
            r"\midrule",
            rf"\multicolumn{{5}}{{l}}{{\small e$\to$t best on {e2t_wins}/{n} benchmarks; "
            rf"MMD best on {mmd_wins}/{n}; t$\to$e best on {t2e_wins}/{n}}} \\",
        ]

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def _write_spotlight_table(
    df: pd.DataFrame,
    out_path: Path,
    *,
    benchmark: str,
    caption: str,
    label: str,
    higher_is_better: bool = True,
    cov_description: str = "",
) -> None:
    """
    Spotlight table sorted by Peak PCK, with rank columns for e→t and MMD
    so rank disagreements are immediately visible without Kendall tau.

    Columns: Training set | Peak PCK | e→t (rank) | MMD (rank)

    Rank is shown in small grey subscript next to each value.
    Sample-limited datasets are separated by a midrule.
    """
    sub = (
        df[df["benchmark"] == benchmark]
        .drop_duplicates(subset=["train"])
        .dropna(subset=["perf_mean"])
        .sort_values("perf_mean", ascending=False)
        .reset_index(drop=True)
    )
    if sub.empty:
        print(f"WARNING: no data for spotlight benchmark={benchmark!r}")
        return

    n = len(sub)
    cov_dir_arrow = r"$\downarrow$" if not higher_is_better else r"$\uparrow$"
    sub["rank_perf"] = range(1, n + 1)
    rank_ascending = not higher_is_better
    sub["rank_e2t"] = sub["e2t"].rank(ascending=rank_ascending, method="min")
    sub["rank_t2e"] = sub["t2e"].rank(ascending=rank_ascending, method="min")
    sub["rank_mmd"] = sub["mmd"].rank(ascending=True, method="min")

    # Rank direction depends on metric: coverage fraction/AUC → higher=better; eps_at50 → lower=better
    rank_ascending = not higher_is_better
    sub["rank_e2t"] = sub["e2t"].rank(ascending=rank_ascending, method="min")
    sub["rank_t2e"] = sub["t2e"].rank(ascending=rank_ascending, method="min")

    lines: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        rf"\textbf{{Training set}} & \textbf{{PCK ($\alpha$\,=\,5\%)}} & \textbf{{rank}} & "
        rf"\textbf{{Eval{{\footnotesize$\to$}}Train$^\dagger$ ({cov_dir_arrow})}} & "
        rf"\textbf{{Train{{\footnotesize$\to$}}Eval$^\dagger$ ({cov_dir_arrow})}} & "
        rf"\textbf{{Flow MMD ($\downarrow$)}} \\",
        r"\midrule",
    ]

    main_rows = sub[~sub["train"].isin(SAMPLE_LIMITED)]
    outlier_rows = sub[sub["train"].isin(SAMPLE_LIMITED)]

    def _row(r: pd.Series) -> str:
        name = _tn(str(r["train"]))
        pck_rank = int(r["rank_perf"])
        pck = f"{r['perf_mean']:.1f}" if pd.notna(r.get("perf_mean")) else "--"

        def _cell(val, rank, nan_str="--"):
            if not pd.notna(val) or rank is None:
                return nan_str
            val_str = f"{val:.1f}" if abs(val) >= 10 else f"{val:.3f}"
            s = rf"{val_str} ({rank})"
            return rf"\textbf{{{s}}}" if abs(rank - pck_rank) <= 1 else s

        # eps_at50 NaN means distribution never reaches 50% coverage within 64px
        t2e_nan = r"$>$64\,px" if not higher_is_better else "--"

        e2t_v = r.get("e2t")
        t2e_v = r.get("t2e")
        mmd_v = r.get("mmd")
        e2t_r = int(r["rank_e2t"]) if pd.notna(r.get("rank_e2t")) else None
        t2e_r = int(r["rank_t2e"]) if pd.notna(r.get("rank_t2e")) else None
        mmd_r = int(r["rank_mmd"]) if pd.notna(r.get("rank_mmd")) else None

        return (
            rf"{name} & {pck} & {pck_rank} & "
            rf"{_cell(e2t_v, e2t_r)} & "
            rf"{_cell(t2e_v, t2e_r, nan_str=t2e_nan)} & "
            rf"{_cell(mmd_v, mmd_r)} \\"
        )

    for _, r in main_rows.iterrows():
        lines.append(_row(r))

    if not outlier_rows.empty:
        lines.append(r"\midrule")
        lines.append(
            rf"\multicolumn{{6}}{{l}}{{\textit{{sample-limited ($n \ll$ others)}}}} \\"
        )
        for _, r in outlier_rows.iterrows():
            lines.append(_row(r))

    lines += [
        r"\midrule",
        rf"\multicolumn{{6}}{{p{{0.95\linewidth}}}}{{\small"
        rf" Sorted by PCK ($\alpha$\,=\,5\%, best checkpoint, avg.\ RAFT \& CATs++)."
        rf" \textbf{{Bold}} = metric rank within 1 of PCK rank."
        r" $^\dagger$Flow Coverage = " + cov_description + r"} \\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def _load_snapshots(snapshot_dirs: list[str]) -> pd.DataFrame:
    """
    Scan snapshot directories for training_summary.txt files and extract
    peak PCK per benchmark.

    CATs runs: only use pretrainedTrue / freezeFalse variants (the standard
    baseline configuration).  RAFT runs: all variants included.

    Multiple runs of the same (train_dataset, model_type, benchmark) are
    averaged.  Then model types are averaged to produce one perf_mean per
    (train, benchmark).

    Returns DataFrame with columns: train, benchmark, perf_mean
    """
    rows: list[dict] = []
    skipped = 0

    for snap_dir_str in snapshot_dirs:
        snap_dir = Path(snap_dir_str)
        if not snap_dir.exists():
            print(f"  WARNING: snapshot dir not found: {snap_dir}")
            continue

        for run_dir in sorted(snap_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_path = run_dir / "training_summary.txt"
            if not summary_path.exists():
                continue

            try:
                summary = summary_path.read_text(encoding="utf-8")
            except OSError:
                continue

            # Parse train dataset
            m = re.search(r"Train dataset:\s*(\S+)", summary)
            if not m:
                continue
            train_dataset = m.group(1).strip()

            # Infer model type and apply variant filter
            dir_lower = run_dir.name.lower()
            if "raft" in dir_lower:
                model_type = "raft"
            else:
                # CATs run — only keep pretrainedTrue / freezeFalse
                pretrained = bool(re.search(r"Pretrained backbone:\s*True", summary))
                freeze = bool(re.search(r"Freeze backbone:\s*True", summary))
                if not pretrained or freeze:
                    skipped += 1
                    continue
                model_type = "cats"

            # Parse BEST PERFORMANCE PER BENCHMARK block
            # Format: "flyingthings: 79.72% PCK (epoch 370, ...)"
            for bm in re.finditer(
                r"^(\w+)\s*:\s*([\d.]+)%\s*PCK", summary, re.MULTILINE
            ):
                bench = bm.group(1).strip()
                pck = float(bm.group(2))
                rows.append(
                    {"train": train_dataset, "benchmark": bench,
                     "peak_pck": pck, "model": model_type}
                )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    print(
        f"  Loaded {len(df)} (train, bench, model) records from snapshots "
        f"({skipped} CATs variants skipped)"
    )

    # Average duplicate runs of the same (train, bench, model), then across models
    avg = (
        df.groupby(["train", "benchmark", "model"])["peak_pck"]
        .mean()
        .groupby(level=["train", "benchmark"])
        .mean()
        .reset_index()
        .rename(columns={"peak_pck": "perf_mean"})
    )
    return avg


def _parse_perf_cell(cell: str, idx: int = 0) -> float:
    """Parse a '(peak_pck, auc_obs_norm)' tuple string from a tier0 grid CSV."""
    cell = str(cell).strip()
    if cell in ("", "--", "nan"):
        return float("nan")
    cell = cell.strip("()")
    parts = cell.split(",")
    try:
        return float(parts[idx].strip())
    except (ValueError, IndexError):
        return float("nan")


def _load_performance_csv(perf_path: Path) -> pd.DataFrame:
    """
    Load a pre-built tier0 representative pair grid CSV (wide format).
    Columns: train_dataset, model, <benchmark1>, <benchmark2>, ...
    Each cell is a '(peak_pck, auc_obs_norm)' tuple string.
    Returns DataFrame: train, benchmark, perf_mean
    """
    if not perf_path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(perf_path)
    id_cols = ["train_dataset", "model"]
    bench_cols = [c for c in raw.columns if c not in id_cols]

    rows = []
    for _, r in raw.iterrows():
        train = str(r["train_dataset"])
        model = str(r["model"])
        for bench in bench_cols:
            pck = _parse_perf_cell(r[bench])
            rows.append({"train": train, "benchmark": bench, "pck": pck, "model": model})

    df = pd.DataFrame(rows)
    avg = (
        df.groupby(["train", "benchmark"])["pck"]
        .mean()
        .reset_index()
        .rename(columns={"pck": "perf_mean"})
    )
    return avg


def _load_coverage(cov_path: Path, k: int, eps: str = "2px", use_eps50: bool = True) -> pd.DataFrame:
    """
    Load a coverage CSV.  Returns DataFrame with columns:
        train, benchmark, t2e, e2t
    where **higher values always mean better coverage / more similar distributions**
    (the caller does not need to worry about sign).

    Three formats are supported:

    1. Raw per-pixel epsilon ladder (*_joint_full.csv):
         eval_covered_by_train_eps{N}  →  e2t  (higher = more eval covered = better)
         train_covered_by_eval_eps{N}  →  t2e  (higher = more train covered)
         The --eps argument selects which threshold to use (e.g. "2px", "1px").
         Higher is better for both.

    2. Q50 curve-summary AUC (*_curve_summary_q50.csv):
         eval_to_train_auc  →  e2t  (higher AUC = better)
         train_to_eval_auc  →  t2e  (higher AUC = better)

    3. HOF / DINO / legacy k-NN radius coverage:
         eval_to_train_coverage / train_to_eval_coverage  (higher fraction = better)
         Filtered by k argument.

    MMD is handled separately in _load_mmd() and is LOWER-is-better;
    that inversion is applied in the Kendall tau computation, not here.
    """
    cov = pd.read_csv(cov_path)

    # Detect split/dataset column names
    if "dataset1" in cov.columns:
        train_col, bench_col, split_col = "dataset1", "dataset2", "split2"
    else:
        train_col, bench_col, split_col = "train_dataset", "eval_dataset", "eval_split"

    cov = cov[cov[split_col].isin(["val", "test"])]

    # Format 1: raw epsilon ladder — pick the requested threshold
    e2t_eps_col = f"eval_covered_by_train_eps{eps}"
    t2e_eps_col = f"train_covered_by_eval_eps{eps}"
    if e2t_eps_col in cov.columns:
        t2e_col, e2t_col = t2e_eps_col, e2t_eps_col
        higher_is_better = True
        print(f"  Coverage format: raw epsilon ladder  eps={eps}  "
              f"(higher fraction = better)")

    # Format 2: Q50 curve summary — two sub-metrics available
    elif "eval_to_train_auc" in cov.columns:
        if use_eps50:
            # eps_at50: pixel radius at which 50% of flows are matched — LOWER is better
            t2e_col, e2t_col = "train_to_eval_eps_at50", "eval_to_train_eps_at50"
            higher_is_better = False
            print("  Coverage format: Q50 eps_at50  (pixel radius at 50% coverage; lower = better)")
        else:
            t2e_col, e2t_col = "train_to_eval_auc", "eval_to_train_auc"
            higher_is_better = True
            print("  Coverage format: Q50 AUC  (mean % covered across radii; higher = better)")

    # Format 3: k-NN radius fraction (HOF / DINO / legacy FAISS)
    else:
        if "k" in cov.columns:
            cov = cov[cov["k"] == k]
        t2e_col, e2t_col = "train_to_eval_coverage", "eval_to_train_coverage"
        higher_is_better = True
        print(f"  Coverage format: k-NN radius fraction  k={k}  (higher fraction = better)")

    cov = cov[[train_col, bench_col, t2e_col, e2t_col]].copy()
    cov.columns = ["train", "benchmark", "t2e", "e2t"]
    return cov, higher_is_better


def _load_mmd(mmd_path: Path) -> pd.DataFrame:
    """
    Load an MMD CSV.  Both old and new format use dataset1/dataset2.
    Returns DataFrame: train, benchmark, mmd  (eval splits only)
    """
    mmd = pd.read_csv(mmd_path)
    split_col = "split2" if "split2" in mmd.columns else "eval_split"
    mmd = mmd[mmd[split_col].isin(["val", "test"])]
    mmd = mmd[["dataset1", "dataset2", "mmd"]].copy()
    mmd.columns = ["train", "benchmark", "mmd"]
    return mmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Directional HOF coverage vs flow MMD, validated against raw snapshot performance"
    )
    parser.add_argument(
        "--coverage",
        default=f"{BASE}/density_joint/coverage_v2_flow_only_raw_joint_curve_summary_q50.csv",
        help=(
            "Path to coverage CSV.  Accepts three formats: "
            "(1) bag-of-flows Q50 AUC summary (*_curve_summary_q50.csv), "
            "(2) HOF/DINO k-NN radius coverage (*_rnorm_k5.csv), "
            "(3) legacy FAISS coverage CSV."
        ),
    )
    parser.add_argument(
        "--mmd",
        default=f"{BASE}/mmd/mmd_v2_flow_joint_v1.csv",
        help="Path to flow MMD CSV",
    )
    # Performance source: snapshots (preferred) or pre-built tier0 CSV
    perf_group = parser.add_mutually_exclusive_group()
    perf_group.add_argument(
        "--snapshot-dirs",
        nargs="+",
        metavar="DIR",
        default=None,
        help=(
            "One or more snapshot root directories to scan for training_summary.txt. "
            "Peak PCK is read directly from each run's summary file."
        ),
    )
    perf_group.add_argument(
        "--perf",
        default=None,
        help="Path to pre-built tier0 performance grid CSV (fallback if --snapshot-dirs not given)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Coverage neighbor count (for HOF/DINO k-NN format, default: 5)",
    )
    parser.add_argument(
        "--eps50",
        action="store_true",
        default=True,
        help=(
            "Use eps_at50 (pixel radius at which 50%% of flows are matched) "
            "instead of AUC. Lower = better. Only applies to Q50 curve-summary CSVs. "
            "Default: on."
        ),
    )
    parser.add_argument(
        "--no-eps50",
        dest="eps50",
        action="store_false",
        help="Use AUC instead of eps_at50.",
    )
    parser.add_argument(
        "--eps",
        default="2px",
        help=(
            "Epsilon threshold for raw per-pixel coverage CSVs "
            "(e.g. '1px', '2px', '1p5px'). Ignored for AUC/k-NN formats. "
            "Higher coverage fraction = better for both t2e and e2t. "
            "Default: 2px"
        ),
    )
    parser.add_argument(
        "--exclude-benchmarks",
        nargs="+",
        metavar="BENCH",
        default=["middlebury"],
        help="Benchmarks to exclude from all output and tables (default: middlebury)",
    )
    parser.add_argument(
        "--spotlight-train",
        nargs="+",
        default=["flyingthings", "sintel"],
        help="Training datasets to spotlight (default: flyingthings sintel)",
    )
    parser.add_argument(
        "--spotlight-benchmark",
        default="tss",
        help="Eval benchmark for the spotlight comparison (default: tss)",
    )
    parser.add_argument(
        "--latex-out",
        default=None,
        help=(
            "Directory to write LaTeX tables into. "
            "Writes directional_tau_table.tex and directional_spotlight_<bench>.tex"
        ),
    )
    args = parser.parse_args()

    root = Path(".")

    # -------------------------------------------------------------------------
    # Load coverage (HOF bag-of-flows by default)
    # -------------------------------------------------------------------------
    cov_path = root / args.coverage
    if not cov_path.exists():
        raise SystemExit(f"Coverage CSV not found: {cov_path}")
    cov, cov_higher_is_better = _load_coverage(cov_path, k=args.k, eps=args.eps, use_eps50=args.eps50)
    print(f"Loaded coverage: {len(cov)} rows  [{cov_path.name}]")

    # Human-readable description of the coverage metric for table footnotes
    _eps_col = f"eval_covered_by_train_eps{args.eps}"
    if _eps_col in cov.columns or any(c.startswith("eval_covered_by_train_eps") for c in pd.read_csv(cov_path, nrows=0).columns):
        eps_label = args.eps.replace("p", ".").replace(".x", r"\,px")
        cov_description = (
            rf"fraction of flow vectors within {eps_label} of their nearest neighbour "
            rf"in the other set ($\uparrow$ higher\,=\,better)."
        )
    elif args.eps50:
        cov_description = (
            r"pixel radius at which 50\% of flows find a nearest-neighbour match "
            r"(lower\,=\,better); $>$64\,px means $<$50\% matched at max radius."
        )
    else:
        cov_description = (
            r"mean \% of flows matched within radius $\varepsilon$, "
            r"avg.\ over $\varepsilon \in [0.5,64]$\,px (higher\,=\,better)."
        )

    # -------------------------------------------------------------------------
    # Load flow MMD
    # -------------------------------------------------------------------------
    mmd_path = root / args.mmd
    if not mmd_path.exists():
        raise SystemExit(f"MMD CSV not found: {mmd_path}")
    mmd = _load_mmd(mmd_path)
    print(f"Loaded MMD:      {len(mmd)} rows  [{mmd_path.name}]")

    # -------------------------------------------------------------------------
    # Merge
    # -------------------------------------------------------------------------
    df = cov.merge(mmd, on=["train", "benchmark"], how="left")
    n_missing_mmd = df["mmd"].isna().sum()
    print(f"Merged:          {len(df)} rows  ({n_missing_mmd} missing MMD — shown as '--')")

    # -------------------------------------------------------------------------
    # Load actual performance
    # -------------------------------------------------------------------------
    perf: pd.DataFrame = pd.DataFrame()
    if args.snapshot_dirs:
        print(f"\nLoading performance from {len(args.snapshot_dirs)} snapshot dir(s)...")
        perf = _load_snapshots(args.snapshot_dirs)
    elif args.perf:
        perf_path = root / args.perf
        print(f"\nLoading performance from pre-built CSV: {perf_path}")
        perf = _load_performance_csv(perf_path)

    if perf.empty:
        print("WARNING: no performance data loaded — skipping ground-truth validation\n")
    else:
        df = df.merge(perf, on=["train", "benchmark"], how="left")
        n_matched = df["perf_mean"].notna().sum()
        print(f"Perf rows matched: {n_matched} / {len(df)}\n")

    # -------------------------------------------------------------------------
    # Exclude blacklisted benchmarks
    # -------------------------------------------------------------------------
    if args.exclude_benchmarks:
        before = len(df)
        df = df[~df["benchmark"].isin(args.exclude_benchmarks)]
        print(f"Excluded benchmarks {args.exclude_benchmarks}: {before - len(df)} rows dropped\n")

    # -------------------------------------------------------------------------
    # Per-benchmark ranking table
    # -------------------------------------------------------------------------
    for bench in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == bench].copy()
        if len(sub) < 3:
            continue

        sub = sub.sort_values("t2e", ascending=False).reset_index(drop=True)
        sub["rank_t2e"] = range(1, len(sub) + 1)

        sub = sub.sort_values("e2t", ascending=False).reset_index(drop=True)
        sub["rank_e2t"] = range(1, len(sub) + 1)

        # lower MMD = more similar = "better" training candidate
        sub = sub.sort_values("mmd", ascending=True).reset_index(drop=True)
        sub["rank_mmd"] = range(1, len(sub) + 1)

        sub["disagree_mmd_vs_t2e"] = (sub["rank_t2e"] - sub["rank_mmd"]).abs()
        sub["disagree_e2t_vs_t2e"] = (sub["rank_t2e"] - sub["rank_e2t"]).abs()

        sub = sub.sort_values("t2e", ascending=False)

        print(f"{'='*78}")
        print(f"Eval benchmark: {bench}   (n={len(sub)} training sets)")
        print(
            f"  {'Train':28s}  {'t2e':>6}  {'e2t':>6}  {'MMD':>7}  "
            f"{'r_t2e':>6}  {'r_mmd':>6}  {'|Δr|':>5}  note"
        )
        print(f"  {'-'*74}")
        for _, r in sub.iterrows():
            flag = ""
            if r["disagree_mmd_vs_t2e"] >= 3:
                flag = " ← big MMD rank error"
            print(
                f"  {r['train']:28s}  {r['t2e']:6.3f}  {r['e2t']:6.3f}  "
                f"{r['mmd']:7.4f}  {r['rank_t2e']:6.0f}  {r['rank_mmd']:6.0f}  "
                f"{r['disagree_mmd_vs_t2e']:5.0f}{flag}"
            )
        print()

    # -------------------------------------------------------------------------
    # Spotlight: user-specified training sets on user-specified benchmark
    # -------------------------------------------------------------------------
    bench = args.spotlight_benchmark
    trains = args.spotlight_train
    print(f"\n{'='*78}")
    print(f"SPOTLIGHT: {' vs '.join(trains)}  for  {bench}")
    print(f"{'='*78}")
    spot = df[df["benchmark"] == bench][["train", "t2e", "e2t", "mmd"]]
    spot = spot[spot["train"].isin(trains)].sort_values("t2e", ascending=False)
    if spot.empty:
        print(f"  No data found for benchmark={bench!r} and trains={trains!r}")
    else:
        print(spot.to_string(index=False))
        if len(spot) == 2:
            a, b = spot.iloc[0], spot.iloc[1]
            print(f"\n  MMD difference |{a['train']} - {b['train']}|: {abs(a['mmd'] - b['mmd']):.5f}")
            print(f"  t2e ratio      {a['train']} / {b['train']}:  {a['t2e'] / b['t2e']:.2f}x")
            print(f"  e2t ratio      {a['train']} / {b['train']}:  {a['e2t'] / b['e2t']:.2f}x")
            pct_mmd = abs(a["mmd"] - b["mmd"]) / max(a["mmd"], b["mmd"]) * 100
            pct_t2e = abs(a["t2e"] - b["t2e"]) / max(a["t2e"], b["t2e"]) * 100
            print(f"\n  Relative gap (MMD): {pct_mmd:.1f}%  →  symmetric metric barely distinguishes them")
            print(f"  Relative gap (t2e): {pct_t2e:.1f}%  →  directional metric shows large gap")

    # -------------------------------------------------------------------------
    # Kendall tau: t2e ranking vs MMD ranking per benchmark
    # -------------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("Kendall tau: agreement between t2e ranking and MMD ranking per benchmark")
    print("  tau=+1.0 means perfect agreement  |  tau=-1.0 means perfect inversion")
    print("  (MMD inverted so that lower distance = higher rank = better training set)")
    print(f"{'='*78}")
    print(f"  {'Benchmark':22s}  {'tau':>7}  {'p-val':>8}  {'n':>4}  interpretation")
    print(f"  {'-'*65}")
    for bench in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == bench].dropna(subset=["t2e", "mmd"])
        if len(sub) < 3:
            continue
        cov_rank = sub["t2e"].rank() if cov_higher_is_better else (-sub["t2e"]).rank()
        tau, pval = kendalltau(cov_rank, (-sub["mmd"]).rank())
        interp = (
            "strong agreement" if tau > 0.7
            else "moderate agreement" if tau > 0.4
            else "weak agreement" if tau > 0.1
            else "near-random" if tau > -0.1
            else "DISAGREE"
        )
        sig = "*" if pval < 0.05 else " "
        print(f"  {bench:22s}  {tau:+7.3f}  {pval:8.4f}{sig}  n={len(sub):3d}  {interp}")

    # -------------------------------------------------------------------------
    # Ground-truth validation: how well does each metric predict actual perf?
    # -------------------------------------------------------------------------
    if "perf_mean" in df.columns and df["perf_mean"].notna().any():
        print(f"\n\n{'='*78}")
        print("GROUND-TRUTH VALIDATION: Kendall tau vs actual peak PCK (averaged across models)")
        print("  Each metric ranked; tau vs performance rank tells us which metric best")
        print("  predicts real transfer outcomes.")
        print(f"{'='*78}")
        print(
            f"  {'Benchmark':22s}  {'tau(t2e)':>10}  {'tau(e2t)':>10}  "
            f"{'tau(MMD)':>10}  {'n':>4}  winner"
        )
        print(f"  {'-'*74}")

        for bench in sorted(df["benchmark"].unique()):
            sub = df[df["benchmark"] == bench].drop_duplicates(subset=["train"]).dropna(
                subset=["t2e", "mmd", "perf_mean"]
            )
            if len(sub) < 3:
                continue
            perf_rank = sub["perf_mean"].rank()
            _cov_sign = 1 if cov_higher_is_better else -1
            tau_t2e, p_t2e = kendalltau((_cov_sign * sub["t2e"]).rank(), perf_rank)
            tau_e2t, p_e2t = kendalltau((_cov_sign * sub["e2t"]).rank(), perf_rank)
            tau_mmd, p_mmd = kendalltau((-sub["mmd"]).rank(), perf_rank)

            best = max(
                [("t2e", tau_t2e), ("e2t", tau_e2t), ("MMD", tau_mmd)],
                key=lambda x: x[1],
            )
            sig_t2e = "*" if p_t2e < 0.05 else " "
            sig_e2t = "*" if p_e2t < 0.05 else " "
            sig_mmd = "*" if p_mmd < 0.05 else " "
            print(
                f"  {bench:22s}  "
                f"{tau_t2e:+8.3f}{sig_t2e}  {tau_e2t:+8.3f}{sig_e2t}  "
                f"{tau_mmd:+8.3f}{sig_mmd}  n={len(sub):3d}  "
                f"best={best[0]}({best[1]:+.3f})"
            )

        print()
        print("  (* = p < 0.05)")
        print()

        # Spotlight: show the actual per-training-set perf + coverage for key benchmark
        bench = args.spotlight_benchmark
        print(f"\n{'='*78}")
        print(
            f"SPOTLIGHT DETAIL: actual performance + coverage for benchmark={bench}"
        )
        print(f"{'='*78}")
        sub = df[df["benchmark"] == bench].drop_duplicates(subset=["train"]).dropna(
            subset=["perf_mean"]
        ).sort_values("perf_mean", ascending=False)
        print(
            f"  {'Train':28s}  {'PCK@5%':>7}  {'t2e':>6}  {'e2t':>6}  {'MMD':>7}  "
            f"{'r_perf':>7}  {'r_t2e':>6}"
        )
        print(f"  {'-'*75}")
        sub = sub.reset_index(drop=True)
        sub["rank_perf"] = range(1, len(sub) + 1)
        sub2 = sub.sort_values("t2e", ascending=False).reset_index(drop=True)
        sub2["rank_t2e"] = range(1, len(sub2) + 1)
        sub = sub.merge(sub2[["train", "rank_t2e"]], on="train")
        sub = sub.sort_values("perf_mean", ascending=False)
        for _, r in sub.iterrows():
            t2e_str = f"{r['t2e']:6.3f}" if pd.notna(r.get("t2e")) else "  --  "
            e2t_str = f"{r['e2t']:6.3f}" if pd.notna(r.get("e2t")) else "  --  "
            mmd_str = f"{r['mmd']:7.4f}" if pd.notna(r.get("mmd")) else "   --   "
            print(
                f"  {r['train']:28s}  {r['perf_mean']:7.1f}  {t2e_str}  "
                f"{e2t_str}  {mmd_str}  {r['rank_perf']:7.0f}  {r['rank_t2e']:6.0f}"
            )

    # -------------------------------------------------------------------------
    # Summary of biggest ranking reversals across all benchmarks
    # -------------------------------------------------------------------------
    print(f"\n\n{'='*78}")
    print("Top 10 largest MMD ranking errors (|rank_t2e - rank_mmd| across all benchmarks)")
    print(f"{'='*78}")
    all_rows = []
    for bench in df["benchmark"].unique():
        sub = df[df["benchmark"] == bench].copy()
        if len(sub) < 3:
            continue
        sub = sub.sort_values("t2e", ascending=False).reset_index(drop=True)
        sub["rank_t2e"] = range(1, len(sub) + 1)
        sub = sub.sort_values("mmd", ascending=True).reset_index(drop=True)
        sub["rank_mmd"] = range(1, len(sub) + 1)
        sub["disagree"] = (sub["rank_t2e"] - sub["rank_mmd"]).abs()
        sub["benchmark"] = bench
        all_rows.append(sub)
    if all_rows:
        all_df = pd.concat(all_rows)
        top = all_df.nlargest(10, "disagree")[
            ["benchmark", "train", "t2e", "e2t", "mmd", "rank_t2e", "rank_mmd", "disagree"]
        ]
        print(top.to_string(index=False))

    # -------------------------------------------------------------------------
    # LaTeX output
    # -------------------------------------------------------------------------
    if args.latex_out:
        latex_dir = Path(args.latex_out)
        latex_dir.mkdir(parents=True, exist_ok=True)

        if "perf_mean" in df.columns and df["perf_mean"].notna().any():
            # Table 1: tau comparison across benchmarks
            _write_tau_table(
                df,
                latex_dir / "directional_tau_table.tex",
                caption=(
                    r"Kendall $\tau$ between each distance metric and actual transfer "
                    r"performance (peak PCK, best checkpoint, averaged across architectures). "
                    r"t$\to$e = $\varepsilon$-radius coverage in the train$\to$eval direction; "
                    r"e$\to$t = coverage in the eval$\to$train direction; "
                    r"MMD = symmetric flow MMD. "
                    r"Boldface marks the best metric per row. "
                    r"Decomposing the symmetric MMD into two directed distances reveals that "
                    r"the eval$\to$train direction carries the dominant predictive signal, "
                    r"while t$\to$e is orthogonal or negatively correlated with performance. "
                    r"Significance: $^*p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$."
                ),
                label="tab:directional_tau",
            )

            # Table 2: spotlight detail for the chosen benchmark
            bench = args.spotlight_benchmark
            # Compute tau values for caption dynamically
            _spot = (
                df[df["benchmark"] == bench]
                .drop_duplicates(subset=["train"])
                .dropna(subset=["t2e", "mmd", "perf_mean"])
            )
            if len(_spot) >= 3:
                _pr = _spot["perf_mean"].rank()
                _te2t, _pe2t = kendalltau(_spot["e2t"].rank(), _pr)
                _tmmd, _pmmd = kendalltau((-_spot["mmd"]).rank(), _pr)
                _e2t_str = rf"{_te2t:+.2f}{_sig(_pe2t, math=True)}"
                _mmd_str = rf"{_tmmd:+.2f}{_sig(_pmmd, math=True)}"

                # Find the single worst MMD rank error for the caption example
                # rank 1 = best in both cases (highest PCK, lowest MMD)
                _spot2 = _spot.copy()
                _spot2["rank_perf"] = _spot2["perf_mean"].rank(ascending=False)
                _spot2["rank_mmd"] = _spot2["mmd"].rank(ascending=True)
                _spot2["rank_err"] = (_spot2["rank_perf"] - _spot2["rank_mmd"]).abs()
                _worst = _spot2.loc[_spot2["rank_err"].idxmax()]
                _ex_name = _tn(str(_worst["train"]))
                _ex_pck_r = int(_worst["rank_perf"])
                _ex_mmd_r = int(_worst["rank_mmd"])
                _mmd_example = (
                    rf"For example, {_ex_name} is PCK rank\,{_ex_pck_r} "
                    rf"but Flow MMD rank\,{_ex_mmd_r}. "
                )
            else:
                _e2t_str, _mmd_str = "?", "?"
                _mmd_example = ""
            _write_spotlight_table(
                df,
                latex_dir / f"directional_spotlight_{bench}.tex",
                benchmark=bench,
                caption=(
                    rf"Training datasets ranked by peak PCK on {_bn(bench)} "
                    r"(PCK $\alpha$\,=\,5\%, best checkpoint, avg.\ RAFT \& CATs++). "
                    r"Parenthetical numbers are ranks (1\,=\,best); \textbf{bold} = within 1 of PCK rank. "
                    r"Symmetric flow MMD conflates both directions of distributional overlap. "
                    + _mmd_example
                    + rf"Decomposing into directed flow coverage ({cov_description}), "
                    + r"the Eval$\to$Train direction better identifies "
                    r"whether evaluation flows have nearby matches in the training set."
                ),
                label=f"tab:directional_spotlight_{bench}",
                higher_is_better=cov_higher_is_better,
                cov_description=cov_description,
            )
        else:
            print("WARNING: no performance data loaded — LaTeX tables require --perf.")

        print(f"\nAll LaTeX tables written to: {latex_dir.resolve()}")


if __name__ == "__main__":
    main()
