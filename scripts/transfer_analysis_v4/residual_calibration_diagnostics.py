"""Residual calibration diagnostics for existing v4 prediction rows.

This script is intentionally post-hoc: it reads per-row prediction CSVs that
already exist under a v4 results directory and measures whether the residual
head is a good magnitude predictor, separate from the headline Spearman rank
claim.

Outputs:
  - residual_calibration__<target>__<head>.csv
  - RESIDUAL_CALIBRATION.md
  - calibrated residual scatter / hexbin figures

The gain-calibrated plots use one scalar per split x family x head:

    g_gain = alpha * (g - mean_context(g))
    alpha  = cov(actual_resid, pred_resid) / var(pred_resid)

This is a diagnostic calibration fit on the plotted rows, not a leakage-clean
headline score. It is useful for answering: "is the vertical-line scatter just
ridge shrinkage, or is there no residual magnitude signal?"

Example:
    python scripts/transfer_analysis_v4/residual_calibration_diagnostics.py \
        --results-dir scripts/transfer_analysis_v4/results_fsub_mean_nn \
        --target peak_pck --heads g g_zridge
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SPLITS = ["LOTO", "LOBO", "JOINT"]
FAMILIES = ["motion", "motion_sym", "motion_fid", "motion_w2",
            "appearance", "appearance_sym", "both", "random"]
FAMILY_LABEL = {
    "motion": "motion (flow)",
    "motion_sym": "motion_sym (FID+SW2+MMD)",
    "motion_fid": "motion_fid",
    "motion_w2": "motion_w2 (sliced-W2)",
    "appearance": "appearance (DINO)",
    "appearance_sym": "appearance_sym (FID+SW2+MMD)",
    "both": "both",
    "random": "random",
}
HEAD_LABEL = {
    "g": "ridge",
    "g_zridge": "z-ridge",
    "g_rank": "ranknet",
    "g_gbm": "gbm",
}


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "font.size": 9,
})


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    v = spearmanr(a, b).statistic
    return float(v) if np.isfinite(v) else np.nan


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    v = pearsonr(a, b)[0]
    return float(v) if np.isfinite(v) else np.nan


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    vx = float(np.nanvar(x))
    if vx < 1e-12:
        return np.nan
    return float(np.nanmean((x - np.nanmean(x)) * (y - np.nanmean(y))) / vx)


def _load_panel(results_dir: Path, target: str, split: str, family: str) -> pd.DataFrame | None:
    path = results_dir / "predictions" / target / f"rows_{split}_{family}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _prepare_residuals(df: pd.DataFrame, head: str) -> pd.DataFrame:
    out = df.copy()
    out["actual_resid"] = out["actual"] - out.groupby("context_id")["actual"].transform("mean")
    out["pred_resid"] = out[head] - out.groupby("context_id")[head].transform("mean")
    return out


def summarize_panel(df: pd.DataFrame, head: str) -> dict:
    df = _prepare_residuals(df, head)
    y_all = df["actual_resid"].to_numpy(float)
    x_all = df["pred_resid"].to_numpy(float)
    alpha = _slope(y_all, x_all)
    x_gain = alpha * x_all if np.isfinite(alpha) else np.full_like(x_all, np.nan)

    ctx_rows = []
    for ctx, grp in df.groupby("context_id"):
        if grp["train_dataset"].nunique() < 3:
            continue
        y = grp["actual_resid"].to_numpy(float)
        x = grp["pred_resid"].to_numpy(float)
        sy = float(np.nanstd(y, ddof=1))
        sx = float(np.nanstd(x, ddof=1))
        if not np.isfinite(sy) or sy < 1e-12:
            continue
        ctx_rows.append({
            "ctx": ctx,
            "spearman": _safe_spearman(grp["actual"].to_numpy(float), grp[head].to_numpy(float)),
            "pearson": _safe_pearson(y, x),
            "slope": _slope(y, x),
            "std_ratio": sx / sy if np.isfinite(sx) else np.nan,
        })
    ctx = pd.DataFrame(ctx_rows)

    return {
        "n_rows": int(len(df)),
        "n_contexts": int(df["context_id"].nunique()),
        "ctx_spearman_mean": float(ctx["spearman"].mean()) if len(ctx) else np.nan,
        "ctx_pearson_mean": float(ctx["pearson"].mean()) if len(ctx) else np.nan,
        "ctx_slope_median": float(ctx["slope"].median()) if len(ctx) else np.nan,
        "ctx_slope_mean": float(ctx["slope"].mean()) if len(ctx) else np.nan,
        "ctx_std_ratio_median": float(ctx["std_ratio"].median()) if len(ctx) else np.nan,
        "ctx_std_ratio_mean": float(ctx["std_ratio"].mean()) if len(ctx) else np.nan,
        "cent_spearman": _safe_spearman(y_all, x_all),
        "cent_pearson": _safe_pearson(y_all, x_all),
        "pooled_gain_alpha": float(alpha) if np.isfinite(alpha) else np.nan,
        "pooled_std_ratio": float(np.nanstd(x_all, ddof=1) / np.nanstd(y_all, ddof=1)),
        "pooled_std_ratio_gain": float(np.nanstd(x_gain, ddof=1) / np.nanstd(y_all, ddof=1))
        if np.isfinite(alpha) else np.nan,
        "resid_rmse": float(np.sqrt(np.nanmean((y_all - x_all) ** 2))),
        "resid_rmse_gain": float(np.sqrt(np.nanmean((y_all - x_gain) ** 2)))
        if np.isfinite(alpha) else np.nan,
        "frac_abs_pred_resid_le_1pt": float(np.nanmean(np.abs(x_all) <= 1.0)),
        "frac_abs_actual_resid_le_1pt": float(np.nanmean(np.abs(y_all) <= 1.0)),
    }


def collect_summary(results_dir: Path, target: str, heads: list[str]) -> pd.DataFrame:
    rows = []
    for head in heads:
        for split in SPLITS:
            for family in FAMILIES:
                df = _load_panel(results_dir, target, split, family)
                if df is None or head not in df.columns:
                    continue
                row = summarize_panel(df, head)
                row.update(target=target, head=head, split=split, family=family)
                rows.append(row)
    return pd.DataFrame(rows)


def collect_benchmark_summary(results_dir: Path, target: str, heads: list[str]) -> pd.DataFrame:
    rows = []
    for head in heads:
        for split in SPLITS:
            for family in FAMILIES:
                df = _load_panel(results_dir, target, split, family)
                if df is None or head not in df.columns or "benchmark" not in df.columns:
                    continue
                for benchmark, grp in df.groupby("benchmark"):
                    row = summarize_panel(grp, head)
                    row.update(
                        target=target,
                        head=head,
                        split=split,
                        family=family,
                        benchmark=benchmark,
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def _limit(vals: list[np.ndarray], q: float = 0.995, floor: float = 5.0) -> float:
    x = np.concatenate(vals)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return floor
    return max(floor, float(np.nanquantile(np.abs(x), q)) * 1.08)


def _plot_for_head(results_dir: Path, target: str, head: str, out_dir: Path,
                   kind: str = "scatter") -> Path | None:
    panels = {}
    xs = []
    ys = []
    for split in SPLITS:
        for family in FAMILIES:
            df = _load_panel(results_dir, target, split, family)
            if df is None or head not in df.columns:
                continue
            df = _prepare_residuals(df, head)
            y = df["actual_resid"].to_numpy(float)
            x = df["pred_resid"].to_numpy(float)
            alpha = _slope(y, x)
            df["pred_resid_gain"] = alpha * df["pred_resid"] if np.isfinite(alpha) else np.nan
            panels[(split, family)] = (df, alpha)
            xs.append(df["pred_resid_gain"].to_numpy(float))
            ys.append(df["actual_resid"].to_numpy(float))
    if not panels:
        return None

    xlim = _limit(xs)
    ylim = _limit(ys)
    lim = max(xlim, ylim)
    fig, axes = plt.subplots(len(SPLITS), len(FAMILIES),
                             figsize=(3.4 * len(FAMILIES), 3.25 * len(SPLITS)),
                             squeeze=False, constrained_layout=True)
    for ri, split in enumerate(SPLITS):
        for ci, family in enumerate(FAMILIES):
            ax = axes[ri][ci]
            panel = panels.get((split, family))
            if panel is None:
                ax.set_axis_off()
                continue
            df, alpha = panel
            x = df["pred_resid_gain"].to_numpy(float)
            y = df["actual_resid"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= lim) & (np.abs(y) <= lim)
            if kind == "hexbin":
                if m.sum() > 10:
                    ax.hexbin(x[m], y[m], gridsize=22, cmap="Blues", mincnt=1, linewidths=0)
            else:
                ax.scatter(x[m], y[m], s=14, alpha=0.45, linewidths=0, color="#1b6ca8")
            ax.plot([-lim, lim], [-lim, lim], "--", color="black", lw=0.7, alpha=0.45)
            ax.axhline(0, color="black", lw=0.5, alpha=0.35)
            ax.axvline(0, color="black", lw=0.5, alpha=0.35)

            stats = summarize_panel(df, head)
            label = (
                f"ctx rho = {stats['ctx_spearman_mean']:+.2f}\n"
                f"ctx r = {stats['ctx_pearson_mean']:+.2f}\n"
                f"alpha = {alpha:.2f}"
            )
            ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2))
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[family], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual residual", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(f"post-hoc gain-calibrated residual ({HEAD_LABEL.get(head, head)})",
                              fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Post-hoc gain-calibrated residual {kind} - {target}, {HEAD_LABEL.get(head, head)}\n"
        "Diagnostic only: alpha is fit on the plotted rows; use rank metrics for headline claims.",
        fontsize=10,
        fontweight="bold",
    )
    path = out_dir / f"fig13_gaincal_residual_{kind}__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt(v: float, digits: int = 3) -> str:
    if not np.isfinite(v):
        return "NA"
    return f"{v:+.{digits}f}" if v < 0 else f"{v:.{digits}f}"


def write_markdown(summary: pd.DataFrame, out_path: Path,
                   target: str, results_dir: Path, figure_paths: list[Path],
                   benchmark_summary: pd.DataFrame | None = None) -> None:
    lines = [
        "# Residual Calibration Diagnostics",
        "",
        "Post-hoc diagnostics for the v4 residual head. These metrics are meant",
        "to separate the ranking claim (Spearman) from residual magnitude",
        "calibration (Pearson, slope, and dispersion).",
        "",
        f"- Results dir: `{results_dir}`",
        f"- Target: `{target}`",
        "- Important: gain-calibrated figures fit one scalar alpha on the plotted",
        "  rows. They are diagnostics, not leakage-clean headline scores.",
        "",
        "## How To Read",
        "",
        "- `ctx_spearman_mean`: existing ranking metric, averaged over contexts.",
        "- `ctx_pearson_mean`: per-context linear residual association.",
        "- `ctx_std_ratio_median`: median std(predicted residual) / std(actual residual).",
        "  Values below 1 mean the residual head is under-dispersed.",
        "- `pooled_gain_alpha`: post-hoc scalar needed to calibrate pooled residual",
        "  magnitude. Values above 1 mean ridge shrinkage/compression.",
        "- `resid_rmse_gain`: residual RMSE after post-hoc scalar gain calibration.",
        "",
        "## Summary Tables",
        "",
    ]
    keep_cols = [
        "split", "family", "head", "ctx_spearman_mean", "ctx_pearson_mean",
        "ctx_std_ratio_median", "cent_pearson", "pooled_gain_alpha",
        "pooled_std_ratio", "pooled_std_ratio_gain", "resid_rmse", "resid_rmse_gain",
    ]
    for head in summary["head"].drop_duplicates():
        lines += [f"### Head: `{head}`", ""]
        sub = summary[summary["head"] == head][keep_cols].copy()
        header = "| " + " | ".join(keep_cols) + " |"
        sep = "| " + " | ".join(["---"] * len(keep_cols)) + " |"
        lines += [header, sep]
        for _, r in sub.iterrows():
            vals = []
            for c in keep_cols:
                val = r[c]
                if isinstance(val, float):
                    vals.append(_fmt(val))
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    if figure_paths:
        lines += ["## Figures", ""]
        for path in figure_paths:
            rel = path.relative_to(out_path.parent)
            lines.append(f"- [{path.name}]({rel})")
        lines.append("")

    if benchmark_summary is not None and not benchmark_summary.empty:
        lines += [
            "## LOTO Per-Benchmark Read",
            "",
            "For the interventional setting, LOTO is the closest regime: a new",
            "training source is scored against observed benchmark contexts.",
            "The table below shows the target-benchmark breakdown for the",
            "`motion` family, where the practical search signal should be judged.",
            "",
        ]
        keep = [
            "benchmark", "head", "ctx_spearman_mean", "ctx_pearson_mean",
            "ctx_std_ratio_median", "pooled_gain_alpha", "resid_rmse",
            "resid_rmse_gain",
        ]
        loto_motion = benchmark_summary[
            (benchmark_summary["split"] == "LOTO")
            & (benchmark_summary["family"] == "motion")
        ][keep].copy()
        header = "| " + " | ".join(keep) + " |"
        sep = "| " + " | ".join(["---"] * len(keep)) + " |"
        lines += [header, sep]
        for _, r in loto_motion.sort_values(["head", "benchmark"]).iterrows():
            vals = []
            for c in keep:
                val = r[c]
                if isinstance(val, float):
                    vals.append(_fmt(val))
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    out_path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("scripts/transfer_analysis_v4/results_fsub_mean_nn"))
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--heads", nargs="+", default=["g", "g_zridge"])
    args = ap.parse_args()

    out_dir = args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = collect_summary(args.results_dir, args.target, args.heads)
    if summary.empty:
        raise SystemExit(f"no prediction rows found under {args.results_dir}")
    benchmark_summary = collect_benchmark_summary(args.results_dir, args.target, args.heads)

    figure_paths = []
    for head in args.heads:
        for kind in ("scatter", "hexbin"):
            path = _plot_for_head(args.results_dir, args.target, head, out_dir, kind=kind)
            if path is not None:
                figure_paths.append(path)

    csv_path = out_dir / f"residual_calibration__{args.target}__{'_'.join(args.heads)}.csv"
    summary.to_csv(csv_path, index=False)
    bench_csv_path = out_dir / f"residual_calibration_by_benchmark__{args.target}__{'_'.join(args.heads)}.csv"
    benchmark_summary.to_csv(bench_csv_path, index=False)
    md_path = args.results_dir / "RESIDUAL_CALIBRATION.md"
    write_markdown(summary, md_path, args.target, args.results_dir,
                   figure_paths, benchmark_summary=benchmark_summary)

    print(f"csv    -> {csv_path}")
    print(f"bybench-> {bench_csv_path}")
    print(f"report -> {md_path}")
    for path in figure_paths:
        print(f"figure -> {path}")


if __name__ == "__main__":
    main()
