"""Z-scored residual diagnostics for v4 prediction rows.

This is a plotting diagnostic, not a model change.  For each context
(`context_id` = benchmark|variant), it plots:

    y = (actual - mean(actual_context)) / std(actual_context)
    x = (pred   - mean(pred_context))   / std(actual_context)

Using the actual-context std on both axes keeps calibration visible in
standardized performance units.  Contexts with near-zero actual spread are
skipped because z-scoring would amplify noise.

Example:
    python scripts/transfer_analysis_v4/zscore_residual_diagnostics.py \
        --results-dir scripts/transfer_analysis_v4/results_fsub_mean_nn \
        --target peak_pck --head g
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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


def _zscore_rows(df: pd.DataFrame, head: str, min_ctx_std: float) -> tuple[pd.DataFrame, dict]:
    rows = []
    skipped_low_std = 0
    skipped_small_n = 0
    ctx_rhos = []
    std_ratios = []

    for ctx, grp in df.groupby("context_id"):
        if grp["train_dataset"].nunique() < 3:
            skipped_small_n += 1
            continue
        y = grp["actual"].to_numpy(float)
        x = grp[head].to_numpy(float)
        y_sd = float(np.std(y, ddof=1))
        if not np.isfinite(y_sd) or y_sd < min_ctx_std:
            skipped_low_std += 1
            continue

        x_ctr = x - float(np.mean(x))
        y_ctr = y - float(np.mean(y))
        x_sd = float(np.std(x, ddof=1))
        if np.isfinite(x_sd):
            std_ratios.append(x_sd / y_sd)
        if np.isfinite(x_sd) and x_sd > 1e-12:
            rho = spearmanr(y, x).statistic
            if np.isfinite(rho):
                ctx_rhos.append(float(rho))

        z = grp.copy()
        z["actual_z"] = y_ctr / y_sd
        z["pred_z_actual_scale"] = x_ctr / y_sd
        z["ctx_actual_std"] = y_sd
        rows.append(z)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    stats = {
        "n_rows": int(len(out)),
        "n_contexts": int(out["context_id"].nunique()) if len(out) else 0,
        "skipped_low_std_contexts": int(skipped_low_std),
        "skipped_small_n_contexts": int(skipped_small_n),
        "mean_ctx_spearman": float(np.mean(ctx_rhos)) if ctx_rhos else np.nan,
        "median_std_ratio_pred_over_actual": float(np.median(std_ratios)) if std_ratios else np.nan,
        "mean_std_ratio_pred_over_actual": float(np.mean(std_ratios)) if std_ratios else np.nan,
        "frac_abs_actual_z_le_025": float(np.mean(np.abs(out["actual_z"]) <= 0.25)) if len(out) else np.nan,
        "frac_abs_pred_z_le_025": float(np.mean(np.abs(out["pred_z_actual_scale"]) <= 0.25)) if len(out) else np.nan,
    }
    return out, stats


def _load_panels(pred_dir: Path, target: str, head: str, min_ctx_std: float):
    target_dir = pred_dir / target
    panels = {}
    stats_rows = []
    for split in SPLITS:
        for fam in FAMILIES:
            path = target_dir / f"rows_{split}_{fam}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if head not in df.columns:
                continue
            zdf, stats = _zscore_rows(df, head, min_ctx_std)
            if zdf.empty:
                continue
            panels[(split, fam)] = zdf
            stats.update(split=split, family=fam, head=head, target=target)
            stats_rows.append(stats)
    return panels, pd.DataFrame(stats_rows)


def _panel_limit(panels: dict, col: str, q: float = 0.995, floor: float = 2.5) -> float:
    vals = [p[col].to_numpy(float) for p in panels.values()]
    vals = np.concatenate(vals)
    lim = float(np.quantile(np.abs(vals[np.isfinite(vals)]), q))
    return max(floor, min(5.0, lim * 1.08))


def scatter_figure(panels: dict, out_dir: Path, target: str, head: str) -> Path:
    xlim = _panel_limit(panels, "pred_z_actual_scale")
    ylim = _panel_limit(panels, "actual_z")

    fig, axes = plt.subplots(
        len(SPLITS), len(FAMILIES),
        figsize=(3.4 * len(FAMILIES), 3.25 * len(SPLITS)),
        squeeze=False,
        constrained_layout=True,
    )
    for ri, split in enumerate(SPLITS):
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            df = panels.get((split, fam))
            if df is None:
                ax.set_axis_off()
                continue
            x = df["pred_z_actual_scale"].to_numpy(float)
            y = df["actual_z"].to_numpy(float)
            ax.scatter(x, y, s=14, alpha=0.42, linewidths=0, color="#1b6ca8")
            ax.plot([-xlim, xlim], [-xlim, xlim], "--", color="black", lw=0.7, alpha=0.45)
            ax.axhline(0, color="black", lw=0.5, alpha=0.35)
            ax.axvline(0, color="black", lw=0.5, alpha=0.35)

            rs = []
            ratios = []
            for _, grp in df.groupby("context_id"):
                if grp[head].std() > 1e-12:
                    rho = spearmanr(grp["actual"], grp[head]).statistic
                    if np.isfinite(rho):
                        rs.append(rho)
                ratios.append(float(grp[head].std(ddof=1) / grp["actual"].std(ddof=1)))
            label = (
                f"ctx rho = {np.mean(rs):+.2f}\n"
                f"med sd ratio = {np.median(ratios):.2f}\n"
                f"n_ctx = {df.context_id.nunique()}"
            )
            ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2))
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual residual z", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(f"predicted residual z ({HEAD_LABEL.get(head, head)})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Within-context standardized residual scatter - {target}, mean_nn, {HEAD_LABEL.get(head, head)}\n"
        "Both axes are centered within context; both are divided by the actual context std.",
        fontsize=10,
        fontweight="bold",
    )
    path = out_dir / f"fig11_zresid_scatter__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def hexbin_figure(panels: dict, out_dir: Path, target: str, head: str) -> Path:
    xlim = _panel_limit(panels, "pred_z_actual_scale")
    ylim = _panel_limit(panels, "actual_z")

    fig, axes = plt.subplots(
        len(SPLITS), len(FAMILIES),
        figsize=(3.4 * len(FAMILIES), 3.25 * len(SPLITS)),
        squeeze=False,
        constrained_layout=True,
    )
    for ri, split in enumerate(SPLITS):
        for ci, fam in enumerate(FAMILIES):
            ax = axes[ri][ci]
            df = panels.get((split, fam))
            if df is None:
                ax.set_axis_off()
                continue
            x = df["pred_z_actual_scale"].to_numpy(float)
            y = df["actual_z"].to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y) & (np.abs(x) <= xlim) & (np.abs(y) <= ylim)
            if m.sum() > 10:
                ax.hexbin(x[m], y[m], gridsize=22, cmap="Blues", mincnt=1, linewidths=0)
            ax.plot([-xlim, xlim], [-xlim, xlim], "--", color="black", lw=0.7, alpha=0.45)
            ax.axhline(0, color="black", lw=0.5, alpha=0.35)
            ax.axvline(0, color="black", lw=0.5, alpha=0.35)

            rs = []
            ratios = []
            for _, grp in df.groupby("context_id"):
                if grp[head].std() > 1e-12:
                    rho = spearmanr(grp["actual"], grp[head]).statistic
                    if np.isfinite(rho):
                        rs.append(rho)
                ratios.append(float(grp[head].std(ddof=1) / grp["actual"].std(ddof=1)))
            label = (
                f"ctx rho = {np.mean(rs):+.2f}\n"
                f"med sd ratio = {np.median(ratios):.2f}\n"
                f"n_ctx = {df.context_id.nunique()}"
            )
            ax.text(0.04, 0.96, label, transform=ax.transAxes, va="top", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=2))
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-ylim, ylim)
            if ri == 0:
                ax.set_title(FAMILY_LABEL[fam], fontsize=9, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{split}\nactual residual z", fontsize=9)
            if ri == len(SPLITS) - 1:
                ax.set_xlabel(f"predicted residual z ({HEAD_LABEL.get(head, head)})", fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        f"Within-context standardized residual hexbin - {target}, mean_nn, {HEAD_LABEL.get(head, head)}\n"
        "This removes cross-context scale differences; under-dispersion remains visible as x-axis compression.",
        fontsize=10,
        fontweight="bold",
    )
    path = out_dir / f"fig12_zresid_hexbin__{target}__{head}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("scripts/transfer_analysis_v4/results_fsub_mean_nn"))
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--head", default="g", choices=["g", "g_zridge", "g_rank", "g_gbm"])
    ap.add_argument("--min-ctx-std", type=float, default=1e-6)
    args = ap.parse_args()

    pred_dir = args.results_dir / "predictions"
    out_dir = args.results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    panels, stats = _load_panels(pred_dir, args.target, args.head, args.min_ctx_std)
    if not panels:
        raise SystemExit(f"no panels found under {pred_dir / args.target}")

    scatter_path = scatter_figure(panels, out_dir, args.target, args.head)
    hexbin_path = hexbin_figure(panels, out_dir, args.target, args.head)
    stats_path = out_dir / f"zresid_stats__{args.target}__{args.head}.csv"
    stats.to_csv(stats_path, index=False)

    print(f"scatter -> {scatter_path}")
    print(f"hexbin  -> {hexbin_path}")
    print(f"stats   -> {stats_path}")


if __name__ == "__main__":
    main()
