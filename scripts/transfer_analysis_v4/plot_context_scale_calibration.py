"""Plot residual calibration heads from context_scale_calibration.py outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _metrics(df: pd.DataFrame, head: str) -> dict:
    c = df.copy()
    c["actual_resid"] = c["actual"] - c.groupby("context_id")["actual"].transform("mean")
    c["pred_resid"] = c[head] - c.groupby("context_id")[head].transform("mean")
    m = np.isfinite(c["actual_resid"]) & np.isfinite(c["pred_resid"])
    x = c.loc[m, "pred_resid"].to_numpy(float)
    y = c.loc[m, "actual_resid"].to_numpy(float)
    out = {
        "pearson": float("nan"),
        "spearman": float("nan"),
        "std_ratio": float("nan"),
        "n": int(m.sum()),
    }
    if len(x) >= 3 and np.std(x) > 1e-9 and np.std(y) > 1e-9:
        out["pearson"] = float(pearsonr(x, y)[0])
        out["spearman"] = float(spearmanr(x, y).statistic)
        out["std_ratio"] = float(np.std(x, ddof=1) / np.std(y, ddof=1))
    return out


def _centered(df: pd.DataFrame, head: str) -> pd.DataFrame:
    out = df.copy()
    out["actual_resid"] = out["actual"] - out.groupby("context_id")["actual"].transform("mean")
    out["pred_resid"] = out[head] - out.groupby("context_id")[head].transform("mean")
    return out


def _axis_limit(*arrays: np.ndarray, floor: float = 1.0) -> float:
    vals = np.concatenate([np.asarray(a, float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return floor
    return max(floor, float(np.nanquantile(np.abs(vals), 0.995)) * 1.08)


def plot_pair(df: pd.DataFrame, split: str, variant_filter: str, head: str,
              out_dir: Path) -> list[Path]:
    c = _centered(df, head)
    x = c["pred_resid"].to_numpy(float)
    y = c["actual_resid"].to_numpy(float)
    lim = _axis_limit(x, y)
    stats = _metrics(df, head)
    title = (
        f"{split} / {variant_filter} / {head}\n"
        f"Pearson={stats['pearson']:+.3f}, Spearman={stats['spearman']:+.3f}, "
        f"std ratio={stats['std_ratio']:.3f}, n={stats['n']}"
    )

    paths = []
    for kind in ("scatter", "hexbin"):
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        if kind == "scatter":
            ax.scatter(x, y, s=18, alpha=0.55, linewidths=0)
        else:
            hb = ax.hexbin(x, y, gridsize=42, mincnt=1, cmap="viridis")
            fig.colorbar(hb, ax=ax, label="count")
        ax.plot([-lim, lim], [-lim, lim], color="black", lw=1, alpha=0.6)
        ax.axhline(0, color="0.7", lw=0.8)
        ax.axvline(0, color="0.7", lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("predicted residual, context-centered")
        ax.set_ylabel("actual residual, context-centered")
        ax.set_title(title, fontsize=10)
        fig.tight_layout()
        path = out_dir / f"{kind}_{split}_{variant_filter}_{head}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_grid(panels: dict[tuple[str, str], pd.DataFrame], splits: list[str],
              heads: list[str], variant_filter: str, out_dir: Path,
              kind: str) -> Path:
    fig, axes = plt.subplots(
        len(splits), len(heads),
        figsize=(4.1 * len(heads), 3.7 * len(splits)),
        squeeze=False,
    )

    centered = {}
    lims = []
    for split in splits:
        for head in heads:
            df = panels.get((split, head))
            if df is None:
                continue
            c = _centered(df, head)
            centered[(split, head)] = c
            lims.append(c["pred_resid"].to_numpy(float))
            lims.append(c["actual_resid"].to_numpy(float))
    lim = _axis_limit(*lims)

    for r, split in enumerate(splits):
        for cidx, head in enumerate(heads):
            ax = axes[r][cidx]
            cdf = centered.get((split, head))
            if cdf is None:
                ax.axis("off")
                continue
            x = cdf["pred_resid"].to_numpy(float)
            y = cdf["actual_resid"].to_numpy(float)
            if kind == "scatter":
                ax.scatter(x, y, s=12, alpha=0.45, linewidths=0)
            else:
                ax.hexbin(x, y, gridsize=32, mincnt=1, cmap="viridis")
            ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, alpha=0.6)
            ax.axhline(0, color="0.75", lw=0.6)
            ax.axvline(0, color="0.75", lw=0.6)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            stats = _metrics(cdf, head)
            ax.set_title(
                f"{split} / {head}\n"
                f"r={stats['pearson']:+.2f}, rho={stats['spearman']:+.2f}, "
                f"std={stats['std_ratio']:.2f}",
                fontsize=9,
            )
            if r == len(splits) - 1:
                ax.set_xlabel("predicted residual")
            if cidx == 0:
                ax.set_ylabel("actual residual")
    fig.suptitle(
        f"Context-centered residual calibration grid ({variant_filter})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = out_dir / f"grid_{kind}_{variant_filter}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-dir", type=Path,
                    default=Path("scripts/transfer_analysis_v4/results_fsub_mean_nn/context_scale_calibration"))
    ap.add_argument("--variant-filter", default="drop_false_true",
                    choices=["all_variants", "drop_false_true"])
    ap.add_argument("--splits", nargs="+", default=["LOTO", "LOBO", "JOINT"],
                    choices=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--heads", nargs="+",
                    default=["g", "g_shrink_gain", "g_variant_gain", "g_profilesim_gain"])
    args = ap.parse_args()

    out_dir = args.calib_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    made = []
    rows = []
    panels = {}
    for split in args.splits:
        path = args.calib_dir / f"rows_{split}_{args.variant_filter}.csv"
        df = pd.read_csv(path)
        for head in args.heads:
            if head not in df.columns:
                continue
            panels[(split, head)] = df
            made.extend(plot_pair(df, split, args.variant_filter, head, out_dir))
            row = _metrics(df, head)
            row.update(split=split, variant_filter=args.variant_filter, head=head)
            rows.append(row)

    made.append(plot_grid(panels, args.splits, args.heads, args.variant_filter,
                          out_dir, kind="scatter"))
    made.append(plot_grid(panels, args.splits, args.heads, args.variant_filter,
                          out_dir, kind="hexbin"))

    metrics = pd.DataFrame(rows)
    metrics_path = out_dir / f"plot_metrics_{args.variant_filter}.csv"
    metrics.to_csv(metrics_path, index=False)
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {len(made)} figures under {out_dir}")
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
