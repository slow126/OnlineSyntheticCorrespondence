#!/usr/bin/env python3
"""
Plot single-predictor relationship with optional family fixed effects.

Outputs:
  1) scatter + global linear fit + family-FE parallel lines
  2) within-family demeaned scatter + fit-through-origin line
  3) summary CSV with R^2 diagnostics
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _apply_transform(arr: np.ndarray, mode: str) -> np.ndarray:
    x = arr.astype(float).copy()
    if mode == "raw":
        return x

    if mode == "neg_log1p":
        shift = 0.0
        mn = float(np.min(x))
        if mn <= -1.0:
            shift = -mn + 1e-9
        return -np.log1p(x + shift)

    if mode in {"log1p", "log"}:
        shift = 0.0
        mn = float(np.min(x))
        if mn <= 0.0:
            shift = -mn + 1e-9
        if mode == "log1p":
            return np.log1p(x + shift)
        return np.log(x + shift)

    if mode == "asinh":
        return np.arcsinh(x)

    raise ValueError(f"Unknown transform: {mode}")


def _collapse_rows(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    family_col: str,
    collapse_keys: List[str],
) -> pd.DataFrame:
    if not collapse_keys:
        return df.copy()

    missing = [k for k in collapse_keys if k not in df.columns]
    if missing:
        raise ValueError(f"Missing collapse keys: {missing}")

    keys = list(collapse_keys)
    if family_col not in keys:
        keys.append(family_col)

    return (
        df.groupby(keys, as_index=False, dropna=False)[[x_col, y_col]]
        .mean(numeric_only=True)
        .copy()
    )


def _fit_family_fe(x: np.ndarray, y: np.ndarray, fam: pd.Series) -> Tuple[LinearRegression, pd.DataFrame]:
    dummies = pd.get_dummies(fam.astype(str), drop_first=True)
    X = np.column_stack([x, dummies.to_numpy()])
    model = LinearRegression().fit(X, y)
    return model, dummies


def main() -> None:
    ap = argparse.ArgumentParser(description="Single predictor plot with family fixed effects.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--x-col", required=True)
    ap.add_argument("--y-col", default="auc_normalized_observed")
    ap.add_argument("--family-col", default="model_family_encoder")
    ap.add_argument("--x-transform", default="raw", choices=["raw", "log1p", "neg_log1p", "asinh", "log"])
    ap.add_argument("--y-transform", default="log", choices=["raw", "log1p", "asinh", "log"])
    ap.add_argument("--collapse-key", action="append", default=None)
    ap.add_argument(
        "--family-filter",
        action="append",
        default=None,
        help="Optional exact family value(s) to include. Repeat flag for multiple families.",
    )
    ap.add_argument("--family-lines-top-n", type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    collapse_keys = [k for k in (args.collapse_key or []) if str(k).strip()]
    needed = [args.x_col, args.y_col, args.family_col] + collapse_keys

    df = pd.read_csv(args.input_csv, usecols=list(dict.fromkeys(needed)))
    for c in [args.x_col, args.y_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[args.x_col, args.y_col, args.family_col])
    if args.family_filter:
        allowed = set(str(x) for x in args.family_filter)
        df = df[df[args.family_col].astype(str).isin(allowed)].copy()
    if df.empty:
        raise ValueError("No rows remain after filtering. Check --family-filter values.")
    n_raw = len(df)

    df = _collapse_rows(df, args.x_col, args.y_col, args.family_col, collapse_keys)
    n_collapsed = len(df)

    x = _apply_transform(df[args.x_col].to_numpy(dtype=float), args.x_transform)
    y = _apply_transform(df[args.y_col].to_numpy(dtype=float), args.y_transform)
    fam = df[args.family_col].astype(str)

    # x-only fit
    x_model = LinearRegression().fit(x.reshape(-1, 1), y)
    y_hat_x = x_model.predict(x.reshape(-1, 1))
    r2_x = float(r2_score(y, y_hat_x))

    # family fixed effects fit
    fe_model, dummies = _fit_family_fe(x, y, fam)
    X_fe = np.column_stack([x, dummies.to_numpy()])
    y_hat_fe = fe_model.predict(X_fe)
    r2_fe = float(r2_score(y, y_hat_fe))
    beta = float(fe_model.coef_[0])

    # Family offsets for FE lines
    families = sorted(fam.unique().tolist())
    base_family = families[0]
    offsets = {base_family: 0.0}
    for i, cname in enumerate(dummies.columns):
        offsets[cname] = float(fe_model.coef_[i + 1])
    for f in families:
        offsets.setdefault(f, 0.0)

    # Plot 1: colored scatter + global line + FE family lines
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    fam_counts = fam.value_counts()
    top_fams = fam_counts.head(max(1, args.family_lines_top_n)).index.tolist()

    cmap = plt.get_cmap("tab20")
    fam_to_color = {f: cmap(i % 20) for i, f in enumerate(sorted(fam.unique()))}

    for f in sorted(fam.unique()):
        m = fam == f
        label = f if f in top_fams else None
        ax.scatter(x[m], y[m], s=12, alpha=0.28, color=fam_to_color[f], edgecolors="none", label=label)

    xs = np.linspace(float(np.min(x)), float(np.max(x)), 200)
    ax.plot(xs, x_model.predict(xs.reshape(-1, 1)), color="#f59e0b", linewidth=2.3, label="global x-only fit")

    for f in top_fams:
        b0 = float(fe_model.intercept_) + offsets[f]
        ys = beta * xs + b0
        ax.plot(xs, ys, linestyle="--", linewidth=1.4, color=fam_to_color[f], alpha=0.9)

    ax.set_xlabel(f"{args.x_col}[{args.x_transform}]")
    ax.set_ylabel(f"{args.y_col}[{args.y_transform}]")
    ax.set_title(
        f"Family FE Plot: {args.x_col} -> {args.y_col}\n"
        f"n={n_collapsed} (raw={n_raw}), R2(x)={r2_x:.3f}, R2(x+family)={r2_fe:.3f}"
    )
    ax.grid(alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        # Keep legend compact
        ax.legend(handles[: min(len(handles), 12)], labels[: min(len(labels), 12)], loc="best", frameon=True)
    fig.tight_layout()
    out1 = out_dir / f"{_sanitize(args.x_col)}__family_fe_scatter.png"
    fig.savefig(out1, dpi=180)
    plt.close(fig)

    # Plot 2: within-family demeaned relationship
    x_s = pd.Series(x)
    y_s = pd.Series(y)
    x_dm = x_s - x_s.groupby(fam).transform("mean")
    y_dm = y_s - y_s.groupby(fam).transform("mean")
    xd = x_dm.to_numpy(dtype=float)
    yd = y_dm.to_numpy(dtype=float)

    denom = float(np.sum(xd * xd))
    slope_dm = float(np.sum(xd * yd) / denom) if denom > 0 else float("nan")
    y_dm_hat = slope_dm * xd
    sst_dm = float(np.sum(yd * yd))
    sse_dm = float(np.sum((yd - y_dm_hat) ** 2))
    r2_dm = float(1.0 - sse_dm / sst_dm) if sst_dm > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(7.8, 5.8))
    ax.scatter(xd, yd, s=12, alpha=0.25, color="#2563eb", edgecolors="none")
    xs2 = np.linspace(float(np.min(xd)), float(np.max(xd)), 200)
    ax.plot(xs2, slope_dm * xs2, color="#dc2626", linewidth=2.2, label="within-family fit")
    ax.set_xlabel(f"{args.x_col}[{args.x_transform}] demeaned by family")
    ax.set_ylabel(f"{args.y_col}[{args.y_transform}] demeaned by family")
    ax.set_title(f"Within-Family Effect\nslope={slope_dm:.4f}, R2_within={r2_dm:.3f}")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    out2 = out_dir / f"{_sanitize(args.x_col)}__within_family_demeaned.png"
    fig.savefig(out2, dpi=180)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "input_csv": args.input_csv,
                "x_col": args.x_col,
                "y_col": args.y_col,
                "family_col": args.family_col,
                "x_transform": args.x_transform,
                "y_transform": args.y_transform,
                "collapse_keys": ",".join(collapse_keys),
                "family_filter": ",".join(args.family_filter or []),
                "n_rows_raw": n_raw,
                "n_rows_collapsed": n_collapsed,
                "r2_x_only": r2_x,
                "r2_x_plus_family_fe": r2_fe,
                "r2_gain_family_fe": r2_fe - r2_x,
                "within_family_slope": slope_dm,
                "within_family_r2": r2_dm,
                "plot_family_fe": str(out1),
                "plot_within_family": str(out2),
            }
        ]
    )
    out_csv = out_dir / "family_effects_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()
