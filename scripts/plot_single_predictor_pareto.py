#!/usr/bin/env python3
"""
Quick diagnostic: single predictor vs PCK-AUC with a Pareto frontier overlay.

Default predictors are aligned to the user-requested single metrics:
  - flow distance AUC
  - flow distance at single threshold
  - flow MMD
  - DINO MMD
  - DINO KL
  - HOF motion-k1 style distance
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictorSpec:
    label: str
    column: str
    better: str  # "low" or "high"


DEFAULT_SPECS: List[PredictorSpec] = [
    PredictorSpec("flow_distance_auc", "flow_eval_to_train_auc", "high"),
    PredictorSpec("flow_distance_single_raw", "flow_eval_to_train_mean_dist", "low"),
    PredictorSpec("flow_mmd", "flow_mmd", "low"),
    PredictorSpec("dino_mmd", "dino_mmd", "low"),
    PredictorSpec("dino_kl", "dino_eval_to_train_kl_div", "low"),
    PredictorSpec("hof_motion_k1", "hof_eval_to_train_mean_dist", "low"),
]


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(xr, yr)


def _linear_fit_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if x.size < 2 or np.unique(x).size < 2:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    resid = np.sum((y - y_hat) ** 2)
    tot = np.sum((y - y.mean()) ** 2)
    r2 = float("nan") if tot == 0 else float(1.0 - resid / tot)
    return float(slope), float(intercept), r2


def _pareto_frontier(x: np.ndarray, y: np.ndarray, better: str) -> Tuple[np.ndarray, np.ndarray]:
    if x.size == 0:
        return np.array([]), np.array([])
    idx = np.argsort(x)
    if better == "high":
        idx = idx[::-1]
    x_s = x[idx]
    y_s = y[idx]
    keep_x: List[float] = []
    keep_y: List[float] = []
    best_y = -np.inf
    for xi, yi in zip(x_s, y_s):
        if yi >= best_y:
            keep_x.append(float(xi))
            keep_y.append(float(yi))
            best_y = yi
    xf = np.array(keep_x)
    yf = np.array(keep_y)
    order = np.argsort(xf)
    return xf[order], yf[order]


def _parse_predictor_specs(raw_specs: Iterable[str] | None) -> List[PredictorSpec]:
    if not raw_specs:
        return DEFAULT_SPECS
    out: List[PredictorSpec] = []
    for item in raw_specs:
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"Invalid --predictor-spec '{item}'. Expected format 'label:column:low|high'."
            )
        label, column, better = parts
        better = better.lower()
        if better not in {"low", "high"}:
            raise ValueError(f"Invalid direction '{better}' in '{item}'. Use 'low' or 'high'.")
        out.append(PredictorSpec(label=label, column=column, better=better))
    return out


def _apply_target_transform(
    sub: pd.DataFrame,
    target_col: str,
    transform: str,
    group_col: str,
) -> pd.DataFrame:
    out = sub.copy()
    if transform == "raw":
        out["target_plot"] = out[target_col]
        return out

    if group_col not in out.columns:
        raise ValueError(
            f"group column '{group_col}' is required for transform '{transform}' but was not found."
        )

    g = out.groupby(group_col, dropna=False)
    if transform == "zscore_by_group":
        mu = g[target_col].transform("mean")
        sd = g[target_col].transform("std")
        out["target_plot"] = (out[target_col] - mu) / sd.replace(0.0, np.nan)
        return out

    if transform == "rank_pct_by_group":
        out["target_plot"] = g[target_col].rank(method="average", pct=True)
        return out

    if transform == "demean_by_group":
        mu = g[target_col].transform("mean")
        out["target_plot"] = out[target_col] - mu
        return out

    raise ValueError(
        f"Unknown transform '{transform}'. "
        "Use one of: raw, zscore_by_group, rank_pct_by_group, demean_by_group."
    )


def _parse_transforms(raw: Iterable[str] | None) -> List[str]:
    if not raw:
        return ["raw"]
    allowed = {"raw", "zscore_by_group", "rank_pct_by_group", "demean_by_group"}
    out: List[str] = []
    for t in raw:
        t2 = str(t).strip()
        if t2 not in allowed:
            raise ValueError(
                f"Invalid --target-transform '{t2}'. "
                "Allowed: raw, zscore_by_group, rank_pct_by_group, demean_by_group."
            )
        out.append(t2)
    return out


def _apply_x_transform(
    x: np.ndarray,
    better: str,
    x_transform: str,
) -> Tuple[np.ndarray, str]:
    vals = x.astype(float).copy()
    out_better = better

    if x_transform == "raw":
        return vals, out_better

    min_v = float(np.min(vals))
    shift = 0.0
    if min_v <= -1.0:
        shift = -min_v + 1e-9

    if x_transform == "log1p":
        vals = np.log1p(vals + shift)
        return vals, out_better

    if x_transform == "neg_log1p":
        vals = -np.log1p(vals + shift)
        out_better = "high" if better == "low" else "low"
        return vals, out_better

    if x_transform == "auto_log":
        # Heuristic: compress tails with log; flip "low is better" metrics so x grows with quality.
        if better == "low":
            vals = -np.log1p(vals + shift)
            out_better = "high"
        else:
            vals = np.log1p(vals + shift)
            out_better = "high"
        return vals, out_better

    raise ValueError(
        f"Unknown x transform '{x_transform}'. Use one of: raw, log1p, neg_log1p, auto_log."
    )


def _apply_y_transform(
    y: np.ndarray,
    y_transform: str,
) -> np.ndarray:
    vals = y.astype(float).copy()

    if y_transform == "raw":
        return vals

    min_v = float(np.min(vals))
    shift = 0.0
    if min_v <= 0.0:
        shift = -min_v + 1e-9

    if y_transform == "log1p":
        return np.log1p(vals + shift)

    if y_transform == "log":
        return np.log(vals + shift)

    if y_transform == "asinh":
        return np.arcsinh(vals)

    if y_transform == "auto_log":
        # Simple heuristic for heavy right-tail targets.
        if min_v > 0 and float(pd.Series(vals).skew()) > 1.0:
            return np.log(vals)
        return vals

    raise ValueError(
        f"Unknown y transform '{y_transform}'. Use one of: raw, log1p, log, asinh, auto_log."
    )


def _collapse_rows(
    sub: pd.DataFrame,
    collapse_keys: List[str],
    predictor_col: str,
    target_col: str,
    group_col: str,
    transform: str,
) -> pd.DataFrame:
    if not collapse_keys:
        return sub

    missing = [k for k in collapse_keys if k not in sub.columns]
    if missing:
        raise ValueError(f"Collapse key(s) not found: {missing}")

    keys = list(collapse_keys)
    if transform != "raw" and group_col not in keys:
        keys.append(group_col)

    return (
        sub.groupby(keys, dropna=False, as_index=False)[[predictor_col, target_col]]
        .mean(numeric_only=True)
        .copy()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot single predictor vs PCK-AUC with Pareto frontier.")
    parser.add_argument("--input-csv", required=True, help="Path to auc_with_features.csv")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where plots and summary CSV will be written.",
    )
    parser.add_argument(
        "--target-col",
        default="auc",
        help="Target column for PCK-AUC. Default: auc",
    )
    parser.add_argument(
        "--group-col",
        default="model_family",
        help="Group column used for grouped target transforms. Default: model_family",
    )
    parser.add_argument(
        "--target-transform",
        action="append",
        default=None,
        help=(
            "Target transform to plot (repeatable). "
            "Options: raw, zscore_by_group, rank_pct_by_group, demean_by_group. Default: raw"
        ),
    )
    parser.add_argument(
        "--predictor-spec",
        action="append",
        default=None,
        help="Optional override. Format: label:column:low|high (repeatable).",
    )
    parser.add_argument(
        "--x-transform",
        default="raw",
        help="Predictor transform: raw, log1p, neg_log1p, auto_log. Default: raw",
    )
    parser.add_argument(
        "--y-transform",
        default="raw",
        help="Target value transform: raw, log1p, log, asinh, auto_log. Default: raw",
    )
    parser.add_argument(
        "--collapse-key",
        action="append",
        default=None,
        help=(
            "Optional grouping key used to collapse duplicate rows before plotting. "
            "Repeat this flag to provide multiple keys."
        ),
    )
    args = parser.parse_args()

    in_path = Path(args.input_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    target_col = str(args.target_col)
    group_col = str(args.group_col)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in: {in_path}")

    specs = _parse_predictor_specs(args.predictor_spec)
    transforms = _parse_transforms(args.target_transform)
    x_transform = str(args.x_transform).strip()
    if x_transform not in {"raw", "log1p", "neg_log1p", "auto_log"}:
        raise ValueError(
            f"Invalid --x-transform '{x_transform}'. Use: raw, log1p, neg_log1p, auto_log."
        )
    y_transform = str(args.y_transform).strip()
    if y_transform not in {"raw", "log1p", "log", "asinh", "auto_log"}:
        raise ValueError(
            f"Invalid --y-transform '{y_transform}'. Use: raw, log1p, log, asinh, auto_log."
        )
    collapse_keys = [str(x) for x in (args.collapse_key or []) if str(x).strip()]
    rows: List[dict] = []

    for transform in transforms:
        for spec in specs:
            base_cols = [spec.column, target_col]
            if transform != "raw":
                base_cols.append(group_col)
            base_cols.extend(collapse_keys)
            base_cols = list(dict.fromkeys(base_cols))
            if spec.column not in df.columns:
                rows.append(
                    {
                        "target_transform": transform,
                        "group_col": group_col if transform != "raw" else "",
                        "collapse_keys": ",".join(collapse_keys),
                        "label": spec.label,
                        "column": spec.column,
                        "better": spec.better,
                        "n_rows_raw": 0,
                        "n_rows_collapsed": 0,
                        "n_rows": 0,
                        "status": "missing_column",
                    }
                )
                continue

            sub = df[base_cols].copy()
            sub[spec.column] = pd.to_numeric(sub[spec.column], errors="coerce")
            sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
            sub = sub.dropna(subset=[spec.column, target_col])
            n_raw = len(sub)
            try:
                sub = _collapse_rows(
                    sub,
                    collapse_keys=collapse_keys,
                    predictor_col=spec.column,
                    target_col=target_col,
                    group_col=group_col,
                    transform=transform,
                )
            except Exception as exc:
                rows.append(
                    {
                        "target_transform": transform,
                        "group_col": group_col if transform != "raw" else "",
                        "collapse_keys": ",".join(collapse_keys),
                        "label": spec.label,
                        "column": spec.column,
                        "better": spec.better,
                        "n_rows_raw": n_raw,
                        "n_rows_collapsed": 0,
                        "n_rows": 0,
                        "status": f"collapse_error: {exc}",
                    }
                )
                continue
            n_collapsed = len(sub)
            try:
                sub = _apply_target_transform(sub, target_col=target_col, transform=transform, group_col=group_col)
            except Exception as exc:
                rows.append(
                    {
                        "target_transform": transform,
                        "group_col": group_col if transform != "raw" else "",
                        "collapse_keys": ",".join(collapse_keys),
                        "label": spec.label,
                        "column": spec.column,
                        "better": spec.better,
                        "n_rows_raw": n_raw,
                        "n_rows_collapsed": n_collapsed,
                        "n_rows": len(sub),
                        "status": f"transform_error: {exc}",
                    }
                )
                continue

            sub = sub.dropna(subset=["target_plot"])
            n = len(sub)
            if n < 5:
                rows.append(
                    {
                        "target_transform": transform,
                        "group_col": group_col if transform != "raw" else "",
                        "collapse_keys": ",".join(collapse_keys),
                        "label": spec.label,
                        "column": spec.column,
                        "better": spec.better,
                        "n_rows_raw": n_raw,
                        "n_rows_collapsed": n_collapsed,
                        "n_rows": n,
                        "status": "too_few_rows",
                    }
                )
                continue

            x = sub[spec.column].to_numpy(dtype=float)
            y = sub["target_plot"].to_numpy(dtype=float)
            try:
                x_plot, better_plot = _apply_x_transform(x, spec.better, x_transform)
            except Exception as exc:
                rows.append(
                    {
                        "target_transform": transform,
                        "group_col": group_col if transform != "raw" else "",
                        "collapse_keys": ",".join(collapse_keys),
                        "x_transform": x_transform,
                        "label": spec.label,
                        "column": spec.column,
                        "better": spec.better,
                        "n_rows_raw": n_raw,
                        "n_rows_collapsed": n_collapsed,
                        "n_rows": len(sub),
                        "status": f"x_transform_error: {exc}",
                    }
                )
                continue
            try:
                y_plot = _apply_y_transform(y, y_transform)
            except Exception as exc:
                rows.append(
                    {
                        "target_transform": transform,
                        "group_col": group_col if transform != "raw" else "",
                        "collapse_keys": ",".join(collapse_keys),
                        "x_transform": x_transform,
                        "y_transform": y_transform,
                        "label": spec.label,
                        "column": spec.column,
                        "better": spec.better,
                        "n_rows_raw": n_raw,
                        "n_rows_collapsed": n_collapsed,
                        "n_rows": len(sub),
                        "status": f"y_transform_error: {exc}",
                    }
                )
                continue

            pear = _pearson(x_plot, y_plot)
            spear = _spearman(x_plot, y_plot)
            slope, intercept, r2 = _linear_fit_stats(x_plot, y_plot)
            xf, yf = _pareto_frontier(x_plot, y_plot, better_plot)

            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            ax.scatter(x_plot, y_plot, s=10, alpha=0.25, color="#2563eb", edgecolors="none", label="rows")

            if np.isfinite(slope):
                xs = np.linspace(float(np.min(x_plot)), float(np.max(x_plot)), 200)
                ys = slope * xs + intercept
                ax.plot(xs, ys, color="#f59e0b", linewidth=2.0, label="linear fit")

            if xf.size > 0:
                ax.plot(
                    xf,
                    yf,
                    color="#dc2626",
                    linewidth=2.0,
                    marker="o",
                    markersize=3,
                    label="pareto frontier",
                )

            y_name = f"{target_col}[{transform}|{y_transform}]"
            x_name = f"{spec.column}[{x_transform}]"
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.set_title(
                f"{spec.label} (x={x_transform}) vs {y_name}\n"
                f"n={n} (raw={n_raw}, collapsed={n_collapsed}), "
                f"pearson={pear:.3f}, spearman={spear:.3f}, r2={r2:.3f}, better_x={better_plot}"
            )
            ax.grid(alpha=0.2)
            ax.legend(loc="best", frameon=True)
            fig.tight_layout()

            out_png = out_dir / (
                f"{_sanitize(spec.label)}__vs__{_sanitize(target_col)}__"
                f"{_sanitize(transform)}__x_{_sanitize(x_transform)}__y_{_sanitize(y_transform)}.png"
            )
            fig.savefig(out_png, dpi=180)
            plt.close(fig)

            rows.append(
                {
                    "target_transform": transform,
                    "group_col": group_col if transform != "raw" else "",
                    "collapse_keys": ",".join(collapse_keys),
                    "x_transform": x_transform,
                    "y_transform": y_transform,
                    "label": spec.label,
                    "column": spec.column,
                    "better": spec.better,
                    "better_after_x_transform": better_plot,
                    "n_rows_raw": n_raw,
                    "n_rows_collapsed": n_collapsed,
                    "n_rows": n,
                    "pearson": pear,
                    "spearman": spear,
                    "slope": slope,
                    "intercept": intercept,
                    "r2": r2,
                    "pareto_points": int(xf.size),
                    "plot_path": str(out_png),
                    "status": "ok",
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "single_predictor_pareto_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path}")
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
