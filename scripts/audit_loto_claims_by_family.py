#!/usr/bin/env python3
"""
Audit LOTO claim support/counter evidence from per-family, parameter-matched outputs.

This script reads:
  <input_root>/<model_family>/k*/best_cv_selection_metadata.json
  <input_root>/<model_family>/k*/heldout_model_space_loto__model_family_encoder_<model_family>_fit_points.csv

It writes claim-audit CSVs with:
  - per-run metrics
  - pure-modality deltas (flow vs hof vs dino) per model family
  - support/counter flags for claim checks
  - parameter-matching availability (e.g., strict 2/2/2)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


CONTROL_PREDICTORS = {
    "log_n_samples_eval",
    "log_avg_flows_eval",
    "log_n_samples_train",
    "log_avg_flows_train",
}

HIGHER_IS_BETTER = {
    "spearman_global",
    "rank_spearman_macro",
    "rank_pairwise_cindex_macro",
    "rank_pairwise_cindex_micro",
}


def _safe_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _infer_method(run_dir_name: str) -> str:
    name = str(run_dir_name)
    if name.endswith("__flow_eps_raw_single"):
        return "flow"
    if name.endswith("__dino_eval_dist"):
        return "dino"
    if name.endswith("__hof_train_dist"):
        return "hof"
    if name.endswith("__flow_plus_hof"):
        return "flow_plus_hof"
    if name.endswith("__flow_plus_dino"):
        return "flow_plus_dino"
    if name.endswith("__flow_plus_dino_plus_hof"):
        return "flow_plus_dino_plus_hof"
    return name


def _signal_predictors(resolved_predictors: Iterable[str]) -> List[str]:
    out: List[str] = []
    for p in resolved_predictors:
        pp = str(p).strip()
        if not pp:
            continue
        if pp in CONTROL_PREDICTORS:
            continue
        if "_x_" in pp:
            continue
        out.append(pp)
    return out


def _count_modalities(signal_predictors: Iterable[str]) -> Tuple[int, int, int]:
    n_flow = 0
    n_appearance = 0
    n_hof = 0
    for p in signal_predictors:
        if p.startswith("flow_"):
            n_flow += 1
        elif p.startswith("dino_"):
            n_appearance += 1
        elif p.startswith("hof_"):
            n_hof += 1
    return n_flow, n_appearance, n_hof


def _pick_fit_points_path(bucket_dir: Path, family: str) -> Optional[Path]:
    exact = bucket_dir / f"heldout_model_space_loto__model_family_encoder_{family}_fit_points.csv"
    if exact.exists():
        return exact
    matches = sorted(bucket_dir.glob("heldout_model_space_loto__model_family_encoder_*_fit_points.csv"))
    return matches[0] if matches else None


def _pairwise_cindex(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, int]:
    t = y_true.to_numpy(dtype=float)
    p = y_pred.to_numpy(dtype=float)
    n = len(t)
    if n < 2:
        return float("nan"), 0

    total = 0
    score_sum = 0.0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dy = t[i] - t[j]
            if dy == 0.0:
                continue
            dp = p[i] - p[j]
            total += 1
            if dp == 0.0:
                score_sum += 0.5
            elif dy * dp > 0.0:
                score_sum += 1.0
    if total == 0:
        return float("nan"), 0
    return score_sum / float(total), total


def _abs_rank_pct_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    true_rank = y_true.rank(method="average", ascending=False)
    pred_rank = y_pred.rank(method="average", ascending=False)
    denom = float(max(n - 1, 1))
    err = ((true_rank - 1.0) / denom - (pred_rank - 1.0) / denom).abs().mean()
    return float(err)


def _compute_loto_metrics(points_df: pd.DataFrame) -> Dict[str, float]:
    y_true_col = "y_true" if "y_true" in points_df.columns else "target"
    y_pred_col = "y_pred" if "y_pred" in points_df.columns else "prediction"
    benchmark_col = "benchmark"

    y = pd.to_numeric(points_df[y_true_col], errors="coerce")
    p = pd.to_numeric(points_df[y_pred_col], errors="coerce")
    valid = y.notna() & p.notna()
    y = y[valid]
    p = p[valid]
    work = points_df.loc[valid, [benchmark_col]].copy()
    work["y_true"] = y.to_numpy()
    work["y_pred"] = p.to_numpy()

    out: Dict[str, float] = {}
    if len(work) == 0:
        return {
            "n_rows": 0.0,
            "mae_global": float("nan"),
            "rmse_global": float("nan"),
            "spearman_global": float("nan"),
            "rank_n_groups": 0.0,
            "rank_spearman_macro": float("nan"),
            "rank_pairwise_cindex_macro": float("nan"),
            "rank_pairwise_cindex_micro": float("nan"),
            "rank_abs_rank_pct_error_macro": float("nan"),
        }

    out["n_rows"] = float(len(work))
    out["mae_global"] = float((work["y_true"] - work["y_pred"]).abs().mean())
    out["rmse_global"] = float(((work["y_true"] - work["y_pred"]) ** 2).mean() ** 0.5)
    out["spearman_global"] = float(work["y_true"].corr(work["y_pred"], method="spearman"))

    group_spearman: List[float] = []
    group_cindex: List[float] = []
    group_abs_rank_err: List[float] = []
    cindex_weighted_sum = 0.0
    cindex_weighted_n = 0
    n_groups = 0

    for _, g in work.groupby(benchmark_col, dropna=False):
        if len(g) < 2:
            continue
        n_groups += 1
        s = float(g["y_true"].corr(g["y_pred"], method="spearman"))
        if math.isfinite(s):
            group_spearman.append(s)
        c, n_pairs = _pairwise_cindex(g["y_true"], g["y_pred"])
        if math.isfinite(c):
            group_cindex.append(c)
        if n_pairs > 0 and math.isfinite(c):
            cindex_weighted_sum += c * float(n_pairs)
            cindex_weighted_n += n_pairs
        e = _abs_rank_pct_error(g["y_true"], g["y_pred"])
        if math.isfinite(e):
            group_abs_rank_err.append(e)

    out["rank_n_groups"] = float(n_groups)
    out["rank_spearman_macro"] = float(pd.Series(group_spearman).mean()) if group_spearman else float("nan")
    out["rank_pairwise_cindex_macro"] = float(pd.Series(group_cindex).mean()) if group_cindex else float("nan")
    out["rank_pairwise_cindex_micro"] = (
        float(cindex_weighted_sum / float(cindex_weighted_n)) if cindex_weighted_n > 0 else float("nan")
    )
    out["rank_abs_rank_pct_error_macro"] = (
        float(pd.Series(group_abs_rank_err).mean()) if group_abs_rank_err else float("nan")
    )
    return out


def _oriented_delta(metric: str, a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    raw = float(a - b)
    if metric in HIGHER_IS_BETTER:
        return raw
    return -raw


def _load_rows(input_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for meta_path in sorted(input_root.glob("*/k*/best_cv_selection_metadata.json")):
        family = meta_path.parent.parent.name
        bucket = meta_path.parent.name
        meta = json.loads(meta_path.read_text())
        run_dir = str(meta.get("selected_run_dir", ""))
        if not run_dir:
            continue
        run_name = Path(run_dir).name
        method = _infer_method(run_name)
        resolved = list(meta.get("resolved_predictors", []) or [])
        sig = _signal_predictors(resolved)
        n_flow, n_appearance, n_hof = _count_modalities(sig)

        fit_points = _pick_fit_points_path(meta_path.parent, family)
        if fit_points is None or not fit_points.exists():
            continue
        points_df = pd.read_csv(fit_points)
        metrics = _compute_loto_metrics(points_df)

        rows.append(
            {
                "model_family": family,
                "bucket": bucket,
                "method": method,
                "selected_run_name": run_name,
                "selected_run_dir": run_dir,
                "selection_metric_key": meta.get("selection_metric_key"),
                "selection_metric_column": meta.get("selection_metric_column"),
                "selection_metric_value": _safe_float(meta.get("selection_metric_value")),
                "n_signal": len(sig),
                "n_flow": n_flow,
                "n_appearance": n_appearance,
                "n_hof": n_hof,
                "signal_predictors": ",".join(sig),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _pure_modality_subset(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keep = {"flow", "dino", "hof"}
    out = df[df["method"].isin(keep)].copy()
    return out


def _build_delta_table(pure_df: pd.DataFrame, primary_metric: str, close_gap: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metrics = [
        "rank_pairwise_cindex_macro",
        "rank_spearman_macro",
        "spearman_global",
        "rmse_global",
        "mae_global",
        "rank_abs_rank_pct_error_macro",
    ]
    for family, g in pure_df.groupby("model_family", dropna=False):
        by_method = {m: g[g["method"] == m].iloc[0] for m in ("flow", "hof", "dino") if not g[g["method"] == m].empty}
        if not {"flow", "hof", "dino"}.issubset(by_method.keys()):
            continue
        row: Dict[str, object] = {"model_family": family}
        for m in metrics:
            f = _safe_float(by_method["flow"].get(m))
            h = _safe_float(by_method["hof"].get(m))
            d = _safe_float(by_method["dino"].get(m))
            row[f"{m}__flow"] = f
            row[f"{m}__hof"] = h
            row[f"{m}__dino"] = d
            row[f"{m}__flow_minus_hof_raw"] = f - h if math.isfinite(f) and math.isfinite(h) else float("nan")
            row[f"{m}__hof_minus_dino_raw"] = h - d if math.isfinite(h) and math.isfinite(d) else float("nan")
            row[f"{m}__flow_minus_dino_raw"] = f - d if math.isfinite(f) and math.isfinite(d) else float("nan")
            row[f"{m}__flow_minus_hof_oriented"] = _oriented_delta(m, f, h)
            row[f"{m}__hof_minus_dino_oriented"] = _oriented_delta(m, h, d)
            row[f"{m}__flow_minus_dino_oriented"] = _oriented_delta(m, f, d)

        # primary-claim flags on selected metric
        f = _safe_float(by_method["flow"].get(primary_metric))
        h = _safe_float(by_method["hof"].get(primary_metric))
        d = _safe_float(by_method["dino"].get(primary_metric))
        if primary_metric in HIGHER_IS_BETTER:
            row["support_flow_gt_hof"] = bool(math.isfinite(f) and math.isfinite(h) and f > h)
            row["support_hof_gt_dino"] = bool(math.isfinite(h) and math.isfinite(d) and h > d)
            row["support_flow_gt_dino"] = bool(math.isfinite(f) and math.isfinite(d) and f > d)
            row["support_order_flow_hof_dino"] = bool(
                math.isfinite(f) and math.isfinite(h) and math.isfinite(d) and (f > h > d)
            )
            row["flow_hof_gap"] = f - h if math.isfinite(f) and math.isfinite(h) else float("nan")
        else:
            row["support_flow_gt_hof"] = bool(math.isfinite(f) and math.isfinite(h) and f < h)
            row["support_hof_gt_dino"] = bool(math.isfinite(h) and math.isfinite(d) and h < d)
            row["support_flow_gt_dino"] = bool(math.isfinite(f) and math.isfinite(d) and f < d)
            row["support_order_flow_hof_dino"] = bool(
                math.isfinite(f) and math.isfinite(h) and math.isfinite(d) and (f < h < d)
            )
            row["flow_hof_gap"] = h - f if math.isfinite(f) and math.isfinite(h) else float("nan")
        row["support_hof_close_to_flow"] = bool(
            math.isfinite(_safe_float(row["flow_hof_gap"])) and _safe_float(row["flow_hof_gap"]) <= float(close_gap)
        )
        row["primary_metric"] = primary_metric
        rows.append(row)
    return pd.DataFrame(rows)


def _support_summary(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    flags = [
        "support_flow_gt_hof",
        "support_hof_gt_dino",
        "support_flow_gt_dino",
        "support_order_flow_hof_dino",
        "support_hof_close_to_flow",
    ]
    rows: List[Dict[str, object]] = []
    n = float(len(delta_df))
    for flag in flags:
        if flag not in delta_df.columns:
            continue
        count = int(pd.to_numeric(delta_df[flag], errors="coerce").fillna(0).astype(int).sum())
        rows.append({"claim_flag": flag, "support_count": count, "counter_count": int(n - count), "support_frac": count / n})
    return pd.DataFrame(rows)


def _support_summary_by_metric(delta_df: pd.DataFrame) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()
    metric_bases = [
        "rank_pairwise_cindex_macro",
        "rank_spearman_macro",
        "spearman_global",
        "rmse_global",
        "mae_global",
        "rank_abs_rank_pct_error_macro",
    ]
    rows: List[Dict[str, object]] = []
    n = float(len(delta_df))
    for m in metric_bases:
        a = pd.to_numeric(delta_df.get(f"{m}__flow_minus_hof_oriented"), errors="coerce")
        b = pd.to_numeric(delta_df.get(f"{m}__hof_minus_dino_oriented"), errors="coerce")
        c = pd.to_numeric(delta_df.get(f"{m}__flow_minus_dino_oriented"), errors="coerce")
        support_flow_hof = int((a > 0).fillna(False).sum()) if a is not None else 0
        support_hof_dino = int((b > 0).fillna(False).sum()) if b is not None else 0
        support_flow_dino = int((c > 0).fillna(False).sum()) if c is not None else 0
        support_order = int(((a > 0) & (b > 0)).fillna(False).sum()) if (a is not None and b is not None) else 0
        rows.append(
            {
                "metric": m,
                "support_flow_gt_hof_count": support_flow_hof,
                "support_hof_gt_dino_count": support_hof_dino,
                "support_flow_gt_dino_count": support_flow_dino,
                "support_order_flow_hof_dino_count": support_order,
                "n_families": int(n),
                "support_flow_gt_hof_frac": support_flow_hof / n if n > 0 else float("nan"),
                "support_hof_gt_dino_frac": support_hof_dino / n if n > 0 else float("nan"),
                "support_flow_gt_dino_frac": support_flow_dino / n if n > 0 else float("nan"),
                "support_order_flow_hof_dino_frac": support_order / n if n > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _matching_availability(pure_df: pd.DataFrame, target_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pure_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    per_family: List[Dict[str, object]] = []
    nearest_rows: List[Dict[str, object]] = []
    for fam, g in pure_df.groupby("model_family", dropna=False):
        for modality, col, method_name in [
            ("flow", "n_flow", "flow"),
            ("appearance", "n_appearance", "dino"),
            ("hof", "n_hof", "hof"),
        ]:
            gg = g[g["method"] == method_name].copy()
            if gg.empty:
                per_family.append(
                    {
                        "model_family": fam,
                        "modality": modality,
                        "target_k": int(target_k),
                        "n_candidates": 0,
                        "has_exact_match": False,
                    }
                )
                continue

            vals = pd.to_numeric(gg[col], errors="coerce")
            has_exact = bool((vals == float(target_k)).any())
            per_family.append(
                {
                    "model_family": fam,
                    "modality": modality,
                    "target_k": int(target_k),
                    "n_candidates": int(len(gg)),
                    "has_exact_match": has_exact,
                }
            )

            # nearest candidate by |count-target|, then better pairwise macro
            gg = gg.copy()
            gg["count_col"] = vals
            gg["k_diff"] = (gg["count_col"] - float(target_k)).abs()
            gg = gg.sort_values(["k_diff", "rank_pairwise_cindex_macro"], ascending=[True, False])
            best = gg.iloc[0]
            nearest_rows.append(
                {
                    "model_family": fam,
                    "modality": modality,
                    "target_k": int(target_k),
                    "selected_method": best["method"],
                    "selected_bucket": best["bucket"],
                    "selected_run_name": best["selected_run_name"],
                    "selected_count": _safe_float(best["count_col"]),
                    "k_diff": _safe_float(best["k_diff"]),
                    "rank_pairwise_cindex_macro": _safe_float(best.get("rank_pairwise_cindex_macro")),
                    "rank_spearman_macro": _safe_float(best.get("rank_spearman_macro")),
                    "spearman_global": _safe_float(best.get("spearman_global")),
                    "rmse_global": _safe_float(best.get("rmse_global")),
                    "mae_global": _safe_float(best.get("mae_global")),
                }
            )

    return pd.DataFrame(per_family), pd.DataFrame(nearest_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LOTO claim support/counter by model family.")
    parser.add_argument("--input-root", required=True, help="Path to .../paper_plots_..._by_model_family")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <input-root>/claim_audit)")
    parser.add_argument(
        "--primary-metric",
        default="rank_pairwise_cindex_macro",
        choices=[
            "rank_pairwise_cindex_macro",
            "rank_pairwise_cindex_micro",
            "rank_spearman_macro",
            "spearman_global",
            "rmse_global",
            "mae_global",
        ],
        help="Metric used for support/counter claim flags.",
    )
    parser.add_argument(
        "--close-gap",
        type=float,
        default=0.05,
        help="Max allowed flow-vs-hof primary-metric gap for 'hof close to flow'.",
    )
    parser.add_argument(
        "--target-k",
        type=int,
        default=2,
        help="Target signal count for parameter-matching availability report.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    if not input_root.exists():
        raise SystemExit(f"Missing input root: {input_root}")
    output_dir = Path(args.output_dir) if args.output_dir else (input_root / "claim_audit")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_df = _load_rows(input_root)
    if rows_df.empty:
        raise SystemExit("No rows found. Check input path and expected files.")

    pure_df = _pure_modality_subset(rows_df)
    delta_df = _build_delta_table(pure_df, args.primary_metric, float(args.close_gap))
    summary_df = _support_summary(delta_df)
    summary_by_metric_df = _support_summary_by_metric(delta_df)
    availability_df, nearest_df = _matching_availability(pure_df, int(args.target_k))

    rows_df.sort_values(["model_family", "bucket"]).to_csv(output_dir / "loto_claim_audit_rows.csv", index=False)
    pure_df.sort_values(["model_family", "method"]).to_csv(output_dir / "loto_pure_modality_rows.csv", index=False)
    delta_df.sort_values(["model_family"]).to_csv(output_dir / "loto_pure_modality_deltas_by_family.csv", index=False)
    summary_df.to_csv(output_dir / "loto_support_counter_summary.csv", index=False)
    summary_by_metric_df.to_csv(output_dir / "loto_support_counter_by_metric.csv", index=False)
    availability_df.sort_values(["model_family", "modality"]).to_csv(
        output_dir / "loto_parameter_matching_availability.csv", index=False
    )
    nearest_df.sort_values(["model_family", "modality"]).to_csv(
        output_dir / f"loto_parameter_matching_nearest_k{int(args.target_k)}.csv", index=False
    )

    print(f"Wrote claim audit CSVs to {output_dir}")


if __name__ == "__main__":
    main()
