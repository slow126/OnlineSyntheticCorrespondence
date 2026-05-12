#!/usr/bin/env python3
"""
Batch wrapper for plot_residual_fit_and_rank_errors.py.

Selects best run per parameter-matched predictor bucket (k + composition),
then writes each run's plots into its own subdirectory.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CV_SELECTION_SPECS: dict[str, dict[str, object]] = {
    "loto_pair_win": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["pairwise_win_rate", "pairwise_win_rate_micro"],
        "maximize": True,
    },
    "loto_rank_pct_err": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["abs_rank_pct_error", "abs_rank_pct_error_micro"],
        "maximize": False,
    },
    "loto_rank_spearman": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["rank_spearman", "rank_spearman_fisher", "rank_spearman_micro"],
        "maximize": True,
    },
    "loto_rank_spearman_micro": {
        "summary_file": "prediction_loto_holdout_placement_summary.csv",
        "metric_cols": ["rank_spearman_micro", "rank_spearman", "rank_spearman_fisher"],
        "maximize": True,
    },
    "joint_pair_win": {
        "summary_file": "prediction_jointood_holdout_placement_summary.csv",
        "metric_cols": ["pairwise_win_rate", "pairwise_win_rate_micro"],
        "maximize": True,
    },
    "joint_rank_pct_err": {
        "summary_file": "prediction_jointood_holdout_placement_summary.csv",
        "metric_cols": ["abs_rank_pct_error", "abs_rank_pct_error_micro"],
        "maximize": False,
    },
    "joint_rank_spearman_micro": {
        "summary_file": "prediction_jointood_holdout_placement_summary.csv",
        "metric_cols": ["rank_spearman_micro", "rank_spearman", "rank_spearman_fisher"],
        "maximize": True,
    },
    "lobo_top1": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["top1"],
        "maximize": True,
    },
    "lobo_spearman": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["spearman"],
        "maximize": True,
    },
    "lobo_cindex": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["pairwise_cindex"],
        "maximize": True,
    },
    "lobo_rank_pct_err": {
        "summary_file": "prediction_lobo_rank_summary.csv",
        "metric_cols": ["mean_abs_rank_pct_error", "median_abs_rank_pct_error"],
        "maximize": False,
    },
}


def _parse_csv_list(text: Optional[str]) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _load_metadata(run_dir: Path) -> Dict[str, object]:
    path = run_dir / "run_metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _extract_metric_value(summary_path: Path, metric_cols: Sequence[str]) -> Tuple[float, Optional[str]]:
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return float("nan"), None
    if df.empty:
        return float("nan"), None

    work = df.copy()
    if "fold" in work.columns:
        overall = work[work["fold"].astype(str) == "__overall__"]
        if not overall.empty:
            work = overall

    for col in metric_cols:
        if col not in work.columns:
            continue
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        if len(vals) == 1:
            return float(vals[0]), col
        return float(np.mean(vals)), col
    return float("nan"), None


def _predictors_from_meta(meta: Dict[str, object]) -> List[str]:
    raw = meta.get("predictors", [])
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    if isinstance(raw, str):
        return _parse_csv_list(raw)
    return []


def _bucket_from_predictors(predictors: Sequence[str], max_signal_k: int) -> Optional[Dict[str, object]]:
    preds = [str(p).strip() for p in predictors if str(p).strip()]
    k = len(preds)
    if k <= 0 or k > int(max_signal_k):
        return None

    n_flow = 0
    n_app = 0
    n_other = 0
    has_flow_mmd = False
    has_app_mmd = False
    has_other_mmd = False

    for p in preds:
        if p == "flow_mmd":
            has_flow_mmd = True
            continue
        if p in {"feature_mmd", "dino_mmd"}:
            has_app_mmd = True
            continue
        if p.endswith("_mmd"):
            has_other_mmd = True
            continue
        if p.startswith("flow_") or p.startswith("hof_"):
            n_flow += 1
            continue
        if p.startswith("dino_"):
            n_app += 1
            continue
        n_other += 1

    if n_flow == 0 and n_app == 0 and n_other == 0:
        if has_flow_mmd and has_app_mmd:
            base = "mmd_both_only"
        elif has_flow_mmd:
            base = "mmd_flow_only"
        elif has_app_mmd:
            base = "mmd_appearance_only"
        elif has_other_mmd:
            base = "mmd_other_only"
        else:
            base = "empty"
    else:
        if n_flow > 0 and n_app == 0:
            base = f"flow_only_f{n_flow}"
        elif n_app > 0 and n_flow == 0:
            base = f"appearance_only_a{n_app}"
        elif n_flow > 0 and n_app > 0:
            base = f"hybrid_f{n_flow}_a{n_app}"
        else:
            base = "other_only"
        if has_flow_mmd:
            base += "__mmd_flow"
        if has_app_mmd:
            base += "__mmd_appearance"
        if has_other_mmd:
            base += "__mmd_other"
        if n_other > 0:
            base += f"__other{n_other}"

    bucket = f"k{k:02d}__{base}"
    return {
        "bucket": bucket,
        "k": int(k),
        "n_flow": int(n_flow),
        "n_appearance": int(n_app),
        "n_other": int(n_other),
        "has_flow_mmd": bool(has_flow_mmd),
        "has_appearance_mmd": bool(has_app_mmd),
        "has_other_mmd": bool(has_other_mmd),
    }


def _collect_candidates(run_root: Path, metric_key: str, max_signal_k: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    if metric_key not in CV_SELECTION_SPECS:
        valid = ", ".join(sorted(CV_SELECTION_SPECS.keys()))
        raise ValueError(f"Unknown --best-cv-metric '{metric_key}'. Valid: {valid}")
    spec = CV_SELECTION_SPECS[metric_key]
    summary_file = str(spec["summary_file"])
    metric_cols = [str(x) for x in list(spec["metric_cols"])]
    maximize = bool(spec["maximize"])

    rows: List[Dict[str, object]] = []
    for summary_path in run_root.rglob(summary_file):
        run_dir = summary_path.parent
        if not (run_dir / "auc_with_features.csv").exists():
            continue
        metric_value, metric_col_used = _extract_metric_value(summary_path, metric_cols)
        if not np.isfinite(metric_value):
            continue
        meta = _load_metadata(run_dir)
        predictors = _predictors_from_meta(meta)
        bucket_info = _bucket_from_predictors(predictors, max_signal_k=max_signal_k)
        if bucket_info is None:
            continue
        rows.append(
            {
                "run_dir": str(run_dir),
                "summary_path": str(summary_path),
                "metric_value": float(metric_value),
                "metric_col_used": metric_col_used,
                "predictors": ",".join(predictors),
                **bucket_info,
            }
        )
    if not rows:
        return pd.DataFrame(), {"maximize": maximize, "summary_file": summary_file, "metric_cols": metric_cols}
    return pd.DataFrame(rows), {"maximize": maximize, "summary_file": summary_file, "metric_cols": metric_cols}


def _select_best_per_bucket(candidates: pd.DataFrame, maximize: bool) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    work = candidates.copy()
    work = work.sort_values(
        ["bucket", "metric_value", "run_dir"],
        ascending=[True, (not maximize), True],
        na_position="last",
    )
    out = work.groupby("bucket", as_index=False, dropna=False).head(1).reset_index(drop=True)
    return out.sort_values(["k", "bucket"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch residual/rank plotting by parameter-matched buckets.")
    parser.add_argument("--run-root", required=True, help="Root containing many leakage-free run dirs.")
    parser.add_argument(
        "--selection-csv",
        default="",
        help=(
            "Optional fixed selection manifest (parameter_matched_selection.csv). "
            "When provided, skip auto-bucketing/selection and use these buckets directly."
        ),
    )
    parser.add_argument(
        "--selection-run-root",
        default="",
        help=(
            "Optional root for fixed selection replay runs. "
            "When set with --selection-csv, each bucket maps to "
            "<selection-run-root>/leakage_free_<bucket><selection-run-suffix>."
        ),
    )
    parser.add_argument(
        "--selection-run-suffix",
        default="__density_as_interactions",
        help="Suffix for --selection-run-root mapping (default: __density_as_interactions).",
    )
    parser.add_argument(
        "--best-cv-metric",
        default="loto_pair_win",
        choices=sorted(CV_SELECTION_SPECS.keys()),
        help="CV metric used to rank and pick best run inside each bucket.",
    )
    parser.add_argument("--max-signal-k", type=int, default=8, help="Maximum predictor count k to include.")
    parser.add_argument(
        "--bucket-filter",
        default="",
        help="Optional CSV of bucket name substrings to keep (e.g., mmd_flow_only,hybrid_f1_a1).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output root. Default: <run-root>/paper_plots_param_matched_<best-cv-metric>.",
    )
    parser.add_argument("--color-by", default="both", help="Pass-through to plot script --color-by.")
    parser.add_argument(
        "--context-target-transform",
        choices=["residual", "zscore"],
        default="residual",
        help="Pass-through to plot script --context-target-transform.",
    )
    parser.add_argument(
        "--context-target-plot-space",
        choices=["model_space", "residual", "absolute"],
        default="model_space",
        help="Pass-through to plot script --context-target-plot-space.",
    )
    parser.add_argument("--top-k", type=int, default=8, help="Pass-through to plot script --top-k.")
    parser.add_argument(
        "--prediction-transform",
        choices=["none", "zscore"],
        default="none",
        help="Pass-through to plot script --prediction-transform.",
    )
    parser.add_argument(
        "--rank-detail-file",
        default="prediction_loto_holdout_placement_detail.csv",
        help="Pass-through to plot script --rank-detail-file.",
    )
    parser.add_argument(
        "--single-context",
        default="",
        help="Optional pass-through to plot script --single-context.",
    )
    parser.add_argument(
        "--single-context-color-by",
        default="train_dataset",
        help="Pass-through to plot script --single-context-color-by.",
    )
    parser.add_argument(
        "--heldout-protocols",
        default="",
        help="Optional CSV pass-through to plot script --heldout-protocols.",
    )
    parser.add_argument(
        "--heldout-plot-spaces",
        default="model_space",
        help="CSV pass-through to plot script --heldout-plot-spaces.",
    )
    parser.add_argument(
        "--heldout-model-cv-dir",
        default="",
        help="Optional pass-through to plot script --heldout-model-cv-dir.",
    )
    parser.add_argument(
        "--heldout-model-cv-head",
        default="ridge",
        help="Pass-through to plot script --heldout-model-cv-head.",
    )
    parser.add_argument(
        "--heldout-color-by",
        default="train_dataset",
        help="Pass-through to plot script --heldout-color-by.",
    )
    parser.add_argument(
        "--heldout-shape-by",
        default="",
        help="Pass-through to plot script --heldout-shape-by.",
    )
    parser.add_argument(
        "--heldout-centroid-by",
        default="",
        help="Pass-through to plot script --heldout-centroid-by.",
    )
    parser.add_argument(
        "--heldout-ellipse-by",
        default="",
        help="Pass-through to plot script --heldout-ellipse-by.",
    )
    parser.add_argument(
        "--heldout-ellipse-n-std",
        type=float,
        default=1.25,
        help="Pass-through to plot script --heldout-ellipse-n-std.",
    )
    parser.add_argument(
        "--heldout-ellipse-min-points",
        type=int,
        default=3,
        help="Pass-through to plot script --heldout-ellipse-min-points.",
    )
    parser.add_argument(
        "--heldout-ellipse-face-alpha",
        type=float,
        default=0.10,
        help="Pass-through to plot script --heldout-ellipse-face-alpha.",
    )
    parser.add_argument(
        "--heldout-ellipse-edge-alpha",
        type=float,
        default=0.95,
        help="Pass-through to plot script --heldout-ellipse-edge-alpha.",
    )
    parser.add_argument(
        "--heldout-ellipse-equal-area",
        action="store_true",
        help="Pass-through flag to plot script --heldout-ellipse-equal-area.",
    )
    parser.add_argument(
        "--heldout-ellipse-only",
        action="store_true",
        help="Pass-through flag to plot script --heldout-ellipse-only.",
    )
    parser.add_argument(
        "--heldout-center-benchmark",
        action="store_true",
        help="Pass-through flag to plot script --heldout-center-benchmark.",
    )
    parser.add_argument(
        "--heldout-center-benchmark-protocols",
        default="lobo,jointood",
        help="Pass-through to plot script --heldout-center-benchmark-protocols.",
    )
    parser.add_argument(
        "--heldout-single-context",
        default="",
        help="Pass-through to plot script --heldout-single-context.",
    )
    parser.add_argument(
        "--heldout-collapse-aggregation",
        choices=["none", "mean", "median"],
        default="none",
        help="Pass-through to plot script --heldout-collapse-aggregation.",
    )
    parser.add_argument(
        "--heldout-collapse-group-cols",
        default="",
        help="Pass-through to plot script --heldout-collapse-group-cols.",
    )
    parser.add_argument(
        "--heldout-save-points",
        action="store_true",
        help="Pass-through flag to save heldout fit points CSVs.",
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help="Pass-through flag to enable paper-ready plotting defaults.",
    )
    parser.add_argument(
        "--pretty-dataset-labels",
        action="store_true",
        help="Pass-through flag to prettify train-dataset legend labels.",
    )
    parser.add_argument(
        "--paper-synthetic-label",
        default="",
        help="Pass-through label used for synthetic dataset naming in paper mode.",
    )
    parser.add_argument(
        "--axis-clip-quantile",
        type=float,
        default=0.0,
        help="Pass-through axis clipping quantile for plotting.",
    )
    parser.add_argument(
        "--axis-pad-frac",
        type=float,
        default=0.05,
        help="Pass-through axis padding fraction for plotting (0 trims internal whitespace).",
    )
    parser.add_argument(
        "--axis-independent-limits",
        action="store_true",
        help="Pass-through flag to use independent x/y plot limits.",
    )
    parser.add_argument(
        "--hide-fit-diagnostics",
        action="store_true",
        help="Pass-through flag to hide fit diagnostics text overlays.",
    )
    parser.add_argument(
        "--hide-fit-line",
        action="store_true",
        help="Pass-through flag to hide fitted regression line.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Pass-through flag to hide figure titles.",
    )
    parser.add_argument(
        "--tight-bbox",
        action="store_true",
        help="Pass-through flag to save figures with tight bbox.",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=0.0,
        help="Pass-through base marker size for scatter plots (0 = auto/default).",
    )
    parser.add_argument(
        "--point-alpha",
        type=float,
        default=0.0,
        help="Pass-through point alpha for scatter plots (0 = auto/default).",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.0,
        help="Pass-through global font scale for scatter/legend text (0 = auto/default).",
    )
    parser.add_argument(
        "--legend-font-scale",
        type=float,
        default=1.0,
        help="Pass-through additional legend-only font scale multiplier.",
    )
    parser.add_argument(
        "--color-saturation",
        type=float,
        default=0.0,
        help="Pass-through category color saturation factor (0 = auto/default).",
    )
    parser.add_argument(
        "--mix-label-style",
        choices=["auto", "full", "short", "short_wrap"],
        default="auto",
        help="Pass-through legend mix-label style.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print selected buckets/runs.")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not args.selection_csv and not run_root.exists():
        raise FileNotFoundError(f"Run root not found: {run_root}")

    out_root = Path(args.output_dir) if args.output_dir else run_root / f"paper_plots_param_matched_{args.best_cv_metric}"
    out_root.mkdir(parents=True, exist_ok=True)

    metric_key = str(args.best_cv_metric)
    spec = CV_SELECTION_SPECS[metric_key]
    metric_cols = [str(x) for x in list(spec["metric_cols"])]
    summary_file = str(spec["summary_file"])
    maximize = bool(spec["maximize"])

    if args.selection_csv:
        sel_path = Path(args.selection_csv)
        if not sel_path.exists():
            raise FileNotFoundError(f"selection csv not found: {sel_path}")
        raw_sel = pd.read_csv(sel_path)
        if "bucket" not in raw_sel.columns:
            raise ValueError("--selection-csv must contain a 'bucket' column.")

        rows: List[Dict[str, object]] = []
        mapped_root = Path(args.selection_run_root) if str(args.selection_run_root).strip() else None
        suffix = str(args.selection_run_suffix)

        for _, row in raw_sel.iterrows():
            bucket = str(row.get("bucket") or "").strip()
            if not bucket:
                continue

            if mapped_root is not None:
                run_dir = mapped_root / f"leakage_free_{bucket}{suffix}"
            else:
                run_dir = Path(str(row.get("run_dir") or "")).expanduser()
            if not str(run_dir):
                continue
            if not run_dir.exists():
                print(f"WARNING: missing run dir for bucket {bucket}: {run_dir}")
                continue

            meta = _load_metadata(run_dir)
            predictors = _parse_csv_list(str(row.get("predictors") or "")) or _predictors_from_meta(meta)
            bucket_info = _bucket_from_predictors(predictors, max_signal_k=int(args.max_signal_k))
            if bucket_info is None:
                continue

            summary_path = run_dir / summary_file
            metric_value, metric_col_used = _extract_metric_value(summary_path, metric_cols)
            if not np.isfinite(metric_value):
                metric_value = float("nan")

            merged: Dict[str, object] = dict(row.to_dict())
            merged.update(
                {
                    "run_dir": str(run_dir),
                    "summary_path": str(summary_path),
                    "metric_value": float(metric_value),
                    "metric_col_used": metric_col_used,
                    "predictors": ",".join(predictors),
                    **bucket_info,
                }
            )
            # In fixed-selection mode, keep the explicit bucket label from the
            # selection manifest (e.g. synthetic directional HOF buckets).
            merged["bucket"] = bucket
            rows.append(merged)

        if not rows:
            raise SystemExit("No rows resolved from --selection-csv.")
        selected = pd.DataFrame(rows)
        selected = selected.sort_values(["k", "bucket"]).reset_index(drop=True)
        bucket_candidate_counts = {str(b): 1 for b in selected["bucket"].astype(str).tolist()}
        spec_info = {"maximize": maximize, "summary_file": summary_file, "metric_cols": metric_cols}
    else:
        candidates, spec_info = _collect_candidates(
            run_root=run_root,
            metric_key=metric_key,
            max_signal_k=int(args.max_signal_k),
        )
        if candidates.empty:
            raise SystemExit("No candidates found for requested root/metric/budget.")

        selected = _select_best_per_bucket(candidates, maximize=bool(spec_info["maximize"]))
        bucket_candidate_counts = (
            candidates.groupby("bucket", dropna=False).size().to_dict() if not candidates.empty else {}
        )

    filters = _parse_csv_list(args.bucket_filter)
    if filters:
        mask = np.zeros(len(selected), dtype=bool)
        for f in filters:
            mask = mask | selected["bucket"].astype(str).str.contains(f, regex=False)
        selected = selected[mask].copy().reset_index(drop=True)
    if selected.empty:
        raise SystemExit("No buckets matched --bucket-filter.")

    manifest_path = out_root / "parameter_matched_selection.csv"
    selected.to_csv(manifest_path, index=False)
    print(f"Wrote: {manifest_path}")

    if args.dry_run:
        print("Dry-run selected buckets:")
        for _, row in selected.iterrows():
            print(f"- {row['bucket']}: {row['run_dir']} | {args.best_cv_metric}={row['metric_value']:.6f}")
        return

    for _, row in selected.iterrows():
        bucket = str(row["bucket"])
        run_dir = str(row["run_dir"])
        bucket_out = out_root / bucket
        bucket_out.mkdir(parents=True, exist_ok=True)

        # Keep an explicit per-bucket record of which run was selected.
        selected_meta = {
            "selection_mode": "parameter_matched_bucket_best",
            "selection_root": str(run_root),
            "selected_run_dir": run_dir,
            "bucket": bucket,
            "bucket_k": int(row.get("k", -1)),
            "bucket_n_flow": int(row.get("n_flow", 0)),
            "bucket_n_appearance": int(row.get("n_appearance", 0)),
            "bucket_n_other": int(row.get("n_other", 0)),
            "bucket_has_flow_mmd": bool(row.get("has_flow_mmd", False)),
            "bucket_has_appearance_mmd": bool(row.get("has_appearance_mmd", False)),
            "bucket_has_other_mmd": bool(row.get("has_other_mmd", False)),
            "selection_metric_key": str(args.best_cv_metric),
            "selection_metric_value": float(row["metric_value"]),
            "selection_metric_column": str(row.get("metric_col_used") or ""),
            "selection_direction": "max" if bool(spec_info["maximize"]) else "min",
            "selection_summary_path": str(row.get("summary_path") or ""),
            "bucket_n_candidates": int(bucket_candidate_counts.get(bucket, 1)),
            "resolved_predictors": _parse_csv_list(str(row.get("predictors") or "")),
            "plot_output_dir": str(bucket_out),
            "rank_detail_file": str(args.rank_detail_file),
            "plot_args": {
                "color_by": str(args.color_by),
                "context_target_transform": str(args.context_target_transform),
                "context_target_plot_space": str(args.context_target_plot_space),
                "prediction_transform": str(args.prediction_transform),
                "top_k": int(args.top_k),
                "single_context": str(args.single_context),
                "single_context_color_by": str(args.single_context_color_by),
                "heldout_single_context": str(args.heldout_single_context),
                "heldout_collapse_aggregation": str(args.heldout_collapse_aggregation),
                "heldout_collapse_group_cols": str(args.heldout_collapse_group_cols),
                "heldout_center_benchmark": bool(args.heldout_center_benchmark),
                "heldout_center_benchmark_protocols": str(args.heldout_center_benchmark_protocols),
                "heldout_shape_by": str(args.heldout_shape_by),
                "heldout_centroid_by": str(args.heldout_centroid_by),
                "heldout_ellipse_by": str(args.heldout_ellipse_by),
                "heldout_ellipse_n_std": float(args.heldout_ellipse_n_std),
                "heldout_ellipse_min_points": int(args.heldout_ellipse_min_points),
                "heldout_ellipse_face_alpha": float(args.heldout_ellipse_face_alpha),
                "heldout_ellipse_edge_alpha": float(args.heldout_ellipse_edge_alpha),
                "heldout_ellipse_equal_area": bool(args.heldout_ellipse_equal_area),
                "heldout_ellipse_only": bool(args.heldout_ellipse_only),
                "paper_ready": bool(args.paper_ready),
                "pretty_dataset_labels": bool(args.pretty_dataset_labels),
                "paper_synthetic_label": str(args.paper_synthetic_label),
                "axis_clip_quantile": float(args.axis_clip_quantile),
                "axis_pad_frac": float(args.axis_pad_frac),
                "axis_independent_limits": bool(args.axis_independent_limits),
                "hide_fit_diagnostics": bool(args.hide_fit_diagnostics),
                "hide_fit_line": bool(args.hide_fit_line),
                "hide_title": bool(args.hide_title),
                "tight_bbox": bool(args.tight_bbox),
                "marker_size": float(args.marker_size),
                "point_alpha": float(args.point_alpha),
                "font_scale": float(args.font_scale),
                "legend_font_scale": float(args.legend_font_scale),
                "color_saturation": float(args.color_saturation),
                "mix_label_style": str(args.mix_label_style),
            },
        }
        (bucket_out / "best_cv_selection_metadata.json").write_text(
            json.dumps(selected_meta, indent=2, sort_keys=True)
        )
        (bucket_out / "best_cv_selection_README.txt").write_text(
            "\n".join(
                [
                    "Best-CV selection reference (parameter-matched bucket)",
                    "",
                    f"Selection root: {run_root}",
                    f"Bucket: {bucket}",
                    f"Selected run: {run_dir}",
                    f"Selection metric: {args.best_cv_metric}",
                    f"Metric column used: {row.get('metric_col_used')}",
                    f"Metric value: {float(row['metric_value']):.6f}",
                    f"Direction: {'maximize' if bool(spec_info['maximize']) else 'minimize'}",
                    f"Summary file: {row.get('summary_path')}",
                    f"Candidates in this bucket: {int(bucket_candidate_counts.get(bucket, 1))}",
                    "",
                    "Resolved predictors:",
                    f"- {row.get('predictors')}",
                ]
            )
            + "\n"
        )

        cmd = [
            "python",
            "scripts/plot_residual_fit_and_rank_errors.py",
            "--run-dir",
            run_dir,
            "--best-cv-metric",
            str(args.best_cv_metric),
            "--output-dir",
            str(bucket_out),
            "--color-by",
            str(args.color_by),
            "--context-target-transform",
            str(args.context_target_transform),
            "--context-target-plot-space",
            str(args.context_target_plot_space),
            "--prediction-transform",
            str(args.prediction_transform),
            "--top-k",
            str(int(args.top_k)),
            "--rank-detail-file",
            str(args.rank_detail_file),
        ]
        if str(args.heldout_protocols).strip():
            cmd.extend(["--heldout-protocols", str(args.heldout_protocols)])
        if str(args.single_context).strip():
            cmd.extend(["--single-context", str(args.single_context)])
        if str(args.single_context_color_by).strip():
            cmd.extend(["--single-context-color-by", str(args.single_context_color_by)])
        if str(args.heldout_plot_spaces).strip():
            cmd.extend(["--heldout-plot-spaces", str(args.heldout_plot_spaces)])
        if str(args.heldout_model_cv_dir).strip():
            cmd.extend(["--heldout-model-cv-dir", str(args.heldout_model_cv_dir)])
        if str(args.heldout_model_cv_head).strip():
            cmd.extend(["--heldout-model-cv-head", str(args.heldout_model_cv_head)])
        if str(args.heldout_color_by).strip():
            cmd.extend(["--heldout-color-by", str(args.heldout_color_by)])
        if str(args.heldout_shape_by).strip():
            cmd.extend(["--heldout-shape-by", str(args.heldout_shape_by)])
        if str(args.heldout_centroid_by).strip():
            cmd.extend(["--heldout-centroid-by", str(args.heldout_centroid_by)])
        if str(args.heldout_ellipse_by).strip():
            cmd.extend(["--heldout-ellipse-by", str(args.heldout_ellipse_by)])
            cmd.extend(["--heldout-ellipse-n-std", str(float(args.heldout_ellipse_n_std))])
            cmd.extend(["--heldout-ellipse-min-points", str(int(args.heldout_ellipse_min_points))])
            cmd.extend(["--heldout-ellipse-face-alpha", str(float(args.heldout_ellipse_face_alpha))])
            cmd.extend(["--heldout-ellipse-edge-alpha", str(float(args.heldout_ellipse_edge_alpha))])
            if bool(args.heldout_ellipse_equal_area):
                cmd.append("--heldout-ellipse-equal-area")
            if bool(args.heldout_ellipse_only):
                cmd.append("--heldout-ellipse-only")
        if bool(args.heldout_center_benchmark):
            cmd.append("--heldout-center-benchmark")
            if str(args.heldout_center_benchmark_protocols).strip():
                cmd.extend(
                    [
                        "--heldout-center-benchmark-protocols",
                        str(args.heldout_center_benchmark_protocols),
                    ]
                )
        if str(args.heldout_single_context).strip():
            cmd.extend(["--heldout-single-context", str(args.heldout_single_context)])
        if str(args.heldout_collapse_aggregation).strip().lower() != "none":
            cmd.extend(["--heldout-collapse-aggregation", str(args.heldout_collapse_aggregation).strip().lower()])
        if str(args.heldout_collapse_group_cols).strip():
            cmd.extend(["--heldout-collapse-group-cols", str(args.heldout_collapse_group_cols).strip()])
        if bool(args.heldout_save_points):
            cmd.append("--heldout-save-points")
        if bool(args.paper_ready):
            cmd.append("--paper-ready")
        if bool(args.pretty_dataset_labels):
            cmd.append("--pretty-dataset-labels")
        if str(args.paper_synthetic_label).strip():
            cmd.extend(["--paper-synthetic-label", str(args.paper_synthetic_label)])
        if float(args.axis_clip_quantile) > 0:
            cmd.extend(["--axis-clip-quantile", str(float(args.axis_clip_quantile))])
        if float(args.axis_pad_frac) >= 0:
            cmd.extend(["--axis-pad-frac", str(float(args.axis_pad_frac))])
        if bool(args.axis_independent_limits):
            cmd.append("--axis-independent-limits")
        if bool(args.hide_fit_diagnostics):
            cmd.append("--hide-fit-diagnostics")
        if bool(args.hide_fit_line):
            cmd.append("--hide-fit-line")
        if bool(args.hide_title):
            cmd.append("--hide-title")
        if bool(args.tight_bbox):
            cmd.append("--tight-bbox")
        if float(args.marker_size) > 0:
            cmd.extend(["--marker-size", str(float(args.marker_size))])
        if float(args.point_alpha) > 0:
            cmd.extend(["--point-alpha", str(float(args.point_alpha))])
        if float(args.font_scale) > 0:
            cmd.extend(["--font-scale", str(float(args.font_scale))])
        if float(args.legend_font_scale) > 0:
            cmd.extend(["--legend-font-scale", str(float(args.legend_font_scale))])
        if float(args.color_saturation) > 0:
            cmd.extend(["--color-saturation", str(float(args.color_saturation))])
        if str(args.mix_label_style).strip() and str(args.mix_label_style).strip().lower() != "auto":
            cmd.extend(["--mix-label-style", str(args.mix_label_style).strip().lower()])
        print(f"[bucket {bucket}] run: {run_dir}")
        subprocess.run(cmd, check=True)

    readme = out_root / "README_parameter_matched.txt"
    readme.write_text(
        "\n".join(
            [
                "Parameter-matched residual/rank plotting",
                "",
                f"run_root: {run_root}",
                f"best_cv_metric: {args.best_cv_metric}",
                f"max_signal_k: {args.max_signal_k}",
                f"bucket_filter: {args.bucket_filter}",
                "",
                "Each subdirectory name encodes: k + predictor composition.",
                "Examples:",
                "- k02__appearance_only_a2",
                "- k02__flow_only_f2",
                "- k02__hybrid_f1_a1",
                "- k01__mmd_flow_only",
                "- k02__mmd_both_only",
                "",
                f"Selection manifest: {manifest_path}",
            ]
        )
        + "\n"
    )
    print(f"Wrote: {readme}")


if __name__ == "__main__":
    main()
