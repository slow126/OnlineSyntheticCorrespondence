#!/usr/bin/env python3
"""
Build a paper-ready utility-guided intervention table from a fixed fitted run.

This script uses:
  1) Observed transfer metrics from auc_with_features.csv
  2) Predicted utility from prediction_<source>_rows.csv
  3) The exact 4-predictor recipe from run_metadata.json

Outputs:
  - utility_guided_design_table.csv
  - utility_guided_design_table.tex
  - utility_guided_design_table_metadata.csv
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = (
    "analysis_comprehensive_runs/"
    "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/"
    "density_joint/leakage_free_flow_eps_raw_joint_auc_at95"
)
DEFAULT_OUTPUT_DIR = "analysis/utility_guided_design_tables"
DEFAULT_VARIANTS = (
    "synthetic,synthetic_small_zoom,synthetic_large_zoom,synthetic_random_flipping"
)
DEFAULT_BENCHMARK = "kitti2015"
DEFAULT_MODEL_FAMILY = "catspp"
DEFAULT_PRED_SOURCE = "lobo"
DEFAULT_OBS_COL = "peak_pck"

DEFAULT_INTERVENTION_LABELS = {
    "synthetic": "baseline",
    "synthetic_small_zoom": "target-zoom",
    "synthetic_large_zoom": "high-zoom",
    "synthetic_random_flipping": "flip-mismatch",
    "synthetic_2d_warp": "2D-warp mismatch",
}

DEFAULT_VARIANT_DISPLAY = {
    "synthetic": "SDF-Fractal3D (base)",
    "synthetic_small_zoom": "+ small-zoom",
    "synthetic_large_zoom": "+ large-zoom",
    "synthetic_random_flipping": "+ random-flip",
    "synthetic_2d_warp": "+ 2D-warp",
}


def _split_csv_arg(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required CSV: {path}")
    return pd.read_csv(path)


def _read_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Missing required metadata file: {path}")
    return json.loads(path.read_text())


def _fmt_num(x: object, digits: int = 3) -> str:
    v = _as_float(x)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def _latex_escape(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = []
    for ch in str(text):
        out.append(repl.get(ch, ch))
    return "".join(out)


def _to_texsc_token(text: str) -> str:
    token = str(text).strip()
    token = token.replace("_", r"\_")
    if token.lower() == "catspp":
        token = "cats++"
    return token


def _build_latex_table(
    df: pd.DataFrame, caption: str, label: str, include_rank: bool
) -> str:
    def _fmt_signed(x: object, digits: int) -> str:
        v = _as_float(x)
        if not np.isfinite(v):
            return "--"
        return f"{v:+.{digits}f}"

    work = df.copy()
    work = work.reset_index(drop=True)

    has_appearance = {"A_align_EtoT", "A_align_TtoE", "A_delta_norm"}.issubset(work.columns)

    # Bold best values by metric direction (higher is better unless otherwise noted).
    if has_appearance:
        higher_better_cols = [
            "PCK@5%",
            "DeltaObs",
            "PredResidual",
            "PredCalibrated",
            "F_cov_EtoT_eps1p5",
            "F_cov_TtoE_eps1p0",
            "A_align_EtoT",
            "A_align_TtoE",
        ]
        lower_better_cols = ["A_delta_norm"]
        if include_rank:
            lower_better_cols.append("rank_cal")
    else:
        higher_better_cols = [
            "PCK@5%",
            "DeltaObs",
            "PredResidual",
            "PredCalibrated",
            "F_auc_EtoT",
            "F_auc_TtoE",
        ]
        lower_better_cols = ["Flow_MMD"]

    best_masks: Dict[str, np.ndarray] = {}
    for col in higher_better_cols:
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            best_masks[col] = np.zeros_like(vals, dtype=bool)
            continue
        best = np.nanmax(vals)
        best_masks[col] = np.isclose(vals, best, equal_nan=False, rtol=1e-12, atol=1e-12)
    for col in lower_better_cols:
        vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            best_masks[col] = np.zeros_like(vals, dtype=bool)
            continue
        best = np.nanmin(vals)
        best_masks[col] = np.isclose(vals, best, equal_nan=False, rtol=1e-12, atol=1e-12)

    def _maybe_bold(text: str, cond: bool) -> str:
        return f"\\textbf{{{text}}}" if cond else text

    all_row_lines: List[str] = []
    for i, r in work.iterrows():
        pck_txt = _fmt_num(r["PCK@5%"], 2)
        d_obs_txt = _fmt_signed(r["DeltaObs"], 2)
        pred_res_txt = _fmt_signed(r["PredResidual"], 3)
        pred_cal_txt = _fmt_signed(r["PredCalibrated"], 2)

        if has_appearance:
            rank_txt = _latex_escape(str(r["rank_cal"]))
            cells = [
                _latex_escape(str(r["Variant"])),
                _latex_escape(str(r["Motion intervention"])),
                _maybe_bold(pck_txt, bool(best_masks["PCK@5%"][i])),
                _maybe_bold(d_obs_txt, bool(best_masks["DeltaObs"][i])),
                _maybe_bold(pred_res_txt, bool(best_masks["PredResidual"][i])),
                _maybe_bold(pred_cal_txt, bool(best_masks["PredCalibrated"][i])),
            ]
            if include_rank:
                cells.append(_maybe_bold(rank_txt, bool(best_masks["rank_cal"][i])))
            cells.extend(
                [
                    _maybe_bold(_fmt_num(r["F_cov_EtoT_eps1p5"], 3), bool(best_masks["F_cov_EtoT_eps1p5"][i])),
                    _maybe_bold(_fmt_num(r["F_cov_TtoE_eps1p0"], 3), bool(best_masks["F_cov_TtoE_eps1p0"][i])),
                    _maybe_bold(_fmt_num(r["A_align_EtoT"], 3), bool(best_masks["A_align_EtoT"][i])),
                    _maybe_bold(_fmt_num(r["A_align_TtoE"], 3), bool(best_masks["A_align_TtoE"][i])),
                    _maybe_bold(_fmt_num(r["A_delta_norm"], 3), bool(best_masks["A_delta_norm"][i])),
                ]
            )
        else:
            cells = [
                _latex_escape(str(r["Variant"])),
                _latex_escape(str(r["Motion intervention"])),
                _maybe_bold(pck_txt, bool(best_masks["PCK@5%"][i])),
                _maybe_bold(d_obs_txt, bool(best_masks["DeltaObs"][i])),
                _maybe_bold(pred_res_txt, bool(best_masks["PredResidual"][i])),
                _maybe_bold(pred_cal_txt, bool(best_masks["PredCalibrated"][i])),
                _maybe_bold(_fmt_num(r["F_auc_EtoT"], 3), bool(best_masks["F_auc_EtoT"][i])),
                _maybe_bold(_fmt_num(r["F_auc_TtoE"], 3), bool(best_masks["F_auc_TtoE"][i])),
                _maybe_bold(_fmt_num(r["Flow_MMD"], 4), bool(best_masks["Flow_MMD"][i])),
            ]
        all_row_lines.append(" & ".join(cells) + r" \\")

    base_row = all_row_lines[:1]
    intervention_rows = all_row_lines[1:]

    if has_appearance:
        tab_spec = "l l rr rr rrrrr"
        pred_multicolumn = "\\multicolumn{3}{c}{\\textbf{Predicted}} &" if include_rank else "\\multicolumn{2}{c}{\\textbf{Predicted}} &"
        pred_header = "$\\Delta\\hat r$ & $\\Delta\\hat y_{\\mathrm{cal}}$ &"
        if include_rank:
            tab_spec = "l l rr rrr rrrrr"
            pred_header = "$\\Delta\\hat r$ & $\\Delta\\hat y_{\\mathrm{cal}}$ & rank &"
        cmidrule = "\\cmidrule(lr){3-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-11}\\cmidrule(l){12-12}"
        if not include_rank:
            cmidrule = "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-10}\\cmidrule(l){11-11}"
        lines = [
            "\\begin{table}[t]",
            "\\centering",
            "\\small",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\setlength{\\tabcolsep}{3.6pt}",
            "\\renewcommand{\\arraystretch}{1.03}",
            "\\resizebox{\\linewidth}{!}{%",
            "\\begin{tabular}{" + tab_spec + "}",
            "\\toprule",
            "\\textbf{SDF-Fractal3D} & \\textbf{Motion} &",
            "\\multicolumn{2}{c}{\\textbf{Observed}} &",
            pred_multicolumn,
            "\\multicolumn{4}{c}{\\textbf{Predictors used (directed)}} &",
            "\\textbf{$\\|\\Delta A\\|$} \\\\",
            cmidrule,
            "\\textbf{variant} & \\textbf{intervention} &",
            "PCK@5\\% & $\\Delta$ &",
            pred_header,
            "$F_{E\\to T}^{\\mathrm{cov}@1.5px}\\uparrow$ &",
            "$F_{T\\to E}^{\\mathrm{cov}@1.0px}\\uparrow$ &",
            "$A_{E\\to T}^{\\mathrm{align}}\\uparrow$ &",
            "$A_{T\\to E}^{\\mathrm{align}}\\uparrow$ &",
            "(control) \\\\",
            "\\midrule",
            *base_row,
            "\\midrule",
            *intervention_rows,
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\renewcommand{\\arraystretch}{1.0}",
            "\\end{table}",
            "",
        ]
        return "\n".join(lines)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3.6pt}",
        "\\renewcommand{\\arraystretch}{1.03}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{l l rr rr rrr}",
        "\\toprule",
        "\\textbf{SDF-Fractal3D} & \\textbf{Motion} &",
        "\\multicolumn{2}{c}{\\textbf{Observed}} &",
        "\\multicolumn{2}{c}{\\textbf{Predicted}} &",
        "\\multicolumn{2}{c}{\\textbf{Predictors}} &",
        "\\textbf{Flow MMD} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(l){9-9}",
        "\\textbf{variant} & \\textbf{intervention} &",
        "PCK@5\\% & $\\Delta$ &",
        "$\\Delta\\hat r$ & $\\Delta\\hat y_{\\mathrm{cal}}$ &",
        "$F_{E\\to T}^{\\mathrm{AUC}}\\uparrow$ &",
        "$F_{T\\to E}^{\\mathrm{AUC}}\\uparrow$ &",
        "(control) \\\\",
        "\\midrule",
        *base_row,
        "\\midrule",
        *intervention_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "}",
        "\\renewcommand{\\arraystretch}{1.0}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build utility-guided design table from one fitted run.")
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help="Run directory with auc_with_features and prediction rows.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Target evaluation benchmark E.")
    parser.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY, help="Target model family m.")
    parser.add_argument(
        "--prediction-source",
        default=DEFAULT_PRED_SOURCE,
        choices=["lobo", "loto", "jointood"],
        help="Prediction rows file source.",
    )
    parser.add_argument("--obs-col", default=DEFAULT_OBS_COL, help="Observed transfer metric column (e.g., peak_pck).")
    parser.add_argument("--base", default="synthetic", help="Baseline train dataset.")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS, help="Comma-separated train datasets to include (must include base).")
    parser.add_argument(
        "--include-2d-warp",
        action="store_true",
        help="If set, include synthetic_2d_warp when not explicitly present in --variants.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        help="Optional pretrained filter: true/false (empty = aggregate).",
    )
    parser.add_argument(
        "--freeze",
        default="",
        help="Optional freeze filter: true/false (empty = aggregate).",
    )
    parser.add_argument(
        "--table-label",
        default="tab:utility_guided_design_2flow",
        help="LaTeX label.",
    )
    parser.add_argument(
        "--table-caption",
        default="",
        help="Optional custom caption. If empty, auto-generated caption is used.",
    )
    parser.add_argument(
        "--prediction-scale",
        default="calibrated_to_obs",
        choices=["raw_residual", "calibrated_to_obs"],
        help=(
            "How to report predicted values in the table. "
            "raw_residual = use model residual output directly; "
            "calibrated_to_obs = affine-map predictions into observed metric units."
        ),
    )
    parser.add_argument(
        "--compact-variant-names",
        action="store_true",
        default=True,
        help="Use compact row names (SDF-Fractal3D base + short variation labels).",
    )
    parser.add_argument(
        "--include-rank",
        action="store_true",
        help="Include the rank column in the LaTeX table (off by default).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_path = run_dir / "auc_with_features.csv"
    pred_path = run_dir / f"prediction_{args.prediction_source}_rows.csv"
    meta_path = run_dir / "run_metadata.json"

    auc_df = _read_csv(auc_path)
    pred_df = _read_csv(pred_path)
    meta = _read_metadata(meta_path)

    predictors = [str(x) for x in meta.get("predictors", [])]
    flow_only_predictors = set(predictors) == {"flow_train_to_eval_auc", "flow_eval_to_train_auc"}
    appearance_predictors = set(predictors) == {
        "flow_train_to_eval_eps1px",
        "flow_eval_to_train_eps1p5px",
        "dino_eval_to_train_mean_dist",
        "dino_train_to_eval_mean_dist",
    }
    if not (flow_only_predictors or appearance_predictors):
        raise SystemExit(
            "Unsupported predictor recipe in run metadata. "
            f"Got {len(predictors)} predictors: {predictors}"
        )
    for p in predictors:
        if p not in auc_df.columns:
            raise SystemExit(f"Predictor not found in auc_with_features.csv: {p}")
    if flow_only_predictors and "flow_mmd" not in auc_df.columns:
        raise SystemExit("flow_mmd column not found in auc_with_features.csv.")

    if args.obs_col not in auc_df.columns:
        raise SystemExit(f"Observed column not found: {args.obs_col}")
    if "prediction" not in pred_df.columns:
        raise SystemExit(f"Prediction column not found in {pred_path}")

    variants = _split_csv_arg(args.variants)
    if args.include_2d_warp and "synthetic_2d_warp" not in variants:
        variants.append("synthetic_2d_warp")
    if args.base not in variants:
        variants = [args.base] + variants
    variants = list(dict.fromkeys(variants))

    # Filters for observed rows (full context, then selected variants).
    obs_ctx = auc_df.copy()
    obs_ctx = obs_ctx[
        (obs_ctx["benchmark"].astype(str) == str(args.benchmark))
        & (obs_ctx["model_family"].astype(str) == str(args.model_family))
    ].copy()
    if args.pretrained.strip():
        want = args.pretrained.strip().lower() in {"1", "true", "t", "yes", "y"}
        obs_ctx = obs_ctx[
            pd.to_numeric(obs_ctx["pretrained"], errors="coerce").astype("float64") == float(want)
        ]
    if args.freeze.strip():
        want = args.freeze.strip().lower() in {"1", "true", "t", "yes", "y"}
        obs_ctx = obs_ctx[
            pd.to_numeric(obs_ctx["freeze"], errors="coerce").astype("float64") == float(want)
        ]
    obs = obs_ctx[obs_ctx["train_dataset"].astype(str).isin(variants)].copy()
    if obs.empty:
        raise SystemExit("No observed rows after applying filters.")

    obs_value_cols = [args.obs_col] + predictors
    if flow_only_predictors:
        obs_value_cols.append("flow_mmd")
    obs_agg = (
        obs.groupby("train_dataset", dropna=False)[obs_value_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    # Filters for prediction rows (full context, then selected variants).
    pred_ctx = pred_df.copy()
    pred_ctx = pred_ctx[
        (pred_ctx["benchmark"].astype(str) == str(args.benchmark))
        & (pred_ctx["model_family"].astype(str) == str(args.model_family))
    ].copy()
    if args.pretrained.strip():
        want = args.pretrained.strip().lower() in {"1", "true", "t", "yes", "y"}
        pred_ctx = pred_ctx[
            pd.to_numeric(pred_ctx["pretrained"], errors="coerce").astype("float64") == float(want)
        ]
    if args.freeze.strip():
        want = args.freeze.strip().lower() in {"1", "true", "t", "yes", "y"}
        pred_ctx = pred_ctx[
            pd.to_numeric(pred_ctx["freeze"], errors="coerce").astype("float64") == float(want)
        ]
    pred = pred_ctx[pred_ctx["train_dataset"].astype(str).isin(variants)].copy()
    if pred.empty:
        raise SystemExit("No prediction rows after applying filters.")

    pred_agg = (
        pred.groupby("train_dataset", dropna=False)[["prediction"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"prediction": "prediction_raw"})
    )

    # Build calibration map from all train datasets in this context (same filters),
    # so predicted deltas can be shown in observed metric units.
    obs_cal = (
        obs_ctx.groupby("train_dataset", dropna=False)[[args.obs_col]]
        .mean(numeric_only=True)
        .reset_index()
    )
    pred_cal = (
        pred_ctx.groupby("train_dataset", dropna=False)[["prediction"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"prediction": "prediction_raw"})
    )
    calib_df = obs_cal.merge(pred_cal, on="train_dataset", how="inner")
    if calib_df.empty:
        raise SystemExit("No rows for prediction-to-observation calibration.")
    x = pd.to_numeric(calib_df["prediction_raw"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(calib_df[args.obs_col], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        slope, intercept = 0.0, float(np.nanmean(y)) if y.size > 0 else 0.0
    else:
        x_std = float(np.nanstd(x))
        if not np.isfinite(x_std) or x_std <= 1e-12:
            slope, intercept = 0.0, float(np.nanmean(y))
        else:
            slope, intercept = np.polyfit(x, y, deg=1)
            slope = float(slope)
            intercept = float(intercept)

    merged = obs_agg.merge(pred_agg, on="train_dataset", how="inner")
    if merged.empty:
        raise SystemExit("No joined rows after merging observed and prediction tables.")

    if args.base not in set(merged["train_dataset"].astype(str)):
        raise SystemExit(f"Baseline dataset '{args.base}' missing after filtering.")

    base_row = merged[merged["train_dataset"].astype(str) == args.base].iloc[0]
    base_obs = _as_float(base_row[args.obs_col])
    base_pred = _as_float(base_row["prediction_raw"])
    merged["DeltaObs"] = pd.to_numeric(merged[args.obs_col], errors="coerce") - base_obs
    merged["prediction_calibrated"] = (
        slope * pd.to_numeric(merged["prediction_raw"], errors="coerce") + intercept
    )
    base_pred_cal = _as_float(
        merged.loc[merged["train_dataset"].astype(str) == args.base, "prediction_calibrated"].iloc[0]
    )
    merged["PredResidual"] = pd.to_numeric(merged["prediction_raw"], errors="coerce") - base_pred
    merged["PredCalibrated"] = (
        pd.to_numeric(merged["prediction_calibrated"], errors="coerce") - base_pred_cal
    )

    # Predicted rank among non-base rows (descending calibrated delta).
    non_base = merged[merged["train_dataset"].astype(str) != args.base].copy()
    non_base = non_base.sort_values("PredCalibrated", ascending=False).reset_index(drop=True)
    rank_map = {str(r["train_dataset"]): i + 1 for i, (_, r) in enumerate(non_base.iterrows())}
    merged["rank_cal"] = merged["train_dataset"].astype(str).map(rank_map).astype("Int64").astype(str)
    merged.loc[merged["train_dataset"].astype(str) == args.base, "rank_cal"] = "--"

    # Final columns.
    variant_col = merged["train_dataset"].astype(str)
    if args.compact_variant_names:
        variant_col = variant_col.map(DEFAULT_VARIANT_DISPLAY).fillna(variant_col)
    out_cols: Dict[str, pd.Series] = {
        "train_dataset_key": merged["train_dataset"].astype(str),
        "Variant": variant_col,
        "Motion intervention": merged["train_dataset"].astype(str).map(DEFAULT_INTERVENTION_LABELS).fillna("intervention"),
        "PCK@5%": pd.to_numeric(merged[args.obs_col], errors="coerce"),
        "DeltaObs": pd.to_numeric(merged["DeltaObs"], errors="coerce"),
        "PredResidual": pd.to_numeric(merged["PredResidual"], errors="coerce"),
        "PredCalibrated": pd.to_numeric(merged["PredCalibrated"], errors="coerce"),
        "rank_cal": merged["rank_cal"].astype(str),
    }
    if flow_only_predictors:
        out_cols["F_auc_EtoT"] = pd.to_numeric(merged["flow_eval_to_train_auc"], errors="coerce")
        out_cols["F_auc_TtoE"] = pd.to_numeric(merged["flow_train_to_eval_auc"], errors="coerce")
        out_cols["Flow_MMD"] = pd.to_numeric(merged["flow_mmd"], errors="coerce")
    else:
        base_a_bt_dist = _as_float(base_row["dino_eval_to_train_mean_dist"])
        base_a_tb_dist = _as_float(base_row["dino_train_to_eval_mean_dist"])
        # Report appearance as alignment score so all four predictors are "higher is better".
        # Monotone map from distance d to alignment a in (0, 1]: a = 1 / (1 + d).
        merged["A_align_BtoT"] = 1.0 / (
            1.0 + pd.to_numeric(merged["dino_eval_to_train_mean_dist"], errors="coerce")
        )
        merged["A_align_TtoB"] = 1.0 / (
            1.0 + pd.to_numeric(merged["dino_train_to_eval_mean_dist"], errors="coerce")
        )
        base_a_bt_align = 1.0 / (1.0 + base_a_bt_dist)
        base_a_tb_align = 1.0 / (1.0 + base_a_tb_dist)
        merged["A_delta_norm"] = np.sqrt(
            (pd.to_numeric(merged["A_align_BtoT"], errors="coerce") - base_a_bt_align) ** 2
            + (pd.to_numeric(merged["A_align_TtoB"], errors="coerce") - base_a_tb_align) ** 2
        )
        out_cols["F_cov_EtoT_eps1p5"] = pd.to_numeric(merged["flow_eval_to_train_eps1p5px"], errors="coerce")
        out_cols["F_cov_TtoE_eps1p0"] = pd.to_numeric(merged["flow_train_to_eval_eps1px"], errors="coerce")
        out_cols["A_align_EtoT"] = pd.to_numeric(merged["A_align_BtoT"], errors="coerce")
        out_cols["A_align_TtoE"] = pd.to_numeric(merged["A_align_TtoB"], errors="coerce")
        out_cols["A_delta_norm"] = pd.to_numeric(merged["A_delta_norm"], errors="coerce")
    out = pd.DataFrame(out_cols)

    # Preserve user variant order.
    order = {v: i for i, v in enumerate(variants)}
    out["__ord"] = out["train_dataset_key"].map(lambda x: order.get(str(x), 10_000))
    out = out.sort_values("__ord").drop(columns=["__ord", "train_dataset_key"]).reset_index(drop=True)

    caption = args.table_caption.strip()
    if not caption:
        b_tex = _to_texsc_token(args.benchmark)
        m_tex = _to_texsc_token(args.model_family)
        if flow_only_predictors:
            caption = (
                r"\textbf{Utility-guided motion tuning in SDF-Fractal3D} "
                f"for context $c=(E,m)=(\\textsc{{{b_tex}}},\\textsc{{{m_tex}}})$ "
                r"using a fixed 2-predictor flow estimator (F$\times$2). "
                r"We vary only the motion sampler while keeping appearance generation fixed. "
                r"Columns report observed transfer (Peak PCK@5\% and $\Delta$ vs.\ base), "
                r"predicted improvement (residual units and calibrated PCK points), "
                r"the two directed flow predictors (AUC; higher is better), "
                r"and Flow MMD as a control (lower is better)."
            )
        else:
            caption = (
                r"\textbf{Utility-guided motion tuning in SDF-Fractal3D} "
                f"for context $c=(E,m)=(\\textsc{{{b_tex}}},\\textsc{{{m_tex}}})$ "
                r"using a fixed 4-predictor estimator (F$\times$2 + A$\times$2). "
                r"We vary only the motion sampler while keeping appearance generation fixed. "
                r"Columns report observed transfer (Peak PCK@5\% and $\Delta$ vs.\ base), "
                r"predicted improvement (residual units and calibrated PCK points; "
                r"\emph{ranked by} $\Delta\hat y_{\mathrm{cal}}$), and the four directed predictors. "
                r"Flow predictors are BFV coverage fractions within a fixed pixel radius (higher is better) "
                r"and appearance predictors are DINO alignment scores (higher is better); "
                r"$\|\Delta A\|$ is appearance drift relative to the base variant."
            )

    tex = _build_latex_table(out, caption=caption, label=args.table_label, include_rank=args.include_rank)

    csv_out = out_dir / "utility_guided_design_table.csv"
    tex_out = out_dir / "utility_guided_design_table.tex"
    meta_out = out_dir / "utility_guided_design_table_metadata.csv"

    out.to_csv(csv_out, index=False)
    tex_out.write_text(tex)

    meta_rows = [
        {"key": "run_dir", "value": str(run_dir)},
        {"key": "benchmark", "value": args.benchmark},
        {"key": "model_family", "value": args.model_family},
        {"key": "prediction_source", "value": args.prediction_source},
        {"key": "obs_col", "value": args.obs_col},
        {"key": "base", "value": args.base},
        {"key": "variants", "value": ",".join(variants)},
        {"key": "predictors", "value": ",".join(predictors)},
        {"key": "appearance_reporting_transform", "value": "alignment = 1/(1+dino_mean_dist)" if appearance_predictors else "(not used)"},
        {"key": "flow_mmd_control", "value": bool(flow_only_predictors)},
        {"key": "prediction_scale", "value": args.prediction_scale},
        {"key": "compact_variant_names", "value": bool(args.compact_variant_names)},
        {"key": "include_rank", "value": bool(args.include_rank)},
        {"key": "calibration_slope", "value": slope},
        {"key": "calibration_intercept", "value": intercept},
        {"key": "calibration_n", "value": int(x.size)},
        {"key": "pretrained_filter", "value": args.pretrained or "(none)"},
        {"key": "freeze_filter", "value": args.freeze or "(none)"},
    ]
    pd.DataFrame(meta_rows).to_csv(meta_out, index=False)

    print(f"Wrote: {csv_out}")
    print(f"Wrote: {tex_out}")
    print(f"Wrote: {meta_out}")


if __name__ == "__main__":
    main()
