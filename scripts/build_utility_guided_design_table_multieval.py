#!/usr/bin/env python3
"""
Build a multi-eval utility-guided design table for fixed motion interventions.

This variant is specialized for the 4-predictor (2F+2A) run and emits one
combined table with grouped benchmark blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_RUN_DIR = (
    "analysis_comprehensive_runs/"
    "ridge_resid_weighted_ridge_a10_no_family_no_density_zscore_zeroshot_v3/"
    "density_joint/leakage_free_combo_flow_eps_raw_single__dino_rnorm_k5"
)
DEFAULT_OUTPUT_DIR = "analysis/utility_guided_design_tables"
DEFAULT_BENCHMARKS = "kitti2015,tss,pfpascal"
DEFAULT_VARIANTS = "synthetic,synthetic_small_zoom,synthetic_large_zoom,synthetic_random_flipping"
DEFAULT_MODEL_FAMILY = "catspp"

VARIANT_NAMES = {
    "synthetic": "SDF-Fractal3D (base)",
    "synthetic_small_zoom": "+ small-zoom",
    "synthetic_large_zoom": "+ large-zoom",
    "synthetic_random_flipping": "+ random-flip",
}
INTERVENTION_NAMES = {
    "synthetic": "baseline",
    "synthetic_small_zoom": "target-zoom",
    "synthetic_large_zoom": "high-zoom",
    "synthetic_random_flipping": "flip-mismatch",
}
NUM_COLS = [
    "PCK@5%",
    "DeltaObs",
    "PredResidual",
    "PredCalibrated",
    "F_cov_EtoT_eps1p5",
    "F_cov_TtoE_eps1p0",
    "A_align_EtoT",
    "A_align_TtoE",
]


def _split_csv(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _as_float(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt_num(x: object, d: int = 3) -> str:
    v = _as_float(x)
    if not np.isfinite(v):
        return "--"
    return f"{v:.{d}f}"


def _fmt_signed(x: object, d: int = 2) -> str:
    v = _as_float(x)
    if not np.isfinite(v):
        return "--"
    return f"{v:+.{d}f}"


def _maybe_bold(txt: str, cond: bool) -> str:
    return f"\\textbf{{{txt}}}" if cond else txt


def _build_block(
    auc_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    benchmark: str,
    model_family: str,
    variants: List[str],
    obs_col: str = "peak_pck",
) -> pd.DataFrame:
    obs_ctx = auc_df[
        (auc_df["benchmark"].astype(str) == benchmark)
        & (auc_df["model_family"].astype(str) == model_family)
    ].copy()
    pred_ctx = pred_df[
        (pred_df["benchmark"].astype(str) == benchmark)
        & (pred_df["model_family"].astype(str) == model_family)
    ].copy()
    obs = obs_ctx[obs_ctx["train_dataset"].astype(str).isin(variants)].copy()
    pred = pred_ctx[pred_ctx["train_dataset"].astype(str).isin(variants)].copy()
    if obs.empty or pred.empty:
        raise SystemExit(f"No rows for benchmark={benchmark}, model_family={model_family}")

    obs_agg = (
        obs.groupby("train_dataset", dropna=False)[
            [obs_col, "flow_eval_to_train_eps1p5px", "flow_train_to_eval_eps1px", "dino_eval_to_train_mean_dist", "dino_train_to_eval_mean_dist"]
        ]
        .mean(numeric_only=True)
        .reset_index()
    )
    pred_agg = (
        pred.groupby("train_dataset", dropna=False)[["prediction"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"prediction": "prediction_raw"})
    )

    obs_cal = (
        obs_ctx.groupby("train_dataset", dropna=False)[[obs_col]]
        .mean(numeric_only=True)
        .reset_index()
    )
    pred_cal = (
        pred_ctx.groupby("train_dataset", dropna=False)[["prediction"]]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"prediction": "prediction_raw"})
    )
    calib = obs_cal.merge(pred_cal, on="train_dataset", how="inner")
    x = pd.to_numeric(calib["prediction_raw"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(calib[obs_col], errors="coerce").to_numpy(float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2 or float(np.nanstd(x)) <= 1e-12:
        slope, intercept = 0.0, float(np.nanmean(y)) if y.size else 0.0
    else:
        slope, intercept = np.polyfit(x, y, deg=1)
        slope = float(slope)
        intercept = float(intercept)

    merged = obs_agg.merge(pred_agg, on="train_dataset", how="inner")
    base = merged[merged["train_dataset"].astype(str) == "synthetic"].iloc[0]
    base_obs = _as_float(base[obs_col])
    base_pred = _as_float(base["prediction_raw"])

    merged["prediction_calibrated"] = slope * pd.to_numeric(merged["prediction_raw"], errors="coerce") + intercept
    base_pred_cal = _as_float(merged.loc[merged["train_dataset"].astype(str) == "synthetic", "prediction_calibrated"].iloc[0])
    merged["DeltaObs"] = pd.to_numeric(merged[obs_col], errors="coerce") - base_obs
    merged["PredResidual"] = pd.to_numeric(merged["prediction_raw"], errors="coerce") - base_pred
    merged["PredCalibrated"] = pd.to_numeric(merged["prediction_calibrated"], errors="coerce") - base_pred_cal

    merged["A_align_EtoT"] = 1.0 / (1.0 + pd.to_numeric(merged["dino_eval_to_train_mean_dist"], errors="coerce"))
    merged["A_align_TtoE"] = 1.0 / (1.0 + pd.to_numeric(merged["dino_train_to_eval_mean_dist"], errors="coerce"))
    base_align_e2t = _as_float(merged.loc[merged["train_dataset"].astype(str) == "synthetic", "A_align_EtoT"].iloc[0])
    base_align_t2e = _as_float(merged.loc[merged["train_dataset"].astype(str) == "synthetic", "A_align_TtoE"].iloc[0])
    merged["A_delta_norm"] = np.sqrt(
        (pd.to_numeric(merged["A_align_EtoT"], errors="coerce") - base_align_e2t) ** 2
        + (pd.to_numeric(merged["A_align_TtoE"], errors="coerce") - base_align_t2e) ** 2
    )

    out = pd.DataFrame(
        {
            "Eval": benchmark,
            "train_dataset_key": merged["train_dataset"].astype(str),
            "Variant": merged["train_dataset"].astype(str).map(VARIANT_NAMES).fillna(merged["train_dataset"].astype(str)),
            "Motion intervention": merged["train_dataset"].astype(str).map(INTERVENTION_NAMES).fillna("intervention"),
            "PCK@5%": pd.to_numeric(merged[obs_col], errors="coerce"),
            "DeltaObs": pd.to_numeric(merged["DeltaObs"], errors="coerce"),
            "PredResidual": pd.to_numeric(merged["PredResidual"], errors="coerce"),
            "PredCalibrated": pd.to_numeric(merged["PredCalibrated"], errors="coerce"),
            "F_cov_EtoT_eps1p5": pd.to_numeric(merged["flow_eval_to_train_eps1p5px"], errors="coerce"),
            "F_cov_TtoE_eps1p0": pd.to_numeric(merged["flow_train_to_eval_eps1px"], errors="coerce"),
            "A_align_EtoT": pd.to_numeric(merged["A_align_EtoT"], errors="coerce"),
            "A_align_TtoE": pd.to_numeric(merged["A_align_TtoE"], errors="coerce"),
            "A_delta_norm": pd.to_numeric(merged["A_delta_norm"], errors="coerce"),
        }
    )
    order = {v: i for i, v in enumerate(variants)}
    out["__ord"] = out["train_dataset_key"].map(lambda x: order.get(str(x), 10_000))
    out = out.sort_values("__ord").drop(columns=["__ord", "train_dataset_key"]).reset_index(drop=True)
    return out


def _build_tex(df: pd.DataFrame, label: str) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Utility-guided motion tuning in SDF-Fractal3D across multiple eval benchmarks} for $m=\textsc{cats++}$ using a fixed 4-predictor estimator (F$\times$2 + A$\times$2). Rows are grouped by eval benchmark $E\in\{\textsc{kitti2015},\textsc{tss},\textsc{pfpascal}\}$. We vary only motion sampling while keeping appearance generation fixed.}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\renewcommand{\arraystretch}{1.03}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l l rr rr rrrrr}",
        r"\toprule",
        r"\textbf{SDF-Fractal3D} & \textbf{Motion} &",
        r"\multicolumn{2}{c}{\textbf{Observed}} &",
        r"\multicolumn{2}{c}{\textbf{Predicted}} &",
        r"\multicolumn{4}{c}{\textbf{Predictors used (directed)}} &",
        r"\textbf{$\|\Delta A\|$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-10}\cmidrule(l){11-11}",
        r"\textbf{variant} & \textbf{intervention} &",
        r"PCK@5\% & $\Delta$ &",
        r"$\Delta\hat r$ & $\Delta\hat y_{\mathrm{cal}}$ &",
        r"$F_{E\to T}^{\mathrm{cov}@1.5px}\uparrow$ &",
        r"$F_{T\to E}^{\mathrm{cov}@1.0px}\uparrow$ &",
        r"$A_{E\to T}^{\mathrm{align}}\uparrow$ &",
        r"$A_{T\to E}^{\mathrm{align}}\uparrow$ &",
        r"(control) \\",
        r"\midrule",
    ]
    for b in pd.unique(df["Eval"].astype(str)):
        block = df[df["Eval"].astype(str) == b].reset_index(drop=True)
        b_tex = b.replace("_", r"\_")
        best_masks: Dict[str, np.ndarray] = {}
        for col in NUM_COLS:
            vals = pd.to_numeric(block[col], errors="coerce").to_numpy(float)
            best_masks[col] = np.isclose(vals, np.nanmax(vals), equal_nan=False, rtol=1e-12, atol=1e-12)
        vals = pd.to_numeric(block["A_delta_norm"], errors="coerce").to_numpy(float)
        best_masks["A_delta_norm"] = np.isclose(vals, np.nanmin(vals), equal_nan=False, rtol=1e-12, atol=1e-12)

        lines.append(rf"\multicolumn{{11}}{{l}}{{\textit{{$E=\textsc{{{b_tex}}}$}}}} \\")
        for i, r in block.iterrows():
            row = [
                str(r["Variant"]),
                str(r["Motion intervention"]),
                _maybe_bold(_fmt_num(r["PCK@5%"], 2), bool(best_masks["PCK@5%"][i])),
                _maybe_bold(_fmt_signed(r["DeltaObs"], 2), bool(best_masks["DeltaObs"][i])),
                _maybe_bold(_fmt_signed(r["PredResidual"], 3), bool(best_masks["PredResidual"][i])),
                _maybe_bold(_fmt_signed(r["PredCalibrated"], 2), bool(best_masks["PredCalibrated"][i])),
                _maybe_bold(_fmt_num(r["F_cov_EtoT_eps1p5"], 3), bool(best_masks["F_cov_EtoT_eps1p5"][i])),
                _maybe_bold(_fmt_num(r["F_cov_TtoE_eps1p0"], 3), bool(best_masks["F_cov_TtoE_eps1p0"][i])),
                _maybe_bold(_fmt_num(r["A_align_EtoT"], 3), bool(best_masks["A_align_EtoT"][i])),
                _maybe_bold(_fmt_num(r["A_align_TtoE"], 3), bool(best_masks["A_align_TtoE"][i])),
                _maybe_bold(_fmt_num(r["A_delta_norm"], 3), bool(best_masks["A_delta_norm"][i])),
            ]
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"}", r"\renewcommand{\arraystretch}{1.0}", r"\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS)
    ap.add_argument("--model-family", default=DEFAULT_MODEL_FAMILY)
    ap.add_argument("--prediction-source", default="lobo", choices=["lobo", "loto", "jointood"])
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--table-label", default="tab:utility_guided_design_4pred_multieval")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_df = pd.read_csv(run_dir / "auc_with_features.csv")
    pred_df = pd.read_csv(run_dir / f"prediction_{args.prediction_source}_rows.csv")

    benchmarks = _split_csv(args.benchmarks)
    variants = _split_csv(args.variants)
    blocks = [_build_block(auc_df, pred_df, b, args.model_family, variants) for b in benchmarks]
    out = pd.concat(blocks, ignore_index=True)

    csv_path = out_dir / "utility_guided_design_table_multieval.csv"
    tex_path = out_dir / "utility_guided_design_table_multieval.tex"
    meta_path = out_dir / "utility_guided_design_table_multieval_metadata.csv"
    out.to_csv(csv_path, index=False)
    tex_path.write_text(_build_tex(out, label=args.table_label))
    pd.DataFrame(
        [
            {"key": "run_dir", "value": str(run_dir)},
            {"key": "benchmarks", "value": ",".join(benchmarks)},
            {"key": "model_family", "value": args.model_family},
            {"key": "prediction_source", "value": args.prediction_source},
            {"key": "variants", "value": ",".join(variants)},
        ]
    ).to_csv(meta_path, index=False)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {tex_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
