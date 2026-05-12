#!/usr/bin/env python3
"""
Compile experiment results into results.md — readable tables for the paper.

Reads:
  results/summary_table.csv           (primary source)
  results/{split}/{model}/{fg}/metrics.csv  (per-context detail, if needed)

Writes:
  results/results.md

Usage:
    python scripts/transfer_analysis_v3/compile_results.py \
        [--results-dir scripts/transfer_analysis_v3/results] \
        [--output scripts/transfer_analysis_v3/results/results.md]
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRIC_DISPLAY = {
    "spearman":       ("Spearman ↑",   "{:.3f}"),
    "rank_mae":       ("Rank MAE ↓",   "{:.2f}"),
    "norm_rank_mae":  ("Norm Rank MAE ↓", "{:.3f}"),
    "kendall":        ("Kendall ↑",    "{:.3f}"),
    "ndcg_3":         ("NDCG@3 ↑",    "{:.3f}"),
    "ndcg_5":         ("NDCG@5 ↑",    "{:.3f}"),
}

SPLIT_DISPLAY = {
    "loto":          "LOTO",
    "loto_grouped":  "LOTO (grouped)",
    "lobo":          "LOBO",
    "loco":          "LOCO",
    "lomo":          "LOMO",
}

MODEL_DISPLAY = {
    "ridge":          "Ridge",
    "bradley_terry":  "Bradley-Terry",
    "plackett_luce":  "Plackett-Luce",
    "kernel_ridge":   "Kernel Ridge",
    "random":         "Random",
    "global_prior":   "Global Prior",
}

FEATURE_DISPLAY = {
    "motion":            "Motion (NN + ε-coverage)",
    "motion_km":         "Motion k-means weighted",
    "appearance":        "Appearance (DINO)",
    "density":           "Density (log N)",
    "symmetric_mmd":     "Symmetric MMD",
    "symmetric_ot":      "Symmetric FID + SW2",
    "symmetric_all":     "Symmetric (all)",
    "motion_appearance": "Motion + Appearance",
    "all":               "All features",
}

MODEL_ORDER   = ["random", "global_prior", "ridge", "bradley_terry", "plackett_luce", "kernel_ridge"]
FEATURE_ORDER = ["density", "symmetric_mmd", "symmetric_ot", "symmetric_all",
                 "motion", "motion_km", "appearance", "motion_appearance", "all"]
SPLIT_ORDER   = ["loto", "loto_grouped", "lobo", "loco", "lomo"]


def fmt(val, fmt_str="{:.3f}") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return fmt_str.format(val)


def bold_best(series: pd.Series, higher_is_better: bool = True) -> list[str]:
    """Return list of formatted strings with **bold** for best value."""
    vals = series.values.astype(float)
    valid = ~np.isnan(vals)
    if not valid.any():
        return ["—"] * len(vals)
    best_idx = np.nanargmax(vals) if higher_is_better else np.nanargmin(vals)
    out = []
    for i, v in enumerate(vals):
        s = fmt(v)
        out.append(f"**{s}**" if (i == best_idx and valid[i]) else s)
    return out


def make_markdown_table(df: pd.DataFrame, row_label: str = "") -> str:
    """Convert a dataframe to a markdown table string."""
    cols = df.columns.tolist()
    header = f"| {row_label} | " + " | ".join(str(c) for c in cols) + " |"
    sep    = f"|{'-' * (len(row_label) + 2)}|" + "|".join("-" * (len(str(c)) + 2) for c in cols) + "|"
    lines  = [header, sep]
    for idx, row in df.iterrows():
        line = f"| {idx} | " + " | ".join(str(row[c]) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def table_model_comparison(df: pd.DataFrame, splits: list[str], metric: str = "spearman") -> str:
    """Rows = models, columns = splits. Uses best feature group per model per split."""
    col_label, fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))
    rows = {}
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        row = {}
        for split in splits:
            s = sub[sub["split"] == split]
            if s.empty:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            col = f"{metric}_mean"
            if col not in s.columns:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            best_val = s[col].max()
            best_fg  = s.loc[s[col].idxmax(), "feature_group"] if not s[col].isna().all() else "?"
            row[SPLIT_DISPLAY.get(split, split)] = f"{fmt(best_val, fmt_str)} *({FEATURE_DISPLAY.get(best_fg, best_fg)})*"
        rows[MODEL_DISPLAY.get(model, model)] = row

    if not rows:
        return "_No data_\n"
    result_df = pd.DataFrame(rows).T
    return make_markdown_table(result_df, "Model") + "\n"


def table_feature_ablation(df: pd.DataFrame, splits: list[str],
                            model: str = "ridge", metric: str = "spearman") -> str:
    """Rows = feature groups, columns = splits. Fixed model."""
    col_label, fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))
    col = f"{metric}_mean"
    ci_lo = f"{metric}_ci_lo"
    ci_hi = f"{metric}_ci_hi"
    sub = df[df["model"] == model]
    rows = {}
    for fg in FEATURE_ORDER:
        fsub = sub[sub["feature_group"] == fg]
        if fsub.empty:
            continue
        row = {}
        for split in splits:
            ssub = fsub[fsub["split"] == split]
            if ssub.empty or col not in ssub.columns:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            v = ssub[col].values[0]
            lo = ssub[ci_lo].values[0] if ci_lo in ssub.columns else float("nan")
            hi = ssub[ci_hi].values[0] if ci_hi in ssub.columns else float("nan")
            if not np.isnan(lo) and not np.isnan(hi):
                row[SPLIT_DISPLAY.get(split, split)] = f"{fmt(v, fmt_str)} [{fmt(lo, fmt_str)}, {fmt(hi, fmt_str)}]"
            else:
                row[SPLIT_DISPLAY.get(split, split)] = fmt(v, fmt_str)
        rows[FEATURE_DISPLAY.get(fg, fg)] = row

    if not rows:
        return "_No data_\n"
    result_df = pd.DataFrame(rows).T
    return make_markdown_table(result_df, "Feature group") + "\n"


def table_full_metrics(df: pd.DataFrame, split: str, model: str, fg: str) -> str:
    """All metrics for a given (split, model, feature_group)."""
    sub = df[(df["split"] == split) & (df["model"] == model) & (df["feature_group"] == fg)]
    if sub.empty:
        return "_No data_\n"
    row = sub.iloc[0]
    lines = []
    for metric, (label, fmt_str) in METRIC_DISPLAY.items():
        col = f"{metric}_mean"
        ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
        if col not in row.index:
            continue
        v = row[col]
        lo = row.get(ci_lo, float("nan"))
        hi = row.get(ci_hi, float("nan"))
        if not np.isnan(float(lo)) and not np.isnan(float(hi)):
            lines.append(f"| {label} | {fmt(v, fmt_str)} | [{fmt(lo, fmt_str)}, {fmt(hi, fmt_str)}] |")
        else:
            lines.append(f"| {label} | {fmt(v, fmt_str)} | — |")
    if not lines:
        return "_No metrics_\n"
    return "| Metric | Mean | 95% CI |\n|--------|------|--------|\n" + "\n".join(lines) + "\n"


def table_spearman_heatmap(df: pd.DataFrame, split: str, metric: str = "spearman") -> str:
    """Rows = models, columns = feature groups. Marks best per row."""
    col = f"{metric}_mean"
    sub = df[df["split"] == split]
    if sub.empty or col not in sub.columns:
        return "_No data_\n"

    model_rows  = [m for m in MODEL_ORDER   if m in sub["model"].values]
    fg_cols     = [f for f in FEATURE_ORDER if f in sub["feature_group"].values]
    if not model_rows or not fg_cols:
        return "_No data_\n"

    col_headers = [FEATURE_DISPLAY.get(f, f) for f in fg_cols]
    header = "| Model | " + " | ".join(col_headers) + " |"
    sep    = "|-------|" + "|".join("-" * (len(h) + 2) for h in col_headers) + "|"
    lines  = [header, sep]

    for model in model_rows:
        msub = sub[sub["model"] == model]
        vals = []
        for fg in fg_cols:
            fgsub = msub[msub["feature_group"] == fg]
            if fgsub.empty or col not in fgsub.columns:
                vals.append(float("nan"))
            else:
                vals.append(float(fgsub[col].values[0]))
        # Bold best (ignore baselines for best-marking)
        best_idx = int(np.nanargmax(vals)) if not all(np.isnan(vals)) else -1
        cells = []
        for j, v in enumerate(vals):
            s = fmt(v)
            cells.append(f"**{s}**" if j == best_idx and not np.isnan(v) else s)
        lines.append(f"| {MODEL_DISPLAY.get(model, model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_summary(results_dir: Path) -> pd.DataFrame | None:
    p = results_dir / "summary_table.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_metrics_from_dirs(results_dir: Path) -> pd.DataFrame:
    """Fallback: scan individual metrics.csv files if summary_table.csv missing."""
    rows = []
    for csv_path in results_dir.glob("*/*/*/metrics.csv"):
        parts = csv_path.parts
        # results_dir / split / model / feature_group / metrics.csv
        fg    = parts[-2]
        model = parts[-3]
        split = parts[-4]
        df = pd.read_csv(csv_path)
        agg = {
            "split": split, "model": model, "feature_group": fg,
            "n_contexts": len(df),
        }
        for metric in METRIC_DISPLAY:
            if metric in df.columns:
                agg[f"{metric}_mean"]   = df[metric].mean()
                agg[f"{metric}_median"] = df[metric].median()
        rows.append(agg)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
        default="scripts/transfer_analysis_v3/results")
    parser.add_argument("--output",
        default="scripts/transfer_analysis_v3/results/results.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    df = load_summary(results_dir)
    if df is None or df.empty:
        print("summary_table.csv not found — scanning individual metrics.csv files...")
        df = load_metrics_from_dirs(results_dir)
    if df is None or df.empty:
        print("No results found. Run run_experiments.py first.")
        return

    available_splits  = [s for s in SPLIT_ORDER  if s in df["split"].values]
    available_models  = [m for m in MODEL_ORDER   if m in df["model"].values]
    available_fgroups = [f for f in FEATURE_ORDER if f in df["feature_group"].values]

    print(f"  {len(df)} rows | splits: {available_splits} | models: {available_models}")

    # -----------------------------------------------------------------------
    # Build results.md
    # -----------------------------------------------------------------------
    sections = []

    sections.append(textwrap.dedent(f"""\
        # Transfer Estimator Results

        Generated from `{results_dir}`

        **Splits evaluated:** {', '.join(SPLIT_DISPLAY.get(s, s) for s in available_splits)}
        **Models:** {', '.join(MODEL_DISPLAY.get(m, m) for m in available_models)}
        **Feature groups:** {', '.join(FEATURE_DISPLAY.get(f, f) for f in available_fgroups)}

        Metrics: Spearman ↑, Rank MAE ↓, Norm Rank MAE ↓, Kendall ↑, NDCG@3 ↑, NDCG@5 ↑.
        95% CI bootstrapped over held-out entities (train datasets for LOTO, benchmarks for LOBO, etc.).

    """))

    # -----------------------------------------------------------------------
    # Section 1: Spearman heatmap per split
    # -----------------------------------------------------------------------
    sections.append("## 1. Spearman by Model × Feature Group\n")
    sections.append(
        "*Best value per model shown in **bold**. Each cell = mean Spearman across held-out contexts.*\n\n"
    )
    for split in available_splits:
        sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
        sections.append(table_spearman_heatmap(df, split, "spearman"))
        sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 2: Model comparison (best feature group per model)
    # -----------------------------------------------------------------------
    sections.append("## 2. Model Comparison (best feature group per cell)\n\n")
    for metric in ["spearman", "rank_mae"]:
        label = METRIC_DISPLAY[metric][0]
        sections.append(f"### {label}\n\n")
        sections.append(table_model_comparison(df, available_splits, metric))
        sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 3: Feature ablation (Ridge, full metrics)
    # -----------------------------------------------------------------------
    best_model = "ridge" if "ridge" in available_models else (available_models[0] if available_models else None)
    if best_model:
        sections.append(f"## 3. Feature Ablation — {MODEL_DISPLAY.get(best_model, best_model)}\n\n")
        sections.append(
            "*Spearman mean [95% CI bootstrapped over held-out entities]. "
            "Rows sorted by feature group type.*\n\n"
        )
        for metric in ["spearman", "rank_mae"]:
            label = METRIC_DISPLAY[metric][0]
            sections.append(f"### {label}\n\n")
            sections.append(table_feature_ablation(df, available_splits, best_model, metric))
            sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 4: Full metrics for best config per split
    # -----------------------------------------------------------------------
    sections.append("## 4. Full Metrics — Best Configuration per Split\n\n")
    best_nonbaseline = [m for m in available_models
                        if m not in {"random", "global_prior"}]
    for split in available_splits:
        sub = df[(df["split"] == split) & df["model"].isin(best_nonbaseline)]
        if sub.empty:
            continue
        col = "spearman_mean"
        if col not in sub.columns or sub[col].isna().all():
            continue
        best_row = sub.loc[sub[col].idxmax()]
        model = best_row["model"]
        fg    = best_row["feature_group"]
        sections.append(
            f"### {SPLIT_DISPLAY.get(split, split)} — "
            f"{MODEL_DISPLAY.get(model, model)}, {FEATURE_DISPLAY.get(fg, fg)}\n\n"
        )
        sections.append(table_full_metrics(df, split, model, fg))
        sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 5: Ranking objective comparison
    # -----------------------------------------------------------------------
    ranking_models = [m for m in ["ridge", "bradley_terry", "plackett_luce", "kernel_ridge"]
                      if m in available_models]
    if len(ranking_models) > 1:
        sections.append("## 5. Ranking Objective Comparison (all features)\n\n")
        sub = df[df["feature_group"] == "all"]
        if not sub.empty:
            for split in available_splits:
                ssub = sub[sub["split"] == split]
                col = "spearman_mean"
                if col not in ssub.columns:
                    continue
                sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
                rows_out = {}
                for model in ranking_models:
                    msub = ssub[ssub["model"] == model]
                    if msub.empty:
                        continue
                    v    = msub[col].values[0]
                    lo   = msub["spearman_ci_lo"].values[0] if "spearman_ci_lo" in msub.columns else float("nan")
                    hi   = msub["spearman_ci_hi"].values[0] if "spearman_ci_hi" in msub.columns else float("nan")
                    rmae = msub["rank_mae_mean"].values[0]  if "rank_mae_mean" in msub.columns else float("nan")
                    rows_out[MODEL_DISPLAY.get(model, model)] = {
                        "Spearman": f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]",
                        "Rank MAE": fmt(rmae, "{:.2f}"),
                    }
                if rows_out:
                    sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Model"))
                    sections.append("\n\n")

    # -----------------------------------------------------------------------
    # Section 6: Baseline sanity check
    # -----------------------------------------------------------------------
    sections.append("## 6. Baseline Sanity Checks\n\n")
    sections.append("*Spearman for random and global-prior baselines. "
                    "Random should be ≈ 0; global prior captures generic dataset quality.*\n\n")
    baseline_models = [m for m in ["random", "global_prior"] if m in available_models]
    if baseline_models:
        col = "spearman_mean"
        sub = df[df["model"].isin(baseline_models) & (df["feature_group"] == "motion")]
        if sub.empty:
            sub = df[df["model"].isin(baseline_models)]
        rows_out = {}
        for model in baseline_models:
            msub = sub[sub["model"] == model]
            row = {}
            for split in available_splits:
                ssub = msub[msub["split"] == split]
                if ssub.empty or col not in ssub.columns:
                    row[SPLIT_DISPLAY.get(split, split)] = "—"
                else:
                    row[SPLIT_DISPLAY.get(split, split)] = fmt(ssub[col].values[0])
            rows_out[MODEL_DISPLAY.get(model, model)] = row
        if rows_out:
            sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Baseline"))
            sections.append("\n\n")

    # -----------------------------------------------------------------------
    # Section 7: Data coverage summary
    # -----------------------------------------------------------------------
    sections.append("## 7. Experiment Coverage\n\n")
    counts = df.groupby(["split", "model"]).size().unstack(fill_value=0)
    sections.append("*Number of feature-group configs completed per split × model:*\n\n")
    sections.append(counts.to_markdown() + "\n\n")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    content = "\n".join(sections)
    out_path.write_text(content)
    print(f"\n✓ Results written to {out_path}")
    print(f"  Sections: Spearman heatmaps, model comparison, feature ablation, "
          f"full metrics, ranking objective comparison, baselines, coverage")


if __name__ == "__main__":
    main()
