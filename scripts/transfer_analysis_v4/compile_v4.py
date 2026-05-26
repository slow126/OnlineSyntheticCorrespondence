"""Compile results.md from v4 outputs.

Walks `summary.csv` (long-form: target × split × family × label × head) and
`bootstrap_gap.csv` (target × split × head), writes one results.md with each
section repeated per-target so the user can compare auc_normalized vs peak_pck
side-by-side.

Run:
    python scripts/transfer_analysis_v4/compile_v4.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ["LOTO", "LOBO", "JOINT"]
FAMILIES = ["motion", "appearance", "both", "random"]
FAMILY_LABEL = {
    "motion":     "**motion (flow)**",
    "appearance": "appearance (DINO)",
    "both":       "both",
    "random":     "random (control)",
}
HEAD_LABEL = {"g": "ridge", "g_zridge": "z-ridge", "g_rank": "ranknet", "g_gbm": "gbm"}
SPLIT_BLURB = {
    "LOTO":  "leave-one-source-out — new training dataset (Park-Marcotte C2)",
    "LOBO":  "leave-one-benchmark-out — new target benchmark (Park-Marcotte C2)",
    "JOINT": "joint heldout — both endpoints unseen (Park-Marcotte C3 / Pahikkala S4)",
}


def _fmt_ci(v, lo, hi):
    if not all(np.isfinite([v, lo, hi])):
        return "—"
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _row_get(idx_df, key, col):
    try:
        return float(idx_df.loc[key, col])
    except (KeyError, TypeError):
        return float("nan")


# ---------------------------------------------------------------------------
def _table_headline(summary: pd.DataFrame, target: str, head: str) -> str:
    main = summary[(summary["label"] == "main") &
                   (summary["target"] == target) &
                   (summary["head"] == head)].set_index(["split", "family"])
    lines = ["| family | LOTO | LOBO | JOINT |", "|---|---|---|---|"]
    for fam in FAMILIES:
        row = [FAMILY_LABEL[fam]]
        for split in SPLITS:
            try:
                r = main.loc[(split, fam)]
                row.append(_fmt_ci(r["ctx_rho_g"], r["ctx_rho_g_lo"], r["ctx_rho_g_hi"]))
            except KeyError:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_gap(gap: pd.DataFrame, target: str, head: str) -> str:
    sub = gap[(gap["target"] == target) & (gap["head"] == head)]
    lines = ["| regime | gap (motion − appearance) | P(gap > 0) |",
             "|---|---|---|"]
    for split in SPLITS:
        s = sub[sub["split"] == split]
        if s.empty:
            lines.append(f"| {split} | — | — |")
            continue
        r = s.iloc[0]
        ci = _fmt_ci(r["ctx_rho_g_gap"], r["ctx_rho_g_gap_lo"], r["ctx_rho_g_gap_hi"])
        p = f"{r['ctx_rho_g_gap_p_gt_0']:.3f}"
        lines.append(f"| **{split}** | {ci} | {p} |")
    return "\n".join(lines)


def _table_calibration(summary: pd.DataFrame, target: str) -> str:
    """abs_r for each head (ridge vs z-ridge) per (split, family)."""
    main = summary[(summary["label"] == "main") & (summary["target"] == target)]
    lines = ["| family | LOTO ridge / z-ridge | LOBO ridge / z-ridge | JOINT ridge / z-ridge |",
             "|---|---|---|---|"]
    for fam in FAMILIES:
        row = [FAMILY_LABEL[fam]]
        for split in SPLITS:
            try:
                r_r = main[(main.split == split) & (main.family == fam) &
                           (main.head == "g")].iloc[0]
                r_c = main[(main.split == split) & (main.family == fam) &
                           (main.head == "g_zridge")].iloc[0]
                row.append(f"{r_r['abs_r_Lg']:+.3f}  /  {r_c['abs_r_Lg']:+.3f}")
            except (KeyError, IndexError):
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_shuffle(summary: pd.DataFrame, target: str, head: str) -> str:
    sh = summary[(summary["label"] == "shuffle") &
                 (summary["target"] == target) &
                 (summary["head"] == head)].set_index(["split", "family"])
    lines = ["| family | LOTO | LOBO | JOINT |", "|---|---|---|---|"]
    for fam in FAMILIES:
        row = [FAMILY_LABEL[fam]]
        for split in SPLITS:
            try:
                r = sh.loc[(split, fam)]
                row.append(_fmt_ci(r["ctx_rho_g"], r["ctx_rho_g_lo"], r["ctx_rho_g_hi"]))
            except KeyError:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_density(summary: pd.DataFrame, target: str) -> str:
    main = summary[(summary["label"] == "main") &
                   (summary["target"] == target) &
                   (summary["head"] == "g")].set_index(["split", "family"])
    if not any((sp, fam) in main.index
               for sp in SPLITS for fam in ("density", "motion", "motion_density")):
        return ""
    lines = ["| split | density alone | motion alone | motion + density |",
             "|---|---|---|---|"]
    for split in SPLITS:
        row = [f"**{split}**"]
        for fam in ("density", "motion", "motion_density"):
            try:
                r = main.loc[(split, fam)]
                row.append(_fmt_ci(r["ctx_rho_g"], r["ctx_rho_g_lo"], r["ctx_rho_g_hi"]))
            except KeyError:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def render_target_section(summary: pd.DataFrame, gap: pd.DataFrame,
                          target: str, fig_dir: Path) -> str:
    figs = {p.stem: p.name for p in sorted(fig_dir.glob("*.png"))}

    def fig_link(stem, alt):
        if stem in figs:
            return f"![{alt}](figures/{figs[stem]})"
        return f"_(figure {stem} missing)_"

    sec = []
    sec.append(f"\n# ========== TARGET: `{target}` ==========\n")

    # Headline
    sec.append("## Headline — within-context ranking ρ (g only)\n")
    sec.append("### Ridge\n")
    sec.append(_table_headline(summary, target, "g") + "\n")
    sec.append("### RankNet (pairwise)\n")
    sec.append(_table_headline(summary, target, "g_rank") + "\n")
    sec.append("### GBM (nonlinear ceiling check)\n")
    sec.append(_table_headline(summary, target, "g_gbm") + "\n")
    sec.append("*Reads: a more flexible model doesn't beat ridge on motion — "
               "confirms ridge is near the practical ceiling at this N. GBM may "
               "find slight extra signal in appearance but the motion ≫ "
               "appearance ordering survives.*\n")
    sec.append("### Motion − appearance gap (paired bootstrap, ridge)\n")
    sec.append(_table_gap(gap, target, "g") + "\n")
    sec.append("### Motion − appearance gap (paired bootstrap, ranknet)\n")
    sec.append(_table_gap(gap, target, "g_rank") + "\n")
    sec.append("### Motion − appearance gap (paired bootstrap, gbm)\n")
    sec.append(_table_gap(gap, target, "g_gbm") + "\n")
    sec.append(f"\n{fig_link(f'fig1_headline_bars__{target}', 'headline bars')}\n")

    # Rank-rank scatters
    sec.append("\n## Rank scatters — predicted-rank vs actual-rank, within-context\n")
    sec.append(
        "Each context ranks its training sources 1..n. The rank scatter strips "
        "magnitude noise (the source of the “vertical band” problem on raw "
        "residual scatters) so the ordering signal is visible directly.\n"
    )
    sec.append(f"\n{fig_link(f'fig6_rank_scatter__{target}__g', 'rank scatter ridge')}\n")
    sec.append("*Ridge: linear least-squares predictions converted to rank.*\n")
    sec.append(f"\n{fig_link(f'fig6_rank_scatter__{target}__g_rank', 'rank scatter ranknet')}\n")
    sec.append("*RankNet: pairwise logistic on within-context feature differences "
               "(direct rank optimization).*\n")

    # Per-context ρ histogram + Top-K hit rate (the honest performance views)
    sec.append("\n## Performance distribution — per-context ρ histogram\n")
    sec.append(
        "The headline ρ in Table 1 is the **mean** of per-context Spearman "
        "values. The histograms show the full distribution: how many contexts "
        "land in the +0.5 to +1.0 range vs the −0.5 to 0 range. Wins and losses "
        "by benchmark are visible from the per-bar colors.\n"
    )
    sec.append(f"\n{fig_link(f'fig8_per_context_rho_hist__{target}__g', 'per-context ρ histogram (ridge)')}\n")
    sec.append(f"\n{fig_link(f'fig8_per_context_rho_hist__{target}__g_rank', 'per-context ρ histogram (ranknet)')}\n")

    sec.append("\n## Top-K training-source hit rate\n")
    sec.append(
        "Practical view: for each context, what fraction of the model's top-K "
        "predicted training datasets are also in the actual top-K? Random "
        "chance baselines (`k/n`) drawn as dotted lines.\n"
    )
    sec.append(f"\n{fig_link(f'fig9_topk_hit_rate__{target}__g', 'top-K hit rate (ridge)')}\n")
    sec.append(f"\n{fig_link(f'fig9_topk_hit_rate__{target}__g_rank', 'top-K hit rate (ranknet)')}\n")

    # Hexbin density view
    sec.append("\n## Hexbin density (residual)\n")
    sec.append(
        "Density-view of fig3a. The trend of high-density cells along the "
        "diagonal *is* the ctx ρ; bulky off-diagonal density tells you the model "
        "is wrong at scale.\n"
    )
    sec.append(f"\n{fig_link(f'fig10_residual_hexbin__{target}__g', 'hexbin density (ridge)')}\n")
    sec.append(f"\n{fig_link(f'fig10_residual_hexbin__{target}__g_zridge', 'hexbin density (z-ridge)')}\n")

    # Residual scatter (legacy view)
    sec.append("\n## Residual scatters (legacy magnitude view)\n")
    sec.append(f"\n{fig_link(f'fig3a_residual_scatter_g__{target}', 'ridge residual g')}\n")
    sec.append("*Raw ridge residuals — predicted g vs actual residual. "
               "Heterogeneous within-context spread is what makes some panels look "
               "compressed.*\n")
    sec.append(f"\n{fig_link(f'fig3a_residual_scatter_g_zridge__{target}', 'z-ridge residual')}\n")
    sec.append("*Z-ridge: target z-scored per context before fitting; one global "
               "slope on standardized data; predictions un-standardized per context. "
               "Built-in handling of heterogeneous within-context variance "
               "(spair std ~ 0.2 vs synthetic std ~ 13).*\n")

    # Calibration
    sec.append("\n## Calibration: ridge vs z-ridge (abs r of L + g)\n")
    sec.append(_table_calibration(summary, target) + "\n")
    sec.append(f"\n{fig_link(f'fig7_zridge_compare__{target}', 'z-ridge compare')}\n")
    sec.append(f"\n{fig_link(f'fig2_global_scatter__{target}', 'global L+g scatter')}\n")

    # Controls
    sec.append("\n## Controls — shuffle-target leakage check (ridge)\n")
    sec.append(_table_shuffle(summary, target, "g") + "\n")
    sec.append("*All cells should ≈ 0.*\n")
    sec.append(f"\n{fig_link(f'fig4_controls__{target}', 'controls')}\n")

    # Density confound
    density_table = _table_density(summary, target)
    if density_table:
        sec.append("\n## Density confound — motion vs dataset-size proxy\n")
        sec.append(density_table + "\n")
        sec.append(f"\n{fig_link(f'fig5_density_confound__{target}', 'density confound')}\n")

    return "\n".join(sec)


def render(summary: pd.DataFrame, gap: pd.DataFrame, fig_dir: Path) -> str:
    targets = sorted(summary["target"].unique())

    out = []
    out.append("# Transfer Analysis v4 — Results\n")
    out.append(
        "Two-target run (`auc_normalized` vs `peak_pck`) with three model heads "
        "per fold:\n\n"
        "- **ridge** — RidgeCV on winsorized + standardized within-context "
        "demeaned features. Winsorization at 1st/99th percentile (using training "
        "rows only) prevents heavy-tailed DINO KL outliers from dominating "
        "appearance predictions.\n"
        "- **z-ridge** — within-context z-score ridge. The target is divided by "
        "its per-context std before fitting (computed on the held-out fold's "
        "training rows), so each context contributes equally to the loss "
        "regardless of its raw variance. Predictions un-standardize per context "
        "at test time. Handles the spair-vs-synthetic variance heterogeneity "
        "without requiring per-benchmark interactions.\n"
        "- **ranknet** — pairwise RankNet (logistic on within-context feature "
        "differences). Direct rank-loss optimization, no magnitude calibration.\n"
        "- **gbm** — HistGradientBoostingRegressor on the same preprocessed "
        "features (max_depth=4, lr=0.05, 200 iters w/ early stopping). "
        "Nonlinear *ceiling-check*: if GBM doesn't beat ridge, the data isn't "
        "leaving nonlinear structure on the table at this N.\n"
    )
    out.append("\n## Cross-target sanity check\n")
    out.append(
        "The two targets measure different things — `auc_normalized` integrates "
        "PCK over training (mixes speed-of-convergence with final quality), "
        "`peak_pck` is the best PCK reached during training (final-quality only). "
        "If the motion claim survives both, it isn't an artifact of either metric.\n"
    )

    # Cross-target quick-look table: motion ridge ctx_rho per target × split
    main = summary[(summary["label"] == "main") & (summary["head"] == "g")]
    quick = ["| target | LOTO | LOBO | JOINT |", "|---|---|---|---|"]
    for t in targets:
        for fam in ("motion", "appearance"):
            row = [f"`{t}` × {fam}"]
            for sp in SPLITS:
                r = main[(main.target == t) & (main.split == sp) &
                         (main.family == fam)]
                if r.empty:
                    row.append("—")
                else:
                    rr = r.iloc[0]
                    row.append(_fmt_ci(rr["ctx_rho_g"], rr["ctx_rho_g_lo"],
                                       rr["ctx_rho_g_hi"]))
            quick.append("| " + " | ".join(row) + " |")
    out.append("\n".join(quick) + "\n")

    for target in targets:
        out.append(render_target_section(summary, gap, target, fig_dir))

    out.append("\n---\n\n## Files\n")
    out.append("- `summary.csv` — long-form summary (target × split × family × label × head)\n")
    out.append("- `bootstrap_gap.csv` — motion − appearance gap CIs per (target, split, head)\n")
    out.append("- `predictions/<target>/rows_<split>_<family>.csv` — per-row predictions "
               "with columns `g`, `g_cal`, `g_rank`, `L`, `actual`\n")
    out.append("- `figures/*.png`\n")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="scripts/transfer_analysis_v4/results")
    args = ap.parse_args()
    root = Path(".").resolve()
    res = root / args.results
    summary = pd.read_csv(res / "summary.csv")
    gap = pd.read_csv(res / "bootstrap_gap.csv") if (res / "bootstrap_gap.csv").exists() else pd.DataFrame()
    body = render(summary, gap, res / "figures")
    (res / "results.md").write_text(body)
    print(f"results -> {res / 'results.md'}")


if __name__ == "__main__":
    main()
