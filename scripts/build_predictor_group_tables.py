#!/usr/bin/env python3
"""
Build claim-oriented tables by parsing predictor composition from method_summary.csv.

Outputs:
- grouped method table with derived modality counts/group labels
- best method per derived group (on primary metric)
- pure-modality best-per-k table (flow_only / appearance_only / hof_only)
- matched-k deltas between pure modalities
- support/counter summary for selected claims
- best-by-composition table (exact modality-count tuple)
- one-step incremental deltas for domain additions (+1 flow / +1 appearance / +1 hof / +1 mmd)
- anchored deltas from pure bases to shared final compositions
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


CONTROL_PREDICTORS = {
    "log_n_samples_eval",
    "log_avg_flows_eval",
    "log_n_samples_train",
    "log_avg_flows_train",
}

DEFAULT_METRICS = [
    "loto_rank_pairwise_cindex",
    "loto_rank_spearman",
    "loto_spearman",
    "loto_mae",
    "loto_rmse",
]

COMPOSITION_COLS = [
    "n_flow",
    "n_appearance",
    "n_hof",
    "n_flow_mmd",
    "n_appearance_mmd",
    "n_other_mmd",
]


def _split_predictors(text: object) -> List[str]:
    if text is None or pd.isna(text):
        return []
    return [t.strip() for t in str(text).split(",") if t and str(t).strip()]


def _is_error_metric(name: str) -> bool:
    key = str(name).lower()
    return any(x in key for x in ("mae", "rmse", "regret", "abs_rank"))


def _higher_is_better(name: str) -> bool:
    return not _is_error_metric(name)


def _parse_group_row(row: pd.Series) -> pd.Series:
    preds = _split_predictors(row.get("predictors"))
    signal = [p for p in preds if p not in CONTROL_PREDICTORS and "_x_" not in p]

    flow = sum(1 for p in signal if p.startswith("flow_") and "mmd" not in p)
    appearance = sum(1 for p in signal if p.startswith("dino_") and "mmd" not in p)
    hof = sum(1 for p in signal if p.startswith("hof_") and "mmd" not in p)

    flow_mmd = sum(1 for p in signal if p == "flow_mmd" or (p.startswith("flow_") and "mmd" in p))
    appearance_mmd = sum(
        1
        for p in signal
        if p == "dino_mmd" or p == "feature_mmd" or (p.startswith("dino_") and "mmd" in p)
    )
    other_mmd = sum(
        1
        for p in signal
        if "mmd" in p and not (p.startswith("flow_") or p.startswith("dino_") or p in {"flow_mmd", "dino_mmd", "feature_mmd"})
    )

    mmd_total = flow_mmd + appearance_mmd + other_mmd
    k_modal = flow + appearance + hof
    k_total = k_modal + mmd_total

    if k_modal == 0 and mmd_total > 0:
        if flow_mmd > 0 and appearance_mmd > 0:
            group = "mmd_both_only"
        elif flow_mmd > 0:
            group = "mmd_flow_only"
        elif appearance_mmd > 0:
            group = "mmd_appearance_only"
        else:
            group = "mmd_other_only"
    elif flow > 0 and appearance == 0 and hof == 0:
        group = "flow_only" if mmd_total == 0 else "flow_only_plus_mmd"
    elif appearance > 0 and flow == 0 and hof == 0:
        group = "appearance_only" if mmd_total == 0 else "appearance_only_plus_mmd"
    elif hof > 0 and flow == 0 and appearance == 0:
        group = "hof_only" if mmd_total == 0 else "hof_only_plus_mmd"
    elif flow > 0 and appearance > 0 and hof == 0:
        group = "flow_appearance_hybrid" if mmd_total == 0 else "flow_appearance_hybrid_plus_mmd"
    elif flow > 0 and hof > 0 and appearance == 0:
        group = "flow_hof_hybrid" if mmd_total == 0 else "flow_hof_hybrid_plus_mmd"
    elif flow == 0 and appearance > 0 and hof > 0:
        group = "appearance_hof_hybrid" if mmd_total == 0 else "appearance_hof_hybrid_plus_mmd"
    elif flow > 0 and appearance > 0 and hof > 0:
        group = "flow_appearance_hof_hybrid" if mmd_total == 0 else "flow_appearance_hof_hybrid_plus_mmd"
    else:
        group = "other"

    return pd.Series(
        {
            "derived_group": group,
            "k_modal": int(k_modal),
            "k_total": int(k_total),
            "n_flow": int(flow),
            "n_appearance": int(appearance),
            "n_hof": int(hof),
            "n_flow_mmd": int(flow_mmd),
            "n_appearance_mmd": int(appearance_mmd),
            "n_other_mmd": int(other_mmd),
            "n_mmd_total": int(mmd_total),
        }
    )


def _best_by_metric(df: pd.DataFrame, metric: str, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work[work[metric].notna()].copy()
    if work.empty:
        return work
    asc = not _higher_is_better(metric)
    work = work.sort_values(metric, ascending=asc)
    return work.groupby(list(group_cols), as_index=False, dropna=False).head(1).reset_index(drop=True)


def _oriented_delta(metric: str, a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    raw = float(a - b)
    return raw if _higher_is_better(metric) else -raw


def _build_pure_deltas(best_pure_by_k: pd.DataFrame, metrics: Sequence[str], primary_metric: str, close_gap: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for k, g in best_pure_by_k.groupby("k_modal", dropna=False):
        have = set(g["derived_group"].astype(str).tolist())
        if not {"flow_only", "appearance_only", "hof_only"}.issubset(have):
            continue
        flow = g[g["derived_group"] == "flow_only"].iloc[0]
        app = g[g["derived_group"] == "appearance_only"].iloc[0]
        hof = g[g["derived_group"] == "hof_only"].iloc[0]

        row: Dict[str, object] = {
            "k_modal": int(k),
            "flow_method": flow.get("method"),
            "appearance_method": app.get("method"),
            "hof_method": hof.get("method"),
        }
        for m in metrics:
            f = float(flow.get(m)) if pd.notna(flow.get(m)) else float("nan")
            a = float(app.get(m)) if pd.notna(app.get(m)) else float("nan")
            h = float(hof.get(m)) if pd.notna(hof.get(m)) else float("nan")
            row[f"{m}__flow"] = f
            row[f"{m}__appearance"] = a
            row[f"{m}__hof"] = h
            row[f"{m}__flow_minus_hof_oriented"] = _oriented_delta(m, f, h)
            row[f"{m}__hof_minus_appearance_oriented"] = _oriented_delta(m, h, a)
            row[f"{m}__flow_minus_appearance_oriented"] = _oriented_delta(m, f, a)

        pf = float(flow.get(primary_metric)) if pd.notna(flow.get(primary_metric)) else float("nan")
        pa = float(app.get(primary_metric)) if pd.notna(app.get(primary_metric)) else float("nan")
        ph = float(hof.get(primary_metric)) if pd.notna(hof.get(primary_metric)) else float("nan")
        if _higher_is_better(primary_metric):
            row["support_flow_gt_hof"] = bool(math.isfinite(pf) and math.isfinite(ph) and pf > ph)
            row["support_hof_gt_appearance"] = bool(math.isfinite(ph) and math.isfinite(pa) and ph > pa)
            row["support_flow_gt_appearance"] = bool(math.isfinite(pf) and math.isfinite(pa) and pf > pa)
            row["support_order_flow_hof_appearance"] = bool(
                math.isfinite(pf) and math.isfinite(ph) and math.isfinite(pa) and pf > ph > pa
            )
            gap = pf - ph if math.isfinite(pf) and math.isfinite(ph) else float("nan")
        else:
            row["support_flow_gt_hof"] = bool(math.isfinite(pf) and math.isfinite(ph) and pf < ph)
            row["support_hof_gt_appearance"] = bool(math.isfinite(ph) and math.isfinite(pa) and ph < pa)
            row["support_flow_gt_appearance"] = bool(math.isfinite(pf) and math.isfinite(pa) and pf < pa)
            row["support_order_flow_hof_appearance"] = bool(
                math.isfinite(pf) and math.isfinite(ph) and math.isfinite(pa) and pf < ph < pa
            )
            gap = ph - pf if math.isfinite(pf) and math.isfinite(ph) else float("nan")
        row["flow_hof_gap"] = gap
        row["support_hof_close_to_flow"] = bool(math.isfinite(gap) and gap <= float(close_gap))
        row["primary_metric"] = primary_metric
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_support(deltas_df: pd.DataFrame) -> pd.DataFrame:
    if deltas_df.empty:
        return pd.DataFrame()
    n = float(len(deltas_df))
    flags = [
        "support_flow_gt_hof",
        "support_hof_gt_appearance",
        "support_flow_gt_appearance",
        "support_order_flow_hof_appearance",
        "support_hof_close_to_flow",
    ]
    rows: List[Dict[str, object]] = []
    for f in flags:
        s = int(pd.to_numeric(deltas_df[f], errors="coerce").fillna(0).astype(int).sum())
        rows.append({"flag": f, "support_count": s, "counter_count": int(n - s), "support_frac": s / n})
    return pd.DataFrame(rows)


def _build_incremental_edges(best_comp: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if best_comp.empty:
        return pd.DataFrame()

    idx: Dict[tuple, pd.Series] = {}
    for _, r in best_comp.iterrows():
        key = tuple(int(r[c]) for c in COMPOSITION_COLS)
        idx[key] = r

    domains = [
        ("flow", "n_flow"),
        ("appearance", "n_appearance"),
        ("hof", "n_hof"),
        ("flow_mmd", "n_flow_mmd"),
        ("appearance_mmd", "n_appearance_mmd"),
        ("other_mmd", "n_other_mmd"),
    ]

    rows: List[Dict[str, object]] = []
    for key, base in idx.items():
        base_counts = {c: int(base[c]) for c in COMPOSITION_COLS}
        for dname, dcol in domains:
            aug_counts = dict(base_counts)
            aug_counts[dcol] += 1
            aug_key = tuple(aug_counts[c] for c in COMPOSITION_COLS)
            aug = idx.get(aug_key)
            if aug is None:
                continue

            row: Dict[str, object] = {
                "added_domain": dname,
                "base_method": base.get("method"),
                "aug_method": aug.get("method"),
            }
            for c in COMPOSITION_COLS:
                row[f"base_{c}"] = base_counts[c]
                row[f"aug_{c}"] = int(aug[c])
            for m in metrics:
                b = float(base.get(m)) if pd.notna(base.get(m)) else float("nan")
                a = float(aug.get(m)) if pd.notna(aug.get(m)) else float("nan")
                row[f"{m}__base"] = b
                row[f"{m}__aug"] = a
                row[f"{m}__raw_delta"] = a - b if math.isfinite(a) and math.isfinite(b) else float("nan")
                row[f"{m}__oriented_delta"] = _oriented_delta(m, a, b)
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_incremental_edges(edges_df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if edges_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for dname, g in edges_df.groupby("added_domain", dropna=False):
        row: Dict[str, object] = {"added_domain": dname, "n_edges": int(len(g))}
        for m in metrics:
            col = f"{m}__oriented_delta"
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{m}__oriented_delta_mean"] = float(vals.mean()) if vals.notna().any() else float("nan")
            row[f"{m}__oriented_delta_median"] = float(vals.median()) if vals.notna().any() else float("nan")
            row[f"{m}__oriented_delta_pos_frac"] = float((vals > 0).mean()) if vals.notna().any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _build_anchored_pure_deltas(best_comp: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if best_comp.empty:
        return pd.DataFrame()

    idx: Dict[tuple, pd.Series] = {}
    for _, r in best_comp.iterrows():
        key = tuple(int(r[c]) for c in COMPOSITION_COLS)
        idx[key] = r

    rows: List[Dict[str, object]] = []
    for _, final in best_comp.iterrows():
        # Keep this focused on no-mmd/no-hof flow+appearance finals.
        if int(final["n_hof"]) != 0:
            continue
        if int(final["n_flow_mmd"]) != 0 or int(final["n_appearance_mmd"]) != 0 or int(final["n_other_mmd"]) != 0:
            continue
        f = int(final["n_flow"])
        a = int(final["n_appearance"])
        if f <= 0 or a <= 0:
            continue

        flow_base_key = (f, 0, 0, 0, 0, 0)
        app_base_key = (0, a, 0, 0, 0, 0)
        flow_base = idx.get(flow_base_key)
        app_base = idx.get(app_base_key)
        if flow_base is None or app_base is None:
            continue

        row: Dict[str, object] = {
            "final_method": final.get("method"),
            "flow_base_method": flow_base.get("method"),
            "appearance_base_method": app_base.get("method"),
            "final_n_flow": f,
            "final_n_appearance": a,
        }
        for m in metrics:
            vf = float(final.get(m)) if pd.notna(final.get(m)) else float("nan")
            vfb = float(flow_base.get(m)) if pd.notna(flow_base.get(m)) else float("nan")
            vab = float(app_base.get(m)) if pd.notna(app_base.get(m)) else float("nan")
            row[f"{m}__final"] = vf
            row[f"{m}__flow_base"] = vfb
            row[f"{m}__appearance_base"] = vab
            row[f"{m}__delta_vs_flow_base_oriented"] = _oriented_delta(m, vf, vfb)
            row[f"{m}__delta_vs_appearance_base_oriented"] = _oriented_delta(m, vf, vab)
            if math.isfinite(vfb) and math.isfinite(vab):
                row[f"{m}__flow_base_minus_appearance_base_oriented"] = _oriented_delta(m, vfb, vab)
            else:
                row[f"{m}__flow_base_minus_appearance_base_oriented"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build predictor-group claim tables from method_summary.csv")
    parser.add_argument("--summary", required=True, help="Path to method_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metric columns to carry into tables",
    )
    parser.add_argument(
        "--primary-metric",
        default="loto_rank_pairwise_cindex",
        help="Primary metric used for best-row selection and support flags",
    )
    parser.add_argument(
        "--close-gap",
        type=float,
        default=0.03,
        help="Threshold for considering hof close to flow on primary metric",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Missing summary: {summary_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"Empty summary: {summary_path}")
    if args.primary_metric not in df.columns:
        raise SystemExit(f"Primary metric not found: {args.primary_metric}")

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    metrics = [m for m in metrics if m in df.columns]
    if args.primary_metric not in metrics:
        metrics = [args.primary_metric] + metrics

    parsed = df.apply(_parse_group_row, axis=1)
    grouped = pd.concat([df.copy(), parsed], axis=1)
    for c in metrics:
        grouped[c] = pd.to_numeric(grouped[c], errors="coerce")

    group_counts = grouped["derived_group"].value_counts(dropna=False).rename_axis("derived_group").reset_index(name="n_rows")
    best_by_group = _best_by_metric(grouped, args.primary_metric, ["derived_group"])
    group_means = grouped.groupby("derived_group", dropna=False)[metrics].mean(numeric_only=True).reset_index()

    pure = grouped[grouped["derived_group"].isin(["flow_only", "appearance_only", "hof_only"])].copy()
    best_pure_by_k = _best_by_metric(pure, args.primary_metric, ["k_modal", "derived_group"])
    pure_deltas = _build_pure_deltas(best_pure_by_k, metrics, args.primary_metric, float(args.close_gap))
    support_summary = _summarize_support(pure_deltas)

    # MMD baseline quick comparison table.
    mmd = grouped[grouped["derived_group"].isin(["mmd_flow_only", "mmd_appearance_only", "mmd_both_only"])].copy()
    best_mmd = _best_by_metric(mmd, args.primary_metric, ["derived_group"])

    best_by_composition = _best_by_metric(grouped, args.primary_metric, COMPOSITION_COLS)
    incremental_edges = _build_incremental_edges(best_by_composition, metrics)
    incremental_summary = _summarize_incremental_edges(incremental_edges, metrics)
    anchored_pure_deltas = _build_anchored_pure_deltas(best_by_composition, metrics)

    grouped.to_csv(out_dir / "method_summary_with_derived_groups.csv", index=False)
    group_counts.to_csv(out_dir / "derived_group_counts.csv", index=False)
    best_by_group.to_csv(out_dir / "best_by_derived_group.csv", index=False)
    group_means.to_csv(out_dir / "means_by_derived_group.csv", index=False)
    best_pure_by_k.to_csv(out_dir / "pure_modalities_best_by_k.csv", index=False)
    pure_deltas.to_csv(out_dir / "pure_modalities_matched_k_deltas.csv", index=False)
    support_summary.to_csv(out_dir / "pure_modalities_support_counter_summary.csv", index=False)
    best_mmd.to_csv(out_dir / "mmd_baseline_best_rows.csv", index=False)
    best_by_composition.to_csv(out_dir / "best_by_composition.csv", index=False)
    incremental_edges.to_csv(out_dir / "incremental_domain_addition_deltas.csv", index=False)
    incremental_summary.to_csv(out_dir / "incremental_domain_addition_summary.csv", index=False)
    anchored_pure_deltas.to_csv(out_dir / "anchored_deltas_flow_vs_appearance_paths.csv", index=False)

    print(f"Wrote predictor-group tables to {out_dir}")


if __name__ == "__main__":
    main()
