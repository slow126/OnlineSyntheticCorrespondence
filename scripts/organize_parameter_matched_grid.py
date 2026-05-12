#!/usr/bin/env python3
"""
Organize an existing parameter-matched selection grid by predictor composition.

Reads:
  - parameter_matched_selection.csv
  - each row's summary_path (optional, for additional overall metrics)

Writes:
  - parsed rows with domain counts/signatures
  - flow-backbone grid views
  - flow-backbone deltas vs each backbone's own base row
  - addon consistency summaries across backbones
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


CONTROL_PREDICTORS = {
    "log_n_samples_eval",
    "log_avg_flows_eval",
    "log_n_samples_train",
    "log_avg_flows_train",
}


def _split_predictors(text: object) -> List[str]:
    if text is None or pd.isna(text):
        return []
    return [t.strip() for t in str(text).split(",") if t and str(t).strip()]


def _read_overall_summary(summary_path: Path) -> Dict[str, float]:
    if not summary_path.exists():
        return {}
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return {}
    if df.empty:
        return {}
    if "fold" not in df.columns:
        return {}
    overall = df[df["fold"] == "__overall__"]
    if overall.empty:
        return {}
    row = overall.iloc[0]
    return {
        "rank_pairwise_win_rate": float(row.get("pairwise_win_rate_micro", row.get("pairwise_win_rate", float("nan")))),
        "rank_spearman": float(row.get("rank_spearman_micro", row.get("rank_spearman", float("nan")))),
        "rank_regret": float(row.get("regret_micro", row.get("regret", float("nan")))),
    }


def _parse_row(row: pd.Series) -> Dict[str, object]:
    preds = _split_predictors(row.get("predictors"))
    controls = [p for p in preds if p in CONTROL_PREDICTORS]
    interactions = [p for p in preds if "_x_" in p]
    signal = [p for p in preds if p not in CONTROL_PREDICTORS and "_x_" not in p]

    flow = [p for p in signal if p.startswith("flow_") and "mmd" not in p]
    appearance = [p for p in signal if p.startswith("dino_") and "mmd" not in p]
    hof = [p for p in signal if p.startswith("hof_") and "mmd" not in p]

    flow_mmd = [p for p in signal if p == "flow_mmd" or (p.startswith("flow_") and "mmd" in p)]
    appearance_mmd = [
        p
        for p in signal
        if p in {"dino_mmd", "feature_mmd"} or (p.startswith("dino_") and "mmd" in p)
    ]
    other_mmd = [
        p
        for p in signal
        if "mmd" in p and p not in set(flow_mmd + appearance_mmd)
    ]
    other = [
        p
        for p in signal
        if not (
            p in set(flow + appearance + hof + flow_mmd + appearance_mmd + other_mmd)
        )
    ]

    flow_sig = "|".join(sorted(flow))
    appearance_sig = "|".join(sorted(appearance))
    hof_sig = "|".join(sorted(hof))
    mmd_sig = "|".join(sorted(flow_mmd + appearance_mmd + other_mmd))

    n_flow = len(flow)
    n_appearance = len(appearance)
    n_hof = len(hof)
    n_mmd = len(flow_mmd) + len(appearance_mmd) + len(other_mmd)

    # Backbone id: prioritize flow signature when present; otherwise appearance/hof/mmd signatures.
    if flow_sig:
        backbone_kind = "flow_backbone"
        backbone_id = flow_sig
    elif appearance_sig and not hof_sig and n_mmd == 0:
        backbone_kind = "appearance_backbone"
        backbone_id = appearance_sig
    elif hof_sig and not appearance_sig and n_mmd == 0:
        backbone_kind = "hof_backbone"
        backbone_id = hof_sig
    elif mmd_sig and not appearance_sig and not hof_sig:
        backbone_kind = "mmd_backbone"
        backbone_id = mmd_sig
    else:
        backbone_kind = "other_backbone"
        backbone_id = f"other::{appearance_sig}::{hof_sig}::{mmd_sig}"

    if flow_sig:
        addon_label_parts: List[str] = []
        if n_appearance > 0:
            addon_label_parts.append(f"+appearance({n_appearance})")
        if n_hof > 0:
            addon_label_parts.append(f"+hof({n_hof})")
        if len(flow_mmd) > 0:
            addon_label_parts.append(f"+flow_mmd({len(flow_mmd)})")
        if len(appearance_mmd) > 0:
            addon_label_parts.append(f"+appearance_mmd({len(appearance_mmd)})")
        if len(other_mmd) > 0:
            addon_label_parts.append(f"+other_mmd({len(other_mmd)})")
        addon_label = "base_flow_only" if not addon_label_parts else "".join(addon_label_parts)
    else:
        addon_label = "n/a"

    return {
        "predictors_raw": ",".join(preds),
        "signal_predictors": ",".join(signal),
        "n_signal": len(signal),
        "n_controls": len(controls),
        "n_interactions": len(interactions),
        "n_flow": n_flow,
        "n_appearance": n_appearance,
        "n_hof": n_hof,
        "n_flow_mmd": len(flow_mmd),
        "n_appearance_mmd": len(appearance_mmd),
        "n_other_mmd": len(other_mmd),
        "n_mmd_total": n_mmd,
        "n_other": len(other),
        "flow_signature": flow_sig,
        "appearance_signature": appearance_sig,
        "hof_signature": hof_sig,
        "mmd_signature": mmd_sig,
        "backbone_kind": backbone_kind,
        "backbone_id": backbone_id,
        "addon_label": addon_label,
    }


def _build_flow_deltas_vs_base(flow_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for backbone_id, g in flow_df.groupby("backbone_id", dropna=False):
        base = g[g["addon_label"] == "base_flow_only"]
        if base.empty:
            continue
        # best base row by metric
        base = base.sort_values(metric_col, ascending=False).iloc[0]
        base_metric = float(base[metric_col])
        for _, r in g.iterrows():
            if r["addon_label"] == "base_flow_only":
                continue
            cur = float(r[metric_col]) if pd.notna(r[metric_col]) else float("nan")
            rows.append(
                {
                    "backbone_id": backbone_id,
                    "base_bucket": base.get("bucket"),
                    "base_method": Path(str(base.get("run_dir", ""))).name,
                    "addon_bucket": r.get("bucket"),
                    "addon_method": Path(str(r.get("run_dir", ""))).name,
                    "addon_label": r.get("addon_label"),
                    f"{metric_col}__base": base_metric,
                    f"{metric_col}__addon": cur,
                    f"{metric_col}__delta": cur - base_metric if math.isfinite(cur) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _summarize_addon_consistency(deltas_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    if deltas_df.empty:
        return pd.DataFrame()
    dcol = f"{metric_col}__delta"
    rows: List[Dict[str, object]] = []
    for addon, g in deltas_df.groupby("addon_label", dropna=False):
        vals = pd.to_numeric(g[dcol], errors="coerce")
        rows.append(
            {
                "addon_label": addon,
                "n_backbones": int(vals.notna().sum()),
                "delta_mean": float(vals.mean()) if vals.notna().any() else float("nan"),
                "delta_median": float(vals.median()) if vals.notna().any() else float("nan"),
                "delta_pos_frac": float((vals > 0).mean()) if vals.notna().any() else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("delta_mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize existing parameter-matched comparison grid")
    parser.add_argument("--selection-csv", required=True, help="Path to parameter_matched_selection.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--primary-metric-col",
        default="metric_value",
        help="Metric column to use for deltas (default uses selection metric values)",
    )
    parser.add_argument(
        "--augment-from-summary",
        action="store_true",
        help="Also parse each summary_path overall row into additional rank metrics",
    )
    args = parser.parse_args()

    sel_path = Path(args.selection_csv)
    if not sel_path.exists():
        raise SystemExit(f"Missing selection CSV: {sel_path}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sel = pd.read_csv(sel_path)
    if sel.empty:
        raise SystemExit(f"Empty selection CSV: {sel_path}")

    parsed = sel.apply(_parse_row, axis=1, result_type="expand")
    work = pd.concat([sel.copy(), parsed], axis=1)
    work["metric_value"] = pd.to_numeric(work["metric_value"], errors="coerce")

    if args.augment_from_summary:
        extra_rows: List[Dict[str, float]] = []
        for _, r in work.iterrows():
            extra_rows.append(_read_overall_summary(Path(str(r.get("summary_path", "")))))
        extra_df = pd.DataFrame(extra_rows)
        work = pd.concat([work, extra_df], axis=1)

    # Flow-backbone focused views.
    flow = work[work["backbone_kind"] == "flow_backbone"].copy()
    metric_col = args.primary_metric_col
    if metric_col not in work.columns:
        raise SystemExit(f"Primary metric column not found: {metric_col}")
    flow[metric_col] = pd.to_numeric(flow[metric_col], errors="coerce")
    flow = flow.sort_values([metric_col], ascending=False)

    flow_pivot = flow.pivot_table(
        index=["backbone_id"],
        columns="addon_label",
        values=metric_col,
        aggfunc="max",
    ).reset_index()

    flow_deltas = _build_flow_deltas_vs_base(flow, metric_col)
    flow_consistency = _summarize_addon_consistency(flow_deltas, metric_col)

    # High-level organization outputs.
    work.sort_values(["backbone_kind", "backbone_id", "metric_value"], ascending=[True, True, False]).to_csv(
        out_dir / "organized_parameter_matched_rows.csv", index=False
    )
    work.groupby(["backbone_kind", "addon_label"], dropna=False).size().reset_index(name="n_rows").to_csv(
        out_dir / "organized_group_counts.csv", index=False
    )
    flow.to_csv(out_dir / "flow_backbone_grid_rows.csv", index=False)
    flow_pivot.to_csv(out_dir / "flow_backbone_grid_pivot.csv", index=False)
    flow_deltas.to_csv(out_dir / "flow_backbone_deltas_vs_base.csv", index=False)
    flow_consistency.to_csv(out_dir / "flow_backbone_addon_consistency.csv", index=False)

    print(f"Wrote organized grid tables to {out_dir}")


if __name__ == "__main__":
    main()

