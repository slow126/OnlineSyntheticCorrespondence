#!/usr/bin/env python3
"""
Analyze smoothness tables to compare synthetic 3D vs 2D warps.

Reads the summary table CSV produced by plot_smoothness_metrics.py and
computes per-encoder deltas against SPAIR-only, plus a direct comparison
between synthetic 3D mixes and 2D warp mixes.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from plot_smoothness_metrics import (
    parse_checkpoint_name,
    format_mix_label,
    encoder_group_label,
)


SYNTH_CONDITIONS = {"spair_synthetic", "synthetic_only"}
SYNTH_2D_CONDITIONS = {"synthetic_2dwarp"}
IMAGENET_2D_CONDITIONS = {"spair_2dwarp", "2dwarp_only"}
BASE_ONLY_CONDITIONS = {"synthetic_only", "synthetic_2dwarp", "2dwarp_only"}


def _resolve_pck_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("mean_pck", "best_pck"):
        if col in df.columns:
            return col
    return None


def _pick_best(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
    if df.empty:
        return None
    return df.sort_values(metric, ascending=True).iloc[0]


def _summarize_encoder(group_df: pd.DataFrame, min_n: int, pck_col: Optional[str]) -> Dict[str, object]:
    group_df = group_df.copy()
    group_df = group_df[group_df["n"].fillna(0) >= min_n]
    encoder = group_df["encoder_group"].iloc[0] if not group_df.empty else "unknown"

    baseline = group_df[group_df["condition"] == "spair_only"]
    baseline_row = _pick_best(baseline, "mean_tv")
    baseline_pck = float(baseline_row[pck_col]) if (pck_col and baseline_row is not None and pd.notna(baseline_row[pck_col])) else None

    synth_rows = group_df[group_df["condition"].isin(SYNTH_CONDITIONS)]
    synth2d_rows = group_df[group_df["condition"].isin(SYNTH_2D_CONDITIONS)]
    imagenet2d_rows = group_df[group_df["condition"].isin(IMAGENET_2D_CONDITIONS)]

    best_synth_tv = _pick_best(synth_rows, "mean_tv")
    best_synth_lap = _pick_best(synth_rows, "mean_laplacian")
    best_synth2d_tv = _pick_best(synth2d_rows, "mean_tv")
    best_synth2d_lap = _pick_best(synth2d_rows, "mean_laplacian")
    best_imagenet2d_tv = _pick_best(imagenet2d_rows, "mean_tv")
    best_imagenet2d_lap = _pick_best(imagenet2d_rows, "mean_laplacian")

    def _delta(row, baseline_row, col):
        if row is None or baseline_row is None:
            return None
        return float(row[col] - baseline_row[col])

    def _delta_pck(row):
        if not pck_col or row is None or baseline_pck is None or pd.isna(row.get(pck_col)):
            return None
        return float(row[pck_col] - baseline_pck)

    def _best_joint(rows: pd.DataFrame, metric: str) -> Optional[pd.Series]:
        if baseline_row is None or not pck_col or baseline_pck is None:
            return None
        filtered = rows.copy()
        filtered = filtered[pd.notna(filtered[pck_col])]
        filtered["delta_tv"] = filtered["mean_tv"] - baseline_row["mean_tv"]
        filtered["delta_lap"] = filtered["mean_laplacian"] - baseline_row["mean_laplacian"]
        filtered["delta_pck"] = filtered[pck_col] - baseline_pck
        filtered = filtered[filtered["delta_pck"] > 0]
        filtered = filtered[(filtered["delta_tv"] < 0) | (filtered["delta_lap"] < 0)]
        if filtered.empty:
            return None
        return filtered.sort_values(metric, ascending=True).iloc[0]

    best_synth_joint_tv = _best_joint(synth_rows, "mean_tv")
    best_synth_joint_lap = _best_joint(synth_rows, "mean_laplacian")
    best_synth2d_joint_tv = _best_joint(synth2d_rows, "mean_tv")
    best_synth2d_joint_lap = _best_joint(synth2d_rows, "mean_laplacian")
    best_imagenet2d_joint_tv = _best_joint(imagenet2d_rows, "mean_tv")
    best_imagenet2d_joint_lap = _best_joint(imagenet2d_rows, "mean_laplacian")

    def _joint_score(row):
        if row is None:
            return None
        return float(row["mean_tv"] + row["mean_laplacian"])

    synth_joint_score = _joint_score(best_synth_joint_tv) if best_synth_joint_tv is not None else _joint_score(best_synth_joint_lap)
    synth2d_joint_score = _joint_score(best_synth2d_joint_tv) if best_synth2d_joint_tv is not None else _joint_score(best_synth2d_joint_lap)
    imagenet2d_joint_score = _joint_score(best_imagenet2d_joint_tv) if best_imagenet2d_joint_tv is not None else _joint_score(best_imagenet2d_joint_lap)

    def _rank_triplet(a, b, c):
        items = [("synthetic_3d", a), ("synthetic_2d", b), ("imagenet_2d", c)]
        items = [(name, val) for name, val in items if val is not None]
        if not items:
            return None, "no_joint_improvement"
        items.sort(key=lambda x: x[1])
        order = " > ".join([name for name, _ in items])
        return order, items[0][0]

    order, winner = _rank_triplet(synth_joint_score, synth2d_joint_score, imagenet2d_joint_score)

    return {
        "encoder_group": encoder,
        "baseline_label": baseline_row["label"] if baseline_row is not None else None,
        "baseline_tv": float(baseline_row["mean_tv"]) if baseline_row is not None else None,
        "baseline_laplacian": float(baseline_row["mean_laplacian"]) if baseline_row is not None else None,
        "baseline_pck": baseline_pck,
        "best_synth_label_tv": best_synth_tv["label"] if best_synth_tv is not None else None,
        "best_synth_tv": float(best_synth_tv["mean_tv"]) if best_synth_tv is not None else None,
        "best_synth_label_lap": best_synth_lap["label"] if best_synth_lap is not None else None,
        "best_synth_laplacian": float(best_synth_lap["mean_laplacian"]) if best_synth_lap is not None else None,
        "best_synth2d_label_tv": best_synth2d_tv["label"] if best_synth2d_tv is not None else None,
        "best_synth2d_tv": float(best_synth2d_tv["mean_tv"]) if best_synth2d_tv is not None else None,
        "best_synth2d_label_lap": best_synth2d_lap["label"] if best_synth2d_lap is not None else None,
        "best_synth2d_laplacian": float(best_synth2d_lap["mean_laplacian"]) if best_synth2d_lap is not None else None,
        "best_imagenet2d_label_tv": best_imagenet2d_tv["label"] if best_imagenet2d_tv is not None else None,
        "best_imagenet2d_tv": float(best_imagenet2d_tv["mean_tv"]) if best_imagenet2d_tv is not None else None,
        "best_imagenet2d_label_lap": best_imagenet2d_lap["label"] if best_imagenet2d_lap is not None else None,
        "best_imagenet2d_laplacian": float(best_imagenet2d_lap["mean_laplacian"]) if best_imagenet2d_lap is not None else None,
        "delta_synth_vs_spair_tv": _delta(best_synth_tv, baseline_row, "mean_tv"),
        "delta_synth_vs_spair_laplacian": _delta(best_synth_lap, baseline_row, "mean_laplacian"),
        "delta_synth_vs_spair_tv": _delta(best_synth_tv, baseline_row, "mean_tv"),
        "delta_synth2d_vs_spair_tv": _delta(best_synth2d_tv, baseline_row, "mean_tv"),
        "delta_imagenet2d_vs_spair_tv": _delta(best_imagenet2d_tv, baseline_row, "mean_tv"),
        "delta_synth_vs_spair_laplacian": _delta(best_synth_lap, baseline_row, "mean_laplacian"),
        "delta_synth2d_vs_spair_laplacian": _delta(best_synth2d_lap, baseline_row, "mean_laplacian"),
        "delta_imagenet2d_vs_spair_laplacian": _delta(best_imagenet2d_lap, baseline_row, "mean_laplacian"),
        "best_synth_pck": float(best_synth_tv[pck_col]) if (pck_col and best_synth_tv is not None and pd.notna(best_synth_tv[pck_col])) else None,
        "best_synth2d_pck": float(best_synth2d_tv[pck_col]) if (pck_col and best_synth2d_tv is not None and pd.notna(best_synth2d_tv[pck_col])) else None,
        "best_imagenet2d_pck": float(best_imagenet2d_tv[pck_col]) if (pck_col and best_imagenet2d_tv is not None and pd.notna(best_imagenet2d_tv[pck_col])) else None,
        "delta_synth_vs_spair_pck": _delta_pck(best_synth_tv),
        "delta_synth2d_vs_spair_pck": _delta_pck(best_synth2d_tv),
        "delta_imagenet2d_vs_spair_pck": _delta_pck(best_imagenet2d_tv),
        "best_synth_joint_label_tv": best_synth_joint_tv["label"] if best_synth_joint_tv is not None else None,
        "best_synth_joint_tv": float(best_synth_joint_tv["mean_tv"]) if best_synth_joint_tv is not None else None,
        "best_synth_joint_pck_tv": float(best_synth_joint_tv[pck_col]) if (pck_col and best_synth_joint_tv is not None) else None,
        "best_synth_joint_label_lap": best_synth_joint_lap["label"] if best_synth_joint_lap is not None else None,
        "best_synth_joint_laplacian": float(best_synth_joint_lap["mean_laplacian"]) if best_synth_joint_lap is not None else None,
        "best_synth_joint_pck_lap": float(best_synth_joint_lap[pck_col]) if (pck_col and best_synth_joint_lap is not None) else None,
        "best_synth2d_joint_label_tv": best_synth2d_joint_tv["label"] if best_synth2d_joint_tv is not None else None,
        "best_synth2d_joint_tv": float(best_synth2d_joint_tv["mean_tv"]) if best_synth2d_joint_tv is not None else None,
        "best_synth2d_joint_pck_tv": float(best_synth2d_joint_tv[pck_col]) if (pck_col and best_synth2d_joint_tv is not None) else None,
        "best_synth2d_joint_label_lap": best_synth2d_joint_lap["label"] if best_synth2d_joint_lap is not None else None,
        "best_synth2d_joint_laplacian": float(best_synth2d_joint_lap["mean_laplacian"]) if best_synth2d_joint_lap is not None else None,
        "best_synth2d_joint_pck_lap": float(best_synth2d_joint_lap[pck_col]) if (pck_col and best_synth2d_joint_lap is not None) else None,
        "best_imagenet2d_joint_label_tv": best_imagenet2d_joint_tv["label"] if best_imagenet2d_joint_tv is not None else None,
        "best_imagenet2d_joint_tv": float(best_imagenet2d_joint_tv["mean_tv"]) if best_imagenet2d_joint_tv is not None else None,
        "best_imagenet2d_joint_pck_tv": float(best_imagenet2d_joint_tv[pck_col]) if (pck_col and best_imagenet2d_joint_tv is not None) else None,
        "best_imagenet2d_joint_label_lap": best_imagenet2d_joint_lap["label"] if best_imagenet2d_joint_lap is not None else None,
        "best_imagenet2d_joint_laplacian": float(best_imagenet2d_joint_lap["mean_laplacian"]) if best_imagenet2d_joint_lap is not None else None,
        "best_imagenet2d_joint_pck_lap": float(best_imagenet2d_joint_lap[pck_col]) if (pck_col and best_imagenet2d_joint_lap is not None) else None,
        "order": order,
        "winner": winner,
    }


def _write_summary_txt(summary_df: pd.DataFrame, output_path: Path, pck_col: Optional[str], full_df: Optional[pd.DataFrame] = None):
    lines = []
    lines.append("Smoothness regularization summary (compact)\n")
    if not pck_col:
        lines.append("NOTE: No PCK column found in input. PCK deltas are omitted.\n")
    winner_counts = summary_df["winner"].value_counts(dropna=False).to_dict()
    lines.append(f"Winner counts: {winner_counts}\n")
    for _, row in summary_df.iterrows():
        lines.append(f"Encoder: {row['encoder_group']}")
        lines.append(f"  Baseline: {row['baseline_label']} (TV={row['baseline_tv']}, Lap={row['baseline_laplacian']}, PCK={row['baseline_pck']})")
        lines.append(f"  Joint-order (best→worst): {row['order']}")
        if pck_col:
            lines.append(f"  Best synth joint (TV+PCK): {row['best_synth_joint_label_tv']} -> {row['best_synth_joint_tv']} (PCK {row['best_synth_joint_pck_tv']})")
            lines.append(f"  Best synth2d joint (TV+PCK): {row['best_synth2d_joint_label_tv']} -> {row['best_synth2d_joint_tv']} (PCK {row['best_synth2d_joint_pck_tv']})")
            lines.append(f"  Best imagenet2d joint (TV+PCK): {row['best_imagenet2d_joint_label_tv']} -> {row['best_imagenet2d_joint_tv']} (PCK {row['best_imagenet2d_joint_pck_tv']})")
        if pck_col:
            lines.append(f"  Best synth joint (Lap+PCK): {row['best_synth_joint_label_lap']} -> {row['best_synth_joint_laplacian']} (PCK {row['best_synth_joint_pck_lap']})")
            lines.append(f"  Best synth2d joint (Lap+PCK): {row['best_synth2d_joint_label_lap']} -> {row['best_synth2d_joint_laplacian']} (PCK {row['best_synth2d_joint_pck_lap']})")
            lines.append(f"  Best imagenet2d joint (Lap+PCK): {row['best_imagenet2d_joint_label_lap']} -> {row['best_imagenet2d_joint_laplacian']} (PCK {row['best_imagenet2d_joint_pck_lap']})")
        if full_df is not None:
            encoder_df = full_df[full_df["encoder_group"] == row["encoder_group"]].copy()
            if not encoder_df.empty:
                encoder_df["score"] = encoder_df["mean_tv"] + encoder_df["mean_laplacian"]
                if pck_col and pck_col in encoder_df.columns and encoder_df[pck_col].notna().any():
                    ordered = encoder_df.sort_values(pck_col, ascending=False)
                    lines.append("  Full order by PCK:")
                else:
                    ordered = encoder_df.sort_values("score", ascending=True)
                    lines.append("  Full order by TV+Lap:")
                for _, full_row in ordered.iterrows():
                    lines.append(
                        f"    - {full_row['label']} ({full_row['condition']}): "
                        f"PCK={full_row.get(pck_col)}, TV={full_row['mean_tv']}, Lap={full_row['mean_laplacian']}"
                    )
        lines.append("\n" + "-" * 72 + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _write_base_only_comparison(df: pd.DataFrame, output_path: Path, pck_col: Optional[str]) -> None:
    base_df = df[df["condition"].isin(BASE_ONLY_CONDITIONS)].copy()
    if base_df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("No base-only rows found.\n")
        return

    rows = []
    for _, row in base_df.iterrows():
        rows.append({
            "encoder_group": row["encoder_group"],
            "condition": row["condition"],
            "label": row.get("label"),
            "mean_tv": row.get("mean_tv"),
            "mean_laplacian": row.get("mean_laplacian"),
            "pck": row.get(pck_col) if pck_col else None,
            "n": row.get("n", 1),
        })
    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)


def _write_base_only_summary(df: pd.DataFrame, output_path: Path, pck_col: Optional[str]) -> None:
    base_df = df[df["condition"].isin(BASE_ONLY_CONDITIONS)].copy()
    lines = ["Base-only performance summary\n"]
    if base_df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("No base-only rows found.\n")
        return

    for encoder, group_df in base_df.groupby("encoder_group", dropna=False):
        group_df = group_df.copy()
        if not group_df.empty:
            group_df["score"] = group_df["mean_tv"] + group_df["mean_laplacian"]
        if pck_col and pck_col in group_df.columns:
            group_df = group_df[pd.notna(group_df[pck_col])]
            if not group_df.empty:
                best_row = group_df.sort_values(pck_col, ascending=False).iloc[0]
                reason = f"best {pck_col}"
            else:
                best_row = None
        else:
            best_row = None

        if best_row is None and not group_df.empty:
            best_row = group_df.sort_values("score", ascending=True).iloc[0]
            reason = "lowest tv+lap"
        elif best_row is None:
            reason = "no data"

        lines.append(f"Encoder: {encoder}")
        if best_row is None:
            lines.append("  Best: n/a")
        else:
            lines.append(f"  Best: {best_row['label']} ({best_row['condition']})")
            lines.append(f"  TV={best_row['mean_tv']}, Lap={best_row['mean_laplacian']}, PCK={best_row.get(pck_col)}")
            lines.append(f"  Reason: {reason}")
        if not group_df.empty:
            if pck_col and pck_col in group_df.columns:
                ordered = group_df.sort_values(pck_col, ascending=False)
                lines.append("  Order by PCK:")
            else:
                ordered = group_df.sort_values("score", ascending=True)
                lines.append("  Order by TV+Lap:")
            for _, row in ordered.iterrows():
                lines.append(
                    f"    - {row['label']} ({row['condition']}): "
                    f"PCK={row.get(pck_col)}, TV={row['mean_tv']}, Lap={row['mean_laplacian']}"
                )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Analyze smoothness regularization effects.")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to smoothness_comparison_*_table.csv",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path for summary (default: alongside input).",
    )
    parser.add_argument(
        "--output-txt",
        default=None,
        help="Output TXT path for readable summary (default: alongside input).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=1,
        help="Minimum n per row to include in comparisons.",
    )
    parser.add_argument(
        "--write-base-only",
        action="store_true",
        help="Also write a base-only (synthetic/2dwarp) comparison CSV.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    df = pd.read_csv(input_path)

    if df.empty:
        raise SystemExit("Input CSV is empty.")

    if "encoder_group" not in df.columns or "condition" not in df.columns:
        if "checkpoint_name" not in df.columns:
            raise SystemExit("Input CSV must include encoder_group/condition or checkpoint_name columns.")
        rows = []
        for _, row in df.iterrows():
            condition, mix_ratio, model_type, pretrained, freeze = parse_checkpoint_name(
                row["checkpoint_name"]
            )
            label = format_mix_label(condition, mix_ratio, row["checkpoint_name"])
            group = encoder_group_label(model_type, pretrained, freeze)
            rows.append({
                "encoder_group": group,
                "condition": condition,
                "mix_ratio": mix_ratio,
                "label": label,
                "mean_tv": row["mean_tv"],
                "mean_laplacian": row["mean_laplacian"],
                "best_pck": row.get("best_pck"),
                "n": 1,
            })
        df = pd.DataFrame(rows)

    summaries = []
    pck_col = _resolve_pck_column(df)
    for encoder, group_df in df.groupby("encoder_group", dropna=False):
        summaries.append(_summarize_encoder(group_df, args.min_n, pck_col))

    summary_df = pd.DataFrame(summaries)

    output_csv = Path(args.output_csv) if args.output_csv else input_path.with_name(input_path.stem + "_regularization_summary.csv")
    output_txt = Path(args.output_txt) if args.output_txt else input_path.with_name(input_path.stem + "_regularization_summary.txt")

    summary_df.to_csv(output_csv, index=False)
    _write_summary_txt(summary_df, output_txt, pck_col, full_df=df)
    if args.write_base_only:
        base_only_output = input_path.with_name(input_path.stem + "_base_only.csv")
        _write_base_only_comparison(df, base_only_output, pck_col)
        base_only_summary = input_path.with_name(input_path.stem + "_base_only_summary.txt")
        _write_base_only_summary(df, base_only_summary, pck_col)

    print(f"Saved summary CSV to: {output_csv}")
    print(f"Saved summary TXT to: {output_txt}")
    if args.write_base_only:
        print(f"Saved base-only CSV to: {base_only_output}")
        print(f"Saved base-only summary to: {base_only_summary}")


if __name__ == "__main__":
    main()
