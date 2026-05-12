#!/usr/bin/env python3
"""
Build consistent, reviewer-readable claim tables with fixed columns per table.

No new experiments are run. Inputs are precomputed aggregate tables.
Outputs:
  - claim_rank_deltas_table.csv/.tex
  - claim_error_deltas_table.csv/.tex
  - coverage_counts_table.csv/.tex
  - include_all_tables.tex
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return "--"
    return f"{v:.{nd}f}"


def _fmt_pct(v: float, nd: int = 1) -> str:
    if v is None or not math.isfinite(v):
        return "--"
    return f"{100.0 * v:.{nd}f}%"


def _row_style(n: float, min_n: int) -> str:
    if not math.isfinite(n):
        return "context"
    if n < min_n:
        return "sparse"
    return "standard"


def _write_tex(df: pd.DataFrame, out_path: Path, caption: str, label: str, colfmt: str) -> None:
    tex_df = df.copy()
    for col in tex_df.columns:
        if col.endswith("_mean"):
            tex_df[col] = pd.to_numeric(tex_df[col], errors="coerce").map(_fmt)
        elif col.endswith("_pos_frac"):
            tex_df[col] = pd.to_numeric(tex_df[col], errors="coerce").map(_fmt_pct)
    tab = tex_df.to_latex(index=False, escape=True, column_format=colfmt)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\resizebox{\linewidth}{!}{%",
        tab.strip(),
        r"}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _directional_rows(method_summary: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    df = pd.read_csv(method_summary)
    asym = df[df["symmetry"] == "asym"].copy()
    sym = df[df["symmetry"] == "sym"].copy()

    c_asym = _num(asym["loto_rank_pairwise_cindex"]).dropna()
    c_sym = _num(sym["loto_rank_pairwise_cindex"]).dropna()
    s_asym = _num(asym["loto_rank_spearman"]).dropna()
    s_sym = _num(sym["loto_rank_spearman"]).dropna()
    mae_asym = _num(asym["loto_mae"]).dropna()
    mae_sym = _num(sym["loto_mae"]).dropna()
    rmse_asym = _num(asym["loto_rmse"]).dropna()
    rmse_sym = _num(sym["loto_rmse"]).dropna()

    n = float(min(len(c_asym), len(c_sym), len(s_asym), len(s_sym)))
    rank_row = {
        "test": "Directional vs symmetric",
        "comparison": "Asymmetric predictors vs symmetric predictors",
        "n": int(n),
        "cindex_delta_mean": float(c_asym.mean() - c_sym.mean()),
        "rank_spearman_delta_mean": float(s_asym.mean() - s_sym.mean()),
        "cindex_pos_frac": float("nan"),
        "rank_spearman_pos_frac": float("nan"),
    }
    # Error metrics oriented so positive is better => symmetric - asymmetric
    err_row = {
        "test": "Directional vs symmetric",
        "comparison": "Asymmetric predictors vs symmetric predictors",
        "n": int(min(len(mae_asym), len(mae_sym), len(rmse_asym), len(rmse_sym))),
        "mae_delta_mean": float(mae_sym.mean() - mae_asym.mean()),
        "rmse_delta_mean": float(rmse_sym.mean() - rmse_asym.mean()),
        "mae_pos_frac": float("nan"),
        "rmse_pos_frac": float("nan"),
    }
    return rank_row, err_row


def _pure_rows(pure_path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    df = pd.read_csv(pure_path)
    pairs = [("flow", "appearance"), ("flow", "hof"), ("hof", "appearance")]

    rank_rows: List[Dict[str, object]] = []
    err_rows: List[Dict[str, object]] = []
    for left, right in pairs:
        c = _num(df[f"loto_rank_pairwise_cindex__{left}_minus_{right}_oriented"]).dropna()
        s = _num(df[f"loto_rank_spearman__{left}_minus_{right}_oriented"]).dropna()
        mae = _num(df[f"loto_mae__{left}_minus_{right}_oriented"]).dropna()
        rmse = _num(df[f"loto_rmse__{left}_minus_{right}_oriented"]).dropna()

        n_rank = int(min(len(c), len(s))) if len(c) and len(s) else 0
        n_err = int(min(len(mae), len(rmse))) if len(mae) and len(rmse) else 0

        rank_rows.append(
            {
                "test": "Pure modality, matched-k",
                "comparison": f"{left.upper()} vs {right.upper()}",
                "n": n_rank,
                "cindex_delta_mean": float(c.mean()) if len(c) else float("nan"),
                "rank_spearman_delta_mean": float(s.mean()) if len(s) else float("nan"),
                "cindex_pos_frac": float((c > 0).mean()) if len(c) else float("nan"),
                "rank_spearman_pos_frac": float((s > 0).mean()) if len(s) else float("nan"),
            }
        )
        err_rows.append(
            {
                "test": "Pure modality, matched-k",
                "comparison": f"{left.upper()} vs {right.upper()}",
                "n": n_err,
                "mae_delta_mean": float(mae.mean()) if len(mae) else float("nan"),
                "rmse_delta_mean": float(rmse.mean()) if len(rmse) else float("nan"),
                "mae_pos_frac": float((mae > 0).mean()) if len(mae) else float("nan"),
                "rmse_pos_frac": float((rmse > 0).mean()) if len(rmse) else float("nan"),
            }
        )
    return rank_rows, err_rows


def _incremental_rows(inc_path: Path) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    df = pd.read_csv(inc_path)
    preferred_order = ["appearance", "hof", "flow", "appearance_mmd", "flow_mmd", "other_mmd"]
    domains = [d for d in preferred_order if d in set(df["added_domain"].astype(str).tolist())]
    # Keep any extra domains that may appear in future runs.
    for d in sorted(set(df["added_domain"].astype(str).tolist())):
        if d not in domains:
            domains.append(d)
    rank_rows: List[Dict[str, object]] = []
    err_rows: List[Dict[str, object]] = []
    for d in domains:
        g = df[df["added_domain"] == d]
        if g.empty:
            continue
        r = g.iloc[0]
        n_edges = int(r["n_edges"]) if pd.notna(r["n_edges"]) else 0
        rank_rows.append(
            {
                "test": "Incremental one-step addition",
                "comparison": f"+{d.upper()} from matched neighbors",
                "n": n_edges,
                "cindex_delta_mean": float(r["loto_rank_pairwise_cindex__oriented_delta_mean"]),
                "rank_spearman_delta_mean": float(r["loto_rank_spearman__oriented_delta_mean"]),
                "cindex_pos_frac": float(r["loto_rank_pairwise_cindex__oriented_delta_pos_frac"]),
                "rank_spearman_pos_frac": float(r["loto_rank_spearman__oriented_delta_pos_frac"]),
            }
        )
        err_rows.append(
            {
                "test": "Incremental one-step addition",
                "comparison": f"+{d.upper()} from matched neighbors",
                "n": n_edges,
                "mae_delta_mean": float(r["loto_mae__oriented_delta_mean"]),
                "rmse_delta_mean": float(r["loto_rmse__oriented_delta_mean"]),
                "mae_pos_frac": float(r["loto_mae__oriented_delta_pos_frac"]),
                "rmse_pos_frac": float(r["loto_rmse__oriented_delta_pos_frac"]),
            }
        )
    return rank_rows, err_rows


def _coverage_rows(method_summary: Path, organized: Path) -> pd.DataFrame:
    ms = pd.read_csv(method_summary)
    og = pd.read_csv(organized)
    out = [
        {"scope": "Full method pool", "item": "Total methods", "count": int(len(ms))},
        {"scope": "Full method pool", "item": "Asymmetric methods", "count": int((ms["symmetry"] == "asym").sum())},
        {"scope": "Full method pool", "item": "Symmetric methods", "count": int((ms["symmetry"] == "sym").sum())},
        {"scope": "Full method pool", "item": "Mixed-symmetry methods", "count": int((ms["symmetry"] == "mixed").sum())},
        {"scope": "Parameter-matched subset", "item": "Rows", "count": int(len(og))},
        {
            "scope": "Parameter-matched subset",
            "item": "Unique flow backbones",
            "count": int(og[og["backbone_kind"] == "flow_backbone"]["backbone_id"].nunique()),
        },
        {
            "scope": "Parameter-matched subset",
            "item": "Unique appearance backbones",
            "count": int(og[og["backbone_kind"] == "appearance_backbone"]["backbone_id"].nunique()),
        },
        {
            "scope": "Parameter-matched subset",
            "item": "Unique MMD backbones",
            "count": int(og[og["backbone_kind"] == "mmd_backbone"]["backbone_id"].nunique()),
        },
    ]
    return pd.DataFrame(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-schema claim tables")
    parser.add_argument("--predictor-group-dir", required=True)
    parser.add_argument("--organized-grid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-n-standard", type=int, default=3, help="N threshold below which rows are flagged sparse")
    args = parser.parse_args()

    pdir = Path(args.predictor_group_dir)
    odir = Path(args.organized_grid_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    method_summary = pdir / "method_summary_with_derived_groups.csv"
    pure_path = pdir / "pure_modalities_matched_k_deltas.csv"
    inc_path = pdir / "incremental_domain_addition_summary.csv"
    organized = odir / "organized_parameter_matched_rows.csv"

    rank_dir_row, err_dir_row = _directional_rows(method_summary)
    rank_pure_rows, err_pure_rows = _pure_rows(pure_path)
    rank_inc_rows, err_inc_rows = _incremental_rows(inc_path)

    rank_df = pd.DataFrame([rank_dir_row] + rank_pure_rows + rank_inc_rows)
    rank_df["evidence"] = rank_df["n"].map(lambda n: _row_style(float(n), args.min_n_standard))
    rank_cols = [
        "test",
        "comparison",
        "n",
        "cindex_delta_mean",
        "rank_spearman_delta_mean",
        "cindex_pos_frac",
        "rank_spearman_pos_frac",
        "evidence",
    ]
    rank_df = rank_df[rank_cols]

    err_df = pd.DataFrame([err_dir_row] + err_pure_rows + err_inc_rows)
    err_df["evidence"] = err_df["n"].map(lambda n: _row_style(float(n), args.min_n_standard))
    err_cols = [
        "test",
        "comparison",
        "n",
        "mae_delta_mean",
        "rmse_delta_mean",
        "mae_pos_frac",
        "rmse_pos_frac",
        "evidence",
    ]
    err_df = err_df[err_cols]

    # Combined table: one row-set with both ranking and error metrics.
    rank_work = rank_df.copy().rename(columns={"n": "n_rank", "evidence": "evidence_rank"})
    err_work = err_df.copy().rename(columns={"n": "n_error", "evidence": "evidence_error"})
    combined_df = rank_work.merge(
        err_work[
            [
                "test",
                "comparison",
                "n_error",
                "mae_delta_mean",
                "rmse_delta_mean",
                "mae_pos_frac",
                "rmse_pos_frac",
                "evidence_error",
            ]
        ],
        on=["test", "comparison"],
        how="outer",
    )
    combined_df["n"] = pd.to_numeric(combined_df["n_rank"], errors="coerce")
    missing_n = combined_df["n"].isna()
    combined_df.loc[missing_n, "n"] = pd.to_numeric(combined_df.loc[missing_n, "n_error"], errors="coerce")
    combined_df["n"] = combined_df["n"].map(lambda v: int(v) if pd.notna(v) else v)
    combined_df["evidence"] = combined_df["n"].map(
        lambda n: _row_style(float(n), args.min_n_standard) if pd.notna(n) else "context"
    )
    order_map = {(r["test"], r["comparison"]): i for i, r in rank_df.reset_index().iterrows()}
    combined_df["__order"] = combined_df.apply(
        lambda r: order_map.get((r["test"], r["comparison"]), 10_000),
        axis=1,
    )
    combined_df = combined_df.sort_values("__order").drop(columns=["__order"])
    combined_df = combined_df[
        [
            "test",
            "comparison",
            "n",
            "cindex_delta_mean",
            "rank_spearman_delta_mean",
            "mae_delta_mean",
            "rmse_delta_mean",
            "cindex_pos_frac",
            "rank_spearman_pos_frac",
            "mae_pos_frac",
            "rmse_pos_frac",
            "evidence",
        ]
    ]

    cov_df = _coverage_rows(method_summary, organized)

    rank_csv = out / "claim_rank_deltas_table.csv"
    err_csv = out / "claim_error_deltas_table.csv"
    combined_csv = out / "claim_combined_deltas_table.csv"
    cov_csv = out / "coverage_counts_table.csv"
    rank_df.to_csv(rank_csv, index=False)
    err_df.to_csv(err_csv, index=False)
    combined_df.to_csv(combined_csv, index=False)
    cov_df.to_csv(cov_csv, index=False)

    rank_tex = out / "claim_rank_deltas_table.tex"
    err_tex = out / "claim_error_deltas_table.tex"
    combined_tex = out / "claim_combined_deltas_table.tex"
    cov_tex = out / "coverage_counts_table.tex"
    _write_tex(
        rank_df,
        rank_tex,
        caption="Ranking Metrics: Oriented Mean Delta Improvements",
        label="tab:claim_rank_deltas",
        colfmt="llrccccl",
    )
    _write_tex(
        err_df,
        err_tex,
        caption="Error Metrics: Oriented Mean Delta Improvements",
        label="tab:claim_error_deltas",
        colfmt="llrccccl",
    )
    _write_tex(
        combined_df,
        combined_tex,
        caption="Combined Metrics: Oriented Mean Delta Improvements",
        label="tab:claim_combined_deltas",
        colfmt="llrccccccccl",
    )
    _write_tex(
        cov_df,
        cov_tex,
        caption="Coverage and Sample Counts for Claim Tables",
        label="tab:coverage_counts",
        colfmt="llr",
    )

    include_path = out / "include_all_tables.tex"
    include_path.write_text(
        "\n\n".join(
            [
                combined_tex.read_text(encoding="utf-8").strip(),
                cov_tex.read_text(encoding="utf-8").strip(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {rank_csv}")
    print(f"Wrote {err_csv}")
    print(f"Wrote {combined_csv}")
    print(f"Wrote {cov_csv}")
    print(f"Wrote {rank_tex}")
    print(f"Wrote {err_tex}")
    print(f"Wrote {combined_tex}")
    print(f"Wrote {cov_tex}")
    print(f"Wrote {include_path}")


if __name__ == "__main__":
    main()
