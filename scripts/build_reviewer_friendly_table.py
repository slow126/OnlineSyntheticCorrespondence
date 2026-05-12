#!/usr/bin/env python3
"""
Build a reviewer-friendly main results table from existing aggregated outputs.

Reads only precomputed tables (no new experiments).
Writes:
  - reviewer_friendly_main_table.csv
  - reviewer_friendly_main_table.tex
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or not math.isfinite(v):
        return "--"
    return f"{v:.{nd}f}"


def _fmt_pct(v: float, nd: int = 2) -> str:
    if v is None or not math.isfinite(v):
        return "--"
    return f"{100.0 * v:.{nd}f}%"


def _verdict(delta_c: float, delta_s: float, pos_c: float | None = None, pos_s: float | None = None) -> str:
    if math.isfinite(delta_c) and math.isfinite(delta_s):
        if delta_c > 0 and delta_s > 0:
            if (pos_c is None or not math.isfinite(pos_c) or pos_c >= 0.8) and (
                pos_s is None or not math.isfinite(pos_s) or pos_s >= 0.8
            ):
                return "Supports claim"
            return "Supports claim (moderate)"
        if abs(delta_c) < 0.005 and abs(delta_s) < 0.01:
            return "Neutral / mixed"
        return "Counter / weak support"
    return "Insufficient"


def _build_rows(predictor_dir: Path, organized_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    # A) Directional vs symmetric (global mean contrast)
    ms = predictor_dir / "method_summary_with_derived_groups.csv"
    if ms.exists():
        df = pd.read_csv(ms)
        if "symmetry" in df.columns:
            asym = df[df["symmetry"] == "asym"]
            sym = df[df["symmetry"] == "sym"]
            c_asym = _num(asym["loto_rank_pairwise_cindex"]).dropna()
            c_sym = _num(sym["loto_rank_pairwise_cindex"]).dropna()
            s_asym = _num(asym["loto_rank_spearman"]).dropna()
            s_sym = _num(sym["loto_rank_spearman"]).dropna()
            if len(c_asym) and len(c_sym) and len(s_asym) and len(s_sym):
                dc = float(c_asym.mean() - c_sym.mean())
                ds = float(s_asym.mean() - s_sym.mean())
                rows.append(
                    {
                        "test": "Directional vs symmetric",
                        "comparison": "Asymmetric predictors vs symmetric predictors",
                        "n": f"{len(c_asym)} vs {len(c_sym)}",
                        "delta_cindex": dc,
                        "delta_rank_spearman": ds,
                        "pos_frac_cindex": float("nan"),
                        "pos_frac_rank_spearman": float("nan"),
                        "verdict": _verdict(dc, ds),
                    }
                )

    # B) Pure modality matched-k deltas
    pm = predictor_dir / "pure_modalities_matched_k_deltas.csv"
    if pm.exists():
        df = pd.read_csv(pm)
        pair_specs = [
            ("flow", "appearance", "Pure modality, matched-k"),
            ("hof", "appearance", "Pure modality, matched-k"),
            ("flow", "hof", "Pure modality, matched-k"),
        ]
        for left, right, test_name in pair_specs:
            ccol = f"loto_rank_pairwise_cindex__{left}_minus_{right}_oriented"
            scol = f"loto_rank_spearman__{left}_minus_{right}_oriented"
            if ccol not in df.columns or scol not in df.columns:
                continue
            c = _num(df[ccol]).dropna()
            s = _num(df[scol]).dropna()
            if c.empty or s.empty:
                continue
            pc = float((c > 0).mean())
            ps = float((s > 0).mean())
            dc = float(c.mean())
            ds = float(s.mean())
            rows.append(
                {
                    "test": test_name,
                    "comparison": f"{left.upper()} vs {right.upper()}",
                    "n": str(int(min(len(c), len(s)))),
                    "delta_cindex": dc,
                    "delta_rank_spearman": ds,
                    "pos_frac_cindex": pc,
                    "pos_frac_rank_spearman": ps,
                    "verdict": _verdict(dc, ds, pc, ps),
                }
            )

    # C) Incremental one-step additions (graph edges)
    inc = predictor_dir / "incremental_domain_addition_summary.csv"
    if inc.exists():
        df = pd.read_csv(inc)
        for domain in ["appearance", "hof", "flow"]:
            g = df[df["added_domain"] == domain]
            if g.empty:
                continue
            r = g.iloc[0]
            dc = float(r.get("loto_rank_pairwise_cindex__oriented_delta_mean", float("nan")))
            ds = float(r.get("loto_rank_spearman__oriented_delta_mean", float("nan")))
            pc = float(r.get("loto_rank_pairwise_cindex__oriented_delta_pos_frac", float("nan")))
            ps = float(r.get("loto_rank_spearman__oriented_delta_pos_frac", float("nan")))
            n_edges = int(r.get("n_edges", 0)) if pd.notna(r.get("n_edges", float("nan"))) else 0
            rows.append(
                {
                    "test": "Incremental one-step addition",
                    "comparison": f"+{domain.upper()} from matched composition neighbors",
                    "n": str(n_edges),
                    "delta_cindex": dc,
                    "delta_rank_spearman": ds,
                    "pos_frac_cindex": pc,
                    "pos_frac_rank_spearman": ps,
                    "verdict": "Insufficient (sparse N)" if n_edges < 3 else _verdict(dc, ds, pc, ps),
                }
            )

    # D) Coverage row from parameter-matched organization
    org = organized_dir / "organized_parameter_matched_rows.csv"
    if org.exists():
        df = pd.read_csv(org)
        if "backbone_kind" in df.columns and "backbone_id" in df.columns:
            flow_n = int(df[df["backbone_kind"] == "flow_backbone"]["backbone_id"].nunique())
            app_n = int(df[df["backbone_kind"] == "appearance_backbone"]["backbone_id"].nunique())
            mmd_n = int(df[df["backbone_kind"] == "mmd_backbone"]["backbone_id"].nunique())
            rows.append(
                {
                    "test": "Parameter-matched coverage",
                    "comparison": f"Backbones covered: flow={flow_n}, appearance={app_n}, mmd={mmd_n}",
                    "n": str(int(len(df))),
                    "delta_cindex": float("nan"),
                    "delta_rank_spearman": float("nan"),
                    "pos_frac_cindex": float("nan"),
                    "pos_frac_rank_spearman": float("nan"),
                    "verdict": "Context row",
                }
            )

    return rows


def _write_latex(df: pd.DataFrame, out_tex: Path) -> None:
    pretty = df.copy()
    pretty["Delta C-index"] = pretty["delta_cindex"].map(_fmt)
    pretty["Delta Rank Spearman"] = pretty["delta_rank_spearman"].map(_fmt)
    pretty["PosFrac C-index"] = pretty["pos_frac_cindex"].map(_fmt_pct)
    pretty["PosFrac Rank Spearman"] = pretty["pos_frac_rank_spearman"].map(_fmt_pct)
    pretty = pretty[
        [
            "test",
            "comparison",
            "n",
            "Delta C-index",
            "Delta Rank Spearman",
            "PosFrac C-index",
            "PosFrac Rank Spearman",
            "verdict",
        ]
    ].rename(
        columns={
            "test": "Test",
            "comparison": "Comparison",
            "n": "N",
            "verdict": "Verdict",
        }
    )

    tab = pretty.to_latex(index=False, escape=True, column_format="llrccccl")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Reviewer-Friendly Summary of Main Results}",
        r"\label{tab:reviewer_friendly_main}",
        r"\small",
        r"\begin{minipage}{0.98\linewidth}",
        r"\textit{All deltas are oriented so positive is better. PosFrac is the fraction of valid comparisons with positive oriented delta.}",
        r"\end{minipage}",
        r"\vspace{0.3em}",
        tab.strip(),
        r"\end{table}",
        "",
    ]
    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reviewer-friendly camera-ready results table")
    parser.add_argument("--predictor-group-dir", required=True)
    parser.add_argument("--organized-grid-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    predictor_dir = Path(args.predictor_group_dir)
    organized_dir = Path(args.organized_grid_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_rows(predictor_dir, organized_dir)
    if not rows:
        raise SystemExit("No rows generated; check inputs.")
    df = pd.DataFrame(rows)
    csv_path = out_dir / "reviewer_friendly_main_table.csv"
    tex_path = out_dir / "reviewer_friendly_main_table.tex"
    df.to_csv(csv_path, index=False)
    _write_latex(df, tex_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
