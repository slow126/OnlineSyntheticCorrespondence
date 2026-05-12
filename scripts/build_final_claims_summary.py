#!/usr/bin/env python3
"""
Build a compact, claim-centric summary from current analysis outputs.

Inputs:
  - heldout_model_cv_* outputs
  - final_utility_sweep_* outputs
  - paper_tables_eccv_* table_1_hypothesis_validation.csv files

Outputs:
  - final_claims_summary.csv
  - final_claims_summary.md
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _f(x: object) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _fmt(x: object, digits: int = 3) -> str:
    v = _f(x)
    if not np.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def _pick_claim_rate(claims_df: pd.DataFrame, scope: str, claim: str) -> float:
    if claims_df.empty:
        return math.nan
    sub = claims_df[
        (claims_df["scope"].astype(str) == scope)
        & (claims_df["claim"].astype(str) == claim)
    ]
    if sub.empty:
        return math.nan
    return _f(sub.iloc[0].get("support_rate"))


def _pick_perm_sig_rate(
    perm_df: pd.DataFrame, scope: str, metric: str = "sig_rate_both"
) -> float:
    if perm_df.empty:
        return math.nan
    sub = perm_df[perm_df["scope"].astype(str) == scope]
    if sub.empty:
        return math.nan
    return _f(sub.iloc[0].get(metric))


def _paper_support_fraction(table_paths: List[str], hypothesis: str) -> Tuple[float, int, int]:
    n_support = 0
    n_total = 0
    for p in table_paths:
        df = _read_csv(Path(p))
        if df.empty or "hypothesis" not in df.columns or "verdict" not in df.columns:
            continue
        sub = df[df["hypothesis"].astype(str) == hypothesis].copy()
        if sub.empty:
            continue
        n_total += len(sub)
        n_support += int(sub["verdict"].astype(str).str.contains("Supports", na=False).sum())
    if n_total == 0:
        return math.nan, 0, 0
    return float(n_support / n_total), n_support, n_total


def _hybrid_best_rate(best_df: pd.DataFrame, protocol: str, heads: List[str]) -> float:
    if best_df.empty:
        return math.nan
    req = {"protocol", "head", "lane"}
    if not req.issubset(set(best_df.columns)):
        return math.nan
    sub = best_df[
        (best_df["protocol"].astype(str) == protocol)
        & (best_df["head"].astype(str).isin([str(h) for h in heads]))
    ]
    if sub.empty:
        return math.nan
    ok = (sub["lane"].astype(str) == "hybrid").astype(float)
    if ok.empty:
        return math.nan
    return float(ok.mean())


def _utility_synthetic_gap(subgroup_df: pd.DataFrame) -> float:
    if subgroup_df.empty:
        return math.nan
    req = {"lane", "k_target", "subgroup", "raw_mae"}
    if not req.issubset(set(subgroup_df.columns)):
        return math.nan
    out: List[float] = []
    for (lane, k), sub in subgroup_df.groupby(["lane", "k_target"], dropna=False):
        syn = sub[sub["subgroup"].astype(str) == "synthetic_only"]
        non = sub[sub["subgroup"].astype(str) == "nonsynthetic_only"]
        if syn.empty or non.empty:
            continue
        out.append(_f(non.iloc[0]["raw_mae"]) - _f(syn.iloc[0]["raw_mae"]))
    if not out:
        return math.nan
    arr = np.array([x for x in out if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return math.nan
    return float(arr.mean())


def _utility_rank_cindex_mean(selected_df: pd.DataFrame) -> float:
    if selected_df.empty or "mean_jointood_rank_pairwise_cindex" not in selected_df.columns:
        return math.nan
    arr = pd.to_numeric(selected_df["mean_jointood_rank_pairwise_cindex"], errors="coerce")
    if arr.notna().sum() == 0:
        return math.nan
    return float(arr.mean())


def _utility_calibration_worse_frac(selected_df: pd.DataFrame) -> float:
    if selected_df.empty or "strict_oof_delta_mae" not in selected_df.columns:
        return math.nan
    arr = pd.to_numeric(selected_df["strict_oof_delta_mae"], errors="coerce")
    arr = arr[arr.notna()]
    if arr.empty:
        return math.nan
    return float((arr < 0.0).mean())


def _utility_sanity_success_frac(sanity_df: pd.DataFrame) -> float:
    if sanity_df.empty:
        return math.nan
    req = {"p_value_mae_lower_than_perm", "p_value_spearman_higher_than_perm"}
    if not req.issubset(set(sanity_df.columns)):
        return math.nan
    p_mae = pd.to_numeric(sanity_df["p_value_mae_lower_than_perm"], errors="coerce")
    p_sp = pd.to_numeric(sanity_df["p_value_spearman_higher_than_perm"], errors="coerce")
    ok = (p_mae <= 0.05) & (p_sp <= 0.05)
    ok = ok[p_mae.notna() & p_sp.notna()]
    if ok.empty:
        return math.nan
    return float(ok.mean())


def _status_from_rate(rate: float, strong: float = 0.8, mixed: float = 0.6) -> str:
    if not np.isfinite(rate):
        return "insufficient_data"
    if rate >= strong:
        return "supported"
    if rate >= mixed:
        return "mixed"
    return "not_supported"


def _status_portability(primary_sig: float, stress_sig: float) -> str:
    if not np.isfinite(primary_sig) or not np.isfinite(stress_sig):
        return "insufficient_data"
    if primary_sig >= 0.25 and stress_sig <= 0.05:
        return "supported_with_limitation"
    if primary_sig >= 0.15:
        return "mixed"
    return "not_supported"


def _status_portability_ranking_split(
    primary_rank_sig: float,
    stress_rank_sig: float,
    primary_both_sig: float,
    stress_both_sig: float,
) -> str:
    """
    Ranking-first portability status.

    Split cutoffs acknowledge protocol difficulty:
      - primary (model_train_benchmark): stronger ranking threshold
      - stress (model_only): weaker ranking threshold
    """
    if not np.isfinite(primary_rank_sig) or not np.isfinite(stress_rank_sig):
        return "insufficient_data"

    if primary_rank_sig >= 0.60 and stress_rank_sig >= 0.20:
        if np.isfinite(primary_both_sig) and np.isfinite(stress_both_sig):
            if primary_both_sig >= 0.25 and stress_both_sig >= 0.10:
                return "supported"
            return "supported_with_limitation"
        return "supported_with_limitation"

    if primary_rank_sig >= 0.45 and stress_rank_sig >= 0.15:
        return "mixed"

    if primary_rank_sig >= 0.35:
        return "supported_with_limitation"

    return "not_supported"


def _status_dual_support(primary_rate: float, stress_rate: float) -> str:
    if not np.isfinite(primary_rate) or not np.isfinite(stress_rate):
        return "insufficient_data"
    if primary_rate >= 0.8 and stress_rate >= 0.8:
        return "supported"
    if primary_rate >= 0.6 and stress_rate >= 0.6:
        return "mixed"
    return "not_supported"


def _status_asym_vs_mmd(primary_rate: float, paper_h1_rate: float) -> str:
    if not np.isfinite(primary_rate) and not np.isfinite(paper_h1_rate):
        return "insufficient_data"
    if np.isfinite(primary_rate) and np.isfinite(paper_h1_rate):
        if primary_rate >= 0.75 and paper_h1_rate >= 0.75:
            return "supported"
        if primary_rate >= 0.6 and paper_h1_rate >= 0.7:
            return "mixed"
        return "not_supported"
    if np.isfinite(primary_rate):
        return _status_from_rate(primary_rate, strong=0.75, mixed=0.6)
    return _status_from_rate(paper_h1_rate, strong=0.75, mixed=0.7)


def _status_asym_typical(primary_rate: float, paper_h1_rate: float) -> str:
    if not np.isfinite(primary_rate) and not np.isfinite(paper_h1_rate):
        return "insufficient_data"
    if np.isfinite(primary_rate) and np.isfinite(paper_h1_rate):
        if primary_rate >= 0.65 and paper_h1_rate >= 0.80:
            return "supported_with_limitation"
        if primary_rate >= 0.60 and paper_h1_rate >= 0.70:
            return "mixed"
        return "not_supported"
    if np.isfinite(primary_rate):
        if primary_rate >= 0.65:
            return "supported_with_limitation"
        if primary_rate >= 0.60:
            return "mixed"
        return "not_supported"
    if paper_h1_rate >= 0.80:
        return "supported_with_limitation"
    if paper_h1_rate >= 0.70:
        return "mixed"
    return "not_supported"


def _status_synthetic_gap(gap: float) -> str:
    if not np.isfinite(gap):
        return "insufficient_data"
    if gap >= 2.0:
        return "supported"
    if gap >= 1.0:
        return "mixed"
    return "not_supported"


def _status_hybrid_not_universal(hybrid_vs_motion_rate: float) -> str:
    if not np.isfinite(hybrid_vs_motion_rate):
        return "insufficient_data"
    if hybrid_vs_motion_rate < 0.5:
        return "supported"
    if hybrid_vs_motion_rate < 0.7:
        return "mixed"
    return "not_supported"


def _build_claim_rows(
    heldout_claims: pd.DataFrame,
    heldout_perm_claims: pd.DataFrame,
    heldout_best: pd.DataFrame,
    utility_selected: pd.DataFrame,
    utility_subgroup: pd.DataFrame,
    utility_sanity: pd.DataFrame,
    paper_tables: List[str],
    primary_scope: str,
    stress_scope: str,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    perm_primary_both = _pick_perm_sig_rate(
        heldout_perm_claims, primary_scope, "sig_rate_both"
    )
    perm_stress_both = _pick_perm_sig_rate(
        heldout_perm_claims, stress_scope, "sig_rate_both"
    )
    perm_primary_rank = _pick_perm_sig_rate(
        heldout_perm_claims, primary_scope, "sig_rate_spearman"
    )
    perm_stress_rank = _pick_perm_sig_rate(
        heldout_perm_claims, stress_scope, "sig_rate_spearman"
    )

    motion_primary = _pick_claim_rate(heldout_claims, primary_scope, "motion_beats_appearance")
    motion_stress = _pick_claim_rate(heldout_claims, stress_scope, "motion_beats_appearance")
    rows.append(
        {
            "claim_id": "C1",
            "claim": "Motion features are a stronger and more consistent transfer signal than appearance.",
            "paper_takeaway": "Motion-based predictors consistently beat appearance-based predictors across held-out evaluations.",
            "status": _status_dual_support(motion_primary, motion_stress),
            "primary_scope": primary_scope,
            "primary_metric": "heldout.motion_beats_appearance.support_rate",
            "primary_value": motion_primary,
            "primary_rule": "higher is better; >=0.80 strong",
            "secondary_scope": "",
            "secondary_metric": "",
            "secondary_value": math.nan,
            "secondary_rule": "",
            "stress_scope": stress_scope,
            "stress_metric": "heldout.motion_beats_appearance.support_rate",
            "stress_value": motion_stress,
            "stress_rule": "higher is better; >=0.80 strong",
            "decision_rule": "supported if primary>=0.80 and stress>=0.80",
            "evidence_summary": (
                f"motion>appearance support_rate primary={_fmt(motion_primary)}, "
                f"stress={_fmt(motion_stress)}"
            ),
            "caveat": "Absolute extrapolation to unseen model families can still be weak.",
        }
    )

    asym_primary = _pick_claim_rate(heldout_claims, primary_scope, "asym_beats_mmd")
    asym_stress = _pick_claim_rate(heldout_claims, stress_scope, "asym_beats_mmd")
    h1_frac, h1_sup, h1_tot = _paper_support_fraction(paper_tables, "H1")
    rows.append(
        {
            "claim_id": "C2",
            "claim": "Directed/asymmetric distances are more informative than symmetric MMD in matched comparisons.",
            "paper_takeaway": "Directed/asymmetric distances are usually better than MMD, but not in every matched case.",
            "status": _status_asym_typical(asym_primary, h1_frac),
            "primary_scope": primary_scope,
            "primary_metric": "heldout.asym_beats_mmd.support_rate",
            "primary_value": asym_primary,
            "primary_rule": "higher is better; >=0.65 indicates typical advantage",
            "secondary_scope": "paper_tables_jointood",
            "secondary_metric": "table_1.H1.support_fraction",
            "secondary_value": h1_frac,
            "secondary_rule": "higher is better; >=0.80 strong external support",
            "stress_scope": stress_scope,
            "stress_metric": "heldout.asym_beats_mmd.support_rate",
            "stress_value": asym_stress,
            "stress_rule": "higher is better; context only",
            "decision_rule": "supported_with_limitation if heldout primary>=0.65 and paper H1>=0.80",
            "evidence_summary": (
                f"asym>mmd support_rate primary={_fmt(asym_primary)}, stress={_fmt(asym_stress)}; "
                f"paper H1 support={h1_sup}/{h1_tot} ({_fmt(h1_frac)})"
            ),
            "caveat": "Directional metrics are typically better, not universally better in every setting.",
        }
    )

    rows.append(
        {
            "claim_id": "C3",
            "claim": "Ranking portability is useful in harder held-out cells, but weak on pure unseen-model extrapolation.",
            "paper_takeaway": "Ranking signal transfers in harder held-out cells; pure unseen-model extrapolation is weaker.",
            "status": _status_portability_ranking_split(
                perm_primary_rank,
                perm_stress_rank,
                perm_primary_both,
                perm_stress_both,
            ),
            "primary_scope": primary_scope,
            "primary_metric": "permutation.sig_rate_spearman",
            "primary_value": perm_primary_rank,
            "primary_rule": "ranking-first: primary cutoff >=0.60",
            "secondary_scope": primary_scope,
            "secondary_metric": "permutation.sig_rate_both",
            "secondary_value": perm_primary_both,
            "secondary_rule": "context only for absolute+rank joint signal",
            "stress_scope": stress_scope,
            "stress_metric": "permutation.sig_rate_spearman",
            "stress_value": perm_stress_rank,
            "stress_rule": "ranking-first: stress cutoff >=0.20",
            "decision_rule": (
                "ranking split cutoffs: primary(sig_rate_spearman)>=0.60, "
                "stress(sig_rate_spearman)>=0.20; use sig_rate_both as limitation context"
            ),
            "evidence_summary": (
                f"perm rank-signal primary={_fmt(perm_primary_rank)}, stress={_fmt(perm_stress_rank)}; "
                f"perm both-metrics primary={_fmt(perm_primary_both)}, stress={_fmt(perm_stress_both)}"
            ),
            "caveat": "Use model-only as limitation/stress-test, not as sole portability headline.",
        }
    )

    hybrid_primary = _hybrid_best_rate(
        heldout_best, protocol="model_train_benchmark", heads=["ols", "ridge"]
    )
    hybrid_stress = _hybrid_best_rate(
        heldout_best, protocol="model_only", heads=["ols", "ridge"]
    )
    rows.append(
        {
            "claim_id": "C4",
            "claim": "Hybrid methods are frequently strongest on absolute MAE among selected candidates.",
            "paper_takeaway": "Hybrid candidates frequently achieve the best absolute MAE in held-out selection.",
            "status": _status_dual_support(hybrid_primary, hybrid_stress),
            "primary_scope": "model_train_benchmark_ols_ridge_best_by_head",
            "primary_metric": "heldout.hybrid_best_rate",
            "primary_value": hybrid_primary,
            "primary_rule": "higher is better; 1.0 means all best-by-head picks are hybrid",
            "secondary_scope": "",
            "secondary_metric": "",
            "secondary_value": math.nan,
            "secondary_rule": "",
            "stress_scope": "model_only_ols_ridge_best_by_head",
            "stress_metric": "heldout.hybrid_best_rate",
            "stress_value": hybrid_stress,
            "stress_rule": "higher is better; stress check on pure unseen-model protocol",
            "decision_rule": "supported if hybrid_best_rate >=0.80 in primary and stress scopes",
            "evidence_summary": (
                f"hybrid best-rate (ols/ridge) primary={_fmt(hybrid_primary)}, stress={_fmt(hybrid_stress)}"
            ),
            "caveat": "This is a best-candidate result; it does not imply hybrid always beats every matched comparator.",
        }
    )

    rank_cidx = _utility_rank_cindex_mean(utility_selected)
    sanity_frac = _utility_sanity_success_frac(utility_sanity)
    status = "insufficient_data"
    if np.isfinite(rank_cidx) and np.isfinite(sanity_frac):
        if rank_cidx >= 0.54 and sanity_frac >= 0.8:
            status = "supported_with_limitation"
        elif rank_cidx > 0.52 and sanity_frac >= 0.6:
            status = "supported_with_limitation"
        else:
            status = "not_supported"
    rows.append(
        {
            "claim_id": "C5",
            "claim": "Utility function provides practical ranking signal for dataset selection.",
            "paper_takeaway": "Utility is useful for ranking candidates, but absolute calibration should remain secondary.",
            "status": status,
            "primary_scope": "final_utility_selected_exact_k",
            "primary_metric": "selected.mean_jointood_rank_pairwise_cindex",
            "primary_value": rank_cidx,
            "primary_rule": "higher is better; >=0.54 indicates useful ranking in this study",
            "secondary_scope": "",
            "secondary_metric": "",
            "secondary_value": math.nan,
            "secondary_rule": "",
            "stress_scope": "final_utility_permutation_sanity",
            "stress_metric": "sanity.permutation_success_fraction_both_metrics",
            "stress_value": sanity_frac,
            "stress_rule": "higher is better; >=0.80 strong",
            "decision_rule": "supported_with_limitation if c-index>=0.54 and sanity>=0.80",
            "evidence_summary": (
                f"mean selected pairwise_cindex={_fmt(rank_cidx)}, sanity both-metric success={_fmt(sanity_frac)}"
            ),
            "caveat": "Treat as ranking utility first; absolute calibration remains secondary.",
        }
    )

    return rows


def _to_readable_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    out = pd.DataFrame(
        {
            "claim_id": rows["claim_id"],
            "status": rows["status"],
            "paper_takeaway": rows["paper_takeaway"],
            "main_test": rows["primary_scope"],
            "main_metric": rows["primary_metric"],
            "main_value": rows["primary_value"],
            "stress_test": rows["stress_scope"],
            "stress_metric": rows["stress_metric"],
            "stress_value": rows["stress_value"],
            "key_evidence": rows["evidence_summary"],
            "caveat": rows["caveat"],
        }
    )
    return out


def _write_markdown(
    path: Path,
    rows_readable: pd.DataFrame,
    audit_path: Path,
    sources: List[str],
    limitations: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Final Claims Summary")
    lines.append("")
    lines.append("Status levels: `supported`, `supported_with_limitation`, `mixed`, `not_supported`, `insufficient_data`.")
    lines.append("")
    lines.append("## Quick Read")
    lines.append("- `main_test`: main validation split used for the claim.")
    lines.append("- `main_metric`: metric used for the status decision.")
    lines.append("- `stress_test`: harder/secondary split used to check robustness.")
    lines.append("")
    lines.append("## Claims")
    if rows_readable.empty:
        lines.append("- No claim rows generated.")
    else:
        cols = [
            "claim_id",
            "status",
            "paper_takeaway",
            "main_test",
            "main_metric",
            "main_value",
            "stress_test",
            "stress_metric",
            "stress_value",
            "key_evidence",
            "caveat",
        ]
        lines.append(rows_readable[cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Audit File")
    lines.append(f"- Full decision table with rules/secondary metrics: `{audit_path}`")
    if limitations:
        lines.append("")
        lines.append("## Non-Headline Limitations")
        for item in limitations:
            lines.append(f"- {item}")
    lines.append("")
    lines.append("## Sources")
    for s in sources:
        lines.append(f"- `{s}`")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final claim-centric summary table.")
    parser.add_argument(
        "--heldout-dir",
        default="analysis_comprehensive_runs/heldout_model_cv_v6_full_perm200_bench",
        help="Held-out CV output directory.",
    )
    parser.add_argument(
        "--utility-dir",
        default="analysis_comprehensive_runs/final_utility_sweep_v2_rankmetrics",
        help="Final utility sweep output directory.",
    )
    parser.add_argument(
        "--paper-table-glob",
        default="analysis_comprehensive_runs/hof_motion_v3_density_jointood_full_*_v1/paper_tables_eccv_*_jointood_mae/table_1_hypothesis_validation.csv",
        help="Glob for ECCV hypothesis tables.",
    )
    parser.add_argument(
        "--primary-scope",
        default="model_train_benchmark_ols_ridge",
        help="Primary held-out scope used for claim evidence.",
    )
    parser.add_argument(
        "--stress-scope",
        default="model_only_ols_ridge",
        help="Stress-test held-out scope used for claim evidence.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_comprehensive_runs/final_claims_v1",
        help="Output directory for final claims summary files.",
    )
    args = parser.parse_args()

    heldout_dir = Path(args.heldout_dir)
    utility_dir = Path(args.utility_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heldout_claims = _read_csv(heldout_dir / "heldout_model_cv_claims_summary.csv")
    heldout_perm_claims = _read_csv(heldout_dir / "heldout_model_cv_permutation_claims_summary.csv")
    heldout_best = _read_csv(heldout_dir / "heldout_model_cv_best_by_protocol_head.csv")
    utility_selected = _read_csv(utility_dir / "selected_exact_k_with_calibrated_diagnostics.csv")
    utility_subgroup = _read_csv(utility_dir / "subgroup_robustness_selected_exact_k.csv")
    utility_sanity = _read_csv(utility_dir / "sanity_checks_permutation_selected_exact_k.csv")
    paper_tables = sorted(glob.glob(str(args.paper_table_glob)))

    rows = _build_claim_rows(
        heldout_claims=heldout_claims,
        heldout_perm_claims=heldout_perm_claims,
        heldout_best=heldout_best,
        utility_selected=utility_selected,
        utility_subgroup=utility_subgroup,
        utility_sanity=utility_sanity,
        paper_tables=paper_tables,
        primary_scope=str(args.primary_scope),
        stress_scope=str(args.stress_scope),
    )
    out_df = pd.DataFrame(rows)
    out_readable = _to_readable_table(out_df)
    out_readable.to_csv(out_dir / "final_claims_summary.csv", index=False)
    out_df.to_csv(out_dir / "final_claims_summary_audit.csv", index=False)

    sources = [
        str(heldout_dir / "heldout_model_cv_claims_summary.csv"),
        str(heldout_dir / "heldout_model_cv_permutation_claims_summary.csv"),
        str(utility_dir / "selected_exact_k_with_calibrated_diagnostics.csv"),
        str(utility_dir / "subgroup_robustness_selected_exact_k.csv"),
        str(utility_dir / "sanity_checks_permutation_selected_exact_k.csv"),
        f"glob:{args.paper_table_glob}",
    ]
    synth_gap = _utility_synthetic_gap(utility_subgroup)
    model_only_rank = _pick_perm_sig_rate(
        heldout_perm_claims, str(args.stress_scope), "sig_rate_spearman"
    )
    limitations: List[str] = []
    if np.isfinite(synth_gap):
        limitations.append(
            f"Synthetic/non-synthetic subgroup gap exists in this benchmark mix (mean raw MAE gap={_fmt(synth_gap)})."
        )
    if np.isfinite(model_only_rank):
        limitations.append(
            f"Pure unseen-model extrapolation is weaker than model+train+benchmark holdout (model-only rank-signal={_fmt(model_only_rank)})."
        )

    _write_markdown(
        out_dir / "final_claims_summary.md",
        out_readable,
        out_dir / "final_claims_summary_audit.csv",
        sources,
        limitations,
    )
    print(f"Wrote final claims summary to: {out_dir}")


if __name__ == "__main__":
    main()
