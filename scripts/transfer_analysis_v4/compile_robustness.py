"""Compile the leave-one-generator-family-out + cluster-bootstrap robustness runs
into one comparison table.

Reads (all optional — missing files are skipped):
    results/summary.csv                       full-data, source-level CIs
    results/bootstrap_gap.csv                 full-data motion-appearance gap
    results/summary_cluster.csv               full-data, FAMILY-level CIs (Tier 3)
    results/bootstrap_gap_cluster.csv         full-data gap, family-level CIs
    results_robust_drop_<fam>/summary.csv     each leave-one-family-out refit (Tier 2)
    results_robust_drop_<fam>/bootstrap_gap.csv

Headline row = (split=LOTO, family=motion, label=main, head=g): the within-context
Spearman of the motion predictor (ctx_rho_g) and the motion-appearance gap. The
question each robustness row answers: does the headline survive this perturbation?

Usage:
    python scripts/transfer_analysis_v4/compile_robustness.py \
        --base-out scripts/transfer_analysis_v4/results_robust \
        --families "sdf3d warp2d kubric realflow semantic" \
        --targets "peak_pck"
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("scripts/transfer_analysis_v4/results")


def _headline(summary_csv: Path, gap_csv: Path, target: str, split: str = "LOTO"):
    """Pull (motion ctx_rho_g [CI], appearance ctx_rho_g, gap [CI] P>0, n_rows)."""
    out = {}
    if summary_csv.exists():
        s = pd.read_csv(summary_csv)
        s = s[(s.target == target) & (s.split == split) & (s.label == "main")
              & (s["head"] == "g")]
        for fam, key in [("motion", "m"), ("appearance", "a")]:
            r = s[s.family == fam]
            if len(r):
                r = r.iloc[0]
                out[f"{key}_rho"] = r.get("ctx_rho_g", np.nan)
                out[f"{key}_lo"] = r.get("ctx_rho_g_lo", np.nan)
                out[f"{key}_hi"] = r.get("ctx_rho_g_hi", np.nan)
                out["n_rows"] = r.get("n_rows", np.nan)
    if gap_csv.exists():
        g = pd.read_csv(gap_csv)
        g = g[(g.target == target) & (g.split == split) & (g["head"] == "g")]
        if len(g):
            g = g.iloc[0]
            out["gap"] = g.get("ctx_rho_g_gap", np.nan)
            out["gap_lo"] = g.get("ctx_rho_g_gap_lo", np.nan)
            out["gap_hi"] = g.get("ctx_rho_g_gap_hi", np.nan)
            out["gap_p_gt_0"] = g.get("ctx_rho_g_gap_p_gt_0", np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-out", default="scripts/transfer_analysis_v4/results_robust")
    ap.add_argument("--families", default="sdf3d warp2d kubric realflow semantic")
    ap.add_argument("--targets", default="peak_pck")
    ap.add_argument("--split", default="LOTO")
    args = ap.parse_args()

    base = Path(args.base_out)
    base.mkdir(parents=True, exist_ok=True)
    fams = args.families.split()
    targets = args.targets.split()

    rows = []
    for target in targets:
        # full-data, source-level CIs (the baseline everything is compared to)
        full = _headline(RESULTS / "summary.csv", RESULTS / "bootstrap_gap.csv",
                         target, args.split)
        full.update(target=target, perturbation="full (11 sources)", drop=None)
        rows.append(full)

        # full-data, FAMILY-level CIs (Tier 3 cluster bootstrap)
        clus = _headline(RESULTS / "summary_cluster.csv",
                         RESULTS / "bootstrap_gap_cluster.csv", target, args.split)
        clus.update(target=target, perturbation="full (cluster CI, ~5 families)",
                    drop=None)
        rows.append(clus)

        # Tier 2: each leave-one-generator-family-out refit
        for fam in fams:
            d = Path(f"{args.base_out}_drop_{fam}")
            h = _headline(d / "summary.csv", d / "bootstrap_gap.csv",
                          target, args.split)
            h.update(target=target, perturbation=f"drop family: {fam}", drop=fam)
            rows.append(h)

    df = pd.DataFrame(rows)
    csv = base / "ROBUSTNESS_SUMMARY.csv"
    df.to_csv(csv, index=False)

    # Readable markdown
    def fmt(v, p=3):
        return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.{p}f}"

    lines = ["# Robustness Summary",
             "",
             "Headline = motion within-context Spearman (`ctx_rho_g`) on the LOTO "
             "split, and the motion−appearance gap. The result is robust if motion "
             "ρ stays clearly positive and the gap stays > 0 under every perturbation.",
             ""]
    for target in targets:
        sub = df[df.target == target]
        lines += [f"## target = {target}", "",
                  "| perturbation | motion ρ [95% CI] | appear. ρ | motion−appear gap [CI] | P(gap>0) | n_rows |",
                  "|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            m = f"{fmt(r.get('m_rho'))} [{fmt(r.get('m_lo'))}, {fmt(r.get('m_hi'))}]"
            a = fmt(r.get("a_rho"))
            gp = f"{fmt(r.get('gap'))} [{fmt(r.get('gap_lo'))}, {fmt(r.get('gap_hi'))}]"
            pg = "—" if pd.isna(r.get("gap_p_gt_0")) else f"{r.get('gap_p_gt_0'):.3f}"
            nr = "—" if pd.isna(r.get("n_rows")) else f"{int(r.get('n_rows'))}"
            lines.append(f"| {r['perturbation']} | {m} | {a} | {gp} | {pg} | {nr} |")
        lines.append("")
    md = base / "ROBUSTNESS_SUMMARY.md"
    md.write_text("\n".join(lines))

    print(df.to_string(index=False))
    print(f"\nwrote {csv}\nwrote {md}")


if __name__ == "__main__":
    main()
