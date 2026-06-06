"""Compile ABLATION_density.md combining:

  1. Feature-side stability (from analysis_v3/density_invariance_pair_sharded/
     stability_*.csv) — at what N does each pairwise-self-distance feature
     match its baseline (ρ >= threshold) value?
  2. Fitted-side stability — at what N does each family's ctx_rho_g stop
     moving (across the 5 matched density levels we ran)?

Run after the lean density sweep finishes. Looks for:
  results_lean_canon_mixed/summary_points.csv  (or summary.csv)
  results_lean_dL{1..5}_mixed/summary_points.csv

Writes:
  scripts/transfer_analysis_v4/ABLATION_density.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DENS_LEVELS = [
    ("dL1", 50000,    25000),
    ("dL2", 200000,   100000),
    ("dL3", 1000000,  500000),
    ("dL4", 4000000,  2000000),
    ("dL5", 8000000,  4000000),
]


def load_feature_stability(stab_dir: Path, threshold: float = 0.90) -> dict:
    """Read stability_{space}_train_eval__eval_eval.csv files."""
    out = {}
    for space in ["flow", "dino"]:
        p = stab_dir / f"stability_{space}_train_eval__eval_eval.csv"
        if not p.exists():
            print(f"  missing: {p}")
            continue
        df = pd.read_csv(p)
        out[space] = df
    return out


def min_n_for_stable(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """For each (pair_type, metric), find min level where rho >= threshold AND
    stays above for all higher levels. Returns one row per (pair_type, metric).
    """
    rows = []
    for (pair_type, metric), sub in df.groupby(["pair_type", "metric"]):
        sub_sorted = sub.sort_values("level").reset_index(drop=True)
        # Find first level whose rho >= threshold AND all subsequent >= threshold.
        min_n = None
        for i in range(len(sub_sorted)):
            tail = sub_sorted.iloc[i:]
            if (tail["rho"] >= threshold).all():
                min_n = int(sub_sorted.iloc[i]["level"])
                break
        rows.append(dict(pair_type=pair_type, metric=metric, min_n=min_n,
                         worst_rho=float(sub_sorted["rho"].min())))
    return pd.DataFrame(rows)


def load_fitted_summary(root: Path) -> pd.DataFrame:
    """Read summary_points.csv from canon + 5 dL dirs."""
    frames = []
    pairs = [("canon", root / "results_lean_canon_mixed")]
    for code, fn, dn in DENS_LEVELS:
        pairs.append((code, root / f"results_lean_{code}_mixed"))
    for code, d in pairs:
        p = d / "summary_points.csv"
        if not p.exists():
            p = d / "summary.csv"
        if not p.exists():
            print(f"  missing: {d}")
            continue
        df = pd.read_csv(p)
        df["__level"] = code
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fitted_stability_table(fitted: pd.DataFrame, target: str,
                           tol: float = 0.05) -> pd.DataFrame:
    """For each (family, split), report ρ_g across canon, dL1..dL5 and
    flag the smallest dL where ρ_g is within `tol` of the canonical value
    AND remains within tol for all higher dLs.
    """
    if fitted.empty:
        return pd.DataFrame()
    sub = fitted[(fitted["target"] == target) & (fitted["label"] == "main")
                 & fitted["ctx_rho_g"].notna()]
    order = ["canon", "dL1", "dL2", "dL3", "dL4", "dL5"]
    rows = []
    for (fam, split), g in sub.groupby(["family", "split"]):
        g = g.set_index("__level").reindex(order)
        vals = {lvl: g.loc[lvl, "ctx_rho_g"] if lvl in g.index else np.nan
                for lvl in order}
        canon_v = vals["canon"]
        # Find smallest dL whose |ρ_g - canon| <= tol AND stays within tol thereafter.
        min_dL = None
        dL_order = ["dL1", "dL2", "dL3", "dL4", "dL5"]
        for i, lvl in enumerate(dL_order):
            tail = dL_order[i:]
            diffs = [abs(vals[t] - canon_v) for t in tail
                     if pd.notna(vals[t]) and pd.notna(canon_v)]
            if diffs and max(diffs) <= tol:
                min_dL = lvl
                break
        row = dict(family=fam, split=split, **vals,
                   min_stable_dL=min_dL,
                   span=float(max(vals.values()) - min(vals.values()))
                       if all(pd.notna(v) for v in vals.values()) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def render(feat_stab: dict, feat_min: dict, fitted_tab: dict,
           thresh: float, tol: float) -> str:
    out = []
    out.append("# Transfer Analysis v4 — Density Ablation\n")
    out.append("Two stability axes:\n")
    out.append("- **Feature-side**: for each pairwise self-distance metric, "
               "at what N does the metric's value match its asymptote "
               f"(Spearman ρ ≥ {thresh})? Reads from the existing "
               "`analysis_v3/density_invariance_pair_sharded/stability_*.csv`.")
    out.append("- **Fitted-side**: for each family, at what N does the "
               f"within-context ridge ρ_g stop moving (|Δρ_g| ≤ {tol} from "
               "canonical density, monotone)? Reads from the lean density "
               "sweep results.\n")

    out.append("Density diagonal levels:\n")
    out.append("| code | flow N | dino N |")
    out.append("|---|---|---|")
    for code, fn, dn in DENS_LEVELS:
        out.append(f"| {code} | {fn:,} | {dn:,} |")

    out.append("\n---\n\n## 1. Feature-side stability (Spearman ρ vs baseline)\n")
    for space, df in feat_stab.items():
        out.append(f"\n### {space.upper()}\n")
        # pivot: rows=metric, cols=level, values=rho — limit to train_eval pair_type
        for pt in sorted(df["pair_type"].unique()):
            sub = df[df["pair_type"] == pt]
            pv = sub.pivot_table(index="metric", columns="level",
                                 values="rho").sort_index(axis=1)
            out.append(f"\n_pair_type = {pt}_\n")
            header = ["metric"] + [f"N={int(c):,}" for c in pv.columns]
            out.append("| " + " | ".join(header) + " |")
            out.append("|" + "---|" * len(header))
            for m in pv.index:
                row = [m]
                for c in pv.columns:
                    v = pv.loc[m, c]
                    row.append(f"{v:.3f}" if pd.notna(v) else "—")
                out.append("| " + " | ".join(row) + " |")
        # Minimum-N table for this space
        mn = feat_min.get(space)
        if mn is not None:
            out.append(f"\n_Minimum N for ρ ≥ {thresh} (per pair_type × metric):_\n")
            out.append("| pair_type | metric | min N | worst ρ |")
            out.append("|---|---|---|---|")
            for _, r in mn.iterrows():
                mn_str = f"{r['min_n']:,}" if pd.notna(r['min_n']) else "*never reaches threshold*"
                out.append(f"| {r['pair_type']} | {r['metric']} | "
                           f"{mn_str} | {r['worst_rho']:.3f} |")

    out.append("\n---\n\n## 2. Fitted-side stability (ctx_rho_g across densities)\n")
    out.append(f"\n_For each family × split: ρ_g at canonical density and at "
               f"each of the 5 matched diagonal levels. min_stable_dL = "
               f"smallest dL whose |ρ_g − canon| ≤ {tol} and stays ≤ {tol} for "
               "all higher dLs._\n")
    for tgt, tab in fitted_tab.items():
        out.append(f"\n### Target: `{tgt}`\n")
        if tab.empty:
            out.append("_no rows_\n")
            continue
        cols = ["family", "split", "canon", "dL1", "dL2", "dL3", "dL4", "dL5",
                "min_stable_dL", "span"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "---|" * len(cols))
        # sort by family then split
        SPLIT_ORDER = {"LOTO": 0, "LOBO": 1, "JOINT": 2}
        tab2 = tab.copy()
        tab2["__so"] = tab2["split"].map(SPLIT_ORDER).fillna(99)
        tab2 = tab2.sort_values(["family", "__so"])
        for _, r in tab2.iterrows():
            row = [r["family"], r["split"]]
            for lvl in ["canon", "dL1", "dL2", "dL3", "dL4", "dL5"]:
                v = r.get(lvl, np.nan)
                row.append(f"{v:+.3f}" if pd.notna(v) else "—")
            row.append(str(r.get("min_stable_dL") or "—"))
            row.append(f"{r['span']:.3f}" if pd.notna(r.get("span")) else "—")
            out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="scripts/transfer_analysis_v4")
    ap.add_argument("--stab-dir",
                    default="analysis_v3/density_invariance_pair_sharded")
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="ρ threshold for feature-side stability")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="|Δρ_g| tolerance for fitted-side stability")
    ap.add_argument("--targets", nargs="+", default=["peak_pck", "auc_normalized"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.root)
    stab_dir = Path(args.stab_dir)

    print("Loading feature-side stability CSVs...")
    feat_stab = load_feature_stability(stab_dir, args.threshold)
    feat_min = {sp: min_n_for_stable(df, args.threshold)
                for sp, df in feat_stab.items()}

    print("Loading fitted-side summary_points.csv from lean sweep dirs...")
    fitted = load_fitted_summary(root)
    if fitted.empty:
        print("  WARNING: no lean density results found; "
              "fitted-side stability will be empty")
    fitted_tab = {tgt: fitted_stability_table(fitted, tgt, args.tol)
                  for tgt in args.targets}

    md = render(feat_stab, feat_min, fitted_tab, args.threshold, args.tol)
    out_path = args.out or (root / "ABLATION_density.md")
    Path(out_path).write_text(md)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
