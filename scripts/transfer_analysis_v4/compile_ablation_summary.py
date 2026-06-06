"""Compile a single ABLATION.md summarizing results across multiple v4 result
directories (one per L-mode / feature-subset / targeted-subset combination).

Scans `scripts/transfer_analysis_v4/results*/summary.csv`, parses each
directory's name to recover the ablation it represents (or falls back to the
dir name itself), and renders compact side-by-side tables.

Run:
    python scripts/transfer_analysis_v4/compile_ablation_summary.py
    python scripts/transfer_analysis_v4/compile_ablation_summary.py --out ABLATION.md
    python scripts/transfer_analysis_v4/compile_ablation_summary.py \
        --dirs results_mixed results_targeted_informed results_targeted_kl

The ABLATION.md focuses on the questions:
  1. Does L choice affect g (it shouldn't — g is L-invariant)?
  2. How does the level-only ρ_L change across L modes?
  3. How does pooled abs_r change across L modes?
  4. Within targeted_informed, which feature subset carries the source-cluster signal?
  5. Is g overfitting (does ridge ρ_g drop when --feature-subset restricts)?
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd


SPLITS = ["LOTO", "LOBO", "JOINT"]
FAMILIES = [
    "motion", "appearance", "both", "random", "density", "motion_density",
    "motion_sym", "motion_mmd", "motion_fid", "motion_w2",
    "appearance_sym", "appearance_mmd", "appearance_fid", "appearance_w2",
    "appearance_nullk",
]
FAMILY_LABEL = {
    "motion":         "**motion (flow)**",
    "appearance":     "appearance (DINO)",
    "both":           "both",
    "random":         "random (control)",
    "density":        "density",
    "motion_density": "motion + density",
    "motion_sym":     "**motion_sym (FID+SW2+MMD)**",
    "motion_mmd":     "motion_mmd",
    "motion_fid":     "motion_fid",
    "motion_w2":      "motion_w2",
    "appearance_sym": "appearance_sym (FID+SW2+MMD)",
    "appearance_mmd": "appearance_mmd",
    "appearance_fid": "appearance_fid",
    "appearance_w2":  "appearance_w2",
    "appearance_nullk": "appearance_nullk",
}


def _fmt_ci(v, lo, hi):
    if not np.isfinite(v):
        return "—"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        # CIs missing (e.g. lean point-estimate runs) — fall back to point value
        return f"{v:+.3f}"
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _fmt_pt(v):
    return f"{v:+.3f}" if np.isfinite(v) else "—"


def _label_for(dir_name: str) -> str:
    """Map a results dir name to a human-readable ablation label."""
    base = os.path.basename(dir_name.rstrip("/"))
    name = base.replace("results_", "").replace("results", "main")
    if name == "" or name == "main":
        return "mixed (default)"
    if name.startswith("fsub_"):
        return f"feature-subset={name[len('fsub_'):]}"
    # Common heuristic mappings:
    if name.startswith("targeted_"):
        rest = name[len("targeted_"):]
        if rest in ("informed", "all"):
            return "targeted_informed (all features)"
        return f"targeted_informed (targeted-subset={rest})"
    if name.startswith("feature_subset_"):
        return f"feature-subset={name[len('feature_subset_'):]}"
    if name in ("mixed", "symmetric_informed", "symmetric_uninformed", "targeted_informed"):
        return name
    return base


def _read_summary(d: Path) -> pd.DataFrame | None:
    p = d / "summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["__dir"] = d.name
    df["__label"] = _label_for(d.name)
    return df


def _wide_by_split(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    return df.pivot_table(index="family", columns="split", values=value_col).reindex(FAMILIES)


def _section_header(title: str) -> str:
    return f"\n## {title}\n"


def _table_metric_by_ablation(all_df: pd.DataFrame, target: str,
                              head: str, value: str, families: list[str],
                              splits: list[str] = SPLITS,
                              with_ci: bool = False) -> str:
    """Per-ablation table: columns = (split × family), one row per ablation."""
    main = all_df[(all_df["label"] == "main") &
                  (all_df["target"] == target) &
                  (all_df["head"] == head)]
    cols = [(s, f) for s in splits for f in families]
    header = ["ablation"]
    for s, f in cols:
        header.append(f"{s}/{f.split('_')[0]}")
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * (len(header)))
    for lbl in sorted(main["__label"].unique()):
        sub = main[main["__label"] == lbl]
        row = [lbl]
        for s, f in cols:
            cell = sub[(sub["split"] == s) & (sub["family"] == f)]
            if cell.empty:
                row.append("—")
            else:
                v = float(cell[value].iloc[0])
                if with_ci and f"{value}_lo" in cell.columns:
                    lo = float(cell[f"{value}_lo"].iloc[0])
                    hi = float(cell[f"{value}_hi"].iloc[0])
                    row.append(_fmt_ci(v, lo, hi))
                else:
                    row.append(_fmt_pt(v))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_metric_by_ablation_split_rows(all_df: pd.DataFrame, target: str,
                                         head: str, value: str,
                                         families: list[str],
                                         splits: list[str] = SPLITS,
                                         with_ci: bool = False) -> str:
    """Per-ablation table with one row per split and columns = families.

    This keeps the symmetric/FID/W2 comparison readable without creating a
    very wide split × family table.
    """
    main = all_df[(all_df["label"] == "main") &
                  (all_df["target"] == target) &
                  (all_df["head"] == head)]
    present = [f for f in families if f in set(main["family"])]
    if not present:
        return "_No rows available yet. Re-run v4 after the corresponding feature columns land._"

    header = ["ablation", "split"] + present
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "---|" * len(header))
    for lbl in sorted(main["__label"].unique()):
        for split in splits:
            sub = main[(main["__label"] == lbl) & (main["split"] == split)]
            if not any((sub["family"] == f).any() for f in present):
                continue
            row = [lbl, split]
            for fam in present:
                cell = sub[sub["family"] == fam]
                if cell.empty:
                    row.append("—")
                else:
                    v = float(cell[value].iloc[0])
                    if with_ci and f"{value}_lo" in cell.columns:
                        lo = float(cell[f"{value}_lo"].iloc[0])
                        hi = float(cell[f"{value}_hi"].iloc[0])
                        row.append(_fmt_ci(v, lo, hi))
                    else:
                        row.append(_fmt_pt(v))
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render(all_df: pd.DataFrame) -> str:
    out = []
    out.append("# Transfer Analysis v4 — L / Feature Ablation Summary\n")
    labels = sorted(all_df["__label"].unique())
    out.append(f"Combining {all_df['__dir'].nunique()} result directories:\n")
    for lbl in labels:
        dirs = sorted(all_df[all_df["__label"] == lbl]["__dir"].unique())
        out.append(f"- **{lbl}** — `{', '.join(dirs)}`")
    out.append("")

    targets = sorted(all_df["target"].unique())
    headline_families = ["motion", "appearance", "random"]

    out.append("---")
    out.append(
        "\n## Quick read\n\n"
        "- **ctx_rho_g (ridge)** should be ~identical across L modes — g is L-invariant. Any drift is just rounding/sampling.\n"
        "- **ctx_rho_g (ridge)** should change with `--feature-subset` if the dropped features carry signal. If it doesn't move much, ridge was already down-weighting them.\n"
        "- **ctx_rho_L** is the *level-only* ranking score and changes a lot across L modes — that's the comparison the ablations are actually testing.\n"
        "- **abs_r_Lg** is calibration. Lower under uniform L; comparable across informed variants.\n"
    )

    for target in targets:
        out.append(f"\n---\n\n# Target: `{target}`\n")

        out.append(_section_header("1. ridge ctx_rho_g — feature claim (should be L-invariant)"))
        out.append(_table_metric_by_ablation(all_df, target, "g", "ctx_rho_g",
                                             headline_families))

        out.append(_section_header("2. symmetric / FID / W2 feature families — ridge ctx_rho_g"))
        out.append(_table_metric_by_ablation_split_rows(
            all_df, target, "g", "ctx_rho_g",
            ["motion_sym", "motion_fid", "motion_w2", "motion_mmd",
             "appearance_sym", "appearance_fid", "appearance_w2", "appearance_mmd",
             "appearance_nullk"],
            with_ci=True,
        ))

        out.append(_section_header("3. ctx_rho_L — level-only ranking ρ"))
        out.append(_table_metric_by_ablation(all_df, target, "g", "ctx_rho_L",
                                             headline_families))

        out.append(_section_header("4. abs_r_Lg — pooled calibration"))
        out.append(_table_metric_by_ablation(all_df, target, "g", "abs_r_Lg",
                                             headline_families))

        # Per-target / family detail with CIs (motion only — the headline family)
        out.append(_section_header("5. motion ridge ctx_rho_g — with 95% CIs"))
        sub = all_df[(all_df["label"] == "main") & (all_df["target"] == target) &
                     (all_df["head"] == "g") & (all_df["family"] == "motion")]
        lines = ["| ablation | LOTO | LOBO | JOINT |", "|---|---|---|---|"]
        for lbl in sorted(sub["__label"].unique()):
            row = [lbl]
            for s in SPLITS:
                r = sub[(sub["__label"] == lbl) & (sub["split"] == s)]
                if r.empty:
                    row.append("—")
                else:
                    rr = r.iloc[0]
                    row.append(_fmt_ci(rr["ctx_rho_g"], rr["ctx_rho_g_lo"], rr["ctx_rho_g_hi"]))
            lines.append("| " + " | ".join(row) + " |")
        out.append("\n".join(lines))

    out.append("\n---\n\n## Files referenced\n")
    for d in sorted(all_df["__dir"].unique()):
        out.append(f"- `scripts/transfer_analysis_v4/{d}/results.md` — full per-mode report")
        out.append(f"- `scripts/transfer_analysis_v4/{d}/summary.csv` — long-form metrics")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="scripts/transfer_analysis_v4")
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="explicit list of result subdirs (basenames). "
                         "If omitted, scans for results*/.")
    ap.add_argument("--out", default=None,
                    help="output file (default: <root>/ABLATION.md)")
    args = ap.parse_args()

    root = Path(args.root)
    if args.dirs:
        dirs = [root / d for d in args.dirs]
    else:
        dirs = sorted(root.glob("results*"))
        # Filter to dirs with summary.csv
        dirs = [d for d in dirs if d.is_dir() and (d / "summary.csv").exists()]

    if not dirs:
        raise SystemExit(f"no result directories with summary.csv found under {root}/")

    print(f"loading {len(dirs)} result directory/ies:")
    frames = []
    for d in dirs:
        df = _read_summary(d)
        if df is not None and not df.empty:
            print(f"  {d.name}: {len(df)} rows, {df['target'].nunique()} target(s)")
            frames.append(df)
    if not frames:
        raise SystemExit("no usable summary.csv files")
    all_df = pd.concat(frames, ignore_index=True)

    out = args.out or (root / "ABLATION.md")
    Path(out).write_text(render(all_df))
    print(f"\nABLATION summary -> {out}")


if __name__ == "__main__":
    main()
