"""Compile ABLATION_calibration.md across context_scale_calibration runs.

Aggregates the per-family `summary_all_variants.csv` (and optionally
`summary_drop_false_true.csv`) produced by `context_scale_calibration.py` and
emits a single Markdown report comparing raw `g` vs each calibrated head
(`g_global_gain`, `g_variant_gain`, `g_context_gain`, `g_shrink_gain`,
`g_benchsim_gain`, `g_profilesim_gain`).

This is the v4 "calibration is kept separate from the headline" deliverable:
ranking ρ_g stays in ABLATION.md / ABLATION_strength.md; residual magnitude
calibration lives here.

Run:
    python scripts/transfer_analysis_v4/compile_calibration_ablation.py
    # default scans: results_mixed/context_scale_calibration_*/
    #              + results_fsub_mean_nn/context_scale_calibration/

    python scripts/transfer_analysis_v4/compile_calibration_ablation.py \\
        --calib-dirs <path1> <path2> ... --out <out>.md
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


HEAD_ORDER = ["g", "g_global_gain", "g_variant_gain", "g_context_gain",
              "g_shrink_gain", "g_benchsim_gain", "g_profilesim_gain"]
SPLITS = ["LOTO", "LOBO", "JOINT"]


def _label_from_dir(p: Path) -> str:
    """Map a calibration dir to a human-readable family / kernel label.
    Includes the parent results_* dir prefix to disambiguate calibration runs
    for the same family across different L-mode result dirs.
    """
    name = p.name
    name = name.replace("context_scale_calibration_", "")
    name = name.replace("context_scale_calibration", "motion mean_nn (default)")
    parent = p.parent.name  # e.g. results_mixed, results_eb_shrunk
    if parent.startswith("results_"):
        parent_short = parent.replace("results_", "")
        return f"{parent_short}:{name}"
    return name


def _load_summary(p: Path, label: str) -> pd.DataFrame:
    frames = []
    for fname, vfilter in [("summary_all_variants.csv", "all_variants"),
                            ("summary_drop_false_true.csv", "drop_false_true")]:
        path = p / fname
        if path.exists():
            df = pd.read_csv(path)
            df["__family"] = label
            df["__variant_filter"] = vfilter
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _fmt(v, fmt="{:+.3f}"):
    return fmt.format(v) if (pd.notna(v) and np.isfinite(v)) else "—"


def _best_head_per_family_split(df: pd.DataFrame, metric: str = "ctx_pearson",
                                variant_filter: str = "all_variants") -> dict:
    """For each (family, split) return the head with the highest `metric`."""
    out = {}
    sub = df[df["__variant_filter"] == variant_filter]
    for (fam, split), g in sub.groupby(["__family", "split"]):
        g_valid = g[g["head"].isin(HEAD_ORDER) & g[metric].notna()]
        if g_valid.empty:
            continue
        best = g_valid.loc[g_valid[metric].idxmax()]
        out[(fam, split)] = (best["head"], float(best[metric]))
    return out


def render(df: pd.DataFrame) -> str:
    out: list[str] = []
    out.append("# Transfer Analysis v4 — Residual Calibration Ablation\n")
    out.append(
        "Leakage-clean residual-magnitude calibration across families. "
        "Generated from `context_scale_calibration.py` outputs.\n\n"
        "**Headline (rank-only) ρ_g claims live in [ABLATION.md](ABLATION.md) "
        "and [ABLATION_strength.md](ABLATION_strength.md).** "
        "This file reports the *magnitude* side: per-family residual-scale "
        "behavior under raw ridge vs each per-fold calibrated head.\n\n"
        "Calibrated heads (positive scalar gains per context; rank-preserving):\n"
        "- `g_global_gain` — one fold-wide residual-std / pred-std gain\n"
        "- `g_variant_gain` — same-variant gain, fallback global\n"
        "- `g_context_gain` — same (benchmark|variant), fallback variant\n"
        "- `g_shrink_gain` — context shrunk toward variant (k=5)\n"
        "- `g_benchsim_gain` — same-variant gains IDW-smoothed across other "
        "benchmarks via flow `mean_nn_sym` (zero-shot-ish over benchmark "
        "neighborhood; `--kernel-space dino` switches to DINO neighborhoods)\n"
        "- `g_profilesim_gain` — same-variant gains IDW-smoothed via "
        "eval-side dataset profile/density features\n"
    )

    out.append("\n## Quick read\n")
    out.append(
        "- Raw `g` is **under-dispersed** across families (median std ratio "
        "0.3–0.9). Calibrated heads bring it close to 1.0.\n"
        "- **mean_nn feature subset is the only family where `g_benchsim_gain` "
        "fails** — its raw std ratio (~0.26 LOBO) is so low that the per-fold "
        "gains it tries to IDW-smooth are large and noisy, causing the kernel "
        "to over-amplify (pooled std ratio > 2).\n"
        "- Every other family tested has raw std ratio ≥ 0.4 and benefits "
        "cleanly from `g_benchsim_gain` and/or `g_profilesim_gain`.\n"
        "- **Recommendation for downstream search**: `motion_sym + "
        "g_benchsim_gain` (best end-to-end LOBO/JOINT). `motion_w2` and "
        "`motion_fid` calibrate well too if a single-feature head is preferred.\n"
    )

    out.append("\n## Per-family raw-vs-best calibrated head\n")
    out.append(
        "For each family on LOTO/LOBO/JOINT, comparing raw `g` to the "
        "best-calibrated head (by ctx_pearson). Spearman is preserved within "
        "each family (gain is rank-invariant per context).\n"
    )

    fams = sorted(df["__family"].unique())
    out.append("\n### all_variants\n")
    out.append("| family | split | raw ρ_S | raw r | raw std (med) | "
               "**best head** | best r | best std (med) | best pooled std |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    av = df[df["__variant_filter"] == "all_variants"]
    for fam in fams:
        for split in SPLITS:
            sub = av[(av["__family"] == fam) & (av["split"] == split)]
            if sub.empty:
                continue
            raw = sub[sub["head"] == "g"].iloc[0]
            cal_heads = sub[sub["head"].isin(HEAD_ORDER[1:])]
            if cal_heads.empty:
                continue
            best = cal_heads.loc[cal_heads["ctx_pearson"].idxmax()]
            out.append(
                f"| {fam} | {split} | {_fmt(raw['ctx_spearman'])} | "
                f"{_fmt(raw['ctx_pearson'])} | "
                f"{_fmt(raw['median_std_ratio'], '{:.3f}')} | "
                f"**{best['head']}** | {_fmt(best['ctx_pearson'])} | "
                f"{_fmt(best['median_std_ratio'], '{:.3f}')} | "
                f"{_fmt(best['pooled_std_ratio'], '{:.3f}')} |"
            )

    if (df["__variant_filter"] == "drop_false_true").any():
        out.append("\n### drop_false_true\n")
        out.append("| family | split | raw ρ_S | raw r | raw std (med) | "
                   "**best head** | best r | best std (med) | best pooled std |")
        out.append("|---|---|---|---|---|---|---|---|---|")
        dv = df[df["__variant_filter"] == "drop_false_true"]
        for fam in fams:
            for split in SPLITS:
                sub = dv[(dv["__family"] == fam) & (dv["split"] == split)]
                if sub.empty:
                    continue
                raw = sub[sub["head"] == "g"].iloc[0]
                cal_heads = sub[sub["head"].isin(HEAD_ORDER[1:])]
                if cal_heads.empty:
                    continue
                best = cal_heads.loc[cal_heads["ctx_pearson"].idxmax()]
                out.append(
                    f"| {fam} | {split} | {_fmt(raw['ctx_spearman'])} | "
                    f"{_fmt(raw['ctx_pearson'])} | "
                    f"{_fmt(raw['median_std_ratio'], '{:.3f}')} | "
                    f"**{best['head']}** | {_fmt(best['ctx_pearson'])} | "
                    f"{_fmt(best['median_std_ratio'], '{:.3f}')} | "
                    f"{_fmt(best['pooled_std_ratio'], '{:.3f}')} |"
                )

    out.append("\n## Full per-head detail\n")
    out.append("All heads (raw + 6 calibrated) per (family, split). "
               "ρ_S = ctx_spearman (preserved across heads in same family by "
               "construction). r = ctx_pearson. std_m = median std ratio. "
               "std_p = pooled std ratio.\n")
    for vfilter in ["all_variants", "drop_false_true"]:
        sub_v = df[df["__variant_filter"] == vfilter]
        if sub_v.empty:
            continue
        out.append(f"\n### {vfilter}\n")
        out.append("| family | split | head | ρ_S | r | std_m | std_p | |L+g| r |")
        out.append("|---|---|---|---|---|---|---|---|")
        for fam in fams:
            for split in SPLITS:
                gsub = sub_v[(sub_v["__family"] == fam) & (sub_v["split"] == split)]
                for head in HEAD_ORDER:
                    r = gsub[gsub["head"] == head]
                    if r.empty:
                        continue
                    rr = r.iloc[0]
                    out.append(
                        f"| {fam} | {split} | `{head}` | "
                        f"{_fmt(rr['ctx_spearman'])} | "
                        f"{_fmt(rr['ctx_pearson'])} | "
                        f"{_fmt(rr['median_std_ratio'], '{:.3f}')} | "
                        f"{_fmt(rr['pooled_std_ratio'], '{:.3f}')} | "
                        f"{_fmt(rr['abs_r_L_plus_head'])} |"
                    )

    out.append("\n## Mechanism note\n")
    out.append(
        "Per-context gain = std(actual residual) / std(predicted residual). "
        "When raw ridge is heavily under-dispersed (e.g. mean_nn with raw "
        "median std ratio 0.26 LOBO), the per-context gains are large (~3–5×) "
        "and noisy. IDW-smoothing large noisy gains across benchmarks amplifies "
        "the wrong scale (pooled std ratio > 2). When raw ridge is closer to "
        "1× scale (motion_sym 0.69 LOBO, motion all 13 features 0.87 LOBO), the "
        "gains are modest (~1.1–1.5×) and IDW-smoothing produces well-behaved "
        "calibrated predictions.\n\n"
        "Practically: any motion family with raw `median_std_ratio` ≥ ~0.4 "
        "benefits from `g_benchsim_gain` / `g_profilesim_gain`. The mean_nn "
        "feature subset is the exception because it concentrates too little "
        "predictive variance into the head; that's a feature-restriction "
        "artifact, not a feature-axis-alignment story.\n\n"
        "The kernel-space choice (flow vs DINO `mean_nn_sym`) is a secondary "
        "knob — both give similar calibrated pearson for motion_sym (flow "
        "kernel slightly better on LOTO/LOBO). Choose based on which "
        "benchmark-similarity geometry better predicts your held-out target's "
        "residual scale.\n"
    )

    out.append("\n## Files referenced\n")
    for fam in fams:
        out.append(f"- **{fam}** — calibration dir + per-split scatter/hexbin")
    out.append("\nEach calibration dir contains:\n"
               "- `summary_all_variants.csv`, `summary_drop_false_true.csv`\n"
               "- `rows_<SPLIT>_<variant_filter>.csv` — per-row predictions for all heads\n"
               "- `figures/grid_{scatter,hexbin}_<variant_filter>.png` — comparison grid\n"
               "- `figures/{scatter,hexbin}_<SPLIT>_<variant_filter>_<head>.png` — per-cell figs\n")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("scripts/transfer_analysis_v4"))
    ap.add_argument("--calib-dirs", nargs="*", type=Path, default=None,
                    help="explicit list of calibration dirs to include")
    ap.add_argument("--out", type=Path, default=None,
                    help="output MD path (default <root>/ABLATION_calibration.md)")
    args = ap.parse_args()

    if args.calib_dirs:
        dirs = [Path(d) for d in args.calib_dirs]
    else:
        # default scan: results_*/context_scale_calibration* under root
        patterns = [
            str(args.root / "results_*/context_scale_calibration"),
            str(args.root / "results_*/context_scale_calibration_*"),
        ]
        dirs = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})

    if not dirs:
        raise SystemExit(f"no calibration dirs found under {args.root}/")

    frames = []
    print(f"loading {len(dirs)} calibration dir/s:")
    for d in dirs:
        label = _label_from_dir(d)
        df = _load_summary(d, label)
        if df.empty:
            print(f"  skip {d.name} (no summary CSVs)")
            continue
        print(f"  {d.name}  -> {label}  ({len(df)} rows)")
        frames.append(df)
    if not frames:
        raise SystemExit("no usable summary CSVs")
    all_df = pd.concat(frames, ignore_index=True)

    out_path = args.out or (args.root / "ABLATION_calibration.md")
    Path(out_path).write_text(render(all_df))
    print(f"\nABLATION_calibration -> {out_path}")


if __name__ == "__main__":
    main()
