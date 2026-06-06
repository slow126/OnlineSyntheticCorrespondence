"""Strength-of-correspondence tests for the g (within-context ridge) head.

For each result directory + family, this computes:

  - point estimate ctx_rho_g
  - bootstrap 95% CI on ctx_rho_g
  - P(rho_g > 0) from the bootstrap distribution (one-sided)
  - shuffle-null distribution stats: mean ± std of rho_g over the
    `_shuffle.csv` (within-context label-permuted) predictions, refit
    across the same bootstrap reps for a fair comparison
  - z-score = (real_rho - null_mean) / null_std

Plus three paired-gap tests per split / dir:

  - motion - appearance
  - motion_sym - appearance_sym
  - motion - random

Each gap test: paired-entity bootstrap, 95% CI on the difference and a
one-sided p-value P(gap > 0). The motion-appearance gap mirrors what
bootstrap_gap.csv already does; the other two extend it.

Writes:
  <root>/strength_per_family.csv
  <root>/strength_paired_gaps.csv
  <root>/ABLATION_strength.md

Run:
  python scripts/transfer_analysis_v4/strength_tests.py \
      --dirs results_mixed results_eb_shrunk ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap import Prepared, _metrics_from_rows, _spearman  # noqa: E402


HEADS = ["g"]                       # ridge only — the headline head
SPLITS = ["LOTO", "LOBO", "JOINT"]
PAIRS = [
    ("motion", "appearance"),
    ("motion_sym", "appearance_sym"),
    ("motion", "random"),
]


def _bootstrap_rho_dist(prep: Prepared, head: str, n_boot: int,
                        seed: int) -> np.ndarray:
    """Return the bootstrap distribution of ctx_rho_g (n_boot values)."""
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        rows = prep.sample_rows(rng)
        m = _metrics_from_rows(*prep.slice(rows, head))
        out[b] = m["ctx_rho_g"]
    return out


def _paired_gap_dist(prep_m: Prepared, prep_a: Prepared, head: str,
                     n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of (motion_rho - appearance_rho), paired entities."""
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        picked = rng.choice(len(prep_m.entities), size=len(prep_m.entities),
                            replace=True)
        ents = prep_m.entities[picked]
        rows_m = np.concatenate([prep_m.entity_to_rows[e] for e in ents])
        rows_a = np.concatenate([prep_a.entity_to_rows[e] for e in ents])
        mm = _metrics_from_rows(*prep_m.slice(rows_m, head))
        ma = _metrics_from_rows(*prep_a.slice(rows_a, head))
        out[b] = mm["ctx_rho_g"] - ma["ctx_rho_g"]
    return out


def _summarize(dist: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, lo, hi) — mean and 95% CI from finite samples."""
    s = dist[np.isfinite(dist)]
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.quantile(s, [0.025, 0.975])
    return float(s.mean()), float(lo), float(hi)


def _p_gt_zero(dist: np.ndarray) -> float:
    s = dist[np.isfinite(dist)]
    return float((s > 0).mean()) if s.size else float("nan")


def per_family(dirs: list[Path], n_boot: int, seed: int, target: str) -> pd.DataFrame:
    rows = []
    for d in dirs:
        pred_dir = d / "predictions" / target
        if not pred_dir.exists():
            print(f"  skip {d.name}: no predictions/{target}/")
            continue
        # Discover families from main rows_*.csv (not _shuffle, not _uniformL)
        fams = set()
        for p in pred_dir.glob("rows_*.csv"):
            stem = p.stem  # rows_LOBO_motion or rows_LOBO_motion_shuffle
            if stem.endswith("_shuffle") or stem.endswith("_uniformL"):
                continue
            parts = stem.split("_", 2)  # ["rows", split, family]
            if len(parts) == 3:
                fams.add(parts[2])
        for split in SPLITS:
            for fam in sorted(fams):
                main = pred_dir / f"rows_{split}_{fam}.csv"
                shuf = pred_dir / f"rows_{split}_{fam}_shuffle.csv"
                if not main.exists():
                    continue
                df_m = pd.read_csv(main)
                prep_m = Prepared(df_m, split)
                # Real distribution
                seed_real = seed + abs(hash((d.name, split, fam, "real"))) % 10000
                dist_real = _bootstrap_rho_dist(prep_m, "g", n_boot, seed_real)
                rho_pt, lo, hi = _summarize(dist_real)
                p_gt0 = _p_gt_zero(dist_real)
                # Null distribution (from shuffle preds)
                null_mean = null_std = z = float("nan")
                if shuf.exists():
                    df_s = pd.read_csv(shuf)
                    prep_s = Prepared(df_s, split)
                    seed_null = seed + abs(hash((d.name, split, fam, "null"))) % 10000
                    dist_null = _bootstrap_rho_dist(prep_s, "g", n_boot, seed_null)
                    sd = dist_null[np.isfinite(dist_null)]
                    if sd.size:
                        null_mean = float(sd.mean())
                        null_std = float(sd.std(ddof=1)) if sd.size > 1 else float("nan")
                        if null_std and null_std > 0:
                            z = (rho_pt - null_mean) / null_std
                rows.append(dict(
                    dir=d.name, split=split, family=fam, target=target,
                    rho_g=rho_pt, rho_g_lo=lo, rho_g_hi=hi, p_rho_gt_0=p_gt0,
                    null_mean=null_mean, null_std=null_std, z_vs_null=z,
                ))
                print(f"  {d.name:30s} {split:6s} {fam:18s} "
                      f"ρ_g={rho_pt:+.3f} [{lo:+.3f},{hi:+.3f}] "
                      f"P(>0)={p_gt0:.3f}  z={z:+.2f}")
    return pd.DataFrame(rows)


def paired_gaps(dirs: list[Path], n_boot: int, seed: int,
                target: str) -> pd.DataFrame:
    rows = []
    for d in dirs:
        pred_dir = d / "predictions" / target
        if not pred_dir.exists():
            continue
        for split in SPLITS:
            for fam_m, fam_a in PAIRS:
                m_path = pred_dir / f"rows_{split}_{fam_m}.csv"
                a_path = pred_dir / f"rows_{split}_{fam_a}.csv"
                if not (m_path.exists() and a_path.exists()):
                    continue
                df_m = pd.read_csv(m_path)
                df_a = pd.read_csv(a_path)
                prep_m = Prepared(df_m, split)
                prep_a = Prepared(df_a, split)
                if not np.array_equal(prep_m.entities, prep_a.entities):
                    print(f"  {d.name} {split} {fam_m}/{fam_a}: entity mismatch, skipping")
                    continue
                sd = seed + abs(hash((d.name, split, fam_m, fam_a))) % 10000
                dist = _paired_gap_dist(prep_m, prep_a, "g", n_boot, sd)
                gap_pt, lo, hi = _summarize(dist)
                p_gt0 = _p_gt_zero(dist)
                rows.append(dict(
                    dir=d.name, split=split, target=target,
                    family_m=fam_m, family_a=fam_a,
                    gap_g=gap_pt, gap_g_lo=lo, gap_g_hi=hi, p_gap_gt_0=p_gt0,
                ))
                print(f"  {d.name:30s} {split:6s} {fam_m}-{fam_a:18s} "
                      f"gap={gap_pt:+.3f} [{lo:+.3f},{hi:+.3f}] P(>0)={p_gt0:.3f}")
    return pd.DataFrame(rows)


def render_md(per_fam: pd.DataFrame, gaps: pd.DataFrame) -> str:
    out = []
    out.append("# Transfer Analysis v4 — Strength of g-correspondence\n")
    out.append("Per-family bootstrap CIs, P(ρ_g > 0), and shuffle-null "
               "comparison; plus paired-gap tests for motion-vs-appearance "
               "and the motion_sym-vs-appearance_sym comparison.\n")

    targets = sorted(per_fam["target"].unique())
    headline_fams = ["motion", "motion_sym", "motion_fid", "motion_w2", "motion_mmd",
                     "appearance", "appearance_sym", "appearance_fid",
                     "appearance_w2", "appearance_mmd",
                     "both", "random", "density"]

    for tgt in targets:
        out.append(f"\n## Target: `{tgt}`\n")
        # Use the canonical mixed dir if available; else first dir alphabetically.
        dirs = sorted(per_fam[per_fam["target"] == tgt]["dir"].unique())
        canon = "results_mixed" if "results_mixed" in dirs else dirs[0]
        sub = per_fam[(per_fam["target"] == tgt) & (per_fam["dir"] == canon)]
        out.append(f"_From `{canon}` (canonical mixed L-mode)._\n")
        out.append("\n### Per-family strength (ctx_rho_g)\n")
        out.append("| family | split | ρ_g [95% CI] | P(ρ_g > 0) | null mean ± std | z vs null |")
        out.append("|---|---|---|---|---|---|")
        for fam in headline_fams:
            for split in SPLITS:
                r = sub[(sub["family"] == fam) & (sub["split"] == split)]
                if r.empty:
                    continue
                rr = r.iloc[0]
                ci = (f"{rr['rho_g']:+.3f} [{rr['rho_g_lo']:+.3f}, "
                      f"{rr['rho_g_hi']:+.3f}]")
                p = f"{rr['p_rho_gt_0']:.3f}"
                null = (f"{rr['null_mean']:+.3f} ± {rr['null_std']:.3f}"
                        if pd.notna(rr['null_std']) else "—")
                z = f"{rr['z_vs_null']:+.2f}" if pd.notna(rr['z_vs_null']) else "—"
                out.append(f"| {fam} | {split} | {ci} | {p} | {null} | {z} |")

        out.append("\n### Paired gaps (motion-side − appearance-side)\n")
        out.append("| gap | split | gap_g [95% CI] | P(gap > 0) |")
        out.append("|---|---|---|---|")
        gsub = gaps[(gaps["target"] == tgt) & (gaps["dir"] == canon)]
        for _, rr in gsub.iterrows():
            ci = (f"{rr['gap_g']:+.3f} [{rr['gap_g_lo']:+.3f}, "
                  f"{rr['gap_g_hi']:+.3f}]")
            p = f"{rr['p_gap_gt_0']:.3f}"
            out.append(f"| {rr['family_m']} − {rr['family_a']} | "
                       f"{rr['split']} | {ci} | {p} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="scripts/transfer_analysis_v4")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="basenames under --root")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--targets", nargs="+", default=["peak_pck", "auc_normalized"])
    ap.add_argument("--out", default=None,
                    help="output MD path (default <root>/ABLATION_strength.md)")
    args = ap.parse_args()
    root = Path(args.root)
    dirs = [root / d for d in args.dirs]
    print(f"strength tests across {len(dirs)} dirs × {len(args.targets)} targets, "
          f"n_boot={args.n_boot}\n")

    pf_frames = []
    pg_frames = []
    for tgt in args.targets:
        print(f"\n=== Target: {tgt} ===\n--- per-family ---")
        pf = per_family(dirs, args.n_boot, args.seed, tgt)
        pf_frames.append(pf)
        print(f"--- paired gaps ---")
        pg = paired_gaps(dirs, args.n_boot, args.seed, tgt)
        pg_frames.append(pg)

    per_fam_df = pd.concat(pf_frames, ignore_index=True)
    gaps_df = pd.concat(pg_frames, ignore_index=True)
    per_fam_df.to_csv(root / "strength_per_family.csv", index=False)
    gaps_df.to_csv(root / "strength_paired_gaps.csv", index=False)
    print(f"\n-> {root}/strength_per_family.csv")
    print(f"-> {root}/strength_paired_gaps.csv")

    out_path = args.out or (root / "ABLATION_strength.md")
    Path(out_path).write_text(render_md(per_fam_df, gaps_df))
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
