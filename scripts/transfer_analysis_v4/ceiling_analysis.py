"""
Predictability ceiling for the transfer-ranking estimand.

WHY: the within-context source ranking can only be predicted as well as it is
REPRODUCIBLE. We treat each model variant (and, once trained, each architecture)
as a *replicate* of the same (source, benchmark) dyad. If sources rank
consistently across replicates, that agreement is reproducible signal a dataset
feature could in principle capture; the replicate-to-replicate scramble is noise
no static feature should be expected to predict. So:

    ceiling  = mean inter-replicate Spearman of the source ranking, per benchmark
    motion's fraction-of-ceiling = (motion benchmark-level rho) / ceiling

We also report the SOURCE MAIN-EFFECT decomposition: the share of within-cell
variance that is generic source quality (a dummy per source) = the ceiling for any
feature *of the source alone*. The remainder is dyadic interaction + replicate noise.

CAVEAT: the current replicates (CATs++ freeze/pretrain toggles + RAFT) are
CORRELATED, so the ceiling is an optimistic UPPER bound and motion's fraction is a
conservative FLOOR. Re-run with independent architectures (GLU-Net / FlowFormer++ /
PWCNet) — they enter the rows table as new `variant` values automatically — for the
unbiased number. `--collapse-arch` computes the ceiling across architecture families
only (first token of the variant), the more conservative cross-architecture version.

Usage:
    python scripts/transfer_analysis_v4/ceiling_analysis.py \
        --rows scripts/transfer_analysis_v4/results/predictions/peak_pck/rows_LOTO_motion.csv \
        [--replicate-col variant] [--collapse-arch] \
        [--out scripts/transfer_analysis_v4/CEILING.md]
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def reproducibility_ceiling(d, rep_col, min_src=4):
    """Per-benchmark mean pairwise inter-replicate Spearman of the source ranking."""
    per_bench, allr = {}, []
    for k, g in d.groupby("benchmark"):
        piv = g.pivot_table(index="train_dataset", columns=rep_col,
                            values="actual", aggfunc="mean")
        reps = [r for r in piv.columns if piv[r].notna().sum() >= min_src]
        pr = []
        for a, b in combinations(reps, 2):
            s = piv[[a, b]].dropna()
            if len(s) >= min_src and s[a].std() > 0 and s[b].std() > 0:
                pr.append(spearmanr(s[a], s[b]).statistic)
        if pr:
            per_bench[k] = dict(ceiling=float(np.mean(pr)), n_reps=len(reps),
                                n_pairs=len(pr))
            allr += pr
    pooled = float(np.nanmean(allr)) if allr else float("nan")
    return per_bench, pooled


def motion_benchmark_rho(d, min_src=4):
    """Motion rho predicting the replicate-averaged source ranking within a benchmark."""
    rs = {}
    for k, g in d.groupby("benchmark"):
        a = g.groupby("train_dataset")["actual"].mean()
        p = g.groupby("train_dataset")[["L", "g"]].mean().sum(axis=1)
        s = pd.concat([a.rename("a"), p.rename("p")], axis=1).dropna()
        if len(s) >= min_src and s["a"].std() > 0 and s["p"].std() > 0:
            rs[k] = float(spearmanr(s["a"], s["p"]).statistic)
    return rs


def source_main_effect_share(d):
    """Share of within-cell variance that is generic source quality (= ceiling for any
       SOURCE-LEVEL feature). cell = context_id (benchmark|variant)."""
    d = d.copy()
    cell = (d["context_id"] if "context_id" in d.columns
            else d["benchmark"].astype(str) + "|" + d["variant"].astype(str))
    d["dev"] = d["actual"] - d.groupby(cell)["actual"].transform("mean")
    ss_tot = float((d["dev"] ** 2).sum())
    src_mean = d.groupby("train_dataset")["dev"].transform("mean")
    ss_src = float((src_mean ** 2).sum())
    return (ss_src / ss_tot) if ss_tot > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="scripts/transfer_analysis_v4/results/"
                    "predictions/peak_pck/rows_LOTO_motion.csv",
                    help="motion rows file (needs actual, L, g, train_dataset, "
                         "benchmark, variant).")
    ap.add_argument("--replicate-col", default="variant",
                    help="column treated as the replicate dimension.")
    ap.add_argument("--collapse-arch", action="store_true",
                    help="collapse `variant` to architecture family (first token "
                         "before '|') and use THAT as the replicate — the more "
                         "conservative cross-architecture ceiling.")
    ap.add_argument("--min-src", type=int, default=4,
                    help="min sources shared by a replicate pair to score it.")
    ap.add_argument("--out", default="scripts/transfer_analysis_v4/CEILING.md")
    args = ap.parse_args()

    d = pd.read_csv(args.rows)
    rep = args.replicate_col
    if args.collapse_arch:
        d["arch"] = d["variant"].astype(str).str.split("|").str[0]
        rep = "arch"

    per_bench, pooled = reproducibility_ceiling(d, rep, args.min_src)
    motion = motion_benchmark_rho(d, args.min_src)
    src_share = source_main_effect_share(d)

    # assemble per-benchmark table
    rows = []
    for k in sorted(set(per_bench) | set(motion)):
        c = per_bench.get(k, {}).get("ceiling", np.nan)
        m = motion.get(k, np.nan)
        frac = (m / c) if (c and not np.isnan(c) and c != 0) else np.nan
        rows.append(dict(benchmark=k, ceiling=c, motion_rho=m, frac_of_ceiling=frac,
                         n_reps=per_bench.get(k, {}).get("n_reps", 0)))
    tab = pd.DataFrame(rows).sort_values("ceiling", ascending=False)
    pooled_motion = float(np.nanmean(list(motion.values())))
    pooled_frac = pooled_motion / pooled if pooled else float("nan")

    n_rep = d[rep].nunique()
    hdr = (f"# Predictability Ceiling\n\n"
           f"Replicate dimension: **{rep}** ({n_rep} levels"
           f"{' — architecture-collapsed' if args.collapse_arch else ''}). "
           f"Source: `{args.rows}`.\n\n"
           f"**Ceiling = how reproducibly model {rep}s agree on the source ranking "
           f"within a benchmark.** Motion can't beat what the target itself doesn't "
           f"reproduce. `frac_of_ceiling = motion_rho / ceiling`.\n\n")
    body = "| benchmark | ceiling ρ | motion ρ | motion / ceiling | n_reps |\n"
    body += "|---|---|---|---|---|\n"
    for _, r in tab.iterrows():
        body += (f"| {r.benchmark} | {r.ceiling:+.3f} | {r.motion_rho:+.3f} | "
                 f"{r.frac_of_ceiling:.0%} | {int(r.n_reps)} |\n")
    body += (f"| **POOLED** | **{pooled:+.3f}** | **{pooled_motion:+.3f}** | "
             f"**{pooled_frac:.0%}** | {n_rep} |\n")

    foot = (f"\n**Source main-effect share:** {src_share:.1%} of within-cell variance "
            f"is generic source quality (ceiling for any *source-level* feature); the "
            f"rest is dyadic interaction + replicate noise.\n\n"
            f"> [!warning] Current {rep}s are correlated (CATs++ toggles + RAFT), so "
            f"the ceiling is an optimistic UPPER bound and motion's fraction a "
            f"conservative FLOOR. Re-run with GLU-Net / FlowFormer++ / PWCNet (they "
            f"enter as new `variant` rows automatically) for the unbiased number.\n")

    Path(args.out).write_text(hdr + body + foot)
    tab.to_csv(Path(args.out).with_suffix(".csv"), index=False)
    print(hdr + body + foot)
    print(f"wrote {args.out} and {Path(args.out).with_suffix('.csv')}")


if __name__ == "__main__":
    main()
