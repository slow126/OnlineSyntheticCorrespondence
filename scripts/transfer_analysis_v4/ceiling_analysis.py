#!/usr/bin/env python3
"""Empirical predictability references for variant-specific transfer rankings.

This intentionally does not claim a mathematical ceiling. It reports:

1. The headline feature score: mean within-context Spearman(actual, g).
2. Pairwise inter-variant agreement: target reproducibility, not a bound.
3. Held-variant consensus: predict each variant's source ranking from the
   rankings of all other variants. This is the most useful empirical oracle
   because it is leave-one-variant-out and uses no held-variant outcomes.
4. Architecture-balanced held-variant consensus: the same calculation after
   giving each architecture family equal weight, so CATs++ toggles do not
   dominate the consensus.
5. Classical reliability approximations, clearly labeled as assumption-heavy
   sensitivity calculations rather than primary results.

Usage:
    python scripts/transfer_analysis_v4/ceiling_analysis.py \
        --rows scripts/transfer_analysis_v4/results/predictions/peak_pck/rows_LOTO_motion.csv \
        --out scripts/transfer_analysis_v4/CEILING_REVISED.md
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def safe_spearman(a: pd.Series, b: pd.Series, min_src: int) -> float:
    pair = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(pair) < min_src or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return float("nan")
    rho = spearmanr(pair["a"], pair["b"]).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def context_feature_scores(d: pd.DataFrame, min_src: int) -> pd.DataFrame:
    rows = []
    for context, group in d.groupby("context_id"):
        rho = safe_spearman(
            group.set_index("train_dataset")["actual"],
            group.set_index("train_dataset")["g"],
            min_src,
        )
        if np.isfinite(rho):
            rows.append({
                "context_id": context,
                "benchmark": group["benchmark"].iloc[0],
                "variant": group["variant"].iloc[0],
                "feature_rho": rho,
                "n_sources": group["train_dataset"].nunique(),
            })
    return pd.DataFrame(rows)


def pairwise_agreement(
    d: pd.DataFrame,
    rep_col: str,
    min_src: int,
) -> pd.DataFrame:
    rows = []
    for benchmark, group in d.groupby("benchmark"):
        pivot = group.pivot_table(
            index="train_dataset",
            columns=rep_col,
            values="actual",
            aggfunc="mean",
        )
        reps = [c for c in pivot if pivot[c].notna().sum() >= min_src]
        rhos = []
        for left, right in combinations(reps, 2):
            rho = safe_spearman(pivot[left], pivot[right], min_src)
            if np.isfinite(rho):
                rhos.append(rho)
        if rhos:
            rows.append({
                "benchmark": benchmark,
                "pairwise_rho": float(np.mean(rhos)),
                "n_replicates": len(reps),
                "n_pairs": len(rhos),
            })
    return pd.DataFrame(rows)


def _rank_columns(pivot: pd.DataFrame) -> pd.DataFrame:
    return pivot.apply(lambda col: col.rank(method="average", pct=True))


def held_variant_consensus(
    d: pd.DataFrame,
    min_src: int,
    balance_groups: bool,
) -> pd.DataFrame:
    """Score each held variant against rank consensus from all other variants."""
    rows = []
    for benchmark, group in d.groupby("benchmark"):
        actual = group.pivot_table(
            index="train_dataset",
            columns="variant",
            values="actual",
            aggfunc="mean",
        )
        pred = group.pivot_table(
            index="train_dataset",
            columns="variant",
            values="g",
            aggfunc="mean",
        )
        ranks = _rank_columns(actual)

        for held in actual:
            other_cols = [c for c in actual if c != held]
            if len(other_cols) < 2:
                continue

            other_ranks = ranks[other_cols]
            if balance_groups:
                group_names = pd.Index(
                    [str(c).split("|", 1)[0] for c in other_cols],
                    name="replicate_group",
                )
                grouped = other_ranks.T.groupby(group_names).mean().T
                consensus = grouped.mean(axis=1)
                support = grouped.notna().sum(axis=1)
                required_support = len(grouped.columns)
                n_consensus_units = len(grouped.columns)
            else:
                consensus = other_ranks.mean(axis=1)
                support = other_ranks.notna().sum(axis=1)
                required_support = 2
                n_consensus_units = len(other_cols)

            consensus = consensus.where(support >= required_support)
            held_actual = actual[held]
            held_pred = pred[held] if held in pred else pd.Series(dtype=float)
            consensus_rho = safe_spearman(held_actual, consensus, min_src)
            feature_rho = safe_spearman(held_actual, held_pred, min_src)
            if np.isfinite(consensus_rho):
                rows.append({
                    "benchmark": benchmark,
                    "held_variant": held,
                    "held_group": str(held).split("|", 1)[0],
                    "consensus_rho": consensus_rho,
                    "feature_rho": feature_rho,
                    "n_sources": int(
                        pd.concat([held_actual, consensus], axis=1).dropna().shape[0]
                    ),
                    "n_consensus_units": n_consensus_units,
                    "balanced": balance_groups,
                })
    return pd.DataFrame(rows)


def held_group_consensus(d: pd.DataFrame, min_src: int) -> pd.DataFrame:
    """Aggregate variants within architecture, then hold out one architecture."""
    rows = []
    for benchmark, group in d.groupby("benchmark"):
        actual = group.pivot_table(
            index="train_dataset",
            columns="variant",
            values="actual",
            aggfunc="mean",
        )
        pred = group.pivot_table(
            index="train_dataset",
            columns="variant",
            values="g",
            aggfunc="mean",
        )
        actual_ranks = _rank_columns(actual)
        pred_ranks = _rank_columns(pred)
        groups = sorted({str(c).split("|", 1)[0] for c in actual})
        if len(groups) < 2:
            continue

        actual_by_group = pd.DataFrame({
            name: actual_ranks[
                [c for c in actual_ranks if str(c).split("|", 1)[0] == name]
            ].mean(axis=1)
            for name in groups
        })
        pred_by_group = pd.DataFrame({
            name: pred_ranks[
                [c for c in pred_ranks if str(c).split("|", 1)[0] == name]
            ].mean(axis=1)
            for name in groups
        })

        for held in groups:
            others = [name for name in groups if name != held]
            consensus = actual_by_group[others].mean(axis=1)
            consensus_rho = safe_spearman(
                actual_by_group[held], consensus, min_src
            )
            feature_rho = safe_spearman(
                actual_by_group[held], pred_by_group[held], min_src
            )
            if np.isfinite(consensus_rho):
                rows.append({
                    "benchmark": benchmark,
                    "held_group": held,
                    "consensus_rho": consensus_rho,
                    "feature_rho": feature_rho,
                    "n_sources": int(
                        pd.concat(
                            [actual_by_group[held], consensus],
                            axis=1,
                        ).dropna().shape[0]
                    ),
                    "n_other_groups": len(others),
                })
    return pd.DataFrame(rows)


def source_main_effect_share(d: pd.DataFrame) -> float:
    """Descriptive source-main-effect share after removing context means."""
    centered = d.copy()
    centered["dev"] = (
        centered["actual"]
        - centered.groupby("context_id")["actual"].transform("mean")
    )
    ss_total = float(np.square(centered["dev"]).sum())
    source_mean = centered.groupby("train_dataset")["dev"].transform("mean")
    ss_source = float(np.square(source_mean).sum())
    return ss_source / ss_total if ss_total > 0 else float("nan")


def benchmark_bootstrap(
    per_benchmark: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    if per_benchmark.empty or n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    values = per_benchmark.set_index("benchmark")
    benches = values.index.to_numpy()
    draws = {c: [] for c in values.columns}
    for _ in range(n_boot):
        sample = rng.choice(benches, size=len(benches), replace=True)
        selected = values.loc[sample]
        for col in values:
            draws[col].append(float(selected[col].mean()))
    return {
        col: tuple(np.nanpercentile(vals, [2.5, 97.5]))
        for col, vals in draws.items()
    }


def fmt(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:+.{digits}f}"


def pct(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.0%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows",
        default=(
            "scripts/transfer_analysis_v4/results/predictions/"
            "peak_pck/rows_LOTO_motion.csv"
        ),
    )
    parser.add_argument("--min-src", type=int, default=4)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="scripts/transfer_analysis_v4/CEILING_REVISED.md",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.rows)
    required = {
        "actual", "g", "train_dataset", "context_id", "benchmark", "variant",
    }
    missing = required - set(data.columns)
    if missing:
        raise SystemExit(f"rows file is missing columns: {sorted(missing)}")

    data["replicate_group"] = data["variant"].astype(str).str.split("|").str[0]
    feature = context_feature_scores(data, args.min_src)
    pair_variant = pairwise_agreement(data, "variant", args.min_src)
    pair_group = pairwise_agreement(data, "replicate_group", args.min_src)
    consensus = held_variant_consensus(data, args.min_src, balance_groups=False)
    consensus_bal = held_variant_consensus(data, args.min_src, balance_groups=True)
    consensus_group = held_group_consensus(data, args.min_src)

    feature_mean = float(feature["feature_rho"].mean())
    consensus_mean = float(consensus["consensus_rho"].mean())
    consensus_bal_mean = float(consensus_bal["consensus_rho"].mean())
    matched_feature = float(consensus["feature_rho"].mean())
    matched_feature_bal = float(consensus_bal["feature_rho"].mean())
    fraction = matched_feature / consensus_mean if consensus_mean > 0 else np.nan
    fraction_bal = (
        matched_feature_bal / consensus_bal_mean
        if consensus_bal_mean > 0 else np.nan
    )
    consensus_group_mean = float(consensus_group["consensus_rho"].mean())
    feature_group_mean = float(consensus_group["feature_rho"].mean())
    fraction_group = (
        feature_group_mean / consensus_group_mean
        if consensus_group_mean > 0
        else np.nan
    )

    by_benchmark = (
        feature.groupby("benchmark")["feature_rho"].mean().rename("feature_rho")
        .to_frame()
        .join(
            pair_variant.set_index("benchmark")["pairwise_rho"]
            .rename("pairwise_variant_rho"),
            how="outer",
        )
        .join(
            pair_group.set_index("benchmark")["pairwise_rho"]
            .rename("pairwise_group_rho"),
            how="outer",
        )
        .join(
            consensus.groupby("benchmark")["consensus_rho"].mean()
            .rename("consensus_rho"),
            how="outer",
        )
        .join(
            consensus_bal.groupby("benchmark")["consensus_rho"].mean()
            .rename("balanced_consensus_rho"),
            how="outer",
        )
        .join(
            consensus_group.groupby("benchmark")["consensus_rho"].mean()
            .rename("held_arch_consensus_rho"),
            how="outer",
        )
        .join(
            consensus_group.groupby("benchmark")["feature_rho"].mean()
            .rename("held_arch_feature_rho"),
            how="outer",
        )
        .reset_index()
    )
    by_benchmark["feature_over_consensus"] = (
        by_benchmark["feature_rho"] / by_benchmark["consensus_rho"]
    )
    intervals = benchmark_bootstrap(
        by_benchmark.drop(columns=["feature_over_consensus"]),
        args.n_boot,
        args.seed,
    )

    pair_r = float(pair_variant["pairwise_rho"].mean())
    median_reps = int(round(pair_variant["n_replicates"].median()))
    single_ceiling = np.sqrt(max(pair_r, 0.0))
    aggregate_reliability = (
        median_reps * pair_r / (1 + (median_reps - 1) * pair_r)
        if pair_r > 0
        else np.nan
    )
    aggregate_ceiling = (
        np.sqrt(aggregate_reliability)
        if np.isfinite(aggregate_reliability)
        else np.nan
    )
    source_share = source_main_effect_share(data)

    def ci(name: str) -> str:
        bounds = intervals.get(name)
        return "" if bounds is None else f" [{bounds[0]:+.3f}, {bounds[1]:+.3f}]"

    lines = [
        "# Empirical Predictability References",
        "",
        f"Input: `{args.rows}`",
        "",
        "## Camera-ready recommendation",
        "",
        "Do **not** report a hard ceiling or say that the predictor reaches a fixed "
        "percentage of all possible feature performance. The data do not identify "
        "that quantity. Report the held-variant consensus as an **empirical oracle "
        "reference**: it predicts one model variant's source ranking from the source "
        "rankings produced by the other variants, without using the held variant's "
        "outcomes.",
        "",
        f"- Motion feature predictor: **rho = {feature_mean:+.3f}** mean within context.",
        f"- Held-variant rank consensus: **rho = {consensus_mean:+.3f}**"
        f"{ci('consensus_rho')}.",
        f"- Motion on the same eligible contexts: **rho = {matched_feature:+.3f}**, "
        f"or **{pct(fraction)} of this empirical reference**.",
        f"- Architecture-balanced consensus sensitivity: "
        f"**rho = {consensus_bal_mean:+.3f}**{ci('balanced_consensus_rho')}; "
        f"motion is **{pct(fraction_bal)}** of that reference.",
        f"- Architecture-aggregated holdout sensitivity: motion "
        f"**rho = {feature_group_mean:+.3f}** versus consensus "
        f"**rho = {consensus_group_mean:+.3f}**"
        f"{ci('held_arch_consensus_rho')} ({pct(fraction_group)}).",
        "",
        "The percentage is descriptive, not a theorem: the consensus itself can be "
        "beaten by features that predict stable structure shared across variants, "
        "and it changes as architectures or training variants are added.",
        "",
        "## Descriptive checks",
        "",
        f"- Pairwise inter-variant ranking agreement: **rho = "
        f"{pair_variant['pairwise_rho'].mean():+.3f}**"
        f"{ci('pairwise_variant_rho')}. This is reproducibility, not a ceiling.",
        f"- Pairwise architecture-family agreement: **rho = "
        f"{pair_group['pairwise_rho'].mean():+.3f}**"
        f"{ci('pairwise_group_rho')}.",
        f"- Generic source main-effect share: **{source_share:.1%}** of "
        "within-context squared variation. This is a variance decomposition, not a "
        "rank-correlation ceiling.",
        "",
        "## Assumption-heavy reliability sensitivity",
        "",
        f"Using mean pairwise Spearman rho ({pair_r:+.3f}) as if it were classical "
        f"parallel-test reliability gives sqrt(rho) = **{single_ceiling:.3f}** for "
        "one noisy variant. A Spearman-Brown aggregation over the median "
        f"{median_reps} variants gives reliability {aggregate_reliability:.3f} and "
        f"sqrt(reliability) = **{aggregate_ceiling:.3f}**. These are appendix-only "
        "sensitivities because variants are not independent parallel measurements "
        "and Spearman correlation is not the classical reliability coefficient.",
        "",
        "## Per-benchmark results",
        "",
        "| benchmark | motion rho | pairwise variants | pairwise arch | "
        "held-variant consensus | arch-balanced consensus | held-arch consensus |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_benchmark.sort_values("benchmark").itertuples():
        lines.append(
            f"| {row.benchmark} | {fmt(row.feature_rho)} | "
            f"{fmt(row.pairwise_variant_rho)} | {fmt(row.pairwise_group_rho)} | "
            f"{fmt(row.consensus_rho)} | {fmt(row.balanced_consensus_rho)} | "
            f"{fmt(row.held_arch_consensus_rho)} |"
        )
    lines.extend([
        "",
        "Bootstrap intervals resample the 10 benchmark identities and therefore "
        "describe variation across this benchmark panel. They do not establish "
        "generalization to arbitrary future architectures or benchmarks.",
        "",
    ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    by_benchmark.to_csv(out.with_suffix(".csv"), index=False)
    consensus.to_csv(out.with_name(f"{out.stem}_held_variant.csv"), index=False)
    consensus_bal.to_csv(
        out.with_name(f"{out.stem}_held_variant_arch_balanced.csv"),
        index=False,
    )
    consensus_group.to_csv(
        out.with_name(f"{out.stem}_held_architecture.csv"),
        index=False,
    )
    print("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
