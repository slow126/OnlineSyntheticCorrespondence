"""Gap-stratified pairwise ranking accuracy + empirical ordering reproducibility.

Answers the reviewer criticism "0.70 pairwise accuracy is not very useful" by
decomposing it: pairwise decisions between sources whose true transfer gap is
large are the decisions that matter, and (hypothesis) the predictor gets those
right; the misses concentrate on near-tie pairs whose ordering is not even
reproducible across independently trained models.

Two outputs, both stratified by the true |peak_pck gap| of the pair:

1. PREDICTOR accuracy: within each context (benchmark|model|pretrained|freeze),
   for every source pair (i, j), is sign(g_i - g_j) == sign(actual_i - actual_j)?
   Read from a v4 predictions dir (rows_{split}_{family}.csv).

2. EMPIRICAL reproducibility: from the transfer table itself, for the same
   benchmark and the same source pair observed under two different training
   contexts (variants), do the two contexts agree on the ordering? Split into
   same-architecture pairs and cross-architecture pairs. This is the ceiling:
   a predictor cannot be expected to rank a pair more reliably than retraining
   does.

Usage:
    python scripts/transfer_analysis_v4/pairwise_gap_analysis.py \
        --rows-dir scripts/transfer_analysis_v4/results_glunet_clean/predictions/peak_pck \
        --out scripts/transfer_analysis_v4/results_glunet_clean/pairwise_gap_analysis.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transfer_analysis_v3"))
from transfer_predictor_prototype import variant_key  # noqa: E402

GAP_BINS = [0.0, 1.0, 2.0, 5.0, 10.0, np.inf]
GAP_LABELS = ["0-1", "1-2", "2-5", "5-10", ">10"]

PURE_TRAIN_DATASETS = [
    "flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
    "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
    "synthetic_random_flipping", "synthetic_small_zoom",
]


def bin_label(gap: float) -> str:
    idx = np.searchsorted(GAP_BINS, gap, side="right") - 1
    return GAP_LABELS[min(idx, len(GAP_LABELS) - 1)]


def predictor_pairs(rows: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """One record per within-context source pair."""
    recs = []
    for ctx, g in rows.groupby("context_id"):
        g = g.drop_duplicates("train_dataset")
        for a, b in itertools.combinations(g.itertuples(), 2):
            gap = abs(a.actual - b.actual)
            if gap == 0:
                continue
            pa, pb = getattr(a, pred_col), getattr(b, pred_col)
            correct = (pa - pb) * (a.actual - b.actual) > 0
            recs.append((ctx, a.train_dataset, b.train_dataset, gap,
                         bin_label(gap), bool(correct)))
    return pd.DataFrame(recs, columns=["context_id", "src_a", "src_b",
                                       "gap", "gap_bin", "correct"])


def empirical_pairs(table: pd.DataFrame, target: str) -> pd.DataFrame:
    """Ordering agreement between pairs of contexts on the same benchmark.

    For each benchmark, each pair of variants (v1, v2), each source pair seen
    in both: does v2 order the pair the same way v1 does? The gap is taken
    from the *reference* context (v1) and each unordered context pair yields
    two records (each side as reference) so the stratification is symmetric.
    """
    table = table.dropna(subset=[target])
    recs = []
    for bench, bt in table.groupby("benchmark"):
        variants = sorted(bt["variant"].unique())
        perf = {(r.variant, r.train_dataset): getattr(r, target)
                for r in bt.itertuples()}
        srcs_by_v = {v: sorted(bt.loc[bt["variant"] == v, "train_dataset"].unique())
                     for v in variants}
        for v1, v2 in itertools.combinations(variants, 2):
            arch1, arch2 = v1.split("|")[0], v2.split("|")[0]
            kind = "same_arch" if arch1 == arch2 else "cross_arch"
            shared = sorted(set(srcs_by_v[v1]) & set(srcs_by_v[v2]))
            for i, j in itertools.combinations(shared, 2):
                d1 = perf[(v1, i)] - perf[(v1, j)]
                d2 = perf[(v2, i)] - perf[(v2, j)]
                if d1 == 0 or d2 == 0:
                    continue
                agree = (d1 > 0) == (d2 > 0)
                for ref_gap in (abs(d1), abs(d2)):
                    recs.append((bench, v1, v2, kind, i, j, ref_gap,
                                 bin_label(ref_gap), bool(agree)))
    return pd.DataFrame(recs, columns=["benchmark", "v1", "v2", "kind",
                                       "src_a", "src_b", "gap", "gap_bin",
                                       "agree"])


def stratified(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    out = (df.groupby(group_cols + ["gap_bin"])[value_col]
             .agg(acc="mean", n="size").reset_index())
    tot = df.groupby(group_cols)[value_col].agg(acc="mean", n="size").reset_index()
    tot["gap_bin"] = "ALL"
    return pd.concat([out, tot], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True,
                    help="v4 predictions dir holding rows_{split}_{family}.csv")
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--families", nargs="+", default=["motion"])
    ap.add_argument("--splits", nargs="+", default=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--pred-col", default="g")
    ap.add_argument("--pure-only", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows_dir = Path(args.rows_dir)

    # --- 1. predictor accuracy by gap ---------------------------------------
    pred_parts = []
    for split in args.splits:
        for fam in args.families:
            f = rows_dir / f"rows_{split}_{fam}.csv"
            if not f.exists():
                print(f"  missing {f}, skipping")
                continue
            pairs = predictor_pairs(pd.read_csv(f), args.pred_col)
            pairs["split"] = split
            pairs["family"] = fam
            pred_parts.append(pairs)
    pred_pairs = pd.concat(pred_parts, ignore_index=True)
    pred_strat = stratified(pred_pairs, "correct", ["split", "family"])
    pred_strat["measure"] = "predictor_accuracy"

    # --- 2. empirical ordering reproducibility by gap -----------------------
    table = pd.read_csv(args.table)
    if args.pure_only:
        table = table[table["train_dataset"].isin(PURE_TRAIN_DATASETS)]
    table = table.copy()
    table["variant"] = table.apply(variant_key, axis=1)
    emp = empirical_pairs(table, args.target)
    emp_strat = stratified(emp, "agree", ["kind"])
    emp_strat["measure"] = "empirical_reproducibility"
    emp_strat["split"] = "n/a"
    emp_strat["family"] = emp_strat.pop("kind")

    out = pd.concat([pred_strat, emp_strat], ignore_index=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")

    bins_order = GAP_LABELS + ["ALL"]

    def show(df, title, key):
        print(f"\n== {title} ==")
        piv = df.pivot_table(index=key, columns="gap_bin", values="acc")
        npiv = df.pivot_table(index=key, columns="gap_bin", values="n")
        piv = piv.reindex(columns=[b for b in bins_order if b in piv.columns])
        for idx, r in piv.iterrows():
            cells = "  ".join(f"{b}:{r[b]:.3f}(n={int(npiv.loc[idx, b])})"
                              for b in piv.columns if np.isfinite(r[b]))
            print(f"  {idx}: {cells}")

    show(pred_strat, f"Predictor pairwise accuracy ({args.pred_col}) by |{args.target} gap|",
         ["split", "family"])
    show(emp_strat, "Empirical ordering reproducibility by |gap| (independent training contexts)",
         ["family"])


if __name__ == "__main__":
    main()
