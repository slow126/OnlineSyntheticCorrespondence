"""Standalone prototype for the *triangle prior* idea on LOTO / LOBO.

Goal
----
Test whether conditioning neighbor borrowing on benchmark relevance (the
"triangle") beats the current pure axis-IDW prior, on both axes:

  * TRAIN side (drives LOTO): predict perf(i, k) by borrowing perf(n, k) from
    known training datasets n.  Triangle reweights neighbor n by how relevant
    n is to benchmark k.
        w(n) = sim_train(i->n) * sim_pair(n->k) ** beta
  * EVAL side (drives LOBO): predict perf(i, k) by borrowing perf(i, e) from
    known benchmarks e.  Triangle reweights neighbor e by how relevant the
    candidate train i is to benchmark e.
        w(e) = sim_eval(k->e) * sim_pair(i->e) ** beta

`sim_pair` is the train_eval distance already present in
pairwise_self_distances.csv.  The full triangle's third term sim_pair(i->k)
is constant over neighbors, so it is used only as an optional gate (blend the
neighbor prior with the fallback prior by candidate i's own relevance to k).

This script evaluates the PRIOR ONLY (no residual ridge) so that the
comparison isolates the borrowing mechanism — exactly the thing that is
currently no better than uniform on LOTO.

The IDW weight transform mirrors RidgePairwiseDistModel._idw_stats in
run_experiments.py so numbers are comparable to the real pipeline.

Usage:
    python scripts/transfer_analysis_v3/triangle_prior_prototype.py
    python scripts/transfer_analysis_v3/triangle_prior_prototype.py --metric mean_nn_sym
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# --------------------------------------------------------------------------
# Distance / IDW machinery (mirrors run_experiments.py)
# --------------------------------------------------------------------------

# Reverse-direction column for asymmetric metrics, so lookup[(b,a)] is correct.
REVERSE_COL = {
    "mean_nn_sym": "mean_nn_sym",
    "mean_nn_a_to_b": "mean_nn_b_to_a",
    "mean_nn_b_to_a": "mean_nn_a_to_b",
    "flow_mmd_self": "flow_mmd_self",
    "flow_fid_self": "flow_fid_self",
    "flow_sliced_w2_self": "flow_sliced_w2_self",
}

# Metrics where higher = more similar (coverage). Everything else is a distance.
SIMILARITY_METRICS = {
    "a_covered_by_b_eps1px", "b_covered_by_a_eps1px", "sym_eps1px",
    "a_covered_by_b_eps4px", "b_covered_by_a_eps4px", "sym_eps4px",
    "a_covered_by_b_eps16px", "b_covered_by_a_eps16px", "sym_eps16px",
}


def build_lookup(dist_df: pd.DataFrame, space: str, pair_type: str,
                 metric_col: str) -> dict[tuple[str, str], float]:
    """{(a, b): value} for one (space, pair_type), both directions resolved."""
    rev_col = REVERSE_COL.get(metric_col, metric_col)
    grp = dist_df[(dist_df["space"] == space) & (dist_df["pair_type"] == pair_type)]
    lk: dict[tuple[str, str], float] = {}
    for _, row in grp.iterrows():
        a, b = row["dataset_a"], row["dataset_b"]
        direct = float(row[metric_col]) if metric_col in row.index and pd.notna(row[metric_col]) else np.nan
        lk[(a, b)] = direct
        if rev_col == metric_col:
            lk[(b, a)] = direct
        else:
            rev = float(row[rev_col]) if rev_col in row.index and pd.notna(row[rev_col]) else np.nan
            lk[(b, a)] = rev
    return lk


def idw_weights(dists: np.ndarray, is_similarity: bool, mode: str) -> np.ndarray:
    """Neighbor weights from a distance/similarity vector — matches _idw_stats."""
    if mode == "uniform":
        return np.ones_like(dists, dtype=np.float64)
    if is_similarity:
        return np.maximum(dists, 1e-8)
    # distance: lower = closer. shift so min = 1e-3, weight = 1/shifted.
    shifted = dists - dists.min() + 1e-3
    return 1.0 / shifted


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class TriangleConfig:
    name: str
    base_mode: str = "idw"        # "idw" | "uniform"
    base_metric: str = "mean_nn_sym"   # train_train / eval_eval metric
    relevance: bool = False       # multiply by sim_pair(neighbor relevance)?
    rel_metric: str = "mean_nn_sym"    # train_eval metric for the triangle term
    beta: float = 1.0             # exponent on relevance weight
    gate: bool = False            # blend with fallback by candidate's own relevance


# --------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------

class TriangleData:
    def __init__(self, table: pd.DataFrame, dist_df: pd.DataFrame, space: str):
        self.space = space
        # context_id encodes (benchmark, model_variant). Map both directions.
        self.ctx_bench: dict[str, str] = {}
        self.ctx_variant: dict[str, tuple] = {}
        for cid, grp in table.groupby("context_id"):
            self.ctx_bench[cid] = grp["benchmark"].iloc[0]
            v = grp.iloc[0]
            self.ctx_variant[cid] = (
                str(v.get("model_family", "")), str(v.get("pretrained", "")),
                str(v.get("freeze", "")),
            )
        # (variant, benchmark) -> context_id, for finding sibling contexts.
        self.var_bench_to_ctx: dict[tuple, str] = {
            (self.ctx_variant[c], self.ctx_bench[c]): c for c in self.ctx_bench
        }
        # perf[(train_dataset, context_id)] = mean auc_normalized
        perf_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
        for _, r in table.iterrows():
            if pd.notna(r["auc_normalized"]):
                perf_vals[(r["train_dataset"], r["context_id"])].append(float(r["auc_normalized"]))
        self.perf = {k: float(np.mean(v)) for k, v in perf_vals.items()}

        self.trains = sorted(table["train_dataset"].unique())
        self.benchmarks = sorted(table["benchmark"].unique())
        self.contexts = sorted(table["context_id"].unique())

        # Lookups are cached per (pair_type, metric) so each config can request
        # its own base/relevance metric without rebuilding the data object.
        self._dist_df = dist_df
        self._lk_cache: dict[tuple[str, str], dict] = {}

        # fallback priors
        bench_vals: dict[str, list[float]] = defaultdict(list)
        train_vals: dict[str, list[float]] = defaultdict(list)
        for (td, cid), v in self.perf.items():
            bench_vals[self.ctx_bench[cid]].append(v)
            train_vals[td].append(v)
        self.eval_mean = {b: float(np.mean(v)) for b, v in bench_vals.items()}
        self.train_mean = {t: float(np.mean(v)) for t, v in train_vals.items()}
        self.global_mean = float(np.mean(list(self.perf.values())))

    def lk(self, pair_type: str, metric: str) -> dict[tuple[str, str], float]:
        key = (pair_type, metric)
        if key not in self._lk_cache:
            self._lk_cache[key] = build_lookup(self._dist_df, self.space, pair_type, metric)
        return self._lk_cache[key]

    # ---- prior builders -------------------------------------------------

    def _weighted(self, base_d, rel_d, perfs, cfg: TriangleConfig) -> float | None:
        if not perfs:
            return None
        base_is_sim = cfg.base_metric in SIMILARITY_METRICS
        rel_is_sim = cfg.rel_metric in SIMILARITY_METRICS
        base_d = np.asarray(base_d, dtype=np.float64)
        perfs = np.asarray(perfs, dtype=np.float64)
        w = idw_weights(base_d, base_is_sim, cfg.base_mode)
        if cfg.relevance:
            rel_d = np.asarray(rel_d, dtype=np.float64)
            rw = idw_weights(rel_d, rel_is_sim, "idw")
            # normalise to unit mean so beta controls relative emphasis cleanly
            rw = rw / rw.mean() if rw.mean() > 0 else rw
            w = w * np.power(np.maximum(rw, 1e-12), cfg.beta)
        tot = w.sum()
        return float((w * perfs).sum() / tot) if tot > 0 else float(perfs.mean())

    def _gate_conf(self, cand_rel: float, fold_rels: list[float],
                   is_sim: bool) -> float:
        """Confidence in [0,1] from candidate's own relevance vs fold neighbors."""
        if not np.isfinite(cand_rel) or not fold_rels:
            return 1.0
        fold = np.asarray([r for r in fold_rels if np.isfinite(r)], dtype=np.float64)
        if fold.size == 0:
            return 1.0
        all_vals = np.append(fold, cand_rel)
        cand_w = idw_weights(all_vals, is_sim, "idw")[-1]
        ref = np.median(idw_weights(all_vals, is_sim, "idw"))
        return float(np.clip(cand_w / (cand_w + ref + 1e-12), 0.0, 1.0))

    def predict_train_side(self, i: str, ctx: str, fold_trains: list[str],
                           cfg: TriangleConfig) -> float:
        """LOTO: borrow perf(n, ctx) over training-dataset neighbors n of i."""
        k = self.ctx_bench[ctx]
        tt = self.lk("train_train", cfg.base_metric)
        te = self.lk("train_eval", cfg.rel_metric)
        rel_is_sim = cfg.rel_metric in SIMILARITY_METRICS
        base_d, rel_d, perfs = [], [], []
        for n in fold_trains:
            if n == i:
                continue
            p = self.perf.get((n, ctx))
            if p is None or not np.isfinite(p):
                continue
            bd = tt.get((i, n))
            if bd is None or not np.isfinite(bd):
                continue
            rd = te.get((n, k))
            if cfg.relevance and (rd is None or not np.isfinite(rd)):
                continue
            base_d.append(bd)
            rel_d.append(rd if rd is not None else np.nan)
            perfs.append(p)
        prior = self._weighted(base_d, rel_d, perfs, cfg)
        fallback = self.eval_mean.get(k, self.global_mean)
        if prior is None:
            return fallback
        if cfg.gate:
            cand_rel = te.get((i, k), np.nan)
            conf = self._gate_conf(cand_rel, rel_d, rel_is_sim)
            return conf * prior + (1.0 - conf) * fallback
        return prior

    def predict_eval_side(self, i: str, ctx: str, fold_benchmarks: list[str],
                          cfg: TriangleConfig) -> float:
        """LOBO: borrow perf(i, sibling-ctx) over benchmark neighbors e of k."""
        k = self.ctx_bench[ctx]
        var = self.ctx_variant[ctx]
        ee = self.lk("eval_eval", cfg.base_metric)
        te = self.lk("train_eval", cfg.rel_metric)
        rel_is_sim = cfg.rel_metric in SIMILARITY_METRICS
        base_d, rel_d, perfs = [], [], []
        for e in fold_benchmarks:
            if e == k:
                continue
            sib = self.var_bench_to_ctx.get((var, e))
            if sib is None:
                continue
            p = self.perf.get((i, sib))
            if p is None or not np.isfinite(p):
                continue
            bd = ee.get((k, e))
            if bd is None or not np.isfinite(bd):
                continue
            rd = te.get((i, e))
            if cfg.relevance and (rd is None or not np.isfinite(rd)):
                continue
            base_d.append(bd)
            rel_d.append(rd if rd is not None else np.nan)
            perfs.append(p)
        prior = self._weighted(base_d, rel_d, perfs, cfg)
        fallback = self.train_mean.get(i, self.global_mean)
        if prior is None:
            return fallback
        if cfg.gate:
            cand_rel = te.get((i, k), np.nan)
            conf = self._gate_conf(cand_rel, rel_d, rel_is_sim)
            return conf * prior + (1.0 - conf) * fallback
        return prior


# --------------------------------------------------------------------------
# Fold runners
# --------------------------------------------------------------------------

def run_loto(data: TriangleData, cfg: TriangleConfig) -> pd.DataFrame:
    rows = []
    for i in data.trains:
        fold_trains = [t for t in data.trains if t != i]
        for ctx in data.contexts:
            if (i, ctx) not in data.perf:
                continue
            pred = data.predict_train_side(i, ctx, fold_trains, cfg)
            rows.append((i, ctx, data.ctx_bench[ctx], data.perf[(i, ctx)], pred))
    return pd.DataFrame(rows, columns=["train_dataset", "context_id", "benchmark",
                                       "actual", "pred"])


def run_own_distance(data: TriangleData, side: str,
                     metric: str = "mean_nn_sym") -> pd.DataFrame:
    """Rank candidates within each context by their OWN (i->k) train_eval distance.

    This is not a borrowing prior — it is the gate term sim_pair(i->k) used
    directly as a within-context ranking score.  For LOTO it ranks training
    datasets within a benchmark context; the score is -distance (closer = higher
    predicted AUC).  No fold logic is needed because each row's own distance is
    not leaked by holding out other rows.  MAE is not meaningful for a raw
    feature score, so only the ranking metrics should be read.
    """
    te = data.lk("train_eval", metric)
    is_sim = metric in SIMILARITY_METRICS
    rows = []
    for (i, ctx), actual in data.perf.items():
        k = data.ctx_bench[ctx]
        d = te.get((i, k))
        if d is None or not np.isfinite(d):
            continue
        score_ = d if is_sim else -d   # closer => higher predicted rank
        rows.append((i, ctx, k, actual, score_))
    return pd.DataFrame(rows, columns=["train_dataset", "context_id", "benchmark",
                                       "actual", "pred"])


def run_lobo(data: TriangleData, cfg: TriangleConfig) -> pd.DataFrame:
    rows = []
    for k in data.benchmarks:
        fold_benchmarks = [b for b in data.benchmarks if b != k]
        held_ctxs = [c for c in data.contexts if data.ctx_bench[c] == k]
        for ctx in held_ctxs:
            for i in data.trains:
                if (i, ctx) not in data.perf:
                    continue
                pred = data.predict_eval_side(i, ctx, fold_benchmarks, cfg)
                rows.append((i, ctx, k, data.perf[(i, ctx)], pred))
    return pd.DataFrame(rows, columns=["train_dataset", "context_id", "benchmark",
                                       "actual", "pred"])


# --------------------------------------------------------------------------
# Metrics (mirror evaluate_context: rank within context_id)
# --------------------------------------------------------------------------

def within_context_spearman(df: pd.DataFrame) -> float:
    scores = []
    for _, grp in df.groupby("context_id"):
        if grp["train_dataset"].nunique() < 3:
            continue
        rho = spearmanr(grp["actual"], grp["pred"]).statistic
        if not np.isnan(rho):
            scores.append(rho)
    return float(np.nanmean(scores)) if scores else float("nan")


def context_centered_spearman(df: pd.DataFrame) -> float:
    c = df.copy()
    c["ar"] = c["actual"] - c.groupby("context_id")["actual"].transform("mean")
    c["pr"] = c["pred"] - c.groupby("context_id")["pred"].transform("mean")
    rho = spearmanr(c["pr"], c["ar"]).statistic
    return float(rho) if not np.isnan(rho) else float("nan")


def context_mae(df: pd.DataFrame) -> float:
    maes = df.groupby("context_id").apply(
        lambda g: np.mean(np.abs(g["actual"] - g["pred"])), include_groups=False)
    return float(maes.mean())


def train_mean_auc_error(df: pd.DataFrame) -> float:
    err = df.groupby("train_dataset").apply(
        lambda g: abs(g["actual"].mean() - g["pred"].mean()), include_groups=False)
    return float(err.mean())


def score(df: pd.DataFrame) -> dict:
    return {
        "MAE": context_mae(df),
        "ctx_spearman": within_context_spearman(df),
        "centered_spearman": context_centered_spearman(df),
        "train_mean_err": train_mean_auc_error(df),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def make_configs(metric: str) -> list[TriangleConfig]:
    directed = "mean_nn_a_to_b" if metric == "mean_nn_sym" else metric
    return [
        TriangleConfig("uniform", base_mode="uniform"),
        TriangleConfig("idw (symmetric)", base_mode="idw", base_metric=metric),
        TriangleConfig("idw (directed)", base_mode="idw", base_metric=directed),
        TriangleConfig("triangle (idw * rel)", base_mode="idw", base_metric=metric,
                       relevance=True, rel_metric=metric, beta=1.0),
        TriangleConfig("triangle beta=0.5", base_mode="idw", base_metric=metric,
                       relevance=True, rel_metric=metric, beta=0.5),
        TriangleConfig("gate only (i->k)", base_mode="idw", base_metric=metric,
                       relevance=False, rel_metric=metric, gate=True),
        TriangleConfig("triangle + gate", base_mode="idw", base_metric=metric,
                       relevance=True, rel_metric=metric, beta=1.0, gate=True),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--space", default="flow")
    ap.add_argument("--metric", default="mean_nn_sym",
                    help="base + relevance metric (train_train/eval_eval/train_eval)")
    args = ap.parse_args()

    root = Path(".").resolve()
    table = pd.read_csv(root / args.table)
    dist_df = pd.read_csv(root / args.dist)

    data = TriangleData(table, dist_df, args.space)
    print(f"space={args.space}  metric={args.metric}")
    print(f"trains={len(data.trains)}  benchmarks={len(data.benchmarks)}  "
          f"contexts={len(data.contexts)}  perf cells={len(data.perf)}")
    te = data.lk("train_eval", args.metric)
    n_te = sum(1 for v in te.values() if np.isfinite(v))
    print(f"train_eval relevance pairs available: {n_te}\n")

    configs = make_configs(args.metric)
    for split, runner in [("LOTO (train-side)", run_loto), ("LOBO (eval-side)", run_lobo)]:
        print(f"================ {split} ================")
        print(f"{'config':<24}{'MAE':>8}{'ctx_rho':>10}{'cent_rho':>10}{'tr_err':>9}")
        for cfg in configs:
            # eval-side relevance/triangle only meaningful for LOBO; train-side for LOTO.
            df = runner(data, cfg)
            m = score(df)
            print(f"{cfg.name:<24}{m['MAE']:>8.3f}{m['ctx_spearman']:>10.3f}"
                  f"{m['centered_spearman']:>10.3f}{m['train_mean_err']:>9.3f}")
        # Reference: rank by candidate's OWN (i->k) distance — the gate term as a
        # direct ranking score. MAE omitted (raw feature, not calibrated).
        own = score(run_own_distance(data, split.split()[0].lower(), args.metric))
        print(f"{'own (i->k) rank [ref]':<24}{'   --':>8}{own['ctx_spearman']:>10.3f}"
              f"{own['centered_spearman']:>10.3f}{'   --':>9}")
        print()


if __name__ == "__main__":
    main()
