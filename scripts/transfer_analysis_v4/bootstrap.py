"""Entity-resampled bootstrap CIs for the headline ctx_rho / cent_rho / abs_r,
broken out by target × head.

Walks `predictions/<target>/` for each target subdirectory; for each per-row
prediction CSV computes bootstrap CIs treating the GENERALIZATION axis as the
entity (sources for LOTO, benchmarks for LOBO, source×benchmark pairs for
JOINT). Reports metrics for each of the three score columns:

    g        — ridge prediction
    g_cal    — ridge × per-context gain calibration
    g_rank   — pairwise RankNet score

Outputs:
    summary.csv        — long-form: target × split × family × label × head
                         with point estimates and 95% CIs
    bootstrap_gap.csv  — motion − appearance gap CIs per (target, split, head)

Run:
    python scripts/transfer_analysis_v4/bootstrap.py [--n-boot 2000]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from experiments import family_of   # generator-family grouping (FAMILY_MAP)


HEADS = ["g", "g_zridge", "g_rank", "g_gbm"]


# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    if (a.max() - a.min()) < 1e-12 or (b.max() - b.min()) < 1e-12:
        return float("nan")
    ra = rankdata(a); rb = rankdata(b)
    da = ra - ra.mean(); db = rb - rb.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum())
    if denom < 1e-12:
        return float("nan")
    return float((da * db).sum() / denom)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    av, bv = a[m], b[m]
    if (av.max() - av.min()) < 1e-12 or (bv.max() - bv.min()) < 1e-12:
        return float("nan")
    da = av - av.mean(); db = bv - bv.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum())
    if denom < 1e-12:
        return float("nan")
    return float((da * db).sum() / denom)


# ---------------------------------------------------------------------------
class Prepared:
    """Numpy-array view of a predictions CSV."""

    def __init__(self, df: pd.DataFrame, split: str, cluster: bool = False):
        df = df.reset_index(drop=True)
        self.split = split
        self.cluster = cluster
        self.actual = df["actual"].to_numpy(float)
        self.g = df["g"].to_numpy(float)
        self.g_zridge = df["g_zridge"].to_numpy(float) if "g_zridge" in df.columns else self.g.copy()
        self.g_rank = df["g_rank"].to_numpy(float) if "g_rank" in df.columns else np.zeros_like(self.g)
        self.g_gbm = df["g_gbm"].to_numpy(float) if "g_gbm" in df.columns else np.zeros_like(self.g)
        self.L = df["L"].to_numpy(float)
        self.train = df["train_dataset"].to_numpy()
        self.bench = df["benchmark"].to_numpy()
        self.ctx = df["context_id"].to_numpy()
        # The resampling unit. cluster=True collapses correlated same-generator
        # sources into one family so the bootstrap counts ~5 effective sources,
        # not 11 — honest CIs under within-family correlation. Only the SOURCE
        # axis is clustered (LOTO/JOINT); benchmarks (LOBO) are left as-is.
        src = (np.array([family_of(t) for t in self.train]) if cluster
               else self.train)
        if split == "LOTO":
            ent = src
        elif split == "LOBO":
            ent = self.bench
        else:
            ent = np.char.add(np.char.add(src.astype(str), "|"),
                              self.bench.astype(str))
        self.entity = ent
        self.entities = np.array(sorted(set(ent.tolist())))
        self.entity_to_rows = {e: np.where(ent == e)[0] for e in self.entities}

    def score(self, head: str) -> np.ndarray:
        return getattr(self, head)

    def slice(self, rows: np.ndarray, head: str):
        return (self.actual[rows], self.score(head)[rows], self.L[rows],
                self.ctx[rows], self.train[rows])

    def sample_rows(self, rng) -> np.ndarray:
        picked = rng.choice(len(self.entities), size=len(self.entities), replace=True)
        parts = [self.entity_to_rows[self.entities[i]] for i in picked]
        return np.concatenate(parts)


def _cent_rho(actual, pred, ctx):
    a_resid = np.empty_like(actual); p_resid = np.empty_like(pred)
    for c in np.unique(ctx):
        m = ctx == c
        a_resid[m] = actual[m] - actual[m].mean()
        p_resid[m] = pred[m] - pred[m].mean()
    return _spearman(p_resid, a_resid)


def _metrics_from_rows(actual, g, L, ctx, train) -> dict:
    Lg = L + g
    uniq = np.unique(ctx)
    rho_g, rho_L, rho_Lg = [], [], []
    for c in uniq:
        m = ctx == c
        if np.unique(train[m]).size < 3:
            continue
        a = actual[m]; gm = g[m]; Lm = L[m]; Lgm = Lg[m]
        rho_g.append(_spearman(a, gm))
        rho_L.append(_spearman(a, Lm))
        rho_Lg.append(_spearman(a, Lgm))
    return dict(
        ctx_rho_g=float(np.nanmean(rho_g)) if rho_g else float("nan"),
        ctx_rho_L=float(np.nanmean(rho_L)) if rho_L else float("nan"),
        ctx_rho_Lg=float(np.nanmean(rho_Lg)) if rho_Lg else float("nan"),
        cent_rho_g=_cent_rho(actual, g, ctx),
        abs_r_Lg=_pearson(actual, Lg),
    )


def bootstrap(prep: Prepared, head: str, n_boot: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    keys = ("ctx_rho_g", "cent_rho_g", "ctx_rho_L", "ctx_rho_Lg", "abs_r_Lg")
    samples = {k: np.full(n_boot, np.nan) for k in keys}
    for b in range(n_boot):
        rows = prep.sample_rows(rng)
        m = _metrics_from_rows(*prep.slice(rows, head))
        for k in keys:
            samples[k][b] = m[k]
    pt = _metrics_from_rows(prep.actual, prep.score(head), prep.L,
                            prep.ctx, prep.train)
    out = {}
    for k in keys:
        out[k] = pt[k]
        s = samples[k][np.isfinite(samples[k])]
        if s.size:
            lo, hi = np.quantile(s, [0.025, 0.975])
            out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
        else:
            out[f"{k}_lo"] = out[f"{k}_hi"] = float("nan")
    return out


def bootstrap_gap(prep_m: Prepared, prep_a: Prepared, head: str,
                  n_boot: int, seed: int) -> dict:
    """Paired bootstrap: same entity sample for both motion and appearance."""
    assert prep_m.split == prep_a.split
    assert np.array_equal(prep_m.entities, prep_a.entities)
    rng = np.random.default_rng(seed)
    keys = ("ctx_rho_g", "cent_rho_g", "abs_r_Lg")
    diffs = {k: np.full(n_boot, np.nan) for k in keys}
    for b in range(n_boot):
        picked = rng.choice(len(prep_m.entities), size=len(prep_m.entities),
                            replace=True)
        ents = prep_m.entities[picked]
        rows_m = np.concatenate([prep_m.entity_to_rows[e] for e in ents])
        rows_a = np.concatenate([prep_a.entity_to_rows[e] for e in ents])
        mm = _metrics_from_rows(*prep_m.slice(rows_m, head))
        ma = _metrics_from_rows(*prep_a.slice(rows_a, head))
        for k in keys:
            diffs[k][b] = mm[k] - ma[k]
    pt_m = _metrics_from_rows(prep_m.actual, prep_m.score(head), prep_m.L,
                              prep_m.ctx, prep_m.train)
    pt_a = _metrics_from_rows(prep_a.actual, prep_a.score(head), prep_a.L,
                              prep_a.ctx, prep_a.train)
    out = {}
    for k in keys:
        out[f"{k}_gap"] = pt_m[k] - pt_a[k]
        s = diffs[k][np.isfinite(diffs[k])]
        if s.size:
            lo, hi = np.quantile(s, [0.025, 0.975])
            out[f"{k}_gap_lo"], out[f"{k}_gap_hi"] = float(lo), float(hi)
            out[f"{k}_gap_p_gt_0"] = float(np.mean(s > 0))
        else:
            out[f"{k}_gap_lo"] = out[f"{k}_gap_hi"] = out[f"{k}_gap_p_gt_0"] = float("nan")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="scripts/transfer_analysis_v4/results")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", action="store_true",
                    help="Cluster bootstrap (Tier 3): resample at the GENERATOR "
                         "FAMILY level (~5) instead of the source level (11) so CIs "
                         "are honest under within-family correlation. Affects LOTO "
                         "and the source axis of JOINT. Writes *_cluster.csv.")
    ap.add_argument("--families", nargs="+", default=None, metavar="FAM",
                    help="Restrict the per-cell bootstrap to these feature families "
                         "(default: all ~20). The robustness pipeline only consumes "
                         "'motion' and 'appearance' (+ their paired gap), so passing "
                         "'--families motion appearance' is ~6-8x faster with no loss.")
    args = ap.parse_args()

    root = Path(".").resolve()
    out_dir = root / args.results
    preds_root = out_dir / "predictions"

    if not preds_root.exists():
        raise SystemExit(f"no predictions found at {preds_root}")
    targets = [p.name for p in preds_root.iterdir() if p.is_dir()]
    if not targets:
        raise SystemExit(f"no per-target subdirs under {preds_root}")
    tag = "_cluster" if args.cluster else ""
    print(f"bootstrap  targets={targets}  n_boot={args.n_boot}"
          f"{'  [CLUSTER: resampling generator families]' if args.cluster else ''}\n")

    splits = ["LOTO", "LOBO", "JOINT"]
    families = ["motion", "appearance", "both", "random",
                "density", "motion_density",
                "size", "supervision_density",
                "motion_size", "motion_supdensity",
                "motion_km", "motion_sym", "motion_mmd", "motion_fid", "motion_w2",
                "appearance_mmd", "appearance_nullk",
                "appearance_sym", "appearance_fid", "appearance_w2"]
    if args.families:
        keep = set(args.families)
        unknown = keep - set(families)
        if unknown:
            raise SystemExit(f"--families: unknown {sorted(unknown)}; valid: {families}")
        families = [f for f in families if f in keep]
        print(f"  [--families filter: {families}]")
    labels = [("main", ""), ("shuffle", "_shuffle"), ("uniform_level", "_uniformL")]

    # 1. Per-cell CIs --------------------------------------------------------
    rows = []
    for target in targets:
        pred_dir = preds_root / target
        for split in splits:
            for fam in families:
                for label, suffix in labels:
                    path = pred_dir / f"rows_{split}_{fam}{suffix}.csv"
                    if not path.exists():
                        continue
                    df = pd.read_csv(path)
                    prep = Prepared(df, split, cluster=args.cluster)
                    for head in HEADS:
                        if head not in df.columns and head != "g":
                            continue
                        seed = (args.seed
                                + abs(hash((target, split, fam, label, head))) % 10_000)
                        m = bootstrap(prep, head, args.n_boot, seed)
                        m.update(target=target, split=split, family=fam,
                                 label=label, head=head, n_rows=len(df))
                        rows.append(m)
                        if label == "main":
                            print(f"  {target:<14} {split:<6} {fam:<11} "
                                  f"{head:<7}  "
                                  f"ctx ρ_g = {m['ctx_rho_g']:+.3f} "
                                  f"[{m['ctx_rho_g_lo']:+.3f}, {m['ctx_rho_g_hi']:+.3f}]")

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / f"summary{tag}.csv", index=False)
    print(f"\nsummary -> {out_dir}/summary{tag}.csv  ({len(summary_df)} cells)")

    # 2. Motion − appearance gap CIs (paired) -------------------------------
    print("\nMotion − appearance gap (paired bootstrap):")
    gap_rows = []
    for target in targets:
        pred_dir = preds_root / target
        for split in splits:
            m_path = pred_dir / f"rows_{split}_motion.csv"
            a_path = pred_dir / f"rows_{split}_appearance.csv"
            if not (m_path.exists() and a_path.exists()):
                continue
            prep_m = Prepared(pd.read_csv(m_path), split, cluster=args.cluster)
            prep_a = Prepared(pd.read_csv(a_path), split, cluster=args.cluster)
            for head in HEADS:
                seed = (args.seed
                        + abs(hash(("gap", target, split, head))) % 10_000)
                g = bootstrap_gap(prep_m, prep_a, head, args.n_boot, seed)
                g.update(target=target, split=split, head=head,
                         n_boot=args.n_boot)
                gap_rows.append(g)
                if head == "g":
                    print(f"  {target:<14} {split:<6}  "
                          f"ctx ρ_g gap = {g['ctx_rho_g_gap']:+.3f} "
                          f"[{g['ctx_rho_g_gap_lo']:+.3f}, "
                          f"{g['ctx_rho_g_gap_hi']:+.3f}]  "
                          f"P(>0) = {g['ctx_rho_g_gap_p_gt_0']:.3f}")

    pd.DataFrame(gap_rows).to_csv(out_dir / f"bootstrap_gap{tag}.csv", index=False)
    print(f"\ngap -> {out_dir}/bootstrap_gap{tag}.csv")


if __name__ == "__main__":
    main()
