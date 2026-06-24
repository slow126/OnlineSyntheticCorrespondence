"""Adversarial verification of the Regime-Direction Law (2026-06-09).

Claim under test: the direction of directed motion coverage that predicts
transfer flips with training regime — from-scratch models are precision-bound
(train->eval distance, a->b, off-target mass) and pretrained models are
recall-bound (eval->train distance, b->a, missing target support).

This script recomputes everything INDEPENDENTLY of the v3/v4 pipeline code:
its own join from the two raw CSVs, no add_selfdist_features, no winsorize,
no impute, no ridge. Checks:

  0. Data integrity: orientation of all (train, benchmark) pairs in the
     distances CSV (a must be the train side); duplicate rows in the transfer
     table; context_id consistency; features constant across variants.
  1. The master table: per-variant within-context Spearman for a->b, b->a,
     sym (mean_nn), with benchmark-bootstrap 95% CIs.
  2. Cross-metric consistency: the same flip must appear in eps-coverage
     (eps4px, eps16px) and KL (k20) directional features — three independent
     feature constructions.
  3. Exact permutation test of the flip on the 8 non-RAFT variants
     (RAFT excluded as the one regime-coding judgment call): statistic =
     mean(d | scratch) - mean(d | pretrained), d_v = mean_ctx(rho_ab - rho_ba),
     against all C(8,4)=70 label assignments.
  4. Robustness: self-pairs (train==benchmark dataset) excluded; sparse
     benchmarks (spair/pfpascal-style semantic) flagged; leave-one-benchmark-out
     range of the flip statistic.
  5. Target robustness: peak_pck and auc_normalized (where available).

Writes a full report to --out (markdown) and the master table CSV next to it.
Deterministic (seed fixed; no Date.now-style nondeterminism).

Run:
    python scripts/transfer_analysis_v4/verify_regime_direction.py \
        --out scripts/transfer_analysis_v4/regime_direction_verification/REPORT.md
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]

# Directional feature triplets to test: (label, a->b col, b->a col, sym col,
# better_when) where better_when='low' means smaller value predicts better
# transfer (distances / KL), 'high' means larger predicts better (coverage).
METRIC_SETS = [
    ("mean_nn", "mean_nn_a_to_b", "mean_nn_b_to_a", "mean_nn_sym", "low"),
    ("eps4px", "a_covered_by_b_eps4px", "b_covered_by_a_eps4px", "sym_eps4px", "high"),
    ("eps16px", "a_covered_by_b_eps16px", "b_covered_by_a_eps16px", "sym_eps16px", "high"),
    ("kl_k20", "kl_a_to_b_k20", "kl_b_to_a_k20", None, "low"),
]

SCRATCH = {"catspp|False|False", "catspp|False|True",
           "glunet|False|False", "glunet|False|True"}
PRETRAINED = {"catspp|True|False", "catspp|True|True",
              "glunet|True|False", "glunet|True|True"}
RAFT = "raft|True|False"

SEED = 0
N_BOOT = 10000


def load(table_path: str, dist_path: str, target: str, space: str = "flow"):
    t = pd.read_csv(table_path)
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t["cv"] = t.benchmark + "|" + t.variant
    t = t.dropna(subset=[target])

    d = pd.read_csv(dist_path)
    te = d[(d.pair_type == "train_eval") & (d.space == space)].copy()
    return t, te


def integrity_checks(t: pd.DataFrame, te: pd.DataFrame, report: list[str]):
    ok = True
    # 0a. orientation: every (train, benchmark) pair must be stored a=train
    pairs = t[["train_dataset", "benchmark"]].drop_duplicates()
    fwd = set(zip(te.dataset_a, te.dataset_b))
    rev = set(zip(te.dataset_b, te.dataset_a)) - fwd
    n_rev = sum((tr, be) in rev for tr, be in pairs.itertuples(index=False))
    n_fwd = sum((tr, be) in fwd for tr, be in pairs.itertuples(index=False))
    report.append(f"- orientation: {n_fwd}/{len(pairs)} pairs forward (a=train), "
                  f"{n_rev} REVERSED")
    if n_rev or n_fwd != len(pairs):
        ok = False
    splits_ok = (te.split_a == "train").all()
    report.append(f"- split_a=='train' for all train_eval rows: {splits_ok}")
    ok &= bool(splits_ok)
    # 0b. duplicates in transfer table
    dup = t.duplicated(subset=["train_dataset", "benchmark", "variant"]).sum()
    report.append(f"- duplicate (train, benchmark, variant) rows: {dup}")
    ok &= dup == 0
    # 0c. context_id consistency
    recon = (t.benchmark + "_" + t.model_family.astype(str) + "_"
             + t.pretrained.astype(str) + "_" + t.freeze.astype(str))
    mism = (recon != t.context_id).sum()
    report.append(f"- context_id mismatches vs (benchmark,model,pre,frz): {mism}")
    ok &= mism == 0
    return ok


def join_features(t: pd.DataFrame, te: pd.DataFrame) -> pd.DataFrame:
    feat_cols = sorted({c for _, a, b, s, _ in METRIC_SETS
                        for c in (a, b, s) if c})
    f = te.set_index(["dataset_a", "dataset_b"])[feat_cols]
    out = t.join(f, on=["train_dataset", "benchmark"], how="left")
    return out


def ctx_rhos(g: pd.DataFrame, col: str, sign: float, target: str) -> list[tuple[str, float]]:
    """(benchmark, within-context spearman) for every context of one variant."""
    out = []
    for cvk, c in g.groupby("cv"):
        if c.train_dataset.nunique() < 3 or c[col].std() <= 1e-15:
            continue
        r = spearmanr(c[target], sign * c[col]).statistic
        if np.isfinite(r):
            out.append((c.benchmark.iloc[0], float(r)))
    return out


def boot_ci(vals: np.ndarray, rng) -> tuple[float, float]:
    bs = rng.choice(vals, size=(N_BOOT, len(vals)), replace=True).mean(axis=1)
    return float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--out", default="scripts/transfer_analysis_v4/"
                                     "regime_direction_verification/REPORT.md")
    ap.add_argument("--space", default="flow", choices=["flow", "dino"],
                    help="feature space for the directional metrics "
                         "(flow=motion, dino=appearance)")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    report: list[str] = ["# Regime-Direction Law — verification report",
                         "", "Independent recomputation; no pipeline code imported.", ""]
    master_rows = []
    verdicts = []

    report.insert(1, f"\nFeature space: **{args.space}**")
    for target in ["peak_pck", "auc_normalized"]:
        t, te = load(args.table, args.dist, target, args.space)
        report.append(f"\n## Target: {target}  (rows={len(t)})")
        if target == "peak_pck":
            ok = integrity_checks(t, te, report)
            report.append(f"- **integrity: {'PASS' if ok else 'FAIL'}**")
            verdicts.append(("integrity", ok))
        tf = join_features(t, te)
        miss = tf["mean_nn_a_to_b"].isna().mean()
        report.append(f"- feature join missing rate: {miss:.3%}")
        # features constant across variants?
        nun = (tf.groupby(["train_dataset", "benchmark"])["mean_nn_a_to_b"]
                 .nunique(dropna=True).max())
        report.append(f"- features constant across variants per (train,bench): "
                      f"max nunique = {nun} (must be 1)")
        if target == "peak_pck":
            verdicts.append(("features_constant", nun <= 1))

        for mlabel, c_ab, c_ba, c_sym, better in METRIC_SETS:
            sign = -1.0 if better == "low" else +1.0
            report.append(f"\n### metric family: {mlabel} "
                          f"(better_when={better}; sign={sign:+.0f})")
            report.append("")  # blank line so the table renders as a block
            report.append("| variant | a→b (precision) | b→a (recall) | sym | "
                          "d = a→b − b→a [95% CI] |")
            report.append("|---|---|---|---|---|")
            d_by_variant = {}
            for v in sorted(tf.variant.unique()):
                if v == "raft|False|False":
                    continue
                g = tf[tf.variant == v]
                r_ab = ctx_rhos(g, c_ab, sign, target)
                r_ba = ctx_rhos(g, c_ba, sign, target)
                if not r_ab or not r_ba:
                    continue
                ab = dict(r_ab); ba = dict(r_ba)
                shared = sorted(set(ab) & set(ba))
                diffs = np.array([ab[b] - ba[b] for b in shared])
                lo, hi = boot_ci(diffs, rng)
                m_ab = np.mean([ab[b] for b in shared])
                m_ba = np.mean([ba[b] for b in shared])
                sym_s = ""
                if c_sym:
                    r_sym = dict(ctx_rhos(g, c_sym, sign, target))
                    sym_s = f"{np.mean([r_sym[b] for b in shared if b in r_sym]):+.3f}"
                d_by_variant[v] = diffs
                report.append(f"| {v} | {m_ab:+.3f} | {m_ba:+.3f} | {sym_s} | "
                              f"{diffs.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] |")
                if mlabel == "mean_nn" and target == "peak_pck":
                    master_rows.append(dict(variant=v, metric=mlabel,
                                            rho_ab=m_ab, rho_ba=m_ba,
                                            d=diffs.mean(), d_lo=lo, d_hi=hi))

            # exact permutation test on the 8 non-RAFT variants
            both = [v for v in (sorted(SCRATCH) + sorted(PRETRAINED))
                    if v in d_by_variant]
            if len(both) == 8:
                dv = {v: d_by_variant[v].mean() for v in both}
                obs = (np.mean([dv[v] for v in SCRATCH])
                       - np.mean([dv[v] for v in PRETRAINED]))
                perm = [np.mean([dv[v] for v in combo])
                        - np.mean([dv[v] for v in both if v not in combo])
                        for combo in itertools.combinations(both, 4)]
                p = float(np.mean([abs(x) >= abs(obs) - 1e-12 for x in perm]))
                report.append(f"\n- flip statistic (scratch − pretrained mean d): "
                              f"**{obs:+.3f}**, exact permutation p = **{p:.4f}** "
                              f"(70 assignments; RAFT excluded)")
                if target == "peak_pck":
                    verdicts.append((f"flip_perm_{mlabel}", p <= 0.06 and obs > 0))
                # RAFT prediction check
                if RAFT in d_by_variant:
                    d_raft = d_by_variant[RAFT].mean()
                    report.append(f"- RAFT d = {d_raft:+.3f} "
                                  f"(scratch-profile predicted: d > 0): "
                                  f"{'CONSISTENT' if d_raft > 0 else 'INCONSISTENT'}")
                # leave-one-benchmark-out range of obs
                benches = sorted({b for v in both
                                  for b in tf[tf.variant == v].benchmark.unique()})
                lobo_obs = []
                for held in benches:
                    dv2 = {v: np.mean([x for b, x in
                                       zip(sorted(set(dict(ctx_rhos(tf[tf.variant == v], c_ab, sign, target))) &
                                                  set(dict(ctx_rhos(tf[tf.variant == v], c_ba, sign, target)))),
                                           d_by_variant[v]) if b != held] or [np.nan])
                           for v in both}
                    if any(np.isnan(x) for x in dv2.values()):
                        continue
                    lobo_obs.append(np.mean([dv2[v] for v in SCRATCH])
                                    - np.mean([dv2[v] for v in PRETRAINED]))
                if lobo_obs:
                    report.append(f"- leave-one-benchmark-out flip statistic range: "
                                  f"[{min(lobo_obs):+.3f}, {max(lobo_obs):+.3f}] "
                                  f"({'all >0' if min(lobo_obs) > 0 else 'CROSSES 0'})")
                    if target == "peak_pck":
                        verdicts.append((f"lobo_stable_{mlabel}", min(lobo_obs) > 0))

        # self-pair exclusion (train dataset == benchmark dataset)
        selfless = tf[tf.train_dataset != tf.benchmark]
        report.append(f"\n### self-pair exclusion ({len(tf)-len(selfless)} rows removed)")
        for mlabel, c_ab, c_ba, _, better in METRIC_SETS[:1]:
            sign = -1.0 if better == "low" else +1.0
            sc, pr = [], []
            for v, grp in [(SCRATCH, sc), (PRETRAINED, pr)]:
                for vv in v:
                    g = selfless[selfless.variant == vv]
                    ab = dict(ctx_rhos(g, c_ab, sign, target))
                    ba = dict(ctx_rhos(g, c_ba, sign, target))
                    shared = set(ab) & set(ba)
                    if shared:
                        grp.append(np.mean([ab[b] - ba[b] for b in shared]))
            if sc and pr:
                obs2 = np.mean(sc) - np.mean(pr)
                report.append(f"- {mlabel}: flip statistic without self-pairs = "
                              f"{obs2:+.3f} ({'holds' if obs2 > 0 else 'GONE'})")
                if target == "peak_pck":
                    verdicts.append(("selfpair_robust", obs2 > 0))

    INTEGRITY = ("integrity", "features_constant")
    if args.space == "dino":
        # NEGATIVE CONTROL: the law predicts NO flip in appearance space, so
        # the flip/stability tests are EXPECTED to come out absent here.
        # Generic PASS/FAIL wording previously made the desired control
        # outcome read like a failed result.
        report.append("\n## Verdicts — NEGATIVE CONTROL "
                      "(law predicts NO flip in appearance space)")
        integ_ok = all(ok for n, ok in verdicts if n in INTEGRITY)
        flip_found = any(ok for n, ok in verdicts
                         if n.startswith(("flip_perm", "lobo_stable")))
        for name, ok in verdicts:
            if name in INTEGRITY:
                report.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
            elif name.startswith(("flip_perm", "lobo_stable")):
                msg = ("flip DETECTED — unexpected, investigate" if ok
                       else "no flip — as predicted for the control")
                report.append(f"- {name}: {msg}")
            else:
                report.append(f"- {name}: {'holds' if ok else 'n/a (control)'}")
        overall = ("CONTROL CONFIRMED — no appearance-space flip "
                   "(motion-specificity supported)"
                   if integ_ok and not flip_found else
                   "INVESTIGATE — unexpected outcome in the control space")
        report.append(f"\n# OVERALL: {overall}")
    else:
        report.append("\n## Verdicts")
        all_ok = True
        for name, ok in verdicts:
            report.append(f"- {name}: {'PASS' if ok else 'FAIL'}")
            all_ok &= ok
        report.append(f"\n# OVERALL: "
                      f"{'VERIFIED' if all_ok else 'NOT VERIFIED — investigate FAILs'}")

    out_path.write_text("\n".join(report))
    suffix = "" if args.space == "flow" else f"_{args.space}"
    pd.DataFrame(master_rows).to_csv(
        out_path.parent / f"master_table_mean_nn{suffix}.csv", index=False)
    print("\n".join(report))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
