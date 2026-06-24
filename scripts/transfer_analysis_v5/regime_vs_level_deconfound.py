"""Regime vs transfer-LEVEL deconfound for the regime-direction law (v5).

A hostile reviewer can claim the regime->direction flip is a dynamic-range /
transfer-level artifact: at the VARIANT grain, regime and mean transfer level
are nearly collinear (REGIME_DIRECTION_FINDING.md records spearman(flip d,
mean transfer level) = -0.80 across the 9 variants). This script attacks the
claim at the CONTEXT grain (context = variant x benchmark, 9 x 9 = 81), where
level varies widely WITHIN regime.

Per context (within the 11 canonical sources):
    level          = mean peak_pck over sources
    r_recall       = spearman(peak_pck, -mean_nn_b_to_a)   # d(B->T), recall
    r_precision    = spearman(peak_pck, -mean_nn_a_to_b)   # d(T->B), precision
    direction_gap  = r_recall - r_precision    # >0 => recall direction better

Sign conventions copied verbatim from asym_vs_sym_table.py (spearman of target
against NEGATIVE distance; dataset_a = train source, dataset_b = benchmark in
analysis_v3/pairwise_self_distances.csv, pair_type == "train_eval", space ==
"flow"). Sanity check: per-variant means of r_precision / r_recall must
reproduce results/asym_vs_sym.csv.

Analyses:
  (a) per-context table (saved to *_contexts.csv)
  (b) OLS direction_gap ~ regime + level (pooled; nonrobust + HC1 + cluster by
      benchmark + cluster by variant; plus per-benchmark OLS) and partial
      Spearman (gap vs regime | level, gap vs level | regime)
  (c) within-regime spearman(direction_gap, level) for scratch / pretrained
      contexts separately (+ within-variant level correlations, where regime
      is held exactly fixed)
  (d) leave-one-variant-out selection-rule comparison: level-threshold rule
      (threshold + orientation fit on the other 8 variants' contexts) vs the
      fit-free regime rule, scored by mean within-context spearman of the
      chosen-direction predictor on the held-out variant's contexts.

Run:
    nice -n 15 /home/spencer/miniconda3/envs/cuda/bin/python \
        scripts/transfer_analysis_v5/regime_vs_level_deconfound.py
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

warnings.filterwarnings("ignore")

ROOT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]

PREC = "mean_nn_a_to_b"   # d(T->B): train mass far from benchmark = precision
REC = "mean_nn_b_to_a"    # d(B->T): benchmark mass uncovered = recall


def regime_of(v: str) -> str:
    arch, pre, _ = v.split("|")
    return "scratch" if (pre == "False" or arch == "raft") else "pretrained"


def sp(x, y) -> float:
    r = spearmanr(x, y).statistic
    return float(r)


def partial_spearman(x, y, z):
    """Partial Spearman r_xy.z via partial Pearson on ranks; t-test p, df=n-3."""
    from scipy.stats import t as tdist
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]
    rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    n = len(x)
    tval = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    p = 2 * tdist.sf(abs(tval), n - 3)
    return float(r), float(p)


def load_contexts(table_path, dist_path, target="peak_pck"):
    t = pd.read_csv(table_path)
    t = t[t.train_dataset.isin(PURE)].copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"]
    t["cv"] = t.benchmark + "|" + t.variant
    d = pd.read_csv(dist_path)
    te = d[(d.pair_type == "train_eval") & (d.space == "flow")]
    f = te.set_index(["dataset_a", "dataset_b"])[[PREC, REC, "mean_nn_sym"]]
    t = t.join(f, on=["train_dataset", "benchmark"], how="left")
    t = t.dropna(subset=[target, PREC])
    return t


def vctx(g, col, target="peak_pck"):
    """Verbatim re-implementation of asym_vs_sym_table.py's per-variant mean."""
    rs = [spearmanr(c[target], -c[col]).statistic
          for _, c in g.groupby("cv")
          if c.train_dataset.nunique() >= 3 and c[col].std() > 1e-12]
    rs = [r for r in rs if np.isfinite(r)]
    return float(np.nanmean(rs)) if rs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(
        ROOT / "scripts/transfer_analysis_v3/transfer_table_nomid.csv"))
    ap.add_argument("--dist", default=str(
        ROOT / "analysis_v3/pairwise_self_distances.csv"))
    ap.add_argument("--asym-csv", default=str(
        ROOT / "scripts/transfer_analysis_v5/results/asym_vs_sym.csv"))
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--out", default=str(
        ROOT / "scripts/transfer_analysis_v5/results/regime_vs_level_deconfound.csv"))
    args = ap.parse_args()
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_ctx = out_csv.with_name("regime_vs_level_deconfound_contexts.csv")
    out_txt = out_csv.with_name("regime_vs_level_deconfound_summary.txt")

    lines = []

    def log(s=""):
        print(s)
        lines.append(str(s))

    t = load_contexts(args.table, args.dist, args.target)
    log(f"table: {args.table}")
    log(f"dist : {args.dist}")
    log(f"rows after PURE-11 filter + raft|False|False drop + dropna: {len(t)}")
    log(f"variants: {sorted(t.variant.unique())}")
    log(f"benchmarks: {sorted(t.benchmark.unique())}")

    # ------------------------------------------------------------------ (a)
    rows = []
    for (bench, var), c in t.groupby(["benchmark", "variant"]):
        if c.train_dataset.nunique() < 3:
            continue
        ok_p = c[PREC].std() > 1e-12
        ok_r = c[REC].std() > 1e-12
        r_p = sp(c[args.target], -c[PREC]) if ok_p else np.nan
        r_r = sp(c[args.target], -c[REC]) if ok_r else np.nan
        rows.append(dict(
            benchmark=bench, variant=var, regime=regime_of(var),
            n_sources=int(c.train_dataset.nunique()),
            level=float(c[args.target].mean()),
            level_std=float(c[args.target].std()),
            r_precision=r_p, r_recall=r_r,
            direction_gap=(r_r - r_p) if (np.isfinite(r_p) and np.isfinite(r_r))
            else np.nan,
        ))
    ctx = pd.DataFrame(rows)
    ctx["regime_pretrained"] = (ctx.regime == "pretrained").astype(int)
    ctx.to_csv(out_ctx, index=False)
    n_all = len(ctx)
    ctx = ctx.dropna(subset=["direction_gap", "level"]).reset_index(drop=True)
    log(f"\ncontexts: {n_all} total, {len(ctx)} with finite direction_gap "
        f"({int((ctx.regime == 'scratch').sum())} scratch, "
        f"{int((ctx.regime == 'pretrained').sum())} pretrained)")
    log(f"wrote per-context table -> {out_ctx}")

    # ------------------------------------------------- sanity vs asym_vs_sym
    log("\n=== SANITY CHECK: per-variant means must reproduce "
        "results/asym_vs_sym.csv ===")
    ref = pd.read_csv(args.asym_csv).set_index("variant")
    san_rows = []
    for v, g in t.groupby("variant"):
        mine_p, mine_r = vctx(g, PREC, args.target), vctx(g, REC, args.target)
        ctx_v = ctx[ctx.variant == v]
        san_rows.append(dict(
            variant=v,
            my_precision=mine_p, ref_precision=ref.loc[v, "precision"],
            my_recall=mine_r, ref_recall=ref.loc[v, "recall"],
            my_mean_gap=float(ctx_v.direction_gap.mean()),
            ref_recall_minus_precision=float(ref.loc[v, "recall"]
                                             - ref.loc[v, "precision"]),
        ))
    san = pd.DataFrame(san_rows)
    san["abs_diff_precision"] = (san.my_precision - san.ref_precision).abs()
    san["abs_diff_recall"] = (san.my_recall - san.ref_recall).abs()
    san["abs_diff_gap"] = (san.my_mean_gap - san.ref_recall_minus_precision).abs()
    log(san.to_string(index=False, float_format=lambda x: f"{x:+.6f}"))
    max_p = san.abs_diff_precision.max()
    max_r = san.abs_diff_recall.max()
    max_g = san.abs_diff_gap.max()
    log(f"max |precision diff| = {max_p:.3e}   max |recall diff| = {max_r:.3e}"
        f"   max |gap diff| = {max_g:.3e}")
    sanity_pass = bool(max_p < 1e-9 and max_r < 1e-9 and max_g < 1e-9)
    log(f"SANITY {'PASS' if sanity_pass else 'FAIL'}")
    if not sanity_pass:
        log("WARNING: per-variant means do not reproduce asym_vs_sym.csv -- "
            "check table/dist inputs before trusting downstream numbers.")

    # ------------------------------------------------------------------ (b)
    import statsmodels.api as sm

    summary = []  # tidy metric rows for the main CSV

    def add(metric, value, n=None, detail=""):
        summary.append(dict(metric=metric, value=value, n=n, detail=detail))

    add("sanity_max_abs_diff_precision", max_p, len(san), "vs asym_vs_sym.csv")
    add("sanity_max_abs_diff_recall", max_r, len(san), "vs asym_vs_sym.csv")
    add("sanity_max_abs_diff_gap", max_g, len(san),
        "my per-context mean gap vs CSV recall-precision")
    add("sanity_pass", float(sanity_pass), len(san), "")

    log("\n=== (b) POOLED OLS: direction_gap ~ regime_pretrained + level "
        f"(n={len(ctx)} contexts) ===")
    y = ctx.direction_gap.values
    X = sm.add_constant(ctx[["regime_pretrained", "level"]].astype(float))
    # z-scored level for comparable coefficient magnitudes
    Xz = X.copy()
    lvl_mu, lvl_sd = ctx.level.mean(), ctx.level.std()
    Xz["level"] = (Xz["level"] - lvl_mu) / lvl_sd
    fit = sm.OLS(y, X).fit()
    fitz = sm.OLS(y, Xz).fit()
    covs = {
        "nonrobust": fit,
        "HC1": sm.OLS(y, X).fit(cov_type="HC1"),
        "cluster(benchmark)": sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds=dict(groups=ctx.benchmark)),
        "cluster(variant)": sm.OLS(y, X).fit(
            cov_type="cluster", cov_kwds=dict(groups=ctx.variant)),
    }
    log(f"R^2 = {fit.rsquared:.4f}   coef(regime_pretrained) = "
        f"{fit.params['regime_pretrained']:+.4f}   coef(level, per PCK pt) = "
        f"{fit.params['level']:+.5f}")
    log(f"standardized: coef(regime) = {fitz.params['regime_pretrained']:+.4f}"
        f"   coef(level, per SD={lvl_sd:.2f} PCK) = {fitz.params['level']:+.4f}")
    for name, f_ in covs.items():
        for term in ["regime_pretrained", "level"]:
            log(f"  [{name:>18s}] {term:<17s} coef={f_.params[term]:+.5f}  "
                f"se={f_.bse[term]:.5f}  t={f_.tvalues[term]:+.3f}  "
                f"p={f_.pvalues[term]:.4g}")
            add(f"ols_{term}_coef", float(f_.params[term]), len(ctx), name)
            add(f"ols_{term}_p", float(f_.pvalues[term]), len(ctx), name)
    add("ols_r2", float(fit.rsquared), len(ctx), "pooled")
    add("ols_regime_coef_std", float(fitz.params["regime_pretrained"]),
        len(ctx), "level z-scored")
    add("ols_level_coef_std", float(fitz.params["level"]), len(ctx),
        f"per 1 SD = {lvl_sd:.3f} PCK pts")

    # regime-only and level-only single-predictor fits (variance explained)
    f_reg = sm.OLS(y, sm.add_constant(
        ctx[["regime_pretrained"]].astype(float))).fit()
    f_lvl = sm.OLS(y, sm.add_constant(ctx[["level"]].astype(float))).fit()
    log(f"single-predictor R^2: regime-only = {f_reg.rsquared:.4f}   "
        f"level-only = {f_lvl.rsquared:.4f}   both = {fit.rsquared:.4f}")
    add("r2_regime_only", float(f_reg.rsquared), len(ctx), "")
    add("r2_level_only", float(f_lvl.rsquared), len(ctx), "")
    add("r2_both", float(fit.rsquared), len(ctx), "")

    log("\n--- raw + partial Spearman (context grain) ---")
    s_gap_reg = sp(ctx.direction_gap, ctx.regime_pretrained)
    s_gap_lvl = sp(ctx.direction_gap, ctx.level)
    s_reg_lvl = sp(ctx.regime_pretrained, ctx.level)
    log(f"spearman(gap, regime)            = {s_gap_reg:+.4f}")
    log(f"spearman(gap, level)             = {s_gap_lvl:+.4f}")
    log(f"spearman(regime, level)          = {s_reg_lvl:+.4f}   "
        "(collinearity at context grain)")
    pr_reg, p_reg = partial_spearman(ctx.direction_gap, ctx.regime_pretrained,
                                     ctx.level)
    pr_lvl, p_lvl = partial_spearman(ctx.direction_gap, ctx.level,
                                     ctx.regime_pretrained)
    log(f"partial spearman(gap, regime | level) = {pr_reg:+.4f}  p={p_reg:.4g}")
    log(f"partial spearman(gap, level | regime) = {pr_lvl:+.4f}  p={p_lvl:.4g}")
    add("spearman_gap_regime", s_gap_reg, len(ctx), "raw")
    add("spearman_gap_level", s_gap_lvl, len(ctx), "raw")
    add("spearman_regime_level", s_reg_lvl, len(ctx), "collinearity check")
    add("partial_spearman_gap_regime_given_level", pr_reg, len(ctx),
        f"p={p_reg:.4g}")
    add("partial_spearman_gap_level_given_regime", pr_lvl, len(ctx),
        f"p={p_lvl:.4g}")

    log("\n--- per-benchmark OLS (9 contexts each) ---")
    pb_rows = []
    for bench, g in ctx.groupby("benchmark"):
        if g.regime_pretrained.nunique() < 2 or len(g) < 4:
            continue
        fb = sm.OLS(g.direction_gap.values, sm.add_constant(
            g[["regime_pretrained", "level"]].astype(float))).fit()
        pb_rows.append(dict(
            benchmark=bench, n=len(g),
            coef_regime=float(fb.params["regime_pretrained"]),
            coef_level=float(fb.params["level"]),
            sp_gap_regime=sp(g.direction_gap, g.regime_pretrained),
            sp_gap_level=sp(g.direction_gap, g.level)))
    pb = pd.DataFrame(pb_rows)
    log(pb.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    log(f"per-benchmark coef(regime): {int((pb.coef_regime > 0).sum())}/"
        f"{len(pb)} positive, median {pb.coef_regime.median():+.4f}")
    log(f"per-benchmark coef(level) : {int((pb.coef_level > 0).sum())}/"
        f"{len(pb)} positive, median {pb.coef_level.median():+.5f}")
    add("per_bench_regime_coef_median", float(pb.coef_regime.median()),
        len(pb), f"{int((pb.coef_regime > 0).sum())}/{len(pb)} positive")
    add("per_bench_level_coef_median", float(pb.coef_level.median()),
        len(pb), f"{int((pb.coef_level > 0).sum())}/{len(pb)} positive")

    # ------------------------------------------------------------------ (c)
    log("\n=== (c) WITHIN-REGIME level test ===")
    for rg, g in ctx.groupby("regime"):
        s = sp(g.direction_gap, g.level)
        log(f"{rg:<10s} (n={len(g):2d} contexts): "
            f"spearman(direction_gap, level) = {s:+.4f}   "
            f"level range [{g.level.min():.2f}, {g.level.max():.2f}], "
            f"mean gap {g.direction_gap.mean():+.4f}")
        add(f"within_{rg}_spearman_gap_level", s, len(g),
            f"level range [{g.level.min():.2f}, {g.level.max():.2f}]")
        add(f"within_{rg}_mean_gap", float(g.direction_gap.mean()), len(g), "")

    log("\n--- within-variant (regime exactly fixed; 9 benchmarks each) ---")
    wv_rows = []
    for v, g in ctx.groupby("variant"):
        if len(g) >= 4:
            wv_rows.append(dict(variant=v, regime=regime_of(v), n=len(g),
                                sp_gap_level=sp(g.direction_gap, g.level)))
    wv = pd.DataFrame(wv_rows)
    log(wv.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    for rg, g in wv.groupby("regime"):
        log(f"mean within-variant spearman(gap, level), {rg}: "
            f"{g.sp_gap_level.mean():+.4f} (n={len(g)} variants)")
        add(f"within_variant_mean_sp_gap_level_{rg}",
            float(g.sp_gap_level.mean()), len(g), "mean over variants")

    # ------------------------------------------------------------------ (d)
    log("\n=== (d) LEAVE-ONE-VARIANT-OUT: level-threshold rule vs regime "
        "rule ===")
    # context-level direction predictors are r_recall / r_precision (already
    # the within-context spearman of the chosen-direction predictor).

    def level_rule_score(g, thr, orient):
        """orient=+1: level >= thr -> recall direction; -1: -> precision."""
        pick_recall = (g.level >= thr) if orient > 0 else (g.level < thr)
        return float(np.where(pick_recall, g.r_recall, g.r_precision).mean())

    variants = sorted(ctx.variant.unique())
    lovo_rows = []
    for v in variants:
        tr = ctx[ctx.variant != v]
        te = ctx[ctx.variant == v]
        cand = np.unique(tr.level.values)
        thrs = np.concatenate([[cand.min() - 1],
                               (cand[:-1] + cand[1:]) / 2,
                               [cand.max() + 1]])
        best = max(((level_rule_score(tr, th, o), th, o)
                    for th in thrs for o in (+1, -1)),
                   key=lambda x: x[0])
        _, th_b, o_b = best
        lvl_sc = level_rule_score(te, th_b, o_b)
        reg_sc = float(np.where(te.regime == "pretrained",
                                te.r_recall, te.r_precision).mean())
        ora_sc = float(np.maximum(te.r_recall, te.r_precision).mean())
        lovo_rows.append(dict(
            heldout_variant=v, regime=regime_of(v), n_ctx=len(te),
            fitted_threshold=float(th_b), orientation=int(o_b),
            train_score=float(best[0]),
            level_rule_heldout=lvl_sc, regime_rule_heldout=reg_sc,
            oracle_heldout=ora_sc))
    lovo = pd.DataFrame(lovo_rows)
    log(lovo.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    w = lovo.n_ctx.values
    lvl_mean = float(np.average(lovo.level_rule_heldout, weights=w))
    reg_mean = float(np.average(lovo.regime_rule_heldout, weights=w))
    ora_mean = float(np.average(lovo.oracle_heldout, weights=w))
    log(f"\nLOVO mean within-context spearman (context-weighted over "
        f"{int(w.sum())} held-out contexts):")
    log(f"  level-threshold rule : {lvl_mean:+.4f}")
    log(f"  regime rule          : {reg_mean:+.4f}")
    log(f"  per-context oracle   : {ora_mean:+.4f}")
    log(f"  regime - level       : {reg_mean - lvl_mean:+.4f}")
    nv_reg = int((lovo.regime_rule_heldout > lovo.level_rule_heldout).sum())
    nv_lvl = int((lovo.level_rule_heldout > lovo.regime_rule_heldout).sum())
    log(f"  held-out variants won: regime {nv_reg}, level {nv_lvl}, "
        f"tie {len(lovo) - nv_reg - nv_lvl}")
    add("lovo_level_rule_mean", lvl_mean, int(w.sum()),
        "threshold+orientation fit LOVO")
    add("lovo_regime_rule_mean", reg_mean, int(w.sum()), "fit-free")
    add("lovo_oracle_mean", ora_mean, int(w.sum()), "per-context best dir")
    add("lovo_regime_minus_level", reg_mean - lvl_mean, int(w.sum()),
        f"variants won: regime {nv_reg} / level {nv_lvl}")

    # caveat quantified: the level rule needs the held-out variant's own mean
    # transfer (an outcome); the regime rule needs only the config string.
    log("\nNOTE: the level rule is handed the held-out variant's own mean "
        "peak_pck per context (an outcome of training); the regime rule uses "
        "only the config string. The comparison is generous to the level "
        "rule.")

    # ------------------------------------------------------------- outputs
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    out_txt.write_text("\n".join(lines) + "\n")
    log(f"\nwrote {out_csv}")
    log(f"wrote {out_txt}")


if __name__ == "__main__":
    main()
