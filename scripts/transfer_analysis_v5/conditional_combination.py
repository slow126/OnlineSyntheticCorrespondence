"""Is there a better way to combine precision and recall than rule/average?

Spencer's hypothesis (2026-06-10): a hierarchy — "once precision needs are
met, recall takes over" — i.e. the average is too naive and the one-direction
rule too rigid; some conditional combination might beat both.

The descriptor is model-free, so conditioning is restricted to (a) the
training regime (known a priori), (b) the candidates' own distance profiles
(data-side), and (c) the context's anchor level L (observed reference runs —
the one cheap model diagnostic the pipeline already requires for absolute
prediction).

Candidates evaluated, all on within-context z-scored directed mean-NN
distances (zP = off-target mass / precision, zR = missing support / recall;
score is higher-better):

  fit-free (no folds needed):
    rule          matched direction per regime (-zP scratch, -zR pretrained)
    mismatched    the opposite (control)
    average       -(zP+zR)/2  (== symmetric mean-NN up to scaling)
    tilted 2:1    matched direction weighted 2x the other
    bottleneck    -max(zP, zR)   Liebig: the candidate's WORSE constraint
                  binds — the per-candidate version of the hierarchy
    best-of       -min(zP, zR)   (control for bottleneck)
    lex P-first   precision met (zP <= ctx median) -> rank by recall;
                  else rank by precision  [the hierarchy, stated directly]
    lex R-first   the mirror image
    dispersion-w  weight each direction by its within-context raw
                  coefficient of variation (more spread = more signal)

  fitted per regime (held-out folds, LOTO + JOINT):
    linear2       OLS on zP, zR        (= Table 4 of the paper)
    linear2+int   OLS on zP, zR, zP*zR (interaction = soft conditionality)
    linear2+L     OLS on zP, zR, zP*Lz, zR*Lz (level-conditioned weights:
                  "as the context's level rises, does recall take over?")
    gbm           depth-2 gradient boosting on zP, zR (nonparametric upper
                  bound: if THIS can't beat linear, there is no exploitable
                  conditional structure at this n)

Metric: mean within-context Spearman rho vs peak_pck, per regime and pooled;
median top-1 regret for the leading contenders.

    python scripts/transfer_analysis_v5/conditional_combination.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from per_regime_linear import AB, BA, folds, load  # noqa: E402

RES = Path(__file__).parent / "results"
RNG = np.random.default_rng(0)


def ctx_rho(df, col):
    """Mean within-context Spearman of score col vs peak_pck."""
    out = []
    for _, c in df.groupby("cv"):
        c = c.drop_duplicates("train_dataset")
        if c.train_dataset.nunique() < 3 or c[col].std() <= 1e-12:
            continue
        out.append(spearmanr(c.peak_pck, c[col]).statistic)
    return float(np.mean(out)), len(out)


def median_regret(df, col):
    regs = []
    for _, c in df.groupby("cv"):
        c = c.drop_duplicates("train_dataset")
        if c.train_dataset.nunique() < 3:
            continue
        regs.append(c.peak_pck.max() - c.loc[c[col].idxmax(), "peak_pck"])
    return float(np.median(regs))


def main():
    t, L = load()

    # within-context z-scores (lower distance = better, so scores negate)
    for raw, z in [(AB, "zP"), (BA, "zR")]:
        g = t.groupby("cv")[raw]
        sd = g.transform("std")
        t[z] = (t[raw] - g.transform("mean")) / sd.where(sd > 0)

    matched = np.where(t.regime == "scratch", -t.zP, -t.zR)
    other = np.where(t.regime == "scratch", -t.zR, -t.zP)
    t["s_rule"] = matched
    t["s_mismatched"] = other
    t["s_average"] = -(t.zP + t.zR) / 2
    t["s_tilted21"] = (2 * matched + other) / 3
    t["s_bottleneck"] = -np.maximum(t.zP, t.zR)
    t["s_bestof"] = -np.minimum(t.zP, t.zR)

    # lexicographic: primary met (<= ctx median) -> rank by secondary
    for name, prim, sec in [("s_lexP", "zP", "zR"), ("s_lexR", "zR", "zP")]:
        med = t.groupby("cv")[prim].transform("median")
        ok = t[prim] <= med
        t[name] = np.where(ok, 100.0 - t[sec], -t[prim])

    # dispersion-weighted average (raw CV per context per direction)
    for raw, w in [(AB, "wP"), (BA, "wR")]:
        g = t.groupby("cv")[raw]
        t[w] = g.transform("std") / g.transform("mean")
    t["s_dispw"] = -(t.wP * t.zP + t.wR * t.zR) / (t.wP + t.wR)

    FREE = [("rule (matched direction)", "s_rule"),
            ("mismatched direction", "s_mismatched"),
            ("average (symmetric)", "s_average"),
            ("tilted 2:1 to matched", "s_tilted21"),
            ("bottleneck: worse of two", "s_bottleneck"),
            ("best of two", "s_bestof"),
            ("lexicographic, precision first", "s_lexP"),
            ("lexicographic, recall first", "s_lexR"),
            ("dispersion-weighted average", "s_dispw")]

    recs = []
    print(f"{'score':<32} {'scratch':>9} {'pretrained':>11} {'pooled':>8}")
    print("-" * 64)
    for name, col in FREE:
        sub = t.dropna(subset=[col])
        sc, _ = ctx_rho(sub[sub.regime == "scratch"], col)
        pr, _ = ctx_rho(sub[sub.regime == "pretrained"], col)
        po, n = ctx_rho(sub, col)
        recs.append(dict(kind="fit-free", score=name, split="-",
                         scratch=sc, pretrained=pr, pooled=po, n_ctx=n))
        print(f"{name:<32} {sc:>+9.3f} {pr:>+11.3f} {po:>+8.3f}")

    # ---------------- fitted, held-out, per regime ----------------
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        HAVE_SK = True
    except ImportError:
        HAVE_SK = False

    # context-level anchor (z-scored within regime, context-constant)
    lbar = (L["LOTO"].rename("Lrow").reset_index())
    t2 = t.merge(lbar, on=["train_dataset", "benchmark", "variant"],
                 how="left")
    t2["Lctx"] = t2.groupby("cv").Lrow.transform("mean")
    t2["Lz"] = (t2.Lctx - t2.groupby("regime").Lctx.transform("mean")) \
        / t2.groupby("regime").Lctx.transform("std")

    t2["zPzR"] = t2.zP * t2.zR
    t2["zPL"] = t2.zP * t2.Lz
    t2["zRL"] = t2.zR * t2.Lz

    DESIGNS = [("linear2 (zP,zR)", ["zP", "zR"], "ols"),
               ("linear2 + interaction", ["zP", "zR", "zPzR"], "ols"),
               ("linear2 + level-conditioned", ["zP", "zR", "zPL", "zRL"],
                "ols")]
    if HAVE_SK:
        DESIGNS.append(("gbm depth-2 (zP,zR)", ["zP", "zR"], "gbm"))

    def fitpred(tr, ts, feats, kind):
        mu = tr.groupby("cv").peak_pck.transform("mean")
        y = (tr.peak_pck - mu).values
        X, Xt = tr[feats].values, ts[feats].values
        if kind == "ols":
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            return Xt @ w
        m = GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                      learning_rate=0.05, random_state=0)
        m.fit(X, y)
        return m.predict(Xt)

    print()
    print(f"{'fitted (held out)':<32} {'split':<6} {'scratch':>9} "
          f"{'pretrained':>11} {'pooled':>8}")
    print("-" * 70)
    base = t2.dropna(subset=["zP", "zR", "Lz"])
    for name, feats, kind in DESIGNS:
        for sp in ["LOTO", "JOINT"]:
            df = base.copy()
            df["pred"] = np.nan
            for regime in ["scratch", "pretrained"]:
                rmask = df.regime == regime
                for trm, tsm in folds(df, sp):
                    tr, ts = trm & rmask, tsm & rmask
                    if ts.sum() == 0 or tr.sum() < 30:
                        continue
                    df.loc[ts, "pred"] = fitpred(df[tr], df[ts], feats, kind)
            sub = df.dropna(subset=["pred"])
            sc, _ = ctx_rho(sub[sub.regime == "scratch"], "pred")
            pr, _ = ctx_rho(sub[sub.regime == "pretrained"], "pred")
            po, n = ctx_rho(sub, "pred")
            recs.append(dict(kind=kind, score=name, split=sp, scratch=sc,
                             pretrained=pr, pooled=po, n_ctx=n))
            print(f"{name:<32} {sp:<6} {sc:>+9.3f} {pr:>+11.3f} {po:>+8.3f}")

    # regret for the leading fit-free contenders
    print()
    print(f"{'median top-1 regret (PCK)':<32} {'scratch':>9} {'pretrained':>11}")
    print("-" * 56)
    for name, col in FREE[:1] + FREE[2:3] + FREE[4:5] + FREE[6:8]:
        sub = t.dropna(subset=[col])
        s = median_regret(sub[sub.regime == "scratch"], col)
        p = median_regret(sub[sub.regime == "pretrained"], col)
        print(f"{name:<32} {s:>9.2f} {p:>11.2f}")

    pd.DataFrame(recs).to_csv(RES / "conditional_combination.csv", index=False)
    print(f"\nwrote {RES / 'conditional_combination.csv'}")


if __name__ == "__main__":
    main()
