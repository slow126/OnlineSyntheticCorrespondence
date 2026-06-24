"""Leave-one-architecture-out: do the per-regime combination weights transfer?

The p-hacking check for the per-regime linear story (Spencer, 2026-06-10):
fit the 2-coefficient per-regime model on two architectures, evaluate its
within-context ranking on the third, and inspect the fitted weights per fold.
Features are within-context z-scored directed mean-NN distances (zP =
off-target mass, zR = missing support), so weight ratios are comparable
across folds and not driven by the ~50-100x raw scale gap between the two
directions.

Verdict from the 2026-06-10 run:
  - SCRATCH: weights transfer and are ~equal (ratio 0.8-1.3 in every fold);
    held-out rho +0.51/+0.65/+0.64 ~= the fixed symmetric average. The
    cross-architecture-stable scratch combiner IS the average.
  - PRETRAINED: fitted weights do NOT transfer (only 2 pretrained archs;
    fit-on-GLU-Net gives a wrong-signed precision weight and held-out +0.25
    on CATs++; fit-on-CATs++ gives balanced weights and +0.46 on GLU-Net),
    while the fit-free recall rule scores +0.44/+0.67 on the same held-out
    contexts. The cross-architecture-defensible pretrained choice is the
    fit-free recall direction.
  - Together these justify exactly the paper's regime-aware policy and
    nothing fancier. NOTE: coefficient RATIOS from globally-z-scored fits
    (e.g. the 7:1 scratch ratio in the regime-linear table) are
    normalization-sensitive; the sign pattern and the single-direction flip
    are not.

    python scripts/transfer_analysis_v5/loao_weight_transfer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from per_regime_linear import AB, BA, load  # noqa: E402

RES = Path(__file__).parent / "results"


def ctx_rho(df, col):
    out = []
    for _, c in df.groupby("cv"):
        c = c.drop_duplicates("train_dataset")
        if c.train_dataset.nunique() < 3 or c[col].std() <= 1e-12:
            continue
        out.append(spearmanr(c.peak_pck, c[col]).statistic)
    return (float(np.mean(out)) if out else float("nan")), len(out)


def main():
    t, _ = load()
    t["arch"] = t.variant.str.split("|").str[0]
    for raw, z in [(AB, "zP"), (BA, "zR")]:
        g = t.groupby("cv")[raw]
        sd = g.transform("std")
        t[z] = (t[raw] - g.transform("mean")) / sd.where(sd > 0)
    t = t.dropna(subset=["zP", "zR"])

    recs = []
    for arch in ["catspp", "glunet", "raft"]:
        for regime in ["scratch", "pretrained"]:
            tr = t[(t.arch != arch) & (t.regime == regime)]
            ts = t[(t.arch == arch) & (t.regime == regime)].copy()
            if len(ts) == 0:
                continue
            mu = tr.groupby("cv").peak_pck.transform("mean")
            w, *_ = np.linalg.lstsq(tr[["zP", "zR"]].values,
                                    (tr.peak_pck - mu).values, rcond=None)
            ts["pred"] = ts[["zP", "zR"]].values @ w
            ts["rule"] = -ts.zP if regime == "scratch" else -ts.zR
            ts["avg"] = -(ts.zP + ts.zR) / 2
            rho, n = ctx_rho(ts, "pred")
            rule_rho, _ = ctx_rho(ts, "rule")
            avg_rho, _ = ctx_rho(ts, "avg")
            recs.append(dict(held_out_arch=arch, regime=regime,
                             w_P=w[0], w_R=w[1],
                             ratio_P_over_R=w[0] / w[1] if abs(w[1]) > 1e-9
                             else np.inf,
                             heldout_rho_fit=rho, heldout_rho_rule=rule_rho,
                             heldout_rho_avg=avg_rho, n_ctx=n))
            print(f"{arch:<8} {regime:<11} wP {w[0]:+6.2f} wR {w[1]:+6.2f}  "
                  f"fit {rho:+.3f}  rule {rule_rho:+.3f}  avg {avg_rho:+.3f}"
                  f"  (n={n})")
    pd.DataFrame(recs).to_csv(RES / "loao_weight_transfer.csv", index=False)
    print(f"\nwrote {RES / 'loao_weight_transfer.csv'}")


if __name__ == "__main__":
    main()
