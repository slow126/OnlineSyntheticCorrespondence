"""JOINT two-way similarity anchor — proper generator (replaces the session
artifact joint_anchor_v2.csv).

In the JOINT setting both the training set i and the benchmark k are unseen,
but their FEATURES are not: we know how motion-similar k is to every other
benchmark (eval<->eval distances) and how similar i is to every other training
set (train<->train distances). The anchor pulls observed cells through BOTH
similarities at once.

Two constructions, both in-fold only (rows with train==i or bench==k are
never used):

  additive (the old L2):  variant in-fold mean
                          + IDW-borrowed benchmark effect (eval<->eval kernel)
                          + IDW-borrowed source effect    (train<->train kernel)

  two-way kernel (L2K):   direct kernel smoother over in-fold cells of the
                          same variant: L(i,k,v) =
                          sum_{j!=i, e!=k} w_eval(k,e) * w_train(i,j) * perf(j,e,v)
                          (product kernel = "rows similar in BOTH coordinates
                          dominate", instead of two marginal corrections)

Output: results/joint_anchor_v2.csv with columns
  src, bench, variant, actual, L2 (additive), L2K (two-way kernel), L2g, L2Kg
and a printed comparison. Middlebury excluded (eval bug).

    python scripts/transfer_analysis_v5/joint_anchor.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
RES = ROOT / "scripts/transfer_analysis_v5/results"

PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]
EPS = 1e-9


def sim_matrix(d: pd.DataFrame, pair_type: str, metric: str = "mean_nn_sym"):
    """symmetric distance lookup {frozenset(a,b): dist} from the distances CSV"""
    s = d[(d.pair_type == pair_type) & (d.space == "flow")]
    out = {}
    for _, r in s.iterrows():
        out[frozenset((r.dataset_a, r.dataset_b))] = float(r[metric])
    return out


def idw(target, others, dist, power=1.0):
    """inverse-distance weights of `others` relative to `target`"""
    w = {}
    for o in others:
        if o == target:
            continue
        key = frozenset((target, o))
        if key in dist:
            w[o] = 1.0 / (dist[key] + EPS) ** power
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()} if tot > 0 else {}


def main():
    t = pd.read_csv(ROOT / "scripts/transfer_analysis_v3/transfer_table_nomid.csv")
    t = t[t.train_dataset.isin(PURE)].dropna(subset=["peak_pck"]).copy()
    t["variant"] = (t.model_family.astype(str) + "|" + t.pretrained.astype(str)
                    + "|" + t.freeze.astype(str))
    t = t[t.variant != "raft|False|False"]

    d = pd.read_csv(ROOT / "analysis_v3/pairwise_self_distances.csv")
    d_ee = sim_matrix(d, "eval_eval")
    d_tt = sim_matrix(d, "train_train")

    benches = sorted(t.benchmark.unique())
    rows = []
    for v, tv in t.groupby("variant"):
        cell = tv.set_index(["train_dataset", "benchmark"]).peak_pck
        for i in PURE:
            for k in benches:
                if (i, k) not in cell.index:
                    continue
                infold = tv[(tv.train_dataset != i) & (tv.benchmark != k)]
                if len(infold) < 4:
                    continue
                w_e = idw(k, infold.benchmark.unique(), d_ee)
                w_t = idw(i, infold.train_dataset.unique(), d_tt)

                # additive: variant mean + borrowed bench/source effects
                vm = infold.peak_pck.mean()
                bmeans = infold.groupby("benchmark").peak_pck.mean()
                smeans = infold.groupby("train_dataset").peak_pck.mean()
                be = sum(w * (bmeans[e] - vm) for e, w in w_e.items()
                         if e in bmeans)
                se = sum(w * (smeans[j] - vm) for j, w in w_t.items()
                         if j in smeans)
                l2 = vm + be + se

                # two-way product kernel over in-fold cells
                num = den = 0.0
                for (j, e), y in infold.set_index(
                        ["train_dataset", "benchmark"]).peak_pck.items():
                    w = w_t.get(j, 0.0) * w_e.get(e, 0.0)
                    num += w * y
                    den += w
                l2k = num / den if den > 0 else vm

                rows.append(dict(src=i, bench=k, variant=v,
                                 actual=float(cell.loc[(i, k)]),
                                 L2=l2, L2K=l2k))

    out = pd.DataFrame(rows)
    for c in ("L2", "L2K"):
        r = pearsonr(out[c], out.actual)[0]
        mae = float(np.abs(out[c] - out.actual).mean())
        print(f"{c}: r = {r:+.3f}   MAE = {mae:.2f}   (n={len(out)})")
    out.to_csv(RES / "joint_anchor_v2.csv", index=False)
    print(f"wrote {RES / 'joint_anchor_v2.csv'}")


if __name__ == "__main__":
    main()
