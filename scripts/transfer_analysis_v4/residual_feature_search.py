"""
Residual feature search: does ANY available dataset descriptor explain the
within-cell magnitude that motion-g leaves on the table?

residual r = actual - L - g   (L = leave-one-out cell mean, g = motion ridge).
We within-cell demean r and each candidate feature, then fit a leakage-clean
leave-one-source-out ridge of r on each feature FAMILY, and report the
out-of-fold within-cell R^2.

- Control "motion (g already saw)": flow_* features -> should be ~0 (g used them).
- Real question: do appearance / density / isolation / zero-flow families mop up r?

If a non-motion family gives a clearly positive out-of-fold R^2, that's a new
feature worth folding into g. If all ~0, ranking (rho~0.4) is the ceiling and the
residual is training-dynamics, not characterizable from dataset statistics.

Usage:
    python scripts/transfer_analysis_v4/residual_feature_search.py --split LOTO
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV


def family_cols(cols):
    fam = {"appearance(dino)": [], "density/size": [], "flow_isolation": [],
           "dino_isolation": [], "zero_flow": [], "motion(g-saw)[control]": []}
    for c in cols:
        cl = c.lower()
        if c in ("auc_normalized", "peak_pck", "context_id", "train_dataset",
                 "benchmark", "model_family", "pretrained", "freeze",
                 "random_train", "random_eval"):
            continue
        if "isolation" in cl and "dino" in cl:        fam["dino_isolation"].append(c)
        elif "isolation" in cl and "flow" in cl:      fam["flow_isolation"].append(c)
        elif "zero_image_frac" in cl:                 fam["zero_flow"].append(c)
        elif "dino" in cl:                            fam["appearance(dino)"].append(c)
        elif cl.startswith("log_") or "vectors" in cl or "samples" in cl:
            fam["density/size"].append(c)
        elif "flow" in cl:                            fam["motion(g-saw)[control]"].append(c)
    return {k: v for k, v in fam.items() if v}


def within_demean(df, cols, group="context_id"):
    out = df.copy()
    for c in cols + ["r"]:
        out[c] = out[c] - out.groupby(group)[c].transform("mean")
    return out


def loso_r2(df, cols):
    """Leave-one-source-out ridge of (within-cell-demeaned) r on cols. Out-of-fold R^2."""
    X = df[cols].to_numpy(float)
    # impute NaN with column mean (fit later restricts to train, but mean is benign here)
    yhat = np.full(len(df), np.nan)
    srcs = df["train_dataset"].to_numpy()
    for s in np.unique(srcs):
        tr, te = srcs != s, srcs == s
        Xtr, Xte = X[tr].copy(), X[te].copy()
        mu = np.nanmean(Xtr, axis=0); sd = np.nanstd(Xtr, axis=0); sd[sd == 0] = 1
        for M in (Xtr, Xte):
            inds = np.where(np.isnan(M))
            M[inds] = np.take(mu, inds[1])
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        reg = RidgeCV(alphas=[0.1, 1, 10, 100, 1000]).fit(Xtr, df["r"].to_numpy()[tr])
        yhat[te] = reg.predict(Xte)
    y = df["r"].to_numpy()
    m = np.isfinite(yhat)
    ss_res = float(((y[m] - yhat[m]) ** 2).sum()); ss_tot = float((y[m] ** 2).sum())
    from scipy.stats import spearmanr
    rho = spearmanr(yhat[m], y[m])[0]
    return 1 - ss_res / (ss_tot + 1e-9), float(rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="LOTO", choices=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--rows", default="results_mixed/predictions/peak_pck")
    ap.add_argument("--table", default="../transfer_analysis_v3/transfer_table.csv")
    args = ap.parse_args()

    rd = Path(args.rows)
    rows = pd.read_csv(rd / f"rows_{args.split}_motion.csv")
    rows = rows[rows.variant.str.startswith(("catspp", "raft"))].copy()
    rows["r"] = rows["actual"] - rows["L"] - rows["g"]   # residual g didn't explain

    tab = pd.read_csv(Path(args.rows).parents[2] / "transfer_analysis_v3" / "transfer_table.csv") \
          if not Path(args.table).exists() else pd.read_csv(args.table)
    feat_cols = [c for c in tab.columns if c not in
                 ("auc_normalized", "peak_pck", "context_id")]
    feats = tab[["train_dataset", "benchmark"] +
                [c for c in feat_cols if c not in ("train_dataset", "benchmark")]] \
            .drop_duplicates(["train_dataset", "benchmark"])
    df = rows.merge(feats, on=["train_dataset", "benchmark"], how="left", suffixes=("", "_f"))

    fams = family_cols(tab.columns)
    # within-cell demean once on the union of all used cols
    allcols = sorted({c for v in fams.values() for c in v if c in df.columns})
    df = within_demean(df, allcols)

    print(f"\n{'='*70}\nRESIDUAL FEATURE SEARCH  ({args.split}, motion residual r=actual-L-g)\n"
          f"{len(df)} rows, {df.train_dataset.nunique()} sources. "
          f"Total within-cell residual var = {float((df['r']**2).mean()):.1f}\n{'='*70}")
    print(f"\n{'feature family':28s} {'n_feat':>6s} {'oof_R2':>8s} {'oof_rho':>8s}")
    print("-" * 54)
    res = []
    for name, cols in fams.items():
        cols = [c for c in cols if c in df.columns and df[c].notna().any()]
        if not cols:
            continue
        r2, rho = loso_r2(df, cols)
        res.append((name, len(cols), r2, rho))
        print(f"{name:28s} {len(cols):6d} {r2:8.3f} {rho:8.3f}")

    # combined non-motion families
    noncon = [c for n, cols in fams.items() if "control" not in n
              for c in cols if c in df.columns and df[c].notna().any()]
    if noncon:
        r2, rho = loso_r2(df, noncon)
        print("-" * 54)
        print(f"{'ALL non-motion combined':28s} {len(noncon):6d} {r2:8.3f} {rho:8.3f}")

    print("\nReads: control 'motion(g-saw)' ~0 confirms g already used those.")
    print("Any non-motion family with oof_R2 >> 0 = a real new magnitude feature.")
    print("All ~0 (or negative) = within-cell magnitude is NOT in dataset statistics ->")
    print("ranking (rho~0.4) is the ceiling; residual is training dynamics.")


if __name__ == "__main__":
    main()
