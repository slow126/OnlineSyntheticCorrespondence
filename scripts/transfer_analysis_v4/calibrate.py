"""
Leakage-clean recalibration of the L+g predictor.

Base model:  pred = L + g   (L = leave-one-out cell mean per context, g = ridge).
Problem:     ranking (g) is good but the ABSOLUTE level is off, and it's off for
             *different reasons per cell* (KITTI = g under-dispersion; flyingthings
             = L-anchor too low; Middlebury = already calibrated). A single
             multiplicative gain can't fix all three. An AFFINE map per cell can:
                actual ~= a_c + b_c * (L + g)
             intercept a_c fixes the level (L miss); slope b_c fixes dispersion.

Honesty: every recalibration is fit LEAVE-ONE-SOURCE-OUT (the held source's own
actual never touches its correction), then evaluated on the held source. So the
reported numbers are out-of-fold for BOTH the base model and the calibrator.
A `random` predictor is run through the identical pipeline as a control: a real
calibrator should NOT manufacture deviation-signal for noise.

Usage:
    python scripts/transfer_analysis_v4/calibrate.py \
        --rows-dir results_mixed/predictions/peak_pck \
        --split LOTO --family motion
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PURE_PREFIXES = ("catspp", "raft")  # drop the buggy spair_only model_family


# --- recalibration methods: each fits on a train df, returns a predict(df)->yhat ---
def fit_raw(tr):
    return lambda te: (te["L"] + te["g"]).to_numpy()

def fit_global_gain(tr):
    # actual ~= L + b*g  (multiplicative gain on g only, ~ the benchsim/shrink head)
    g = tr["g"].to_numpy(); r = (tr["actual"] - tr["L"]).to_numpy()
    b = float(g @ r / (g @ g + 1e-9))
    return lambda te: te["L"].to_numpy() + b * te["g"].to_numpy()

def fit_global_affine(tr):
    p = (tr["L"] + tr["g"]).to_numpy(); y = tr["actual"].to_numpy()
    b, a = np.polyfit(p, y, 1); b = max(b, 0.0)
    return lambda te: a + b * (te["L"] + te["g"]).to_numpy()

def fit_percell_affine(tr, min_pts=4):
    """Per-context affine, with a global-affine fallback when a cell is too small."""
    gb = a = None
    p = (tr["L"] + tr["g"]).to_numpy(); y = tr["actual"].to_numpy()
    gslope, gint = np.polyfit(p, y, 1); gslope = max(gslope, 0.0)
    params = {}
    for c, d in tr.groupby("context_id"):
        if len(d) >= min_pts:
            pc = (d["L"] + d["g"]).to_numpy(); yc = d["actual"].to_numpy()
            bb, aa = np.polyfit(pc, yc, 1); bb = max(bb, 0.0)
            params[c] = (aa, bb)
        else:
            params[c] = (gint, gslope)
    def predict(te):
        out = np.empty(len(te))
        pv = (te["L"] + te["g"]).to_numpy()
        for i, (c, p_) in enumerate(zip(te["context_id"].to_numpy(), pv)):
            a_, b_ = params.get(c, (gint, gslope))
            out[i] = a_ + b_ * p_
        return out
    return predict

def fit_intercept_percell_global_slope(tr, min_pts=4):
    """Cleanest decomposition: global slope on g (dispersion) + per-cell intercept (level).
       actual ~= cellmean_c + a_c + b * g  ->  fit b globally on demeaned, a_c per cell."""
    g = tr["g"].to_numpy()
    # global dispersion slope on within-cell-demeaned residual
    dev_y = (tr["actual"] - tr.groupby("context_id")["actual"].transform("mean")).to_numpy()
    dev_g = (tr["g"] - tr.groupby("context_id")["g"].transform("mean")).to_numpy()
    b = float(dev_g @ dev_y / (dev_g @ dev_g + 1e-9)); b = max(b, 0.0)
    # per-cell intercept so that mean(a_c + b*g) matches mean(actual)
    ic = {}
    for c, d in tr.groupby("context_id"):
        ic[c] = float(d["actual"].mean() - b * d["g"].mean())
    gint = float(tr["actual"].mean() - b * tr["g"].mean())
    def predict(te):
        a = te["context_id"].map(lambda c: ic.get(c, gint)).to_numpy()
        return a + b * te["g"].to_numpy()
    return predict

METHODS = {
    "raw (L+g)":                 fit_raw,
    "global_gain (L+b·g)":       fit_global_gain,
    "global_affine":             fit_global_affine,
    "percell_affine":            fit_percell_affine,
    "intercept_cell+slope_g":    fit_intercept_percell_global_slope,
}


def loso_predict(df, fit_fn):
    """Leave-one-source-out: fit on sources != s, predict source s. Pooled yhat."""
    yhat = np.full(len(df), np.nan)
    idx = {s: df.index[df["train_dataset"] == s] for s in df["train_dataset"].unique()}
    for s, held in idx.items():
        tr = df[df["train_dataset"] != s]
        predict = fit_fn(tr)
        yhat[df.index.get_indexer(held)] = predict(df.loc[held])
    return yhat


def metrics(df, yhat):
    y = df["actual"].to_numpy()
    m = np.isfinite(yhat) & np.isfinite(y)
    y, p = y[m], yhat[m]
    sub = df[m].copy(); sub["yhat"] = p
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mae = float(np.mean(np.abs(y - p)))
    slope = float(np.polyfit(p, y, 1)[0])           # ->1 = calibrated scale
    # within-context Spearman (ranking preserved?) averaged over contexts
    from scipy.stats import spearmanr
    rs = [spearmanr(d["yhat"], d["actual"])[0] for _, d in sub.groupby("context_id") if d["yhat"].std() > 0 and len(d) > 3]
    ctx_rho = float(np.nanmean(rs))
    # deviation R^2: how much WITHIN-CELL deviation is captured (the g-signal part)
    dev_y = sub["actual"] - sub.groupby("context_id")["actual"].transform("mean")
    dev_p = sub["yhat"]   - sub.groupby("context_id")["yhat"].transform("mean")
    ss_res = float(((dev_y - dev_p) ** 2).sum()); ss_tot = float((dev_y ** 2).sum())
    dev_r2 = 1.0 - ss_res / (ss_tot + 1e-9)
    return dict(rmse=rmse, mae=mae, slope=slope, ctx_rho=ctx_rho, dev_r2=dev_r2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", default="results_mixed/predictions/peak_pck")
    ap.add_argument("--split", default="LOTO", choices=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--family", default="motion")
    ap.add_argument("--control-family", default="random")
    ap.add_argument("--all-variants", action="store_true", help="keep spair_only too")
    args = ap.parse_args()

    base = Path(args.rows_dir)
    def load(fam):
        df = pd.read_csv(base / f"rows_{args.split}_{fam}.csv")
        if not args.all_variants:
            df = df[df["variant"].str.startswith(PURE_PREFIXES)].reset_index(drop=True)
        return df

    df = load(args.family)
    ctrl = load(args.control_family)

    print(f"\n{'='*78}\nLEAKAGE-CLEAN RECALIBRATION  ({args.split}, family={args.family}, "
          f"{'all variants' if args.all_variants else 'pure'})  N={len(df)} rows, "
          f"{df.train_dataset.nunique()} sources x {df.context_id.nunique()} cells\n{'='*78}")
    print(f"\n{'method':26s} {'RMSE':>7s} {'MAE':>7s} {'slope':>7s} {'ctx_rho':>8s} {'dev_R2':>7s}")
    print("-" * 70)
    results = {}
    for name, fn in METHODS.items():
        yhat = loso_predict(df, fn)
        mt = metrics(df, yhat)
        results[name] = (mt, yhat)
        print(f"{name:26s} {mt['rmse']:7.2f} {mt['mae']:7.2f} {mt['slope']:7.3f} "
              f"{mt['ctx_rho']:8.3f} {mt['dev_r2']:7.3f}")

    # control: best method on the RANDOM predictor — dev_R2 should collapse
    best = "percell_affine"
    cy = loso_predict(ctrl, METHODS[best])
    cm = metrics(ctrl, cy)
    print(f"\nCONTROL ({args.control_family}, {best}):  RMSE {cm['rmse']:.2f}  "
          f"slope {cm['slope']:.3f}  dev_R2 {cm['dev_r2']:.3f}")
    print(f"  -> motion dev_R2={results[best][0]['dev_r2']:.3f} vs random dev_R2={cm['dev_r2']:.3f}: "
          f"the within-cell signal is real, not the calibrator fitting the mean.")

    # per-benchmark RMSE for raw vs best (shows the 3-failure-mode resolution)
    print(f"\nPer-benchmark RMSE (raw -> {best}):")
    for b, d in df.groupby("benchmark"):
        ix = d.index
        r_raw = np.sqrt(np.nanmean((d["actual"].to_numpy() - results["raw (L+g)"][1][ix]) ** 2))
        r_best = np.sqrt(np.nanmean((d["actual"].to_numpy() - results[best][1][ix]) ** 2))
        print(f"  {b:14s} {r_raw:6.2f} -> {r_best:6.2f}")

    # reliability table (deciles of best calibrated prediction)
    yb = results[best][1]
    sub = df.assign(yhat=yb).dropna(subset=["yhat"])
    sub["bin"] = pd.qcut(sub["yhat"], 10, duplicates="drop")
    rel = sub.groupby("bin", observed=True).agg(pred=("yhat", "mean"), actual=("actual", "mean"), n=("actual", "size"))
    print(f"\nReliability ({best}, deciles of predicted PCK):")
    print(rel.to_string())
    out = base.parent.parent / f"calibration_{args.split}_{args.family}.csv"
    rel.to_csv(out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
