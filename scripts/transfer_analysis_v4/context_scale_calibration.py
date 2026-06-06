"""Leakage-clean residual scale calibration diagnostic.

This replays a small v4 slice and asks whether ridge residual magnitudes are
underdispersed because the raw head needs a fold-training scale correction.

Default slice:

    peak_pck, motion family, mean_nn feature subset, pure sources only

For each held-out fold, gains are fit only on that fold's training rows:

    g_global_gain  = global fold residual std / prediction std
    g_variant_gain = same-variant residual std / prediction std
    g_context_gain = same-context residual std / prediction std, fallback variant
    g_shrink_gain  = context gain shrunk toward variant gain
    g_benchsim_gain = gain smoothed over similar benchmarks with same variant
    g_profilesim_gain = gain smoothed over benchmark profile/density similarity

The gains are positive scale factors, so within-context ranking is unchanged
when the same gain is applied to every candidate in a context. This script is
therefore a calibration diagnostic, not a new ranking head.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "transfer_analysis_v3"))
sys.path.insert(0, str(ROOT / "scripts" / "transfer_analysis_v4"))

from experiments import (  # noqa: E402
    _fit_ridge,
    _predict_ridge,
    add_random_features,
    feature_cols,
    PURE_TRAIN_DATASETS,
)
from transfer_predictor_prototype import add_selfdist_features, variant_key  # noqa: E402
from triangle_prior_prototype import build_lookup, idw_weights, SIMILARITY_METRICS  # noqa: E402


EPS = 1e-9


def _std_gain(y: np.ndarray, p: np.ndarray, max_gain: float) -> float | None:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return None
    sy = float(np.std(y[m], ddof=1))
    sp = float(np.std(p[m], ddof=1))
    if sy < EPS or sp < EPS:
        return None
    return float(np.clip(sy / sp, 0.0, max_gain))


def _gain_map(df: pd.DataFrame, key: str, max_gain: float) -> dict[str, float]:
    out = {}
    for name, g in df.groupby(key):
        gain = _std_gain(g["y"].to_numpy(float), g["g_raw"].to_numpy(float), max_gain)
        if gain is not None:
            out[str(name)] = gain
    return out


def _context_centered_pearson(df: pd.DataFrame, pred_col: str) -> float:
    c = df.copy()
    c["ar"] = c["actual"] - c.groupby("context_id")["actual"].transform("mean")
    c["pr"] = c[pred_col] - c.groupby("context_id")[pred_col].transform("mean")
    m = np.isfinite(c["ar"]) & np.isfinite(c["pr"])
    if m.sum() < 3 or c.loc[m, "pr"].std() < EPS:
        return float("nan")
    return float(pearsonr(c.loc[m, "pr"], c.loc[m, "ar"])[0])


def _within_context_spearman(df: pd.DataFrame, pred_col: str) -> float:
    vals = []
    for _, g in df.groupby("context_id"):
        if g["train_dataset"].nunique() < 3 or g[pred_col].std() < EPS:
            continue
        rho = spearmanr(g["actual"], g[pred_col]).statistic
        if np.isfinite(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else float("nan")


def _median_std_ratio(df: pd.DataFrame, pred_col: str) -> float:
    vals = []
    for _, g in df.groupby("context_id"):
        y_std = float(g["actual"].std(ddof=1))
        p_std = float(g[pred_col].std(ddof=1))
        if y_std > EPS and p_std > EPS:
            vals.append(p_std / y_std)
    return float(np.median(vals)) if vals else float("nan")


def _pooled_std_ratio(df: pd.DataFrame, pred_col: str) -> float:
    c = df.copy()
    c["ar"] = c["actual"] - c.groupby("context_id")["actual"].transform("mean")
    c["pr"] = c[pred_col] - c.groupby("context_id")[pred_col].transform("mean")
    sy = float(c["ar"].std(ddof=1))
    sp = float(c["pr"].std(ddof=1))
    return sp / sy if sy > EPS else float("nan")


def _summarize(df: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in pred_cols:
        abs_pred = df["L"] + df[col]
        m = np.isfinite(abs_pred) & np.isfinite(df["actual"])
        abs_r = float("nan")
        if m.sum() >= 3 and abs_pred[m].std() > EPS:
            abs_r = float(pearsonr(abs_pred[m], df.loc[m, "actual"])[0])
        rows.append({
            "head": col,
            "ctx_spearman": _within_context_spearman(df, col),
            "ctx_pearson": _context_centered_pearson(df, col),
            "median_std_ratio": _median_std_ratio(df, col),
            "pooled_std_ratio": _pooled_std_ratio(df, col),
            "abs_r_L_plus_head": abs_r,
            "mean_abs_head": float(np.nanmean(np.abs(df[col]))),
        })
    return pd.DataFrame(rows)


def _prepare_table(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(args.table)
    if args.pure_only:
        table = table[table["train_dataset"].isin(PURE_TRAIN_DATASETS)].copy()
    table["variant"] = table.apply(variant_key, axis=1)
    if args.exclude_false_true:
        table = table[~table["variant"].str.contains(r"\|False\|True$", regex=True)].copy()
    table["cv"] = table["benchmark"] + "|" + table["variant"]
    dist = pd.read_csv(args.dist)
    table = add_selfdist_features(table, dist)
    table = add_random_features(table, seed=42)
    table = table.dropna(subset=[args.target]).copy()
    table["auc_normalized"] = table[args.target]
    return table, dist


def _build_eval_profile_lookup(table: pd.DataFrame) -> dict[tuple[str, str], float]:
    cols = [
        c for c in (
            "log_eval_n_samples",
            "log_eval_n_vectors",
            "log_eval_valid_vectors_per_sample_capped",
            "log_eval_valid_vectors_mean",
            "log_eval_valid_vectors_p90",
        )
        if c in table.columns
    ]
    if not cols:
        return {}
    bench = table.groupby("benchmark")[cols].first().dropna(how="all")
    X = bench.to_numpy(float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd < EPS] = 1.0
    Z = np.nan_to_num((X - mu) / sd)
    out = {}
    names = bench.index.tolist()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i == j:
                continue
            out[(a, b)] = float(np.linalg.norm(Z[i] - Z[j]))
    return out


def _fit_fold(infold: pd.DataFrame, held: pd.DataFrame, f_cols: list[str],
              max_gain: float, shrink_k: float,
              missing_xmean: str = "held"):
    cm = infold.groupby("cv")["auc_normalized"].mean().to_dict()
    xmean = {cv: infold.loc[infold.cv == cv, f_cols].mean()
             for cv in infold.cv.unique()}
    grand_x = infold[f_cols].mean()
    for cv in held.cv.unique():
        if cv not in xmean:
            if missing_xmean == "grand":
                xmean[cv] = grand_x
            else:
                xmean[cv] = held.loc[held.cv == cv, f_cols].mean()

    Xtr = np.asarray([
        [getattr(r, c) - xmean[r.cv][c] for c in f_cols]
        for r in infold.itertuples()
    ], float)
    ytr = np.asarray([
        float(r.auc_normalized) - cm[r.cv]
        for r in infold.itertuples()
    ], float)
    model = _fit_ridge(Xtr, ytr)
    gtr = _predict_ridge(model, Xtr)

    tr = pd.DataFrame({
        "cv": infold["cv"].to_numpy(),
        "variant": infold["variant"].to_numpy(),
        "y": ytr,
        "g_raw": gtr,
    })
    global_gain = _std_gain(ytr, gtr, max_gain) or 1.0
    variant_gain = _gain_map(tr, "variant", max_gain)
    context_gain = _gain_map(tr, "cv", max_gain)
    variant_counts = tr.groupby("variant").size().to_dict()
    context_counts = tr.groupby("cv").size().to_dict()

    Xte = np.asarray([
        [getattr(r, c) - xmean[r.cv][c] for c in f_cols]
        for r in held.itertuples()
    ], float)
    gte = _predict_ridge(model, Xte) if len(Xte) else np.array([])
    return {
        "cm": cm,
        "global_gain": global_gain,
        "variant_gain": variant_gain,
        "context_gain": context_gain,
        "variant_counts": variant_counts,
        "context_counts": context_counts,
        "g_test": gte,
    }


def _benchsim_gain(benchmark: str, variant: str, fit: dict,
                   ee: dict, ee_is_sim: bool) -> float:
    """IDW average of gains from nearby benchmarks with the same variant.

    Same-benchmark contexts are excluded, so this tests whether benchmark motion
    similarity predicts calibration scale rather than memorizing target scale.
    """
    ds, gs = [], []
    suffix = "|" + variant
    for cv, gain in fit["context_gain"].items():
        if not cv.endswith(suffix):
            continue
        other_benchmark = cv[:-len(suffix)]
        if other_benchmark == benchmark:
            continue
        d = ee.get((benchmark, other_benchmark))
        if d is None or not np.isfinite(d):
            continue
        ds.append(float(d))
        gs.append(float(gain))
    if not ds:
        return fit["variant_gain"].get(variant, fit["global_gain"])
    w = idw_weights(np.asarray(ds, float), ee_is_sim, "idw")
    return float((w * np.asarray(gs, float)).sum() / w.sum())


def _profilesim_gain(benchmark: str, variant: str, fit: dict,
                     profile_dist: dict[tuple[str, str], float]) -> float:
    ds, gs = [], []
    suffix = "|" + variant
    for cv, gain in fit["context_gain"].items():
        if not cv.endswith(suffix):
            continue
        other_benchmark = cv[:-len(suffix)]
        if other_benchmark == benchmark:
            continue
        d = profile_dist.get((benchmark, other_benchmark))
        if d is None or not np.isfinite(d):
            continue
        ds.append(float(d))
        gs.append(float(gain))
    if not ds:
        return fit["variant_gain"].get(variant, fit["global_gain"])
    w = idw_weights(np.asarray(ds, float), False, "idw")
    return float((w * np.asarray(gs, float)).sum() / w.sum())


def _gain_for(cv: str, benchmark: str, variant: str, fit: dict,
              shrink_k: float, ee: dict, ee_is_sim: bool,
              profile_dist: dict[tuple[str, str], float]) -> dict[str, float]:
    global_gain = fit["global_gain"]
    vg = fit["variant_gain"].get(variant, global_gain)
    cg = fit["context_gain"].get(cv, vg)
    n = float(fit["context_counts"].get(cv, 0))
    w = n / (n + shrink_k) if n > 0 else 0.0
    bg = _benchsim_gain(benchmark, variant, fit, ee, ee_is_sim)
    pg = _profilesim_gain(benchmark, variant, fit, profile_dist)
    return {
        "global": global_gain,
        "variant": vg,
        "context": cg,
        "shrink": w * cg + (1.0 - w) * vg,
        "benchsim": bg,
        "profilesim": pg,
    }


def run_loto(table: pd.DataFrame, f_cols: list[str], ee: dict,
             ee_is_sim: bool, profile_dist: dict, args) -> pd.DataFrame:
    rows = []
    for held_src in sorted(table["train_dataset"].unique()):
        infold = table[table["train_dataset"] != held_src]
        held = table[table["train_dataset"] == held_src]
        fit = _fit_fold(infold, held, f_cols, args.max_gain, args.shrink_k)
        gmean = float(infold["auc_normalized"].mean())
        for r, g_raw in zip(held.itertuples(), fit["g_test"]):
            gains = _gain_for(r.cv, r.benchmark, r.variant, fit,
                              args.shrink_k, ee, ee_is_sim, profile_dist)
            L = fit["cm"].get(r.cv, gmean)
            rows.append(_row(r, g_raw, L, gains))
    return pd.DataFrame(rows)


def run_lobo(table: pd.DataFrame, f_cols: list[str], ee: dict,
             ee_is_sim: bool, profile_dist: dict, args) -> pd.DataFrame:
    rows = []
    for held_bench in sorted(table["benchmark"].unique()):
        infold = table[table["benchmark"] != held_bench]
        held = table[table["benchmark"] == held_bench]
        fit = _fit_fold(infold, held, f_cols, args.max_gain, args.shrink_k)
        perf = {(r.train_dataset, r.benchmark, r.variant): float(r.auc_normalized)
                for r in infold.itertuples()}
        fold_benchmarks = sorted(infold["benchmark"].unique())
        gmean = float(infold["auc_normalized"].mean())

        for r, g_raw in zip(held.itertuples(), fit["g_test"]):
            ds, ps = [], []
            for e in fold_benchmarks:
                p = perf.get((r.train_dataset, e, r.variant))
                d = ee.get((r.benchmark, e))
                if p is not None and d is not None and np.isfinite(p) and np.isfinite(d):
                    ds.append(d)
                    ps.append(p)
            if ds:
                w = idw_weights(np.asarray(ds, float), ee_is_sim, "idw")
                L = float((w * np.asarray(ps, float)).sum() / w.sum())
            else:
                L = gmean
            gains = _gain_for(r.cv, r.benchmark, r.variant, fit,
                              args.shrink_k, ee, ee_is_sim, profile_dist)
            rows.append(_row(r, g_raw, L, gains))
    return pd.DataFrame(rows)


def run_joint(table: pd.DataFrame, f_cols: list[str], ee: dict,
              ee_is_sim: bool, profile_dist: dict, args) -> pd.DataFrame:
    rows = []
    for held_src in sorted(table["train_dataset"].unique()):
        for held_bench in sorted(table["benchmark"].unique()):
            infold = table[(table["train_dataset"] != held_src)
                           & (table["benchmark"] != held_bench)]
            held = table[(table["train_dataset"] == held_src)
                         & (table["benchmark"] == held_bench)]
            if infold.empty or held.empty:
                continue
            fit = _fit_fold(infold, held, f_cols, args.max_gain, args.shrink_k,
                            missing_xmean="grand")
            grand = float(infold["auc_normalized"].mean())
            gamma = (infold.groupby("variant")["auc_normalized"].mean() - grand).to_dict()
            for r, g_raw in zip(held.itertuples(), fit["g_test"]):
                L = grand + gamma.get(r.variant, 0.0)
                gains = _gain_for(r.cv, r.benchmark, r.variant, fit,
                                  args.shrink_k, ee, ee_is_sim, profile_dist)
                rows.append(_row(r, g_raw, L, gains))
    return pd.DataFrame(rows)


def _row(r, g_raw: float, L: float, gains: dict[str, float]) -> dict:
    return {
        "train_dataset": r.train_dataset,
        "context_id": r.cv,
        "benchmark": r.benchmark,
        "variant": r.variant,
        "actual": float(r.auc_normalized),
        "L": float(L),
        "g": float(g_raw),
        "g_global_gain": float(g_raw * gains["global"]),
        "g_variant_gain": float(g_raw * gains["variant"]),
        "g_context_gain": float(g_raw * gains["context"]),
        "g_shrink_gain": float(g_raw * gains["shrink"]),
        "g_benchsim_gain": float(g_raw * gains["benchsim"]),
        "g_profilesim_gain": float(g_raw * gains["profilesim"]),
        "gain_global": float(gains["global"]),
        "gain_variant": float(gains["variant"]),
        "gain_context": float(gains["context"]),
        "gain_shrink": float(gains["shrink"]),
        "gain_benchsim": float(gains["benchsim"]),
        "gain_profilesim": float(gains["profilesim"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path,
                    default=ROOT / "scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", type=Path,
                    default=ROOT / "analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--target", default="peak_pck")
    ap.add_argument("--family", default="motion")
    ap.add_argument("--feature-subset", default="mean_nn")
    ap.add_argument("--splits", nargs="+", default=["LOTO", "LOBO", "JOINT"],
                    choices=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--pure-only", action="store_true", default=True)
    ap.add_argument("--include-mixed", dest="pure_only", action="store_false")
    ap.add_argument("--exclude-false-true", action="store_true")
    ap.add_argument("--max-gain", type=float, default=10.0)
    ap.add_argument("--shrink-k", type=float, default=5.0)
    ap.add_argument("--kernel-space", default="flow", choices=["flow", "dino"],
                    help="space for the benchsim IDW kernel (mean_nn_sym between benchmarks)")
    ap.add_argument("--kernel-metric", default="mean_nn_sym",
                    help="distance metric for the benchsim IDW kernel")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "scripts/transfer_analysis_v4/results_fsub_mean_nn"
                    / "context_scale_calibration")
    args = ap.parse_args()

    table, dist = _prepare_table(args)
    f_cols = feature_cols(table, args.family, feature_subset=args.feature_subset)
    if not f_cols:
        raise SystemExit(f"no feature columns for {args.family}/{args.feature_subset}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ee = build_lookup(dist, args.kernel_space, "eval_eval", args.kernel_metric)
    ee_is_sim = args.kernel_metric in SIMILARITY_METRICS
    profile_dist = _build_eval_profile_lookup(table)

    heads = [
        "g",
        "g_global_gain",
        "g_variant_gain",
        "g_context_gain",
        "g_shrink_gain",
        "g_benchsim_gain",
        "g_profilesim_gain",
    ]
    summaries = []
    for split in args.splits:
        if split == "LOTO":
            df = run_loto(table, f_cols, ee, ee_is_sim, profile_dist, args)
        elif split == "LOBO":
            df = run_lobo(table, f_cols, ee, ee_is_sim, profile_dist, args)
        else:
            df = run_joint(table, f_cols, ee, ee_is_sim, profile_dist, args)
        label = "drop_false_true" if args.exclude_false_true else "all_variants"
        df.to_csv(args.out_dir / f"rows_{split}_{label}.csv", index=False)
        sm = _summarize(df, heads)
        sm.insert(0, "split", split)
        sm.insert(1, "variant_filter", label)
        sm.insert(2, "n_rows", len(df))
        sm.insert(3, "n_contexts", df["context_id"].nunique())
        summaries.append(sm)

    out = pd.concat(summaries, ignore_index=True)
    out_path = args.out_dir / (
        "summary_drop_false_true.csv" if args.exclude_false_true else "summary_all_variants.csv"
    )
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
