"""Unified-A sweep: g (within estimator) + L (observed-or-back-off cell band).

Adds two new dimensions on top of the original v4:

- Multiple **targets**: pass `--targets auc_normalized peak_pck`. Predictions
  are written to `predictions/<target>/` so downstream (bootstrap, figures,
  compile) can compare targets side-by-side.
- Multiple **heads**: ridge (default), pairwise RankNet, and ridge with
  per-benchmark gain calibration. All three score columns are written into
  the same per-row CSV so downstream tooling can pick.

One model, four families (motion / appearance / both / random), three splits
(LOTO / LOBO / JOINT), plus shuffle-target leakage control. Writes per-row
prediction CSVs and a point-estimate summary; bootstrap CIs are layered on
later by `bootstrap.py`.

Per-row prediction CSV columns:
    actual, L, g, g_cal, g_rank, train_dataset, context_id, benchmark, variant

- `g`      = ridge prediction (raw)
- `g_cal`  = ridge × per-context gain (gain = std(actual_resid_in_fold) /
             std(g_in_fold) for the held-out fold's training rows; fallback
             to median gain for unseen contexts)
- `g_rank` = pairwise RankNet score (logistic regression on pairwise feature
             differences; rank-equivalent to standard linear RankNet)

Run:
    python scripts/transfer_analysis_v4/experiments.py --targets auc_normalized peak_pck

Optional ablations:
    --family-matched-prior           (appearance gets DINO-IDW for L)
    --no-ranknet                     (skip RankNet head; ridge only)
    --no-zridge                    (skip gain-calibrated ridge column)

Inputs (must exist from v3):
    scripts/transfer_analysis_v3/transfer_table.csv
    analysis_v3/pairwise_self_distances.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "transfer_analysis_v3"))
from triangle_prior_prototype import (  # noqa: E402
    build_lookup, idw_weights, SIMILARITY_METRICS,
)
from transfer_predictor_prototype import (  # noqa: E402
    variant_key, add_selfdist_features, SELFDIST_METRICS,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAMILIES = ["motion", "appearance", "both", "random",
            "density", "motion_density",
            "size", "supervision_density",
            "motion_size", "motion_supdensity",
            # additional motion feature spaces beyond the 13 self-dist set
            "motion_km",          # k-means coverage (6 features)
            "motion_sym",         # symmetric distances combined (FID + SW2 + MMD, 3)
            "motion_mmd",         # flow MMD only (1)
            "motion_fid",         # flow FID only (1)
            "motion_w2",          # flow sliced W2 only (1)
            # additional appearance feature spaces beyond the 13 self-dist set
            "appearance_mmd",     # dino MMD only (1)
            "appearance_nullk",   # null-calibrated DINO coverage (8 features)
            "appearance_sym",     # DINO symmetric distances combined (FID + SW2 + MMD, 3)
            "appearance_fid",     # DINO FID only (1) — requires dino_fid column
            "appearance_w2",      # DINO sliced W2 only (1) — requires dino_sliced_w2 column
            ]

# The 11 "pure" training sources from the original v4 analysis. Mixed
# variants (e.g. spair_synthetic_70_30) are in the table after the v3
# table rebuild but were not part of the headline scope. Use --pure-only
# to restrict to these.
PURE_TRAIN_DATASETS = [
    "flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
    "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
    "synthetic_random_flipping", "synthetic_small_zoom",
]

# Generator-family grouping for robustness. The 11 "pure" sources are not 11
# independent draws: several are variants from the same generator and share
# appearance/motion statistics, so treating them as independent overstates how
# many effective sources back the result. The honest unit of resampling /
# hold-out is the GENERATOR FAMILY (~5), not the source (11). Used by:
#   - experiments.py --drop-family  (leave-one-generator-family-out, Tier 2)
#   - bootstrap.py    --cluster     (cluster bootstrap CIs, Tier 3)
FAMILY_MAP = {
    "synthetic":                  "sdf3d",     # SDF-3D procedural + its zoom/flip variants
    "synthetic_large_zoom":       "sdf3d",
    "synthetic_small_zoom":       "sdf3d",
    "synthetic_random_flipping":  "sdf3d",
    "synthetic_2d_warp":          "warp2d",    # 2D image-warp augmentations
    "imagenet2dwarp":             "warp2d",
    "movi_f":                     "kubric",    # Kubric / MOVi physical renderer
    "flyingthings":               "realflow",  # real/quasi-real optical-flow datasets
    "pointodyssey":               "realflow",
    "sintel":                     "realflow",
    "spair":                      "semantic",  # semantic-keypoint (no dense flow)
}

def family_of(src: str) -> str:
    """Generator family for a source (falls back to the source name itself)."""
    return FAMILY_MAP.get(src, src)

SPLITS = ["LOTO", "LOBO", "JOINT"]
N_RANDOM_FEATS = 13            # dim-match to motion / appearance

# Per-family similarity space for L. Random / density have no meaningful
# similarity space ("none") and always use the uninformed level. Other
# families use their own feature space (so appearance doesn't piggyback on
# flow). See L_MODE below for what "informed" means.
FAMILY_SPACE = {
    "motion":               "flow",
    "appearance":           "dino",
    "both":                 "flow",  # motion+appearance; flow is the discriminator
    "random":               None,
    "density":              None,
    "size":                 None,
    "supervision_density":  None,
    "motion_density":       "flow",
    "motion_size":          "flow",
    "motion_supdensity":    "flow",
    # New motion variants — still flow space conceptually
    "motion_km":            "flow",
    "motion_sym":           "flow",
    "motion_mmd":           "flow",
    "motion_fid":           "flow",
    "motion_w2":            "flow",
    # New appearance variants — DINO space
    "appearance_mmd":       "dino",
    "appearance_fid":       "dino",
    "appearance_w2":        "dino",
    "appearance_sym":       "dino",
    "appearance_nullk":     "dino",
}

# Three L modes for ablating "feature-informed level":
#   mixed                  — LOTO=cell_mean(uninformed), LOBO=per-family IDW
#                            (current main; asymmetric across CV regimes)
#   symmetric_informed     — LOTO=per-family sim_train_IDW (over source-source
#                            similarity), LOBO=per-family sim_eval_IDW.
#                            Symmetric: both regimes use feature-informed L.
#   symmetric_uninformed   — LOTO=cell_mean, LOBO=uniform.
#                            Symmetric: neither uses feature similarity for L.
# Families with FAMILY_SPACE=None (random/density) always fall back to the
# uninformed level (cell_mean / uniform) regardless of mode.
L_MODES = ["mixed", "symmetric_informed", "symmetric_uninformed",
           "targeted_informed", "eb_shrunk", "density_idw"]

# Feature subsets — used by --feature-subset (affects f_cols globally so
# g/zridge/ranknet/gbm/targeted_idw all see only those columns) and by
# --targeted-subset (further restricts which dimensions enter the
# targeted_idw distance norm).
#
# Coarse subsets within the 13 SELFDIST_METRICS:
#   all          — all 13 features
#   mean_nn      — mean_nn × {sym, a→b, b→a}     (3)
#   coverage     — eps × {1,4,16 px} × {a→b, b→a} (6)
#   kl           — kl × {k5, k20} × {a→b, b→a}    (4)
#   asym_only    — drop the 1 _sym feature        (12)
#
# Fine-grained subsets for "test each metric individually":
#   mean_nn_sym  — only the symmetric mean_nn       (1)
#   mean_nn_asym — directional mean_nn (no _sym)   (2)
#   eps_1px      — only 1px coverage               (2)
#   eps_4px      — only 4px coverage               (2)
#   eps_16px     — only 16px coverage              (2)
#   kl_k5        — only k=5 KL                     (2)
#   kl_k20       — only k=20 KL                    (2)
SUBSET_NAMES = ["all",
                # coarse
                "mean_nn", "coverage", "kl", "asym_only",
                # fine
                "mean_nn_sym", "mean_nn_asym",
                "eps_1px", "eps_4px", "eps_16px",
                "kl_k5", "kl_k20"]


def _apply_subset(cols: list[str], subset: str) -> list[str]:
    if subset == "all":
        return cols
    if subset == "asym_only":
        out = [c for c in cols if "_sym" not in c]
    elif subset == "mean_nn":
        out = [c for c in cols if "mean_nn" in c]
    elif subset == "mean_nn_sym":
        out = [c for c in cols if "mean_nn" in c and c.endswith("_sym")]
    elif subset == "mean_nn_asym":
        out = [c for c in cols if "mean_nn" in c and not c.endswith("_sym")]
    elif subset == "coverage":
        out = [c for c in cols if "covered_by" in c]
    elif subset == "eps_1px":
        out = [c for c in cols if "eps1px" in c]
    elif subset == "eps_4px":
        out = [c for c in cols if "eps4px" in c]
    elif subset == "eps_16px":
        out = [c for c in cols if "eps16px" in c]
    elif subset == "kl":
        out = [c for c in cols if "_kl_" in c or c.startswith("kl_")]
    elif subset == "kl_k5":
        out = [c for c in cols if ("_kl_" in c or c.startswith("kl_")) and "_k5" in c]
    elif subset == "kl_k20":
        out = [c for c in cols if ("_kl_" in c or c.startswith("kl_")) and "_k20" in c]
    else:
        raise ValueError(f"unknown subset: {subset}")
    # Fall back to original if no columns match (density/random naming)
    return out if len(out) >= 1 else cols


def targeted_subset_indices(f_cols: list[str], subset: str) -> list[int]:
    """Within an already-subset f_cols, indices that further restrict the
    targeted_idw norm. If --feature-subset and --targeted-subset agree, this
    returns all indices."""
    if subset == "all":
        return list(range(len(f_cols)))
    keep = set(_apply_subset(f_cols, subset))
    return [i for i, c in enumerate(f_cols) if c in keep]


def level_config_for(fam: str, mode: str) -> tuple[str, str, str]:
    """Returns (lobo_space, lobo_kind, loto_kind).
    space ∈ {"flow", "dino", "density", "none"}
    kind  ∈ {"idw","uniform","cell_mean","sim_idw","targeted_idw","eb_shrunk"}
      cell_mean    — LOTO uninformed (in-fold cell mean)
      sim_idw      — LOTO informed via train-train distance (k-agnostic)
                     in the same `space` as the family
      targeted_idw — LOTO informed via similarity of the (i→k) feature
                     vectors themselves; k-conditioned, multi-metric,
                     directional (uses asymmetric mean_nn, eps coverage,
                     KL — same 13-dim space as g)
      eb_shrunk    — LOTO uses Empirical Bayes shrinkage of cell_mean
                     toward grand_mean (Efron-Morris 1973). Attenuates
                     the LOO anti-correlation by mixing with the grand
                     mean using λ = σ²_b / (σ²_b + σ²_w / n).
    """
    space = FAMILY_SPACE.get(fam) or "none"
    if mode == "density_idw":
        # Use density-based IDW for both LOTO and LOBO, ignoring family space.
        # Random/density families have no meaningful "their own distance space"
        # so they still fall back to uninformed.
        if space == "none":
            return ("none", "uniform", "cell_mean")
        return ("density", "idw", "sim_idw")
    if mode == "eb_shrunk":
        # Empirical-Bayes shrinkage of cell_mean toward grand_mean for LOTO.
        # LOBO unchanged (per-family IDW).
        if space == "none":
            return (space, "uniform", "eb_shrunk")
        return (space, "idw", "eb_shrunk")
    if mode == "symmetric_uninformed" or space == "none":
        return (space, "uniform", "cell_mean")
    if mode == "symmetric_informed":
        return (space, "idw", "sim_idw")
    if mode == "targeted_informed":
        return (space, "idw", "targeted_idw")
    # mixed (default)
    return (space, "idw", "cell_mean")

DENSITY_COLS = [
    "log_train_n_samples",
    "log_train_n_vectors",
    "log_train_valid_vectors_per_sample_capped",
    "log_train_valid_vectors_mean",
    "log_train_valid_vectors_p90",
    "log_eval_n_samples",
    "log_eval_n_vectors",
    "log_eval_valid_vectors_per_sample_capped",
    "log_eval_valid_vectors_mean",
    "log_eval_valid_vectors_p90",
]

# Pure dataset-size proxies — totals, no per-sample normalization.
# Lets us isolate "this is just a 'bigger dataset transfers better' story."
SIZE_COLS = [
    "log_train_n_samples",
    "log_train_n_vectors",
    "log_eval_n_samples",
    "log_eval_n_vectors",
]

# Per-sample supervision density — sparse vs dense supervision regime.
# A spair-trained model gets ~12 keypoints/pair; a flyingthings-trained model
# gets ~250k flow vectors/pair. This family isolates that effect from totals.
SUPERVISION_DENSITY_COLS = [
    "log_train_valid_vectors_per_sample_capped",
    "log_train_valid_vectors_mean",
    "log_train_valid_vectors_p90",
    "log_eval_valid_vectors_per_sample_capped",
    "log_eval_valid_vectors_mean",
    "log_eval_valid_vectors_p90",
]

# k-means-weighted flow coverage features (already in transfer_table from the
# v3 coverage_v2_flow_only_raw_joint_kmeans_full.csv). 6 features: 3 scales × 2 dirs.
MOTION_KM_COLS = [
    "flow_km_eval_covered_by_train_eps1px_weighted",
    "flow_km_train_covered_by_eval_eps1px_weighted",
    "flow_km_eval_covered_by_train_eps4px_weighted",
    "flow_km_train_covered_by_eval_eps4px_weighted",
    "flow_km_eval_covered_by_train_eps16px_weighted",
    "flow_km_train_covered_by_eval_eps16px_weighted",
]

# Symmetric distance metrics — single-feature families for individual
# metric ablation. From v3's symmetric_distances.csv + flow_mmd_results.
MOTION_SYM_COLS = ["flow_fid", "flow_sliced_w2", "flow_mmd"]
MOTION_MMD_COLS = ["flow_mmd"]
MOTION_FID_COLS = ["flow_fid"]
MOTION_W2_COLS = ["flow_sliced_w2"]

APPEARANCE_MMD_COLS = ["dino_mmd"]
APPEARANCE_FID_COLS = ["dino_fid"]
APPEARANCE_W2_COLS = ["dino_sliced_w2"]
APPEARANCE_SYM_COLS = ["dino_fid", "dino_sliced_w2", "dino_mmd"]
# Null-calibrated DINO coverage (8 features: 4 thresholds × 2 dirs).
# These were the original DINO coverage features in v3 before the self-dist
# refactor brought in eps coverage.
APPEARANCE_NULLK_COLS = [
    "dino_eval_covered_by_train_null80",
    "dino_train_covered_by_eval_null80",
    "dino_eval_covered_by_train_null90",
    "dino_train_covered_by_eval_null90",
    "dino_eval_covered_by_train_null95",
    "dino_train_covered_by_eval_null95",
    "dino_eval_covered_by_train_null99",
    "dino_train_covered_by_eval_null99",
]


def feature_cols(table: pd.DataFrame, family: str,
                 feature_subset: str = "all") -> list[str]:
    if family == "motion":
        cols = [f"se_flow_{m}" for m in SELFDIST_METRICS
                if f"se_flow_{m}" in table.columns]
    elif family == "appearance":
        cols = [f"se_dino_{m}" for m in SELFDIST_METRICS
                if f"se_dino_{m}" in table.columns]
    elif family == "both":
        cols = (feature_cols(table, "motion", feature_subset)
                + feature_cols(table, "appearance", feature_subset))
        return cols  # already subset-applied
    elif family == "random":
        cols = [c for c in table.columns if c.startswith("rnd_")]
    elif family == "density":
        cols = [c for c in DENSITY_COLS if c in table.columns]
    elif family == "size":
        cols = [c for c in SIZE_COLS if c in table.columns]
    elif family == "supervision_density":
        cols = [c for c in SUPERVISION_DENSITY_COLS if c in table.columns]
    elif family == "motion_density":
        cols = (feature_cols(table, "motion", feature_subset)
                + feature_cols(table, "density"))
        return cols
    elif family == "motion_size":
        cols = (feature_cols(table, "motion", feature_subset)
                + feature_cols(table, "size"))
        return cols
    elif family == "motion_supdensity":
        cols = (feature_cols(table, "motion", feature_subset)
                + feature_cols(table, "supervision_density"))
        return cols
    elif family == "motion_km":
        cols = [c for c in MOTION_KM_COLS if c in table.columns]
    elif family == "motion_sym":
        cols = [c for c in MOTION_SYM_COLS if c in table.columns]
    elif family == "motion_mmd":
        cols = [c for c in MOTION_MMD_COLS if c in table.columns]
    elif family == "motion_fid":
        cols = [c for c in MOTION_FID_COLS if c in table.columns]
    elif family == "motion_w2":
        cols = [c for c in MOTION_W2_COLS if c in table.columns]
    elif family == "appearance_mmd":
        cols = [c for c in APPEARANCE_MMD_COLS if c in table.columns]
    elif family == "appearance_fid":
        cols = [c for c in APPEARANCE_FID_COLS if c in table.columns]
    elif family == "appearance_w2":
        cols = [c for c in APPEARANCE_W2_COLS if c in table.columns]
    elif family == "appearance_sym":
        cols = [c for c in APPEARANCE_SYM_COLS if c in table.columns]
    elif family == "appearance_nullk":
        cols = [c for c in APPEARANCE_NULLK_COLS if c in table.columns]
    else:
        raise ValueError(f"unknown family: {family}")
    return _apply_subset(cols, feature_subset)


def add_random_features(table: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    for j in range(N_RANDOM_FEATS):
        table[f"rnd_{j}"] = rng.standard_normal(len(table))
    return table


# ---------------------------------------------------------------------------
# Density-based pairwise distance lookups for the density_idw L mode.
# Uses the same per-dataset features that the `density` family uses (log
# sample count + log valid vectors/sample). After standardization we compute
# pairwise Euclidean distances between sources (for LOTO sim_idw) and between
# benchmarks (for LOBO IDW).
# ---------------------------------------------------------------------------
def _build_density_lookups(table: pd.DataFrame):
    """Return (tt_density, ee_density) — pairwise Euclidean distance dicts in
    standardized density space. Distance interpretation: lower=closer."""
    # Train-side features per source
    src_cols = [
        c for c in ("log_train_n_samples", "log_train_n_vectors",
                    "log_train_valid_vectors_per_sample_capped",
                    "log_train_valid_vectors_mean", "log_train_valid_vectors_p90")
        if c in table.columns
    ]
    src_table = (table.groupby("train_dataset")[src_cols].first()
                       .dropna(how="all"))
    # Eval-side features per benchmark
    eval_cols = [
        c for c in ("log_eval_n_samples", "log_eval_n_vectors",
                    "log_eval_valid_vectors_per_sample_capped",
                    "log_eval_valid_vectors_mean", "log_eval_valid_vectors_p90")
        if c in table.columns
    ]
    bench_table = (table.groupby("benchmark")[eval_cols].first()
                         .dropna(how="all"))

    def _pairwise_euclid(df: pd.DataFrame) -> dict:
        if df.empty:
            return {}
        X = df.values.astype(float)
        # Standardize by feature, ignoring NaN
        mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0)
        sd[sd < 1e-9] = 1.0
        Z = (X - mu) / sd
        Z = np.nan_to_num(Z)
        out = {}
        names = df.index.tolist()
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i == j:
                    continue
                d = float(np.linalg.norm(Z[i] - Z[j]))
                out[(a, b)] = d
        return out

    return _pairwise_euclid(src_table), _pairwise_euclid(bench_table)


# ---------------------------------------------------------------------------
# Feature preprocessing with leakage-clean winsorization
# ---------------------------------------------------------------------------
# DINO KL features have heavy tails (movi_f's KL vs others is ~3-4× the typical
# scale; synthetic→synthetic gives negative KL from small-sample estimator
# noise). Without clipping, a handful of rows can dominate the ridge fit,
# producing wildly miscalibrated appearance predictions (std 250+ on JOINT vs
# motion's std 12). We clip per-feature at training-fold 1st/99th percentile
# — leakage-clean (uses only in-fold rows) and modest (~5 rows clipped per
# side out of 500).
WINSOR_LO, WINSOR_HI = 0.01, 0.99


def _winsor_bounds(X: np.ndarray):
    return (np.nanquantile(X, WINSOR_LO, axis=0),
            np.nanquantile(X, WINSOR_HI, axis=0))


def _apply_winsor(X: np.ndarray, q_lo: np.ndarray, q_hi: np.ndarray):
    return np.clip(X, q_lo, q_hi)


# ---------------------------------------------------------------------------
# Core ridge head + within-context fitting
# ---------------------------------------------------------------------------
def _fit_ridge(X: np.ndarray, y: np.ndarray):
    q_lo, q_hi = _winsor_bounds(X)
    Xw = _apply_winsor(X, q_lo, q_hi)
    imp = SimpleImputer(strategy="median").fit(Xw)
    Xi = np.nan_to_num(imp.transform(Xw))
    scl = StandardScaler().fit(Xi)
    Xs = scl.transform(Xi)
    reg = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]).fit(Xs, y)
    return imp, scl, reg, q_lo, q_hi


def _predict_ridge(model, X: np.ndarray) -> np.ndarray:
    imp, scl, reg, q_lo, q_hi = model
    Xw = _apply_winsor(X, q_lo, q_hi)
    Xi = np.nan_to_num(imp.transform(Xw))
    return reg.predict(scl.transform(Xi))


# ---------------------------------------------------------------------------
# RankNet (pairwise logistic): linear scorer fit on pairwise feature diffs
# ---------------------------------------------------------------------------
def _fit_ranknet(X: np.ndarray, y: np.ndarray, ctx: np.ndarray, model):
    """Pairwise RankNet via logistic regression on within-context pair diffs.
    `model` is the ridge tuple (imp, scl, reg, q_lo, q_hi) — we reuse its
    winsor bounds + scaler so RankNet sees the same feature representation as
    ridge. Returns a weight vector w; predict via Xs @ w."""
    imp, scl, _, q_lo, q_hi = model
    Xs = scl.transform(np.nan_to_num(imp.transform(_apply_winsor(X, q_lo, q_hi))))
    pair_X = []
    pair_y = []
    for c in np.unique(ctx):
        idx = np.where(ctx == c)[0]
        if idx.size < 2:
            continue
        yc = y[idx]; Xc = Xs[idx]
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                if yc[i] == yc[j]:
                    continue
                pair_X.append(Xc[i] - Xc[j])
                pair_y.append(1 if yc[i] > yc[j] else 0)
    if not pair_y:
        return None
    pair_X = np.asarray(pair_X, float)
    pair_y = np.asarray(pair_y, int)
    lr = LogisticRegression(C=1.0, fit_intercept=False, max_iter=1000,
                            solver="lbfgs").fit(pair_X, pair_y)
    return lr.coef_[0]


def _predict_ranknet(w, X: np.ndarray, model) -> np.ndarray:
    if w is None:
        return np.zeros(len(X))
    imp, scl, _, q_lo, q_hi = model
    Xs = scl.transform(np.nan_to_num(imp.transform(_apply_winsor(X, q_lo, q_hi))))
    return Xs @ w


# ---------------------------------------------------------------------------
# GBM head — nonlinear ceiling check
# ---------------------------------------------------------------------------
def _fit_gbm(X: np.ndarray, y: np.ndarray, model):
    """Histogram gradient boosting on the same winsorized+scaled features that
    ridge sees. Conservative hyperparameters for N≈500: depth=4, leaves=15,
    learning_rate=0.05, 200 iters with early stopping on validation. The point
    is a *ceiling* check, not a tuned production model."""
    imp, scl, _, q_lo, q_hi = model
    Xs = scl.transform(np.nan_to_num(imp.transform(_apply_winsor(X, q_lo, q_hi))))
    gbm = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_depth=4,
        max_leaf_nodes=15,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=0,
    ).fit(Xs, y)
    return gbm


def _predict_gbm(gbm, X: np.ndarray, model) -> np.ndarray:
    if gbm is None:
        return np.zeros(len(X))
    imp, scl, _, q_lo, q_hi = model
    Xs = scl.transform(np.nan_to_num(imp.transform(_apply_winsor(X, q_lo, q_hi))))
    return gbm.predict(Xs)


# ---------------------------------------------------------------------------
# Within-context z-score ridge — variance-heterogeneous calibration baked in
# ---------------------------------------------------------------------------
def _fit_zridge(X: np.ndarray, y: np.ndarray, ctx: np.ndarray, model):
    """Fit ridge on within-context target z-scores: each context contributes
    equally to the loss regardless of its raw target variance. This handles
    the case where spair contexts have std(actual)≈0.2 while synthetic
    contexts have std≈13 — a single global slope can't be right for both, so
    we normalize per context, fit one global slope on the normalized data,
    then un-normalize per context at predict time.

    Returns (reg_z, std_y_by_ctx, std_y_fallback)."""
    imp, scl, _, q_lo, q_hi = model
    Xs = scl.transform(np.nan_to_num(imp.transform(_apply_winsor(X, q_lo, q_hi))))
    std_y_by_ctx = {}
    z_y = np.zeros_like(y)
    for c in np.unique(ctx):
        m = ctx == c
        sy = float(y[m].std())
        if sy < 1e-9 or not np.isfinite(sy) or m.sum() < 3:
            z_y[m] = y[m]
            continue
        std_y_by_ctx[c] = sy
        z_y[m] = y[m] / sy
    fb = float(np.median(list(std_y_by_ctx.values()))) if std_y_by_ctx else 1.0
    reg_z = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]).fit(Xs, z_y)
    return reg_z, std_y_by_ctx, fb


def _predict_zridge(reg_z, X: np.ndarray, ctx: np.ndarray,
                    std_y_by_ctx: dict, std_y_fb: float, model) -> np.ndarray:
    imp, scl, _, q_lo, q_hi = model
    Xs = scl.transform(np.nan_to_num(imp.transform(_apply_winsor(X, q_lo, q_hi))))
    z_pred = reg_z.predict(Xs)
    stds = np.array([std_y_by_ctx.get(c, std_y_fb) for c in ctx], float)
    return z_pred * stds


# ---------------------------------------------------------------------------
# LOTO / LOBO CV (held axis observable on the OTHER side)
# ---------------------------------------------------------------------------
def cv_oneaxis(table: pd.DataFrame, f_cols: list[str], hold: str,
               ee: dict, ee_is_sim: bool, level_kind: str = "idw",
               tt: dict | None = None, tt_is_sim: bool = False,
               loto_kind: str = "cell_mean",
               targeted_subset: str = "all",
               do_ranknet: bool = True, do_zridge: bool = True,
               do_gbm: bool = True) -> pd.DataFrame:
    rows = []
    for held_val in sorted(table[hold].unique()):
        infold = table[table[hold] != held_val]
        held = table[table[hold] == held_val]
        if infold.empty or held.empty:
            continue
        assert held_val not in set(infold[hold])

        cm = infold.groupby("cv")["auc_normalized"].mean().to_dict()
        gmean = float(infold["auc_normalized"].mean())
        xmean = {cv: infold.loc[infold.cv == cv, f_cols].mean()
                 for cv in infold.cv.unique()}
        for cv in held.cv.unique():
            if cv not in xmean:
                xmean[cv] = held.loc[held.cv == cv, f_cols].mean()

        Xtr_list, ytr_list, ctx_tr_list = [], [], []
        for r in infold.itertuples():
            xm = xmean[r.cv]
            Xtr_list.append([getattr(r, c) - xm[c] for c in f_cols])
            ytr_list.append(float(r.auc_normalized) - cm[r.cv])
            ctx_tr_list.append(r.cv)
        Xtr = np.asarray(Xtr_list, float)
        ytr = np.asarray(ytr_list, float)
        ctx_tr = np.asarray(ctx_tr_list)

        model = _fit_ridge(Xtr, ytr)
        zridge = None
        if do_zridge:
            zridge = _fit_zridge(Xtr, ytr, ctx_tr, model)
        w_rank = None
        if do_ranknet:
            w_rank = _fit_ranknet(Xtr, ytr, ctx_tr, model)
        gbm = None
        if do_gbm:
            gbm = _fit_gbm(Xtr, ytr, model)

        # L_A: observed-or-back-off cell band.
        perf = {(r.train_dataset, r.benchmark, r.variant): float(r.auc_normalized)
                for r in infold.itertuples()}
        fold_benchmarks = sorted(infold["benchmark"].unique())
        fold_sources = sorted(infold["train_dataset"].unique())

        # EB shrinkage parameters (only used when loto_kind == "eb_shrunk").
        # Compute σ²_b (between-cell variance of cell_means) and σ²_w (mean
        # within-cell variance of the in-fold sources), shared across the fold.
        eb_lambda_by_cv: dict = {}
        if loto_kind == "eb_shrunk":
            cell_means = []
            cell_within_vars = []
            cell_ns = {}
            for cv_key, gp in infold.groupby("cv"):
                vals = gp["auc_normalized"].dropna().values
                if len(vals) < 2:
                    continue
                cell_means.append(float(vals.mean()))
                cell_within_vars.append(float(vals.var(ddof=1)))
                cell_ns[cv_key] = len(vals)
            if cell_means:
                sig_b = float(np.var(cell_means, ddof=1)) if len(cell_means) > 1 else 0.0
                sig_w = float(np.mean(cell_within_vars))
                for cv_key, n in cell_ns.items():
                    denom = sig_b + (sig_w / max(n, 1))
                    eb_lambda_by_cv[cv_key] = sig_b / denom if denom > 1e-12 else 1.0

        # Standardized feature vectors per (source, benchmark) for targeted_idw.
        # We reuse the ridge model's preprocessor (winsor + impute + scale)
        # so the feature space matches what g sees. Raw (un-demeaned) features
        # so each (i, k) has its own absolute coverage profile of k.
        feat_by_sk = {}
        subset_idx: list[int] = []
        if loto_kind == "targeted_idw":
            imp_g, scl_g, _, q_lo_g, q_hi_g = model
            for r in table.itertuples():
                raw = np.asarray([[getattr(r, c) for c in f_cols]], float)
                raw_w = _apply_winsor(raw, q_lo_g, q_hi_g)
                feat_by_sk[(r.train_dataset, r.benchmark)] = \
                    scl_g.transform(np.nan_to_num(imp_g.transform(raw_w)))[0]
            subset_idx = targeted_subset_indices(f_cols, targeted_subset)

        def level_A(i, k, v, cv):
            if cv in cm:                              # LOTO branch: benchmark observed
                if loto_kind == "sim_idw" and tt is not None:
                    ds, ps = [], []
                    for j in fold_sources:
                        if j == i:
                            continue
                        p = perf.get((j, k, v)); d = tt.get((i, j))
                        if p is not None and np.isfinite(p) and d is not None and np.isfinite(d):
                            ds.append(d); ps.append(p)
                    if ds:
                        ps_arr = np.asarray(ps, float)
                        w = idw_weights(np.asarray(ds, float), tt_is_sim, "idw")
                        return float((w * ps_arr).sum() / w.sum())
                if loto_kind == "targeted_idw" and (i, k) in feat_by_sk:
                    # kNN-IDW in standardized (i→k) feature space.
                    # subset_idx restricts which feature dimensions enter the
                    # norm — lets us ask "which kind of asymmetric overlap
                    # info actually carries the source-clustering signal?"
                    f_ik = feat_by_sk[(i, k)]
                    if subset_idx:
                        f_ik = f_ik[subset_idx]
                    ds, ps = [], []
                    for j in fold_sources:
                        if j == i:
                            continue
                        f_jk = feat_by_sk.get((j, k))
                        if f_jk is None:
                            continue
                        if subset_idx:
                            f_jk = f_jk[subset_idx]
                        p = perf.get((j, k, v))
                        if p is None or not np.isfinite(p):
                            continue
                        d = float(np.linalg.norm(f_ik - f_jk))
                        if not np.isfinite(d):
                            continue
                        ds.append(d); ps.append(p)
                    if ds:
                        ps_arr = np.asarray(ps, float)
                        w = idw_weights(np.asarray(ds, float), False, "idw")
                        return float((w * ps_arr).sum() / w.sum())
                if loto_kind == "eb_shrunk":
                    # Empirical-Bayes shrinkage: λ·cell_mean + (1-λ)·grand_mean.
                    # Attenuates LOO anti-correlation; ρ_L goes from -1.0 toward 0.
                    lam = eb_lambda_by_cv.get(cv, 1.0)
                    return lam * cm[cv] + (1.0 - lam) * gmean
                return cm[cv]                         # default LOTO: in-fold cell mean
            ds, ps = [], []                           # LOBO: borrow over benchmarks
            for e in fold_benchmarks:
                if e == k:
                    continue
                p = perf.get((i, e, v)); d = ee.get((k, e))
                if p is not None and np.isfinite(p) and d is not None and np.isfinite(d):
                    ds.append(d); ps.append(p)
            if not ds:
                return gmean
            ps = np.asarray(ps, float)
            if level_kind == "uniform":
                return float(ps.mean())
            w = idw_weights(np.asarray(ds, float), ee_is_sim, "idw")
            return float((w * ps).sum() / w.sum())

        # Vectorize prediction on held rows for speed
        Xte = np.asarray([
            [getattr(r, c) - xmean[r.cv][c] for c in f_cols]
            for r in held.itertuples()
        ], float)
        ctx_te = np.asarray([r.cv for r in held.itertuples()])
        g_test = _predict_ridge(model, Xte) if len(Xte) else np.array([])
        g_test_rank = (_predict_ranknet(w_rank, Xte, model)
                       if (do_ranknet and len(Xte)) else np.zeros(len(Xte)))
        if zridge is not None and len(Xte):
            reg_z, std_y_map, std_y_fb = zridge
            g_test_zridge = _predict_zridge(reg_z, Xte, ctx_te,
                                            std_y_map, std_y_fb, model)
        else:
            g_test_zridge = g_test.copy()
        g_test_gbm = (_predict_gbm(gbm, Xte, model)
                      if (do_gbm and len(Xte)) else np.zeros(len(Xte)))

        for r, g_val, gz_val, gr_val, gb_val in zip(
                held.itertuples(), g_test, g_test_zridge, g_test_rank, g_test_gbm):
            L = level_A(r.train_dataset, r.benchmark, r.variant, r.cv)
            rows.append((r.train_dataset, r.cv, r.benchmark, r.variant,
                         float(r.auc_normalized), float(g_val), float(gz_val),
                         float(gr_val), float(gb_val), L))

    return pd.DataFrame(rows, columns=["train_dataset", "context_id", "benchmark",
                                       "variant", "actual", "g", "g_zridge",
                                       "g_rank", "g_gbm", "L"])


# ---------------------------------------------------------------------------
# JOINT CV (both endpoints unseen — C3 / S4)
# ---------------------------------------------------------------------------
def cv_joint(table: pd.DataFrame, f_cols: list[str],
             do_ranknet: bool = True, do_zridge: bool = True,
             do_gbm: bool = True) -> pd.DataFrame:
    rows = []
    for i_test in sorted(table["train_dataset"].unique()):
        for k_test in sorted(table["benchmark"].unique()):
            infold = table[(table["train_dataset"] != i_test) &
                           (table["benchmark"] != k_test)]
            held = table[(table["train_dataset"] == i_test) &
                         (table["benchmark"] == k_test)]
            if infold.empty or held.empty:
                continue

            cm = infold.groupby("cv")["auc_normalized"].mean().to_dict()
            grand = float(infold["auc_normalized"].mean())
            xmean = {cv: infold.loc[infold.cv == cv, f_cols].mean()
                     for cv in infold.cv.unique()}
            grand_x = infold[f_cols].mean()
            for cv in held.cv.unique():
                xmean[cv] = grand_x

            Xtr_list, ytr_list, ctx_tr_list = [], [], []
            for r in infold.itertuples():
                xm = xmean[r.cv]
                Xtr_list.append([getattr(r, c) - xm[c] for c in f_cols])
                ytr_list.append(float(r.auc_normalized) - cm[r.cv])
                ctx_tr_list.append(r.cv)
            Xtr = np.asarray(Xtr_list, float)
            ytr = np.asarray(ytr_list, float)
            ctx_tr = np.asarray(ctx_tr_list)

            model = _fit_ridge(Xtr, ytr)
            zridge = None
            if do_zridge:
                zridge = _fit_zridge(Xtr, ytr, ctx_tr, model)
            w_rank = None
            if do_ranknet:
                w_rank = _fit_ranknet(Xtr, ytr, ctx_tr, model)
            gbm = None
            if do_gbm:
                gbm = _fit_gbm(Xtr, ytr, model)

            gamma = (infold.groupby("variant")["auc_normalized"].mean()
                     - grand).to_dict()

            Xte = np.asarray([
                [getattr(r, c) - xmean[r.cv][c] for c in f_cols]
                for r in held.itertuples()
            ], float)
            ctx_te = np.asarray([r.cv for r in held.itertuples()])
            g_test = _predict_ridge(model, Xte) if len(Xte) else np.array([])
            g_test_rank = (_predict_ranknet(w_rank, Xte, model)
                           if (do_ranknet and len(Xte)) else np.zeros(len(Xte)))
            if zridge is not None and len(Xte):
                reg_z, std_y_map, std_y_fb = zridge
                g_test_zridge = _predict_zridge(reg_z, Xte, ctx_te,
                                                std_y_map, std_y_fb, model)
            else:
                g_test_zridge = g_test.copy()
            g_test_gbm = (_predict_gbm(gbm, Xte, model)
                          if (do_gbm and len(Xte)) else np.zeros(len(Xte)))

            for r, g_val, gz_val, gr_val, gb_val in zip(
                    held.itertuples(), g_test, g_test_zridge, g_test_rank, g_test_gbm):
                L = grand + gamma.get(r.variant, 0.0)
                rows.append((r.train_dataset, r.cv, r.benchmark, r.variant,
                             float(r.auc_normalized), float(g_val),
                             float(gz_val), float(gr_val), float(gb_val), L))

    return pd.DataFrame(rows, columns=["train_dataset", "context_id", "benchmark",
                                       "variant", "actual", "g", "g_zridge",
                                       "g_rank", "g_gbm", "L"])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def within_context_spearman(df: pd.DataFrame, pred_col: str) -> float:
    rs = []
    for _, g in df.groupby("context_id"):
        if g["train_dataset"].nunique() < 3:
            continue
        if g[pred_col].std() < 1e-12:
            continue
        rho = spearmanr(g["actual"], g[pred_col]).statistic
        if np.isfinite(rho):
            rs.append(rho)
    return float(np.nanmean(rs)) if rs else float("nan")


def context_centered_spearman(df: pd.DataFrame, pred_col: str) -> float:
    c = df.copy()
    c["ar"] = c["actual"] - c.groupby("context_id")["actual"].transform("mean")
    c["pr"] = c[pred_col] - c.groupby("context_id")[pred_col].transform("mean")
    if c["pr"].std() < 1e-12:
        return float("nan")
    rho = spearmanr(c["pr"], c["ar"]).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def context_mae(df: pd.DataFrame, pred_col: str) -> float:
    return float(df.groupby("context_id").apply(
        lambda g: np.mean(np.abs(g["actual"] - g[pred_col])),
        include_groups=False).mean())


def global_abs_r(df: pd.DataFrame, pred_col: str) -> float:
    a = df["actual"].values; p = df[pred_col].values
    m = np.isfinite(a) & np.isfinite(p)
    if m.sum() < 3 or np.std(p[m]) < 1e-9:
        return float("nan")
    return float(pearsonr(p[m], a[m])[0])


def summarize(df: pd.DataFrame) -> dict:
    """Headline metrics for one (split, family) row. Reports per-head ctx ρ
    for ridge, z-ridge, and ranknet, plus calibration abs_r."""
    abs_pred = df["L"] + df["g"]
    abs_pred_z = df["L"] + df["g_zridge"]
    df_abs = df.assign(abs_pred=abs_pred, abs_pred_z=abs_pred_z)
    return dict(
        ctx_rho_g=within_context_spearman(df, "g"),
        cent_rho_g=context_centered_spearman(df, "g"),
        ctx_rho_L=within_context_spearman(df, "L"),
        ctx_rho_Lg=within_context_spearman(df_abs, "abs_pred"),
        MAE_Lg=context_mae(df_abs, "abs_pred"),
        abs_r_Lg=global_abs_r(df_abs, "abs_pred"),
        ctx_rho_g_zridge=within_context_spearman(df, "g_zridge"),
        cent_rho_g_zridge=context_centered_spearman(df, "g_zridge"),
        abs_r_Lg_zridge=global_abs_r(df_abs, "abs_pred_z"),
        MAE_Lg_zridge=context_mae(df_abs.assign(abs_pred=abs_pred_z), "abs_pred"),
        ctx_rho_g_rank=within_context_spearman(df, "g_rank"),
        cent_rho_g_rank=context_centered_spearman(df, "g_rank"),
        ctx_rho_g_gbm=within_context_spearman(df, "g_gbm"),
        cent_rho_g_gbm=context_centered_spearman(df, "g_gbm"),
        abs_r_Lg_gbm=global_abs_r(
            df.assign(abs_pred_gbm=df["L"] + df["g_gbm"]), "abs_pred_gbm"),
        n_rows=len(df),
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
def shuffle_within_context(table: pd.DataFrame, rng) -> pd.DataFrame:
    t = table.copy()
    t["auc_normalized"] = (t.groupby("cv")["auc_normalized"]
                            .transform(lambda s: rng.permutation(s.values)))
    return t


def run_one(table: pd.DataFrame, family: str, split: str,
            ee: dict, ee_is_sim: bool, label: str = "",
            level_kind: str = "idw",
            tt: dict | None = None, tt_is_sim: bool = False,
            loto_kind: str = "cell_mean",
            targeted_subset: str = "all",
            feature_subset: str = "all",
            do_ranknet: bool = True, do_zridge: bool = True,
            do_gbm: bool = True) -> tuple[pd.DataFrame, dict]:
    f_cols = feature_cols(table, family, feature_subset=feature_subset)
    if not f_cols:
        return pd.DataFrame(), {}
    if split == "JOINT":
        df = cv_joint(table, f_cols, do_ranknet=do_ranknet,
                      do_zridge=do_zridge, do_gbm=do_gbm)
    else:
        hold = "train_dataset" if split == "LOTO" else "benchmark"
        df = cv_oneaxis(table, f_cols, hold, ee, ee_is_sim, level_kind=level_kind,
                        tt=tt, tt_is_sim=tt_is_sim, loto_kind=loto_kind,
                        targeted_subset=targeted_subset,
                        do_ranknet=do_ranknet, do_zridge=do_zridge, do_gbm=do_gbm)
    if df.empty:
        return df, {}
    metrics = summarize(df)
    metrics.update(split=split, family=family, label=label or "main",
                   n_features=len(f_cols))
    return df, metrics


def run_one_target(root: Path, table_in: pd.DataFrame, dist_df: pd.DataFrame,
                   args, target: str, ee_flow, ee_flow_is_sim,
                   ee_dino, ee_dino_is_sim,
                   tt_flow, tt_flow_is_sim, tt_dino, tt_dino_is_sim,
                   tt_density=None, ee_density=None, tt_density_is_sim=False) -> None:
    """Run the full sweep for a single target column. Writes per-row CSVs
    under `predictions/<target>/` and a `summary_points_<target>.csv` next to
    them."""
    out_dir = root / args.out
    pred_dir = out_dir / "predictions" / target
    pred_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    table = table_in.dropna(subset=[target]).copy()
    # Internally everything reads 'auc_normalized' as the target column.
    table["auc_normalized"] = table[target]
    table_sh = shuffle_within_context(table, rng)

    print(f"\n========== target = {target}  (rows={len(table)}) ==========")
    summary_rows = []

    def _prior_for(fam):
        """Pick (ee, ee_is_sim, lobo_kind, tt, tt_is_sim, loto_kind) per
        (family, mode). Honors --flat-flow-prior (legacy) by forcing flow-IDW
        for both LOBO and LOTO sim_idw."""
        if args.flat_flow_prior:
            return (ee_flow, ee_flow_is_sim, "idw",
                    tt_flow, tt_flow_is_sim, "cell_mean")
        space, lobo_kind, loto_kind = level_config_for(fam, args.l_mode)
        if space == "dino":
            return (ee_dino, ee_dino_is_sim, lobo_kind,
                    tt_dino, tt_dino_is_sim, loto_kind)
        if space == "density":
            return (ee_density or {}, tt_density_is_sim, lobo_kind,
                    tt_density or {}, tt_density_is_sim, loto_kind)
        # flow or none — flow lookups always passed (only used if kind=="sim_idw"/"idw")
        return (ee_flow, ee_flow_is_sim, lobo_kind,
                tt_flow, tt_flow_is_sim, loto_kind)

    # 1. Main sweep ----------------------------------------------------------
    for split in args.splits:
        print(f"---- {split} ----")
        for fam in args.families:
            ee, ee_is_sim, lvl, tt, tt_is_sim, loto_lvl = _prior_for(fam)
            df, m = run_one(table, fam, split, ee, ee_is_sim, label="main",
                            level_kind=lvl, tt=tt, tt_is_sim=tt_is_sim,
                            loto_kind=loto_lvl,
                            targeted_subset=args.targeted_subset,
                            feature_subset=args.feature_subset,
                            do_ranknet=args.use_ranknet,
                            do_zridge=not args.no_zridge,
                            do_gbm=not args.no_gbm)
            if df.empty:
                continue
            df.to_csv(pred_dir / f"rows_{split}_{fam}.csv", index=False)
            summary_rows.append(m)
            print(f"  {fam:<11}  ridge={m['ctx_rho_g']:+.3f}  "
                  f"zridge={m['ctx_rho_g_zridge']:+.3f}  "
                  f"rank={m['ctx_rho_g_rank']:+.3f}  "
                  f"gbm={m['ctx_rho_g_gbm']:+.3f}")

    # 2. Shuffle control -----------------------------------------------------
    print(f"\n---- SHUFFLE CONTROL ----")
    for split in args.splits:
        for fam in args.families:
            ee, ee_is_sim, lvl, tt, tt_is_sim, loto_lvl = _prior_for(fam)
            df, m = run_one(table_sh, fam, split, ee, ee_is_sim,
                            label="shuffle", level_kind=lvl,
                            tt=tt, tt_is_sim=tt_is_sim, loto_kind=loto_lvl,
                            targeted_subset=args.targeted_subset,
                            feature_subset=args.feature_subset,
                            do_ranknet=args.use_ranknet,
                            do_zridge=not args.no_zridge,
                            do_gbm=not args.no_gbm)
            if df.empty:
                continue
            df.to_csv(pred_dir / f"rows_{split}_{fam}_shuffle.csv", index=False)
            summary_rows.append(m)
            print(f"  {split:<6} {fam:<11}  ridge ρ_g={m['ctx_rho_g']:+.3f}  "
                  f"rank ρ_g={m['ctx_rho_g_rank']:+.3f}")

    # 3. Uniform-level ablation (LOBO only) ----------------------------------
    print(f"\n---- UNIFORM-LEVEL ABLATION (LOBO) ----")
    for split in ["LOBO"]:
        if split not in args.splits:
            continue
        for fam in args.families:
            df, m = run_one(table, fam, split, ee_flow, ee_flow_is_sim,
                            label="uniform_level", level_kind="uniform",
                            do_ranknet=args.use_ranknet,
                            do_zridge=not args.no_zridge,
                            do_gbm=not args.no_gbm)
            if df.empty:
                continue
            df.to_csv(pred_dir / f"rows_{split}_{fam}_uniformL.csv", index=False)
            summary_rows.append(m)
            print(f"  {split:<6} {fam:<11}  ridge ρ_L={m['ctx_rho_L']:+.3f}  "
                  f"abs_r_Lg={m['abs_r_Lg']:.3f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df["target"] = target
    summary_df.to_csv(out_dir / f"summary_points_{target}.csv", index=False)
    print(f"point-estimate summary -> {out_dir}/summary_points_{target}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="scripts/transfer_analysis_v3/transfer_table.csv")
    ap.add_argument("--dist", default="analysis_v3/pairwise_self_distances.csv")
    ap.add_argument("--out", default="scripts/transfer_analysis_v4/results")
    ap.add_argument("--target", default=None,
                    help="single target (legacy); prefer --targets")
    ap.add_argument("--targets", nargs="+",
                    default=["auc_normalized", "peak_pck"],
                    help="target columns to run; predictions saved per-target")
    ap.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    ap.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    ap.add_argument("--prior-space", default="flow")
    ap.add_argument("--prior-metric", default="mean_nn_sym")
    ap.add_argument("--family-matched-prior", action="store_true",
                    help="(legacy) ignored — per-family prior is now the default")
    ap.add_argument("--flat-flow-prior", action="store_true",
                    help="legacy mode: use flow-IDW for ALL families (motion, "
                         "appearance, random, ...). Default = each family uses "
                         "its own space (DINO for appearance; uniform L for random "
                         "and density) so no family piggy-backs on motion.")
    ap.add_argument("--l-mode", default="mixed", choices=L_MODES,
                    help="L mechanism per CV regime:\n"
                         "  mixed (default): LOTO=cell_mean, LOBO=per-family IDW\n"
                         "  symmetric_informed: LOTO=per-family sim_train_IDW, "
                         "LOBO=per-family sim_eval_IDW\n"
                         "  symmetric_uninformed: LOTO=cell_mean, LOBO=uniform "
                         "for all families\n"
                         "  targeted_informed: LOTO=kNN-IDW in standardized "
                         "(i→k) feature space, LOBO=per-family IDW")
    ap.add_argument("--feature-subset", default="all", choices=SUBSET_NAMES,
                    help="Restrict which features f_cols includes (affects g, "
                         "z-ridge, ranknet, gbm, and the targeted_idw vector). "
                         "Use this to test whether g is overfitting with the "
                         "full 13-dim feature set.")
    ap.add_argument("--targeted-subset", default="all", choices=SUBSET_NAMES,
                    help="Further restrict which feature dimensions enter the "
                         "targeted_idw distance norm. Only matters when "
                         "--l-mode=targeted_informed. Independent of "
                         "--feature-subset; if --feature-subset already "
                         "restricts f_cols, this further filters within.")
    ap.add_argument("--no-ranknet", action="store_true",
                    help="(legacy) ignored; ranknet is now opt-in via --use-ranknet")
    ap.add_argument("--use-ranknet", action="store_true",
                    help="opt-in pairwise RankNet head (off by default — it "
                         "doesn't move the headline and the rank scatter is "
                         "unreadable at N=10/context)")
    ap.add_argument("--no-zridge", action="store_true",
                    help="skip within-context z-score ridge head")
    ap.add_argument("--no-gbm", action="store_true",
                    help="skip GBM (nonlinear ceiling-check) head")
    ap.add_argument("--pure-only", action="store_true",
                    help="Restrict training datasets to the original 11 pure "
                         "sources (drop mixed variants like spair_synthetic_70_30). "
                         "Matches the analysis scope used in CLAIMS.md.")
    ap.add_argument("--drop-family", nargs="+", default=None,
                    metavar="FAMILY",
                    help="Leave-one-generator-family-out robustness (Tier 2): drop "
                         "ALL sources whose generator family is in this list before "
                         "fitting. Families: " + ", ".join(sorted(set(FAMILY_MAP.values()))) +
                         ". Tests that no single family drives the result.")
    ap.add_argument("--drop-source", nargs="+", default=None, metavar="SOURCE",
                    help="Drop specific train_dataset(s) before fitting "
                         "(drop-one-source robustness, Tier 1).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.target is not None:
        args.targets = [args.target]

    root = Path(".").resolve()
    out_dir = root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(root / args.table).copy()
    if args.pure_only:
        before = len(table)
        table = table[table["train_dataset"].isin(PURE_TRAIN_DATASETS)].copy()
        print(f"--pure-only: filtered {before} -> {len(table)} rows "
              f"({table['train_dataset'].nunique()} pure sources)")
    if args.drop_family:
        fams = set(args.drop_family)
        unknown = fams - set(FAMILY_MAP.values())
        if unknown:
            raise SystemExit(f"--drop-family: unknown families {sorted(unknown)}; "
                             f"valid: {sorted(set(FAMILY_MAP.values()))}")
        dropped = sorted(s for s in table["train_dataset"].unique()
                         if family_of(s) in fams)
        before = table["train_dataset"].nunique()
        table = table[~table["train_dataset"].isin(dropped)].copy()
        print(f"--drop-family {sorted(fams)}: dropped {len(dropped)} sources "
              f"{dropped} ({before} -> {table['train_dataset'].nunique()} sources)")
    if args.drop_source:
        srcs = set(args.drop_source)
        before = table["train_dataset"].nunique()
        table = table[~table["train_dataset"].isin(srcs)].copy()
        print(f"--drop-source {sorted(srcs)}: "
              f"{before} -> {table['train_dataset'].nunique()} sources")
    table["variant"] = table.apply(variant_key, axis=1)
    table["cv"] = table["benchmark"] + "|" + table["variant"]
    dist_df = pd.read_csv(root / args.dist)
    table = add_selfdist_features(table, dist_df)
    table = add_random_features(table, seed=42)

    ee_flow = build_lookup(dist_df, "flow", "eval_eval", args.prior_metric)
    ee_flow_is_sim = args.prior_metric in SIMILARITY_METRICS
    ee_dino = build_lookup(dist_df, "dino", "eval_eval", args.prior_metric)
    ee_dino_is_sim = args.prior_metric in SIMILARITY_METRICS

    tt_flow = build_lookup(dist_df, "flow", "train_train", args.prior_metric)
    tt_flow_is_sim = args.prior_metric in SIMILARITY_METRICS
    tt_dino = build_lookup(dist_df, "dino", "train_train", args.prior_metric)
    tt_dino_is_sim = args.prior_metric in SIMILARITY_METRICS

    # Density-based pairwise distances (for --l-mode density_idw). Built once
    # from the per-dataset density features in the transfer table itself.
    tt_density, ee_density = _build_density_lookups(table)
    tt_density_is_sim = False  # raw Euclidean distance, lower=closer

    print(f"v4 sweep  targets={args.targets}  families={args.families}  "
          f"splits={args.splits}")
    print(f"L mode: {args.l_mode}  feature-subset: {args.feature_subset}  "
          f"targeted-subset: {args.targeted_subset}")
    if args.flat_flow_prior:
        print(f"prior: FLAT flow-IDW for all families (legacy mode)")
    else:
        print(f"prior: per-family — motion/both/motion_density=flow, "
              f"appearance=DINO, random/density=uniform-L")
    print(f"heads: ridge always; ranknet={'yes' if args.use_ranknet else 'no'};  "
          f"zridge={'no' if args.no_zridge else 'yes'};  "
          f"gbm={'no' if args.no_gbm else 'yes'}")

    summary_all = []
    for target in args.targets:
        if target not in table.columns:
            print(f"  SKIP {target}: column not in table")
            continue
        run_one_target(root, table, dist_df, args, target,
                       ee_flow, ee_flow_is_sim, ee_dino, ee_dino_is_sim,
                       tt_flow, tt_flow_is_sim, tt_dino, tt_dino_is_sim,
                       tt_density=tt_density, ee_density=ee_density,
                       tt_density_is_sim=tt_density_is_sim)
        summary_all.append(pd.read_csv(out_dir / f"summary_points_{target}.csv"))

    if summary_all:
        combined = pd.concat(summary_all, ignore_index=True)
        combined.to_csv(out_dir / "summary_points.csv", index=False)
        print(f"\ncombined point-estimate summary -> {out_dir}/summary_points.csv")


if __name__ == "__main__":
    main()
