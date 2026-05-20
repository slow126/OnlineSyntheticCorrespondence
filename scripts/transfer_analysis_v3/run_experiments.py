#!/usr/bin/env python3
"""
Transfer ranking estimator experiments — strict split-first pipeline.

Leakage rule: ALL preprocessing (imputation, scaling), pairwise/listwise
construction, and global priors are computed from training rows only.

Splits:
  loto           leave one train_dataset out (~11 folds)
  loto_grouped   same but collapse variant datasets to their base name
  lobo           leave one benchmark out (~9 folds)
  loco           leave one context_id out (~50 folds)
  loco_cell      leave one (train_dataset, benchmark) cell out (99 folds; pooled eval)
  joint_cell     leave one train_dataset AND one benchmark axis out; test their cell
  lomo           leave one model_family out (~8 folds)

Models:
  ridge          RidgeCV on within-context rank scores
  bradley_terry  Logistic regression on (x_i - x_j) pairs
  plackett_luce  Custom torch PL loss, linear model
  kernel_ridge   KernelRidge(rbf), grid search
  random         Random scores (baseline)
  global_prior   Mean train-dataset rank from training fold (baseline)

Feature groups: motion, motion_km, appearance, density,
                symmetric_mmd, symmetric_fid, symmetric_w2,
                motion_appearance, all

Usage:
    python scripts/transfer_analysis_v3/run_experiments.py \
        [--table scripts/transfer_analysis_v3/transfer_table.csv] \
        [--splits loto lobo] [--models ridge] [--feature-groups motion all] \
        [--output-dir scripts/transfer_analysis_v3/results] \
        [--debug]
"""

import argparse
import hashlib
import itertools
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kendalltau
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import SplineTransformer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

def resolve_feature_groups(table_cols: list[str]) -> dict[str, list[str]]:
    """Build feature groups by matching against actual column names in the table.

    Design principle: each group tests ONE concept.  Composites are explicitly
    motivated combinations, not exploratory mixes.

    Directed (asymmetric, train↔eval) groups — flow space:
      flow_nn   — mean NN distance (continuous, no threshold)
      flow_eps  — ε-ball coverage at 1/4/16 px (binary fraction, threshold-based)
      flow_km   — k-means density-weighted ε-coverage (same thresholds)
      flow_kl   — kNN KL divergence (information-theoretic, density-sensitive)
                  [populated after re-running coverage with kl.enabled: true]

    Directed — DINO appearance space:
      dino_nn   — mean NN distance in DINO space
      dino_cov  — null-calibrated cosine coverage
      dino_kl   — kNN KL divergence in DINO space [pending]

    Symmetric (undirected) baselines — no directional signal:
      sym_flow  — flow MMD + flow FID + flow SW2
      sym_dino  — DINO MMD + DINO FID + DINO SW2
      sym_mmd   — flow MMD + DINO MMD
      sym_fid   — flow FID + DINO FID
      sym_w2    — flow SW2 + DINO SW2

    Dataset size:
      density   — log(N_train) + log(N_eval)

    Motivated composites (full directed description of one modality):
      motion          — flow_nn + flow_eps  (all directed flow features)
      motion_km       — flow_nn + flow_km   (k-means variant)
      appearance      — dino_nn + dino_cov  (all directed DINO features)
      motion_appearance — motion + appearance
      all             — all directed + all symmetric + density
    """
    def match(patterns: list[str]) -> list[str]:
        cols = []
        for pat in patterns:
            matched = [c for c in table_cols if c == pat or c.startswith(pat)]
            cols.extend(matched)
        return list(dict.fromkeys(cols))  # deduplicate, preserve order

    EPS = ["eps1px", "eps4px", "eps16px"]

    # --- Directed: flow space ---
    flow_nn  = match(["flow_mean_nn_eval_to_train_k1", "flow_mean_nn_train_to_eval_k1"])
    # Exclude _weighted variants (those belong to flow_km); startswith would pull them in.
    flow_eps = [c for c in match([f"flow_eval_covered_by_train_{t}" for t in EPS]
                               + [f"flow_train_covered_by_eval_{t}" for t in EPS])
                if not c.endswith("_weighted")]
    flow_km  = match([f"flow_km_eval_covered_by_train_{t}_weighted" for t in EPS]
                   + [f"flow_km_train_covered_by_eval_{t}_weighted" for t in EPS])
    # kNN KL: populated after re-running coverage with kl.enabled: true + k_values: [5, 20]
    flow_kl  = match(["flow_kl_eval_to_train_k5",  "flow_kl_train_to_eval_k5",
                       "flow_kl_eval_to_train_k20", "flow_kl_train_to_eval_k20"])

    # --- Directed: DINO appearance space ---
    dino_nn  = match(["dino_mean_nn_eval_to_train_k1", "dino_mean_nn_train_to_eval_k1"])
    # Null-calibrated coverage preferred; falls back to qnorm if not yet computed.
    _null    = match(["dino_eval_covered_by_train_null90", "dino_train_covered_by_eval_null90",
                      "dino_eval_covered_by_train_null95", "dino_train_covered_by_eval_null95"])
    _qnorm   = match(["dino_eval_covered_by_train_qnorm_k1", "dino_train_covered_by_eval_qnorm_k1"])
    dino_cov = _null if _null else _qnorm
    # kNN KL in DINO space [pending]
    dino_kl  = match(["dino_kl_eval_to_train_k5",  "dino_kl_train_to_eval_k5",
                       "dino_kl_eval_to_train_k20", "dino_kl_train_to_eval_k20"])

    # --- Symmetric baselines (individual) ---
    flow_mmd_only  = match(["flow_mmd"])
    dino_mmd_only  = match(["dino_mmd"])
    flow_fid_only  = match(["flow_fid"])
    dino_fid_only  = match(["dino_fid"])
    flow_w2_only   = match(["flow_sliced_w2"])
    dino_w2_only   = match(["dino_sliced_w2"])
    # --- Symmetric baselines (combined) ---
    sym_flow = match(["flow_mmd", "flow_fid", "flow_sliced_w2"])
    sym_dino = match(["dino_mmd", "dino_fid", "dino_sliced_w2"])
    sym_mmd  = match(["flow_mmd", "dino_mmd"])
    sym_fid  = match(["flow_fid", "dino_fid"])
    sym_w2   = match(["flow_sliced_w2", "dino_sliced_w2"])

    # --- Density ---
    density       = match(["log_train_n_vectors", "log_eval_n_vectors"])
    density_train = match(["log_train_n_vectors"])
    density_eval  = match(["log_eval_n_vectors"])
    density_idw   = density  # same features as density; IDW uses log_n_dist
    random_idw    = match(["random_train", "random_eval"])  # random features + random IDW distances

    # --- Vector profile controls ---
    sample_count       = match(["log_train_n_samples", "log_eval_n_samples"])
    sample_count_train = match(["log_train_n_samples"])
    sample_count_eval  = match(["log_eval_n_samples"])
    vector_density_simple_train = match(["log_train_valid_vectors_per_sample_capped"])
    vector_density_simple_eval = match(["log_eval_valid_vectors_per_sample_capped"])
    vector_density_simple = vector_density_simple_train + vector_density_simple_eval
    train_profile_simple = sample_count_train + vector_density_simple_train
    eval_profile_simple = sample_count_eval + vector_density_simple_eval
    profile_simple = sample_count + vector_density_simple
    vector_density_train = match([
        "log_train_valid_vectors_per_sample",
        "log_train_sampled_vectors_per_sample",
        "log_train_retained_vectors_per_sample",
        "log_train_valid_vectors_mean",
        "log_train_valid_vectors_median",
        "log_train_valid_vectors_p10",
        "log_train_valid_vectors_p90",
        "log_train_valid_vectors_p95",
        "log_train_sampled_vectors_mean",
        "log_train_sampled_vectors_median",
        "train_zero_image_frac",
    ])
    vector_density_eval = match([
        "log_eval_valid_vectors_per_sample",
        "log_eval_sampled_vectors_per_sample",
        "log_eval_retained_vectors_per_sample",
        "log_eval_valid_vectors_mean",
        "log_eval_valid_vectors_median",
        "log_eval_valid_vectors_p10",
        "log_eval_valid_vectors_p90",
        "log_eval_valid_vectors_p95",
        "log_eval_sampled_vectors_mean",
        "log_eval_sampled_vectors_median",
        "eval_zero_image_frac",
    ])
    vector_density = vector_density_train + vector_density_eval
    train_profile = sample_count_train + vector_density_train
    eval_profile = sample_count_eval + vector_density_eval
    profile_density = sample_count + vector_density

    # --- Composites ---
    all_sym  = sym_flow + sym_dino  # deduplicated below
    groups = {
        # Single-concept directed groups
        "flow_nn":           flow_nn,
        "flow_eps":          flow_eps,
        "flow_km":           flow_km,
        "flow_kl":           flow_kl,
        "dino_nn":           dino_nn,
        "dino_cov":          dino_cov,
        "dino_kl":           dino_kl,
        # Symmetric baselines — individual (space × metric)
        "flow_mmd_only":     flow_mmd_only,
        "dino_mmd_only":     dino_mmd_only,
        "flow_fid_only":     flow_fid_only,
        "dino_fid_only":     dino_fid_only,
        "flow_w2_only":      flow_w2_only,
        "dino_w2_only":      dino_w2_only,
        # Symmetric baselines — combined (per modality, per metric)
        "sym_flow":          sym_flow,
        "sym_dino":          sym_dino,
        "sym_mmd":           sym_mmd,
        "sym_fid":           sym_fid,
        "sym_w2":            sym_w2,
        # Dataset size — combined and split by axis
        "density":           density,
        "density_train":     density_train,
        "density_eval":      density_eval,
        # Pure density IDW: log N features + |log N| IDW distances (no flow anywhere)
        "density_idw":       density_idw,
        # Random IDW: log N features + shuffled random distances (mechanism-only baseline)
        "random_idw":        random_idw,
        # Sample/profile controls. These separate "how many examples" from
        # "how many flow vectors per example" so sparse datasets like SPair are
        # not collapsed into total-vector count alone.
        "sample_count":          sample_count,
        "sample_count_train":    sample_count_train,
        "sample_count_eval":     sample_count_eval,
        "vector_density_simple": vector_density_simple,
        "train_profile_simple":  train_profile_simple,
        "eval_profile_simple":   eval_profile_simple,
        "profile_simple":        profile_simple,
        "vector_density":        vector_density,
        "vector_density_train":  vector_density_train,
        "vector_density_eval":   vector_density_eval,
        "train_profile":         train_profile,
        "eval_profile":          eval_profile,
        "profile_density":       profile_density,
        # Motivated composites
        "motion":            flow_nn + flow_eps,
        "motion_km":         flow_nn + flow_km,
        "appearance":        dino_nn + dino_cov,
        "motion_appearance": flow_nn + flow_eps + dino_nn + dino_cov,
        "flow_mmd_profile":  flow_mmd_only + train_profile_simple,
        "flow_fid_profile":  flow_fid_only + train_profile_simple,
        "flow_w2_profile":   flow_w2_only + train_profile_simple,
        "flow_kl_profile":   flow_kl + train_profile_simple,
        "motion_km_profile": flow_nn + flow_km + train_profile_simple,
        "all":               flow_nn + flow_eps + dino_nn + dino_cov + density + list(dict.fromkeys(all_sym)),
    }
    # Deduplicate columns and filter to those that actually exist in the table.
    existing = set(table_cols)
    groups = {k: list(dict.fromkeys(c for c in v if c in existing))
              for k, v in groups.items()}
    return groups


# ---------------------------------------------------------------------------
# Split generators — yield (fold_id, train_df, test_df)
# ---------------------------------------------------------------------------

def _base_name(dataset: str) -> str:
    """Collapse variant suffixes: flyingthings_synthetic_70_30 → flyingthings."""
    return re.sub(r"(_synthetic_\d+_\d+|_30_70|_50_50|_70_30|_2d_warp.*|_imagenet.*)", "", dataset)


def iter_loto_folds(df: pd.DataFrame, grouped: bool = False) -> Iterator[tuple]:
    col = "train_dataset"
    if grouped:
        df = df.copy()
        df["_group"] = df[col].map(_base_name)
        groups = sorted(df["_group"].unique())
        for g in groups:
            test_mask = df["_group"] == g
            yield g, df[~test_mask].drop(columns="_group"), df[test_mask].drop(columns="_group")
    else:
        for entity in sorted(df[col].unique()):
            test_mask = df[col] == entity
            yield entity, df[~test_mask], df[test_mask]


def iter_lobo_folds(df: pd.DataFrame) -> Iterator[tuple]:
    for entity in sorted(df["benchmark"].unique()):
        test_mask = df["benchmark"] == entity
        yield entity, df[~test_mask], df[test_mask]


def iter_loco_folds(df: pd.DataFrame) -> Iterator[tuple]:
    for entity in sorted(df["context_id"].unique()):
        test_mask = df["context_id"] == entity
        yield entity, df[~test_mask], df[test_mask]


def iter_loco_cell_folds(df: pd.DataFrame) -> Iterator[tuple]:
    """Leave one (train_dataset, benchmark) cell out.

    99 folds (11 trains × 9 benchmarks). Each fold holds out all model variants
    for one train-benchmark pair. When all fold predictions are pooled, evaluation
    groups by context_id (benchmark × model variant) and ranks the 11 training
    datasets — each training dataset's prediction for that context was made without
    seeing that (train, benchmark) cell, making this a strict matrix-completion eval.
    """
    cells = sorted({(row.train_dataset, row.benchmark) for _, row in df.iterrows()})
    for train, bench in cells:
        test_mask = (df["train_dataset"] == train) & (df["benchmark"] == bench)
        yield f"{train}|{bench}", df[~test_mask], df[test_mask]


def iter_joint_cell_folds(df: pd.DataFrame) -> Iterator[tuple]:
    """Leave one train axis and one benchmark axis out; test only their cell.

    For a cell (train=A, benchmark=B), the fit fold excludes every row with
    train_dataset=A and every row with benchmark=B.  The test fold is only
    A×B.  Pooled across folds, this evaluates predictions for each context where
    both the train dataset and benchmark were absent from that fold's fit pool.
    """
    cells = sorted({(row.train_dataset, row.benchmark) for _, row in df.iterrows()})
    for train, bench in cells:
        test_mask = (df["train_dataset"] == train) & (df["benchmark"] == bench)
        train_mask = (df["train_dataset"] != train) & (df["benchmark"] != bench)
        yield f"{train}|{bench}", df[train_mask], df[test_mask]


def iter_lomo_folds(df: pd.DataFrame) -> Iterator[tuple]:
    for entity in sorted(df["model_family"].unique()):
        test_mask = df["model_family"] == entity
        yield entity, df[~test_mask], df[test_mask]


SPLIT_FNS = {
    "loto":         lambda df: iter_loto_folds(df, grouped=False),
    "loto_grouped": lambda df: iter_loto_folds(df, grouped=True),
    "lobo":         iter_lobo_folds,
    "loco":         iter_loco_folds,
    "loco_cell":    iter_loco_cell_folds,
    "joint_cell":   iter_joint_cell_folds,
    "lomo":         iter_lomo_folds,
}


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def fit_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple:
    X = train_df[feature_cols].values.astype(np.float64)
    imputer = SimpleImputer(strategy="median").fit(X)
    scaler  = StandardScaler().fit(imputer.transform(X))
    return imputer, scaler


def apply_preprocessor(df: pd.DataFrame, feature_cols: list[str], preprocessor: tuple) -> np.ndarray:
    imputer, scaler = preprocessor
    X = df[feature_cols].values.astype(np.float64)
    return scaler.transform(imputer.transform(X)).astype(np.float32)


# ---------------------------------------------------------------------------
# Within-context rank target
# ---------------------------------------------------------------------------

def make_rank_scores(df: pd.DataFrame, target_col: str = "auc_normalized",
                     context_col: str = "context_id") -> pd.Series:
    """Per context: best candidate → 1.0, worst → 0.0. Skip contexts with n < 2."""
    scores: dict[int, float] = {}
    for _, grp in df.groupby(context_col):
        n = len(grp)
        if n < 2:
            continue
        ranks = grp[target_col].rank(ascending=True, method="average")
        for idx, r in ranks.items():
            scores[idx] = (r - 1.0) / (n - 1.0)
    return pd.Series(scores, name="rank_score")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class RidgeRankModel:
    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        y = make_rank_scores(train_df).reindex(train_df.index)
        valid = y.notna()
        self._model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(
            X_train[valid.values], y[valid].values)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test)


class RidgeAbsModel:
    """Plain RidgeCV baseline that predicts absolute target values."""

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        y = train_df["auc_normalized"].copy().reindex(train_df.index)
        valid = y.notna()
        self._model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]).fit(
            X_train[valid.values], y[valid].values)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test)


class BradleyTerryModel:
    """Pairwise logistic regression on (x_i - x_j) differences.
    Pairs are built inside fit() so they are always from training rows only."""

    def __init__(self, margin: float = 0.05):
        self.margin = margin

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        idx_to_row = {orig_idx: pos for pos, orig_idx in enumerate(train_df.index)}
        diffs, labels = [], []
        for _, grp in train_df.groupby("context_id"):
            positions = [(idx_to_row[i], i) for i in grp.index]
            y_vals = {i: train_df.loc[i, "auc_normalized"] for _, i in positions}
            for (pos_i, idx_i), (pos_j, idx_j) in itertools.combinations(positions, 2):
                yi, yj = y_vals[idx_i], y_vals[idx_j]
                if abs(yi - yj) <= self.margin:
                    continue
                d = X_train[pos_i] - X_train[pos_j]
                label = 1 if yi > yj else 0
                # Add both directions so sklearn always sees two classes.
                # f(d,1) + f(-d,0) = 1 by sigmoid symmetry — no information added,
                # just satisfies sklearn's requirement for ≥2 classes.
                diffs.append(d);  labels.append(label)
                diffs.append(-d); labels.append(1 - label)
        if len(diffs) < 10:
            self._coef = np.zeros(X_train.shape[1], dtype=np.float32)
            return
        D = np.array(diffs, dtype=np.float32)
        y = np.array(labels)
        lr = LogisticRegression(penalty="l2", C=1.0, fit_intercept=False,
                                solver="lbfgs", max_iter=500)
        lr.fit(D, y)
        self._coef = lr.coef_[0].astype(np.float32)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        return (X_test @ self._coef).astype(np.float64)


class PlackettLuceModel:
    """Linear model trained with Plackett-Luce listwise loss using torch."""

    def __init__(self, lambda_: float = 1.0, epochs: int = 200, lr: float = 1e-2):
        self.lambda_ = lambda_
        self.epochs = epochs
        self.lr = lr

    @staticmethod
    def _pl_loss(scores: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        """Negative log Plackett-Luce likelihood for a single ranked list."""
        s = scores[order]  # best-first
        log_tail = torch.logcumsumexp(s.flip(0), dim=0).flip(0)
        return (log_tail - s).sum()

    def _fit_one(self, X: np.ndarray, df: pd.DataFrame, lambda_: float) -> np.ndarray:
        d = X.shape[1]
        beta = nn.Parameter(torch.zeros(d))
        opt = torch.optim.Adam([beta], lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)

        # Precompute per-context positions and orderings once — O(N) total.
        # Moving this inside the epoch loop caused O(N²) per epoch (list.index scan).
        idx_map = {idx: pos for pos, idx in enumerate(df.index)}
        groups = []
        for _, grp in df.groupby("context_id"):
            positions = torch.tensor([idx_map[i] for i in grp.index], dtype=torch.long)
            order = torch.tensor(np.argsort(-grp["auc_normalized"].values), dtype=torch.long)
            groups.append((positions, order))

        for _ in range(self.epochs):
            opt.zero_grad()
            scores = X_t @ beta
            total_loss = torch.tensor(0.0, requires_grad=False)
            for positions, order in groups:
                ctx_scores = scores[positions]
                loss = PlackettLuceModel._pl_loss(ctx_scores, order)
                total_loss = total_loss + loss
            reg = 0.5 * lambda_ * torch.sum(beta ** 2)
            (total_loss + reg).backward()
            opt.step()
        return beta.detach().numpy().astype(np.float32)

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        # Simple lambda selection via 3-fold inner CV on rank Spearman
        lambda_grid = [0.01, 0.1, 1.0, 10.0]
        best_lambda, best_score = lambda_grid[0], -np.inf
        df_reset = train_df.reset_index(drop=False)
        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        context_ids = train_df["context_id"].values

        for lam in lambda_grid:
            fold_scores = []
            for tr_idx, va_idx in kf.split(X_train):
                tr_df = df_reset.iloc[tr_idx].set_index("index")
                beta = self._fit_one(X_train[tr_idx], tr_df, lam)
                va_scores = X_train[va_idx] @ beta
                va_df = train_df.iloc[va_idx].copy()
                va_df["pred_score"] = va_scores
                rho = _context_spearman(va_df)
                fold_scores.append(rho)
            mean_rho = np.nanmean(fold_scores)
            if mean_rho > best_score:
                best_score, best_lambda = mean_rho, lam

        self._beta = self._fit_one(X_train, train_df, best_lambda)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        return (X_test @ self._beta).astype(np.float64)


class KernelRidgeModel:
    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        y = make_rank_scores(train_df).reindex(train_df.index)
        valid = y.notna()
        X_tr = X_train[valid.values]
        y_tr = y[valid].values

        best_score, best_params = -np.inf, (1.0, 0.1)
        alphas = [0.1, 1.0, 10.0]
        gammas = [None, 0.1]  # None → 'scale'

        kf = KFold(n_splits=3, shuffle=True, random_state=0)
        for alpha in alphas:
            for gamma in gammas:
                g = 1.0 / X_tr.shape[1] if gamma is None else gamma
                scores = []
                for tr, va in kf.split(X_tr):
                    kr = KernelRidge(kernel="rbf", alpha=alpha, gamma=g)
                    kr.fit(X_tr[tr], y_tr[tr])
                    scores.append(float(np.corrcoef(kr.predict(X_tr[va]), y_tr[va])[0, 1]))
                mean_score = np.nanmean(scores)
                if mean_score > best_score:
                    best_score, best_params = mean_score, (alpha, g)

        alpha, gamma = best_params
        self._model = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)
        self._model.fit(X_tr, y_tr)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        return self._model.predict(X_test)


class TwoWayMixedRidgeModel:
    """Regularized two-way effects model for absolute transfer prediction.

    This is a small mixed-effects approximation using Ridge on:
      selected numeric features
      + train_dataset one-hot effects
      + benchmark one-hot effects
      + correspondence-model variant one-hot effects

    The L2 penalty shrinks entity effects toward zero, so held-out train
    datasets/benchmarks fall back to the global/features/known-axis estimate
    instead of an IDW nearest-neighbor rule.
    """

    _CAT_COLS = ["train_dataset", "benchmark", "model_family", "pretrained", "freeze"]

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        target_col: str | None = None,
    ) -> None:
        self._feature_cols = list(feature_cols or [])
        self._target_col = target_col
        self._cat_values: dict[str, list[str]] = {}
        self._imputer = None
        self._scaler = None
        self._model = None

    def _cat_matrix(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        parts = []
        for col in self._CAT_COLS:
            values = df[col].astype(str).fillna("__NA__")
            if fit:
                self._cat_values[col] = sorted(values.unique().tolist())
            cats = self._cat_values.get(col, [])
            idx = {v: i for i, v in enumerate(cats)}
            mat = np.zeros((len(df), len(cats)), dtype=np.float64)
            for r, val in enumerate(values):
                j = idx.get(val)
                if j is not None:
                    mat[r, j] = 1.0
            parts.append(mat)
        return np.hstack(parts) if parts else np.zeros((len(df), 0), dtype=np.float64)

    def _design(self, df: pd.DataFrame, fit: bool) -> np.ndarray:
        numeric = df[self._feature_cols].values.astype(np.float64)
        cats = self._cat_matrix(df, fit=fit)
        return np.hstack([numeric, cats]) if cats.shape[1] else numeric

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        del X_train
        target = self._target_col if self._target_col in train_df.columns else "auc_normalized"
        y = train_df[target].copy().reindex(train_df.index)
        valid = y.notna().values
        X = self._design(train_df, fit=True)
        self._imputer = SimpleImputer(strategy="median").fit(X[valid])
        X_imp = self._imputer.transform(X[valid])
        self._scaler = StandardScaler().fit(X_imp)
        X_proc = self._scaler.transform(X_imp)
        self._model = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]).fit(
            X_proc, y[valid].values
        )

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self._design(test_df, fit=False)
        X_proc = self._scaler.transform(self._imputer.transform(X))
        return self._model.predict(X_proc)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TwoWayMixedRidgeModel requires predict_score_df(test_df)")


class TensorProductKRRModel:
    """
    KernelRidge with tensor product kernel K_train ⊗ K_eval.

    K_train(i,j) = RBF(d_sym(train_i, train_j)) where d_sym is a pairwise
    distance between training datasets (from pairwise_self_distances.csv).
    K_eval(a,b)  = RBF(d_sym(eval_a,  eval_b)).
    K_full[(i,a),(j,b)] = K_train(i,j) * K_eval(a,b)

    This gives the model an explicit prior over how training datasets and
    benchmarks relate to each other — improving generalisation in LOTO and LOBO.
    """

    def __init__(
        self,
        self_dist_df: pd.DataFrame,
        kernel_col: str,
        kernel_type: str = "rbf_distance",
        alpha_vals: list | None = None,
        target_col: str | None = None,
    ) -> None:
        """
        Args:
            self_dist_df: DataFrame with (space, dataset_a, dataset_b, kernel_col)
                          Must include BOTH train-train and eval-eval rows.
            kernel_col:   Column name in self_dist_df to use as distance/similarity.
            kernel_type:  "rbf_distance" — RBF(gamma * d) where higher col = farther.
                          "rbf_similarity" — RBF(gamma * (1-s)) where col in [0,1].
            alpha_vals:   Ridge regularization values to try in inner CV.
        """
        self._df = self_dist_df.copy()
        self._kernel_col = kernel_col
        self._kernel_type = kernel_type
        self._alpha_vals = alpha_vals or [0.01, 0.1, 1.0, 10.0, 100.0]
        self._target_col = target_col
        # Build lookup dict: (dataset_a, dataset_b) → value (symmetric)
        self._lookup: dict[tuple[str, str], float] = {}
        for _, row in self_dist_df.iterrows():
            a, b, v = row["dataset_a"], row["dataset_b"], row.get(kernel_col, np.nan)
            if np.isfinite(v):
                self._lookup[(a, b)] = float(v)
                self._lookup[(b, a)] = float(v)
        # Self-distance for each dataset (0 for distance, 1 for similarity)
        self._self_val = 0.0 if kernel_type == "rbf_distance" else 1.0

    def _get_dist(self, a: str, b: str) -> float:
        if a == b:
            return self._self_val
        return self._lookup.get((a, b), np.nan)

    def _build_kernel(
        self,
        names_row: list[str],
        names_col: list[str],
        gamma: float,
    ) -> np.ndarray:
        """Build (n_row, n_col) RBF kernel matrix."""
        n, m = len(names_row), len(names_col)
        D = np.full((n, m), np.nan, dtype=np.float64)
        for i, a in enumerate(names_row):
            for j, b in enumerate(names_col):
                D[i, j] = self._get_dist(a, b)
        # Impute missing with column/row mean, then overall mean
        col_means = np.nanmean(D, axis=0, keepdims=True)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        mask = ~np.isfinite(D)
        D = np.where(mask, col_means, D)
        D = np.where(~np.isfinite(D), 0.0, D)
        if self._kernel_type == "rbf_similarity":
            D = 1.0 - D  # convert similarity to distance
        return np.exp(-gamma * D)

    def _auto_gamma(self, names: list[str]) -> float:
        """1/median of pairwise distances, clamped away from 0."""
        vals = []
        for a, b in itertools.combinations(names, 2):
            v = self._get_dist(a, b)
            if np.isfinite(v):
                vals.append(v if self._kernel_type == "rbf_distance" else 1.0 - v)
        if not vals:
            return 1.0
        med = float(np.median(vals))
        return 1.0 / max(med, 1e-8)

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        train_ds   = train_df["train_dataset"].tolist()
        benchmarks = train_df["benchmark"].tolist()

        unique_trains = list(dict.fromkeys(train_ds))
        unique_evals  = list(dict.fromkeys(benchmarks))

        gamma_tr = self._auto_gamma(unique_trains)
        gamma_ev = self._auto_gamma(unique_evals)
        self._gamma_tr = gamma_tr
        self._gamma_ev = gamma_ev
        self._unique_trains = unique_trains
        self._unique_evals  = unique_evals

        K_train = self._build_kernel(unique_trains, unique_trains, gamma_tr)
        K_eval  = self._build_kernel(unique_evals,  unique_evals,  gamma_ev)
        self._K_train = K_train
        self._K_eval  = K_eval

        t_idx = {t: i for i, t in enumerate(unique_trains)}
        e_idx = {e: i for i, e in enumerate(unique_evals)}
        ti = np.array([t_idx[t] for t in train_ds])
        ei = np.array([e_idx[e] for e in benchmarks])

        K_full = K_train[np.ix_(ti, ti)] * K_eval[np.ix_(ei, ei)]

        if self._target_col is not None and self._target_col in train_df.columns:
            y = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            y = make_rank_scores(train_df).reindex(train_df.index)
        valid = y.notna()
        K_tr = K_full[np.ix_(valid.values, valid.values)]
        y_tr = y[valid].values

        self._valid_mask = valid.values
        self._ti_valid = ti[valid.values]
        self._ei_valid = ei[valid.values]

        if len(y_tr) < 4:
            self._model = KernelRidge(kernel="precomputed", alpha=1.0)
            self._model.fit(K_tr, y_tr)
            return

        best_alpha, best_score = self._alpha_vals[0], -np.inf
        kf = KFold(n_splits=min(3, len(y_tr)), shuffle=True, random_state=0)
        for alpha in self._alpha_vals:
            fold_scores = []
            for tr, va in kf.split(K_tr):
                kr = KernelRidge(kernel="precomputed", alpha=alpha)
                kr.fit(K_tr[np.ix_(tr, tr)], y_tr[tr])
                p = kr.predict(K_tr[np.ix_(va, tr)])
                if len(p) > 1:
                    r = float(np.corrcoef(p, y_tr[va])[0, 1])
                    fold_scores.append(r if np.isfinite(r) else 0.0)
            if fold_scores and np.nanmean(fold_scores) > best_score:
                best_score = np.nanmean(fold_scores)
                best_alpha = alpha

        self._model = KernelRidge(kernel="precomputed", alpha=best_alpha)
        self._model.fit(K_tr, y_tr)

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        t_idx = {t: i for i, t in enumerate(self._unique_trains)}
        e_idx = {e: i for i, e in enumerate(self._unique_evals)}

        test_ds    = test_df["train_dataset"].tolist()
        test_bm    = test_df["benchmark"].tolist()

        # For unseen datasets/benchmarks, fall back to mean kernel row.
        # This handles LOTO (novel train dataset) and LOBO (novel benchmark).
        def _row_idx(name, idx_dict, K):
            if name in idx_dict:
                return K[idx_dict[name]]
            return K.mean(axis=0)

        m = len(test_ds)
        n = len(self._ti_valid)
        K_test = np.zeros((m, n), dtype=np.float64)
        for i, (td, te) in enumerate(zip(test_ds, test_bm)):
            k_tr_row = _row_idx(td, t_idx, self._K_train)
            k_ev_row = _row_idx(te, e_idx, self._K_eval)
            K_test[i] = k_tr_row[self._ti_valid] * k_ev_row[self._ei_valid]

        return self._model.predict(K_test)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("TensorProductKRRModel requires predict_score_df(test_df)")


# Kernel configurations for TensorProductKRR.
# Each entry: model_name → (space, kernel_col, kernel_type)
KRR_TP_CONFIGS: dict[str, tuple[str, str, str]] = {
    "krr_tp_flow_nn":   ("flow", "mean_nn_sym",    "rbf_distance"),
    "krr_tp_flow_eps":  ("flow", "sym_eps1px",     "rbf_similarity"),
    "krr_tp_flow_eps16":("flow", "sym_eps16px",    "rbf_similarity"),
    "krr_tp_dino_nn":   ("dino", "mean_nn_sym",    "rbf_distance"),
    "krr_tp_dino_eps":  ("dino", "sym_eps1px",     "rbf_similarity"),
}


class RidgePairwiseDistModel:
    """Ridge regression augmented with IDW performance predictions from fold neighbors.

    For a held-out (train_i, benchmark_j) test point, two IDW predictions are formed:

    Train side — for each in-fold train n ≠ i, weight perf(n, j) by 1/dist(i, n):
        idw_pred_train = Σ w(i,n)·perf(n,j) / Σ w(i,n)

    Eval side — for each in-fold eval e ≠ j, weight perf(i, e) by 1/dist(j, e):
        idw_pred_eval  = Σ w(j,e)·perf(i,e) / Σ w(j,e)

    When a performance value is unavailable (e.g., i is held out), falls back to
    the in-fold mean across the available axis.

    Three features per side per selected pairwise feature space:
        idw_pred — IDW-weighted performance prediction
        min_dist — distance to nearest in-fold neighbor (in-distribution proxy)
        perf_std — std of neighbor performances (prediction confidence/spread)

    Default augmentation: 3 features × 2 sides × 2 spaces = 12 columns.
    Use --pairwise-spaces flow for flow-only ablations.

    Variants are instantiated with different metric_col values for clean isolation:
        ridge_pairwise_nn      — mean_nn_sym (symmetric NN distance)
        ridge_pairwise_eps1px  — a_covered_by_b_eps1px (ε-coverage similarity)
        ridge_pairwise_eps16px — a_covered_by_b_eps16px
        ridge_pairwise_kl      — kl_a_to_b_k5 (KL divergence distance)
    """

    # Maps a metric column to its reversed-direction counterpart.
    # Used when pairwise_self_distances.csv stores the pair as (b, a) instead of (a, b).
    _REVERSE_COL: dict[str, str] = {
        "mean_nn_sym":            "mean_nn_sym",           # symmetric
        "mean_nn_a_to_b":         "mean_nn_b_to_a",
        "mean_nn_b_to_a":         "mean_nn_a_to_b",
        "a_covered_by_b_eps1px":  "b_covered_by_a_eps1px",
        "b_covered_by_a_eps1px":  "a_covered_by_b_eps1px",
        "a_covered_by_b_eps16px": "b_covered_by_a_eps16px",
        "b_covered_by_a_eps16px": "a_covered_by_b_eps16px",
        "kl_a_to_b_k5":           "kl_b_to_a_k5",
        "kl_b_to_a_k5":           "kl_a_to_b_k5",
        "sym_eps1px":             "sym_eps1px",
        "sym_eps16px":            "sym_eps16px",
        "log_n_dist":             "log_n_dist",            # symmetric: |log Na - log Nb|
        "sample_count_dist":      "sample_count_dist",     # symmetric: sample-count profile distance
        "vector_density_dist":    "vector_density_dist",   # symmetric: vectors/sample profile distance
        "profile_dist":           "profile_dist",          # symmetric: sample + vectors/sample distance
        "random_dist":            "random_dist",           # symmetric: shuffled random baseline
        "flow_mmd_self":          "flow_mmd_self",         # symmetric MMD self-distance
        "flow_fid_self":          "flow_fid_self",         # symmetric FID self-distance
        "flow_sliced_w2_self":    "flow_sliced_w2_self",   # symmetric sliced-W2 self-distance
    }

    def __init__(
        self,
        self_dist_df: pd.DataFrame,
        feature_cols: list[str],
        metric_col: str,
        is_similarity: bool = False,
        target_col: str | None = None,
        spaces: list[str] | None = None,
        cross_idw: bool = False,
        cross_mode: str = "raw",
        weight_mode: str = "idw",
        use_spline: bool = False,
    ) -> None:
        self._feature_cols = list(feature_cols)
        self._metric_col = metric_col
        self._is_similarity = is_similarity
        self._target_col = target_col
        self._spaces = list(spaces or ["flow", "dino"])
        self._cross_idw = cross_idw
        self._cross_mode = cross_mode
        self._weight_mode = weight_mode
        self._use_spline = use_spline
        self._fold_trains: list[str] = []
        self._fold_evals: list[str] = []
        self._perf_lookup: dict[tuple, float] = {}
        self._train_mean: dict[str, float] = {}
        self._eval_mean: dict[str, float] = {}
        self._imputer = None
        self._scaler = None
        self._spline = None
        self._basis_scaler = None
        self._model = None

        # Build lookup {(space, pair_type): {(a, b): metric_value}} resolving
        # both directions at build time so runtime lookup is a single dict.get().
        rev_col = self._REVERSE_COL.get(metric_col, metric_col)
        self._lookup: dict[tuple[str, str], dict[tuple[str, str], float]] = {}
        for (space, ptype), grp in self_dist_df.groupby(["space", "pair_type"]):
            lk: dict[tuple[str, str], float] = {}
            for _, row in grp.iterrows():
                da, db = row["dataset_a"], row["dataset_b"]
                direct = float(row[metric_col]) if metric_col in row.index and pd.notna(row[metric_col]) else np.nan
                lk[(da, db)] = direct
                if rev_col == metric_col:
                    lk[(db, da)] = direct  # symmetric metric
                else:
                    rev = float(row[rev_col]) if rev_col in row.index and pd.notna(row[rev_col]) else np.nan
                    lk[(db, da)] = rev
            self._lookup[(space, ptype)] = lk

    def _fit_ridge_regressor(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the residual/absolute Ridge stage, optionally with spline features."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        y = np.where(np.isfinite(y), y, np.nan)
        valid_rows = np.isfinite(y)
        X = X[valid_rows]
        y = y[valid_rows]
        keep_cols = ~np.all(~np.isfinite(X), axis=0)
        if keep_cols.any():
            X = X[:, keep_cols]
        else:
            X = np.zeros((len(y), 1), dtype=np.float64)
        self._kept_cols = keep_cols

        self._imputer = SimpleImputer(strategy="median").fit(X)
        X_imp = self._imputer.transform(X)
        X_imp = np.nan_to_num(X_imp, nan=0.0, posinf=0.0, neginf=0.0)
        self._scaler = StandardScaler().fit(X_imp)
        X_proc = self._scaler.transform(X_imp)
        X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)

        if self._use_spline:
            self._spline = SplineTransformer(
                n_knots=4,
                degree=3,
                include_bias=False,
                extrapolation="constant",
            ).fit(X_proc)
            X_model = self._spline.transform(X_proc)
            X_model = np.asarray(X_model)
            self._basis_scaler = StandardScaler().fit(X_model)
            X_model = self._basis_scaler.transform(X_model)
            X_model = np.nan_to_num(X_model, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self._spline = None
            self._basis_scaler = None
            X_model = X_proc

        alphas = getattr(self, "_ridge_alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        try:
            self._model = RidgeCV(alphas=alphas).fit(
                X_model, y
            )
        except Exception:
            # Full anchor-bilinear/random-control designs can be ill-conditioned.
            # Fall back to a conservative solver instead of killing the sweep.
            self._model = Ridge(alpha=float(np.median(alphas)), solver="lsqr").fit(X_model, y)

    def _predict_ridge_regressor(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        keep_cols = getattr(self, "_kept_cols", None)
        if keep_cols is not None and keep_cols.any():
            X = X[:, keep_cols]
        elif keep_cols is not None:
            X = np.zeros((len(X), 1), dtype=np.float64)
        X_proc = self._scaler.transform(self._imputer.transform(X))
        X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)
        if self._use_spline:
            X_model = np.asarray(self._spline.transform(X_proc))
            X_model = self._basis_scaler.transform(X_model)
            X_model = np.nan_to_num(X_model, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            X_model = X_proc
        return self._model.predict(X_model)

    @staticmethod
    def _model_variant_key(row: pd.Series) -> tuple[str, str, str]:
        """Full correspondence-model variant key, excluding benchmark."""
        return (
            str(row.get("model_family", "")),
            str(row.get("pretrained", "")),
            str(row.get("freeze", "")),
        )

    def _idw_stats(self, dists: list[float], perfs: list[float]) -> np.ndarray:
        """Compute [idw_pred, min_dist, perf_std] from parallel distance/perf lists."""
        if not dists:
            return np.array([np.nan, np.nan, np.nan])
        d = np.array(dists, dtype=np.float64)
        p = np.array(perfs, dtype=np.float64)
        if self._weight_mode == "uniform":
            weights = np.ones_like(p, dtype=np.float64)
            min_dist = float(1.0 - d.max()) if self._is_similarity else float(d.min())
        elif self._weight_mode == "random":
            # Deterministic random-neighbor baseline. The seed depends on the
            # unordered candidate set, but weights are independent of distance
            # magnitude/order. This tests the borrowing mechanism without
            # meaningful geometry.
            seed_payload = "|".join(f"{v:.8g}" for v in sorted(d.tolist()))
            seed = int.from_bytes(
                hashlib.blake2b(seed_payload.encode("utf-8"), digest_size=8).digest(),
                byteorder="little",
                signed=False,
            )
            rng = np.random.default_rng(seed)
            weights = rng.random(len(p)) + 1e-8
            min_dist = float(1.0 - d.max()) if self._is_similarity else float(d.min())
        elif self._is_similarity:
            # Coverage metrics: higher = more similar. Use similarity directly as weight.
            weights = np.maximum(d, 1e-8)
            min_dist = 1.0 - float(d.max())  # convert max-similarity to min-distance
        else:
            # Distance metrics (NN, KL): lower = more similar. Shift so min = 1e-3,
            # then weight = 1/shifted_dist. Works for negative KL values.
            shifted = d - d.min() + 1e-3
            weights = 1.0 / shifted
            min_dist = float(d.min())
        total_w = weights.sum()
        idw_pred = float((weights * p).sum() / total_w) if total_w > 0 else float(p.mean())
        perf_std = float(p.std()) if len(p) > 1 else 0.0
        return np.array([idw_pred, min_dist, perf_std])

    def _self_metric_value(self) -> float:
        return 1.0 if self._is_similarity else 0.0

    def _neighbor_list(
        self,
        entity: str,
        fold_entities: list[str],
        lookup: dict[tuple[str, str], float],
        include_self: bool,
    ) -> list[tuple[str, float]]:
        out = []
        for other in fold_entities:
            if other == entity:
                if include_self:
                    out.append((other, self._self_metric_value()))
                continue
            val = lookup.get((entity, other))
            if val is not None and np.isfinite(val):
                out.append((other, val))
        return out

    def _cross_idw_stats(
        self,
        train_dists: list[float],
        eval_dists: list[float],
        perfs: list[float],
    ) -> np.ndarray:
        """Joint train-neighbor × benchmark-neighbor IDW stats."""
        if not perfs:
            return np.array([np.nan, np.nan, np.nan, np.nan])
        td = np.array(train_dists, dtype=np.float64)
        ed = np.array(eval_dists, dtype=np.float64)
        p = np.array(perfs, dtype=np.float64)
        if self._weight_mode == "uniform":
            weights = np.ones_like(p, dtype=np.float64)
            min_train_dist = float(1.0 - td.max()) if self._is_similarity else float(td.min())
            min_eval_dist = float(1.0 - ed.max()) if self._is_similarity else float(ed.min())
        elif self._weight_mode == "random":
            seed_payload = "|".join(
                f"{a:.8g},{b:.8g}" for a, b in sorted(zip(td.tolist(), ed.tolist()))
            )
            seed = int.from_bytes(
                hashlib.blake2b(seed_payload.encode("utf-8"), digest_size=8).digest(),
                byteorder="little",
                signed=False,
            )
            rng = np.random.default_rng(seed)
            weights = rng.random(len(p)) + 1e-8
            min_train_dist = float(1.0 - td.max()) if self._is_similarity else float(td.min())
            min_eval_dist = float(1.0 - ed.max()) if self._is_similarity else float(ed.min())
        elif self._is_similarity:
            train_weights = np.maximum(td, 1e-8)
            eval_weights = np.maximum(ed, 1e-8)
            min_train_dist = 1.0 - float(td.max())
            min_eval_dist = 1.0 - float(ed.max())
            weights = train_weights * eval_weights
        else:
            train_shifted = td - td.min() + 1e-3
            eval_shifted = ed - ed.min() + 1e-3
            train_weights = 1.0 / train_shifted
            eval_weights = 1.0 / eval_shifted
            min_train_dist = float(td.min())
            min_eval_dist = float(ed.min())
            weights = train_weights * eval_weights
        total_w = weights.sum()
        idw_pred = float((weights * p).sum() / total_w) if total_w > 0 else float(p.mean())
        perf_std = float(p.std()) if len(p) > 1 else 0.0
        return np.array([idw_pred, min_train_dist, min_eval_dist, perf_std])

    def _compute_idw_features(
        self,
        df: pd.DataFrame,
        fold_trains: list[str],
        fold_evals: list[str],
        perf_lookup: dict[tuple[str, str], float],
        train_mean: dict[str, float],
        eval_mean: dict[str, float],
    ) -> np.ndarray:
        """Build 12-column IDW augmentation matrix for all rows in df."""
        parts: list[np.ndarray] = []
        for space in self._spaces:
            tt_lk = self._lookup.get((space, "train_train"), {})
            ee_lk = self._lookup.get((space, "eval_eval"), {})

            # Precompute per-entity distance neighbor lists (independent of the other axis).
            unique_trains = list(dict.fromkeys(df["train_dataset"]))
            unique_evals  = list(dict.fromkeys(df["benchmark"]))

            train_nbr: dict[str, list[tuple[str, float]]] = {}
            for ti in unique_trains:
                train_nbr[ti] = self._neighbor_list(
                    ti, fold_trains, tt_lk, include_self=False)

            eval_nbr: dict[str, list[tuple[str, float]]] = {}
            for ej in unique_evals:
                eval_nbr[ej] = self._neighbor_list(
                    ej, fold_evals, ee_lk, include_self=False)

            cross_train_nbr: dict[str, list[tuple[str, float]]] = {}
            cross_eval_nbr: dict[str, list[tuple[str, float]]] = {}
            if self._cross_idw:
                for ti in unique_trains:
                    cross_train_nbr[ti] = self._neighbor_list(
                        ti, fold_trains, tt_lk, include_self=True)
                for ej in unique_evals:
                    cross_eval_nbr[ej] = self._neighbor_list(
                        ej, fold_evals, ee_lk, include_self=True)

            n_rows = len(df)
            train_feats = np.full((n_rows, 3), np.nan)
            eval_feats  = np.full((n_rows, 3), np.nan)
            cross_feats = np.full((n_rows, 4), np.nan)

            for idx, (_, row) in enumerate(df.iterrows()):
                ti, bj = row["train_dataset"], row["benchmark"]
                model_variant = self._model_variant_key(row)
                model_family = model_variant[0]

                # Train side: weight perf(n, bj) by distance from ti to n.
                # Prefer exact model variant; fall back to model family, then cross-variant mean.
                t_dists, t_perfs = [], []
                for n, dv in train_nbr.get(ti, []):
                    pf = perf_lookup.get(
                        (n, bj, *model_variant),
                        perf_lookup.get((n, bj, model_family), perf_lookup.get((n, bj))),
                    )
                    if pf is None:
                        pf = train_mean.get(n)
                    if pf is not None and np.isfinite(pf):
                        t_dists.append(dv)
                        t_perfs.append(pf)
                train_feats[idx] = self._idw_stats(t_dists, t_perfs)

                # Eval side: weight perf(ti, e) by distance from bj to e.
                e_dists, e_perfs = [], []
                for e, dv in eval_nbr.get(bj, []):
                    pf = perf_lookup.get(
                        (ti, e, *model_variant),
                        perf_lookup.get((ti, e, model_family), perf_lookup.get((ti, e))),
                    )
                    if pf is None:
                        pf = eval_mean.get(e)
                    if pf is not None and np.isfinite(pf):
                        e_dists.append(dv)
                        e_perfs.append(pf)
                eval_feats[idx] = self._idw_stats(e_dists, e_perfs)

                if self._cross_idw:
                    eval_prior = np.nan
                    if self._cross_mode == "residual":
                        prior_dists, prior_vals = [], []
                        for e, e_dist in cross_eval_nbr.get(bj, []):
                            val = eval_mean.get(e)
                            if val is not None and np.isfinite(val):
                                prior_dists.append(e_dist)
                                prior_vals.append(val)
                        eval_prior = self._idw_stats(prior_dists, prior_vals)[0]

                    ct_dists, ce_dists, c_perfs = [], [], []
                    for n, t_dist in cross_train_nbr.get(ti, []):
                        for e, e_dist in cross_eval_nbr.get(bj, []):
                            pf = perf_lookup.get(
                                (n, e, *model_variant),
                                perf_lookup.get((n, e, model_family), perf_lookup.get((n, e))),
                            )
                            if pf is not None and np.isfinite(pf):
                                if self._cross_mode == "residual" and e in eval_mean:
                                    pf = pf - eval_mean[e]
                                ct_dists.append(t_dist)
                                ce_dists.append(e_dist)
                                c_perfs.append(pf)
                    cross_row = self._cross_idw_stats(ct_dists, ce_dists, c_perfs)
                    if self._cross_mode == "residual" and np.isfinite(eval_prior):
                        cross_row[0] += eval_prior
                    cross_feats[idx] = cross_row

            parts.extend([train_feats, eval_feats])
            if self._cross_idw:
                parts.append(cross_feats)

        return np.hstack(parts) if parts else np.zeros((len(df), 0), dtype=np.float64)

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        # X_train (preprocessed by run_fold) is ignored — we re-preprocess jointly
        # with the IDW augmentation features so they share the same scaler.
        self._fold_trains = sorted(train_df["train_dataset"].unique())
        self._fold_evals  = sorted(train_df["benchmark"].unique())

        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = make_rank_scores(train_df).reindex(train_df.index)

        self._perf_lookup = {}
        _pair_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
        _family_pair_vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        _variant_pair_vals: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
        for row_idx, row in train_df.iterrows():
            val = perf_series.loc[row_idx]
            if pd.notna(val):
                td  = row["train_dataset"]
                bm  = row["benchmark"]
                model_variant = self._model_variant_key(row)
                model_family = model_variant[0]
                fval = float(val)
                _variant_pair_vals[(td, bm, *model_variant)].append(fval)
                _family_pair_vals[(td, bm, model_family)].append(fval)
                _pair_vals[(td, bm)].append(fval)

        # Lookup priority at prediction time:
        #   1. exact correspondence model variant: (family, pretrained, freeze)
        #   2. model family average
        #   3. model-agnostic average
        for key, vals in _variant_pair_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for key, vals in _family_pair_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for (td, bm), vals in _pair_vals.items():
            self._perf_lookup[(td, bm)] = float(np.mean(vals))

        self._train_mean = {
            t: float(np.mean([v for (td, bm), vs in _pair_vals.items() if td == t for v in vs]))
            for t in self._fold_trains
            if any(td == t for td, _ in _pair_vals)
        }
        self._eval_mean = {
            e: float(np.mean([v for (td, bm), vs in _pair_vals.items() if bm == e for v in vs]))
            for e in self._fold_evals
            if any(bm == e for _, bm in _pair_vals)
        }

        orig = train_df[self._feature_cols].values.astype(np.float64)
        aug  = self._compute_idw_features(
            train_df, self._fold_trains, self._fold_evals,
            self._perf_lookup, self._train_mean, self._eval_mean,
        )
        X_full = np.hstack([orig, aug]) if aug.shape[1] > 0 else orig
        valid = perf_series.notna()

        self._fit_ridge_regressor(X_full[valid.values], perf_series[valid].values)

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        orig  = test_df[self._feature_cols].values.astype(np.float64)
        aug   = self._compute_idw_features(
            test_df, self._fold_trains, self._fold_evals,
            self._perf_lookup, self._train_mean, self._eval_mean,
        )
        X_full = np.hstack([orig, aug]) if aug.shape[1] > 0 else orig
        return self._predict_ridge_regressor(X_full)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("RidgePairwiseDistModel requires predict_score_df(test_df)")


class AnchorBilinearRidgeModel(RidgePairwiseDistModel):
    """Full bilinear ridge over learned anchor-space train/eval coordinates.

    For each fold, a train dataset is represented by its distances/similarities
    to all in-fold training datasets; an eval benchmark is represented by its
    distances/similarities to all in-fold benchmarks.  The model then fits Ridge
    on:

      selected train-eval features
      + train anchor coordinates
      + eval anchor coordinates
      + full train_anchor ⊗ eval_anchor cross-products

    This keeps the interpolatable-space idea but lets Ridge learn which anchor
    directions and train/eval interactions matter, instead of imposing inverse
    distance weights.
    """

    def _axis_matrix(
        self,
        names: list[str],
        anchors: list[str],
        lookup: dict[tuple[str, str], float],
    ) -> np.ndarray:
        X = np.full((len(names), len(anchors)), np.nan, dtype=np.float64)
        self_val = self._self_metric_value()
        for i, name in enumerate(names):
            for j, anchor in enumerate(anchors):
                if name == anchor:
                    X[i, j] = self_val
                else:
                    val = lookup.get((name, anchor))
                    if val is not None and np.isfinite(val):
                        X[i, j] = float(val)
        return X

    @staticmethod
    def _row_outer(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.einsum("bi,bj->bij", a, b).reshape(a.shape[0], a.shape[1] * b.shape[1])

    def _anchor_axes(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        space = self._spaces[0]
        tt_lk = self._lookup.get((space, "train_train"), {})
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        train_axis = self._axis_matrix(
            df["train_dataset"].astype(str).tolist(), self._fold_trains, tt_lk
        )
        eval_axis = self._axis_matrix(
            df["benchmark"].astype(str).tolist(), self._fold_evals, ee_lk
        )
        orig = df[self._feature_cols].values.astype(np.float64)
        return orig, train_axis, eval_axis

    def _build_bilinear_features(self, df: pd.DataFrame) -> np.ndarray:
        orig, train_axis, eval_axis = self._anchor_axes(df)
        cross = self._row_outer(train_axis, eval_axis)
        return np.hstack([orig, train_axis, eval_axis, cross])

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        del X_train
        self._fold_trains = sorted(train_df["train_dataset"].astype(str).unique())
        self._fold_evals = sorted(train_df["benchmark"].astype(str).unique())

        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = train_df["auc_normalized"].copy().reindex(train_df.index)
        valid = perf_series.notna().values
        X_full = self._build_bilinear_features(train_df)
        self._fit_ridge_regressor(X_full[valid], perf_series[valid].values)

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        X_full = self._build_bilinear_features(test_df)
        return self._predict_ridge_regressor(X_full)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("AnchorBilinearRidgeModel requires predict_score_df(test_df)")


class AnchorAdditiveRidgeModel(AnchorBilinearRidgeModel):
    """Anchor ridge without train×eval cross-products."""

    def _build_bilinear_features(self, df: pd.DataFrame) -> np.ndarray:
        orig, train_axis, eval_axis = self._anchor_axes(df)
        return np.hstack([orig, train_axis, eval_axis])


class AnchorLowRankBilinearRidgeModel(AnchorBilinearRidgeModel):
    """Anchor ridge with a low-dimensional train×eval interaction subspace.

    The train/eval anchor coordinate matrices are each compressed by PCA inside
    the training fold, then only their low-dimensional outer product is added.
    With rank=3 this gives 9 interaction features instead of ~100-110.
    """

    def __init__(self, *args, rank: int = 3, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._train_axis_imputer = None
        self._eval_axis_imputer = None
        self._train_axis_scaler = None
        self._eval_axis_scaler = None
        self._train_axis_pca = None
        self._eval_axis_pca = None

    def _fit_reduce_axis(self, X: np.ndarray, axis: str) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        imputer = SimpleImputer(strategy="median").fit(X)
        X_imp = np.nan_to_num(imputer.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(X_imp)
        X_proc = np.nan_to_num(scaler.transform(X_imp), nan=0.0, posinf=0.0, neginf=0.0)
        n_comp = max(1, min(self._rank, X_proc.shape[1], X_proc.shape[0]))
        pca = PCA(n_components=n_comp, random_state=0).fit(X_proc)
        if axis == "train":
            self._train_axis_imputer = imputer
            self._train_axis_scaler = scaler
            self._train_axis_pca = pca
        else:
            self._eval_axis_imputer = imputer
            self._eval_axis_scaler = scaler
            self._eval_axis_pca = pca
        return pca.transform(X_proc)

    def _transform_reduce_axis(self, X: np.ndarray, axis: str) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        if axis == "train":
            imputer = self._train_axis_imputer
            scaler = self._train_axis_scaler
            pca = self._train_axis_pca
        else:
            imputer = self._eval_axis_imputer
            scaler = self._eval_axis_scaler
            pca = self._eval_axis_pca
        X_proc = scaler.transform(imputer.transform(X))
        X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)
        return pca.transform(X_proc)

    def _build_lowrank_features(self, df: pd.DataFrame, fit_axes: bool) -> np.ndarray:
        orig, train_axis, eval_axis = self._anchor_axes(df)
        if fit_axes:
            train_low = self._fit_reduce_axis(train_axis, "train")
            eval_low = self._fit_reduce_axis(eval_axis, "eval")
        else:
            train_low = self._transform_reduce_axis(train_axis, "train")
            eval_low = self._transform_reduce_axis(eval_axis, "eval")
        cross = self._row_outer(train_low, eval_low)
        return np.hstack([orig, train_axis, eval_axis, cross])

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        del X_train
        self._fold_trains = sorted(train_df["train_dataset"].astype(str).unique())
        self._fold_evals = sorted(train_df["benchmark"].astype(str).unique())
        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = train_df["auc_normalized"].copy().reindex(train_df.index)
        valid = perf_series.notna().values
        X_full = self._build_lowrank_features(train_df, fit_axes=True)
        self._fit_ridge_regressor(X_full[valid], perf_series[valid].values)

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        X_full = self._build_lowrank_features(test_df, fit_axes=False)
        return self._predict_ridge_regressor(X_full)


class AnchorBilinearShrunkRidgeModel(AnchorBilinearRidgeModel):
    """Full anchor bilinear ridge with a high-regularization alpha grid."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ridge_alphas = [10.0, 100.0, 1000.0, 10000.0]


class KernelMixedEffectsModel(RidgePairwiseDistModel):
    """Additive kernel mixed-effects model with optional train×eval interaction.

    The covariance is an ANOVA-style sum of components:

        K = wt * K_train + we * K_eval + wv * K_variant
            + wi * (K_train * K_eval)

    where K_train and K_eval are RBF kernels built from the selected
    train-train/eval-eval pairwise distance analog, and K_variant is a small
    correspondence-model variant kernel.  Component weights and ridge alpha are
    chosen by a small inner CV grid.  The interaction weight is capped and can
    be disabled entirely, keeping this much less flexible than pure TP-KRR.
    """

    def __init__(self, *args, include_interaction: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._include_interaction = include_interaction
        self._alpha_vals = [0.1, 1.0, 10.0, 100.0]

    @staticmethod
    def _variant_key(row: pd.Series) -> tuple[str, str, str]:
        return (
            str(row.get("model_family", "")),
            str(row.get("pretrained", "")),
            str(row.get("freeze", "")),
        )

    @staticmethod
    def _variant_kernel_rows(row_df: pd.DataFrame, col_df: pd.DataFrame) -> np.ndarray:
        row_keys = [KernelMixedEffectsModel._variant_key(r) for _, r in row_df.iterrows()]
        col_keys = [KernelMixedEffectsModel._variant_key(r) for _, r in col_df.iterrows()]
        K = np.zeros((len(row_keys), len(col_keys)), dtype=np.float64)
        for i, rv in enumerate(row_keys):
            for j, cv in enumerate(col_keys):
                if rv == cv:
                    K[i, j] = 1.0
                elif rv[0] == cv[0]:
                    K[i, j] = 0.5
        return K

    def _axis_distance_values(self, entities: list[str], lookup: dict[tuple[str, str], float]) -> list[float]:
        vals = []
        for a, b in itertools.combinations(entities, 2):
            if a == b:
                continue
            val = lookup.get((a, b))
            if val is None or not np.isfinite(val):
                continue
            vals.append(1.0 - float(val) if self._is_similarity else float(val))
        return vals

    def _axis_gamma(self, entities: list[str], lookup: dict[tuple[str, str], float]) -> float:
        vals = self._axis_distance_values(entities, lookup)
        if not vals:
            return 1.0
        med = float(np.median(vals))
        return 1.0 / max(abs(med), 1e-8)

    def _axis_fallback_distance(self, entities: list[str], lookup: dict[tuple[str, str], float]) -> float:
        vals = self._axis_distance_values(entities, lookup)
        if not vals:
            return 1.0
        return float(np.median(vals))

    def _axis_kernel(
        self,
        row_names: list[str],
        col_names: list[str],
        lookup: dict[tuple[str, str], float],
        gamma: float,
        fallback_distance: float,
    ) -> np.ndarray:
        K = np.zeros((len(row_names), len(col_names)), dtype=np.float64)
        for i, a in enumerate(row_names):
            for j, b in enumerate(col_names):
                if a == b:
                    dist = 0.0
                else:
                    val = lookup.get((a, b))
                    if val is None or not np.isfinite(val):
                        dist = fallback_distance
                    else:
                        dist = 1.0 - float(val) if self._is_similarity else float(val)
                K[i, j] = np.exp(-gamma * max(float(dist), 0.0))
        return K

    def _component_kernels(self, row_df: pd.DataFrame, col_df: pd.DataFrame) -> tuple[np.ndarray, ...]:
        space = self._spaces[0]
        tt_lk = self._lookup.get((space, "train_train"), {})
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        Kt = self._axis_kernel(
            row_df["train_dataset"].astype(str).tolist(),
            col_df["train_dataset"].astype(str).tolist(),
            tt_lk,
            self._gamma_train,
            self._fallback_train_dist,
        )
        Ke = self._axis_kernel(
            row_df["benchmark"].astype(str).tolist(),
            col_df["benchmark"].astype(str).tolist(),
            ee_lk,
            self._gamma_eval,
            self._fallback_eval_dist,
        )
        Kv = self._variant_kernel_rows(row_df, col_df)
        Ki = Kt * Ke
        return Kt, Ke, Kv, Ki

    @staticmethod
    def _combine_components(components: tuple[np.ndarray, ...], weights: tuple[float, float, float, float]) -> np.ndarray:
        wt, we, wv, wi = weights
        Kt, Ke, Kv, Ki = components
        return wt * Kt + we * Ke + wv * Kv + wi * Ki

    def _candidate_weights(self) -> list[tuple[float, float, float, float]]:
        main = [
            (1.0, 1.0, 1.0, 0.0),
            (2.0, 1.0, 1.0, 0.0),
            (1.0, 2.0, 1.0, 0.0),
            (1.0, 1.0, 2.0, 0.0),
            (1.0, 1.0, 0.5, 0.0),
        ]
        if not self._include_interaction:
            return main

        out = list(main)
        for base in main[:3]:
            for wi in (0.05, 0.10, 0.25):
                out.append((base[0], base[1], base[2], wi))
        return out

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        del X_train
        self._fold_trains = sorted(train_df["train_dataset"].astype(str).unique())
        self._fold_evals = sorted(train_df["benchmark"].astype(str).unique())
        space = self._spaces[0]
        tt_lk = self._lookup.get((space, "train_train"), {})
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        self._gamma_train = self._axis_gamma(self._fold_trains, tt_lk)
        self._gamma_eval = self._axis_gamma(self._fold_evals, ee_lk)
        self._fallback_train_dist = self._axis_fallback_distance(self._fold_trains, tt_lk)
        self._fallback_eval_dist = self._axis_fallback_distance(self._fold_evals, ee_lk)

        if self._target_col is not None and self._target_col in train_df.columns:
            y = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            y = train_df["auc_normalized"].copy().reindex(train_df.index)
        valid = y.notna().values
        self._train_df_valid = train_df.loc[valid].copy()
        y_valid = y[valid].values.astype(np.float64)

        components = self._component_kernels(self._train_df_valid, self._train_df_valid)
        best = (float("inf"), self._candidate_weights()[0], self._alpha_vals[0])
        n = len(y_valid)
        if n >= 6:
            kf = KFold(n_splits=min(3, n), shuffle=True, random_state=0)
            for weights in self._candidate_weights():
                K = self._combine_components(components, weights)
                for alpha in self._alpha_vals:
                    losses = []
                    for tr, va in kf.split(K):
                        y_mean = float(np.mean(y_valid[tr]))
                        kr = KernelRidge(kernel="precomputed", alpha=alpha)
                        kr.fit(K[np.ix_(tr, tr)], y_valid[tr] - y_mean)
                        pred = kr.predict(K[np.ix_(va, tr)]) + y_mean
                        losses.append(float(np.mean((pred - y_valid[va]) ** 2)))
                    score = float(np.mean(losses))
                    if score < best[0]:
                        best = (score, weights, alpha)

        self._weights = best[1]
        self._alpha = best[2]
        K_final = self._combine_components(components, self._weights)
        self._y_mean = float(np.mean(y_valid))
        self._model = KernelRidge(kernel="precomputed", alpha=self._alpha)
        self._model.fit(K_final, y_valid - self._y_mean)

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        components = self._component_kernels(test_df, self._train_df_valid)
        K_test = self._combine_components(components, self._weights)
        return self._model.predict(K_test) + self._y_mean

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("KernelMixedEffectsModel requires predict_score_df(test_df)")


class KernelMixedAdditiveModel(KernelMixedEffectsModel):
    """Kernel mixed-effects model with train/eval/variant main effects only."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["include_interaction"] = False
        super().__init__(*args, **kwargs)


class KernelMixedInteractionModel(KernelMixedEffectsModel):
    """Kernel mixed-effects model with a capped train×eval interaction term."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["include_interaction"] = True
        super().__init__(*args, **kwargs)


class RidgePairwiseCrossDistModel(RidgePairwiseDistModel):
    """Ridge+IDW with raw joint train-neighbor × benchmark-neighbor calibration."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["cross_idw"] = True
        super().__init__(*args, **kwargs)


class RidgePairwiseCrossResidualModel(RidgePairwiseDistModel):
    """2-axis IDW that transfers residual train effects across benchmark neighbors."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["cross_idw"] = True
        kwargs["cross_mode"] = "residual"
        super().__init__(*args, **kwargs)


class RidgePairwiseCrossResidualSplineModel(RidgePairwiseCrossResidualModel):
    """2-axis residual IDW with a spline-expanded Ridge stage."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_spline"] = True
        super().__init__(*args, **kwargs)


class RidgePairwiseUniformModel(RidgePairwiseDistModel):
    """Ridge + neighbor-average controls: same neighbors, uniform weights."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["weight_mode"] = "uniform"
        super().__init__(*args, **kwargs)


class RidgePairwiseRandomModel(RidgePairwiseDistModel):
    """Ridge + random-weight neighbor controls: same panel, no distance geometry."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["weight_mode"] = "random"
        super().__init__(*args, **kwargs)


class IDWPriorResidualModel(RidgePairwiseDistModel):
    """Two-stage: IDW prior (additive offset) + ridge on residuals using flow features only.

    Stage 1 — prior: for each row (i, j), IDW-weight perf(n, j) over in-fold
    training datasets n similar to i. This collapses benchmark-difficulty variance
    and dataset-level performance level into a single additive offset.

    Stage 2 — residual ridge: fit RidgeCV on (y - prior) using ONLY the original
    flow features — no IDW columns. The ridge exclusively models within-benchmark
    ranking: 'does better flow coverage predict above-average transfer?'

    Predict: prior(i, j) + ridge(flow_features(i, j))

    Training uses LOO priors (self excluded from neighbors) to avoid leakage.
    """

    def _build_prior(
        self,
        df: pd.DataFrame,
        fold_trains: list[str],
        perf_lookup: dict,
        train_mean: dict,
        eval_mean: dict,
        loo: bool,
    ) -> np.ndarray:
        fallback = float(np.nanmean(list(eval_mean.values()))) if eval_mean else 0.0
        space = self._spaces[0]
        tt_lk = self._lookup.get((space, "train_train"), {})
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        fold_evals = list(eval_mean.keys())

        def _idw_both(ti, bj, mv, mf, t_nbrs, e_nbrs):
            """Compute train-side and eval-side IDW priors.

            Train-side: perf(similar_training_datasets, bj) → interpolates dataset quality.
            Eval-side:  perf(ti, similar_benchmarks)        → interpolates benchmark difficulty.

            Uses 'has specific data' flags to decide which side(s) to trust:
            - LOTO test: train-side has benchmark-specific perf, eval-side doesn't
              (held-out td has no perf entries) → use train-side only.
            - LOBO test: eval-side has per-td perf for similar benchmarks, train-side
              only has train_mean fallback (held-out bm has no perf entries) → use eval-side only.
            - LOCO-cell / training rows: both sides have specific data → average.
            """
            t_has_specific, t_dists, t_perfs = False, [], []
            for n, dv in t_nbrs:
                pf = perf_lookup.get((n, bj, *mv),
                     perf_lookup.get((n, bj, mf),
                     perf_lookup.get((n, bj))))
                if pf is not None:
                    t_has_specific = True
                else:
                    pf = train_mean.get(n)
                if pf is not None and np.isfinite(pf):
                    t_dists.append(dv)
                    t_perfs.append(pf)

            e_has_specific, e_dists, e_perfs = False, [], []
            for e, dv in e_nbrs:
                pf = perf_lookup.get((ti, e, *mv),
                     perf_lookup.get((ti, e, mf),
                     perf_lookup.get((ti, e)))  )
                if pf is not None:
                    e_has_specific = True
                else:
                    pf = eval_mean.get(e)
                if pf is not None and np.isfinite(pf):
                    e_dists.append(dv)
                    e_perfs.append(pf)

            tp = self._idw_stats(t_dists, t_perfs)[0] if t_dists else None
            ep = self._idw_stats(e_dists, e_perfs)[0] if e_dists else None

            if t_has_specific and e_has_specific:
                return 0.5 * tp + 0.5 * ep
            elif t_has_specific:
                return tp
            elif e_has_specific:
                return ep
            elif tp is not None and ep is not None:
                return 0.5 * tp + 0.5 * ep
            return tp if tp is not None else (ep if ep is not None else None)

        if loo:
            # Training time: LOO on both axes — exclude self from each neighbor list.
            priors = np.full(len(df), np.nan)
            for idx, (_, row) in enumerate(df.iterrows()):
                ti, bj = row["train_dataset"], row["benchmark"]
                mv = self._model_variant_key(row)
                mf = mv[0]
                t_nbrs = self._neighbor_list(ti, fold_trains, tt_lk, include_self=False)
                e_nbrs = self._neighbor_list(bj, fold_evals,  ee_lk, include_self=False)
                p = _idw_both(ti, bj, mv, mf, t_nbrs, e_nbrs)
                priors[idx] = p if p is not None else eval_mean.get(bj, fallback)
            return priors

        # Test time: build neighbor lists for ALL datasets appearing in df so
        # held-out training datasets (LOTO) and held-out benchmarks (LOBO) both
        # get proper IDW from their respective in-fold neighbors.
        all_trains = list(dict.fromkeys(df["train_dataset"]))
        all_evals  = list(dict.fromkeys(df["benchmark"]))
        train_nbr = {ti: self._neighbor_list(ti, fold_trains, tt_lk, include_self=False)
                     for ti in all_trains}
        eval_nbr  = {bj: self._neighbor_list(bj, fold_evals,  ee_lk, include_self=False)
                     for bj in all_evals}

        priors = np.full(len(df), np.nan)
        for idx, (_, row) in enumerate(df.iterrows()):
            ti, bj = row["train_dataset"], row["benchmark"]
            mv = self._model_variant_key(row)
            mf = mv[0]
            p = _idw_both(ti, bj, mv, mf, train_nbr.get(ti, []), eval_nbr.get(bj, []))
            priors[idx] = p if p is not None else eval_mean.get(bj, fallback)
        return priors

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        self._fold_trains = sorted(train_df["train_dataset"].unique())
        self._fold_evals  = sorted(train_df["benchmark"].unique())

        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = make_rank_scores(train_df).reindex(train_df.index)

        self._perf_lookup = {}
        _pair_vals: dict[tuple[str, str], list[float]] = defaultdict(list)
        _family_vals: dict[tuple, list[float]] = defaultdict(list)
        _variant_vals: dict[tuple, list[float]] = defaultdict(list)
        for row_idx, row in train_df.iterrows():
            val = perf_series.loc[row_idx]
            if pd.notna(val):
                td, bm = row["train_dataset"], row["benchmark"]
                mv = self._model_variant_key(row)
                fval = float(val)
                _variant_vals[(td, bm, *mv)].append(fval)
                _family_vals[(td, bm, mv[0])].append(fval)
                _pair_vals[(td, bm)].append(fval)
        for key, vals in _variant_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for key, vals in _family_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for (td, bm), vals in _pair_vals.items():
            self._perf_lookup[(td, bm)] = float(np.mean(vals))

        self._train_mean = {
            t: float(np.mean([v for (td, _), vs in _pair_vals.items()
                               if td == t for v in vs]))
            for t in self._fold_trains if any(td == t for td, _ in _pair_vals)
        }
        self._eval_mean = {
            e: float(np.mean([v for (_, bm), vs in _pair_vals.items()
                               if bm == e for v in vs]))
            for e in self._fold_evals if any(bm == e for _, bm in _pair_vals)
        }

        # LOO IDW prior — self excluded so residuals are unbiased
        priors = self._build_prior(
            train_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=True,
        )
        valid = perf_series.notna().values
        residuals = perf_series.values - priors

        # Ridge on flow features only, fitted to residuals
        X_orig = train_df[self._feature_cols].values.astype(np.float64)
        self._fit_ridge_regressor(X_orig[valid], residuals[valid])

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        priors = self._build_prior(
            test_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=False,
        )
        X_test = test_df[self._feature_cols].values.astype(np.float64)
        return priors + self._predict_ridge_regressor(X_test)

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError("IDWPriorResidualModel requires predict_score_df(test_df)")


class IDWPriorContextModel(IDWPriorResidualModel):
    """Two-stage: context_mean[(j, mv)] prior + global ridge on context-normalised residuals.

    context_mean = mean_n y(n, j, mv) removes both benchmark-difficulty AND
    model-variant scale. The residual ridge sees only within-context ranking
    signal — exactly what Spearman measures — so feature coefficients should be
    more interpretable and Spearman should improve vs IDWPriorResidualModel.
    """

    def _build_prior(
        self,
        df: pd.DataFrame,
        fold_trains: list[str],
        perf_lookup: dict,
        train_mean: dict,
        eval_mean: dict,
        loo: bool,
    ) -> np.ndarray:
        """Use context_mean[(bm, mv)] as prior at both training and test time.

        For known benchmarks: exact context_mean keeps residuals consistent.
        For unknown benchmarks (LOBO held-out): interpolate from similar in-fold
        benchmarks via eval-eval IDW distances — mirrors what ridge_pairwise does
        with its eval-side IDW features.
        Falls back to eval_mean[bm] only when no eval-eval distances exist.
        """
        fallback = float(np.nanmean(list(eval_mean.values()))) if eval_mean else 0.0
        ctx_mean = getattr(self, "_context_mean", {})
        space = self._spaces[0]
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        fold_evals = getattr(self, "_fold_evals", list(eval_mean.keys()))

        priors = np.full(len(df), np.nan)
        for idx, (_, row) in enumerate(df.iterrows()):
            bj = row["benchmark"]
            mv = self._model_variant_key(row)
            key = (bj, *mv)
            if key in ctx_mean:
                priors[idx] = ctx_mean[key]
            else:
                # Benchmark is novel (LOBO): IDW over similar in-fold benchmarks'
                # context_means, weighted by eval-eval distance.
                e_dists, e_vals = [], []
                for e, dv in self._neighbor_list(bj, fold_evals, ee_lk, include_self=False):
                    val = ctx_mean.get((e, *mv))
                    if val is not None and np.isfinite(val):
                        e_dists.append(dv)
                        e_vals.append(val)
                if e_dists:
                    priors[idx] = self._idw_stats(e_dists, e_vals)[0]
                else:
                    priors[idx] = eval_mean.get(bj, fallback)
        return priors

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        # Step 1: parent sets up perf_lookup, imputer, scaler, and a first-pass model
        # (that first-pass model uses eval_mean as prior because _context_mean isn't
        # set yet — getattr fallback in _build_prior handles that gracefully)
        super().fit(X_train, train_df)

        # Step 2: compute context_mean per (benchmark, *model_variant)
        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = make_rank_scores(train_df).reindex(train_df.index)

        _context_vals: dict[tuple, list[float]] = defaultdict(list)
        for row_idx, row in train_df.iterrows():
            val = perf_series.loc[row_idx]
            if pd.notna(val):
                bm = row["benchmark"]
                mv = self._model_variant_key(row)
                _context_vals[(bm, *mv)].append(float(val))
        self._context_mean = {k: float(np.mean(v)) for k, v in _context_vals.items()}

        # Step 3: refit ridge on context-mean residuals (imputer/scaler unchanged)
        priors = self._build_prior(
            train_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=True,
        )
        valid = perf_series.notna().values
        residuals = perf_series.values - priors
        X_orig = train_df[self._feature_cols].values.astype(np.float64)
        self._fit_ridge_regressor(X_orig[valid], residuals[valid])


class IDWPriorContextLocalModel(IDWPriorContextModel):
    """Two-stage: context_mean prior + per-benchmark local RidgeCV.

    Fits one RidgeCV per benchmark on context-mean residuals (~10–80 rows).
    Captures benchmark-specific feature importance for within-context ranking.
    Falls back to the global ridge (from IDWPriorContextModel) for benchmarks
    not seen during training (LOBO held-out benchmark).
    """

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        # Parent does all standard setup and fits global self._model
        super().fit(X_train, train_df)

        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = make_rank_scores(train_df).reindex(train_df.index)

        priors = self._build_prior(
            train_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=True,
        )
        valid = perf_series.notna().values
        residuals = perf_series.values - priors
        X_orig = train_df[self._feature_cols].values.astype(np.float64)
        X_proc_all = self._scaler.transform(self._imputer.transform(X_orig))

        self._local_models: dict[str, RidgeCV] = {}
        for bm in self._fold_evals:
            mask = (train_df["benchmark"].values == bm) & valid
            if mask.sum() >= 5:
                self._local_models[bm] = RidgeCV(
                    alphas=[0.01, 0.1, 1.0, 10.0, 100.0]
                ).fit(X_proc_all[mask], residuals[mask])

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        priors = self._build_prior(
            test_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=False,
        )
        X_test = test_df[self._feature_cols].values.astype(np.float64)
        X_proc = self._scaler.transform(self._imputer.transform(X_test))
        # Start with global predictions as fallback, override with local models
        residual_preds = self._model.predict(X_proc)
        benchmarks = test_df["benchmark"].values
        for bm, local in self._local_models.items():
            mask = benchmarks == bm
            if mask.any():
                residual_preds[mask] = local.predict(X_proc[mask])
        return priors + residual_preds


class UniformPriorResidualModel(IDWPriorResidualModel):
    """Two-stage residual model with uniform neighbor averaging as the prior."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["weight_mode"] = "uniform"
        super().__init__(*args, **kwargs)


class RandomPriorResidualModel(IDWPriorResidualModel):
    """Two-stage residual model with deterministic random neighbor weights."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["weight_mode"] = "random"
        super().__init__(*args, **kwargs)


class IDWPriorTwoWayModel(IDWPriorResidualModel):
    """Axis-aware two-way prior + residual ridge.

    The prior uses the observed axis and interpolates only the missing axis:

    * LOTO / unseen train dataset:
        context_mean(benchmark, model_variant)
        + IDW over similar training datasets' context-demeaned residuals.

    * LOBO / unseen benchmark:
        train_mean(train_dataset, model_variant)
        + IDW over similar benchmarks' train-demeaned residuals.

    * LOCO-cell / both axes observed but the cell is held out:
        average the train-axis and eval-axis estimates.

    This keeps train-dataset variation alive for LOBO instead of falling back
    to a constant context/global mean.
    """

    def _fit_prior_tables(self, train_df: pd.DataFrame, perf_series: pd.Series) -> None:
        self._global_vals: list[float] = []
        self._variant_vals: dict[tuple, list[float]] = defaultdict(list)
        self._family_vals: dict[str, list[float]] = defaultdict(list)
        self._train_exact_vals: dict[tuple, list[float]] = defaultdict(list)
        self._train_family_vals: dict[tuple, list[float]] = defaultdict(list)
        self._train_any_vals: dict[str, list[float]] = defaultdict(list)
        self._ctx_exact_vals: dict[tuple, list[float]] = defaultdict(list)
        self._ctx_family_vals: dict[tuple, list[float]] = defaultdict(list)
        self._ctx_any_vals: dict[str, list[float]] = defaultdict(list)
        self._cell_exact_vals: dict[tuple, list[float]] = defaultdict(list)
        self._cell_family_vals: dict[tuple, list[float]] = defaultdict(list)
        self._cell_any_vals: dict[tuple, list[float]] = defaultdict(list)
        self._row_prior_keys: dict[int, dict[str, tuple | str | float]] = {}

        for row_idx, row in train_df.iterrows():
            val = perf_series.loc[row_idx]
            if pd.isna(val):
                continue
            y = float(val)
            td, bm = row["train_dataset"], row["benchmark"]
            mv = self._model_variant_key(row)
            mf = mv[0]
            train_exact = (td, *mv)
            train_family = (td, mf)
            ctx_exact = (bm, *mv)
            ctx_family = (bm, mf)
            cell_exact = (td, bm, *mv)
            cell_family = (td, bm, mf)
            cell_any = (td, bm)

            self._global_vals.append(y)
            self._variant_vals[mv].append(y)
            self._family_vals[mf].append(y)
            self._train_exact_vals[train_exact].append(y)
            self._train_family_vals[train_family].append(y)
            self._train_any_vals[td].append(y)
            self._ctx_exact_vals[ctx_exact].append(y)
            self._ctx_family_vals[ctx_family].append(y)
            self._ctx_any_vals[bm].append(y)
            self._cell_exact_vals[cell_exact].append(y)
            self._cell_family_vals[cell_family].append(y)
            self._cell_any_vals[cell_any].append(y)
            self._row_prior_keys[row_idx] = {
                "y": y,
                "mv": mv,
                "mf": mf,
                "train_exact": train_exact,
                "train_family": train_family,
                "train_any": td,
                "ctx_exact": ctx_exact,
                "ctx_family": ctx_family,
                "ctx_any": bm,
                "cell_exact": cell_exact,
                "cell_family": cell_family,
                "cell_any": cell_any,
            }

    @staticmethod
    def _mean_vals(vals: list[float] | None, exclude_y: float | None = None) -> float:
        if not vals:
            return float("nan")
        if exclude_y is None:
            return float(np.mean(vals))
        if len(vals) <= 1:
            return float("nan")
        return float((np.sum(vals) - exclude_y) / (len(vals) - 1))

    def _exclude_for(self, row_idx: int | None, key_name: str, key: tuple | str) -> float | None:
        if row_idx is None:
            return None
        info = self._row_prior_keys.get(row_idx)
        if info is None or info.get(key_name) != key:
            return None
        return float(info["y"])

    def _global_mean_for(self, row: pd.Series, row_idx: int | None = None) -> float:
        mv = self._model_variant_key(row)
        mf = mv[0]
        ex = self._exclude_for(row_idx, "mv", mv)
        val = self._mean_vals(self._variant_vals.get(mv), ex)
        if np.isfinite(val):
            return val
        ex = self._exclude_for(row_idx, "mf", mf)
        val = self._mean_vals(self._family_vals.get(mf), ex)
        if np.isfinite(val):
            return val
        ex = self._row_prior_keys.get(row_idx, {}).get("y") if row_idx is not None else None
        val = self._mean_vals(self._global_vals, ex)
        return val if np.isfinite(val) else 0.0

    def _train_mean_for(self, td: str, row: pd.Series, row_idx: int | None = None) -> float:
        mv = self._model_variant_key(row)
        mf = mv[0]
        keys = [
            ("train_exact", (td, *mv), self._train_exact_vals),
            ("train_family", (td, mf), self._train_family_vals),
            ("train_any", td, self._train_any_vals),
        ]
        for key_name, key, table in keys:
            val = self._mean_vals(table.get(key), self._exclude_for(row_idx, key_name, key))
            if np.isfinite(val):
                return val
        return float("nan")

    def _ctx_mean_for(self, bm: str, row: pd.Series, row_idx: int | None = None) -> float:
        mv = self._model_variant_key(row)
        mf = mv[0]
        keys = [
            ("ctx_exact", (bm, *mv), self._ctx_exact_vals),
            ("ctx_family", (bm, mf), self._ctx_family_vals),
            ("ctx_any", bm, self._ctx_any_vals),
        ]
        for key_name, key, table in keys:
            val = self._mean_vals(table.get(key), self._exclude_for(row_idx, key_name, key))
            if np.isfinite(val):
                return val
        return float("nan")

    def _cell_mean_for(self, td: str, bm: str, row: pd.Series,
                       row_idx: int | None = None) -> float:
        mv = self._model_variant_key(row)
        mf = mv[0]
        keys = [
            ("cell_exact", (td, bm, *mv), self._cell_exact_vals),
            ("cell_family", (td, bm, mf), self._cell_family_vals),
            ("cell_any", (td, bm), self._cell_any_vals),
        ]
        for key_name, key, table in keys:
            val = self._mean_vals(table.get(key), self._exclude_for(row_idx, key_name, key))
            if np.isfinite(val):
                return val
        return float("nan")

    def _weighted_mean(self, dists: list[float], vals: list[float]) -> float:
        if not vals:
            return float("nan")
        return float(self._idw_stats(dists, vals)[0])

    def _idw_train_mean_for(self, td: str, row: pd.Series,
                            row_idx: int | None = None) -> float:
        space = self._spaces[0]
        tt_lk = self._lookup.get((space, "train_train"), {})
        dists, vals = [], []
        for n, dv in self._neighbor_list(td, self._fold_trains, tt_lk, include_self=False):
            val = self._train_mean_for(n, row, row_idx)
            if np.isfinite(val):
                dists.append(dv)
                vals.append(val)
        return self._weighted_mean(dists, vals)

    def _idw_ctx_mean_for(self, bm: str, row: pd.Series,
                          row_idx: int | None = None) -> float:
        space = self._spaces[0]
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        dists, vals = [], []
        for e, dv in self._neighbor_list(bm, self._fold_evals, ee_lk, include_self=False):
            val = self._ctx_mean_for(e, row, row_idx)
            if np.isfinite(val):
                dists.append(dv)
                vals.append(val)
        return self._weighted_mean(dists, vals)

    def _train_axis_prior(self, row: pd.Series, row_idx: int | None = None) -> float:
        """Known/evaluable context; interpolate missing train effect within context."""
        td, bm = row["train_dataset"], row["benchmark"]
        ctx = self._ctx_mean_for(bm, row, row_idx)
        if not np.isfinite(ctx):
            return float("nan")

        space = self._spaces[0]
        tt_lk = self._lookup.get((space, "train_train"), {})
        dists, residuals = [], []
        for n, dv in self._neighbor_list(td, self._fold_trains, tt_lk, include_self=False):
            cell = self._cell_mean_for(n, bm, row, row_idx)
            if np.isfinite(cell):
                dists.append(dv)
                residuals.append(cell - ctx)
        resid = self._weighted_mean(dists, residuals)
        if np.isfinite(resid):
            return ctx + resid
        train = self._idw_train_mean_for(td, row, row_idx)
        glob = self._global_mean_for(row, row_idx)
        return train + ctx - glob if np.isfinite(train) else float("nan")

    def _eval_axis_prior(self, row: pd.Series, row_idx: int | None = None) -> float:
        """Known/evaluable train dataset; interpolate missing benchmark/context effect."""
        td, bm = row["train_dataset"], row["benchmark"]
        train = self._train_mean_for(td, row, row_idx)
        if not np.isfinite(train):
            return float("nan")

        space = self._spaces[0]
        ee_lk = self._lookup.get((space, "eval_eval"), {})
        dists, residuals = [], []
        for e, dv in self._neighbor_list(bm, self._fold_evals, ee_lk, include_self=False):
            cell = self._cell_mean_for(td, e, row, row_idx)
            if np.isfinite(cell):
                dists.append(dv)
                residuals.append(cell - train)
        resid = self._weighted_mean(dists, residuals)
        if np.isfinite(resid):
            return train + resid
        ctx = self._idw_ctx_mean_for(bm, row, row_idx)
        glob = self._global_mean_for(row, row_idx)
        return train + ctx - glob if np.isfinite(ctx) else float("nan")

    def _additive_prior(self, row: pd.Series, row_idx: int | None = None) -> float:
        td, bm = row["train_dataset"], row["benchmark"]
        glob = self._global_mean_for(row, row_idx)
        train = self._train_mean_for(td, row, row_idx)
        if not np.isfinite(train):
            train = self._idw_train_mean_for(td, row, row_idx)
        ctx = self._ctx_mean_for(bm, row, row_idx)
        if not np.isfinite(ctx):
            ctx = self._idw_ctx_mean_for(bm, row, row_idx)
        if np.isfinite(train) and np.isfinite(ctx):
            return train + ctx - glob
        if np.isfinite(train):
            return train
        if np.isfinite(ctx):
            return ctx
        return glob

    def _row_prior(self, row: pd.Series, row_idx: int | None = None) -> float:
        train_axis = self._train_axis_prior(row, row_idx)
        eval_axis = self._eval_axis_prior(row, row_idx)
        vals = [v for v in (train_axis, eval_axis) if np.isfinite(v)]
        if vals:
            return float(np.mean(vals))
        return self._additive_prior(row, row_idx)

    def _build_prior(
        self,
        df: pd.DataFrame,
        fold_trains: list[str],
        perf_lookup: dict,
        train_mean: dict,
        eval_mean: dict,
        loo: bool,
    ) -> np.ndarray:
        del fold_trains, perf_lookup, train_mean, eval_mean
        return np.array([
            self._row_prior(row, row_idx if loo else None)
            for row_idx, row in df.iterrows()
        ], dtype=np.float64)

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        self._fold_trains = sorted(train_df["train_dataset"].unique())
        self._fold_evals = sorted(train_df["benchmark"].unique())

        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = make_rank_scores(train_df).reindex(train_df.index)

        self._fit_prior_tables(train_df, perf_series)

        # Keep the parent-style lookup/means populated for debugging and any
        # downstream helper code that expects these attributes.
        self._perf_lookup = {}
        for key, vals in self._cell_exact_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for key, vals in self._cell_family_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for key, vals in self._cell_any_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        self._train_mean = {k: float(np.mean(v)) for k, v in self._train_any_vals.items()}
        self._eval_mean = {k: float(np.mean(v)) for k, v in self._ctx_any_vals.items()}

        priors = self._build_prior(
            train_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=True,
        )
        valid = perf_series.notna().values
        residuals = perf_series.values - priors

        X_orig = train_df[self._feature_cols].values.astype(np.float64)
        self._fit_ridge_regressor(X_orig[valid], residuals[valid])

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        priors = self._build_prior(
            test_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=False,
        )
        X_test = test_df[self._feature_cols].values.astype(np.float64)
        return priors + self._predict_ridge_regressor(X_test)


class UniformPriorTwoWayModel(IDWPriorTwoWayModel):
    """Axis-aware residual model with uniform neighbor averaging."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["weight_mode"] = "uniform"
        super().__init__(*args, **kwargs)


class RandomPriorTwoWayModel(IDWPriorTwoWayModel):
    """Axis-aware residual model with deterministic random neighbor weights."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["weight_mode"] = "random"
        super().__init__(*args, **kwargs)


class IDWPriorTwoWaySplineModel(IDWPriorTwoWayModel):
    """Axis-aware prior + spline-expanded residual ridge."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_spline"] = True
        super().__init__(*args, **kwargs)


class IDWPriorTwoWayRankResidualModel(IDWPriorTwoWayModel):
    """Axis-aware prior + residual ridge with a split-aware ranking residual.

    The absolute prior is the same two-way axis-aware prior as
    IDWPriorTwoWayModel.  The residual stage is fit with two signals:

      1. direct residual regression for absolute calibration;
      2. pairwise residual differences inside the ranking axis relevant to the
         outer split.

    For LOTO, pairs are formed inside (benchmark, model variant), ranking train
    datasets.  For LOBO, pairs are formed inside (train_dataset, model variant),
    ranking benchmarks.  LOCO-style splits use both pair families.
    """

    def __init__(
        self,
        *args,
        split_name: str | None = None,
        max_pairs_per_group: int = 2000,
        rank_margin: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._split_name = split_name or "auto"
        self._max_pairs_per_group = max_pairs_per_group
        self._rank_margin = rank_margin
        self._rank_model = None
        self._rank_imputer = None
        self._rank_scaler = None
        self._residual_combiner = None

    def _rank_group_specs(self) -> list[list[str]]:
        variant = ["model_family", "pretrained", "freeze"]
        split = str(self._split_name)
        if split.startswith("lobo"):
            return [["train_dataset", *variant]]
        if split.startswith("loto"):
            return [["benchmark", *variant]]
        if split in {"loco", "loco_cell", "joint_cell"}:
            return [["benchmark", *variant], ["train_dataset", *variant]]
        return [["benchmark", *variant]]

    @staticmethod
    def _fit_stage(
        X: np.ndarray,
        y: np.ndarray,
        alphas: list[float] | None = None,
        fit_intercept: bool = True,
    ) -> dict:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        y = np.where(np.isfinite(y), y, np.nan)
        valid_rows = np.isfinite(y)
        X = X[valid_rows]
        y = y[valid_rows]
        keep_cols = ~np.all(~np.isfinite(X), axis=0)
        if keep_cols.any():
            X = X[:, keep_cols]
        else:
            X = np.zeros((len(y), 1), dtype=np.float64)

        imputer = SimpleImputer(strategy="median").fit(X)
        X_imp = np.nan_to_num(imputer.transform(X), nan=0.0, posinf=0.0, neginf=0.0)
        scaler = StandardScaler().fit(X_imp)
        X_proc = np.nan_to_num(scaler.transform(X_imp), nan=0.0, posinf=0.0, neginf=0.0)
        try:
            model = RidgeCV(
                alphas=alphas or [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                fit_intercept=fit_intercept,
            ).fit(X_proc, y)
        except Exception:
            model = Ridge(
                alpha=100.0,
                solver="lsqr",
                fit_intercept=fit_intercept,
            ).fit(X_proc, y)
        return {
            "keep_cols": keep_cols,
            "imputer": imputer,
            "scaler": scaler,
            "model": model,
        }

    @staticmethod
    def _predict_stage(stage: dict, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        keep_cols = stage["keep_cols"]
        if keep_cols.any():
            X = X[:, keep_cols]
        else:
            X = np.zeros((len(X), 1), dtype=np.float64)
        X_proc = stage["scaler"].transform(stage["imputer"].transform(X))
        X_proc = np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)
        return stage["model"].predict(X_proc)

    def _rank_design(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        X = np.where(np.isfinite(X), X, np.nan)
        if fit:
            keep_cols = ~np.all(~np.isfinite(X), axis=0)
            if not keep_cols.any():
                keep_cols = np.ones(X.shape[1], dtype=bool)
            self._rank_keep_cols = keep_cols
            X = X[:, keep_cols]
            self._rank_imputer = SimpleImputer(strategy="median").fit(X)
            X_imp = np.nan_to_num(
                self._rank_imputer.transform(X), nan=0.0, posinf=0.0, neginf=0.0
            )
            self._rank_scaler = StandardScaler().fit(X_imp)
        else:
            keep_cols = getattr(self, "_rank_keep_cols", np.ones(X.shape[1], dtype=bool))
            X = X[:, keep_cols]
            X_imp = np.nan_to_num(
                self._rank_imputer.transform(X), nan=0.0, posinf=0.0, neginf=0.0
            )
        X_proc = self._rank_scaler.transform(X_imp)
        return np.nan_to_num(X_proc, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_rank_pairs(
        self,
        X_proc: np.ndarray,
        train_df: pd.DataFrame,
        perf_values: np.ndarray,
        residuals: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        pos_by_idx = {idx: pos for pos, idx in enumerate(train_df.index)}
        diffs: list[np.ndarray] = []
        targets: list[float] = []
        rng = np.random.default_rng(0)

        valid_df = train_df.loc[valid].copy()
        for group_cols in self._rank_group_specs():
            group_cols = [c for c in group_cols if c in valid_df.columns]
            if not group_cols:
                continue
            for _, grp in valid_df.groupby(group_cols, dropna=False):
                if len(grp) < 2:
                    continue
                pairs = []
                for idx_i, idx_j in itertools.combinations(grp.index, 2):
                    pi, pj = pos_by_idx[idx_i], pos_by_idx[idx_j]
                    ydiff = float(perf_values[pi] - perf_values[pj])
                    if abs(ydiff) <= self._rank_margin:
                        continue
                    pairs.append((pi, pj))
                if len(pairs) > self._max_pairs_per_group:
                    keep = rng.choice(len(pairs), size=self._max_pairs_per_group, replace=False)
                    pairs = [pairs[i] for i in sorted(keep)]
                for pi, pj in pairs:
                    d = X_proc[pi] - X_proc[pj]
                    rdiff = float(residuals[pi] - residuals[pj])
                    diffs.append(d)
                    targets.append(rdiff)
                    diffs.append(-d)
                    targets.append(-rdiff)

        if not diffs:
            return (
                np.zeros((0, X_proc.shape[1]), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        return np.vstack(diffs).astype(np.float64), np.asarray(targets, dtype=np.float64)

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        del X_train
        self._fold_trains = sorted(train_df["train_dataset"].unique())
        self._fold_evals = sorted(train_df["benchmark"].unique())

        if self._target_col is not None and self._target_col in train_df.columns:
            perf_series = train_df[self._target_col].copy().reindex(train_df.index)
        else:
            perf_series = make_rank_scores(train_df).reindex(train_df.index)

        self._fit_prior_tables(train_df, perf_series)
        self._perf_lookup = {}
        for key, vals in self._cell_exact_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for key, vals in self._cell_family_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        for key, vals in self._cell_any_vals.items():
            self._perf_lookup[key] = float(np.mean(vals))
        self._train_mean = {k: float(np.mean(v)) for k, v in self._train_any_vals.items()}
        self._eval_mean = {k: float(np.mean(v)) for k, v in self._ctx_any_vals.items()}

        priors = self._build_prior(
            train_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=True,
        )
        perf_values = perf_series.values.astype(np.float64)
        valid = perf_series.notna().values & np.isfinite(priors)
        residuals = perf_values - priors
        X_orig = train_df[self._feature_cols].values.astype(np.float64)

        self._abs_stage = self._fit_stage(X_orig[valid], residuals[valid])

        X_rank = self._rank_design(X_orig, fit=True)
        D, dy = self._build_rank_pairs(X_rank, train_df, perf_values, residuals, valid)
        if len(dy) >= 10 and np.nanstd(dy) > 0:
            try:
                self._rank_model = RidgeCV(
                    alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
                    fit_intercept=False,
                ).fit(D, dy)
            except Exception:
                self._rank_model = Ridge(
                    alpha=100.0,
                    solver="lsqr",
                    fit_intercept=False,
                ).fit(D, dy)
        else:
            self._rank_model = None

        abs_pred = self._predict_stage(self._abs_stage, X_orig)
        rank_pred = (
            self._rank_model.predict(X_rank)
            if self._rank_model is not None
            else np.zeros(len(train_df), dtype=np.float64)
        )
        combo_X = np.column_stack([abs_pred, rank_pred])
        try:
            self._residual_combiner = RidgeCV(
                alphas=[0.1, 1.0, 10.0, 100.0, 1000.0],
            ).fit(combo_X[valid], residuals[valid])
        except Exception:
            self._residual_combiner = Ridge(alpha=100.0, solver="lsqr").fit(
                combo_X[valid], residuals[valid]
            )

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        priors = self._build_prior(
            test_df, self._fold_trains,
            self._perf_lookup, self._train_mean, self._eval_mean,
            loo=False,
        )
        X_test = test_df[self._feature_cols].values.astype(np.float64)
        abs_pred = self._predict_stage(self._abs_stage, X_test)
        X_rank = self._rank_design(X_test, fit=False)
        rank_pred = (
            self._rank_model.predict(X_rank)
            if self._rank_model is not None
            else np.zeros(len(test_df), dtype=np.float64)
        )
        residual_pred = self._residual_combiner.predict(np.column_stack([abs_pred, rank_pred]))
        return priors + residual_pred


class UniformPriorTwoWaySplineModel(UniformPriorTwoWayModel):
    """Axis-aware uniform prior + spline-expanded residual ridge."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_spline"] = True
        super().__init__(*args, **kwargs)


class RandomPriorTwoWaySplineModel(RandomPriorTwoWayModel):
    """Axis-aware random prior + spline-expanded residual ridge."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["use_spline"] = True
        super().__init__(*args, **kwargs)


class RandomBaseline:
    def __init__(self, n_seeds: int = 1000):
        self.n_seeds = n_seeds

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        pass

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        # Averaged scores reduce variance — evaluation handles ranking per context
        rng = np.random.default_rng(0)
        scores = np.zeros(len(X_test))
        for seed in range(self.n_seeds):
            scores += rng.random(len(X_test))
        return scores / self.n_seeds


class GlobalPriorBaseline:
    """Mean rank per train_dataset across training contexts."""

    def fit(self, X_train: np.ndarray, train_df: pd.DataFrame) -> None:
        rank_scores = make_rank_scores(train_df).reindex(train_df.index)
        df = train_df.copy()
        df["_rs"] = rank_scores
        self._prior = df.groupby("train_dataset")["_rs"].mean().to_dict()
        self._fallback = 0.5

    def predict_score_df(self, test_df: pd.DataFrame) -> np.ndarray:
        return test_df["train_dataset"].map(self._prior).fillna(self._fallback).values

    def predict_score(self, X_test: np.ndarray) -> np.ndarray:
        # Used when test_df is not available — returns fallback
        return np.full(len(X_test), self._fallback)


# Per-variant (metric_col, is_similarity) config for IDW-based ridge_pairwise_* models.
# is_similarity=True: metric is a coverage score (higher=closer), weight = sim directly.
# is_similarity=False: metric is a distance (lower=closer), weight = 1/(dist - min + eps).
_PAIRWISE_VARIANT_COLS: dict[str, tuple[str, bool]] = {
    "ridge_pairwise_nn":      ("mean_nn_sym",            False),  # NN distance, symmetric
    "ridge_pairwise_eps1px":  ("a_covered_by_b_eps1px",  True),   # ε-coverage at 1px
    "ridge_pairwise_eps16px": ("a_covered_by_b_eps16px", True),   # ε-coverage at 16px
    "ridge_pairwise_kl":      ("kl_a_to_b_k5",           False),  # KL divergence (can be <0)
}

# Coupled feature-group → pairwise metric mapping for the generic ridge_pairwise
# model. This avoids the redundant metric × feature-group cross product:
# each row uses one concept both as train/eval features and as the IDW
# neighborhood metric, where a train-train/eval-eval pairwise analog exists.
_PAIRWISE_FEATURE_GROUP_COLS: dict[str, tuple[str, bool]] = {
    "flow_nn":       ("mean_nn_sym",           False),
    "flow_eps":      ("a_covered_by_b_eps1px", True),
    "flow_km":       ("a_covered_by_b_eps1px", True),
    "flow_kl":       ("kl_a_to_b_k5",          False),
    "motion":        ("mean_nn_sym",           False),
    "motion_km":     ("mean_nn_sym",           False),
    # Symmetric train-eval feature baselines use matching train-train/eval-eval
    # self-distances after running build_symmetric_self_distances.py.
    "flow_mmd_only": ("flow_mmd_self",         False),
    "flow_fid_only": ("flow_fid_self",         False),
    "flow_w2_only":  ("flow_sliced_w2_self",   False),
    "sym_flow":      ("mean_nn_sym",           False),
    # Density baselines: IDW uses size-based distances (|log N_a - log N_b|),
    # consistent with how flow features use flow-space distances.
    "density_train": ("log_n_dist",            False),
    "density_eval":  ("log_n_dist",            False),
    "density_idw":   ("log_n_dist",            False),
    "random_idw":    ("random_dist",           False),
    "sample_count":         ("sample_count_dist",   False),
    "sample_count_train":   ("sample_count_dist",   False),
    "sample_count_eval":    ("sample_count_dist",   False),
    "vector_density_simple": ("vector_density_simple_dist", False),
    "train_profile_simple":  ("profile_simple_dist",        False),
    "eval_profile_simple":   ("profile_simple_dist",        False),
    "profile_simple":        ("profile_simple_dist",        False),
    "vector_density":       ("vector_density_dist", False),
    "vector_density_train": ("vector_density_dist", False),
    "vector_density_eval":  ("vector_density_dist", False),
    "train_profile":        ("profile_dist",        False),
    "eval_profile":         ("profile_dist",        False),
    "profile_density":      ("profile_dist",        False),
    "flow_mmd_profile":     ("flow_mmd_self",       False),
    "flow_fid_profile":     ("flow_fid_self",       False),
    "flow_w2_profile":      ("flow_sliced_w2_self", False),
    "flow_kl_profile":      ("kl_a_to_b_k5",        False),
    "motion_km_profile":    ("mean_nn_sym",         False),
}

GENERIC_PAIRWISE_MODELS = {
    "kernel_mixed_additive",
    "kernel_mixed_interaction",
    "anchor_additive_ridge",
    "anchor_lowrank_bilinear_ridge",
    "anchor_bilinear_ridge",
    "anchor_bilinear_shrunk_ridge",
    "ridge_pairwise",
    "ridge_pairwise_uniform",
    "ridge_pairwise_random",
    "ridge_pairwise_cross",
    "ridge_pairwise_cross_resid",
    "ridge_pairwise_cross_resid_spline",
    "idw_prior_residual",
    "uniform_prior_residual",
    "random_prior_residual",
    "idw_prior_context",
    "idw_prior_context_local",
    "idw_prior_two_way",
    "idw_prior_two_way_rank",
    "uniform_prior_two_way",
    "random_prior_two_way",
    "idw_prior_two_way_spline",
    "uniform_prior_two_way_spline",
    "random_prior_two_way_spline",
}
PAIRWISE_MODEL_NAMES = {*GENERIC_PAIRWISE_MODELS, *_PAIRWISE_VARIANT_COLS.keys()}


def resolve_pairwise_metric(
    model_name: str,
    feature_group_name: str | None,
) -> tuple[str, bool] | None:
    if model_name in _PAIRWISE_VARIANT_COLS:
        return _PAIRWISE_VARIANT_COLS[model_name]
    if model_name in GENERIC_PAIRWISE_MODELS and feature_group_name is not None:
        return _PAIRWISE_FEATURE_GROUP_COLS.get(feature_group_name)
    return None

MODEL_CLASSES = {
    "ridge":          RidgeRankModel,
    "ridge_abs":      RidgeAbsModel,
    "two_way_mixed_ridge": TwoWayMixedRidgeModel,
    "anchor_additive_ridge": AnchorAdditiveRidgeModel,
    "anchor_lowrank_bilinear_ridge": AnchorLowRankBilinearRidgeModel,
    "anchor_bilinear_ridge": AnchorBilinearRidgeModel,
    "anchor_bilinear_shrunk_ridge": AnchorBilinearShrunkRidgeModel,
    "kernel_mixed_additive": KernelMixedAdditiveModel,
    "kernel_mixed_interaction": KernelMixedInteractionModel,
    "bradley_terry":  BradleyTerryModel,
    "plackett_luce":  PlackettLuceModel,
    "kernel_ridge":   KernelRidgeModel,
    "random":         RandomBaseline,
    "global_prior":   GlobalPriorBaseline,
    # krr_tp_* registered dynamically in main() after loading self-dist CSV
    # ridge_pairwise_* instantiated specially in run_fold with per-variant metric_col
    "ridge_pairwise": RidgePairwiseDistModel,
    "ridge_pairwise_uniform": RidgePairwiseUniformModel,
    "ridge_pairwise_random": RidgePairwiseRandomModel,
    "ridge_pairwise_cross": RidgePairwiseCrossDistModel,
    "ridge_pairwise_cross_resid": RidgePairwiseCrossResidualModel,
    "ridge_pairwise_cross_resid_spline": RidgePairwiseCrossResidualSplineModel,
    "idw_prior_residual":         IDWPriorResidualModel,
    "uniform_prior_residual":     UniformPriorResidualModel,
    "random_prior_residual":      RandomPriorResidualModel,
    "idw_prior_context":          IDWPriorContextModel,
    "idw_prior_context_local":    IDWPriorContextLocalModel,
    "idw_prior_two_way":          IDWPriorTwoWayModel,
    "idw_prior_two_way_rank":     IDWPriorTwoWayRankResidualModel,
    "uniform_prior_two_way":      UniformPriorTwoWayModel,
    "random_prior_two_way":       RandomPriorTwoWayModel,
    "idw_prior_two_way_spline":   IDWPriorTwoWaySplineModel,
    "uniform_prior_two_way_spline": UniformPriorTwoWaySplineModel,
    "random_prior_two_way_spline":  RandomPriorTwoWaySplineModel,
    **{name: RidgePairwiseDistModel for name in _PAIRWISE_VARIANT_COLS},
}

# These models do not consume the selected feature columns for prediction.
# Running them across every feature group creates duplicate work and duplicate
# report rows; the first requested feature group is enough.
FEATURE_INDEPENDENT_MODELS = {
    "random",
    "global_prior",
    *KRR_TP_CONFIGS.keys(),
}

# Ridge+IDW models use a base train-eval feature set plus IDW neighbor features.
# They should not be crossed with every feature group by default; that turns the
# experiment into a hard-to-interpret interaction grid. Prefer one stable flow
# base group and compare the pairwise neighborhood metric variants.
PAIRWISE_BASE_FEATURE_PREFERENCE = ("motion_km", "motion", "flow_km", "flow_nn")


def choose_pairwise_feature_group(requested: list[str], explicit: str | None = None) -> str:
    if explicit:
        return explicit
    requested_set = set(requested)
    for fg in PAIRWISE_BASE_FEATURE_PREFERENCE:
        if fg in requested_set:
            return fg
    return requested[0]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _margin_weighted_kendall(y_true: np.ndarray, y_pred: np.ndarray,
                              tau: float = 0.1) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    weighted_concordant, total_weight = 0.0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(y_true[i] - y_true[j])
            w = min(diff / tau, 1.0)
            if w == 0:
                continue
            total_weight += w
            sign_true = np.sign(y_true[i] - y_true[j])
            sign_pred = np.sign(y_pred[i] - y_pred[j])
            if sign_true == sign_pred and sign_true != 0:
                weighted_concordant += w
    if total_weight == 0:
        return float("nan")
    return 2.0 * weighted_concordant / total_weight - 1.0


def _ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    if len(y_true) < 2:
        return float("nan")
    k = min(k, len(y_true))
    pred_order = np.argsort(-y_pred)[:k]
    ideal_order = np.argsort(-y_true)[:k]
    gains      = (2.0 ** y_true[pred_order]  - 1.0) / np.log2(np.arange(2, k + 2))
    ideal_gains = (2.0 ** y_true[ideal_order] - 1.0) / np.log2(np.arange(2, k + 2))
    idcg = ideal_gains.sum()
    return float(gains.sum() / idcg) if idcg > 0 else float("nan")


def evaluate_context(group: pd.DataFrame) -> dict:
    if group["train_dataset"].nunique() < 3:
        return {}
    y    = group["auc_normalized"].values.astype(np.float64)
    pred = group["pred_score"].values.astype(np.float64)
    n    = len(group)
    true_rank = group["auc_normalized"].rank(ascending=False, method="average").values
    pred_rank = group["pred_score"].rank(ascending=False, method="average").values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sp = spearmanr(true_rank, pred_rank).statistic
        kt = kendalltau(true_rank, pred_rank).statistic
    return {
        "spearman":       float(sp) if not np.isnan(sp) else float("nan"),
        "rank_mae":       float(np.mean(np.abs(true_rank - pred_rank))),
        "norm_rank_mae":  float(np.mean(np.abs(true_rank - pred_rank)) / (n - 1)),
        "kendall":        float(kt) if not np.isnan(kt) else float("nan"),
        "margin_kendall": _margin_weighted_kendall(y, pred, tau=0.1),
        "ndcg_3":         _ndcg_at_k(y, pred, k=3),
        "ndcg_5":         _ndcg_at_k(y, pred, k=5),
        "mae":            float(np.mean(np.abs(y - pred))),
        "rmse":           float(np.sqrt(np.mean((y - pred) ** 2))),
        "n":              n,
    }


def _context_spearman(pred_df: pd.DataFrame) -> float:
    scores = []
    for _, grp in pred_df.groupby("context_id"):
        if grp["train_dataset"].nunique() < 3:
            continue
        m = evaluate_context(grp)
        if m:
            scores.append(m["spearman"])
    return float(np.nanmean(scores)) if scores else float("nan")


def evaluate_predictions(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Returns (context_metrics_df, aggregate_dict)."""
    rows = []
    for ctx, grp in pred_df.groupby("context_id"):
        m = evaluate_context(grp)
        if m:
            m["context_id"] = ctx
            rows.append(m)
    ctx_df = pd.DataFrame(rows)
    if ctx_df.empty:
        return ctx_df, {}
    metric_cols = ["spearman", "rank_mae", "norm_rank_mae", "kendall",
                   "margin_kendall", "ndcg_3", "ndcg_5", "mae", "rmse"]
    agg = {}
    for col in metric_cols:
        vals = ctx_df[col].dropna()
        agg[f"{col}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        agg[f"{col}_median"] = float(vals.median()) if len(vals) else float("nan")
        agg["n_contexts"] = len(ctx_df)
    return ctx_df, agg


def bootstrap_ci(ctx_df: pd.DataFrame, metric_col: str,
                 unit_col: str = "context_id", n_boot: int = 1000) -> tuple[float, float]:
    units = ctx_df[unit_col].unique()
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        sampled = rng.choice(units, size=len(units), replace=True)
        vals = ctx_df.set_index(unit_col).loc[sampled, metric_col].dropna()
        means.append(float(vals.mean()))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Fold runner
# ---------------------------------------------------------------------------

def run_fold(fold_id: str, train_df: pd.DataFrame, test_df: pd.DataFrame,
             model_name: str, feature_cols: list[str],
             feature_group_name: str | None = None,
             split_name: str | None = None,
             self_dist_df: pd.DataFrame | None = None,
             target_col: str | None = None,
             pairwise_spaces: list[str] | None = None,
             exclude_fit_train_datasets: set[str] | None = None,
             debug: bool = False) -> pd.DataFrame:
    assert set(test_df.index).isdisjoint(set(train_df.index)), "Index leakage detected!"
    if exclude_fit_train_datasets:
        before = len(train_df)
        train_df = train_df[~train_df["train_dataset"].isin(exclude_fit_train_datasets)]
        if train_df.empty:
            raise ValueError(
                f"Fold {fold_id!r} has no training rows after excluding "
                f"{sorted(exclude_fit_train_datasets)}"
            )
        if debug and before != len(train_df):
            print(f"  excluded {before - len(train_df)} fit rows from "
                  f"{sorted(exclude_fit_train_datasets)}")

    preprocessor = fit_preprocessor(train_df, feature_cols)
    X_train = apply_preprocessor(train_df, feature_cols, preprocessor)
    X_test  = apply_preprocessor(test_df,  feature_cols, preprocessor)

    model_cls = MODEL_CLASSES[model_name]
    pairwise_metric = resolve_pairwise_metric(model_name, feature_group_name)
    if model_name == "two_way_mixed_ridge":
        model = model_cls(feature_cols=feature_cols, target_col=target_col)
    elif pairwise_metric is not None and self_dist_df is not None:
        metric_col, is_similarity = pairwise_metric
        extra_kwargs = {}
        if model_name == "idw_prior_two_way_rank":
            extra_kwargs["split_name"] = split_name
        model = model_cls(
            self_dist_df, feature_cols,
            metric_col=metric_col, is_similarity=is_similarity,
            target_col=target_col,
            spaces=pairwise_spaces,
            **extra_kwargs,
        )
    else:
        model = model_cls()
    model.fit(X_train, train_df)

    # predict_score: some models need the full test dataframe
    if isinstance(model, (GlobalPriorBaseline, TensorProductKRRModel,
                          RidgePairwiseDistModel, TwoWayMixedRidgeModel)):
        pred_score = model.predict_score_df(test_df)
    else:
        pred_score = model.predict_score(X_test)

    out = test_df[["train_dataset", "benchmark", "model_family", "pretrained",
                    "freeze", "context_id", "auc_normalized"]].copy()
    out["pred_score"] = pred_score
    out["fold_id"]    = fold_id

    if debug:
        agg = evaluate_predictions(out)[1]
        sp = agg.get('spearman_mean', float('nan'))
        note = " (NaN expected for LOTO/LOCO single fold — needs all folds)" if np.isnan(sp) else ""
        print(f"  fold={fold_id!r:30s}  test_rows={len(out):4d}  "
              f"spearman={sp:.3f}  rank_mae={agg.get('rank_mae_mean', float('nan')):.3f}{note}")
    return out


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(df: pd.DataFrame, split_name: str, model_name: str,
                   feature_cols: list[str], feature_group_name: str,
                   out_dir: Path, self_dist_df: pd.DataFrame | None = None,
                   target_col: str | None = None,
                   pairwise_spaces: list[str] | None = None,
                   exclude_fit_train_datasets: set[str] | None = None,
                   debug: bool = False) -> None:
    split_fn = SPLIT_FNS[split_name]
    all_preds = []
    folds = list(split_fn(df))
    n_folds = len(folds)
    for i, (fold_id, train_df, test_df) in enumerate(folds):
        if len(feature_cols) == 0:
            print(f"  WARNING: no features for group — skipping")
            return
        print(f"  {split_name:14s} {model_name:15s} {feature_group_name:20s}  fold {i+1}/{n_folds}", flush=True)
        pred_df = run_fold(str(fold_id), train_df, test_df,
                           model_name, feature_cols, feature_group_name=feature_group_name,
                           split_name=split_name,
                           self_dist_df=self_dist_df, target_col=target_col,
                           pairwise_spaces=pairwise_spaces,
                           exclude_fit_train_datasets=exclude_fit_train_datasets,
                           debug=debug)
        all_preds.append(pred_df)

    if not all_preds:
        return
    pred_df = pd.concat(all_preds, ignore_index=True)

    # Evaluate
    ctx_df, agg = evaluate_predictions(pred_df)
    if not ctx_df.empty:
        # Join per-context metadata needed for bootstrap grouping.
        # benchmark and model_family are unique per context_id across all splits.
        # For LOTO fold_id is NOT unique per context_id (each context appears in
        # every fold), so we bootstrap over contexts instead of folds.
        ctx_meta = (pred_df.drop_duplicates("context_id")
                           .set_index("context_id")[["benchmark", "model_family"]])
        ctx_df = ctx_df.join(ctx_meta, on="context_id")

        bootstrap_unit = {
            "loto": "context_id", "loto_grouped": "context_id",
            "lobo": "benchmark",  "loco": "context_id",
            "loco_cell": "benchmark",  # pool across cells; bootstrap over benchmarks
            "joint_cell": "benchmark",  # pool across cells; bootstrap over benchmarks
            "lomo": "model_family",
        }.get(split_name, "context_id")

        for metric_col in ["spearman", "rank_mae", "norm_rank_mae", "ndcg_3", "mae", "rmse"]:
            if metric_col not in ctx_df.columns:
                continue
            lo, hi = bootstrap_ci(ctx_df, metric_col, unit_col=bootstrap_unit)
            agg[f"{metric_col}_ci_lo"] = lo
            agg[f"{metric_col}_ci_hi"] = hi

    agg["split"] = split_name
    agg["model"] = model_name
    agg["feature_group"] = feature_group_name

    if not debug:
        print(f"  {split_name:14s} {model_name:15s} {feature_group_name:20s} "
              f"Spearman={agg.get('spearman_mean', float('nan')):.3f}  "
              f"RankMAE={agg.get('rank_mae_mean', float('nan')):.3f}")

    # Save
    result_dir = out_dir / split_name / model_name / feature_group_name
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(result_dir / "predictions.csv", index=False)
    if not ctx_df.empty:
        ctx_df.to_csv(result_dir / "metrics.csv", index=False)

    return agg


def _profile_axis_frame(df: pd.DataFrame, axis: str, cols: list[str]) -> pd.DataFrame:
    id_col = "train_dataset" if axis == "train" else "benchmark"
    present = [c for c in cols if c in df.columns]
    if not present:
        return pd.DataFrame()
    out = df[[id_col] + present].drop_duplicates(subset=[id_col]).set_index(id_col)
    rename = {}
    for col in present:
        name = col
        name = name.replace(f"log_{axis}_", "log_")
        name = name.replace(f"{axis}_", "")
        rename[col] = name
    return out.rename(columns=rename)


def _pairwise_profile_distances(profile: pd.DataFrame) -> dict[tuple[str, str], float]:
    if profile.empty:
        return {}
    X = profile.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    names = profile.index.astype(str).tolist()
    out: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            out[(a, b)] = float(np.linalg.norm(X[i] - X[j]))
    return out


def add_profile_distance_columns(self_dist_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add sample/vector profile distances for IDW controls.

    Distances are computed separately for train datasets and eval benchmarks,
    then written onto train_train and eval_eval rows.  This keeps the profile
    control priors from falling back to total-vector-count geometry.
    """
    if self_dist_df.empty:
        return self_dist_df

    sample_cols = {
        "train": ["log_train_n_samples"],
        "eval": ["log_eval_n_samples"],
    }
    simple_vector_cols = {
        "train": ["log_train_valid_vectors_per_sample_capped"],
        "eval": ["log_eval_valid_vectors_per_sample_capped"],
    }
    vector_cols = {
        "train": [
            "log_train_valid_vectors_per_sample",
            "log_train_sampled_vectors_per_sample",
            "log_train_retained_vectors_per_sample",
            "log_train_valid_vectors_mean",
            "log_train_valid_vectors_median",
            "log_train_valid_vectors_p10",
            "log_train_valid_vectors_p90",
            "log_train_valid_vectors_p95",
            "log_train_sampled_vectors_mean",
            "log_train_sampled_vectors_median",
            "train_zero_image_frac",
        ],
        "eval": [
            "log_eval_valid_vectors_per_sample",
            "log_eval_sampled_vectors_per_sample",
            "log_eval_retained_vectors_per_sample",
            "log_eval_valid_vectors_mean",
            "log_eval_valid_vectors_median",
            "log_eval_valid_vectors_p10",
            "log_eval_valid_vectors_p90",
            "log_eval_valid_vectors_p95",
            "log_eval_sampled_vectors_mean",
            "log_eval_sampled_vectors_median",
            "eval_zero_image_frac",
        ],
    }

    out = self_dist_df.copy()
    specs = {
        "sample_count_dist": sample_cols,
        "vector_density_simple_dist": simple_vector_cols,
        "profile_simple_dist": {
            "train": sample_cols["train"] + simple_vector_cols["train"],
            "eval": sample_cols["eval"] + simple_vector_cols["eval"],
        },
        "vector_density_dist": vector_cols,
        "profile_dist": {
            "train": sample_cols["train"] + vector_cols["train"],
            "eval": sample_cols["eval"] + vector_cols["eval"],
        },
    }
    for dist_col, by_axis in specs.items():
        out[dist_col] = np.nan
        for axis, pair_type in [("train", "train_train"), ("eval", "eval_eval")]:
            profile = _profile_axis_frame(df, axis, by_axis[axis])
            lookup = _pairwise_profile_distances(profile)
            if not lookup:
                continue
            mask = out["pair_type"] == pair_type
            out.loc[mask, dist_col] = [
                lookup.get((str(a), str(b)), np.nan)
                for a, b in zip(out.loc[mask, "dataset_a"], out.loc[mask, "dataset_b"])
            ]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table",
        default="scripts/transfer_analysis_v3/transfer_table.csv")
    parser.add_argument("--splits", nargs="+",
        default=["loto", "lobo", "joint_cell", "lomo"],
        choices=list(SPLIT_FNS.keys()))
    parser.add_argument("--models", nargs="+",
        default=["ridge", "bradley_terry", "plackett_luce", "kernel_ridge",
                 "random", "global_prior"],
        # Choices validated after dynamic registration below
    )
    parser.add_argument("--self-dist-csv",
        default="analysis_v3/pairwise_self_distances.csv",
        help="Pairwise self-distances CSV (train-train + eval-eval). "
             "Required for krr_tp_* models; skipped if missing.")
    parser.add_argument("--feature-groups", nargs="+",
        default=["flow_nn", "flow_eps", "flow_km", "flow_kl",
                 "dino_nn", "dino_cov", "dino_kl",
                 "sym_flow", "sym_dino",
                 "sym_mmd", "sym_fid", "sym_w2",
                 "density", "density_train", "density_eval", "density_idw", "random_idw",
                 "sample_count", "sample_count_train", "sample_count_eval",
                 "vector_density_simple", "train_profile_simple",
                 "eval_profile_simple", "profile_simple",
                 "vector_density", "vector_density_train", "vector_density_eval",
                 "train_profile", "eval_profile", "profile_density",
                 "motion", "motion_km", "appearance",
                 "motion_appearance",
                 "flow_mmd_profile", "flow_fid_profile", "flow_w2_profile",
                 "flow_kl_profile", "motion_km_profile",
                 "all"])
    parser.add_argument("--output-dir",
        default="scripts/transfer_analysis_v3/results")
    parser.add_argument("--target", default="auc_normalized",
        help="Column to use as transfer performance target.")
    parser.add_argument("--pairwise-spaces", nargs="+", default=["flow", "dino"],
        choices=["flow", "dino"],
        help="Pairwise spaces used by ridge_pairwise_* IDW augmentations. "
             "Default uses both; pass 'flow' for flow-only ablations.")
    parser.add_argument("--exclude-fit-train-datasets", nargs="+", default=[],
        help="Training datasets to remove from every training/fitting fold while "
             "leaving test rows intact. Example: --exclude-fit-train-datasets spair")
    parser.add_argument("--drop-train-datasets", nargs="+", default=[],
        help="Remove rows with these train_dataset values before splitting/evaluation. "
             "Use this when a training dataset should disappear from reports and plots.")
    parser.add_argument("--expand-feature-independent", action="store_true",
        help="Run feature-independent models (random, global_prior, krr_tp_*) for every "
             "requested feature group. By default they run only on the first feature group.")
    parser.add_argument("--pairwise-feature-group", default=None,
        help="Base feature group for named ridge_pairwise_* variants when not expanding them. "
             "The generic ridge_pairwise model ignores this and loops over requested "
             "feature groups with a coupled feature-group→IDW metric mapping.")
    parser.add_argument("--expand-pairwise-feature-groups", action="store_true",
        help="Run ridge_pairwise_* models across every requested feature group. "
             "Default is one base feature group to avoid redundant IDW grids.")
    parser.add_argument("--debug", action="store_true",
        help="Run one fold per split/model/feature-group and print metrics inline.")
    args = parser.parse_args()

    root = Path(".").resolve()

    # Register tensor-product KRR models if pairwise distances are available
    self_dist_path = root / args.self_dist_csv
    self_dist_df: pd.DataFrame | None = None
    if self_dist_path.exists():
        from functools import partial
        self_dist_df = pd.read_csv(self_dist_path)
        for model_name, (space, kernel_col, kernel_type) in KRR_TP_CONFIGS.items():
            space_df = self_dist_df[self_dist_df["space"] == space]
            if space_df.empty or kernel_col not in space_df.columns:
                print(f"  NOTE: skipping {model_name} — column '{kernel_col}' not in "
                      f"{self_dist_path.name} (space={space})")
                continue
            MODEL_CLASSES[model_name] = partial(
                TensorProductKRRModel, space_df, kernel_col, kernel_type,
                target_col=args.target,
            )
        print(f"Registered {len(KRR_TP_CONFIGS)} krr_tp_* models from {self_dist_path.name}")
    else:
        print(f"NOTE: {self_dist_path.name} not found — krr_tp_* models unavailable. "
              f"Run compute_pairwise_self_distances.py first.")

    table_path = root / args.table
    if not table_path.exists():
        sys.exit(f"Transfer table not found: {table_path}\nRun build_table.py first.")

    df = pd.read_csv(table_path)
    if args.target != "auc_normalized" and args.target in df.columns:
        df["auc_normalized"] = df[args.target]
    if args.drop_train_datasets:
        before = len(df)
        dropped = set(args.drop_train_datasets)
        df = df[~df["train_dataset"].isin(dropped)].copy()
        print(f"Dropped train_dataset rows {sorted(dropped)}: {before} -> {len(df)}")
        if df.empty:
            sys.exit("No rows left after --drop-train-datasets")
    print(f"Loaded {len(df)} rows from {table_path}")
    print(f"  train_datasets: {df['train_dataset'].nunique()}, "
          f"benchmarks: {df['benchmark'].nunique()}, "
          f"contexts: {df['context_id'].nunique()}")
    if self_dist_df is not None:
        self_dist_df = add_profile_distance_columns(self_dist_df, df)
        profile_dist_cols = [
            c for c in [
                "sample_count_dist",
                "vector_density_simple_dist",
                "profile_simple_dist",
                "vector_density_dist",
                "profile_dist",
            ]
            if c in self_dist_df.columns
        ]
        if profile_dist_cols:
            print(f"  Added profile IDW distance columns: {profile_dist_cols}")

    feature_groups = resolve_feature_groups(df.columns.tolist())
    for fg_name, cols in feature_groups.items():
        print(f"  Feature group '{fg_name}': {len(cols)} columns")

    out_dir = root / args.output_dir
    all_agg_rows = []
    exclude_fit_train_datasets = set(args.exclude_fit_train_datasets)
    if exclude_fit_train_datasets:
        print("Excluding from all fit folds: "
              f"{sorted(exclude_fit_train_datasets)}")
    summary_path = out_dir / "summary_table.csv"
    if not args.debug and summary_path.exists():
        try:
            existing_summary = pd.read_csv(summary_path)
            if not existing_summary.empty:
                all_agg_rows.extend(existing_summary.to_dict("records"))
        except Exception:
            pass

    combos = []
    pairwise_fg = choose_pairwise_feature_group(args.feature_groups, args.pairwise_feature_group)
    for split_name, model_name in itertools.product(args.splits, args.models):
        fgroups = args.feature_groups
        if not args.expand_feature_independent and model_name in FEATURE_INDEPENDENT_MODELS:
            fgroups = args.feature_groups[:1]
        if not args.expand_pairwise_feature_groups and model_name in _PAIRWISE_VARIANT_COLS:
            fgroups = [pairwise_fg]
        combos.extend((split_name, model_name, fg_name) for fg_name in fgroups)
    if any(m in _PAIRWISE_VARIANT_COLS for m in args.models) and not args.expand_pairwise_feature_groups:
        print(f"Ridge+IDW models will use base feature group '{pairwise_fg}' "
              "(pass --expand-pairwise-feature-groups to run the full cross product).")
    if any(m in GENERIC_PAIRWISE_MODELS for m in args.models):
        supported = [fg for fg in args.feature_groups if fg in _PAIRWISE_FEATURE_GROUP_COLS]
        skipped = [fg for fg in args.feature_groups if fg not in _PAIRWISE_FEATURE_GROUP_COLS]
        print("Generic ridge_pairwise models will use coupled feature groups: "
              f"{supported or 'none'}")
        if skipped:
            print("  Skipping unsupported ridge_pairwise feature groups with no pairwise analog: "
                  f"{skipped}")
    print(f"\nRunning {len(combos)} experiments "
          f"({'debug: 1 fold each' if args.debug else 'full'})...\n")

    for split_name, model_name, fg_name in combos:
        if fg_name not in feature_groups:
            print(f"  Unknown feature group: {fg_name} — skipping")
            continue
        feature_cols = feature_groups[fg_name]
        if not feature_cols:
            print(f"  No columns for feature group '{fg_name}' (data may be missing) — skipping")
            continue
        if model_name in GENERIC_PAIRWISE_MODELS and fg_name not in _PAIRWISE_FEATURE_GROUP_COLS:
            print(f"  {split_name:14s} {model_name:15s} {fg_name:20s}  "
                  "no coupled pairwise metric — skipping")
            continue

        if args.debug:
            # One fold only
            split_fn = SPLIT_FNS[split_name]
            fold_iter = iter(split_fn(df))
            try:
                fold_id, train_df, test_df = next(fold_iter)
            except StopIteration:
                continue
            run_fold(str(fold_id), train_df, test_df, model_name, feature_cols,
                     feature_group_name=fg_name,
                     split_name=split_name,
                     self_dist_df=self_dist_df if self_dist_path.exists() else None,
                     target_col=args.target,
                     pairwise_spaces=args.pairwise_spaces,
                     exclude_fit_train_datasets=exclude_fit_train_datasets,
                     debug=True)
        else:
            # Resume: skip already-completed experiments and reload their agg row.
            result_dir = out_dir / split_name / model_name / fg_name
            metrics_path = result_dir / "metrics.csv"
            if metrics_path.exists():
                print(f"  {split_name:14s} {model_name:15s} {fg_name:20s}  already done — skipping")
                try:
                    ctx_df = pd.read_csv(metrics_path)
                    numeric_cols = [c for c in ctx_df.columns
                                    if c not in {"context_id", "benchmark", "model_family"}
                                    and pd.api.types.is_numeric_dtype(ctx_df[c])]
                    agg = {
                        "split": split_name, "model": model_name, "feature_group": fg_name,
                        "n_contexts": len(ctx_df),
                        **{f"{c}_mean":   ctx_df[c].mean()   for c in numeric_cols},
                        **{f"{c}_median": ctx_df[c].median() for c in numeric_cols},
                    }
                    all_agg_rows.append(agg)
                except Exception:
                    pass
                continue

            agg = run_experiment(df, split_name, model_name, feature_cols, fg_name,
                                 out_dir,
                                 self_dist_df=self_dist_df if self_dist_path.exists() else None,
                                 target_col=args.target,
                                 pairwise_spaces=args.pairwise_spaces,
                                 exclude_fit_train_datasets=exclude_fit_train_datasets,
                                 debug=False)
            if agg:
                all_agg_rows.append(agg)

    if not args.debug and all_agg_rows:
        summary = pd.DataFrame(all_agg_rows)
        key_cols = ["split", "model", "feature_group"]
        if all(c in summary.columns for c in key_cols):
            summary = summary.drop_duplicates(subset=key_cols, keep="last")
        summary.to_csv(summary_path, index=False)
        print(f"\n✓ Summary saved to {summary_path}")

        # Print a quick pivot of mean Spearman
        try:
            pivot = summary.pivot_table(
                values="spearman_mean",
                index=["model", "feature_group"],
                columns="split",
                aggfunc="first",
            )
            print("\nSpearman summary (mean across contexts):")
            print(pivot.round(3).to_string())
        except Exception:
            pass


if __name__ == "__main__":
    main()
