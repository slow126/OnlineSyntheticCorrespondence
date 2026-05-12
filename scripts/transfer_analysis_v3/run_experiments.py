#!/usr/bin/env python3
"""
Transfer ranking estimator experiments — strict split-first pipeline.

Leakage rule: ALL preprocessing (imputation, scaling), pairwise/listwise
construction, and global priors are computed from training rows only.

Splits:
  loto           leave one train_dataset out (24 folds)
  loto_grouped   same but collapse variant datasets to their base name
  lobo           leave one benchmark out (10 folds)
  loco           leave one context_id out (~300 folds)
  lomo           leave one model_family out (8 folds)

Models:
  ridge          RidgeCV on within-context rank scores
  bradley_terry  Logistic regression on (x_i - x_j) pairs
  plackett_luce  Custom torch PL loss, linear model
  kernel_ridge   KernelRidge(rbf), grid search
  random         Random scores (baseline)
  global_prior   Mean train-dataset rank from training fold (baseline)

Feature groups: motion, motion_km, appearance, density,
                symmetric_mmd, symmetric_ot, symmetric_all,
                motion_appearance, all

Usage:
    python scripts/transfer_analysis_v3/run_experiments.py \
        [--table scripts/transfer_analysis_v3/transfer_table.csv] \
        [--splits loto lobo] [--models ridge] [--feature-groups motion all] \
        [--output-dir scripts/transfer_analysis_v3/results] \
        [--debug]
"""

import argparse
import itertools
import re
import sys
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kendalltau
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

def resolve_feature_groups(table_cols: list[str]) -> dict[str, list[str]]:
    """Build feature groups by matching against actual column names in the table."""
    def match(patterns: list[str]) -> list[str]:
        cols = []
        for pat in patterns:
            matched = [c for c in table_cols if c == pat or c.startswith(pat)]
            cols.extend(matched)
        return list(dict.fromkeys(cols))  # deduplicate, preserve order

    FLOW_EPS_THRESHOLDS = ["eps1px", "eps4px", "eps16px"]
    flow_eps = match([f"flow_eval_covered_by_train_{t}" for t in FLOW_EPS_THRESHOLDS]
                   + [f"flow_train_covered_by_eval_{t}" for t in FLOW_EPS_THRESHOLDS])
    flow_nn  = match(["flow_mean_nn_eval_to_train_k1", "flow_mean_nn_train_to_eval_k1"])
    flow_km_eps = match([f"flow_km_eval_covered_by_train_{t}_weighted" for t in FLOW_EPS_THRESHOLDS]
                       + [f"flow_km_train_covered_by_eval_{t}_weighted" for t in FLOW_EPS_THRESHOLDS])
    # Null-calibrated coverage (preferred): witness-existence semantics,
    # threshold calibrated to the cross-set null cosine distribution.
    # Falls back to qnorm_k1 if null coverage not yet computed.
    null_cov = match(["dino_eval_covered_by_train_null90", "dino_train_covered_by_eval_null90",
                      "dino_eval_covered_by_train_null95", "dino_train_covered_by_eval_null95"])
    qnorm_fallback = match(["dino_eval_covered_by_train_qnorm_k1",
                             "dino_train_covered_by_eval_qnorm_k1"])
    dino_cov = null_cov if null_cov else qnorm_fallback
    dino_feat = match(["dino_mean_nn_eval_to_train_k1", "dino_mean_nn_train_to_eval_k1"]) + dino_cov
    density   = match(["log_train_n_vectors", "log_eval_n_vectors"])
    sym_mmd   = match(["flow_mmd", "dino_mmd"])
    sym_ot    = match(["flow_fid", "flow_sliced_w2", "dino_fid", "dino_sliced_w2"])

    groups = {
        "motion":            flow_nn + flow_eps,
        "motion_km":         flow_nn + flow_km_eps,
        "appearance":        dino_feat,
        "density":           density,
        "symmetric_mmd":     sym_mmd,
        "symmetric_ot":      sym_ot,
        "symmetric_all":     sym_mmd + sym_ot,
        "motion_appearance": flow_nn + flow_eps + dino_feat,
        "all":               flow_nn + flow_eps + dino_feat + density + sym_mmd + sym_ot,
    }
    # Deduplicate and filter to columns that actually exist
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


def iter_lomo_folds(df: pd.DataFrame) -> Iterator[tuple]:
    for entity in sorted(df["model_family"].unique()):
        test_mask = df["model_family"] == entity
        yield entity, df[~test_mask], df[test_mask]


SPLIT_FNS = {
    "loto":         lambda df: iter_loto_folds(df, grouped=False),
    "loto_grouped": lambda df: iter_loto_folds(df, grouped=True),
    "lobo":         iter_lobo_folds,
    "loco":         iter_loco_folds,
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
    def _pl_loss(scores: torch.Tensor, order: torch.Tensor, lambda_: float) -> torch.Tensor:
        """Negative log Plackett-Luce likelihood for a single ranked list."""
        s = scores[order]  # best-first
        n = len(s)
        loss = torch.tensor(0.0)
        for k in range(n):
            loss = loss - s[k] + torch.logsumexp(s[k:], dim=0)
        return loss + 0.5 * lambda_ * torch.sum(scores ** 2)

    def _fit_one(self, X: np.ndarray, df: pd.DataFrame, lambda_: float) -> np.ndarray:
        d = X.shape[1]
        beta = nn.Parameter(torch.zeros(d))
        opt = torch.optim.Adam([beta], lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        for _ in range(self.epochs):
            opt.zero_grad()
            scores = X_t @ beta
            total_loss = torch.tensor(0.0, requires_grad=False)
            for _, grp in df.groupby("context_id"):
                positions = [list(df.index).index(i) for i in grp.index]
                y_vals = grp["auc_normalized"].values
                # Sort descending (best first)
                order = torch.tensor(np.argsort(-y_vals), dtype=torch.long)
                ctx_scores = scores[positions]
                loss = PlackettLuceModel._pl_loss(ctx_scores, order, 0.0)
                total_loss = total_loss + loss
            # Add global L2
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
        alphas = [0.01, 0.1, 1.0, 10.0]
        gammas = [None, 0.01, 0.1, 1.0]  # None → 'scale'

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


MODEL_CLASSES = {
    "ridge":          RidgeRankModel,
    "bradley_terry":  BradleyTerryModel,
    "plackett_luce":  PlackettLuceModel,
    "kernel_ridge":   KernelRidgeModel,
    "random":         RandomBaseline,
    "global_prior":   GlobalPriorBaseline,
}


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
                   "margin_kendall", "ndcg_3", "ndcg_5"]
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
             debug: bool = False) -> pd.DataFrame:
    assert set(test_df.index).isdisjoint(set(train_df.index)), "Index leakage detected!"

    preprocessor = fit_preprocessor(train_df, feature_cols)
    X_train = apply_preprocessor(train_df, feature_cols, preprocessor)
    X_test  = apply_preprocessor(test_df,  feature_cols, preprocessor)

    model_cls = MODEL_CLASSES[model_name]
    model = model_cls()
    model.fit(X_train, train_df)

    # predict_score: GlobalPrior needs the dataframe for train_dataset lookup
    if isinstance(model, GlobalPriorBaseline):
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
                   out_dir: Path, debug: bool = False) -> None:
    split_fn = SPLIT_FNS[split_name]
    all_preds = []
    for fold_id, train_df, test_df in split_fn(df):
        if len(feature_cols) == 0:
            print(f"  WARNING: no features for group — skipping")
            return
        pred_df = run_fold(str(fold_id), train_df, test_df,
                           model_name, feature_cols, debug=debug)
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
            "lobo": "benchmark",  "loco": "context_id", "lomo": "model_family",
        }.get(split_name, "context_id")

        for metric_col in ["spearman", "rank_mae", "norm_rank_mae", "ndcg_3"]:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table",
        default="scripts/transfer_analysis_v3/transfer_table.csv")
    parser.add_argument("--splits", nargs="+",
        default=["loto", "lobo", "loco", "lomo"],
        choices=list(SPLIT_FNS.keys()))
    parser.add_argument("--models", nargs="+",
        default=["ridge", "bradley_terry", "plackett_luce", "kernel_ridge",
                 "random", "global_prior"],
        choices=list(MODEL_CLASSES.keys()))
    parser.add_argument("--feature-groups", nargs="+",
        default=["motion", "motion_km", "appearance", "density",
                 "symmetric_mmd", "symmetric_ot", "symmetric_all",
                 "motion_appearance", "all"])
    parser.add_argument("--output-dir",
        default="scripts/transfer_analysis_v3/results")
    parser.add_argument("--target", default="auc_normalized",
        help="Column to use as transfer performance target.")
    parser.add_argument("--debug", action="store_true",
        help="Run one fold per split/model/feature-group and print metrics inline.")
    args = parser.parse_args()

    root = Path(".").resolve()
    table_path = root / args.table
    if not table_path.exists():
        sys.exit(f"Transfer table not found: {table_path}\nRun build_table.py first.")

    df = pd.read_csv(table_path)
    if args.target != "auc_normalized" and args.target in df.columns:
        df["auc_normalized"] = df[args.target]
    print(f"Loaded {len(df)} rows from {table_path}")
    print(f"  train_datasets: {df['train_dataset'].nunique()}, "
          f"benchmarks: {df['benchmark'].nunique()}, "
          f"contexts: {df['context_id'].nunique()}")

    feature_groups = resolve_feature_groups(df.columns.tolist())
    for fg_name, cols in feature_groups.items():
        print(f"  Feature group '{fg_name}': {len(cols)} columns")

    out_dir = root / args.output_dir
    all_agg_rows = []

    combos = list(itertools.product(args.splits, args.models, args.feature_groups))
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

        if args.debug:
            # One fold only
            split_fn = SPLIT_FNS[split_name]
            fold_iter = iter(split_fn(df))
            try:
                fold_id, train_df, test_df = next(fold_iter)
            except StopIteration:
                continue
            run_fold(str(fold_id), train_df, test_df, model_name, feature_cols, debug=True)
        else:
            agg = run_experiment(df, split_name, model_name, feature_cols, fg_name,
                                 out_dir, debug=False)
            if agg:
                all_agg_rows.append(agg)

    if not args.debug and all_agg_rows:
        summary = pd.DataFrame(all_agg_rows)
        summary_path = out_dir / "summary_table.csv"
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
