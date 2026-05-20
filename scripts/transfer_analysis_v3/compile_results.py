#!/usr/bin/env python3
"""
Compile experiment results into results.md — readable tables for the paper.

Reads:
  results/summary_table.csv           (primary source)
  results/{split}/{model}/{fg}/metrics.csv  (per-context detail, if needed)

Writes:
  results/results.md

Usage:
    python scripts/transfer_analysis_v3/compile_results.py \
        [--results-dir scripts/transfer_analysis_v3/results] \
        [--output scripts/transfer_analysis_v3/results/results.md]
"""

import argparse
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr as _scipy_spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRIC_DISPLAY = {
    "spearman":       ("Spearman ↑",       "{:.3f}"),
    "rank_mae":       ("Rank MAE ↓",       "{:.2f}"),
    "norm_rank_mae":  ("Norm Rank MAE ↓",  "{:.3f}"),
    "kendall":        ("Kendall ↑",        "{:.3f}"),
    "ndcg_3":         ("NDCG@3 ↑",        "{:.3f}"),
    "ndcg_5":         ("NDCG@5 ↑",        "{:.3f}"),
    "mae":            ("MAE ↓ (abs AUC)",  "{:.2f}"),
    "rmse":           ("RMSE ↓ (abs AUC)", "{:.2f}"),
}

# Models that predict in absolute AUC space (not relative rank scores).
# MAE/RMSE are only meaningful for these.
ABSOLUTE_MODELS = frozenset([
    "krr_tp_flow_nn", "krr_tp_flow_eps", "krr_tp_flow_eps16",
    "krr_tp_dino_nn", "krr_tp_dino_eps",
    "ridge_abs", "two_way_mixed_ridge",
    "anchor_additive_ridge", "anchor_lowrank_bilinear_ridge",
    "anchor_bilinear_ridge", "anchor_bilinear_shrunk_ridge",
    "kernel_mixed_additive", "kernel_mixed_interaction",
    "ridge_pairwise", "ridge_pairwise_cross", "ridge_pairwise_cross_resid",
    "ridge_pairwise_cross_resid_spline",
    "ridge_pairwise_uniform", "ridge_pairwise_random",
    "idw_prior_residual",
    "uniform_prior_residual", "random_prior_residual",
    "idw_prior_context",
    "idw_prior_context_local",
    "idw_prior_two_way",
    "idw_prior_two_way_rank",
    "uniform_prior_two_way", "random_prior_two_way",
    "idw_prior_two_way_spline",
    "uniform_prior_two_way_spline", "random_prior_two_way_spline",
    "ridge_pairwise_nn", "ridge_pairwise_eps1px",
    "ridge_pairwise_eps16px", "ridge_pairwise_kl",
])

SPLIT_DISPLAY = {
    "loto":          "LOTO",
    "loto_grouped":  "LOTO (grouped)",
    "lobo":          "LOBO",
    "loco":          "LOCO",
    "loco_cell":     "LOCO-cell",
    "joint_cell":    "Joint-cell",
    "lomo":          "LOMO",
}

MODEL_DISPLAY = {
    "ridge":             "Ridge",
    "ridge_abs":         "Ridge (absolute)",
    "two_way_mixed_ridge": "Two-way mixed ridge",
    "anchor_additive_ridge": "Anchor additive ridge",
    "anchor_lowrank_bilinear_ridge": "Anchor low-rank bilinear ridge",
    "anchor_bilinear_ridge": "Anchor bilinear ridge",
    "anchor_bilinear_shrunk_ridge": "Anchor bilinear ridge (shrunk)",
    "kernel_mixed_additive": "Kernel mixed effects",
    "kernel_mixed_interaction": "Kernel mixed effects + interaction",
    "bradley_terry":     "Bradley-Terry",
    "plackett_luce":     "Plackett-Luce",
    "kernel_ridge":      "Kernel Ridge",
    "random":            "Random",
    "global_prior":      "Global Prior",
    "krr_tp_flow_nn":    "TP-KRR flow NN",
    "krr_tp_flow_eps":   "TP-KRR flow ε-cov 1px",
    "krr_tp_flow_eps16": "TP-KRR flow ε-cov 16px",
    "krr_tp_dino_nn":    "TP-KRR DINO NN",
    "krr_tp_dino_eps":   "TP-KRR DINO ε-cov 1px",
    "ridge_pairwise":         "Ridge + IDW (coupled)",
    "ridge_pairwise_cross":   "Ridge + IDW (2-axis)",
    "ridge_pairwise_cross_resid": "Ridge + IDW (2-axis residual)",
    "ridge_pairwise_cross_resid_spline": "Ridge + IDW (2-axis residual spline)",
    "ridge_pairwise_uniform":     "Ridge + uniform neighbors",
    "ridge_pairwise_random":      "Ridge + random neighbors",
    "idw_prior_residual":         "IDW prior + residual ridge",
    "uniform_prior_residual":     "Uniform prior + residual ridge",
    "random_prior_residual":      "Random-neighbor prior + residual ridge",
    "idw_prior_context":          "IDW prior + context ridge",
    "idw_prior_context_local":    "IDW prior + local ridge",
    "idw_prior_two_way":          "Axis-aware prior + residual ridge",
    "idw_prior_two_way_rank":     "Axis-aware prior + rank residual ridge",
    "uniform_prior_two_way":      "Axis-aware uniform prior + residual ridge",
    "random_prior_two_way":       "Axis-aware random prior + residual ridge",
    "idw_prior_two_way_spline":   "Axis-aware prior + spline residual ridge",
    "uniform_prior_two_way_spline": "Axis-aware uniform prior + spline residual ridge",
    "random_prior_two_way_spline":  "Axis-aware random prior + spline residual ridge",
    "ridge_pairwise_all":     "Ridge + IDW (all metrics)",   # legacy alias
    "ridge_pairwise_nn":      "Ridge + IDW (NN distance)",
    "ridge_pairwise_eps1px":  "Ridge + IDW (ε-cov 1px)",
    "ridge_pairwise_eps16px": "Ridge + IDW (ε-cov 16px)",
    "ridge_pairwise_kl":      "Ridge + IDW (KL div)",
}

FEATURE_DISPLAY = {
    # --- Single-concept directed ---
    "flow_nn":           "Flow: NN distance",
    "flow_eps":          "Flow: ε-coverage",
    "flow_km":           "Flow: k-means ε-coverage",
    "flow_kl":           "Flow: KL divergence",
    "dino_nn":           "DINO: NN distance",
    "dino_cov":          "DINO: coverage",
    "dino_kl":           "DINO: KL divergence",
    # --- Symmetric baselines (individual) ---
    "flow_mmd_only":     "flow MMD",
    "dino_mmd_only":     "DINO MMD",
    "flow_fid_only":     "flow FID",
    "dino_fid_only":     "DINO FID",
    "flow_w2_only":      "flow SW2",
    "dino_w2_only":      "DINO SW2",
    # --- Symmetric baselines (combined) ---
    "sym_flow":          "sym flow (MMD+FID+SW2)",
    "sym_dino":          "sym DINO (MMD+FID+SW2)",
    "sym_mmd":           "sym MMD (flow+DINO)",
    "sym_fid":           "sym FID (flow+DINO)",
    "sym_w2":            "sym SW2 (flow+DINO)",
    # --- Density ---
    "density":           "Density: log N (train+eval)",
    "density_train":     "Density: log N_train",
    "density_eval":      "Density: log N_eval",
    "density_idw":       "Density IDW: log N feature + size IDW",
    "random_idw":        "Random-ID control: random dataset codes + random distances",
    "sample_count":      "Sample count: train+eval images",
    "sample_count_train": "Sample count: train images",
    "sample_count_eval": "Sample count: eval images",
    "vector_density_simple": "Vector density: capped vectors/sample",
    "train_profile_simple": "Train profile: samples + capped vectors/sample",
    "eval_profile_simple": "Eval profile: samples + capped vectors/sample",
    "profile_simple":     "Profile control: samples + capped vectors/sample",
    "vector_density":    "Vector density: vectors/sample train+eval",
    "vector_density_train": "Vector density: train vectors/sample",
    "vector_density_eval": "Vector density: eval vectors/sample",
    "train_profile":     "Train profile: samples + vectors/sample",
    "eval_profile":      "Eval profile: samples + vectors/sample",
    "profile_density":   "Profile control: samples + vectors/sample",
    # --- Composites ---
    "motion":            "Motion (NN + ε-cov, directed)",
    "motion_km":         "Motion k-means (NN + k-means ε-cov, directed)",
    "appearance":        "Appearance (NN + coverage, directed)",
    "motion_appearance": "Motion + Appearance (directed)",
    "flow_mmd_profile":  "flow MMD + train profile",
    "flow_fid_profile":  "flow FID + train profile",
    "flow_w2_profile":   "flow SW2 + train profile",
    "flow_kl_profile":   "Flow KL + train profile",
    "motion_km_profile": "Motion k-means + train profile",
    "all":               "All features",
}

# One-line description of each group (for the legend).
FEATURE_LEGEND = {
    "flow_nn":           "Mean NN distance eval→train and train→eval in flow space [2 features]",
    "flow_eps":          "Fraction of points within ε of nearest match at 1/4/16 px [6 features]",
    "flow_km":           "K-means density-weighted ε-coverage at 1/4/16 px [6 features]",
    "flow_kl":           "kNN KL divergence KL(eval‖train) + KL(train‖eval), flow space [4 features; requires flow KL run]",
    "dino_nn":           "Mean NN distance in DINO space [2 features]",
    "dino_cov":          "Null-calibrated cosine coverage (eval→train, train→eval) [2 features]",
    "dino_kl":           "kNN KL divergence in DINO space, k=5 and k=20 [4 features]",
    "flow_mmd_only":     "MMD in flow space only [1 feature]",
    "dino_mmd_only":     "MMD in DINO space only [1 feature]",
    "flow_fid_only":     "FID (Fréchet distance) in flow space only [1 feature]",
    "dino_fid_only":     "FID (Fréchet distance) in DINO space only [1 feature]",
    "flow_w2_only":      "Sliced Wasserstein-2 in flow space only [1 feature]",
    "dino_w2_only":      "Sliced Wasserstein-2 in DINO space only [1 feature]",
    "sym_flow":          "Flow MMD + Flow FID + Flow SW2 — all symmetric metrics, flow space [3]",
    "sym_dino":          "DINO MMD + DINO FID + DINO SW2 — all symmetric metrics, DINO space [3]",
    "sym_mmd":           "Flow MMD + DINO MMD — undirected baseline, no threshold [2]",
    "sym_fid":           "Flow FID + DINO FID — Fréchet distance [2]",
    "sym_w2":            "Flow SW2 + DINO SW2 — sliced Wasserstein-2 [2]",
    "density":           "log(N_train) + log(N_eval) — dataset size only [2]",
    "density_train":     "log(N_train) only — training set size [1]",
    "density_eval":      "log(N_eval) only — eval benchmark size/density [1]",
    "density_idw":       "log(N) features + |log N_a − log N_b| IDW — no flow information anywhere [2]",
    "random_idw":        "fixed random scalar per dataset/benchmark + shuffled random distances — random identity control [2]",
    "sample_count":      "log image/sample counts for train and eval datasets [2]",
    "sample_count_train": "log image/sample count for the train dataset [1]",
    "sample_count_eval": "log image/sample count for the eval benchmark [1]",
    "vector_density_simple": "capped log valid vectors per image for train and eval datasets [2]",
    "train_profile_simple": "train sample count + capped train valid vectors per image [2]",
    "eval_profile_simple": "eval sample count + capped eval valid vectors per image [2]",
    "profile_simple":     "train+eval sample counts + capped valid vectors per image [4]",
    "vector_density":    "log vectors per image plus zero-image fractions for train and eval datasets",
    "vector_density_train": "train-set vectors per image: valid/sampled/retained counts and quantiles",
    "vector_density_eval": "eval-set vectors per image: valid/sampled/retained counts and quantiles",
    "train_profile":     "train sample count + train vectors-per-image profile",
    "eval_profile":      "eval sample count + eval vectors-per-image profile",
    "profile_density":   "train+eval sample counts and vectors-per-image profile",
    "motion":            "flow_nn + flow_eps — full directed flow description [8]",
    "motion_km":         "flow_nn + flow_km — k-means variant of motion [8]",
    "appearance":        "dino_nn + dino_cov — full directed DINO description [4]",
    "motion_appearance": "motion + appearance — all directed features, no symmetric [12]",
    "flow_mmd_profile":  "flow MMD plus train-set sample/vector profile",
    "flow_fid_profile":  "flow FID plus train-set sample/vector profile",
    "flow_w2_profile":   "flow SW2 plus train-set sample/vector profile",
    "flow_kl_profile":   "flow KL divergence plus train-set sample/vector profile",
    "motion_km_profile": "flow NN + k-means coverage plus train-set sample/vector profile",
    "all":               "All directed (motion+appearance) + all symmetric + density",
}

MODEL_ORDER   = ["random", "global_prior", "ridge", "ridge_abs",
                 "two_way_mixed_ridge",
                 "anchor_additive_ridge", "anchor_lowrank_bilinear_ridge",
                 "anchor_bilinear_ridge", "anchor_bilinear_shrunk_ridge",
                 "kernel_mixed_additive", "kernel_mixed_interaction",
                 "bradley_terry", "plackett_luce", "kernel_ridge",
                 "krr_tp_flow_nn", "krr_tp_flow_eps", "krr_tp_flow_eps16",
                 "krr_tp_dino_nn", "krr_tp_dino_eps",
                 "ridge_pairwise", "ridge_pairwise_uniform", "ridge_pairwise_random",
                 "ridge_pairwise_cross_resid", "ridge_pairwise_cross",
                 "idw_prior_residual", "uniform_prior_residual", "random_prior_residual",
                 "idw_prior_context", "idw_prior_context_local",
                 "idw_prior_two_way", "idw_prior_two_way_rank",
                 "uniform_prior_two_way", "random_prior_two_way",
                 "ridge_pairwise_cross_resid_spline",
                 "idw_prior_two_way_spline",
                 "uniform_prior_two_way_spline", "random_prior_two_way_spline",
                 "ridge_pairwise_nn", "ridge_pairwise_eps1px", "ridge_pairwise_eps16px", "ridge_pairwise_kl"]
FEATURE_ORDER = [
    # Symmetric baselines — individual (space × metric)
    "flow_mmd_only", "dino_mmd_only",
    "flow_fid_only", "dino_fid_only",
    "flow_w2_only",  "dino_w2_only",
    # Symmetric baselines — combined
    "sym_mmd", "sym_fid", "sym_w2", "sym_flow", "sym_dino",
    # Density
    "density", "density_train", "density_eval", "density_idw", "random_idw",
    "sample_count", "sample_count_train", "sample_count_eval",
    "vector_density_simple", "train_profile_simple", "eval_profile_simple", "profile_simple",
    "vector_density", "vector_density_train", "vector_density_eval",
    "train_profile", "eval_profile", "profile_density",
    # Single-concept directed
    "flow_nn", "flow_eps", "flow_km", "flow_kl",
    "dino_nn", "dino_cov", "dino_kl",
    # Composites
    "motion", "motion_km", "appearance", "motion_appearance",
    "flow_mmd_profile", "flow_fid_profile", "flow_w2_profile",
    "flow_kl_profile", "motion_km_profile",
    "all",
]
SPLIT_ORDER   = ["loto", "loto_grouped", "lobo", "joint_cell", "loco_cell", "loco", "lomo"]

DIAGNOSTIC_MODELS = [
    "ridge_abs",
    "two_way_mixed_ridge",
    "anchor_additive_ridge",
    "anchor_lowrank_bilinear_ridge",
    "anchor_bilinear_ridge",
    "anchor_bilinear_shrunk_ridge",
    "kernel_mixed_additive",
    "kernel_mixed_interaction",
    "ridge_pairwise",
    "idw_prior_two_way",
    "idw_prior_two_way_rank",
    "uniform_prior_two_way",
    "random_prior_two_way",
]
DIAGNOSTIC_BASELINES = ["random", "global_prior"]
DIAGNOSTIC_FEATURES = [
    "density_train", "density_eval", "density_idw", "random_idw",
    "sample_count", "vector_density_simple", "train_profile_simple", "profile_simple",
    "flow_fid_only", "flow_w2_only", "flow_kl", "motion_km",
    "flow_fid_profile", "flow_w2_profile", "flow_kl_profile", "motion_km_profile",
]
DIAGNOSTIC_FEATURE_FAMILIES = {
    "Panel / density controls": ["density_train", "density_eval", "density_idw", "random_idw"],
    "Sample / vector profile controls": [
        "sample_count", "vector_density_simple", "train_profile_simple", "profile_simple",
    ],
    "Flow geometry": ["flow_fid_only", "flow_w2_only", "flow_kl", "motion_km"],
    "Flow geometry + train profile": [
        "flow_fid_profile", "flow_w2_profile", "flow_kl_profile", "motion_km_profile",
    ],
}

MODEL_COLORS = {
    "random":               "#aaaaaa",
    "global_prior":         "#888888",
    "ridge":                "#4878cf",
    "ridge_abs":            "#1f77b4",
    "two_way_mixed_ridge":   "#08519c",
    "anchor_additive_ridge": "#9e9ac8",
    "anchor_lowrank_bilinear_ridge": "#807dba",
    "anchor_bilinear_ridge": "#6a51a3",
    "anchor_bilinear_shrunk_ridge": "#4a1486",
    "kernel_mixed_additive": "#756bb1",
    "kernel_mixed_interaction": "#54278f",
    "bradley_terry":        "#6acc65",
    "plackett_luce":        "#d65f5f",
    "kernel_ridge":         "#b47cc7",
    "krr_tp_flow_nn":       "#ff7f0e",
    "krr_tp_flow_eps":      "#ffbb78",
    "krr_tp_flow_eps16":    "#ffd700",
    "krr_tp_dino_nn":       "#d62728",
    "krr_tp_dino_eps":      "#ff9896",
    "ridge_pairwise_nn":    "#2ca02c",
    "ridge_pairwise":       "#2ca02c",
    "ridge_pairwise_uniform": "#9ecae1",
    "ridge_pairwise_random":  "#fdae6b",
    "ridge_pairwise_cross": "#006d2c",
    "ridge_pairwise_cross_resid": "#238b45",
    "ridge_pairwise_cross_resid_spline": "#54278f",
    "idw_prior_residual":         "#74c476",
    "uniform_prior_residual":     "#6baed6",
    "random_prior_residual":      "#fd8d3c",
    "idw_prior_context":          "#41ab5d",
    "idw_prior_context_local":    "#006837",
    "idw_prior_two_way":          "#00441b",
    "idw_prior_two_way_rank":     "#7a0177",
    "uniform_prior_two_way":      "#2171b5",
    "random_prior_two_way":       "#e6550d",
    "idw_prior_two_way_spline":   "#252525",
    "uniform_prior_two_way_spline": "#6baed6",
    "random_prior_two_way_spline":  "#fd8d3c",
    "ridge_pairwise_eps1px":  "#98df8a",
    "ridge_pairwise_eps16px": "#17becf",
    "ridge_pairwise_kl":    "#9edae5",
}

CALIBRATION_STYLE = {
    "ridge_abs":                  ("#1f77b4", "o", "-"),
    "two_way_mixed_ridge":        ("#08519c", "D", "-"),
    "anchor_additive_ridge":      ("#9e9ac8", "P", "--"),
    "anchor_lowrank_bilinear_ridge": ("#807dba", "P", "-."),
    "anchor_bilinear_ridge":      ("#6a51a3", "P", "-"),
    "anchor_bilinear_shrunk_ridge": ("#4a1486", "P", ":"),
    "kernel_mixed_additive":      ("#756bb1", "^", "-"),
    "kernel_mixed_interaction":   ("#54278f", "^", "--"),
    "ridge_pairwise":             ("#d62728", "s", "-"),
    "ridge_pairwise_uniform":     ("#6baed6", "s", "--"),
    "ridge_pairwise_random":      ("#fd8d3c", "s", ":"),
    "ridge_pairwise_cross_resid": ("#9467bd", "^", "-"),
    "ridge_pairwise_cross_resid_spline": ("#54278f", "^", ":"),
    "ridge_pairwise_cross":       ("#8c564b", "v", "--"),
    "idw_prior_residual":         ("#2ca02c", "D", "-"),
    "uniform_prior_residual":     ("#2171b5", "D", "--"),
    "random_prior_residual":      ("#e6550d", "D", ":"),
    "idw_prior_context":          ("#ff7f0e", "P", "--"),
    "idw_prior_context_local":    ("#17becf", "X", "-."),
    "idw_prior_two_way":          ("#000000", "*", "-"),
    "idw_prior_two_way_rank":     ("#7a0177", "P", "-"),
    "uniform_prior_two_way":      ("#08519c", "*", "--"),
    "random_prior_two_way":       ("#a63603", "*", ":"),
    "idw_prior_two_way_spline":   ("#252525", "X", "-"),
    "uniform_prior_two_way_spline": ("#6baed6", "X", "--"),
    "random_prior_two_way_spline":  ("#fd8d3c", "X", ":"),
    "ridge_pairwise_nn":          ("#bcbd22", "h", "--"),
    "ridge_pairwise_eps1px":      ("#e377c2", ">", "--"),
    "ridge_pairwise_eps16px":     ("#7f7f7f", "<", "--"),
    "ridge_pairwise_kl":          ("#aec7e8", "p", "--"),
    "krr_tp_flow_nn":             ("#ff9896", "8", ":"),
    "krr_tp_flow_eps":            ("#c5b0d5", "H", ":"),
    "krr_tp_flow_eps16":          ("#c49c94", "d", ":"),
}

CALIBRATION_LABEL = {
    "ridge_abs":                  "Ridge abs",
    "two_way_mixed_ridge":        "Mixed ridge",
    "anchor_additive_ridge":      "Anchor add",
    "anchor_lowrank_bilinear_ridge": "Anchor low-rank",
    "anchor_bilinear_ridge":      "Bilinear ridge",
    "anchor_bilinear_shrunk_ridge": "Bilinear shrunk",
    "kernel_mixed_additive":      "Kernel mixed",
    "kernel_mixed_interaction":   "Kernel mixed+int",
    "ridge_pairwise":             "Ridge+IDW",
    "ridge_pairwise_uniform":     "Ridge+uniform",
    "ridge_pairwise_random":      "Ridge+random",
    "ridge_pairwise_cross_resid": "2-axis resid",
    "ridge_pairwise_cross_resid_spline": "2-axis spline",
    "ridge_pairwise_cross":       "2-axis",
    "idw_prior_residual":         "IDW prior",
    "uniform_prior_residual":     "Uniform prior",
    "random_prior_residual":      "Random prior",
    "idw_prior_context":          "Context prior",
    "idw_prior_context_local":    "Local prior",
    "idw_prior_two_way":          "Axis-aware",
    "idw_prior_two_way_rank":     "Axis rank",
    "uniform_prior_two_way":      "Axis uniform",
    "random_prior_two_way":       "Axis random",
    "idw_prior_two_way_spline":   "Axis spline",
    "uniform_prior_two_way_spline": "Axis uniform spline",
    "random_prior_two_way_spline":  "Axis random spline",
    "ridge_pairwise_nn":          "IDW NN",
    "ridge_pairwise_eps1px":      "IDW eps1",
    "ridge_pairwise_eps16px":     "IDW eps16",
    "ridge_pairwise_kl":          "IDW KL",
    "krr_tp_flow_nn":             "TP-KRR NN",
    "krr_tp_flow_eps":            "TP-KRR eps",
    "krr_tp_flow_eps16":          "TP-KRR eps16",
}


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def _rescale_pred(x: "pd.Series", y: "pd.Series") -> "tuple[pd.Series, bool]":
    """Min-max rescale x to [y.min(), y.max()] when scales differ (e.g. relative ridge score).

    Returns (x_out, was_rescaled). Rescaling is a monotone transform so Spearman
    is unchanged; it maps relative predictions into AUC space so residuals and
    calibration bins are interpretable in AUC units.
    """
    x_range = float(x.max() - x.min()) if float(x.max() - x.min()) > 0 else 1.0
    y_range = float(y.max() - y.min()) if float(y.max() - y.min()) > 0 else 1.0
    mean_diff = abs(float(x.mean()) - float(y.mean()))
    scales_match = (
        max(x_range, y_range) / min(x_range, y_range) < 5
        and mean_diff < 2 * max(x_range, y_range)
    )
    if scales_match:
        return x, False
    x_rescaled = (x - x.min()) / x_range * y_range + float(y.min())
    return x_rescaled, True


def _load_preds(results_dir: Path, model: str, split: str,
                summary_df: pd.DataFrame,
                selection_metric: str = "spearman") -> "pd.DataFrame | None":
    """Load predictions.csv for the best feature group of (model, split)."""
    row = _best_result_row(summary_df, model, split, selection_metric)
    if row is None:
        return None
    path = results_dir / split / model / row["feature_group"] / "predictions.csv"
    return pd.read_csv(path) if path.exists() else None


def _load_preds_for_feature(results_dir: Path, model: str, split: str,
                            feature_group: str) -> "pd.DataFrame | None":
    """Load predictions.csv for an explicit feature group."""
    path = results_dir / split / model / feature_group / "predictions.csv"
    return pd.read_csv(path) if path.exists() else None


def _best_result_row(summary_df: pd.DataFrame, model: str, split: str,
                     selection_metric: str = "spearman") -> "pd.Series | None":
    """Best summary row for model/split under a metric.

    For absolute-scale models, figures generally pass selection_metric="mae"
    so the selected feature group is the best-calibrated configuration. Ranking
    models usually pass "spearman".
    """
    col = f"{selection_metric}_mean"
    sub = summary_df[(summary_df["model"] == model) & (summary_df["split"] == split)]
    if sub.empty or col not in sub.columns or sub[col].isna().all():
        return None
    lower_is_better = selection_metric in {"mae", "rmse", "rank_mae", "norm_rank_mae"}
    best_idx = sub[col].idxmin() if lower_is_better else sub[col].idxmax()
    return sub.loc[best_idx]


def _auc_matrix(results_dir: Path) -> "pd.DataFrame | None":
    """Scan all predictions.csv files and return mean AUC per (train, benchmark)."""
    frames = []
    for p in results_dir.glob("*/*/*/predictions.csv"):
        try:
            frames.append(pd.read_csv(p, usecols=["train_dataset", "benchmark", "auc_normalized"]))
        except Exception:
            continue
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True).dropna(subset=["auc_normalized"])
    return combined.groupby(["train_dataset", "benchmark"])["auc_normalized"].mean().unstack()


def _cluster_order(names: list, dist_df: "pd.DataFrame | None",
                   pair_type: str, space: str = "flow",
                   metric: str = "mean_nn_sym") -> list:
    """Return hierarchically clustered order of names; falls back to input order."""
    if not HAS_SCIPY or dist_df is None or len(names) < 3:
        return names
    try:
        sub = dist_df[(dist_df["pair_type"] == pair_type) & (dist_df["space"] == space)]
        idx = {n: i for i, n in enumerate(names)}
        n = len(names)
        D = np.zeros((n, n))
        for _, row in sub.iterrows():
            a, b = row.get("dataset_a"), row.get("dataset_b")
            if a in idx and b in idx and pd.notna(row.get(metric, np.nan)):
                v = float(row[metric])
                D[idx[a], idx[b]] = v
                D[idx[b], idx[a]] = v
        if np.isnan(D).any():
            return names
        Z = linkage(squareform(D, checks=False), method="ward")
        return [names[i] for i in leaves_list(Z)]
    except Exception:
        return names


def _fig1_scatter(results_dir: Path, summary_df: pd.DataFrame, fig_dir: Path,
                  available_models: list, available_splits: list,
                  color_by: str = "benchmark",
                  filename: str = "fig1_scatter.png") -> "Path | None":
    """Predicted vs actual AUC scatter, faceted by split × model."""
    scatter_models = [m for m in
                      ["ridge", "ridge_abs", "ridge_pairwise", "ridge_pairwise_cross_resid",
                       "ridge_pairwise_cross_resid_spline",
                       "idw_prior_residual", "idw_prior_context", "idw_prior_context_local",
                       "idw_prior_two_way", "idw_prior_two_way_spline",
                       "ridge_pairwise_nn",
                       "ridge_pairwise_eps1px"]
                      if m in available_models]
    target_splits = [s for s in ["loto", "lobo", "joint_cell", "loco_cell", "loco"] if s in available_splits]
    if not scatter_models or not target_splits:
        return None

    n_m, n_s = len(scatter_models), len(target_splits)
    fig, axes = plt.subplots(n_s, n_m, figsize=(3.5 * n_m, 3.5 * n_s), squeeze=False)

    if color_by not in {"benchmark", "train_dataset"}:
        raise ValueError(f"Unsupported scatter color column: {color_by}")
    color_label = "Benchmark" if color_by == "benchmark" else "Training dataset"
    palette_lookup: dict = {}
    for ri, split in enumerate(target_splits):
        for ci, model in enumerate(scatter_models):
            ax = axes[ri][ci]
            selection_metric = "mae" if model in ABSOLUTE_MODELS and "mae_mean" in summary_df.columns else "spearman"
            best_row = _best_result_row(summary_df, model, split,
                                        selection_metric=selection_metric)
            preds = _load_preds(results_dir, model, split, summary_df,
                                selection_metric=selection_metric)
            if preds is None or preds.empty:
                ax.text(0.5, 0.5, "No run for this split", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                if ri == 0:
                    ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=8)
                ax.set_axis_off()
                continue

            x = preds["pred_score"].astype(float)
            y = preds["auc_normalized"].astype(float)
            x_range = float(x.max() - x.min()) if float(x.max() - x.min()) > 0 else 1.0
            y_range = float(y.max() - y.min()) if float(y.max() - y.min()) > 0 else 1.0
            groups = sorted(preds[color_by].unique())
            if not palette_lookup:
                palette_name = "tab10" if len(groups) <= 10 else "tab20"
                palette = sns.color_palette(palette_name, len(groups))
                palette_lookup.update(dict(zip(groups, palette)))

            for group_name in groups:
                mask = preds[color_by] == group_name
                ax.scatter(x[mask], y[mask],
                           c=[palette_lookup.get(group_name, "#999")],
                           alpha=0.55, s=16, linewidths=0)

            # For scatter: don't rescale — use independent axes when scales differ
            # so the raw prediction distribution is visible.
            _, scales_match = _rescale_pred(x, y)
            scales_match = not scales_match  # _rescale_pred returns was_rescaled

            if scales_match:
                # Shared-axis plot with y=x diagonal
                pad = max(x_range, y_range) * 0.05
                lo = min(float(x.min()), float(y.min())) - pad
                hi = max(float(x.max()), float(y.max())) + pad
                ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.8, alpha=0.45)
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                xlabel = "Predicted AUC"
            else:
                # Independent axes: pred is on a different scale (e.g. relative ridge score)
                # Draw an OLS regression line so the trend is still visible
                x_pad = x_range * 0.05
                y_pad = y_range * 0.05
                ax.set_xlim(float(x.min()) - x_pad, float(x.max()) + x_pad)
                ax.set_ylim(float(y.min()) - y_pad, float(y.max()) + y_pad)
                xv, yv = x.values, y.values
                m_ols = float(np.cov(xv, yv)[0, 1] / np.var(xv)) if np.var(xv) > 0 else 0.0
                b_ols = float(yv.mean()) - m_ols * float(xv.mean())
                xs = np.array([float(x.min()), float(x.max())])
                ax.plot(xs, m_ols * xs + b_ols, "--", color="black", lw=0.8, alpha=0.45)
                xlabel = "Pred Score (relative)"

            if HAS_SCIPY and len(x) >= 3:
                r, _ = _scipy_spearmanr(x.values, y.values)
                ax.text(0.05, 0.95, f"ρ = {r:.2f}", transform=ax.transAxes,
                        va="top", fontsize=8, fontweight="bold")

            if ri == 0:
                metric_label = "MAE" if selection_metric == "mae" else "ρ"
                fg = best_row["feature_group"] if best_row is not None else "?"
                ax.set_title(
                    f"{MODEL_DISPLAY.get(model, model)}\n"
                    f"{FEATURE_DISPLAY.get(fg, fg)} ({metric_label}-selected)",
                    fontsize=8,
                )
            if ci == 0:
                ax.set_ylabel(f"{SPLIT_DISPLAY.get(split, split)}\nActual AUC", fontsize=8)
            if ri == n_s - 1:
                ax.set_xlabel(xlabel, fontsize=8)
            ax.tick_params(labelsize=7)

    if palette_lookup:
        patches = [mpatches.Patch(color=c, label=b)
                   for b, c in palette_lookup.items()]
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 6),
                   fontsize=7, bbox_to_anchor=(0.5, -0.03), framealpha=0.9,
                   title=color_label, title_fontsize=7)

    fig.suptitle(f"Predicted vs Actual AUC — colored by {color_label.lower()}",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    path = fig_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig1_random_idw_scatter(results_dir: Path, summary_df: pd.DataFrame,
                             fig_dir: Path, available_models: list,
                             available_splits: list) -> "Path | None":
    """Predicted vs actual scatter for the random_idw control feature group."""
    del summary_df
    scatter_models = [m for m in
                      ["ridge_pairwise", "ridge_pairwise_cross_resid",
                       "ridge_pairwise_cross_resid_spline",
                       "idw_prior_residual", "idw_prior_two_way",
                       "idw_prior_two_way_spline"]
                      if m in available_models]
    target_splits = [s for s in ["loto", "lobo", "joint_cell", "loco_cell"] if s in available_splits]
    if not scatter_models or not target_splits:
        return None

    n_m, n_s = len(scatter_models), len(target_splits)
    fig, axes = plt.subplots(n_s, n_m, figsize=(3.25 * n_m, 3.15 * n_s),
                             squeeze=False)
    palette_lookup: dict = {}

    for ri, split in enumerate(target_splits):
        for ci, model in enumerate(scatter_models):
            ax = axes[ri][ci]
            preds = _load_preds_for_feature(results_dir, model, split, "random_idw")
            if preds is None or preds.empty:
                ax.text(0.5, 0.5, "No random_idw run", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=8)
                ax.set_axis_off()
                continue

            x = preds["pred_score"].astype(float)
            y = preds["auc_normalized"].astype(float)
            groups = sorted(preds["benchmark"].unique())
            if not palette_lookup:
                palette_name = "tab10" if len(groups) <= 10 else "tab20"
                palette_lookup.update(dict(zip(groups, sns.color_palette(palette_name, len(groups)))))

            for group_name in groups:
                mask = preds["benchmark"] == group_name
                ax.scatter(x[mask], y[mask], c=[palette_lookup.get(group_name, "#999")],
                           alpha=0.55, s=15, linewidths=0)

            x_range = float(x.max() - x.min()) if float(x.max() - x.min()) > 0 else 1.0
            y_range = float(y.max() - y.min()) if float(y.max() - y.min()) > 0 else 1.0
            pad = max(x_range, y_range) * 0.05
            lo = min(float(x.min()), float(y.min())) - pad
            hi = max(float(x.max()), float(y.max())) + pad
            ax.plot([lo, hi], [lo, hi], "--", color="black", lw=0.8, alpha=0.45)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

            if HAS_SCIPY and len(x) >= 3:
                r, _ = _scipy_spearmanr(x.values, y.values)
                mae = float(np.mean(np.abs(y.values - x.values)))
                ax.text(0.05, 0.95, f"ρ = {r:.2f}\nMAE = {mae:.2f}",
                        transform=ax.transAxes, va="top", fontsize=7,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=1.5))

            if ri == 0:
                ax.set_title(MODEL_DISPLAY.get(model, model), fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{SPLIT_DISPLAY.get(split, split)}\nActual AUC", fontsize=8)
            if ri == n_s - 1:
                ax.set_xlabel("Predicted AUC", fontsize=8)
            ax.tick_params(labelsize=7)

    if palette_lookup:
        patches = [mpatches.Patch(color=c, label=b) for b, c in palette_lookup.items()]
        fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 6),
                   fontsize=7, bbox_to_anchor=(0.5, -0.03), framealpha=0.9,
                   title="Benchmark", title_fontsize=7)

    fig.suptitle("Random-IDW Control: Predicted vs Actual AUC",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    path = fig_dir / "fig1c_random_idw_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig2_auc_matrix(results_dir: Path, fig_dir: Path,
                     dist_df: "pd.DataFrame | None") -> "Path | None":
    """Ground-truth AUC heatmap with similarity-ordered rows/cols."""
    matrix = _auc_matrix(results_dir)
    if matrix is None or matrix.empty:
        return None

    trains = _cluster_order(list(matrix.index), dist_df, "train_train")
    evals  = _cluster_order(list(matrix.columns), dist_df, "eval_eval")
    matrix = matrix.reindex(index=trains, columns=evals)

    fig, ax = plt.subplots(figsize=(max(8, len(evals) * 1.1), max(5, len(trains) * 0.75)))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.4, linecolor="white",
                cbar_kws={"label": "AUC (normalized)", "shrink": 0.75})
    ax.set_xlabel("Benchmark", fontsize=10)
    ax.set_ylabel("Training Dataset", fontsize=10)
    ax.set_title("Ground-Truth Transfer Performance Matrix\n"
                 "(ordered by flow-space distribution similarity)", fontsize=10)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    plt.tight_layout()
    path = fig_dir / "fig2_auc_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig3_residuals(results_dir: Path, summary_df: pd.DataFrame, fig_dir: Path,
                    available_models: list, available_splits: list,
                    dist_df: "pd.DataFrame | None") -> "Path | None":
    """Side-by-side residual heatmaps: Ridge vs best IDW variant."""
    compare = [m for m in ["ridge_abs", "ridge_pairwise", "ridge_pairwise_nn", "ridge_pairwise_eps1px"]
               if m in available_models]
    split = next((s for s in ["loto", "lobo", "joint_cell", "loco_cell"] if s in available_splits), None)
    if not compare or split is None:
        return None

    matrix = _auc_matrix(results_dir)
    if matrix is None or matrix.empty:
        return None
    train_ord = _cluster_order(list(matrix.index), dist_df, "train_train")
    eval_ord  = _cluster_order(list(matrix.columns), dist_df, "eval_eval")

    resid_maps, max_abs = [], 0.0
    rescaled_models = set()
    for model in compare:
        selection_metric = "mae" if model in ABSOLUTE_MODELS and "mae_mean" in summary_df.columns else "spearman"
        preds = _load_preds(results_dir, model, split, summary_df,
                            selection_metric=selection_metric)
        if preds is None or preds.empty:
            resid_maps.append(None)
            continue
        preds = preds.copy()
        x = preds["pred_score"].astype(float)
        y = preds["auc_normalized"].astype(float)
        x, was_rescaled = _rescale_pred(x, y)
        if was_rescaled:
            rescaled_models.add(model)
        preds["residual"] = x - y
        rm = preds.groupby(["train_dataset", "benchmark"])["residual"].mean().unstack()
        rm = rm.reindex(index=train_ord, columns=eval_ord)
        resid_maps.append(rm)
        cur = rm.abs().max().max()
        if np.isfinite(cur):
            max_abs = max(max_abs, cur)

    if max_abs == 0:
        return None

    n_rows = max((rm.shape[0] for rm in resid_maps if rm is not None and not rm.empty), default=0)
    n_cols = max((rm.shape[1] for rm in resid_maps if rm is not None and not rm.empty), default=0)
    fig_h = max(5.0, min(12.0, 0.38 * max(n_rows, 1)))
    fig_w = max(5.5 * len(compare), 0.48 * max(n_cols, 1) * len(compare))
    fig, axes = plt.subplots(1, len(compare), figsize=(fig_w, fig_h), squeeze=False)
    for ci, (model, rm) in enumerate(zip(compare, resid_maps)):
        ax = axes[0][ci]
        if rm is None or rm.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        # Cell annotations are useful on the pure 11x10 matrix but unreadable on
        # larger mixed-dataset matrices. Disable them once the grid gets dense.
        annot_cells = rm.size <= 120
        sns.heatmap(rm, annot=annot_cells, fmt=".1f", cmap="RdBu_r", ax=ax,
                    center=0, vmin=-max_abs, vmax=max_abs,
                    linewidths=0.4, linecolor="white",
                    annot_kws={"fontsize": 5},
                    cbar_kws={"label": "Pred − Actual", "shrink": 0.75})
        ax.set_title(f"{MODEL_DISPLAY.get(model, model)}\n"
                     f"({SPLIT_DISPLAY.get(split, split)})", fontsize=9)
        ax.tick_params(axis="x", rotation=35, labelsize=7)
        ax.tick_params(axis="y", rotation=0, labelsize=7)
        if ci > 0:
            ax.set_ylabel("")

    rescale_note = ""
    if rescaled_models:
        names = ", ".join(MODEL_DISPLAY.get(m, m) for m in sorted(rescaled_models))
        rescale_note = f"\n({names}: pred score min-max rescaled to AUC range)"
    fig.suptitle(f"Prediction Residuals: mean(Predicted − Actual) per cell{rescale_note}",
                 fontsize=10)
    plt.tight_layout()
    path = fig_dir / "fig3_residuals.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig4_model_bar(summary_df: pd.DataFrame, fig_dir: Path,
                    available_models: list, available_splits: list) -> "Path | None":
    """Horizontal bar chart: best Spearman per model per split."""
    col = "spearman_mean"
    if col not in summary_df.columns:
        return None
    plot_models = [m for m in MODEL_ORDER
                   if m in available_models and m not in {"random", "global_prior"}]
    if not plot_models or not available_splits:
        return None

    rows = []
    for model in plot_models:
        for split in available_splits:
            sub = summary_df[(summary_df["model"] == model) & (summary_df["split"] == split)]
            if sub.empty or col not in sub.columns or sub[col].isna().all():
                continue
            best_idx = sub[col].idxmax()
            v   = float(sub.loc[best_idx, col])
            lo  = float(sub.loc[best_idx, "spearman_ci_lo"]) if "spearman_ci_lo" in sub.columns else np.nan
            hi  = float(sub.loc[best_idx, "spearman_ci_hi"]) if "spearman_ci_hi" in sub.columns else np.nan
            rows.append({"model": model, "split": split, "spearman": v, "ci_lo": lo, "ci_hi": hi})

    if not rows:
        return None
    plot_df = pd.DataFrame(rows)

    n_s = len(available_splits)
    fig, axes = plt.subplots(1, n_s, figsize=(5 * n_s, max(4, len(plot_models) * 0.45)),
                             sharey=False, squeeze=False)

    for ci, split in enumerate(available_splits):
        ax = axes[0][ci]
        sub = (plot_df[plot_df["split"] == split]
               .set_index("model")
               .reindex([m for m in plot_models if m in plot_df["model"].values]))
        sub = sub.dropna(subset=["spearman"])
        if sub.empty:
            ax.set_axis_off()
            continue

        y_pos  = np.arange(len(sub))
        colors = [MODEL_COLORS.get(m, "#888888") for m in sub.index]
        ax.barh(y_pos, sub["spearman"], color=colors, height=0.6, alpha=0.85)

        for yi, (model, row) in enumerate(sub.iterrows()):
            lo, hi = row.get("ci_lo", np.nan), row.get("ci_hi", np.nan)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(row["spearman"], yi,
                            xerr=[[row["spearman"] - lo], [hi - row["spearman"]]],
                            fmt="none", color="black", capsize=3, lw=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_DISPLAY.get(m, m) for m in sub.index], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Spearman ρ", fontsize=9)
        ax.set_title(SPLIT_DISPLAY.get(split, split), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.suptitle("Model Comparison — Best Spearman per Split (with 95% CI)", fontsize=11, y=1.01)
    plt.tight_layout()
    path = fig_dir / "fig4_model_bar.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig5_calibration(results_dir: Path, summary_df: pd.DataFrame, fig_dir: Path,
                      available_models: list, available_splits: list) -> "Path | None":
    """Calibration curves: mean-predicted vs mean-actual in equal-count bins.

    Only includes models that predict in absolute AUC space (IDW variants, KRR-TP).
    Ridge/BT/PL produce relative rank scores; min-max rescaling skews their
    distribution into a narrow band, making calibration bins collapse into a
    near-vertical line. Calibration is only meaningful for absolute-scale models.
    """
    cal_models = [m for m in
                  ["ridge_abs", "ridge_pairwise", "ridge_pairwise_cross_resid",
                   "ridge_pairwise_cross_resid_spline",
                   "idw_prior_residual", "idw_prior_context", "idw_prior_context_local",
                   "idw_prior_two_way", "idw_prior_two_way_spline",
                   "ridge_pairwise_cross",
                   "ridge_pairwise_nn", "ridge_pairwise_eps1px",
                   "ridge_pairwise_eps16px", "ridge_pairwise_kl",
                   "krr_tp_flow_nn", "krr_tp_flow_eps", "krr_tp_flow_eps16"]
                  if m in available_models]
    target_splits = [s for s in ["loto", "lobo", "joint_cell", "loco_cell"] if s in available_splits]
    if not cal_models or not target_splits:
        return None

    n_s = len(target_splits)
    fig, axes = plt.subplots(1, n_s, figsize=(5.5 * n_s, 4.5), squeeze=False)

    for ci, split in enumerate(target_splits):
        ax = axes[0][ci]
        lo_global, hi_global = np.inf, -np.inf
        has_line = False

        for model in cal_models:
            preds = _load_preds(results_dir, model, split, summary_df,
                                selection_metric="mae")
            if preds is None or preds.empty:
                continue
            x = preds["pred_score"].astype(float).values
            y = preds["auc_normalized"].astype(float).values
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) < 20:
                continue

            n_bins = 10
            order = np.argsort(x)
            bin_size = len(x) // n_bins
            mx, my = [], []
            for b in range(n_bins):
                sl = order[b * bin_size: (b + 1) * bin_size if b < n_bins - 1 else None]
                mx.append(x[sl].mean())
                my.append(y[sl].mean())

            color, marker, linestyle = CALIBRATION_STYLE.get(
                model, (MODEL_COLORS.get(model, "#888"), "o", "-")
            )
            label = CALIBRATION_LABEL.get(model, MODEL_DISPLAY.get(model, model))
            ax.plot(mx, my, marker=marker, linestyle=linestyle, color=color,
                    lw=1.7, ms=5.5, markeredgecolor="white", markeredgewidth=0.4,
                    alpha=0.92, label=label)
            # Direct labels make the many IDW-family curves distinguishable even
            # when the legend colors are close or the lines overlap.
            ax.annotate(
                label,
                xy=(mx[-1], my[-1]),
                xytext=(5, (len(ax.lines) % 5 - 2) * 3),
                textcoords="offset points",
                color=color,
                fontsize=6.5,
                va="center",
                clip_on=True,
            )
            lo_global = min(lo_global, min(mx), min(my))
            hi_global = max(hi_global, max(mx), max(my))
            has_line = True

        if not has_line:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.set_axis_off()
            continue

        pad = (hi_global - lo_global) * 0.05
        lims = (lo_global - pad, hi_global + pad)
        ax.plot(lims, lims, "--", color="black", lw=0.8, alpha=0.45, label="Ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Mean Predicted AUC", fontsize=9)
        ax.set_ylabel("Mean Actual AUC", fontsize=9)
        ax.set_title(SPLIT_DISPLAY.get(split, split), fontsize=10)
        ax.legend(fontsize=6.5, loc="upper left", frameon=True, framealpha=0.85,
                  title="Best MAE feature group", title_fontsize=7)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)

    fig.suptitle("Calibration: Mean Predicted vs Mean Actual AUC (10 equal-count bins; pooled by split)",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    path = fig_dir / "fig5_calibration.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig5_calibration_spread(results_dir: Path, summary_df: pd.DataFrame, fig_dir: Path,
                             available_models: list, available_splits: list) -> "Path | None":
    """Small-multiple calibration ribbons showing within-bin spread.

    Each panel bins predictions by predicted AUC. The solid line is mean actual
    AUC per bin, the dark band is the 25-75% range, and the light band is the
    10-90% range. This shows dispersion hidden by mean-only calibration curves.
    """
    spread_models = [m for m in
                     ["ridge_abs", "ridge_pairwise", "ridge_pairwise_cross_resid",
                      "ridge_pairwise_cross_resid_spline",
                      "idw_prior_residual", "idw_prior_two_way",
                      "idw_prior_two_way_spline"]
                     if m in available_models]
    target_splits = [s for s in ["loto", "lobo", "joint_cell", "loco_cell"] if s in available_splits]
    if not spread_models or not target_splits:
        return None

    n_s, n_m = len(target_splits), len(spread_models)
    fig, axes = plt.subplots(n_s, n_m, figsize=(3.2 * n_m, 3.0 * n_s),
                             squeeze=False, sharex=False, sharey=False)

    for ri, split in enumerate(target_splits):
        for ci, model in enumerate(spread_models):
            ax = axes[ri][ci]
            best_row = _best_result_row(summary_df, model, split, selection_metric="mae")
            preds = _load_preds(results_dir, model, split, summary_df,
                                selection_metric="mae")
            if preds is None or preds.empty or best_row is None:
                ax.text(0.5, 0.5, "No run", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                ax.set_axis_off()
                continue

            x = preds["pred_score"].astype(float).values
            y = preds["auc_normalized"].astype(float).values
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) < 20:
                ax.text(0.5, 0.5, "Too few points", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                ax.set_axis_off()
                continue

            n_bins = min(10, max(4, len(x) // 12))
            order = np.argsort(x)
            bins = np.array_split(order, n_bins)
            mx, my, q10, q25, q75, q90 = [], [], [], [], [], []
            for sl in bins:
                xb = x[sl]
                yb = y[sl]
                mx.append(float(np.mean(xb)))
                my.append(float(np.mean(yb)))
                q10.append(float(np.quantile(yb, 0.10)))
                q25.append(float(np.quantile(yb, 0.25)))
                q75.append(float(np.quantile(yb, 0.75)))
                q90.append(float(np.quantile(yb, 0.90)))

            color, marker, linestyle = CALIBRATION_STYLE.get(
                model, (MODEL_COLORS.get(model, "#888"), "o", "-")
            )
            ax.fill_between(mx, q10, q90, color=color, alpha=0.10, linewidth=0)
            ax.fill_between(mx, q25, q75, color=color, alpha=0.24, linewidth=0)
            ax.plot(mx, my, marker=marker, linestyle=linestyle, color=color,
                    lw=1.8, ms=4.8, markeredgecolor="white", markeredgewidth=0.4)

            lo = min(np.min(x), np.min(y), np.min(q10))
            hi = max(np.max(x), np.max(y), np.max(q90))
            pad = (hi - lo) * 0.05 if hi > lo else 1.0
            lims = (lo - pad, hi + pad)
            ax.plot(lims, lims, "--", color="black", lw=0.75, alpha=0.40)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            if ri == n_s - 1:
                ax.set_xlabel("Predicted AUC", fontsize=8)
            if ci == 0:
                ax.set_ylabel(f"{SPLIT_DISPLAY.get(split, split)}\nActual AUC", fontsize=8)

            title = CALIBRATION_LABEL.get(model, MODEL_DISPLAY.get(model, model))
            if ri == 0:
                ax.set_title(title, fontsize=9)
            mae = best_row.get("mae_mean", float("nan"))
            sp = best_row.get("spearman_mean", float("nan"))
            fg = FEATURE_DISPLAY.get(best_row.get("feature_group", ""), best_row.get("feature_group", ""))
            ax.text(0.03, 0.97, f"MAE {fmt(mae, '{:.2f}')}\nρ {fmt(sp)}\n{fg}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=6.4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=1.5))
            ax.grid(alpha=0.25)
            ax.tick_params(labelsize=7)

    fig.suptitle("Calibration Spread: mean line, 25-75% band, 10-90% band",
                 fontsize=10, y=1.01)
    plt.tight_layout()
    path = fig_dir / "fig5b_calibration_spread.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_figures(results_dir: Path, summary_df: pd.DataFrame,
                     available_models: list, available_splits: list,
                     dist_csv: "Path | None" = None) -> dict:
    """Generate all five figures; return {name: Path} for each that succeeded."""
    if not HAS_MATPLOTLIB:
        print("  (matplotlib not available — skipping figures)")
        return {}

    sns.set_theme(style="whitegrid", font_scale=0.9)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    dist_df = None
    if dist_csv is not None and dist_csv.exists():
        try:
            dist_df = pd.read_csv(dist_csv)
        except Exception:
            pass

    jobs = [
        ("fig1", _fig1_scatter,    (results_dir, summary_df, fig_dir, available_models, available_splits)),
        ("fig1_train", _fig1_scatter,
         (results_dir, summary_df, fig_dir, available_models, available_splits,
          "train_dataset", "fig1_scatter_by_train.png")),
        ("fig1_random", _fig1_random_idw_scatter,
         (results_dir, summary_df, fig_dir, available_models, available_splits)),
        ("fig2", _fig2_auc_matrix, (results_dir, fig_dir, dist_df)),
        ("fig3", _fig3_residuals,  (results_dir, summary_df, fig_dir, available_models, available_splits, dist_df)),
        ("fig4", _fig4_model_bar,  (summary_df, fig_dir, available_models, available_splits)),
        ("fig5", _fig5_calibration,(results_dir, summary_df, fig_dir, available_models, available_splits)),
        ("fig5b", _fig5_calibration_spread,
         (results_dir, summary_df, fig_dir, available_models, available_splits)),
    ]
    out = {}
    print("Generating figures...")
    for name, fn, args in jobs:
        try:
            p = fn(*args)
            if p is not None:
                out[name] = p
                print(f"  ✓ {p.name}")
            else:
                print(f"  – {name}: skipped (no data)")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return out


def figures_md(fig_paths: dict, results_dir: Path) -> str:
    """Build the Figures section with embedded image links."""
    if not fig_paths:
        return ""
    captions = {
        "fig1": (
            "**Figure 1 — Predicted vs Actual AUC**",
            "Each point is a (train, benchmark) test prediction colored by benchmark. "
            "Dashed diagonal = perfect calibration. ρ = Spearman on all test points. "
            "Rows: available held-out splits, usually LOTO, LOBO, and LOCO-cell. "
            "Columns: selected models at their best feature group for that split: "
            "absolute-scale models are selected by lowest MAE, while ranking models are "
            "selected by highest Spearman. The selected feature group is shown in the "
            "column title. Missing panels mean that model was not run for that split. "
            "Ranking models (BT, PL, Ridge rank) produce relative scores so they won't "
            "hug the diagonal even when their Spearman is high.",
        ),
        "fig1_train": (
            "**Figure 1b — Predicted vs Actual AUC, Colored by Training Set**",
            "Same panels and feature-group selection as Figure 1, but points are colored "
            "by training dataset instead of benchmark. This view is useful for spotting "
            "row-wise dataset bias, such as one training set being consistently over- or "
            "under-predicted across benchmarks and model variants.",
        ),
        "fig1_random": (
            "**Figure 1c — Random-IDW Control Scatter**",
            "Same absolute prediction scatter as Figure 1, but forced to the `random_idw` "
            "feature group for the IDW-family models. This isolates how much of the apparent "
            "absolute prediction quality comes from the borrowing mechanism rather than from "
            "meaningful flow-space neighborhoods.",
        ),
        "fig2": (
            "**Figure 2 — Ground-Truth Transfer Performance Matrix**",
            "Mean AUC for each (training dataset, benchmark) pair, averaged across model families. "
            "Rows and columns are ordered by hierarchical clustering on flow-space NN distances — "
            "training sets that look similar are adjacent, as are similar benchmarks. "
            "Block structure here is evidence that the IDW neighborhood assumption holds.",
        ),
        "fig3": (
            "**Figure 3 — Prediction Residuals: Predicted − Actual**",
            "Mean error per (train, benchmark) cell. Red = over-prediction, blue = under-prediction. "
            "Rows/cols use the same similarity ordering as Figure 2. "
            "Systematic row or column blocks mean the model is missing a bias for that "
            "training set or benchmark; random scatter is ideal.",
        ),
        "fig4": (
            "**Figure 4 — Model Comparison: Best Spearman per Split**",
            "Each bar shows the best Spearman across all feature groups for that model/split pair. "
            "Error bars are 95% bootstrap CIs. "
            "Baseline group (grey), Ridge family (blue), IDW pairwise (green), TP-KRR (orange).",
        ),
        "fig5": (
            "**Figure 5 — Calibration Curves (absolute-scale models)**",
            "IDW and TP-KRR predictions binned into 10 equal-count bins by predicted AUC; "
            "mean actual AUC plotted against mean predicted AUC. "
            "A perfectly calibrated model follows the dashed diagonal. "
            "Ridge/BT/PL are excluded — they produce relative rank scores that are not "
            "interpretable as absolute AUC predictions regardless of rescaling.",
        ),
        "fig5b": (
            "**Figure 5b — Calibration Spread (absolute-scale models)**",
            "Small multiples for the main absolute predictors. Predictions are binned by "
            "predicted AUC. The solid line is mean actual AUC; the darker ribbon is the "
            "25-75% range of actual AUC within the bin, and the lighter ribbon is the "
            "10-90% range. This shows whether models with similar average calibration "
            "still differ in pointwise spread.",
        ),
    }
    lines = ["## Figures\n"]
    for name in ["fig1", "fig1_train", "fig1_random", "fig2", "fig3", "fig4", "fig5", "fig5b"]:
        if name not in fig_paths:
            continue
        rel = fig_paths[name].relative_to(results_dir)
        title, caption = captions[name]
        lines.append(f"\n### {title}\n")
        lines.append(f"![{title}]({rel})\n")
        lines.append(f"\n{caption}\n")
    return "\n".join(lines) + "\n"


def fmt(val, fmt_str="{:.3f}") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    return fmt_str.format(val)


def bold_best(series: pd.Series, higher_is_better: bool = True) -> list[str]:
    """Return list of formatted strings with **bold** for best value."""
    vals = series.values.astype(float)
    valid = ~np.isnan(vals)
    if not valid.any():
        return ["—"] * len(vals)
    best_idx = np.nanargmax(vals) if higher_is_better else np.nanargmin(vals)
    out = []
    for i, v in enumerate(vals):
        s = fmt(v)
        out.append(f"**{s}**" if (i == best_idx and valid[i]) else s)
    return out


def make_markdown_table(df: pd.DataFrame, row_label: str = "") -> str:
    """Convert a dataframe to a markdown table string."""
    cols = df.columns.tolist()
    header = f"| {row_label} | " + " | ".join(str(c) for c in cols) + " |"
    sep    = f"|{'-' * (len(row_label) + 2)}|" + "|".join("-" * (len(str(c)) + 2) for c in cols) + "|"
    lines  = [header, sep]
    for idx, row in df.iterrows():
        line = f"| {idx} | " + " | ".join(str(row[c]) for c in cols) + " |"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def table_model_comparison(df: pd.DataFrame, splits: list[str], metric: str = "spearman") -> str:
    """Rows = models, columns = splits. Uses best feature group per model per split."""
    col_label, fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))
    rows = {}
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        row = {}
        for split in splits:
            s = sub[sub["split"] == split]
            if s.empty:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            col = f"{metric}_mean"
            if col not in s.columns or s[col].isna().all():
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            lower_is_better = metric in {"mae", "rmse", "rank_mae", "norm_rank_mae"}
            best_val = s[col].min() if lower_is_better else s[col].max()
            best_idx = s[col].idxmin() if lower_is_better else s[col].idxmax()
            best_fg  = s.loc[best_idx, "feature_group"]
            row[SPLIT_DISPLAY.get(split, split)] = f"{fmt(best_val, fmt_str)} *({FEATURE_DISPLAY.get(best_fg, best_fg)})*"
        rows[MODEL_DISPLAY.get(model, model)] = row

    if not rows:
        return "_No data_\n"
    result_df = pd.DataFrame(rows).T
    return make_markdown_table(result_df, "Model") + "\n"


def table_feature_ablation(df: pd.DataFrame, splits: list[str],
                            model: str = "ridge", metric: str = "spearman") -> str:
    """Rows = feature groups, columns = splits. Fixed model."""
    col_label, fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))
    col = f"{metric}_mean"
    ci_lo = f"{metric}_ci_lo"
    ci_hi = f"{metric}_ci_hi"
    sub = df[df["model"] == model]
    rows = {}
    for fg in FEATURE_ORDER:
        fsub = sub[sub["feature_group"] == fg]
        if fsub.empty:
            continue
        row = {}
        for split in splits:
            ssub = fsub[fsub["split"] == split]
            if ssub.empty or col not in ssub.columns:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            v = ssub[col].values[0]
            lo = ssub[ci_lo].values[0] if ci_lo in ssub.columns else float("nan")
            hi = ssub[ci_hi].values[0] if ci_hi in ssub.columns else float("nan")
            if not np.isnan(lo) and not np.isnan(hi):
                row[SPLIT_DISPLAY.get(split, split)] = f"{fmt(v, fmt_str)} [{fmt(lo, fmt_str)}, {fmt(hi, fmt_str)}]"
            else:
                row[SPLIT_DISPLAY.get(split, split)] = fmt(v, fmt_str)
        rows[FEATURE_DISPLAY.get(fg, fg)] = row

    if not rows:
        return "_No data_\n"
    result_df = pd.DataFrame(rows).T
    return make_markdown_table(result_df, "Feature group") + "\n"


def table_full_metrics(df: pd.DataFrame, split: str, model: str, fg: str) -> str:
    """All metrics for a given (split, model, feature_group)."""
    sub = df[(df["split"] == split) & (df["model"] == model) & (df["feature_group"] == fg)]
    if sub.empty:
        return "_No data_\n"
    row = sub.iloc[0]
    lines = []
    for metric, (label, fmt_str) in METRIC_DISPLAY.items():
        col = f"{metric}_mean"
        ci_lo, ci_hi = f"{metric}_ci_lo", f"{metric}_ci_hi"
        if col not in row.index:
            continue
        v = row[col]
        lo = row.get(ci_lo, float("nan"))
        hi = row.get(ci_hi, float("nan"))
        if not np.isnan(float(lo)) and not np.isnan(float(hi)):
            lines.append(f"| {label} | {fmt(v, fmt_str)} | [{fmt(lo, fmt_str)}, {fmt(hi, fmt_str)}] |")
        else:
            lines.append(f"| {label} | {fmt(v, fmt_str)} | — |")
    if not lines:
        return "_No metrics_\n"
    return "| Metric | Mean | 95% CI |\n|--------|------|--------|\n" + "\n".join(lines) + "\n"


def table_spearman_heatmap(df: pd.DataFrame, split: str, metric: str = "spearman",
                           higher_is_better: bool = True,
                           model_filter: list | None = None) -> str:
    """Rows = models, columns = feature groups. Marks best per row."""
    col = f"{metric}_mean"
    sub = df[df["split"] == split]
    if sub.empty or col not in sub.columns:
        return "_No data_\n"

    order = model_filter if model_filter is not None else MODEL_ORDER
    model_rows  = [m for m in order       if m in sub["model"].values]
    fg_cols     = [f for f in FEATURE_ORDER if f in sub["feature_group"].values]
    if not model_rows or not fg_cols:
        return "_No data_\n"

    col_headers = [FEATURE_DISPLAY.get(f, f) for f in fg_cols]
    header = "| Model | " + " | ".join(col_headers) + " |"
    sep    = "|-------|" + "|".join("-" * (len(h) + 2) for h in col_headers) + "|"
    lines  = [header, sep]

    fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))[1]
    for model in model_rows:
        msub = sub[sub["model"] == model]
        vals = []
        for fg in fg_cols:
            fgsub = msub[msub["feature_group"] == fg]
            if fgsub.empty or col not in fgsub.columns:
                vals.append(float("nan"))
            else:
                vals.append(float(fgsub[col].values[0]))
        best_idx = (int(np.nanargmax(vals)) if higher_is_better else int(np.nanargmin(vals))) \
                   if not all(np.isnan(vals)) else -1
        cells = []
        for j, v in enumerate(vals):
            s = fmt(v, fmt_str)
            cells.append(f"**{s}**" if j == best_idx and not np.isnan(v) else s)
        lines.append(f"| {MODEL_DISPLAY.get(model, model)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table_density_confound_check(df: pd.DataFrame, splits: list[str],
                                  models: list[str] | None = None) -> str:
    """Side-by-side: density_eval / density_train vs best flow features.

    Answers: 'Is log(N_eval) alone competitive with real flow features?'
    If density_eval ≈ flow_km, results may be driven by benchmark difficulty
    rather than distribution coverage.
    """
    if models is None:
        models = ["ridge_abs", "ridge_pairwise"]
    present_models = [m for m in models if m in df["model"].values]
    if not present_models:
        return "_No absolute models present._\n"

    # Always show idw_prior_residual alongside the others if present
    if "idw_prior_residual" not in models and "idw_prior_residual" in df["model"].values:
        models = list(models) + ["idw_prior_residual"]
        present_models = [m for m in models if m in df["model"].values]

    compare_fgs = [
        "random_idw",
        "density_idw",
        "density_eval",
        "density_train",
        "sample_count",
        "vector_density",
        "train_profile",
        "profile_density",
        "flow_mmd_only",
        "flow_fid_only",
        "flow_w2_only",
        "flow_fid_profile",
        "flow_w2_profile",
        "sym_flow",
        "flow_nn",
        "flow_eps",
        "flow_km",
        "flow_kl",
        "motion_km",
        "motion_km_profile",
    ]
    col   = "mae_mean"
    ci_lo = "mae_ci_lo"
    ci_hi = "mae_ci_hi"
    fmt_str = METRIC_DISPLAY["mae"][1]

    lines: list[str] = []
    for split in splits:
        sub = df[df["split"] == split]
        if sub.empty or col not in sub.columns:
            continue
        col_headers = [MODEL_DISPLAY.get(m, m) for m in present_models]
        header = f"#### {SPLIT_DISPLAY.get(split, split)}\n\n"
        header += "| Feature group | " + " | ".join(col_headers) + " |\n"
        header += "|---|" + "|".join("---" for _ in present_models) + "|\n"
        rows_out: list[str] = []
        for fg in compare_fgs:
            fgsub = sub[sub["feature_group"] == fg]
            if fgsub.empty:
                continue
            cells: list[str] = []
            row_vals: list[float] = []
            for model in present_models:
                msub = fgsub[fgsub["model"] == model]
                if msub.empty or msub[col].isna().all():
                    cells.append("—")
                    row_vals.append(float("nan"))
                else:
                    v  = float(msub[col].values[0])
                    lo = float(msub[ci_lo].values[0]) if ci_lo in msub.columns else float("nan")
                    hi = float(msub[ci_hi].values[0]) if ci_hi in msub.columns else float("nan")
                    row_vals.append(v)
                    if np.isfinite(lo) and np.isfinite(hi):
                        cells.append(f"{fmt(v, fmt_str)} [{fmt(lo, fmt_str)}, {fmt(hi, fmt_str)}]")
                    else:
                        cells.append(fmt(v, fmt_str))
            valid = [(i, v) for i, v in enumerate(row_vals) if np.isfinite(v)]
            if valid:
                best_i = min(valid, key=lambda x: x[1])[0]
                cells[best_i] = f"**{cells[best_i]}**"
            label = FEATURE_DISPLAY.get(fg, fg)
            separator = " ← *density baseline*" if fg in ("density_eval", "density_train", "density_idw") else ""
            if fg in ("sample_count", "vector_density", "train_profile", "profile_density"):
                separator = " ← *profile control*"
            if fg == "random_idw":
                separator = " ← *random-ID control*"
            rows_out.append(f"| {label}{separator} | " + " | ".join(cells) + " |")
        if rows_out:
            lines.append(header + "\n".join(rows_out) + "\n")

    if not lines:
        return "_No density confound data._\n"
    return (
        "*`log(N_eval)` = number of labeled keypoints in the benchmark. "
        "If this baseline matches flow features, predictions are driven by "
        "benchmark difficulty (size/density), not distribution coverage. "
        "`density_idw` uses size-based neighborhoods, while `random_idw` is a fixed "
        "random dataset/benchmark-code control. Symmetric rows test whether MMD/FID/W2 "
        "baselines explain the same signal as directed flow coverage.*\n\n"
        + "\n".join(lines)
    )


def table_idw_prior_variants(df: pd.DataFrame, splits: list[str]) -> str:
    """Compare the three two-stage IDW prior variants + ridge_pairwise as reference.

    Rows = splits, columns = model variants. Best feature group per model selected
    by lowest MAE. Reports MAE + Spearman so both dimensions are visible.
    """
    variants = ["ridge_pairwise", "ridge_pairwise_uniform", "ridge_pairwise_random",
                "ridge_pairwise_cross_resid", "ridge_pairwise_cross_resid_spline",
                "idw_prior_residual", "uniform_prior_residual", "random_prior_residual",
                "idw_prior_context", "idw_prior_context_local",
                "idw_prior_two_way", "uniform_prior_two_way", "random_prior_two_way",
                "idw_prior_two_way_spline",
                "uniform_prior_two_way_spline", "random_prior_two_way_spline"]
    present = [m for m in variants if m in df["model"].values]
    if not present:
        return "_No two-stage model results._\n"

    mae_col = "mae_mean"
    sp_col  = "spearman_mean"

    col_headers = [MODEL_DISPLAY.get(m, m) for m in present]
    header  = "| Split | " + " | ".join(col_headers) + " |\n"
    header += "|---|" + "|".join("---" for _ in present) + "|\n"

    rows = []
    for split in splits:
        cells = [SPLIT_DISPLAY.get(split, split)]
        for model in present:
            sub = df[(df["model"] == model) & (df["split"] == split)]
            if sub.empty or mae_col not in sub.columns or sub[mae_col].isna().all():
                cells.append("—")
            else:
                best = sub.loc[sub[mae_col].idxmin()]
                mae = best[mae_col]
                sp  = best.get(sp_col, float("nan"))
                fg  = best["feature_group"]
                cells.append(
                    f"MAE {fmt(mae, '{:.2f}')} / ρ {fmt(sp, '{:.3f}')} "
                    f"*({FEATURE_DISPLAY.get(fg, fg)})*"
                )
        rows.append("| " + " | ".join(cells) + " |")

    return header + "\n".join(rows) + "\n"


def table_objective_conclusions(df: pd.DataFrame, splits: list[str]) -> str:
    """Computed conclusion tables: winners, null lift, feature families, splines."""
    real_sym = ["flow_mmd_only", "flow_fid_only", "flow_w2_only"]
    real_directed = ["flow_nn", "flow_eps", "flow_km", "flow_kl", "motion", "motion_km"]
    profile_composites = [
        "flow_mmd_profile", "flow_fid_profile", "flow_w2_profile",
        "flow_kl_profile", "motion_km_profile",
    ]
    real_features = real_sym + real_directed + profile_composites
    null_features = [
        "random_idw", "density_idw", "density_train", "density_eval",
        "sample_count", "sample_count_train", "sample_count_eval",
        "vector_density", "vector_density_train", "vector_density_eval",
        "train_profile", "eval_profile", "profile_density",
    ]
    comparison_models = [
        "ridge_pairwise", "ridge_pairwise_cross_resid",
        "ridge_pairwise_cross_resid_spline",
        "idw_prior_residual", "idw_prior_two_way",
        "idw_prior_two_way_spline",
    ]
    comparison_models = [m for m in comparison_models if m in df["model"].values]

    parts: list[str] = []

    best_rows = {}
    for split in splits:
        sub = df[(df["split"] == split)
                 & (df["model"].isin(ABSOLUTE_MODELS))
                 & df["mae_mean"].notna()]
        if sub.empty:
            continue
        best_mae = sub.loc[sub["mae_mean"].idxmin()]
        sp_sub = sub[sub["spearman_mean"].notna()]
        best_sp = sp_sub.loc[sp_sub["spearman_mean"].idxmax()] if not sp_sub.empty else None
        best_rows[SPLIT_DISPLAY.get(split, split)] = {
            "Best MAE config": (
                f"{MODEL_DISPLAY.get(best_mae['model'], best_mae['model'])}; "
                f"{FEATURE_DISPLAY.get(best_mae['feature_group'], best_mae['feature_group'])}"
            ),
            "MAE": fmt(best_mae["mae_mean"], "{:.2f}"),
            "ρ": fmt(best_mae.get("spearman_mean", np.nan)),
            "Best ρ config": (
                "—" if best_sp is None else
                f"{MODEL_DISPLAY.get(best_sp['model'], best_sp['model'])}; "
                f"{FEATURE_DISPLAY.get(best_sp['feature_group'], best_sp['feature_group'])}"
            ),
            "Best ρ": "—" if best_sp is None else fmt(best_sp["spearman_mean"]),
        }
    if best_rows:
        parts.append("### Best Configurations\n\n")
        parts.append(make_markdown_table(pd.DataFrame(best_rows).T, "Split") + "\n\n")

    lift_detail = []
    for split in splits:
        for model in comparison_models:
            sub = df[(df["split"] == split) & (df["model"] == model) & df["mae_mean"].notna()]
            real = sub[sub["feature_group"].isin(real_features)]
            null = sub[sub["feature_group"].isin(null_features)]
            if real.empty or null.empty:
                continue
            best_real = real.loc[real["mae_mean"].idxmin()]
            best_null = null.loc[null["mae_mean"].idxmin()]
            lift_detail.append({
                "split": split,
                "model": model,
                "best_real": best_real["feature_group"],
                "best_null": best_null["feature_group"],
                "lift": float(best_null["mae_mean"] - best_real["mae_mean"]),
            })
    lift_df = pd.DataFrame(lift_detail)
    if not lift_df.empty:
        rows = []
        for split, grp in lift_df.groupby("split", sort=False):
            best = grp.loc[grp["lift"].idxmax()]
            rows.append({
                "Split": SPLIT_DISPLAY.get(split, split),
                "Median lift": fmt(grp["lift"].median(), "{:.2f}"),
                "Mean lift": fmt(grp["lift"].mean(), "{:.2f}"),
                "Positive / total": f"{int((grp['lift'] > 0).sum())}/{len(grp)}",
                "Largest lift": (
                    f"{fmt(best['lift'], '{:.2f}')} from "
                    f"{MODEL_DISPLAY.get(best['model'], best['model'])}; "
                    f"{FEATURE_DISPLAY.get(best['best_real'], best['best_real'])} "
                    f"vs {FEATURE_DISPLAY.get(best['best_null'], best['best_null'])}"
                ),
            })
        parts.append("### Feature Lift Over Null Controls\n\n")
        parts.append(
            "*Lift = best density/random-control MAE minus best real-feature MAE "
            "within the same split and model. Positive values mean the distributional "
            "feature improved over panel/density borrowing controls.*\n\n"
        )
        parts.append(make_markdown_table(pd.DataFrame(rows).set_index("Split"), "Split") + "\n\n")

    sym_detail = []
    for split in splits:
        for model in comparison_models:
            sub = df[(df["split"] == split) & (df["model"] == model) & df["mae_mean"].notna()]
            sym = sub[sub["feature_group"].isin(real_sym)]
            direct = sub[sub["feature_group"].isin(real_directed)]
            if sym.empty or direct.empty:
                continue
            best_sym = sym.loc[sym["mae_mean"].idxmin()]
            best_dir = direct.loc[direct["mae_mean"].idxmin()]
            sym_detail.append({
                "split": split,
                "sym_adv": float(best_dir["mae_mean"] - best_sym["mae_mean"]),
                "best_sym": best_sym["feature_group"],
                "best_dir": best_dir["feature_group"],
            })
    sym_df = pd.DataFrame(sym_detail)
    if not sym_df.empty:
        rows = []
        for split, grp in sym_df.groupby("split", sort=False):
            best = grp.loc[grp["sym_adv"].idxmax()]
            rows.append({
                "Split": SPLIT_DISPLAY.get(split, split),
                "Median sym advantage": fmt(grp["sym_adv"].median(), "{:.2f}"),
                "Symmetric wins": f"{int((grp['sym_adv'] > 0).sum())}/{len(grp)}",
                "Strongest symmetric feature": FEATURE_DISPLAY.get(best["best_sym"], best["best_sym"]),
                "Against directed": FEATURE_DISPLAY.get(best["best_dir"], best["best_dir"]),
            })
        parts.append("### Symmetric vs Directed Features\n\n")
        parts.append(
            "*Symmetric advantage = best directed-feature MAE minus best symmetric-feature MAE. "
            "Positive means MMD/FID/SW2 beat directed coverage/NN/KL within the same model.*\n\n"
        )
        parts.append(make_markdown_table(pd.DataFrame(rows).set_index("Split"), "Split") + "\n\n")

    wins = []
    for split in splits:
        for model in comparison_models:
            sub = df[(df["split"] == split)
                     & (df["model"] == model)
                     & (df["feature_group"].isin(real_features))
                     & df["mae_mean"].notna()]
            if sub.empty:
                continue
            wins.append(sub.loc[sub["mae_mean"].idxmin(), "feature_group"])
    if wins:
        win_counts = (pd.Series(wins).value_counts()
                      .rename_axis("Feature group")
                      .reset_index(name="Wins"))
        win_counts["Feature group"] = win_counts["Feature group"].map(
            lambda f: FEATURE_DISPLAY.get(f, f)
        )
        parts.append("### Real-Feature Win Counts\n\n")
        parts.append(
            "*Counts how often each real distributional feature group is the lowest-MAE "
            "real feature across the selected model/split comparisons.*\n\n"
        )
        parts.append(make_markdown_table(
            win_counts.set_index("Feature group"), "Feature group"
        ) + "\n\n")

    spline_pairs = [
        ("ridge_pairwise_cross_resid", "ridge_pairwise_cross_resid_spline"),
        ("idw_prior_two_way", "idw_prior_two_way_spline"),
        ("uniform_prior_two_way", "uniform_prior_two_way_spline"),
        ("random_prior_two_way", "random_prior_two_way_spline"),
    ]
    rows = []
    for split in splits:
        for linear, spline in spline_pairs:
            lin = df[(df["split"] == split) & (df["model"] == linear) & df["mae_mean"].notna()]
            spl = df[(df["split"] == split) & (df["model"] == spline) & df["mae_mean"].notna()]
            if lin.empty or spl.empty:
                continue
            b_lin = lin.loc[lin["mae_mean"].idxmin()]
            b_spl = spl.loc[spl["mae_mean"].idxmin()]
            delta = float(b_lin["mae_mean"] - b_spl["mae_mean"])
            rows.append({
                "Split": SPLIT_DISPLAY.get(split, split),
                "Model family": MODEL_DISPLAY.get(spline, spline),
                "Linear MAE": fmt(b_lin["mae_mean"], "{:.2f}"),
                "Spline MAE": fmt(b_spl["mae_mean"], "{:.2f}"),
                "Spline gain": f"{delta:+.2f}",
                "Spline feature": FEATURE_DISPLAY.get(b_spl["feature_group"], b_spl["feature_group"]),
            })
    if rows:
        parts.append("### Spline Residual Check\n\n")
        parts.append(
            "*Spline gain = best linear-counterpart MAE minus best spline MAE. "
            "Positive means the nonlinear residual stage helped.*\n\n"
        )
        spline_df = pd.DataFrame(rows)
        spline_df.insert(0, "Comparison", spline_df["Split"] + " — " + spline_df["Model family"])
        spline_df = spline_df.drop(columns=["Split", "Model family"]).set_index("Comparison")
        parts.append(make_markdown_table(spline_df, "Comparison") + "\n\n")

    if not parts:
        return "_No objective conclusion summary available for the completed runs._\n"
    return "".join(parts)


def table_absolute_performance(df: pd.DataFrame, splits: list[str]) -> str:
    """Rows = absolute-scale models, columns = splits. Best feature group by MAE, showing MAE + Spearman."""
    mae_col = "mae_mean"
    sp_col  = "spearman_mean"
    abs_order = [m for m in MODEL_ORDER if m in ABSOLUTE_MODELS]
    rows = {}
    for model in abs_order:
        sub = df[df["model"] == model]
        if sub.empty or mae_col not in sub.columns:
            continue
        row = {}
        for split in splits:
            s = sub[(sub["split"] == split) & sub[mae_col].notna()]
            if s.empty:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            best = s.loc[s[mae_col].idxmin()]
            mae  = best[mae_col]
            sp   = best.get(sp_col, float("nan"))
            fg   = best["feature_group"]
            row[SPLIT_DISPLAY.get(split, split)] = (
                f"MAE {fmt(mae, '{:.2f}')} / ρ {fmt(sp, '{:.3f}')} "
                f"*({FEATURE_DISPLAY.get(fg, fg)})*"
            )
        rows[MODEL_DISPLAY.get(model, model)] = row

    if not rows:
        return "_No absolute-scale model data_\n"
    return make_markdown_table(pd.DataFrame(rows).T, "Model") + "\n"


def table_feature_ablation_paired(df: pd.DataFrame, split: str,
                                   models: list[str],
                                   metric: str = "mae") -> str:
    """Rows = feature groups, columns = models. Bold = best per row.

    Compares multiple models (e.g. ridge_abs vs ridge_pairwise) on the same
    split and metric. Best value per row is bolded.
    """
    _, fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))
    col   = f"{metric}_mean"
    ci_lo = f"{metric}_ci_lo"
    ci_hi = f"{metric}_ci_hi"
    lower_is_better = metric in {"mae", "rmse", "rank_mae", "norm_rank_mae"}

    sub = df[df["split"] == split]

    col_headers = [MODEL_DISPLAY.get(m, m) for m in models]
    header = "| Feature group | " + " | ".join(col_headers) + " |"
    sep    = "|---|" + "|".join("-" * (len(h) + 2) for h in col_headers) + "|"
    lines  = [header, sep]

    found_any = False
    for fg in FEATURE_ORDER:
        row_vals: list[float] = []
        row_strs: list[str]   = []
        for model in models:
            fsub = sub[(sub["model"] == model) & (sub["feature_group"] == fg)]
            if fsub.empty or col not in fsub.columns or fsub[col].isna().all():
                row_vals.append(float("nan"))
                row_strs.append("—")
            else:
                v  = float(fsub[col].values[0])
                lo = float(fsub[ci_lo].values[0]) if ci_lo in fsub.columns else float("nan")
                hi = float(fsub[ci_hi].values[0]) if ci_hi in fsub.columns else float("nan")
                row_vals.append(v)
                if np.isfinite(lo) and np.isfinite(hi):
                    row_strs.append(f"{fmt(v, fmt_str)} [{fmt(lo, fmt_str)}, {fmt(hi, fmt_str)}]")
                else:
                    row_strs.append(fmt(v, fmt_str))

        if all(np.isnan(v) for v in row_vals):
            continue
        found_any = True

        valid = [(i, v) for i, v in enumerate(row_vals) if np.isfinite(v)]
        if valid:
            best_i = min(valid, key=lambda x: x[1])[0] if lower_is_better \
                     else max(valid, key=lambda x: x[1])[0]
            row_strs[best_i] = f"**{row_strs[best_i]}**"

        lines.append(f"| {FEATURE_DISPLAY.get(fg, fg)} | " + " | ".join(row_strs) + " |")

    if not found_any:
        return "_No data_\n"
    return "\n".join(lines) + "\n"


def is_diagnostic_report(models: list[str], feature_groups: list[str]) -> bool:
    """True for the narrowed density-vs-geometry sweep."""
    model_set = set(models)
    allowed_models = set(DIAGNOSTIC_MODELS) | set(DIAGNOSTIC_BASELINES)
    required_models = {"ridge_abs", "ridge_pairwise", "idw_prior_two_way"}
    has_required_features = {"train_profile_simple", "profile_simple", "motion_km_profile"}.issubset(
        set(feature_groups)
    )
    return (
        required_models.issubset(model_set)
        and model_set.issubset(allowed_models)
        and has_required_features
    )


def diagnostic_feature_legend(feature_groups: list[str]) -> str:
    present = set(feature_groups)
    lines = []
    for family, fgroups in DIAGNOSTIC_FEATURE_FAMILIES.items():
        visible = [f for f in fgroups if f in present]
        if not visible:
            continue
        labels = ", ".join(f"`{f}`" for f in visible)
        lines.append(f"- **{family}:** {labels}")
    return "\n".join(lines) + "\n"


def _mae_rho_cell(row: pd.Series | None) -> str:
    if row is None or "mae_mean" not in row or pd.isna(row["mae_mean"]):
        return "—"
    return (
        f"MAE {fmt(row['mae_mean'], '{:.2f}')} / "
        f"ρ {fmt(row.get('spearman_mean', np.nan))}"
    )


def _single_result(df: pd.DataFrame, split: str, model: str, feature_group: str) -> pd.Series | None:
    sub = df[
        (df["split"] == split)
        & (df["model"] == model)
        & (df["feature_group"] == feature_group)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def _best_row(
    df: pd.DataFrame,
    split: str,
    model: str,
    feature_groups: list[str],
    metric: str = "mae",
) -> pd.Series | None:
    col = f"{metric}_mean"
    sub = df[
        (df["split"] == split)
        & (df["model"] == model)
        & (df["feature_group"].isin(feature_groups))
    ]
    if sub.empty or col not in sub.columns or sub[col].isna().all():
        return None
    lower_is_better = metric in {"mae", "rmse", "rank_mae", "norm_rank_mae"}
    idx = sub[col].idxmin() if lower_is_better else sub[col].idxmax()
    return sub.loc[idx]


def _signed_delta(before: float, after: float, lower_is_better: bool) -> str:
    if not np.isfinite(before) or not np.isfinite(after):
        return "—"
    delta = before - after if lower_is_better else after - before
    return f"{delta:+.2f}" if lower_is_better else f"{delta:+.3f}"


def table_diagnostic_model_axis(df: pd.DataFrame, splits: list[str]) -> str:
    """Rows = models, columns = splits. Each cell picks that model's best feature."""
    rows = {}
    for model in [m for m in DIAGNOSTIC_MODELS if m in df["model"].values]:
        row = {}
        for split in splits:
            sub = df[
                (df["split"] == split)
                & (df["model"] == model)
                & df["mae_mean"].notna()
            ]
            if sub.empty:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            best = sub.loc[sub["mae_mean"].idxmin()]
            row[SPLIT_DISPLAY.get(split, split)] = (
                f"{_mae_rho_cell(best)} "
                f"*({FEATURE_DISPLAY.get(best['feature_group'], best['feature_group'])})*"
            )
        rows[MODEL_DISPLAY.get(model, model)] = row
    return make_markdown_table(pd.DataFrame(rows).T, "Model") + "\n" if rows else "_No model-axis data._\n"


def table_diagnostic_control_summary(
    df: pd.DataFrame,
    splits: list[str],
    model: str = "idw_prior_two_way",
) -> str:
    """Best controls vs best flow groups for the primary model, split by split."""
    structure_controls = ["random_idw"]
    density_controls = [
        "density_train", "density_eval", "density_idw",
        "sample_count", "vector_density_simple", "train_profile_simple", "profile_simple",
    ]
    flow_only = ["flow_fid_only", "flow_w2_only", "flow_kl", "motion_km"]
    flow_profile = ["flow_fid_profile", "flow_w2_profile", "flow_kl_profile", "motion_km_profile"]

    rows = {}
    for split in splits:
        struct = _best_row(df, split, model, structure_controls, metric="mae")
        ctrl = _best_row(df, split, model, density_controls, metric="mae")
        flow = _best_row(df, split, model, flow_only, metric="mae")
        combo = _best_row(df, split, model, flow_profile, metric="mae")
        ctrl_mae = float(ctrl["mae_mean"]) if ctrl is not None and pd.notna(ctrl.get("mae_mean")) else np.nan
        flow_mae = float(flow["mae_mean"]) if flow is not None and pd.notna(flow.get("mae_mean")) else np.nan
        combo_mae = float(combo["mae_mean"]) if combo is not None and pd.notna(combo.get("mae_mean")) else np.nan
        struct_mae = float(struct["mae_mean"]) if struct is not None and pd.notna(struct.get("mae_mean")) else np.nan
        rows[SPLIT_DISPLAY.get(split, split)] = {
            "Structure null": (
                "—" if struct is None else
                f"{_mae_rho_cell(struct)} *({FEATURE_DISPLAY.get(struct['feature_group'], struct['feature_group'])})*"
            ),
            "Best density/profile control": (
                "—" if ctrl is None else
                f"{_mae_rho_cell(ctrl)} *({FEATURE_DISPLAY.get(ctrl['feature_group'], ctrl['feature_group'])})*"
            ),
            "Best flow-only": (
                "—" if flow is None else
                f"{_mae_rho_cell(flow)} *({FEATURE_DISPLAY.get(flow['feature_group'], flow['feature_group'])})*"
            ),
            "Best flow + profile": (
                "—" if combo is None else
                f"{_mae_rho_cell(combo)} *({FEATURE_DISPLAY.get(combo['feature_group'], combo['feature_group'])})*"
            ),
            "Flow gain vs density/profile": _signed_delta(ctrl_mae, flow_mae, lower_is_better=True),
            "Flow+profile gain vs density/profile": _signed_delta(ctrl_mae, combo_mae, lower_is_better=True),
            "Flow gain vs structure": _signed_delta(struct_mae, flow_mae, lower_is_better=True),
        }
    return make_markdown_table(pd.DataFrame(rows).T, "Split") + "\n" if rows else "_No control summary data._\n"


def table_feature_axis_for_model(
    df: pd.DataFrame,
    splits: list[str],
    model: str = "idw_prior_two_way",
) -> str:
    """Rows = features, columns = splits for one primary model."""
    rows = {}
    ordered_fgroups = [f for f in DIAGNOSTIC_FEATURES if f in df["feature_group"].values]
    for fg in ordered_fgroups:
        row = {}
        for split in splits:
            res = _single_result(df, split, model, fg)
            row[SPLIT_DISPLAY.get(split, split)] = _mae_rho_cell(res)
        rows[FEATURE_DISPLAY.get(fg, fg)] = row
    if not rows:
        return "_No feature-axis data._\n"

    out = pd.DataFrame(rows).T
    # Bold the lowest MAE in each split column.
    for split in splits:
        col = SPLIT_DISPLAY.get(split, split)
        vals = []
        for fg in ordered_fgroups:
            res = _single_result(df, split, model, fg)
            vals.append(float(res["mae_mean"]) if res is not None and pd.notna(res.get("mae_mean")) else np.nan)
        if not np.isfinite(vals).any():
            continue
        best_i = int(np.nanargmin(vals))
        out.iloc[best_i, out.columns.get_loc(col)] = f"**{out.iloc[best_i, out.columns.get_loc(col)]}**"
    return make_markdown_table(out, "Feature group") + "\n"


def table_feature_axis_metric_for_model(
    df: pd.DataFrame,
    splits: list[str],
    model: str = "idw_prior_two_way",
    metric: str = "spearman",
) -> str:
    """Rows = features, columns = splits for one metric. Used for explicit Spearman checks."""
    _, fmt_str = METRIC_DISPLAY.get(metric, (metric, "{:.3f}"))
    col_name = f"{metric}_mean"
    lower_is_better = metric in {"mae", "rmse", "rank_mae", "norm_rank_mae"}
    rows = {}
    ordered_fgroups = [f for f in DIAGNOSTIC_FEATURES if f in df["feature_group"].values]
    for fg in ordered_fgroups:
        row = {}
        for split in splits:
            res = _single_result(df, split, model, fg)
            if res is None or col_name not in res or pd.isna(res[col_name]):
                row[SPLIT_DISPLAY.get(split, split)] = "—"
            else:
                row[SPLIT_DISPLAY.get(split, split)] = fmt(float(res[col_name]), fmt_str)
        rows[FEATURE_DISPLAY.get(fg, fg)] = row
    if not rows:
        return "_No metric data._\n"

    out = pd.DataFrame(rows).T
    for split in splits:
        col = SPLIT_DISPLAY.get(split, split)
        vals = []
        for fg in ordered_fgroups:
            res = _single_result(df, split, model, fg)
            vals.append(float(res[col_name]) if res is not None and col_name in res and pd.notna(res[col_name]) else np.nan)
        if not np.isfinite(vals).any():
            continue
        best_i = int(np.nanargmin(vals) if lower_is_better else np.nanargmax(vals))
        out.iloc[best_i, out.columns.get_loc(col)] = f"**{out.iloc[best_i, out.columns.get_loc(col)]}**"
    return make_markdown_table(out, "Feature group") + "\n"


def table_flow_profile_lift(
    df: pd.DataFrame,
    splits: list[str],
    model: str = "idw_prior_two_way",
) -> str:
    """Does each flow feature improve over train_profile_simple once profile is included?"""
    pairs = [
        ("flow_fid_only", "flow_fid_profile"),
        ("flow_w2_only", "flow_w2_profile"),
        ("flow_kl", "flow_kl_profile"),
        ("motion_km", "motion_km_profile"),
    ]
    rows = []
    for split in splits:
        profile = _single_result(df, split, model, "train_profile_simple")
        profile_mae = float(profile["mae_mean"]) if profile is not None and pd.notna(profile.get("mae_mean")) else np.nan
        for flow_only, flow_profile in pairs:
            base = _single_result(df, split, model, flow_only)
            combo = _single_result(df, split, model, flow_profile)
            base_mae = float(base["mae_mean"]) if base is not None and pd.notna(base.get("mae_mean")) else np.nan
            combo_mae = float(combo["mae_mean"]) if combo is not None and pd.notna(combo.get("mae_mean")) else np.nan
            lift = profile_mae - combo_mae if np.isfinite(profile_mae) and np.isfinite(combo_mae) else np.nan
            rows.append({
                "Split / feature": f"{SPLIT_DISPLAY.get(split, split)} — {FEATURE_DISPLAY.get(flow_only, flow_only)}",
                "Train profile MAE": fmt(profile_mae, "{:.2f}"),
                "Flow-only MAE": fmt(base_mae, "{:.2f}"),
                "Flow + profile MAE": fmt(combo_mae, "{:.2f}"),
                "Gain vs profile": f"{lift:+.2f}" if np.isfinite(lift) else "—",
            })
    if not rows:
        return "_No flow/profile lift data._\n"
    return make_markdown_table(pd.DataFrame(rows).set_index("Split / feature"), "Split / feature") + "\n"


def table_flow_profile_lift_spearman(
    df: pd.DataFrame,
    splits: list[str],
    model: str = "idw_prior_two_way",
) -> str:
    """Spearman counterpart to flow/profile MAE lift."""
    pairs = [
        ("flow_fid_only", "flow_fid_profile"),
        ("flow_w2_only", "flow_w2_profile"),
        ("flow_kl", "flow_kl_profile"),
        ("motion_km", "motion_km_profile"),
    ]
    rows = []
    for split in splits:
        profile = _single_result(df, split, model, "train_profile_simple")
        profile_sp = float(profile["spearman_mean"]) if profile is not None and pd.notna(profile.get("spearman_mean")) else np.nan
        for flow_only, flow_profile in pairs:
            base = _single_result(df, split, model, flow_only)
            combo = _single_result(df, split, model, flow_profile)
            base_sp = float(base["spearman_mean"]) if base is not None and pd.notna(base.get("spearman_mean")) else np.nan
            combo_sp = float(combo["spearman_mean"]) if combo is not None and pd.notna(combo.get("spearman_mean")) else np.nan
            rows.append({
                "Split / feature": f"{SPLIT_DISPLAY.get(split, split)} — {FEATURE_DISPLAY.get(flow_only, flow_only)}",
                "Train profile ρ": fmt(profile_sp),
                "Flow-only ρ": fmt(base_sp),
                "Flow + profile ρ": fmt(combo_sp),
                "ρ gain vs profile": _signed_delta(profile_sp, combo_sp, lower_is_better=False),
            })
    if not rows:
        return "_No flow/profile Spearman data._\n"
    return make_markdown_table(pd.DataFrame(rows).set_index("Split / feature"), "Split / feature") + "\n"


def table_axis_aware_nulls(df: pd.DataFrame, splits: list[str]) -> str:
    """Compare axis-aware IDW geometry against uniform/random panel-borrowing nulls."""
    models = [m for m in ["idw_prior_two_way", "uniform_prior_two_way", "random_prior_two_way"]
              if m in df["model"].values]
    if len(models) < 2:
        return "_No axis-aware null comparison data._\n"
    parts = []
    for split in splits:
        parts.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
        rows = {}
        for fg in [f for f in DIAGNOSTIC_FEATURES if f in df["feature_group"].values]:
            row = {}
            idw_mae = np.nan
            null_maes = []
            for model in models:
                res = _single_result(df, split, model, fg)
                row[MODEL_DISPLAY.get(model, model)] = _mae_rho_cell(res)
                if res is not None and pd.notna(res.get("mae_mean")):
                    if model == "idw_prior_two_way":
                        idw_mae = float(res["mae_mean"])
                    else:
                        null_maes.append(float(res["mae_mean"]))
            best_null = min(null_maes) if null_maes else np.nan
            gain = best_null - idw_mae if np.isfinite(best_null) and np.isfinite(idw_mae) else np.nan
            row["IDW gain vs best null"] = f"{gain:+.2f}" if np.isfinite(gain) else "—"
            rows[FEATURE_DISPLAY.get(fg, fg)] = row
        parts.append(make_markdown_table(pd.DataFrame(rows).T, "Feature group"))
        parts.append("\n\n")
    return "".join(parts)


def table_axis_aware_null_summary(df: pd.DataFrame, splits: list[str]) -> str:
    """Best axis-aware geometry vs best uniform/random null, for MAE and Spearman."""
    null_models = ["uniform_prior_two_way", "random_prior_two_way"]
    rows = {}
    for split in splits:
        idw_mae = _best_row(df, split, "idw_prior_two_way", DIAGNOSTIC_FEATURES, metric="mae")
        null_mae_candidates = [
            _best_row(df, split, m, DIAGNOSTIC_FEATURES, metric="mae") for m in null_models
        ]
        null_mae_candidates = [r for r in null_mae_candidates if r is not None]
        best_null_mae = (
            min(null_mae_candidates, key=lambda r: float(r["mae_mean"]))
            if null_mae_candidates else None
        )
        idw_sp = _best_row(df, split, "idw_prior_two_way", DIAGNOSTIC_FEATURES, metric="spearman")
        null_sp_candidates = [
            _best_row(df, split, m, DIAGNOSTIC_FEATURES, metric="spearman") for m in null_models
        ]
        null_sp_candidates = [r for r in null_sp_candidates if r is not None]
        best_null_sp = (
            max(null_sp_candidates, key=lambda r: float(r["spearman_mean"]))
            if null_sp_candidates else None
        )
        idw_mae_val = float(idw_mae["mae_mean"]) if idw_mae is not None and pd.notna(idw_mae.get("mae_mean")) else np.nan
        null_mae_val = float(best_null_mae["mae_mean"]) if best_null_mae is not None and pd.notna(best_null_mae.get("mae_mean")) else np.nan
        idw_sp_val = float(idw_sp["spearman_mean"]) if idw_sp is not None and pd.notna(idw_sp.get("spearman_mean")) else np.nan
        null_sp_val = float(best_null_sp["spearman_mean"]) if best_null_sp is not None and pd.notna(best_null_sp.get("spearman_mean")) else np.nan
        rows[SPLIT_DISPLAY.get(split, split)] = {
            "Best geometry MAE": (
                "—" if idw_mae is None else
                f"{_mae_rho_cell(idw_mae)} *({FEATURE_DISPLAY.get(idw_mae['feature_group'], idw_mae['feature_group'])})*"
            ),
            "Best null MAE": (
                "—" if best_null_mae is None else
                f"{_mae_rho_cell(best_null_mae)} "
                f"*({MODEL_DISPLAY.get(best_null_mae['model'], best_null_mae['model'])}; "
                f"{FEATURE_DISPLAY.get(best_null_mae['feature_group'], best_null_mae['feature_group'])})*"
            ),
            "MAE gain": _signed_delta(null_mae_val, idw_mae_val, lower_is_better=True),
            "Best geometry ρ": (
                "—" if idw_sp is None else
                f"ρ {fmt(idw_sp_val)} *({FEATURE_DISPLAY.get(idw_sp['feature_group'], idw_sp['feature_group'])})*"
            ),
            "Best null ρ": (
                "—" if best_null_sp is None else
                f"ρ {fmt(null_sp_val)} "
                f"*({MODEL_DISPLAY.get(best_null_sp['model'], best_null_sp['model'])}; "
                f"{FEATURE_DISPLAY.get(best_null_sp['feature_group'], best_null_sp['feature_group'])})*"
            ),
            "ρ gain": _signed_delta(null_sp_val, idw_sp_val, lower_is_better=False),
        }
    return make_markdown_table(pd.DataFrame(rows).T, "Split") + "\n" if rows else "_No null summary data._\n"


def table_diagnostic_baselines(df: pd.DataFrame, splits: list[str]) -> str:
    rows = {}
    for model in [m for m in DIAGNOSTIC_BASELINES if m in df["model"].values]:
        row = {}
        for split in splits:
            sub = df[(df["split"] == split) & (df["model"] == model)]
            if sub.empty:
                row[SPLIT_DISPLAY.get(split, split)] = "—"
                continue
            res = sub.iloc[0]
            row[SPLIT_DISPLAY.get(split, split)] = (
                f"ρ {fmt(res.get('spearman_mean', np.nan))}"
                if model == "random"
                else _mae_rho_cell(res)
            )
        rows[MODEL_DISPLAY.get(model, model)] = row
    return make_markdown_table(pd.DataFrame(rows).T, "Baseline") + "\n" if rows else "_No baselines._\n"


def _pretty(name: str) -> str:
    """Human-readable feature name for tables."""
    name = name.replace("eval_covered_by_train", "ε-cov E→T")
    name = name.replace("train_covered_by_eval", "ε-cov T→E")
    name = name.replace("mean_nn_eval_to_train", "NN-dist E→T")
    name = name.replace("mean_nn_train_to_eval", "NN-dist T→E")
    name = name.replace("_weighted", " (km)")
    name = name.replace("flow_km_", "flow_km/")
    name = name.replace("flow_", "flow/")
    name = name.replace("dino_", "dino/")
    name = name.replace("log_eval_n_vectors",  "density/log_eval_N")
    name = name.replace("log_train_n_vectors", "density/log_train_N")
    name = name.replace("log_eval_n_samples", "profile/log_eval_samples")
    name = name.replace("log_train_n_samples", "profile/log_train_samples")
    name = name.replace("log_eval_valid_vectors_per_sample", "profile/log_eval_vecs_per_img")
    name = name.replace("log_train_valid_vectors_per_sample", "profile/log_train_vecs_per_img")
    name = name.replace("log_eval_sampled_vectors_per_sample", "profile/log_eval_sampled_per_img")
    name = name.replace("log_train_sampled_vectors_per_sample", "profile/log_train_sampled_per_img")
    name = name.replace("log_eval_retained_vectors_per_sample", "profile/log_eval_retained_per_img")
    name = name.replace("log_train_retained_vectors_per_sample", "profile/log_train_retained_per_img")
    name = name.replace("zero_image_frac", "zero-image frac")
    name = name.replace("_null99", " (null p99)")
    name = name.replace("_null95", " (null p95)")
    name = name.replace("_null90", " (null p90)")
    name = name.replace("_null80", " (null p80)")
    name = name.replace("_qnorm_k1", " (qnorm)")
    name = name.replace("kl_eval_to_train", "KL E→T")
    name = name.replace("kl_train_to_eval", "KL T→E")
    name = name.replace("_k5",  " k=5")
    name = name.replace("_k20", " k=20")
    name = name.replace("_eps", " ε")
    return name.replace("_", " ").strip()


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_summary(results_dir: Path) -> pd.DataFrame | None:
    p = results_dir / "summary_table.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_metrics_from_dirs(results_dir: Path) -> pd.DataFrame:
    """Fallback: scan individual metrics.csv files if summary_table.csv missing."""
    rows = []
    for csv_path in results_dir.glob("*/*/*/metrics.csv"):
        parts = csv_path.parts
        # results_dir / split / model / feature_group / metrics.csv
        fg    = parts[-2]
        model = parts[-3]
        split = parts[-4]
        df = pd.read_csv(csv_path)
        agg = {
            "split": split, "model": model, "feature_group": fg,
            "n_contexts": len(df),
        }
        for metric in METRIC_DISPLAY:
            if metric in df.columns:
                agg[f"{metric}_mean"]   = df[metric].mean()
                agg[f"{metric}_median"] = df[metric].median()
        rows.append(agg)
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir",
        default="scripts/transfer_analysis_v3/results")
    parser.add_argument("--output",
        default="scripts/transfer_analysis_v3/results/results.md")
    parser.add_argument("--mi-csv", default=None,
        help="Path to feature_mi.csv from compute_feature_mi.py")
    parser.add_argument("--self-dist-csv", default="analysis_v3/pairwise_self_distances.csv",
        help="Pairwise self-distances CSV used for clustering rows/cols in Figure 2.")
    parser.add_argument("--no-figures", action="store_true",
        help="Skip figure generation.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}...")
    df = load_summary(results_dir)
    if df is None or df.empty:
        print("summary_table.csv not found — scanning individual metrics.csv files...")
        df = load_metrics_from_dirs(results_dir)
    if df is None or df.empty:
        print("No results found. Run run_experiments.py first.")
        return

    available_splits  = [s for s in SPLIT_ORDER  if s in df["split"].values]
    available_models  = [m for m in MODEL_ORDER   if m in df["model"].values]
    available_fgroups = [f for f in FEATURE_ORDER if f in df["feature_group"].values]

    print(f"  {len(df)} rows | splits: {available_splits} | models: {available_models}")

    # Generate figures before building sections so we can embed them
    fig_paths: dict = {}
    if not args.no_figures:
        dist_csv = Path(args.self_dist_csv)
        fig_paths = generate_figures(
            results_dir, df, available_models, available_splits,
            dist_csv=dist_csv if dist_csv.exists() else None,
        )

    # -----------------------------------------------------------------------
    # Build results.md
    # -----------------------------------------------------------------------
    sections = []

    diagnostic_report = is_diagnostic_report(available_models, available_fgroups)
    if diagnostic_report:
        legend_lines = diagnostic_feature_legend(available_fgroups)
    else:
        legend_lines = "\n".join(
            f"- **{FEATURE_DISPLAY.get(f, f)}**: {FEATURE_LEGEND.get(f, '')}"
            for f in available_fgroups if f in FEATURE_LEGEND
        )
    sections.append(textwrap.dedent(f"""\
        # Transfer Estimator Results

        Generated from `{results_dir}`

        **Target metric:** `{args.mi_csv.split("feature_mi_")[1].split("/")[0] if args.mi_csv and "feature_mi_" in args.mi_csv else results_dir.name}`
        **Splits evaluated:** {', '.join(SPLIT_DISPLAY.get(s, s) for s in available_splits)}
        **Models:** {', '.join(MODEL_DISPLAY.get(m, m) for m in available_models)}

        **Primary metric: MAE ↓** (mean |predicted − actual| AUC, in absolute PCK units).
        Absolute-scale models (Ridge+IDW, TP-KRR) predict actual AUC values; MAE is their
        primary metric. Spearman ρ is reported as a secondary metric. Ridge (rank), BT, and PL
        produce relative rank scores — Spearman is their primary metric, MAE is not meaningful.
        95% CI bootstrapped over held-out entities (train datasets for LOTO, benchmarks for LOBO, etc.).

        **Global Prior / LOTO note:** Global Prior Spearman is undefined for LOTO/LOTO-grouped
        because each test fold contains exactly one training dataset (can't rank one item).

        ### Feature Group Legend

    """) + legend_lines + "\n\n")

    if fig_paths:
        sections.append(figures_md(fig_paths, results_dir))

    if diagnostic_report:
        sections.append("## 0. Report Map\n\n")
        sections.append(
            "*This diagnostic report is organized around two experimental axes: "
            "**model axis** = which predictor/borrowing mechanism works best, and "
            "**feature axis** = which distributional/profile signal works best. "
            "Every key comparison is shown split-by-split rather than pooled, because pooled "
            "averages can hide Simpson-paradox structure across benchmarks, training datasets, "
            "and model variants. The interaction table is kept as a compact audit trail rather "
            "than the main story.*\n\n"
        )

        sections.append("## 1. Confound Summary — Is It Structure, Density, or Flow?\n\n")
        sections.append(
            "*Fixed model: `idw_prior_two_way`. This is the highest-level readout for the "
            "current question. `Structure null` uses the random-ID control. "
            "`Best density/profile control` captures dataset size, sample count, and "
            "capped supervision density. `Best flow-only` asks whether motion geometry works by "
            "itself. `Best flow + profile` asks whether flow still helps after the train "
            "profile is included. Positive gains mean lower MAE than the control.*\n\n"
        )
        sections.append(table_diagnostic_control_summary(df, available_splits, model="idw_prior_two_way"))
        sections.append("\n")

        sections.append("## 2. Model Axis — Which Predictor Wins?\n\n")
        sections.append(
            "*Each cell selects the best feature group for that model and split by lowest MAE. "
            "Use this to compare plain feature regression, coupled IDW, and the axis-aware "
            "prior family. The cell includes Spearman so calibration quality and rank "
            "agreement are visible together.*\n\n"
        )
        sections.append(table_diagnostic_model_axis(df, available_splits))
        sections.append("\n")

        sections.append("## 3. Feature Axis — MAE for the Axis-Aware Model\n\n")
        sections.append(
            "*Fixed model: `idw_prior_two_way`. Rows are feature groups; columns are held-out "
            "splits. Bold marks the lowest MAE per split. This isolates the feature question "
            "from the model question.*\n\n"
        )
        sections.append(table_feature_axis_for_model(df, available_splits, model="idw_prior_two_way"))
        sections.append("\n")

        sections.append("## 4. Feature Axis — Spearman for the Axis-Aware Model\n\n")
        sections.append(
            "*Same fixed model and feature groups, but ranked by Spearman. This catches cases "
            "where a feature calibrates absolute PCK well but does not order datasets well, "
            "or vice versa.*\n\n"
        )
        sections.append(table_feature_axis_metric_for_model(
            df, available_splits, model="idw_prior_two_way", metric="spearman"
        ))
        sections.append("\n")

        sections.append("### Flow After Train Profile\n\n")
        sections.append(
            "*Positive gain means `flow + train_profile_simple` beats "
            "`train_profile_simple` alone for "
            "`idw_prior_two_way`. This is the direct check for whether flow geometry adds "
            "signal beyond sample count and vectors-per-sample. The first table is MAE; "
            "the second is the same comparison for Spearman.*\n\n"
        )
        sections.append(table_flow_profile_lift(df, available_splits, model="idw_prior_two_way"))
        sections.append("\n")
        sections.append(table_flow_profile_lift_spearman(df, available_splits, model="idw_prior_two_way"))
        sections.append("\n")

        sections.append("## 5. Geometry vs Borrowing Nulls\n\n")
        sections.append(
            "*Same axis-aware model family, different neighbor weights. "
            "`idw_prior_two_way` uses the actual feature geometry; `uniform_prior_two_way` "
            "uses equal neighbor weights; `random_prior_two_way` uses deterministic random "
            "neighbor weights. Positive IDW gain means geometry beats the best null. "
            "The summary shows best-case MAE and best-case Spearman separately; the detailed "
            "tables show the comparison at each feature group.*\n\n"
        )
        sections.append(table_axis_aware_null_summary(df, available_splits))
        sections.append("\n")
        sections.append(table_axis_aware_nulls(df, available_splits))
        sections.append("\n")

        sections.append("## 6. Model × Feature Detail — MAE\n\n")
        sections.append(
            "*Compact interaction matrix for auditability. Bold is the best model for a "
            "given feature group within each split.*\n\n"
        )
        diag_models = [m for m in DIAGNOSTIC_MODELS if m in available_models]
        for split in available_splits:
            sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
            sections.append(table_feature_ablation_paired(df, split, diag_models, metric="mae"))
            sections.append("\n")

        sections.append("## 7. Model × Feature Detail — Spearman\n\n")
        sections.append(
            "*Same interaction matrix, but for rank agreement. This is intentionally separate "
            "from MAE so a calibration win does not get mistaken for a ranking win.*\n\n"
        )
        for split in available_splits:
            sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
            sections.append(table_feature_ablation_paired(df, split, diag_models, metric="spearman"))
            sections.append("\n")

        sections.append("## 8. Baseline Sanity Checks\n\n")
        sections.append(
            "*Random should have near-zero Spearman; global prior captures generic dataset "
            "quality but cannot rank LOTO single-dataset folds.*\n\n"
        )
        sections.append(table_diagnostic_baselines(df, available_splits))
        sections.append("\n")

        sections.append("## 9. Experiment Coverage\n\n")
        counts = df.groupby(["split", "model"]).size().unstack(fill_value=0)
        sections.append("*Number of feature-group configs completed per split × model:*\n\n")
        sections.append(counts.to_markdown() + "\n\n")

        content = "\n".join(sections)
        out_path.write_text(content)
        print(f"\n✓ Results written to {out_path}")
        print("  Sections: confound summary (1), model axis (2), feature MAE/Spearman (3-4), "
              "geometry nulls (5), model×feature details (6-7), baselines (8), coverage (9)")
        return

    # -----------------------------------------------------------------------
    # Section 0: Objective conclusion summary
    # -----------------------------------------------------------------------
    sections.append("## 0. Objective Summary\n\n")
    sections.append(
        "*Computed from `summary_table.csv`; intended to make the main conclusions "
        "less dependent on visual inspection of the scatter plots.*\n\n"
    )
    sections.append(table_objective_conclusions(df, available_splits))
    sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 1: Absolute Prediction Quality (PRIMARY — MAE)
    # -----------------------------------------------------------------------
    abs_models_present = [m for m in available_models if m in ABSOLUTE_MODELS]
    if abs_models_present:
        sections.append("## 1. Absolute Prediction Quality (Primary Metric: MAE ↓)\n\n")
        sections.append(
            "*MAE = mean |predicted − actual| AUC, in absolute PCK-percentage units. "
            "Lower is better. Each cell shows the best feature group by MAE. "
            "Ridge+IDW and TP-KRR predict in absolute AUC space. "
            "Ridge (rank), BT, and PL produce relative rank scores — their MAE is not comparable.*\n\n"
        )
        sections.append(table_absolute_performance(df, available_splits))
        sections.append("\n")

        sections.append("### MAE by Absolute Model × Feature Group\n\n")
        sections.append("*Bold = lowest MAE per row (best configuration).*\n\n")
        for split in available_splits:
            abs_sub = df[(df["split"] == split) & df["model"].isin(abs_models_present)]
            if abs_sub.empty or "mae_mean" not in abs_sub.columns or abs_sub["mae_mean"].isna().all():
                continue
            sections.append(f"#### {SPLIT_DISPLAY.get(split, split)}\n\n")
            sections.append(table_spearman_heatmap(
                df[df["model"].isin(abs_models_present)],
                split, "mae", higher_is_better=False,
                model_filter=abs_models_present,
            ))
            sections.append("\n")
    else:
        sections.append("## 1. Absolute Prediction Quality\n\n")
        sections.append("*No absolute-scale model results. Run ridge_abs and ridge_pairwise.*\n\n")

    # -----------------------------------------------------------------------
    # Section 2: Feature Ablation — MAE (features, IDW, two-stage, null priors)
    # -----------------------------------------------------------------------
    ablation_models = [m for m in ["ridge_abs",
                                   "two_way_mixed_ridge",
                                   "anchor_bilinear_ridge",
                                   "kernel_mixed_additive",
                                   "kernel_mixed_interaction",
                                   "ridge_pairwise",
                                   "ridge_pairwise_cross_resid",
                                   "ridge_pairwise_cross_resid_spline",
                                   "idw_prior_residual",
                                   "idw_prior_two_way",
                                   "idw_prior_two_way_spline",
                                   "uniform_prior_two_way",
                                   "random_prior_two_way",
                                   "uniform_prior_two_way_spline",
                                   "random_prior_two_way_spline"]
                       if m in available_models]
    if ablation_models:
        sections.append("## 2. Feature Ablation — MAE\n\n")
        sections.append(
            "*MAE mean [95% CI] per feature group. This table compares the features-only "
            "absolute Ridge baseline, coupled IDW models, the two-stage residual priors, "
            "and the uniform/random-neighbor null priors. Standalone symmetric baselines "
            "(`flow_mmd_only`, `flow_fid_only`, `flow_w2_only`) are included when present. "
            "Bold = lowest MAE per row (best model for that feature group).*\n\n"
        )
        for split in available_splits:
            sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
            sections.append(table_feature_ablation_paired(df, split, ablation_models, metric="mae"))
            sections.append("\n")

        null_models = [m for m in ["idw_prior_two_way",
                                   "uniform_prior_two_way",
                                   "random_prior_two_way",
                                   "idw_prior_two_way_spline",
                                   "uniform_prior_two_way_spline",
                                   "random_prior_two_way_spline"]
                       if m in available_models]
        if len(null_models) >= 2:
            sections.append("### 2.x Geometry vs Borrowing Nulls — MAE\n\n")
            sections.append(
                "*Same axis-aware two-stage residual structure, different neighbor weights. "
                "`idw_prior_two_way` uses the metric geometry; `uniform_prior_two_way` uses "
                "equal neighbor weights; `random_prior_two_way` uses deterministic random "
                "neighbor weights. Spline variants use the same priors but replace the "
                "linear residual ridge with a spline-expanded ridge. If uniform/random are "
                "close to IDW, panel borrowing is doing most of the work.*\n\n"
            )
            for split in available_splits:
                sections.append(f"#### {SPLIT_DISPLAY.get(split, split)}\n\n")
                sections.append(table_feature_ablation_paired(df, split, null_models, metric="mae"))
                sections.append("\n")
    else:
        sections.append("## 2. Feature Ablation — MAE\n\n")
        sections.append("*Need both ridge_abs and ridge_pairwise results to compare.*\n\n")

    # -----------------------------------------------------------------------
    # Section 3: Full Metrics — Best Configuration per Split (MAE-selected)
    # -----------------------------------------------------------------------
    sections.append("## 3. Full Metrics — Best Configuration per Split\n\n")
    sections.append("*Best (model, feature group) per split selected by lowest MAE.*\n\n")
    abs_nonbaseline = [m for m in available_models if m in ABSOLUTE_MODELS]
    for split in available_splits:
        sub = df[(df["split"] == split) & df["model"].isin(abs_nonbaseline)]
        if sub.empty:
            continue
        mae_col = "mae_mean"
        if mae_col not in sub.columns or sub[mae_col].isna().all():
            continue
        best_row = sub.loc[sub[mae_col].idxmin()]
        model = best_row["model"]
        fg    = best_row["feature_group"]
        sections.append(
            f"### {SPLIT_DISPLAY.get(split, split)} — "
            f"{MODEL_DISPLAY.get(model, model)}, {FEATURE_DISPLAY.get(fg, fg)}\n\n"
        )
        sections.append(table_full_metrics(df, split, model, fg))
        sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 4: Baseline Sanity Checks
    # -----------------------------------------------------------------------
    sections.append("## 4. Baseline Sanity Checks\n\n")
    sections.append("*Spearman for random and global-prior baselines. "
                    "Random should be ≈ 0; global prior captures generic dataset quality.*\n\n")

    baseline_models = [m for m in ["random", "global_prior"] if m in available_models]
    if baseline_models:
        col = "spearman_mean"
        sub = df[df["model"].isin(baseline_models) & (df["feature_group"] == "motion")]
        if sub.empty:
            sub = df[df["model"].isin(baseline_models)]
        rows_out = {}
        for model in baseline_models:
            msub = sub[sub["model"] == model]
            row = {}
            for split in available_splits:
                ssub = msub[msub["split"] == split]
                if ssub.empty or col not in ssub.columns:
                    row[SPLIT_DISPLAY.get(split, split)] = "—"
                else:
                    row[SPLIT_DISPLAY.get(split, split)] = fmt(ssub[col].values[0])
            rows_out[MODEL_DISPLAY.get(model, model)] = row
        if rows_out:
            sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Baseline"))
            sections.append("\n\n")

    # Density confound check — only shown when density_eval results exist
    density_present = "density_eval" in df["feature_group"].values
    if density_present:
        sections.append("### 4.x Density Confound Check\n\n")
        confound_models = [m for m in ["ridge_abs",
                                       "two_way_mixed_ridge",
                                       "anchor_bilinear_ridge",
                                       "kernel_mixed_additive",
                                       "kernel_mixed_interaction",
                                       "ridge_pairwise",
                                       "ridge_pairwise_cross_resid_spline",
                                       "idw_prior_two_way",
                                       "idw_prior_two_way_spline",
                                       "uniform_prior_two_way",
                                       "random_prior_two_way",
                                       "uniform_prior_two_way_spline",
                                       "random_prior_two_way_spline"]
                           if m in available_models]
        sections.append(table_density_confound_check(df, available_splits, confound_models))
        sections.append("\n")

    # Two-stage IDW prior comparison — only when at least one variant is present
    two_stage_present = any(
        m in available_models
        for m in ["idw_prior_residual", "idw_prior_context",
                  "idw_prior_context_local", "idw_prior_two_way",
                  "uniform_prior_residual", "random_prior_residual",
                  "uniform_prior_two_way", "random_prior_two_way",
                  "idw_prior_two_way_spline",
                  "uniform_prior_two_way_spline", "random_prior_two_way_spline"]
    )
    if two_stage_present:
        sections.append("### 4.y Two-Stage IDW Predictor Comparison\n\n")
        sections.append(textwrap.dedent("""\
            *Two-stage predictors: Stage 1 = an additive prior; Stage 2 = RidgeCV on the
            residual using only the selected distributional features (no IDW columns).
            Final prediction = prior + ridge.*

            | Variant | Stage-1 prior | Stage-2 fit |
            |---|---|---|
            | **IDW prior + residual ridge** | train-side IDW over same-benchmark neighbors | global RidgeCV |
            | **Uniform prior + residual ridge** | same train-side neighbors with equal weights | global RidgeCV |
            | **Random-neighbor prior + residual ridge** | same train-side neighbors with deterministic random weights | global RidgeCV |
            | **IDW prior + context ridge** | `context_mean[j, mv]` — per context (bm × model variant) | global RidgeCV |
            | **IDW prior + local ridge** | `context_mean[j, mv]` | per-benchmark RidgeCV |
            | **Axis-aware prior + residual ridge** | use the observed axis; IDW-interpolate only the missing train/eval axis | global RidgeCV |
            | **Axis-aware uniform/random prior + residual ridge** | same axis-aware prior, but with equal/random neighbor weights | global RidgeCV |
            | **Spline residual variants** | same priors as above | spline-expanded RidgeCV on residuals |

            *Goal: separate calibration offsets from residual feature signal without collapsing
            LOBO predictions to a constant context/global mean. Uniform/random-neighbor
            variants test whether the flow-space geometry improves over the panel borrowing
            mechanism itself.*

        """))
        sections.append(table_idw_prior_variants(df, available_splits))
        sections.append("\n")

    # -----------------------------------------------------------------------
    # Section 5: Ranking Performance (secondary)
    # -----------------------------------------------------------------------
    ranking_models_present = [m for m in ["ridge", "ridge_pairwise", "bradley_terry",
                                           "plackett_luce", "kernel_ridge"]
                               if m in available_models]
    if ranking_models_present:
        sections.append("## 5. Ranking Performance (Secondary Metric: Spearman ρ)\n\n")
        sections.append(
            "*Spearman ρ is reported as a secondary metric. "
            "The primary story is absolute MAE in Sections 1–3. "
            "Ridge (rank) is optimized directly for Spearman; Ridge+IDW is optimized for MAE. "
            "Near-identical training datasets incur equal Spearman penalty regardless of AUC gap — "
            "making MAE a more informative metric for practical dataset selection.*\n\n"
        )

        sections.append("### Spearman by Model × Feature Group\n\n")
        sections.append("*Bold = best (highest) Spearman per row.*\n\n")
        for split in available_splits:
            sections.append(f"#### {SPLIT_DISPLAY.get(split, split)}\n\n")
            sections.append(table_spearman_heatmap(
                df, split, "spearman",
                model_filter=ranking_models_present,
            ))
            sections.append("\n")

        if "ridge" in available_models:
            sections.append("### Feature Ablation — Spearman (Ridge rank model)\n\n")
            sections.append(
                "*Spearman mean [95% CI bootstrapped over held-out entities]. "
                "Rows sorted by feature group type.*\n\n"
            )
            for metric in ["spearman", "rank_mae"]:
                label = METRIC_DISPLAY[metric][0]
                sections.append(f"#### {label}\n\n")
                sections.append(table_feature_ablation(df, available_splits, "ridge", metric))
                sections.append("\n")

        # Ranking objective comparison (BT/PL vs Ridge, if run)
        ranking_obj_models = [m for m in ["ridge", "bradley_terry", "plackett_luce", "kernel_ridge"]
                               if m in available_models]
        if len(ranking_obj_models) > 1:
            sections.append("### Ranking Objective Comparison (all features)\n\n")
            sub = df[df["feature_group"] == "all"]
            if not sub.empty:
                for split in available_splits:
                    ssub = sub[sub["split"] == split]
                    col = "spearman_mean"
                    if col not in ssub.columns:
                        continue
                    sections.append(f"#### {SPLIT_DISPLAY.get(split, split)}\n\n")
                    rows_out = {}
                    for model in ranking_obj_models:
                        msub = ssub[ssub["model"] == model]
                        if msub.empty:
                            continue
                        v    = msub[col].values[0]
                        lo   = msub["spearman_ci_lo"].values[0] if "spearman_ci_lo" in msub.columns else float("nan")
                        hi   = msub["spearman_ci_hi"].values[0] if "spearman_ci_hi" in msub.columns else float("nan")
                        rmae = msub["rank_mae_mean"].values[0]  if "rank_mae_mean" in msub.columns else float("nan")
                        rows_out[MODEL_DISPLAY.get(model, model)] = {
                            "Spearman": f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]",
                            "Rank MAE": fmt(rmae, "{:.2f}"),
                        }
                    if rows_out:
                        sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Model"))
                        sections.append("\n\n")

    # -----------------------------------------------------------------------
    # Section 6: Subsampling stability
    # -----------------------------------------------------------------------
    STABILITY_METRIC_DISPLAY = {
        "mean_nn_dist":        "NN dist. (flow, directed)",
        "eval_covered_eps1px":  "ε-cov 1px eval→train",
        "eval_covered_eps4px":  "ε-cov 4px eval→train",
        "eval_covered_eps16px": "ε-cov 16px eval→train",
        "train_covered_eps1px": "ε-cov 1px train→eval",
        "train_covered_eps4px": "ε-cov 4px train→eval",
        "train_covered_eps16px":"ε-cov 16px train→eval",
        "flow_fid":            "FID (flow, sym.)",
        "flow_sliced_w2":      "SW2 (flow, sym.)",
        "dino_fid":            "FID (DINO, sym.)",
        "dino_sliced_w2":      "SW2 (DINO, sym.)",
    }
    stab_frames = []
    # Stability lives alongside the target dirs, not inside one
    stab_dir = results_dir.parent / "subsampling_stability"
    for fname in ["stability_table.csv", "stability_symmetric.csv"]:
        p = stab_dir / fname
        if p.exists():
            stab_frames.append(pd.read_csv(p))
    if stab_frames:
        stab = pd.concat(stab_frames, ignore_index=True)
        caps = sorted(stab[stab["cap"] > 0]["cap"].unique())
        cap_labels = {row["cap"]: row["cap_label"]
                      for _, row in stab[stab["cap"] > 0].iterrows()}
        sections.append("## 6. Subsampling Stability\n\n")
        sections.append(
            "*Spearman ρ between metric values at each subsample cap vs. full dataset. "
            "Values near 1.0 mean rankings are stable at that sample size. "
            "Covers directed flow coverage metrics and (if run) symmetric FID/SW2.*\n\n"
        )
        rows_out = {}
        for metric in STABILITY_METRIC_DISPLAY:
            sub = stab[stab["metric"] == metric]
            if sub.empty:
                continue
            row = {}
            for cap in caps:
                csub = sub[sub["cap"] == cap]
                row[cap_labels.get(cap, str(cap))] = fmt(csub["spearman"].values[0]) if not csub.empty else "—"
            rows_out[STABILITY_METRIC_DISPLAY[metric]] = row
        if rows_out:
            sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Metric"))
            sections.append("\n\n")

    # -----------------------------------------------------------------------
    # Section 7: Few-shot learning curve
    # -----------------------------------------------------------------------
    fewshot_dir = results_dir.parent / "few_shot"
    fewshot_curve = fewshot_dir / "few_shot_learning_curve.csv"
    fewshot_agg   = fewshot_dir / "few_shot_aggregate.csv"
    if fewshot_curve.exists() and fewshot_agg.exists():
        lc  = pd.read_csv(fewshot_curve)
        agg = pd.read_csv(fewshot_agg)
        k_vals = sorted(lc["k"].unique())
        k_labels = {k: (f"k={k} (zero-shot)" if k == 0 else f"k={k}") for k in k_vals}
        sections.append("## 7. Few-Shot Learning Curve\n\n")
        sections.append(
            "*Performance as annotated (train, eval) pairs k are added. "
            "MAE ↓ is rank MAE; Spearman ↑. "
            "CIs bootstrapped over held-out contexts.*\n\n"
        )
        for metric, label, arrow in [("mae", "Rank MAE ↓", "↓"), ("spearman", "Spearman ↑", "↑")]:
            sections.append(f"### {label}\n\n")
            rows_out = {}
            for fg in lc["feature_group"].unique():
                sub_lc = lc[(lc["feature_group"] == fg) & (lc["metric"] == metric)]
                sub_agg = agg[(agg["feature_group"] == fg) & (agg["metric"] == metric)]
                row = {}
                for k in k_vals:
                    lc_row  = sub_lc[sub_lc["k"] == k]
                    agg_row = sub_agg[sub_agg["k"] == k]
                    if lc_row.empty:
                        row[k_labels[k]] = "—"
                        continue
                    v = lc_row["mean"].values[0]
                    lo = agg_row["ci_lo"].values[0] if not agg_row.empty else float("nan")
                    hi = agg_row["ci_hi"].values[0] if not agg_row.empty else float("nan")
                    if metric == "mae":
                        row[k_labels[k]] = f"{v:.1f} [{lo:.1f}, {hi:.1f}]"
                    else:
                        row[k_labels[k]] = f"{fmt(v)} [{fmt(lo)}, {fmt(hi)}]"
                rows_out[FEATURE_DISPLAY.get(fg, fg)] = row
            if rows_out:
                sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Feature Group"))
                sections.append("\n\n")

    # -----------------------------------------------------------------------
    # Section 8: Feature mutual information
    # -----------------------------------------------------------------------
    mi_path = Path(args.mi_csv) if args.mi_csv else None
    if mi_path and mi_path.exists():
        mi_df = pd.read_csv(mi_path)
        mi_df = mi_df[mi_df["mi_point"] > 0].copy()
        random_mi_df = mi_df[mi_df["feature"].astype(str).str.startswith("random_")].copy()
        mi_df = mi_df[~mi_df["feature"].astype(str).str.startswith("random_")].copy()

        rows_out = {}
        for _, row in mi_df.iterrows():
            ci_str = f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]"
            rows_out[_pretty(row["feature"])] = {
                "MI (point)": f"{row['mi_point']:.3f}",
                "Bootstrap 95% CI": ci_str,
            }
        sections.append("## 8. Feature Predictive MI — MI(feature; AUC)\n\n")
        sections.append(
            "*KSG kNN estimator (k=5). Bootstrap CIs resample at training-dataset level "
            "(n=500). Features sorted by point estimate. Features with MI=0 omitted. "
            "Random train/eval control scalars are excluded from this table because they "
            "are fixed per dataset/benchmark and can act as arbitrary dataset IDs rather "
            "than meaningful predictive features.*\n\n"
        )
        sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Feature"))
        sections.append("\n\n")
        if not random_mi_df.empty:
            ctrl_rows = {}
            for _, row in random_mi_df.iterrows():
                ctrl_rows[_pretty(row["feature"])] = {
                    "MI (point)": f"{row['mi_point']:.3f}",
                    "Bootstrap 95% CI": f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
                }
            sections.append("### Random-ID Control MI\n\n")
            sections.append(
                "*These values are diagnostic only. A high `random_eval` MI means the "
                "global MI estimator is picking up benchmark identity/benchmark difficulty, "
                "not a real random feature signal.*\n\n"
            )
            sections.append(make_markdown_table(pd.DataFrame(ctrl_rows).T, "Feature"))
            sections.append("\n\n")
    elif mi_path:
        sections.append("## 8. Feature Predictive MI\n\n")
        sections.append(f"*MI CSV not found at `{mi_path}` — run compute_feature_mi.py first.*\n\n")

    # -----------------------------------------------------------------------
    # Section 9: Feature redundancy (top correlated pairs)
    # -----------------------------------------------------------------------
    red_path = mi_path.parent / "feature_redundancy.csv" if mi_path else None
    if red_path and red_path.exists():
        red_df = pd.read_csv(red_path, index_col=0)
        feat_names = red_df.columns.tolist()
        pairs = []
        for i, fi in enumerate(feat_names):
            for j, fj in enumerate(feat_names):
                if j <= i:
                    continue
                val = red_df.loc[fi, fj]
                pairs.append((fi, fj, float(val)))
        pairs.sort(key=lambda x: -x[2])
        top_pairs = [(fi, fj, v) for fi, fj, v in pairs if v > 0][:20]
        if top_pairs:
            rows_out = {}
            for fi, fj, v in top_pairs:
                rows_out[f"{_pretty(fi)}  ↔  {_pretty(fj)}"] = {"MI (nats)": f"{v:.3f}"}
            sections.append("## 9. Feature Redundancy — Top Correlated Pairs\n\n")
            sections.append(
                "*Pairwise MI(feature_i; feature_j) — point estimates (KSG k=5). "
                "High MI = features carry similar information about the training-eval pair.*\n\n"
            )
            sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Feature pair"))
            sections.append("\n\n")

    # -----------------------------------------------------------------------
    # Section 10: Tensor Product KRR comparison
    # -----------------------------------------------------------------------
    KRR_TP_MODELS = [m for m in df["model"].unique() if str(m).startswith("krr_tp_")]
    KRR_TP_DISPLAY = {
        "krr_tp_flow_nn":    "TP-KRR flow NN",
        "krr_tp_flow_eps":   "TP-KRR flow ε-cov 1px",
        "krr_tp_flow_eps16": "TP-KRR flow ε-cov 16px",
        "krr_tp_dino_nn":    "TP-KRR DINO NN",
        "krr_tp_dino_eps":   "TP-KRR DINO ε-cov 1px",
    }
    if KRR_TP_MODELS:
        sections.append("## 10. Tensor Product KRR\n\n")
        sections.append(
            "*Kernel Ridge Regression with K_train ⊗ K_eval tensor product kernel. "
            "Each variant uses a different distance measure to define similarity between "
            "training datasets (K_train) and between benchmarks (K_eval). "
            "Compared to Ridge at the same feature group. Best per-split in **bold**.*\n\n"
        )
        sections.append(
            "**Interpretation:** Unlike standard models that use distributional features as "
            "input X, TP-KRR uses explicit train-train and eval-eval distances to build a "
            "structured kernel. This gives a natural prior for LOTO (novel training dataset) "
            "and LOBO (novel benchmark) via kernel extrapolation.\n\n"
        )
        col = "spearman_mean"
        # Show best Spearman per (model, split) across all feature groups
        for split in available_splits:
            sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
            rows_out = {}
            # Add Ridge reference row
            ridge_sub = df[(df["model"] == "ridge") & (df["split"] == split)]
            if not ridge_sub.empty and col in ridge_sub.columns:
                best_ridge = ridge_sub[col].max()
                best_fg    = ridge_sub.loc[ridge_sub[col].idxmax(), "feature_group"]
                rows_out["Ridge (best fg)"] = {
                    "Best Spearman": f"**{fmt(best_ridge)}**",
                    "Feature group": FEATURE_DISPLAY.get(best_fg, best_fg),
                }
            for model in sorted(KRR_TP_MODELS):
                m_sub = df[(df["model"] == model) & (df["split"] == split)]
                if m_sub.empty or col not in m_sub.columns:
                    continue
                best_v  = m_sub[col].max()
                best_fg = m_sub.loc[m_sub[col].idxmax(), "feature_group"]
                rows_out[KRR_TP_DISPLAY.get(model, model)] = {
                    "Best Spearman": fmt(best_v),
                    "Feature group": FEATURE_DISPLAY.get(best_fg, best_fg),
                }
            if rows_out:
                sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Model"))
                sections.append("\n\n")

        # Detailed table: all KRR TP variants × all feature groups for LOBO and LOTO
        for split in [s for s in ["lobo", "loto"] if s in available_splits]:
            sections.append(f"#### {SPLIT_DISPLAY.get(split, split)} — Feature Group Detail\n\n")
            fg_order = [f for f in FEATURE_ORDER if f in df["feature_group"].values]
            tp_models_sorted = sorted(KRR_TP_MODELS)
            ridge_available  = "ridge" in available_models
            all_models_row   = (["ridge"] if ridge_available else []) + tp_models_sorted
            col_headers = [KRR_TP_DISPLAY.get(m, MODEL_DISPLAY.get(m, m)) for m in all_models_row]
            header = "| Feature group | " + " | ".join(col_headers) + " |"
            sep    = "|---|" + "|".join(["---"] * len(col_headers)) + "|"
            lines  = [header, sep]
            for fg in fg_order:
                cells = []
                for model in all_models_row:
                    sub = df[(df["model"] == model) & (df["split"] == split)
                             & (df["feature_group"] == fg)]
                    if sub.empty or col not in sub.columns:
                        cells.append("—")
                    else:
                        cells.append(fmt(sub[col].values[0]))
                lines.append(f"| {FEATURE_DISPLAY.get(fg, fg)} | " + " | ".join(cells) + " |")
            sections.append("\n".join(lines) + "\n\n")
    else:
        sections.append("## 10. Tensor Product KRR\n\n")
        sections.append(
            "*No krr_tp_* results found. Run Step 0d (compute_pairwise_self_distances.py) "
            "then re-run experiments with krr_tp_* models.*\n\n"
        )

    # -----------------------------------------------------------------------
    # Section 11: IDW Pairwise model comparison
    # -----------------------------------------------------------------------
    IDW_MODELS = [m for m in df["model"].unique()
                  if str(m) == "ridge_pairwise"
                  or str(m) == "ridge_pairwise_cross"
                  or str(m) == "ridge_pairwise_cross_resid"
                  or str(m) == "ridge_pairwise_cross_resid_spline"
                  or (str(m).startswith("ridge_pairwise_") and m != "ridge_pairwise_all")]
    IDW_DISPLAY = {
        "ridge_pairwise":         "IDW (coupled)",
        "ridge_pairwise_uniform": "Uniform neighbors",
        "ridge_pairwise_random":  "Random neighbors",
        "ridge_pairwise_cross":   "IDW (2-axis)",
        "ridge_pairwise_cross_resid": "IDW (2-axis residual)",
        "ridge_pairwise_cross_resid_spline": "IDW (2-axis residual spline)",
        "ridge_pairwise_nn":      "IDW (NN distance)",
        "ridge_pairwise_eps1px":  "IDW (ε-cov 1px)",
        "ridge_pairwise_eps16px": "IDW (ε-cov 16px)",
        "ridge_pairwise_kl":      "IDW (KL div)",
    }
    idw_order = ["ridge_pairwise", "ridge_pairwise_uniform", "ridge_pairwise_random",
                 "ridge_pairwise_cross_resid", "ridge_pairwise_cross_resid_spline",
                 "ridge_pairwise_cross",
                 "ridge_pairwise_nn", "ridge_pairwise_eps1px",
                 "ridge_pairwise_eps16px", "ridge_pairwise_kl"]
    if IDW_MODELS:
        sections.append("## 11. Ridge + IDW Pairwise Models\n\n")
        sections.append(
            "*Ridge regression augmented with inverse-distance-weighted predictions from "
            "fold neighbors. The coupled model adds two one-axis IDW predictions: one "
            "from in-fold trains similar to train_i, one from in-fold benchmarks similar "
            "to benchmark_j. The 2-axis model also adds a joint train-neighbor × "
            "benchmark-neighbor IDW prediction, which remains informative when the "
            "held-out benchmark has no direct observed AUC. The residual variant "
            "borrows train effects after subtracting the source benchmark mean, which "
            "reduces direct transfer of the wrong benchmark scale. "
            "Predictions are in absolute AUC units — no rank normalization needed.*\n\n"
        )
        sections.append(
            "*Each variant uses a single distance metric to define neighborhood: "
            "NN distance (geometric), ε-coverage (threshold-based), or KL divergence "
            "(information-theoretic). Per-metric isolation means each variant's result "
            "is a clean test of that metric's value as a neighborhood proxy.*\n\n"
        )
        col = "spearman_mean"
        for split in available_splits:
            sections.append(f"### {SPLIT_DISPLAY.get(split, split)}\n\n")
            rows_out = {}
            # Ridge reference row
            ridge_sub = df[(df["model"] == "ridge") & (df["split"] == split)]
            if not ridge_sub.empty and col in ridge_sub.columns:
                best_v  = ridge_sub[col].max()
                best_fg = ridge_sub.loc[ridge_sub[col].idxmax(), "feature_group"]
                rows_out["Ridge (best fg)"] = {
                    "Best Spearman": fmt(best_v),
                    "Feature group": FEATURE_DISPLAY.get(best_fg, best_fg),
                    "vs Ridge": "—",
                }
            ridge_best = ridge_sub[col].max() if not ridge_sub.empty and col in ridge_sub.columns else float("nan")
            for model in sorted(IDW_MODELS, key=lambda m: idw_order.index(m) if m in idw_order else 99):
                m_sub = df[(df["model"] == model) & (df["split"] == split)]
                if m_sub.empty or col not in m_sub.columns:
                    continue
                best_v  = m_sub[col].max()
                best_fg = m_sub.loc[m_sub[col].idxmax(), "feature_group"]
                delta   = best_v - ridge_best if np.isfinite(ridge_best) else float("nan")
                delta_s = f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}"
                rows_out[IDW_DISPLAY.get(model, model)] = {
                    "Best Spearman": fmt(best_v),
                    "Feature group": FEATURE_DISPLAY.get(best_fg, best_fg),
                    "vs Ridge": delta_s if np.isfinite(delta) else "—",
                }
            if rows_out:
                sections.append(make_markdown_table(pd.DataFrame(rows_out).T, "Model"))
                sections.append("\n\n")

        # Detailed table: all IDW variants × all feature groups for LOTO and LOBO
        for split in [s for s in ["loto", "lobo", "joint_cell"] if s in available_splits]:
            sections.append(f"#### {SPLIT_DISPLAY.get(split, split)} — Spearman by Feature Group\n\n")
            fg_order = [f for f in FEATURE_ORDER if f in df["feature_group"].values]
            idw_sorted = sorted(IDW_MODELS, key=lambda m: idw_order.index(m) if m in idw_order else 99)
            all_models_row = ["ridge"] + idw_sorted
            col_headers = [IDW_DISPLAY.get(m, MODEL_DISPLAY.get(m, m)) for m in all_models_row]
            col_headers[0] = "Ridge"
            header = "| Feature group | " + " | ".join(col_headers) + " |"
            sep    = "|---|" + "|".join(["---"] * len(col_headers)) + "|"
            lines  = [header, sep]
            for fg in fg_order:
                cells = []
                best_in_row = float("-inf")
                row_vals = []
                for model in all_models_row:
                    sub = df[(df["model"] == model) & (df["split"] == split)
                             & (df["feature_group"] == fg)]
                    v = sub[col].values[0] if not sub.empty and col in sub.columns else float("nan")
                    row_vals.append(v)
                    if np.isfinite(v):
                        best_in_row = max(best_in_row, v)
                for v in row_vals:
                    s = fmt(v)
                    cells.append(f"**{s}**" if np.isfinite(v) and v == best_in_row else s)
                lines.append(f"| {FEATURE_DISPLAY.get(fg, fg)} | " + " | ".join(cells) + " |")
            sections.append("\n".join(lines) + "\n\n")
    else:
        sections.append("## 11. Ridge + IDW Pairwise Models\n\n")
        sections.append(
            "*No ridge_pairwise_* results found. Run experiments with "
            "`ridge_pairwise` for coupled feature-group/IDW ablations, or "
            "`ridge_pairwise_nn ridge_pairwise_eps1px ridge_pairwise_eps16px ridge_pairwise_kl` "
            "for fixed-neighborhood diagnostics.*\n\n"
        )

    # -----------------------------------------------------------------------
    # Section 12: Experiment coverage summary
    # -----------------------------------------------------------------------
    sections.append("## 12. Experiment Coverage\n\n")
    counts = df.groupby(["split", "model"]).size().unstack(fill_value=0)
    sections.append("*Number of feature-group configs completed per split × model:*\n\n")
    sections.append(counts.to_markdown() + "\n\n")

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------
    content = "\n".join(sections)
    out_path.write_text(content)
    print(f"\n✓ Results written to {out_path}")
    print(f"  Sections: absolute MAE (1), feature ablation MAE (2), full metrics (3), "
          f"baselines (4), ranking/Spearman (5), stability (6), MI (8), TP-KRR (10), IDW (11)")


if __name__ == "__main__":
    main()
