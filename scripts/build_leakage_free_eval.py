#!/usr/bin/env python3
"""
One-stop analysis pipeline for leakage-free evaluation.

Outputs (under --output-dir):
  - auc_results.csv: per-run, per-benchmark AUC up to N steps
  - auc_with_features.csv: AUC table joined with coverage/MMD predictors (normalized if enabled)
  - auc_with_features_raw.csv: Raw AUC table before benchmark normalization (optional)
  - analysis_normalization.txt: Benchmark normalization settings (optional)
  - curve_stats.csv: per-run, per-benchmark peak/final/drop stats
  - prediction_lobo_summary.csv: leave-one-benchmark-out metrics
  - prediction_lobo_rows.csv: per-row LOBO predictions
  - prediction_loto_summary.csv: leave-one-training-dataset-out metrics
  - prediction_loto_rows.csv: per-row LOTO predictions
  - prediction_loto_holdout_placement_summary.csv: LOTO holdout insertion ranking metrics
  - prediction_loto_holdout_placement_detail.csv: per-task LOTO holdout insertion diagnostics
  - prediction_jointood_summary.csv: leave-one-train-dataset-and-benchmark-out metrics (optional)
  - prediction_jointood_rows.csv: per-row joint-OOD predictions (optional)
  - prediction_jointood_holdout_placement_summary.csv: Joint-OOD holdout insertion ranking metrics
  - prediction_jointood_holdout_placement_detail.csv: per-task Joint-OOD insertion diagnostics
  - regression_summary.txt: OLS/mixedlm regression summary

Optional (mode=all):
  - dev_selected_results.csv, dev_selected_summary.csv
  - fixed_step_results.csv
"""

import argparse
import datetime
import json
import math
import csv
import re
import subprocess
import sys
from collections import defaultdict
from typing import List
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


RUN_NAME_RE = re.compile(
    r"^(?P<train_dataset>.+)_(?P<step_tag>logsteps|steps)(?P<logsteps>\d+)"
    r"_pretrained(?P<pretrained>True|False)_freeze(?P<freeze>True|False)"
    r"_(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$"
)

MODEL_FAMILY_DEFAULT = "catspp"
MODEL_FAMILY_ALIASES = {
    "mixed": MODEL_FAMILY_DEFAULT,
    "mixed_plots": MODEL_FAMILY_DEFAULT,
    "cats": MODEL_FAMILY_DEFAULT,
    "catspp": MODEL_FAMILY_DEFAULT,
    "raft": "raft",
    "flowformer": "flowformer",
}
MODEL_FAMILY_ORDER = (MODEL_FAMILY_DEFAULT, "raft", "flowformer")

COVERAGE_RENAME_MAP = {
    "flow_eval_to_train_coverage": "flow_eval_to_train_over_train_precision",
    "flow_eval_to_train_coverage_logit": "flow_eval_to_train_over_train_precision_logit",
    "flow_train_to_eval_coverage": "flow_train_to_eval_over_eval_recall",
    "flow_train_to_eval_coverage_logit": "flow_train_to_eval_over_eval_recall_logit",
    "resnet_eval_to_train_coverage": "resnet_eval_to_train_over_train_precision",
    "resnet_eval_to_train_coverage_logit": "resnet_eval_to_train_over_train_precision_logit",
    "resnet_train_to_eval_coverage": "resnet_train_to_eval_over_eval_recall",
    "resnet_train_to_eval_coverage_logit": "resnet_train_to_eval_over_eval_recall_logit",
    "dino_eval_to_train_coverage": "dino_eval_to_train_over_train_precision",
    "dino_eval_to_train_coverage_logit": "dino_eval_to_train_over_train_precision_logit",
    "dino_train_to_eval_coverage": "dino_train_to_eval_over_eval_recall",
    "dino_train_to_eval_coverage_logit": "dino_train_to_eval_over_eval_recall_logit",
    "hof_eval_to_train_coverage": "hof_eval_to_train_over_train_precision",
    "hof_eval_to_train_coverage_logit": "hof_eval_to_train_over_train_precision_logit",
    "hof_train_to_eval_coverage": "hof_train_to_eval_over_eval_recall",
    "hof_train_to_eval_coverage_logit": "hof_train_to_eval_over_eval_recall_logit",
}


def normalize_dataset_name(name):
    if name is None:
        return None
    name = str(name).strip().lower()
    name = name.replace('+', '_')
    if name.endswith("_cats"):
        name = name[:-5]
    return name


def parse_eps_values(raw: str) -> List[str]:
    if raw is None:
        return []
    values = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        if token.endswith("px"):
            token = token[:-2]
        try:
            value = float(token)
        except ValueError:
            values.append(token)
            continue
        values.append(f"{value:g}".replace(".", "p"))
    return values


def apply_flow_eps_ring_features(
    df: pd.DataFrame,
    eps_values: List[str],
    weighted: bool = False,
) -> pd.DataFrame:
    """Convert cumulative flow epsilon ladders into per-ring bins in place."""
    if df.empty or not eps_values:
        return df
    out = df.copy()
    suffix = "_weighted" if weighted else ""
    for direction in ("train_to_eval", "eval_to_train"):
        prev = None
        for eps in eps_values:
            col = f"flow_{direction}_eps{eps}px{suffix}"
            if col not in out.columns:
                continue
            current = pd.to_numeric(out[col], errors="coerce")
            if prev is None:
                out[col] = current
            else:
                out[col] = current - prev
            prev = current
    return out


def add_explicit_coverage_columns(df):
    if df.empty:
        return df
    df = df.copy()
    for old, new in COVERAGE_RENAME_MAP.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    return df


def parse_training_summary(summary_path: Path) -> dict:
    info = {}
    if not summary_path.exists():
        return info
    try:
        with summary_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Train dataset:"):
                    info["train_dataset"] = normalize_dataset_name(line.split(":", 1)[1])
                elif line.startswith("Pretrained backbone:"):
                    value = line.split(":", 1)[1].strip()
                    info["pretrained"] = value.lower() in {"true", "1", "yes"}
                elif line.startswith("Freeze backbone:"):
                    value = line.split(":", 1)[1].strip()
                    info["freeze"] = value.lower() in {"true", "1", "yes"}
    except OSError:
        return info
    return info


def parse_run_name(snapshot_dir: Path) -> dict:
    info = {
        "run_id": str(snapshot_dir),
        "run_name": snapshot_dir.name,
        "train_dataset": None,
        "step_tag": None,
        "logsteps": None,
        "pretrained": None,
        "freeze": None,
        "timestamp": None,
        "model_family": None,
    }
    match = RUN_NAME_RE.match(snapshot_dir.name)
    if match:
        info.update(match.groupdict())
        info["train_dataset"] = normalize_dataset_name(info["train_dataset"])
        info["logsteps"] = int(info["logsteps"])
        info["pretrained"] = info["pretrained"] == "True"
        info["freeze"] = info["freeze"] == "True"
    summary_info = parse_training_summary(snapshot_dir / "training_summary.txt")
    config_info = parse_run_config(snapshot_dir / "config.yaml")
    for key in ("train_dataset", "pretrained", "freeze"):
        if summary_info.get(key) is not None:
            if key == "train_dataset":
                # Keep run name if it has mixed ratios; otherwise prefer summary.
                if info.get(key) is None or info.get(key) == "spair_synthetic":
                    info[key] = summary_info[key]
            elif info.get(key) is None:
                info[key] = summary_info[key]
        if info.get(key) is None and config_info.get(key) is not None:
            info[key] = config_info[key]
        elif key == "train_dataset" and config_info.get(key) is not None:
            if info.get(key) is None or info.get(key) == "spair_synthetic":
                info[key] = config_info[key]
    name_dataset = parse_dataset_from_snapshot_name(snapshot_dir.name)
    if name_dataset is not None:
        if info.get("train_dataset") is None:
            info["train_dataset"] = name_dataset
        elif len(name_dataset) > len(str(info.get("train_dataset"))):
            info["train_dataset"] = name_dataset
    info["model_family"] = derive_model_family(snapshot_dir)
    return info


def parse_run_config(config_path: Path) -> dict:
    info = {}
    if not config_path.exists():
        return info
    try:
        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return info
    dataset_cfg = config.get("dataset", {})
    if dataset_cfg.get("mixed", False) or "datasets" in dataset_cfg:
        datasets = dataset_cfg.get("datasets", [])
        percentages = dataset_cfg.get("percentages", [])
        if datasets:
            base_name = "_".join(datasets)
            if percentages and len(percentages) == len(datasets):
                percent_tokens = []
                for value in percentages:
                    try:
                        percent_tokens.append(str(int(round(float(value) * 100))))
                    except (TypeError, ValueError):
                        percent_tokens = []
                        break
                if percent_tokens:
                    base_name = f"{base_name}_{'_'.join(percent_tokens)}"
            info["train_dataset"] = normalize_dataset_name(base_name)
    else:
        dataset_name = dataset_cfg.get("dataset_name")
        if dataset_name:
            info["train_dataset"] = normalize_dataset_name(dataset_name)
    model_cfg = config.get("model", {})
    if "pretrained_backbone" in model_cfg:
        info["pretrained"] = bool(model_cfg.get("pretrained_backbone"))
    if "freeze" in model_cfg:
        info["freeze"] = bool(model_cfg.get("freeze"))
    return info


def parse_dataset_from_snapshot_name(name: str):
    if not name:
        return None
    lower = name.lower()
    for token in ("raft", "flowformer"):
        marker = f"_{token}"
        if marker in lower:
            prefix = name[: lower.index(marker)]
            return normalize_dataset_name(prefix)
    return None


def derive_model_family(snapshot_dir: Path) -> str:
    name = snapshot_dir.name.lower()
    if "raft" in name:
        return "raft"
    if "flowformer" in name:
        return "flowformer"
    return MODEL_FAMILY_DEFAULT


def list_snapshot_dirs(root_dirs):
    snapshot_dirs = []
    for root_dir in root_dirs:
        root = Path(root_dir)
        if not root.exists():
            continue
        for csv_path in root.rglob("validation_results.csv"):
            snapshot_dirs.append(csv_path.parent)
    return sorted(set(snapshot_dirs))


def select_dev_step(df, dev_benchmarks, metric):
    dev_df = df[df["benchmark"].isin(dev_benchmarks)]
    if dev_df.empty:
        return None, None, 0
    dev_score = dev_df.groupby("training_steps")[metric].mean()
    max_score = dev_score.max()
    best_steps = dev_score[dev_score == max_score].index.tolist()
    best_step = min(best_steps)
    return best_step, float(max_score), int(dev_df["benchmark"].nunique())


def find_nearest_step(df, target_step):
    steps = df["training_steps"].dropna().unique()
    if len(steps) == 0:
        return None
    steps = np.array(sorted(steps))
    return int(steps[np.abs(steps - target_step).argmin()])


def compute_auc(sub_df, metric, max_steps, pad_to_max=False):
    sub = sub_df[sub_df["training_steps"] <= max_steps].copy()
    if sub.empty:
        return np.nan, 0, None
    sub = sub.groupby("training_steps", as_index=False)[metric].mean()
    sub = sub.sort_values("training_steps")
    last_step = int(sub["training_steps"].iloc[-1])
    if pad_to_max and last_step < max_steps:
        last_value = sub[metric].iloc[-1]
        pad_row = pd.DataFrame({"training_steps": [max_steps], metric: [last_value]})
        sub = pd.concat([sub, pad_row], ignore_index=True)
    if len(sub) < 2:
        return np.nan, len(sub), last_step
    auc = float(np.trapz(sub[metric].to_numpy(), sub["training_steps"].to_numpy()))
    return auc, len(sub), last_step


def compute_curve_stats(df, metric):
    rows = []
    for bench, sub in df.groupby("benchmark"):
        sub = sub.dropna(subset=[metric]).sort_values("training_steps")
        if sub.empty:
            continue
        peak_idx = sub[metric].idxmax()
        peak_row = sub.loc[peak_idx]
        final_row = sub.iloc[-1]
        rows.append(
            {
                "benchmark": bench,
                "peak_pck": float(peak_row[metric]),
                "peak_training_steps": int(peak_row["training_steps"]),
                "peak_epoch": int(peak_row["epoch"]) if "epoch" in sub.columns else np.nan,
                "final_pck": float(final_row[metric]),
                "final_training_steps": int(final_row["training_steps"]),
                "final_epoch": int(final_row["epoch"]) if "epoch" in sub.columns else np.nan,
                "drop_pck": float(peak_row[metric] - final_row[metric]),
            }
        )
    return rows


def _parse_float(row, key):
    value = row.get(key, np.nan)
    try:
        return float(value) if value not in (None, "") else np.nan
    except (ValueError, TypeError):
        return np.nan


def load_coverage_lookup(csv_path, allow_unsplit=True):
    coverage_lookup = {}
    path = Path(csv_path)
    if not path.exists():
        print(f"Warning: Coverage CSV not found: {csv_path}")
        return coverage_lookup

    try:
        with path.open("r") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            use_flow_eps_format = "train_dataset" in fieldnames and "eval_dataset" in fieldnames
            for row in reader:
                if use_flow_eps_format:
                    train_dataset = normalize_dataset_name(row.get("train_dataset"))
                    train_split = normalize_dataset_name(row.get("train_split"))
                    eval_dataset = normalize_dataset_name(row.get("eval_dataset"))
                    eval_split = normalize_dataset_name(row.get("eval_split"))
                else:
                    train_dataset = normalize_dataset_name(row.get("dataset1"))
                    train_split = normalize_dataset_name(row.get("split1"))
                    eval_dataset = normalize_dataset_name(row.get("dataset2"))
                    eval_split = normalize_dataset_name(row.get("split2"))

                if not train_dataset or not eval_dataset:
                    continue

                train_id = f"{train_dataset}_{train_split}" if train_split else train_dataset
                eval_id = f"{eval_dataset}_{eval_split}" if eval_split else eval_dataset

                train_to_eval = _parse_float(row, "train_to_eval_coverage")
                if np.isnan(train_to_eval):
                    train_to_eval = _parse_float(row, "recall")

                eval_to_train = _parse_float(row, "eval_to_train_coverage")
                if np.isnan(eval_to_train):
                    eval_to_train = _parse_float(row, "precision")

                outside_val = _parse_float(row, "outside")
                k_val = _parse_float(row, "k")
                radius_quantile = _parse_float(row, "radius_quantile")
                radius_train = _parse_float(row, "radius_train")
                radius_eval = _parse_float(row, "radius_eval")
                mean_eval_to_train = _parse_float(row, "mean_nn_eval_to_train")
                median_eval_to_train = _parse_float(row, "median_nn_eval_to_train")
                p90_eval_to_train = _parse_float(row, "p90_nn_eval_to_train")
                mean_train_to_eval = _parse_float(row, "mean_nn_train_to_eval")
                median_train_to_eval = _parse_float(row, "median_nn_train_to_eval")
                p90_train_to_eval = _parse_float(row, "p90_nn_train_to_eval")
                if np.isnan(mean_eval_to_train):
                    mean_eval_to_train = _parse_float(row, "mean_nn_eval_to_train_k1")
                if np.isnan(median_eval_to_train):
                    median_eval_to_train = _parse_float(row, "median_nn_eval_to_train_k1")
                if np.isnan(p90_eval_to_train):
                    p90_eval_to_train = _parse_float(row, "p90_nn_eval_to_train_k1")
                if np.isnan(mean_train_to_eval):
                    mean_train_to_eval = _parse_float(row, "mean_nn_train_to_eval_k1")
                if np.isnan(median_train_to_eval):
                    median_train_to_eval = _parse_float(row, "median_nn_train_to_eval_k1")
                if np.isnan(p90_train_to_eval):
                    p90_train_to_eval = _parse_float(row, "p90_nn_train_to_eval_k1")
                kl_eval_to_train = _parse_float(row, "kl_eval_to_train")
                kl_train_to_eval = _parse_float(row, "kl_train_to_eval")
                kl_eval_to_train_hist = _parse_float(row, "kl_eval_to_train_hist")
                kl_train_to_eval_hist = _parse_float(row, "kl_train_to_eval_hist")
                kl_eval_to_train_hist_log1p = _parse_float(row, "kl_eval_to_train_hist_log1p_linear")
                kl_train_to_eval_hist_log1p = _parse_float(row, "kl_train_to_eval_hist_log1p_linear")
                kl_eval_to_train_hist_radius = _parse_float(row, "kl_eval_to_train_hist_radius")
                kl_train_to_eval_hist_radius = _parse_float(row, "kl_train_to_eval_hist_radius")
                kl_eval_to_train_hist_median = _parse_float(row, "kl_eval_to_train_hist_median")
                kl_train_to_eval_hist_median = _parse_float(row, "kl_train_to_eval_hist_median")
                kl_eval_to_train_hist_rank = _parse_float(row, "kl_eval_to_train_hist_rank")
                kl_train_to_eval_hist_rank = _parse_float(row, "kl_train_to_eval_hist_rank")
                density_l2 = _parse_float(row, "hof_density_l2")
                density_l1 = _parse_float(row, "hof_density_l1")
                density_cos = _parse_float(row, "hof_density_cosine")

                if not allow_unsplit and (not train_split or not eval_split):
                    continue

                eps_eval_to_train = {}
                eps_train_to_eval = {}
                for key, value in row.items():
                    if key is None:
                        continue
                    key = key.strip()
                    if key.startswith("eval_covered_by_train_eps") and key.endswith("px"):
                        eps = key.replace("eval_covered_by_train_eps", "").replace("px", "")
                        eps_eval_to_train[f"eval_to_train_eps{eps}px"] = _parse_float(row, key)
                    elif key.startswith("train_covered_by_eval_eps") and key.endswith("px"):
                        eps = key.replace("train_covered_by_eval_eps", "").replace("px", "")
                        eps_train_to_eval[f"train_to_eval_eps{eps}px"] = _parse_float(row, key)
                    elif key.startswith("eval_covered_by_train_eps") and key.endswith("px_weighted"):
                        eps = key.replace("eval_covered_by_train_eps", "").replace("px_weighted", "")
                        eps_eval_to_train[f"eval_to_train_eps{eps}px_weighted"] = _parse_float(row, key)
                    elif key.startswith("train_covered_by_eval_eps") and key.endswith("px_weighted"):
                        eps = key.replace("train_covered_by_eval_eps", "").replace("px_weighted", "")
                        eps_train_to_eval[f"train_to_eval_eps{eps}px_weighted"] = _parse_float(row, key)
                    elif key.startswith("eval_to_train_auc") or key.startswith("eval_to_train_eps_at"):
                        eps_eval_to_train[key] = _parse_float(row, key)
                    elif key.startswith("train_to_eval_auc") or key.startswith("train_to_eval_eps_at"):
                        eps_train_to_eval[key] = _parse_float(row, key)

                coverage_lookup[(train_id, eval_id)] = {
                    "train_to_eval_coverage": train_to_eval,
                    "eval_to_train_coverage": eval_to_train,
                    "outside": outside_val,
                    "k": k_val,
                    "radius_quantile": radius_quantile,
                    "radius_train": radius_train,
                    "radius_eval": radius_eval,
                    "mean_nn_eval_to_train": mean_eval_to_train,
                    "median_nn_eval_to_train": median_eval_to_train,
                    "p90_nn_eval_to_train": p90_eval_to_train,
                    "mean_nn_train_to_eval": mean_train_to_eval,
                    "median_nn_train_to_eval": median_train_to_eval,
                    "p90_nn_train_to_eval": p90_train_to_eval,
                    "kl_eval_to_train": kl_eval_to_train,
                    "kl_train_to_eval": kl_train_to_eval,
                    "kl_eval_to_train_hist": kl_eval_to_train_hist,
                    "kl_train_to_eval_hist": kl_train_to_eval_hist,
                    "kl_eval_to_train_hist_log1p_linear": kl_eval_to_train_hist_log1p,
                    "kl_train_to_eval_hist_log1p_linear": kl_train_to_eval_hist_log1p,
                    "kl_eval_to_train_hist_radius": kl_eval_to_train_hist_radius,
                    "kl_train_to_eval_hist_radius": kl_train_to_eval_hist_radius,
                    "kl_eval_to_train_hist_median": kl_eval_to_train_hist_median,
                    "kl_train_to_eval_hist_median": kl_train_to_eval_hist_median,
                    "kl_eval_to_train_hist_rank": kl_eval_to_train_hist_rank,
                    "kl_train_to_eval_hist_rank": kl_train_to_eval_hist_rank,
                    "hof_density_l2": density_l2,
                    "hof_density_l1": density_l1,
                    "hof_density_cosine": density_cos,
                    **eps_eval_to_train,
                    **eps_train_to_eval,
                }
                if allow_unsplit:
                    coverage_lookup[(train_dataset, eval_dataset)] = {
                        "train_to_eval_coverage": train_to_eval,
                        "eval_to_train_coverage": eval_to_train,
                        "outside": outside_val,
                        "k": k_val,
                        "radius_quantile": radius_quantile,
                        "radius_train": radius_train,
                        "radius_eval": radius_eval,
                        "mean_nn_eval_to_train": mean_eval_to_train,
                        "median_nn_eval_to_train": median_eval_to_train,
                        "p90_nn_eval_to_train": p90_eval_to_train,
                        "mean_nn_train_to_eval": mean_train_to_eval,
                        "median_nn_train_to_eval": median_train_to_eval,
                        "p90_nn_train_to_eval": p90_train_to_eval,
                        "kl_eval_to_train": kl_eval_to_train,
                        "kl_train_to_eval": kl_train_to_eval,
                        "kl_eval_to_train_hist": kl_eval_to_train_hist,
                        "kl_train_to_eval_hist": kl_train_to_eval_hist,
                        "kl_eval_to_train_hist_log1p_linear": kl_eval_to_train_hist_log1p,
                        "kl_train_to_eval_hist_log1p_linear": kl_train_to_eval_hist_log1p,
                        "kl_eval_to_train_hist_radius": kl_eval_to_train_hist_radius,
                        "kl_train_to_eval_hist_radius": kl_train_to_eval_hist_radius,
                        "kl_eval_to_train_hist_median": kl_eval_to_train_hist_median,
                        "kl_train_to_eval_hist_median": kl_train_to_eval_hist_median,
                        "kl_eval_to_train_hist_rank": kl_eval_to_train_hist_rank,
                        "kl_train_to_eval_hist_rank": kl_train_to_eval_hist_rank,
                        "hof_density_l2": density_l2,
                        "hof_density_l1": density_l1,
                        "hof_density_cosine": density_cos,
                        **eps_eval_to_train,
                        **eps_train_to_eval,
                    }
    except Exception as exc:
        print(f"Warning: could not read coverage CSV {csv_path}: {exc}")

    return coverage_lookup


def load_variogram_lookup(csv_path, allow_unsplit=True):
    lookup = {}
    path = Path(csv_path)
    if not path.exists():
        print(f"Warning: Variogram CSV not found: {csv_path}")
        return lookup

    try:
        with path.open("r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                train_dataset = normalize_dataset_name(row.get("train_dataset"))
                train_split = normalize_dataset_name(row.get("train_split"))
                eval_dataset = normalize_dataset_name(row.get("eval_dataset"))
                eval_split = normalize_dataset_name(row.get("eval_split"))
                if not train_dataset or not eval_dataset:
                    continue

                train_id = f"{train_dataset}_{train_split}" if train_split else train_dataset
                eval_id = f"{eval_dataset}_{eval_split}" if eval_split else eval_dataset

                metrics = {
                    "train_auc": _parse_float(row, "train_auc"),
                    "eval_auc": _parse_float(row, "eval_auc"),
                    "train_auc_norm": _parse_float(row, "train_auc_norm"),
                    "eval_auc_norm": _parse_float(row, "eval_auc_norm"),
                    "auc_diff": _parse_float(row, "auc_diff"),
                    "auc_diff_norm": _parse_float(row, "auc_diff_norm"),
                    "curve_l1": _parse_float(row, "curve_l1"),
                    "curve_l2": _parse_float(row, "curve_l2"),
                    "curve_corr": _parse_float(row, "curve_corr"),
                    "overlap_bins": _parse_float(row, "overlap_bins"),
                    "total_bins": _parse_float(row, "total_bins"),
                }

                if not allow_unsplit and (not train_split or not eval_split):
                    continue

                lookup[(train_id, eval_id)] = metrics
                if allow_unsplit:
                    lookup[(train_dataset, eval_dataset)] = metrics
    except Exception as exc:
        print(f"Warning: could not read variogram CSV {csv_path}: {exc}")

    return lookup


def load_mmd_lookup(csv_path, allow_unsplit=True):
    path = Path(csv_path)
    if not path.exists():
        print(f"Warning: MMD CSV not found: {csv_path}")
        return {}

    mmd_lookup = {}
    try:
        df = pd.read_csv(path)
        has_splits = "split1" in df.columns and "split2" in df.columns
        for _, row in df.iterrows():
            dataset1 = normalize_dataset_name(row.get("dataset1"))
            dataset2 = normalize_dataset_name(row.get("dataset2"))
            if not dataset1 or not dataset2:
                continue
            mmd2 = row.get("mmd2", np.nan)
            try:
                mmd2 = float(mmd2)
            except (ValueError, TypeError):
                continue

            if has_splits:
                split1 = normalize_dataset_name(row.get("split1"))
                split2 = normalize_dataset_name(row.get("split2"))
                if not allow_unsplit and (not split1 or not split2):
                    continue
                if dataset1 == dataset2 and split1 == split2:
                    continue
                dataset1_id = f"{dataset1}_{split1}" if split1 else dataset1
                dataset2_id = f"{dataset2}_{split2}" if split2 else dataset2
                mmd_lookup[(dataset1_id, dataset2_id)] = mmd2
                mmd_lookup[(dataset2_id, dataset1_id)] = mmd2
            else:
                if not allow_unsplit:
                    continue
                if dataset1 == dataset2:
                    continue
                mmd_lookup[(dataset1, dataset2)] = mmd2
                mmd_lookup[(dataset2, dataset1)] = mmd2

            if allow_unsplit:
                mmd_lookup[(dataset1, dataset2)] = mmd2
                mmd_lookup[(dataset2, dataset1)] = mmd2
    except Exception as exc:
        print(f"Warning: could not read MMD CSV {csv_path}: {exc}")

    return mmd_lookup


def lookup_pair(lookup, train_dataset, benchmark, train_split="train", allow_unsplit=True):
    candidates = [
        (f"{train_dataset}_{train_split}", f"{benchmark}_val"),
        (f"{train_dataset}_{train_split}", f"{benchmark}_test"),
        (f"{train_dataset}_{train_split}", benchmark),
    ]
    if allow_unsplit:
        candidates.extend(
            [
                (train_dataset, benchmark),
                (train_dataset, f"{benchmark}_val"),
                (train_dataset, f"{benchmark}_test"),
            ]
        )
    for key in candidates:
        if key in lookup:
            return lookup[key], key
    return None, candidates[-1]


def lookup_mmd(lookup, train_dataset, benchmark, train_split="train", allow_unsplit=True):
    candidates = [
        (f"{train_dataset}_{train_split}", f"{benchmark}_val"),
        (f"{train_dataset}_{train_split}", f"{benchmark}_test"),
        (f"{train_dataset}_{train_split}", benchmark),
    ]
    if allow_unsplit:
        candidates.extend(
            [
                (train_dataset, benchmark),
                (train_dataset, f"{benchmark}_val"),
                (train_dataset, f"{benchmark}_test"),
            ]
        )
    for key in candidates:
        if key in lookup:
            return lookup[key], key
    return None, candidates[-1]


def resolve_train_dataset(train_dataset, known_datasets=None, strict=False):
    if train_dataset is None:
        return []
    candidates = [train_dataset]
    if strict:
        return candidates
    if train_dataset.startswith("synthetic_"):
        candidates.append("synthetic")
    if train_dataset.startswith("spair_synthetic"):
        if train_dataset == "spair_synthetic":
            candidates.append("spair_synthetic_50_50")
        if known_datasets:
            for dataset in sorted(known_datasets):
                if dataset.startswith("spair_synthetic_"):
                    candidates.append(dataset)
    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def gather_known_datasets(*lookups):
    datasets = set()
    for lookup in lookups:
        if not lookup:
            continue
        for (train_id, eval_id) in lookup.keys():
            datasets.add(strip_split_suffix(train_id))
            datasets.add(strip_split_suffix(eval_id))
    return datasets


def strip_split_suffix(name):
    if name is None:
        return name
    for suffix in ("_train", "_training", "_val", "_validation", "_test"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalize_flow_stats_dataset(name):
    name = normalize_dataset_name(name)
    if not name:
        return name
    if name.endswith("_dino"):
        name = name[: -len("_dino")]
    return name


def _flow_stats_candidates(name, allow_unsplit=True, prefer_splits=None):
    name = normalize_flow_stats_dataset(name)
    if not name:
        return []
    prefer_splits = prefer_splits or []
    base = strip_split_suffix(name)
    candidates = [name]
    for suffix in prefer_splits:
        cand = f"{base}_{suffix}"
        if cand not in candidates:
            candidates.append(cand)
    if allow_unsplit and base != name:
        candidates.append(base)
    if allow_unsplit:
        for suffix in ("train", "val", "test"):
            cand = f"{base}_{suffix}"
            if cand not in candidates:
                candidates.append(cand)
    return candidates


def load_flow_stats(stats_dir):
    lookup = {}
    if not stats_dir:
        return lookup
    path = Path(stats_dir)
    if not path.exists():
        print(f"Warning: flow stats dir not found: {stats_dir}")
        return lookup
    for json_path in sorted(path.glob("*.json")):
        try:
            with json_path.open("r") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: could not read flow stats {json_path}: {exc}")
            continue
        dataset = data.get("dataset")
        if not dataset:
            dataset = json_path.stem
            if dataset.startswith("flow_counts_"):
                dataset = dataset[len("flow_counts_") :]
        dataset = normalize_flow_stats_dataset(dataset)
        if not dataset:
            continue
        if dataset in lookup:
            print(f"Warning: duplicate flow stats for {dataset} (keeping {json_path.name})")
        lookup[dataset] = data
    return lookup


def _extract_flow_density_metrics(stats):
    if not stats:
        return np.nan, np.nan
    n_samples = stats.get("images_seen")
    avg_flows = None
    valid_counts = stats.get("valid_counts")
    if isinstance(valid_counts, dict):
        avg_flows = valid_counts.get("mean")
    if avg_flows is None:
        total_valid = stats.get("total_valid_vectors")
        if total_valid is not None and n_samples:
            avg_flows = total_valid / n_samples
    try:
        n_samples = float(n_samples)
    except (TypeError, ValueError):
        n_samples = np.nan
    try:
        avg_flows = float(avg_flows)
    except (TypeError, ValueError):
        avg_flows = np.nan
    return n_samples, avg_flows


def lookup_flow_stats(lookup, dataset, allow_unsplit=True, prefer_splits=None):
    candidates = _flow_stats_candidates(
        dataset, allow_unsplit=allow_unsplit, prefer_splits=prefer_splits
    )
    for candidate in candidates:
        if candidate in lookup:
            return lookup[candidate], candidate
    return None, candidates[-1] if candidates else None


def _safe_log_series(series, eps):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return np.log(np.maximum(values, float(eps)))


def _is_coverage_column(name: str) -> bool:
    lower = str(name).lower()
    return (
        "coverage" in lower
        or "over_eval_recall" in lower
        or "over_train_precision" in lower
    )


def add_flow_density_features(
    df,
    flow_stats_lookup,
    log_eps=1e-6,
    allow_unsplit=True,
    add_interactions=False,
):
    if df.empty or not flow_stats_lookup:
        return df, {}
    df = df.copy()
    missing = defaultdict(int)

    train_names = df["train_dataset"].dropna().unique() if "train_dataset" in df.columns else []
    eval_names = df["benchmark"].dropna().unique() if "benchmark" in df.columns else []

    train_samples = {}
    train_flows = {}
    eval_samples = {}
    eval_flows = {}

    for name in train_names:
        stats, _ = lookup_flow_stats(
            flow_stats_lookup, name, allow_unsplit=allow_unsplit, prefer_splits=["train"]
        )
        if stats is None:
            missing[("train_dataset", name)] += 1
            train_samples[name] = np.nan
            train_flows[name] = np.nan
        else:
            n_samples, avg_flows = _extract_flow_density_metrics(stats)
            train_samples[name] = n_samples
            train_flows[name] = avg_flows

    for name in eval_names:
        stats, _ = lookup_flow_stats(
            flow_stats_lookup, name, allow_unsplit=allow_unsplit, prefer_splits=["val", "test"]
        )
        if stats is None:
            missing[("benchmark", name)] += 1
            eval_samples[name] = np.nan
            eval_flows[name] = np.nan
        else:
            n_samples, avg_flows = _extract_flow_density_metrics(stats)
            eval_samples[name] = n_samples
            eval_flows[name] = avg_flows

    if "train_dataset" in df.columns:
        df["n_samples_train"] = df["train_dataset"].map(train_samples)
        df["avg_flows_train"] = df["train_dataset"].map(train_flows)
    if "benchmark" in df.columns:
        df["n_samples_eval"] = df["benchmark"].map(eval_samples)
        df["avg_flows_eval"] = df["benchmark"].map(eval_flows)

    if "n_samples_train" in df.columns:
        df["log_n_samples_train"] = _safe_log_series(df["n_samples_train"], log_eps)
    if "n_samples_eval" in df.columns:
        df["log_n_samples_eval"] = _safe_log_series(df["n_samples_eval"], log_eps)
    if "avg_flows_train" in df.columns:
        df["log_avg_flows_train"] = _safe_log_series(df["avg_flows_train"], log_eps)
    if "avg_flows_eval" in df.columns:
        df["log_avg_flows_eval"] = _safe_log_series(df["avg_flows_eval"], log_eps)

    if add_interactions and "log_avg_flows_eval" in df.columns:
        for col in df.columns:
            if _is_coverage_column(col):
                df[f"{col}_x_log_avg_flows_eval"] = df[col] * df["log_avg_flows_eval"]

    return df, missing


def logit(value, eps=1e-6):
    if value is None or pd.isna(value):
        return np.nan
    clipped = min(max(float(value), eps), 1.0 - eps)
    return float(np.log(clipped / (1.0 - clipped)))


def add_logit_columns(df, columns, suffix="_logit"):
    for col in columns:
        if col in df.columns:
            df[f"{col}{suffix}"] = df[col].apply(logit)
    return df


def normalize_distances_by_radius(df, eps, radius_floor=0.0):
    df = df.copy()
    floor = max(float(radius_floor), 0.0)
    prefixes = ("flow", "resnet", "dino", "hof")
    suffixes = (
        "eval_to_train_mean_dist",
        "eval_to_train_median_dist",
        "eval_to_train_p90_dist",
        "train_to_eval_mean_dist",
        "train_to_eval_median_dist",
        "train_to_eval_p90_dist",
    )
    for prefix in prefixes:
        radius_col = f"{prefix}_radius_train"
        if radius_col not in df.columns:
            continue
        denom = np.maximum(df[radius_col].astype(float), floor) + float(eps)
        for suffix in suffixes:
            col = f"{prefix}_{suffix}"
            if col in df.columns:
                df[col] = df[col].astype(float) / denom
    return df


def add_distance_ratio_features(df, eps, radius_floor=0.0):
    df = df.copy()
    floor = max(float(radius_floor), 0.0)
    prefixes = ("flow", "resnet", "dino", "hof")
    stats = ("mean", "median", "p90")
    for prefix in prefixes:
        radius_train_col = f"{prefix}_radius_train"
        radius_eval_col = f"{prefix}_radius_eval"
        radius_train = None
        radius_eval = None
        if radius_train_col in df.columns:
            radius_train = np.maximum(df[radius_train_col].astype(float), floor) + float(eps)
        if radius_eval_col in df.columns:
            radius_eval = np.maximum(df[radius_eval_col].astype(float), floor) + float(eps)

        for stat in stats:
            train_col = f"{prefix}_train_to_eval_{stat}_dist"
            eval_col = f"{prefix}_eval_to_train_{stat}_dist"
            if radius_eval is not None and train_col in df.columns:
                df[f"{train_col}_over_radius_eval"] = df[train_col].astype(float) / radius_eval
            if radius_train is not None and train_col in df.columns:
                df[f"{train_col}_over_radius_train"] = (
                    df[train_col].astype(float) / radius_train
                )
            if radius_train is not None and eval_col in df.columns:
                df[f"{eval_col}_over_radius_train"] = df[eval_col].astype(float) / radius_train
            if radius_eval is not None and eval_col in df.columns:
                df[f"{eval_col}_over_radius_eval"] = df[eval_col].astype(float) / radius_eval
            if train_col in df.columns and eval_col in df.columns:
                denom = df[train_col].astype(float) + float(eps)
                df[f"{prefix}_{stat}_dist_asymmetry"] = (
                    df[eval_col].astype(float) + float(eps)
                ) / denom
    return df


def transform_distance_ratio_features(df, mode):
    if mode == "none":
        return df
    df = df.copy()
    ratio_cols = [col for col in df.columns if "_dist_over_radius_" in col]
    if not ratio_cols:
        return df
    if mode == "log1p":
        for col in ratio_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            values = values.where(values >= 0)
            df[col] = np.log1p(values)
        return df
    return df


def transform_radius_features(df, mode, eps):
    if mode == "keep":
        return df
    df = df.copy()
    prefixes = ("flow", "resnet", "dino", "hof")
    radius_cols = ("radius_train", "radius_eval", "radius_quantile")
    if mode == "drop":
        for prefix in prefixes:
            for suffix in radius_cols:
                col = f"{prefix}_{suffix}"
                if col in df.columns:
                    df = df.drop(columns=[col])
        return df
    if mode == "log":
        for prefix in prefixes:
            for suffix in radius_cols:
                col = f"{prefix}_{suffix}"
                if col in df.columns:
                    values = df[col].astype(float)
                    df[col] = np.log(np.maximum(values, float(eps)))
        return df
    return df


def normalize_by_group(df, columns, group_col, mode):
    if mode == "none" or not columns:
        return df
    if group_col not in df.columns:
        return df

    df = df.copy()
    grouped = df.groupby(group_col)[columns]
    means = grouped.transform("mean")
    global_mean = df[columns].mean()
    means = means.fillna(global_mean)

    if mode == "center":
        df[columns] = df[columns] - means
        return df

    stds = grouped.transform(lambda x: x.std(ddof=0))
    global_std = df[columns].std(ddof=0).replace(0, 1.0)
    stds = stds.replace(0, 1.0).fillna(global_std)
    df[columns] = (df[columns] - means) / stds
    return df


def normalize_predictors_within_benchmark(train_df, test_df, predictors, mode):
    if mode == "none" or not predictors:
        return train_df, test_df
    if "benchmark" not in train_df.columns or "benchmark" not in test_df.columns:
        return train_df, test_df

    train_df = train_df.copy()
    test_df = test_df.copy()
    combined = pd.concat(
        [train_df.assign(_split="train"), test_df.assign(_split="test")],
        axis=0,
    )
    if combined.empty:
        return train_df, test_df

    if mode == "rank":
        group_sizes = combined.groupby("benchmark", dropna=False)["benchmark"].transform("size")
        denom = (group_sizes - 1.0).clip(lower=1.0)
        for col in predictors:
            if col not in combined.columns:
                continue
            ranks = combined.groupby("benchmark", dropna=False)[col].rank(
                method="average", ascending=True
            )
            combined[col] = (ranks - 1.0) / denom
    elif mode == "zscore":
        means = combined.groupby("benchmark", dropna=False)[predictors].transform("mean")
        stds = combined.groupby("benchmark", dropna=False)[predictors].transform(
            lambda x: x.std(ddof=0)
        )
        stds = stds.replace(0, 1.0)
        combined[predictors] = (combined[predictors] - means) / stds
    else:
        return train_df, test_df

    train_df = combined[combined["_split"] == "train"].drop(columns=["_split"])
    test_df = combined[combined["_split"] == "test"].drop(columns=["_split"])
    return train_df, test_df


def permute_target_within_group(df, target, group_col, rng):
    df = df.copy()
    if target not in df.columns:
        return df
    if group_col and group_col in df.columns:
        for _, sub in df.groupby(group_col):
            values = sub[target].to_numpy(dtype=float, copy=True)
            rng.shuffle(values)
            df.loc[sub.index, target] = values
        return df
    values = df[target].to_numpy(dtype=float, copy=True)
    rng.shuffle(values)
    df[target] = values
    return df


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _safe_max_abs_delta(a, b):
    vals = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    mask = np.isfinite(vals)
    if mask.sum() == 0:
        return np.nan
    return float(np.max(vals[mask]))


def write_predictor_colinearity(df, predictors, out_path, method="pearson"):
    if df.empty or not predictors:
        return
    cols = [p for p in predictors if p in df.columns]
    if len(cols) < 2:
        return
    sub = df[cols].dropna()
    if len(sub) < 3:
        return
    corr = sub.corr(method=method)
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    corr = corr.mask(mask)
    corr.to_csv(out_path)


def write_distance_diagnostics(df, out_path):
    lines = []
    prefixes = ("flow", "resnet", "dino", "hof")
    directions = ("train_to_eval", "eval_to_train")
    for prefix in prefixes:
        for direction in directions:
            mean_col = f"{prefix}_{direction}_mean_dist"
            median_col = f"{prefix}_{direction}_median_dist"
            p90_col = f"{prefix}_{direction}_p90_dist"
            if mean_col not in df.columns or median_col not in df.columns or p90_col not in df.columns:
                continue
            mean_vals = df[mean_col]
            median_vals = df[median_col]
            p90_vals = df[p90_col]
            lines.append(f"{prefix} {direction}:")
            lines.append(f"  max_abs(mean-median)={_safe_max_abs_delta(mean_vals, median_vals):.6f}")
            lines.append(f"  max_abs(mean-p90)={_safe_max_abs_delta(mean_vals, p90_vals):.6f}")
            lines.append(f"  max_abs(median-p90)={_safe_max_abs_delta(median_vals, p90_vals):.6f}")
            lines.append(f"  corr(mean,median)={_safe_corr(mean_vals, median_vals):.6f}")
            lines.append(f"  corr(mean,p90)={_safe_corr(mean_vals, p90_vals):.6f}")
            lines.append(f"  corr(median,p90)={_safe_corr(median_vals, p90_vals):.6f}")
    if not lines:
        lines.append("No distance diagnostics available.")
    out_path.write_text("\n".join(lines))


def _write_run_metadata(out_dir: Path, predictors: List[str], args: argparse.Namespace) -> None:
    resolved_model = args.prediction_model or args.linear_model
    predictor_list = list(predictors or [])
    n_total = len(predictor_list)
    n_enc_main = len([p for p in predictor_list if p.startswith("enc_")])
    n_mf_main = len([p for p in predictor_list if p.startswith("mf_")])
    n_enc_inter = len([p for p in predictor_list if "__enc_" in p])
    n_mf_inter = len([p for p in predictor_list if "__mf_" in p])
    n_base = n_total - n_enc_main - n_mf_main - n_enc_inter - n_mf_inter
    meta = {
        "target": args.target,
        "prediction_target": args.prediction_target,
        "predictors": predictors,
        "n_predictors": n_total,
        "n_predictors_base": n_base,
        "n_predictors_encoder_main_effects": n_enc_main,
        "n_predictors_model_family_main_effects": n_mf_main,
        "n_predictors_encoder_interactions": n_enc_inter,
        "n_predictors_model_family_interactions": n_mf_inter,
        "rank_target": bool(args.rank_target),
        "rank_target_source": args.rank_target_source,
        "rank_target_group": args.rank_target_group,
        "ranking_group": args.ranking_group,
        "ranking_context_cols": args.ranking_context_cols,
        "pairwise_group_cols": args.pairwise_group_cols,
        "cv_residualize_target_by_context": bool(
            getattr(args, "cv_residualize_target_by_context", False)
        ),
        "cv_residual_context_cols": getattr(args, "cv_residual_context_cols", ""),
        "cv_residual_eval_space": getattr(args, "cv_residual_eval_space", "residual"),
        "cv_residual_target_transform": getattr(args, "cv_residual_target_transform", "residual"),
        "cv_residual_target_std_eps": getattr(args, "cv_residual_target_std_eps", 1e-9),
        "cv_fewshot_context_calibration": bool(
            getattr(args, "cv_fewshot_context_calibration", False)
        ),
        "cv_fewshot_context_calibration_cols": getattr(
            args, "cv_fewshot_context_calibration_cols", ""
        ),
        "cv_fewshot_context_calibration_std_eps": getattr(
            args, "cv_fewshot_context_calibration_std_eps", 1e-9
        ),
        "cv_fewshot_context_calibration_min_group_size": getattr(
            args, "cv_fewshot_context_calibration_min_group_size", 2
        ),
        "cv_fewshot_context_calibration_backoff": bool(
            getattr(args, "cv_fewshot_context_calibration_backoff", True)
        ),
        "cv_fewshot_context_calibration_k": int(
            getattr(args, "cv_fewshot_context_calibration_k", 0)
        ),
        "cv_fewshot_context_calibration_seed": int(
            getattr(args, "cv_fewshot_context_calibration_seed", 0)
        ),
        "loto_single_predictor_baselines": bool(
            getattr(args, "loto_single_predictor_baselines", True)
        ),
        "jointood_single_predictor_baselines": bool(
            getattr(args, "jointood_single_predictor_baselines", True)
        ),
        "rank_target_with_encoder": bool(args.rank_target_with_encoder),
        "rank_target_with_model": bool(args.rank_target_with_model),
        "model": resolved_model,
        "linear_model": args.linear_model,
        "prediction_model": args.prediction_model,
        "joint_ood_holdout": bool(getattr(args, "joint_ood_holdout", False)),
        "collapse_cv_cells": bool(getattr(args, "collapse_cv_cells", False)),
        "ridge_alpha": args.ridge_alpha,
        "standardize": bool(args.standardize),
        "fit_sample_weighting": args.fit_sample_weighting,
        "fit_balance_real_synth": bool(args.fit_balance_real_synth),
        "cv_repeat_aggregation": getattr(args, "cv_repeat_aggregation", "none"),
        "overall_aggregation": args.overall_aggregation,
        "logit_coverage": bool(args.logit_coverage),
        "custom_interactions": getattr(args, "custom_interactions", ""),
        "use_flow_eps_predictors": bool(getattr(args, "use_flow_eps_predictors", False)),
        "use_flow_eps_weighted_predictors": bool(
            getattr(args, "use_flow_eps_weighted_predictors", False)
        ),
        "flow_eps_rings": bool(getattr(args, "flow_eps_rings", False)),
        "use_flow_density_predictors": bool(
            getattr(args, "use_flow_density_predictors", False)
        ),
        "flow_density_interactions": bool(
            getattr(args, "flow_density_interactions", False)
        ),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def build_auc_feature_table(
    auc_df,
    flow_lookup,
    resnet_lookup,
    hof_lookup,
    variogram_lookup,
    flow_mmd_lookup,
    feature_mmd_lookup,
    logit_coverage=False,
    dino_lookup=None,
    dino_mmd_lookup=None,
    strict_dataset_match=False,
    allow_unsplit_coverage=True,
    allow_unsplit_mmd=True,
    distance_radius_norm="none",
    distance_ratio_transform="none",
    radius_transform="keep",
    radius_eps=1e-6,
    radius_floor=0.0,
    rename_coverage=False,
):
    rows = []
    missing = defaultdict(int)
    known_datasets = gather_known_datasets(
        flow_lookup,
        resnet_lookup,
        hof_lookup,
        variogram_lookup,
        flow_mmd_lookup,
        feature_mmd_lookup,
        dino_lookup,
        dino_mmd_lookup,
    )

    for row in auc_df.to_dict(orient="records"):
        train_dataset = normalize_dataset_name(row.get("train_dataset"))
        benchmark = normalize_dataset_name(row.get("benchmark"))
        if not train_dataset or not benchmark:
            continue

        flow_metrics = None
        resnet_metrics = None
        hof_metrics = None
        dino_metrics = None
        variogram_metrics = None
        flow_mmd = None
        feature_mmd = None
        dino_mmd = None
        resolved_train = train_dataset

        for candidate in resolve_train_dataset(
            train_dataset, known_datasets, strict=strict_dataset_match
        ):
            flow_metrics, flow_key = lookup_pair(
                flow_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_coverage
            )
            resnet_metrics, resnet_key = lookup_pair(
                resnet_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_coverage
            )
            if hof_lookup:
                hof_metrics, _ = lookup_pair(
                    hof_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_coverage
                )
            if dino_lookup:
                dino_metrics, _ = lookup_pair(
                    dino_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_coverage
                )
            if variogram_lookup:
                variogram_metrics, _ = lookup_pair(
                    variogram_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_coverage
                )
            flow_mmd, _ = lookup_mmd(
                flow_mmd_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_mmd
            )
            feature_mmd, _ = lookup_mmd(
                feature_mmd_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_mmd
            )
            if dino_mmd_lookup:
                dino_mmd, _ = lookup_mmd(
                    dino_mmd_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_mmd
                )
            if (
                flow_metrics is not None
                or resnet_metrics is not None
                or hof_metrics is not None
                or variogram_metrics is not None
            ):
                resolved_train = candidate
                break

        if flow_metrics is None:
            missing[("flow", train_dataset, benchmark)] += 1
        if resnet_metrics is None:
            missing[("resnet", train_dataset, benchmark)] += 1
        if hof_lookup is not None and hof_metrics is None:
            missing[("hof", train_dataset, benchmark)] += 1
        if dino_lookup is not None and dino_metrics is None:
            missing[("dino", train_dataset, benchmark)] += 1
        if flow_mmd is None:
            missing[("flow_mmd", train_dataset, benchmark)] += 1
        if feature_mmd is None:
            missing[("feature_mmd", train_dataset, benchmark)] += 1
        if dino_mmd_lookup is not None and dino_mmd is None:
            missing[("dino_mmd", train_dataset, benchmark)] += 1
        if variogram_lookup is not None and variogram_metrics is None:
            missing[("variogram", train_dataset, benchmark)] += 1

        row.update({
            "train_dataset": resolved_train,
            "benchmark": benchmark,
            "flow_train_to_eval_coverage": (
                flow_metrics["train_to_eval_coverage"] if flow_metrics else np.nan
            ),
            "flow_eval_to_train_coverage": (
                flow_metrics["eval_to_train_coverage"] if flow_metrics else np.nan
            ),
            "flow_outside_mass": flow_metrics["outside"] if flow_metrics else np.nan,
            "flow_k": flow_metrics["k"] if flow_metrics else np.nan,
            "flow_radius_quantile": flow_metrics["radius_quantile"] if flow_metrics else np.nan,
            "flow_radius_train": flow_metrics["radius_train"] if flow_metrics else np.nan,
            "flow_radius_eval": flow_metrics["radius_eval"] if flow_metrics else np.nan,
            "flow_eval_to_train_mean_dist": (
                flow_metrics["mean_nn_eval_to_train"] if flow_metrics else np.nan
            ),
            "flow_eval_to_train_median_dist": (
                flow_metrics["median_nn_eval_to_train"] if flow_metrics else np.nan
            ),
            "flow_eval_to_train_p90_dist": (
                flow_metrics["p90_nn_eval_to_train"] if flow_metrics else np.nan
            ),
            "flow_train_to_eval_mean_dist": (
                flow_metrics["mean_nn_train_to_eval"] if flow_metrics else np.nan
            ),
            "flow_train_to_eval_median_dist": (
                flow_metrics["median_nn_train_to_eval"] if flow_metrics else np.nan
            ),
            "flow_train_to_eval_p90_dist": (
                flow_metrics["p90_nn_train_to_eval"] if flow_metrics else np.nan
            ),
            "flow_eval_to_train_kl_div": (
                flow_metrics["kl_eval_to_train"] if flow_metrics else np.nan
            ),
            "flow_train_to_eval_kl_div": (
                flow_metrics["kl_train_to_eval"] if flow_metrics else np.nan
            ),
            "flow_eval_to_train_kl_div_hist": (
                flow_metrics.get("kl_eval_to_train_hist", np.nan) if flow_metrics else np.nan
            ),
            "flow_train_to_eval_kl_div_hist": (
                flow_metrics.get("kl_train_to_eval_hist", np.nan) if flow_metrics else np.nan
            ),
            "flow_eval_to_train_kl_div_hist_log1p_linear": (
                flow_metrics.get("kl_eval_to_train_hist_log1p_linear", np.nan)
                if flow_metrics
                else np.nan
            ),
            "flow_train_to_eval_kl_div_hist_log1p_linear": (
                flow_metrics.get("kl_train_to_eval_hist_log1p_linear", np.nan)
                if flow_metrics
                else np.nan
            ),
            "resnet_train_to_eval_coverage": (
                resnet_metrics["train_to_eval_coverage"] if resnet_metrics else np.nan
            ),
            "resnet_eval_to_train_coverage": (
                resnet_metrics["eval_to_train_coverage"] if resnet_metrics else np.nan
            ),
            "resnet_outside_mass": resnet_metrics["outside"] if resnet_metrics else np.nan,
            "resnet_k": resnet_metrics["k"] if resnet_metrics else np.nan,
            "resnet_radius_quantile": resnet_metrics["radius_quantile"] if resnet_metrics else np.nan,
            "resnet_radius_train": resnet_metrics["radius_train"] if resnet_metrics else np.nan,
            "resnet_radius_eval": resnet_metrics["radius_eval"] if resnet_metrics else np.nan,
            "resnet_eval_to_train_mean_dist": (
                resnet_metrics["mean_nn_eval_to_train"] if resnet_metrics else np.nan
            ),
            "resnet_eval_to_train_median_dist": (
                resnet_metrics["median_nn_eval_to_train"] if resnet_metrics else np.nan
            ),
            "resnet_eval_to_train_p90_dist": (
                resnet_metrics["p90_nn_eval_to_train"] if resnet_metrics else np.nan
            ),
            "resnet_train_to_eval_mean_dist": (
                resnet_metrics["mean_nn_train_to_eval"] if resnet_metrics else np.nan
            ),
            "resnet_train_to_eval_median_dist": (
                resnet_metrics["median_nn_train_to_eval"] if resnet_metrics else np.nan
            ),
            "resnet_train_to_eval_p90_dist": (
                resnet_metrics["p90_nn_train_to_eval"] if resnet_metrics else np.nan
            ),
            "resnet_eval_to_train_kl_div": (
                resnet_metrics["kl_eval_to_train"] if resnet_metrics else np.nan
            ),
            "resnet_train_to_eval_kl_div": (
                resnet_metrics["kl_train_to_eval"] if resnet_metrics else np.nan
            ),
            "resnet_eval_to_train_kl_div_hist": (
                resnet_metrics.get("kl_eval_to_train_hist", np.nan) if resnet_metrics else np.nan
            ),
            "resnet_train_to_eval_kl_div_hist": (
                resnet_metrics.get("kl_train_to_eval_hist", np.nan) if resnet_metrics else np.nan
            ),
            "resnet_eval_to_train_kl_div_hist_log1p_linear": (
                resnet_metrics.get("kl_eval_to_train_hist_log1p_linear", np.nan)
                if resnet_metrics
                else np.nan
            ),
            "resnet_train_to_eval_kl_div_hist_log1p_linear": (
                resnet_metrics.get("kl_train_to_eval_hist_log1p_linear", np.nan)
                if resnet_metrics
                else np.nan
            ),
            "hof_train_to_eval_coverage": (
                hof_metrics["train_to_eval_coverage"] if hof_metrics else np.nan
            ),
            "hof_eval_to_train_coverage": (
                hof_metrics["eval_to_train_coverage"] if hof_metrics else np.nan
            ),
            "hof_outside_mass": hof_metrics["outside"] if hof_metrics else np.nan,
            "hof_k": hof_metrics["k"] if hof_metrics else np.nan,
            "hof_radius_quantile": hof_metrics["radius_quantile"] if hof_metrics else np.nan,
            "hof_radius_train": hof_metrics["radius_train"] if hof_metrics else np.nan,
            "hof_radius_eval": hof_metrics["radius_eval"] if hof_metrics else np.nan,
            "hof_eval_to_train_mean_dist": (
                hof_metrics["mean_nn_eval_to_train"] if hof_metrics else np.nan
            ),
            "hof_eval_to_train_median_dist": (
                hof_metrics["median_nn_eval_to_train"] if hof_metrics else np.nan
            ),
            "hof_eval_to_train_p90_dist": (
                hof_metrics["p90_nn_eval_to_train"] if hof_metrics else np.nan
            ),
            "hof_train_to_eval_mean_dist": (
                hof_metrics["mean_nn_train_to_eval"] if hof_metrics else np.nan
            ),
            "hof_train_to_eval_median_dist": (
                hof_metrics["median_nn_train_to_eval"] if hof_metrics else np.nan
            ),
            "hof_train_to_eval_p90_dist": (
                hof_metrics["p90_nn_train_to_eval"] if hof_metrics else np.nan
            ),
            "hof_eval_to_train_kl_div": (
                hof_metrics["kl_eval_to_train"] if hof_metrics else np.nan
            ),
            "hof_train_to_eval_kl_div": (
                hof_metrics["kl_train_to_eval"] if hof_metrics else np.nan
            ),
            "hof_eval_to_train_kl_div_hist": (
                hof_metrics.get("kl_eval_to_train_hist", np.nan) if hof_metrics else np.nan
            ),
            "hof_train_to_eval_kl_div_hist": (
                hof_metrics.get("kl_train_to_eval_hist", np.nan) if hof_metrics else np.nan
            ),
            "hof_eval_to_train_kl_div_hist_log1p_linear": (
                hof_metrics.get("kl_eval_to_train_hist_log1p_linear", np.nan)
                if hof_metrics
                else np.nan
            ),
            "hof_train_to_eval_kl_div_hist_log1p_linear": (
                hof_metrics.get("kl_train_to_eval_hist_log1p_linear", np.nan)
                if hof_metrics
                else np.nan
            ),
            "hof_density_l2": (
                hof_metrics.get("hof_density_l2", np.nan) if hof_metrics else np.nan
            ),
            "hof_density_l1": (
                hof_metrics.get("hof_density_l1", np.nan) if hof_metrics else np.nan
            ),
            "hof_density_cosine": (
                hof_metrics.get("hof_density_cosine", np.nan) if hof_metrics else np.nan
            ),
            "dino_train_to_eval_coverage": (
                dino_metrics["train_to_eval_coverage"] if dino_metrics else np.nan
            ),
            "dino_eval_to_train_coverage": (
                dino_metrics["eval_to_train_coverage"] if dino_metrics else np.nan
            ),
            "dino_outside_mass": dino_metrics["outside"] if dino_metrics else np.nan,
            "dino_k": dino_metrics["k"] if dino_metrics else np.nan,
            "dino_radius_quantile": dino_metrics["radius_quantile"] if dino_metrics else np.nan,
            "dino_radius_train": dino_metrics["radius_train"] if dino_metrics else np.nan,
            "dino_radius_eval": dino_metrics["radius_eval"] if dino_metrics else np.nan,
            "dino_eval_to_train_mean_dist": (
                dino_metrics["mean_nn_eval_to_train"] if dino_metrics else np.nan
            ),
            "dino_eval_to_train_median_dist": (
                dino_metrics["median_nn_eval_to_train"] if dino_metrics else np.nan
            ),
            "dino_eval_to_train_p90_dist": (
                dino_metrics["p90_nn_eval_to_train"] if dino_metrics else np.nan
            ),
            "dino_train_to_eval_mean_dist": (
                dino_metrics["mean_nn_train_to_eval"] if dino_metrics else np.nan
            ),
            "dino_train_to_eval_median_dist": (
                dino_metrics["median_nn_train_to_eval"] if dino_metrics else np.nan
            ),
            "dino_train_to_eval_p90_dist": (
                dino_metrics["p90_nn_train_to_eval"] if dino_metrics else np.nan
            ),
            "dino_eval_to_train_kl_div": (
                dino_metrics["kl_eval_to_train"] if dino_metrics else np.nan
            ),
            "dino_train_to_eval_kl_div": (
                dino_metrics["kl_train_to_eval"] if dino_metrics else np.nan
            ),
            "dino_eval_to_train_kl_div_hist": (
                dino_metrics.get("kl_eval_to_train_hist", np.nan) if dino_metrics else np.nan
            ),
            "dino_train_to_eval_kl_div_hist": (
                dino_metrics.get("kl_train_to_eval_hist", np.nan) if dino_metrics else np.nan
            ),
            "dino_eval_to_train_kl_div_hist_log1p_linear": (
                dino_metrics.get("kl_eval_to_train_hist_log1p_linear", np.nan)
                if dino_metrics
                else np.nan
            ),
            "dino_train_to_eval_kl_div_hist_log1p_linear": (
                dino_metrics.get("kl_train_to_eval_hist_log1p_linear", np.nan)
                if dino_metrics
                else np.nan
            ),
            "variogram_train_auc": (
                variogram_metrics.get("train_auc", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_eval_auc": (
                variogram_metrics.get("eval_auc", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_train_auc_norm": (
                variogram_metrics.get("train_auc_norm", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_eval_auc_norm": (
                variogram_metrics.get("eval_auc_norm", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_auc_diff": (
                variogram_metrics.get("auc_diff", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_auc_diff_norm": (
                variogram_metrics.get("auc_diff_norm", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_curve_l1": (
                variogram_metrics.get("curve_l1", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_curve_l2": (
                variogram_metrics.get("curve_l2", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_curve_corr": (
                variogram_metrics.get("curve_corr", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_overlap_bins": (
                variogram_metrics.get("overlap_bins", np.nan) if variogram_metrics else np.nan
            ),
            "variogram_total_bins": (
                variogram_metrics.get("total_bins", np.nan) if variogram_metrics else np.nan
            ),
            "flow_mmd": flow_mmd,
            "feature_mmd": feature_mmd,
            "dino_mmd": dino_mmd,
        })

        if flow_metrics:
            for key, value in flow_metrics.items():
                if (
                    key.startswith("eval_to_train_eps")
                    or key.startswith("train_to_eval_eps")
                    or key.startswith("eval_to_train_auc")
                    or key.startswith("train_to_eval_auc")
                    or key.startswith("eval_to_train_eps_at")
                    or key.startswith("train_to_eval_eps_at")
                ):
                    row[f"flow_{key}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_distance_ratio_features(df, radius_eps, radius_floor)
    if distance_ratio_transform != "none":
        df = transform_distance_ratio_features(df, distance_ratio_transform)
    if distance_radius_norm == "divide":
        df = normalize_distances_by_radius(df, radius_eps, radius_floor)
    if radius_transform != "keep":
        df = transform_radius_features(df, radius_transform, radius_eps)
    if logit_coverage:
        df = add_logit_columns(
            df,
            [
                "flow_train_to_eval_coverage",
                "flow_eval_to_train_coverage",
                "flow_outside_mass",
                "resnet_train_to_eval_coverage",
                "resnet_eval_to_train_coverage",
                "resnet_outside_mass",
                "dino_train_to_eval_coverage",
                "dino_eval_to_train_coverage",
                "dino_outside_mass",
                "hof_train_to_eval_coverage",
                "hof_eval_to_train_coverage",
                "hof_outside_mass",
            ],
        )
    if rename_coverage:
        df = add_explicit_coverage_columns(df)
    return df, missing


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    x = x[finite]
    y = y[finite]
    x = x - np.mean(x)
    y = y - np.mean(y)
    x_scale = float(np.max(np.abs(x))) if x.size else 0.0
    y_scale = float(np.max(np.abs(y))) if y.size else 0.0
    if x_scale == 0.0 or y_scale == 0.0:
        return np.nan
    x = x / x_scale
    y = y / y_scale
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    x = x[finite]
    y = y[finite]
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def kendall_tau_b(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.nan
    xx = x[finite]
    yy = y[finite]
    n = int(len(xx))
    if n < 2:
        return np.nan

    n_conc = 0
    n_disc = 0
    n_tie_x = 0
    n_tie_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xx[i] - xx[j]
            dy = yy[i] - yy[j]
            tie_x = dx == 0.0
            tie_y = dy == 0.0
            if tie_x:
                n_tie_x += 1
            if tie_y:
                n_tie_y += 1
            if tie_x or tie_y:
                continue
            if dx * dy > 0.0:
                n_conc += 1
            elif dx * dy < 0.0:
                n_disc += 1
    n0 = n * (n - 1) / 2.0
    denom = np.sqrt(max(n0 - n_tie_x, 0.0) * max(n0 - n_tie_y, 0.0))
    if denom == 0.0:
        return np.nan
    return float((n_conc - n_disc) / denom)


def pairwise_cindex(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite = np.isfinite(y_true) & np.isfinite(y_pred)
    if finite.sum() < 2:
        return np.nan
    yt = y_true[finite]
    yp = y_pred[finite]
    n = int(len(yt))
    if n < 2:
        return np.nan

    comparable = 0
    score = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dy_true = yt[i] - yt[j]
            if dy_true == 0.0:
                continue
            comparable += 1
            dy_pred = yp[i] - yp[j]
            if dy_pred == 0.0:
                score += 0.5
            elif dy_true * dy_pred > 0.0:
                score += 1.0
    if comparable == 0:
        return np.nan
    return float(score / comparable)


def mae_rmse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() == 0:
        return np.nan, np.nan
    diff = x[finite] - y[finite]
    mae = float(np.mean(np.abs(diff)))
    scale = float(np.max(np.abs(diff))) if diff.size else 0.0
    if not np.isfinite(scale):
        rmse = np.nan
    elif scale == 0.0:
        rmse = 0.0
    else:
        rmse = float(scale * np.sqrt(np.mean((diff / scale) ** 2)))
    return mae, rmse


def safe_std(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    scale = float(np.max(np.abs(finite)))
    if not np.isfinite(scale):
        return np.nan
    if scale == 0.0:
        return 0.0
    normalized = finite / scale
    centered = normalized - float(np.mean(normalized))
    return float(scale * np.sqrt(np.mean(centered * centered)))


def inv_one_plus_exp(values):
    values = np.asarray(values, dtype=float)
    out = np.empty_like(values)
    pos = values >= 0
    if np.any(pos):
        exp_neg = np.exp(-values[pos])
        out[pos] = exp_neg / (1.0 + exp_neg)
    if np.any(~pos):
        exp_pos = np.exp(values[~pos])
        out[~pos] = 1.0 / (1.0 + exp_pos)
    return out


def fit_linear_model(
    train_df,
    predictors,
    target,
    standardize=True,
    model="ols",
    ridge_alpha=1.0,
    min_std=0.0,
    sample_weight=None,
):
    X = train_df[predictors].to_numpy(dtype=float)
    y = train_df[target].to_numpy(dtype=float)
    w = None
    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        if w.shape[0] != X.shape[0]:
            w = None
        else:
            w[~np.isfinite(w)] = 0.0
            w = np.where(w > 0.0, w, 0.0)
            if float(w.sum()) <= 0.0:
                w = None
    mean = np.zeros(X.shape[1])
    std = np.ones(X.shape[1])
    if standardize:
        if w is None:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
        else:
            w_norm = w / float(w.sum())
            mean = np.sum(X * w_norm[:, None], axis=0)
            var = np.sum(((X - mean) ** 2) * w_norm[:, None], axis=0)
            std = np.sqrt(np.maximum(var, 0.0))
        if min_std > 0:
            std = np.where(std < min_std, float(min_std), std)
        std[std == 0] = 1.0
        X = (X - mean) / std
    X = np.column_stack([np.ones(len(X)), X])
    if w is not None:
        sqrt_w = np.sqrt(w)
        X_fit = X * sqrt_w[:, None]
        y_fit = y * sqrt_w
    else:
        X_fit = X
        y_fit = y
    if model == "ridge":
        alpha = float(ridge_alpha)
        penalty = np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        xtx = X_fit.T @ X_fit
        xty = X_fit.T @ y_fit
        try:
            coef = np.linalg.solve(xtx + alpha * penalty, xty)
        except np.linalg.LinAlgError:
            coef, _, _, _ = np.linalg.lstsq(xtx + alpha * penalty, xty, rcond=None)
    else:
        coef, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    return coef, mean, std


def run_stability_selection(
    df,
    predictors,
    target,
    n_bootstrap=100,
    lasso_alpha_range=None,
    threshold=0.7,
    output_path=None,
):
    """
    Stability selection: run regularized regression on bootstrap samples
    and count how often each predictor is selected.
    
    Returns DataFrame with stability scores for each predictor.
    """
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.exceptions import ConvergenceWarning
        import warnings
    except ImportError:
        print("Warning: scikit-learn not available, skipping stability selection")
        return pd.DataFrame({
            'predictor': predictors,
            'stability_score': np.zeros(len(predictors)),
            'stable': [False] * len(predictors),
        })
    
    if lasso_alpha_range is None:
        lasso_alpha_range = np.logspace(-4, 0, 50)
    
    complete_df = filter_complete_rows(df, predictors, target)
    if complete_df.empty or len(complete_df) < 10:
        return pd.DataFrame({
            'predictor': predictors,
            'stability_score': np.zeros(len(predictors)),
            'stable': [False] * len(predictors),
        })
    
    X = complete_df[predictors].values
    y = complete_df[target].values
    n_samples, n_features = X.shape
    
    selection_matrix = np.zeros((n_bootstrap, n_features))
    
    for i in range(n_bootstrap):
        rng = np.random.RandomState(i)
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[boot_idx]
        y_boot = y[boot_idx]
        
        # Standardize
        mean = X_boot.mean(axis=0)
        std = X_boot.std(axis=0)
        std[std == 0] = 1.0
        X_boot_std = (X_boot - mean) / std
        
        # Fit Lasso with CV
        try:
            lasso = LassoCV(
                alphas=lasso_alpha_range, 
                cv=min(5, len(X_boot) // 2), 
                max_iter=5000,  # Increased from default 1000
                tol=1e-3,       # Relaxed from default 1e-4
                random_state=i
            )
            # Suppress convergence warnings during stability selection
            # Results are still valid even with minor convergence issues
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                lasso.fit(X_boot_std, y_boot)
            # Record selections
            selection_matrix[i] = (np.abs(lasso.coef_) > 1e-10).astype(int)
        except Exception:
            # If Lasso fails, skip this bootstrap iteration
            continue
    
    stability_scores = selection_matrix.mean(axis=0)
    
    results = pd.DataFrame({
        'predictor': predictors,
        'stability_score': stability_scores,
        'stable': stability_scores >= threshold,
        'selection_count': (selection_matrix.sum(axis=0)).astype(int),
    }).sort_values('stability_score', ascending=False)
    
    if output_path:
        results.to_csv(output_path, index=False)
        
        # Also write stable predictors to txt for easy reuse
        stable_preds = results[results['stable']]['predictor'].tolist()
        if stable_preds:
            txt_path = Path(output_path).parent / "stable_predictors.txt"
            txt_path.write_text(",".join(stable_preds))
    
    return results


def compute_predictor_family_comparison(
    df,
    predictor_families,
    target,
    output_path,
    standardize=True,
    ridge_alpha=1.0,
    cv_standardize_mode="global",
):
    """
    Compare predictor families (flow vs dino vs mmd) in separate models.
    
    predictor_families: dict like {'flow': [...predictors], 'dino': [...], 'mmd': [...]}
    
    Returns DataFrame comparing R², LOBO performance, and aggregate weights.
    """
    results = []
    complete_df = df.copy()
    
    # Get all predictors
    all_predictors = []
    for family_preds in predictor_families.values():
        all_predictors.extend(family_preds)
    all_predictors = list(set(all_predictors))
    
    complete_df = filter_complete_rows(complete_df, all_predictors, target)
    
    if complete_df.empty:
        return pd.DataFrame()
    
    for family_name, family_predictors in predictor_families.items():
        available_preds = [p for p in family_predictors if p in complete_df.columns]
        if not available_preds:
            continue
        
        # Fit model with only this family
        X = complete_df[available_preds].values
        y = complete_df[target].values
        
        mean = np.zeros(X.shape[1])
        std = np.ones(X.shape[1])
        
        if standardize:
            mean = X.mean(axis=0)
            std = X.std(axis=0)
            std[std == 0] = 1.0
            X_std = (X - mean) / std
        else:
            X_std = X
        
        # Ridge regression
        X_std = np.column_stack([np.ones(len(X_std)), X_std])
        penalty = np.eye(X_std.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(X_std.T @ X_std + float(ridge_alpha) * penalty, X_std.T @ y)
        
        y_pred = X_std.dot(coef)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        
        # Aggregate weight (sum of absolute coefficients)
        agg_weight = float(np.sum(np.abs(coef[1:])))  # Skip intercept
        
        # LOBO validation
        lobo_summary, _ = run_group_cv(
            complete_df,
            "benchmark",
            available_preds,
            target,
            standardize=standardize,
            standardize_mode=cv_standardize_mode,
            model="ridge",
            ridge_alpha=ridge_alpha,
        )
        
        if not lobo_summary.empty:
            overall = lobo_summary[lobo_summary["benchmark"] == "__overall__"]
            lobo_pearson = float(overall["pearson"].iloc[0]) if not overall.empty else np.nan
            lobo_spearman = float(overall["spearman"].iloc[0]) if not overall.empty else np.nan
        else:
            lobo_pearson = np.nan
            lobo_spearman = np.nan
        
        results.append({
            'family': family_name,
            'n_predictors': len(available_preds),
            'r2': r2,
            'agg_weight': agg_weight,
            'lobo_pearson': lobo_pearson,
            'lobo_spearman': lobo_spearman,
            'predictors': ", ".join(available_preds),
        })
    
    result_df = pd.DataFrame(results).sort_values('lobo_spearman', ascending=False)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
    
    return result_df


def compute_univariate_predictor_comparison(
    df,
    all_predictors,
    target,
    output_path,
    standardize=True,
    ridge_alpha=0.5,
    cv_standardize_mode="global",
):
    """
    Fit each predictor individually in separate LOBO runs.
    
    This shows which predictors are useful on their own, without confounding
    from other correlated predictors.
    
    Returns DataFrame with one row per predictor showing LOBO performance.
    """
    results = []
    
    for predictor in all_predictors:
        if predictor not in df.columns:
            continue
        
        # Filter to complete cases for this predictor
        predictor_df = df[[predictor, target, 'benchmark']].dropna()
        
        if predictor_df.empty or len(predictor_df) < 10:
            continue
        
        # Run LOBO with just this one predictor
        try:
            lobo_summary, _ = run_group_cv(
                predictor_df,
                "benchmark",
                [predictor],
                target,
                standardize=standardize,
                standardize_mode=cv_standardize_mode,
                model="ridge",
                ridge_alpha=ridge_alpha,
            )
            
            if not lobo_summary.empty:
                overall = lobo_summary[lobo_summary["benchmark"] == "__overall__"]
                if not overall.empty:
                    lobo_pearson = float(overall["pearson"].iloc[0])
                    lobo_spearman = float(overall["spearman"].iloc[0])
                    lobo_mae = float(overall["mae"].iloc[0])
                    lobo_rmse = float(overall["rmse"].iloc[0])
                else:
                    lobo_pearson = np.nan
                    lobo_spearman = np.nan
                    lobo_mae = np.nan
                    lobo_rmse = np.nan
            else:
                lobo_pearson = np.nan
                lobo_spearman = np.nan
                lobo_mae = np.nan
                lobo_rmse = np.nan
        except Exception as e:
            print(f"Warning: Failed to fit {predictor}: {e}")
            lobo_pearson = np.nan
            lobo_spearman = np.nan
            lobo_mae = np.nan
            lobo_rmse = np.nan
        
        results.append({
            'predictor': predictor,
            'n_obs': len(predictor_df),
            'lobo_pearson': lobo_pearson,
            'lobo_spearman': lobo_spearman,
            'lobo_mae': lobo_mae,
            'lobo_rmse': lobo_rmse,
        })
    
    result_df = pd.DataFrame(results).sort_values('lobo_spearman', ascending=False)
    
    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved univariate comparison to {output_path}")
    
    return result_df


def compute_univariate_predictor_comparison_by_group(
    df,
    all_predictors,
    target,
    group_col,
    output_path,
    standardize=True,
    ridge_alpha=0.5,
    cv_standardize_mode="global",
):
    """
    Fit each predictor individually in separate group-CV runs.

    This mirrors the LOBO univariate comparison but uses a configurable
    grouping column (e.g., train_dataset for LOTO-style analysis).
    """
    results = []

    for predictor in all_predictors:
        if predictor not in df.columns:
            continue

        predictor_df = df[[predictor, target, group_col]].dropna()
        if predictor_df.empty or len(predictor_df) < 10:
            continue

        try:
            summary, _ = run_group_cv(
                predictor_df,
                group_col,
                [predictor],
                target,
                standardize=standardize,
                standardize_mode=cv_standardize_mode,
                model="ridge",
                ridge_alpha=ridge_alpha,
            )

            if not summary.empty:
                overall = summary[summary[group_col] == "__overall__"]
                if not overall.empty:
                    pearson = float(overall["pearson"].iloc[0])
                    spearman = float(overall["spearman"].iloc[0])
                    mae = float(overall["mae"].iloc[0])
                    rmse = float(overall["rmse"].iloc[0])
                else:
                    pearson = np.nan
                    spearman = np.nan
                    mae = np.nan
                    rmse = np.nan
            else:
                pearson = np.nan
                spearman = np.nan
                mae = np.nan
                rmse = np.nan
        except Exception as e:
            print(f"Warning: Failed to fit {predictor} ({group_col}): {e}")
            pearson = np.nan
            spearman = np.nan
            mae = np.nan
            rmse = np.nan

        results.append({
            "predictor": predictor,
            "n_obs": len(predictor_df),
            f"{group_col}_pearson": pearson,
            f"{group_col}_spearman": spearman,
            f"{group_col}_mae": mae,
            f"{group_col}_rmse": rmse,
        })

    result_df = pd.DataFrame(results).sort_values(
        f"{group_col}_spearman", ascending=False
    )

    if output_path:
        result_df.to_csv(output_path, index=False)
        print(f"Saved group univariate comparison to {output_path}")

    return result_df


def fit_pairwise_rank_model(
    train_df,
    predictors,
    target,
    group_cols,
    option_col,
    standardize=True,
    ridge_alpha=1.0,
    min_std=0.0,
    max_iter=200,
    lr=0.1,
):
    if isinstance(group_cols, str):
        group_cols = [group_cols] if group_cols else []
    group_cols = [c for c in list(group_cols or []) if c]
    agg_cols = list(dict.fromkeys(group_cols + ([option_col] if option_col else [])))
    if not agg_cols:
        raise ValueError("Pairwise rank requires at least one grouping/option column.")
    grouped = (
        train_df.groupby(agg_cols)[predictors + [target]]
        .mean()
        .reset_index(drop=False)
    )
    grouped = grouped.dropna(subset=predictors + [target])
    if grouped.empty:
        n = len(predictors)
        return np.zeros(n), np.zeros(n), np.ones(n)

    X = grouped[predictors].to_numpy(dtype=float)
    y = grouped[target].to_numpy(dtype=float)
    mean = np.zeros(X.shape[1])
    std = np.ones(X.shape[1])
    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        if min_std > 0:
            std = np.where(std < min_std, float(min_std), std)
        std[std == 0] = 1.0
        X = (X - mean) / std

    diffs = []
    labels = []
    if group_cols:
        group_iter = grouped.groupby(group_cols, dropna=False)
    else:
        group_iter = [(None, grouped)]
    for _, sub in group_iter:
        if len(sub) < 2:
            continue
        idx = sub.index.to_numpy()
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                yi = y[idx[i]]
                yj = y[idx[j]]
                if yi == yj:
                    continue
                label = 1.0 if yi > yj else -1.0
                diffs.append(X[idx[i]] - X[idx[j]])
                labels.append(label)

    if not diffs:
        n = X.shape[1]
        return np.zeros(n), mean, std

    diffs = np.asarray(diffs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    w = np.zeros(diffs.shape[1], dtype=float)
    reg = float(ridge_alpha)

    for _ in range(max_iter):
        margins = labels * (diffs @ w)
        grad = -(labels[:, None] * diffs) * inv_one_plus_exp(margins)[:, None]
        grad = grad.mean(axis=0)
        if reg > 0:
            grad = grad + reg * w
        if not np.all(np.isfinite(grad)):
            break
        grad_norm = float(np.linalg.norm(grad))
        if np.isfinite(grad_norm) and grad_norm > 1e3:
            grad = grad * (1e3 / grad_norm)
        step = float(lr) * grad
        if not np.all(np.isfinite(step)):
            break
        w = w - step

    return w, mean, std


def predict_linear_model(df, predictors, coef, mean, std, standardize=True):
    X = df[predictors].to_numpy(dtype=float)
    if standardize:
        X = (X - mean) / std
    X = np.column_stack([np.ones(len(X)), X])
    return X.dot(coef)


def predict_pairwise_rank(df, predictors, coef, mean, std, standardize=True):
    X = df[predictors].to_numpy(dtype=float)
    if standardize:
        X = (X - mean) / std
    return X.dot(coef)


def filter_complete_rows(df, predictors, target):
    mask = df[predictors + [target]].notna().all(axis=1)
    return df[mask].copy()


def _is_synthetic_train_dataset_name(name):
    if name is None:
        return False
    norm = normalize_dataset_name(str(name))
    if norm.startswith("synthetic"):
        return True
    return "_synthetic" in norm


def _inverse_frequency_weights(df, cols):
    if df.empty or not cols:
        return np.ones(len(df), dtype=float)
    if any(col not in df.columns for col in cols):
        return np.ones(len(df), dtype=float)
    if len(cols) == 1:
        col = cols[0]
        counts = df.groupby(col, dropna=False).size()
        row_counts = df[col].map(counts).to_numpy(dtype=float)
    else:
        counts = df.groupby(cols, dropna=False).size()
        keys = pd.MultiIndex.from_frame(df[cols])
        row_counts = counts.reindex(keys).to_numpy(dtype=float)
    row_counts = np.where(np.isfinite(row_counts) & (row_counts > 0), row_counts, 1.0)
    return 1.0 / row_counts


def compute_fit_sample_weights(
    train_df,
    mode="none",
    balance_real_synth=False,
):
    n = len(train_df)
    if n == 0:
        return np.array([], dtype=float)

    weights = np.ones(n, dtype=float)
    mode = str(mode or "none").strip().lower()
    if mode == "inverse_benchmark":
        weights *= _inverse_frequency_weights(train_df, ["benchmark"])
    elif mode == "inverse_train_dataset":
        weights *= _inverse_frequency_weights(train_df, ["train_dataset"])
    elif mode == "inverse_task":
        if "model_family_encoder" in train_df.columns:
            task_cols = ["benchmark", "model_family_encoder"]
        elif "model_family" in train_df.columns:
            task_cols = ["benchmark", "model_family"]
        else:
            task_cols = ["benchmark"]
        weights *= _inverse_frequency_weights(train_df, task_cols)

    if balance_real_synth and "train_dataset" in train_df.columns:
        synth_mask = train_df["train_dataset"].map(_is_synthetic_train_dataset_name).to_numpy(dtype=bool)
        real_mask = ~synth_mask
        synth_sum = float(weights[synth_mask].sum()) if synth_mask.any() else 0.0
        real_sum = float(weights[real_mask].sum()) if real_mask.any() else 0.0
        if synth_sum > 0.0 and real_sum > 0.0:
            weights[synth_mask] *= 0.5 / synth_sum
            weights[real_mask] *= 0.5 / real_sum

    weights[~np.isfinite(weights)] = 0.0
    weights = np.where(weights > 0.0, weights, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return np.ones(n, dtype=float)
    # Keep optimization scale stable across folds.
    return weights * (float(n) / total)


def _first_non_null(series):
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def collapse_cv_rows_to_cells(df, group_cols, numeric_agg="mean"):
    if df.empty:
        return df.copy()

    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        return df.copy()

    agg_mode = str(numeric_agg or "mean").strip().lower()
    if agg_mode not in {"mean", "median"}:
        agg_mode = "mean"

    agg_map = {}
    for col in df.columns:
        if col in group_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_map[col] = agg_mode
        else:
            agg_map[col] = _first_non_null

    grouped = df.groupby(group_cols, dropna=False, sort=False)
    collapsed = grouped.agg(agg_map).reset_index()
    collapsed["cell_n_rows"] = grouped.size().to_numpy()
    return collapsed


def drop_low_variance_predictors(train_df, test_df, predictors, min_std):
    if not predictors or min_std <= 0:
        return train_df, test_df, predictors, []
    stds = train_df[predictors].std(ddof=0)
    keep_mask = stds >= float(min_std)
    keep = [p for p in predictors if keep_mask.get(p, False)]
    dropped = [p for p in predictors if p not in keep]
    return train_df, test_df, keep, dropped


def run_group_cv(
    df,
    group_col,
    predictors,
    target,
    standardize=True,
    standardize_mode="local",
    center_by_group=False,
    center_group_col=None,
    group_norm_mode="none",
    within_benchmark_norm="none",
    encoder_group_norm_mode="none",
    encoder_group_col=None,
    target_group_demean=False,
    target_group_col=None,
    target_context_residualize_cols=None,
    target_context_transform="residual",
    target_context_std_eps=1e-9,
    target_context_eval="absolute",
    fewshot_context_calibration=False,
    fewshot_context_calibration_cols=None,
    fewshot_context_calibration_std_eps=1e-9,
    fewshot_context_calibration_min_group_size=2,
    fewshot_context_calibration_backoff=True,
    fewshot_context_calibration_k=0,
    fewshot_context_calibration_seed=0,
    min_predictor_std=0.0,
    prediction_clip=False,
    prediction_clip_min=None,
    prediction_clip_max=None,
    model="ols",
    ridge_alpha=1.0,
    permute_target=False,
    permute_group_col=None,
    permute_seed=0,
    pairwise_option_col=None,
    pairwise_group_cols=("benchmark",),
    fit_sample_weighting="none",
    fit_balance_real_synth=False,
    overall_aggregation="micro",
    task_prediction_rows_out=None,
    task_prediction_option_col=None,
    task_prediction_task_cols=None,
    task_prediction_holdout_group_col=None,
):
    results = []
    pred_rows = []

    # Global standardization: compute mean/std from ALL data before CV
    global_mean = None
    global_std = None
    df_standardized = df.copy()
    if standardize and standardize_mode == "global":
        complete_df = filter_complete_rows(df, predictors, target)
        if not complete_df.empty:
            X_all = complete_df[predictors].to_numpy(dtype=float)
            global_mean = X_all.mean(axis=0)
            global_std = X_all.std(axis=0)
            if min_predictor_std > 0:
                global_std = np.where(global_std < float(min_predictor_std), float(min_predictor_std), global_std)
            global_std[global_std == 0] = 1.0
            # Apply global standardization to all data
            for i, pred in enumerate(predictors):
                df_standardized[pred] = (df_standardized[pred] - global_mean[i]) / global_std[i]

    groups = sorted(df[group_col].dropna().unique())
    for idx, group in enumerate(groups):
        # Use pre-standardized data if global mode
        fold_df = df_standardized if (standardize and standardize_mode == "global") else df
        train_df = fold_df[fold_df[group_col] != group]
        test_df = fold_df[fold_df[group_col] == group]
        train_df = filter_complete_rows(train_df, predictors, target)
        test_df = filter_complete_rows(test_df, predictors, target)

        if train_df.empty or test_df.empty:
            continue
        if len(train_df) <= len(predictors):
            continue

        target_offsets = None
        target_scales = None
        target_residualizer = None
        y_true = test_df[target].to_numpy(dtype=float)
        if target_context_residualize_cols:
            (
                train_df,
                test_df,
                _,
                context_test_offsets,
                _,
                context_test_scales,
                target_residualizer,
            ) = residualize_target_by_context(
                train_df,
                test_df,
                target,
                target_context_residualize_cols,
                transform=target_context_transform,
                std_eps=target_context_std_eps,
            )
            if target_residualizer is not None:
                if str(target_context_eval or "absolute").strip().lower() == "residual":
                    y_true = test_df[target].to_numpy(dtype=float)
                else:
                    target_offsets = context_test_offsets
                    target_scales = context_test_scales
        elif target_group_demean and target_group_col:
            # Support both single column (string) and multiple columns (list)
            if isinstance(target_group_col, str):
                train_df, test_df, target_offsets = demean_target_by_group(
                    train_df, test_df, target, target_group_col
                )
            else:
                # Multiple groups
                train_df, test_df = demean_target_by_multiple_groups(
                    train_df, test_df, target, target_group_col
                )
                target_offsets = None  # Not tracked for multiple groups

        if permute_target:
            rng = np.random.RandomState(int(permute_seed) + idx)
            train_df = permute_target_within_group(
                train_df, target, permute_group_col, rng
            )

        if within_benchmark_norm != "none":
            train_df, test_df = normalize_predictors_within_benchmark(
                train_df, test_df, predictors, within_benchmark_norm
            )

        train_df, test_df, predictors_fold, dropped = drop_low_variance_predictors(
            train_df, test_df, predictors, min_predictor_std
        )
        if not predictors_fold:
            continue
        if len(train_df) <= len(predictors_fold):
            continue

        if encoder_group_norm_mode != "none" and encoder_group_col:
            train_df, test_df = _normalize_predictors_by_group(
                train_df, test_df, predictors_fold, encoder_group_col, encoder_group_norm_mode
            )
        if group_norm_mode != "none" and center_group_col:
            train_df, test_df = _normalize_predictors_by_group(
                train_df, test_df, predictors_fold, center_group_col, group_norm_mode
            )
        elif center_by_group and center_group_col:
            train_df, test_df = _normalize_predictors_by_group(
                train_df, test_df, predictors_fold, center_group_col, "center"
            )

        # Disable local standardization if global mode is used
        local_standardize = standardize and standardize_mode == "local"
        
        fit_weights = None
        if model != "pairwise_rank":
            fit_weights = compute_fit_sample_weights(
                train_df,
                mode=fit_sample_weighting,
                balance_real_synth=fit_balance_real_synth,
            )

        if model == "pairwise_rank":
            coef, mean, std = fit_pairwise_rank_model(
                train_df,
                predictors_fold,
                target,
                pairwise_group_cols,
                pairwise_option_col,
                standardize=local_standardize,
                ridge_alpha=ridge_alpha,
                min_std=min_predictor_std,
            )
            y_pred = predict_pairwise_rank(test_df, predictors_fold, coef, mean, std, local_standardize)
        else:
            coef, mean, std = fit_linear_model(
                train_df,
                predictors_fold,
                target,
                standardize=local_standardize,
                model=model,
                ridge_alpha=ridge_alpha,
                min_std=min_predictor_std,
                sample_weight=fit_weights,
            )
            y_pred = predict_linear_model(test_df, predictors_fold, coef, mean, std, local_standardize)
        if local_standardize:
            X_test = (test_df[predictors_fold].to_numpy(dtype=float) - mean) / std
            max_abs_z = float(np.nanmax(np.abs(X_test))) if X_test.size else np.nan
        else:
            max_abs_z = np.nan
        if target_scales is not None:
            y_pred = y_pred * target_scales
        if target_offsets is not None:
            y_pred = y_pred + target_offsets

        context_calibrator = None
        fewshot_calibration_mask = np.zeros(len(test_df), dtype=bool)
        if bool(fewshot_context_calibration):
            calib_df = test_df
            calib_y_true = y_true
            calib_y_pred = y_pred
            k_shot = int(fewshot_context_calibration_k)
            if k_shot > 0:
                rng = np.random.RandomState(int(fewshot_context_calibration_seed) + int(idx))
                fewshot_calibration_mask = sample_fewshot_calibration_mask(
                    test_df,
                    context_cols=fewshot_context_calibration_cols,
                    k=k_shot,
                    rng=rng,
                    allow_backoff=fewshot_context_calibration_backoff,
                )
                if np.any(fewshot_calibration_mask):
                    calib_df = test_df.iloc[np.where(fewshot_calibration_mask)[0]]
                    calib_y_true = y_true[fewshot_calibration_mask]
                    calib_y_pred = y_pred[fewshot_calibration_mask]
            context_calibrator = fit_context_prediction_calibrator(
                calib_df,
                calib_y_true,
                calib_y_pred,
                context_cols=fewshot_context_calibration_cols,
                std_eps=fewshot_context_calibration_std_eps,
                min_group_size=fewshot_context_calibration_min_group_size,
                allow_backoff=fewshot_context_calibration_backoff,
            )
            y_pred = apply_context_prediction_calibrator(
                test_df,
                y_pred,
                context_calibrator,
            )
        clip_min = prediction_clip_min
        clip_max = prediction_clip_max
        if prediction_clip and model != "pairwise_rank":
            if clip_min is None:
                clip_min = float(np.nanmin(train_df[target].to_numpy(dtype=float)))
            if clip_max is None:
                clip_max = float(np.nanmax(train_df[target].to_numpy(dtype=float)))
            y_pred = np.clip(y_pred, clip_min, clip_max)

        eval_mask = ~fewshot_calibration_mask if int(fewshot_context_calibration_k) > 0 else np.ones(len(test_df), dtype=bool)
        if not np.any(eval_mask):
            continue
        test_eval_df = test_df.iloc[np.where(eval_mask)[0]]
        y_true_eval = y_true[eval_mask]
        y_pred_eval = y_pred[eval_mask]

        if task_prediction_rows_out is not None:
            option_col = task_prediction_option_col or pairwise_option_col
            task_cols = [c for c in (task_prediction_task_cols or ["benchmark"]) if c in fold_df.columns]
            holdout_group_col = (
                task_prediction_holdout_group_col
                if task_prediction_holdout_group_col in fold_df.columns
                else None
            )
            if option_col and option_col in fold_df.columns and task_cols:
                candidate_df = filter_complete_rows(fold_df, predictors, target)
                if not candidate_df.empty:
                    task_keys = test_eval_df[task_cols].drop_duplicates()
                    candidate_df = candidate_df.merge(task_keys, on=task_cols, how="inner")
                    if not candidate_df.empty:
                        candidate_offsets = None
                        candidate_scales = None
                        if target_residualizer is not None:
                            candidate_df, candidate_offsets, candidate_scales = apply_context_target_residualizer(
                                candidate_df,
                                target,
                                target_residualizer,
                            )
                        if model == "pairwise_rank":
                            candidate_pred = predict_pairwise_rank(
                                candidate_df, predictors_fold, coef, mean, std, local_standardize
                            )
                        else:
                            candidate_pred = predict_linear_model(
                                candidate_df, predictors_fold, coef, mean, std, local_standardize
                            )
                            if prediction_clip:
                                candidate_pred = np.clip(candidate_pred, clip_min, clip_max)
                        if (
                            candidate_scales is not None
                            and str(target_context_eval or "absolute").strip().lower()
                            != "residual"
                        ):
                            candidate_pred = candidate_pred * candidate_scales
                        if (
                            candidate_offsets is not None
                            and str(target_context_eval or "absolute").strip().lower()
                            != "residual"
                        ):
                            candidate_pred = candidate_pred + candidate_offsets
                        if context_calibrator is not None:
                            candidate_pred = apply_context_prediction_calibrator(
                                candidate_df,
                                candidate_pred,
                                context_calibrator,
                            )
                        for row, pred in zip(candidate_df.to_dict(orient="records"), candidate_pred):
                            is_holdout = 0
                            if holdout_group_col is not None:
                                is_holdout = int(row.get(holdout_group_col) == group)
                            elif option_col in row:
                                is_holdout = int(row.get(option_col) == group)
                            row.update(
                                {
                                    "prediction": float(pred),
                                    "target": float(row[target]),
                                    "fold": group,
                                    "is_holdout_option": is_holdout,
                                }
                            )
                            task_prediction_rows_out.append(row)

        pred_nan = int(np.isnan(y_pred_eval).sum())
        pred_inf = int(np.isinf(y_pred_eval).sum())
        target_nan = int(np.isnan(y_true_eval).sum())
        target_inf = int(np.isinf(y_true_eval).sum())
        pred_finite = y_pred_eval[np.isfinite(y_pred_eval)]
        target_finite = y_true_eval[np.isfinite(y_true_eval)]
        pred_min = float(pred_finite.min()) if pred_finite.size else np.nan
        pred_max = float(pred_finite.max()) if pred_finite.size else np.nan
        target_min = float(target_finite.min()) if target_finite.size else np.nan
        target_max = float(target_finite.max()) if target_finite.size else np.nan
        pred_std = safe_std(pred_finite)
        target_std = safe_std(target_finite)

        mae, rmse = mae_rmse(y_true_eval, y_pred_eval)
        pearson = pearson_corr(y_true_eval, y_pred_eval)
        spearman = spearman_corr(y_true_eval, y_pred_eval)

        results.append({
            group_col: group,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_eval_df)),
            "n_calibration": int(fewshot_calibration_mask.sum()),
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "target_min": target_min,
            "target_max": target_max,
            "pred_min": pred_min,
            "pred_max": pred_max,
            "target_std": target_std,
            "pred_std": pred_std,
            "max_abs_zscore_feature": max_abs_z,
            "pred_nan": pred_nan,
            "pred_inf": pred_inf,
            "target_nan": target_nan,
            "target_inf": target_inf,
            "dropped_predictor_count": len(dropped),
        })

        for row, pred in zip(test_eval_df.to_dict(orient="records"), y_pred_eval):
            row.update({
                "prediction": float(pred),
                "target": float(row[target]),
                "fold": group,
                "is_calibration_row": 0,
            })
            pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(results)

    if not pred_df.empty:
        macro_mode = str(overall_aggregation or "micro").strip().lower() == "macro_fold"
        if macro_mode and not summary_df.empty:
            overall_mae = float(summary_df["mae"].mean())
            overall_rmse = float(summary_df["rmse"].mean())
            overall_pearson = float(summary_df["pearson"].mean())
            overall_spearman = float(summary_df["spearman"].mean())
        else:
            overall_mae, overall_rmse = mae_rmse(
                pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
            )
            overall_pearson = pearson_corr(
                pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
            )
            overall_spearman = spearman_corr(
                pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
            )
        overall = {
            group_col: "__overall__",
            "n_train": int(len(df)),
            "n_test": int(len(pred_df)),
            "n_calibration": float(summary_df["n_calibration"].sum()) if "n_calibration" in summary_df.columns else np.nan,
            "mae": overall_mae,
            "rmse": overall_rmse,
            "pearson": overall_pearson,
            "spearman": overall_spearman,
            "target_min": float(pred_df["target"].min()) if not pred_df.empty else np.nan,
            "target_max": float(pred_df["target"].max()) if not pred_df.empty else np.nan,
            "pred_min": float(pred_df["prediction"].min()) if not pred_df.empty else np.nan,
            "pred_max": float(pred_df["prediction"].max()) if not pred_df.empty else np.nan,
            "target_std": safe_std(pred_df["target"].to_numpy()) if not pred_df.empty else np.nan,
            "pred_std": safe_std(pred_df["prediction"].to_numpy()) if not pred_df.empty else np.nan,
            "max_abs_zscore_feature": np.nan,
            "pred_nan": int(np.isnan(pred_df["prediction"]).sum()) if not pred_df.empty else 0,
            "pred_inf": int(np.isinf(pred_df["prediction"]).sum()) if not pred_df.empty else 0,
            "target_nan": int(np.isnan(pred_df["target"]).sum()) if not pred_df.empty else 0,
            "target_inf": int(np.isinf(pred_df["target"]).sum()) if not pred_df.empty else 0,
            "dropped_predictor_count": np.nan,
            "overall_aggregation": "macro_fold" if macro_mode else "micro",
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    return summary_df, pred_df


def run_joint_holdout_cv(
    df,
    train_group_col,
    benchmark_col,
    predictors,
    target,
    standardize=True,
    standardize_mode="local",
    center_by_group=False,
    center_group_col=None,
    group_norm_mode="none",
    within_benchmark_norm="none",
    encoder_group_norm_mode="none",
    encoder_group_col=None,
    target_group_demean=False,
    target_group_col=None,
    target_context_residualize_cols=None,
    target_context_transform="residual",
    target_context_std_eps=1e-9,
    target_context_eval="absolute",
    fewshot_context_calibration=False,
    fewshot_context_calibration_cols=None,
    fewshot_context_calibration_std_eps=1e-9,
    fewshot_context_calibration_min_group_size=2,
    fewshot_context_calibration_backoff=True,
    fewshot_context_calibration_k=0,
    fewshot_context_calibration_seed=0,
    min_predictor_std=0.0,
    prediction_clip=False,
    prediction_clip_min=None,
    prediction_clip_max=None,
    model="ols",
    ridge_alpha=1.0,
    permute_target=False,
    permute_group_col=None,
    permute_seed=0,
    pairwise_option_col=None,
    pairwise_group_cols=("benchmark",),
    fit_sample_weighting="none",
    fit_balance_real_synth=False,
    overall_aggregation="micro",
    task_prediction_rows_out=None,
    task_prediction_option_col=None,
    task_prediction_task_cols=None,
    task_prediction_holdout_group_col=None,
):
    results = []
    pred_rows = []

    global_mean = None
    global_std = None
    df_standardized = df.copy()
    if standardize and standardize_mode == "global":
        complete_df = filter_complete_rows(df, predictors, target)
        if not complete_df.empty:
            X_all = complete_df[predictors].to_numpy(dtype=float)
            global_mean = X_all.mean(axis=0)
            global_std = X_all.std(axis=0)
            if min_predictor_std > 0:
                global_std = np.where(
                    global_std < float(min_predictor_std),
                    float(min_predictor_std),
                    global_std,
                )
            global_std[global_std == 0] = 1.0
            for i, pred in enumerate(predictors):
                df_standardized[pred] = (
                    df_standardized[pred] - global_mean[i]
                ) / global_std[i]

    train_groups = sorted(df[train_group_col].dropna().unique())
    benchmark_groups = sorted(df[benchmark_col].dropna().unique())
    fold_idx = 0
    for train_group in train_groups:
        for benchmark in benchmark_groups:
            holdout_name = f"{train_group}__{benchmark}"
            fold_df = df_standardized if (standardize and standardize_mode == "global") else df
            train_df = fold_df[
                (fold_df[train_group_col] != train_group) & (fold_df[benchmark_col] != benchmark)
            ]
            test_df = fold_df[
                (fold_df[train_group_col] == train_group) & (fold_df[benchmark_col] == benchmark)
            ]
            train_df = filter_complete_rows(train_df, predictors, target)
            test_df = filter_complete_rows(test_df, predictors, target)

            if train_df.empty or test_df.empty:
                fold_idx += 1
                continue
            if len(train_df) <= len(predictors):
                fold_idx += 1
                continue

            target_offsets = None
            target_scales = None
            target_residualizer = None
            y_true = test_df[target].to_numpy(dtype=float)
            if target_context_residualize_cols:
                (
                    train_df,
                    test_df,
                    _,
                    context_test_offsets,
                    _,
                    context_test_scales,
                    target_residualizer,
                ) = residualize_target_by_context(
                    train_df,
                    test_df,
                    target,
                    target_context_residualize_cols,
                    transform=target_context_transform,
                    std_eps=target_context_std_eps,
                )
                if target_residualizer is not None:
                    if str(target_context_eval or "absolute").strip().lower() == "residual":
                        y_true = test_df[target].to_numpy(dtype=float)
                    else:
                        target_offsets = context_test_offsets
                        target_scales = context_test_scales
            elif target_group_demean and target_group_col:
                if isinstance(target_group_col, str):
                    train_df, test_df, target_offsets = demean_target_by_group(
                        train_df, test_df, target, target_group_col
                    )
                else:
                    train_df, test_df = demean_target_by_multiple_groups(
                        train_df, test_df, target, target_group_col
                    )
                    target_offsets = None

            if permute_target:
                rng = np.random.RandomState(int(permute_seed) + fold_idx)
                train_df = permute_target_within_group(
                    train_df, target, permute_group_col, rng
                )

            if within_benchmark_norm != "none":
                train_df, test_df = normalize_predictors_within_benchmark(
                    train_df, test_df, predictors, within_benchmark_norm
                )

            train_df, test_df, predictors_fold, dropped = drop_low_variance_predictors(
                train_df, test_df, predictors, min_predictor_std
            )
            if not predictors_fold:
                fold_idx += 1
                continue
            if len(train_df) <= len(predictors_fold):
                fold_idx += 1
                continue

            if encoder_group_norm_mode != "none" and encoder_group_col:
                train_df, test_df = _normalize_predictors_by_group(
                    train_df,
                    test_df,
                    predictors_fold,
                    encoder_group_col,
                    encoder_group_norm_mode,
                )
            if group_norm_mode != "none" and center_group_col:
                train_df, test_df = _normalize_predictors_by_group(
                    train_df, test_df, predictors_fold, center_group_col, group_norm_mode
                )
            elif center_by_group and center_group_col:
                train_df, test_df = _normalize_predictors_by_group(
                    train_df, test_df, predictors_fold, center_group_col, "center"
                )

            local_standardize = standardize and standardize_mode == "local"
            fit_weights = None
            if model != "pairwise_rank":
                fit_weights = compute_fit_sample_weights(
                    train_df,
                    mode=fit_sample_weighting,
                    balance_real_synth=fit_balance_real_synth,
                )

            if model == "pairwise_rank":
                coef, mean, std = fit_pairwise_rank_model(
                    train_df,
                    predictors_fold,
                    target,
                    pairwise_group_cols,
                    pairwise_option_col,
                    standardize=local_standardize,
                    ridge_alpha=ridge_alpha,
                    min_std=min_predictor_std,
                )
                y_pred = predict_pairwise_rank(
                    test_df, predictors_fold, coef, mean, std, local_standardize
                )
            else:
                coef, mean, std = fit_linear_model(
                    train_df,
                    predictors_fold,
                    target,
                    standardize=local_standardize,
                    model=model,
                    ridge_alpha=ridge_alpha,
                    min_std=min_predictor_std,
                    sample_weight=fit_weights,
                )
                y_pred = predict_linear_model(
                    test_df, predictors_fold, coef, mean, std, local_standardize
                )
            if local_standardize:
                X_test = (test_df[predictors_fold].to_numpy(dtype=float) - mean) / std
                max_abs_z = float(np.nanmax(np.abs(X_test))) if X_test.size else np.nan
            else:
                max_abs_z = np.nan
            if target_scales is not None:
                y_pred = y_pred * target_scales
            if target_offsets is not None:
                y_pred = y_pred + target_offsets

            context_calibrator = None
            fewshot_calibration_mask = np.zeros(len(test_df), dtype=bool)
            if bool(fewshot_context_calibration):
                calib_df = test_df
                calib_y_true = y_true
                calib_y_pred = y_pred
                k_shot = int(fewshot_context_calibration_k)
                if k_shot > 0:
                    rng = np.random.RandomState(int(fewshot_context_calibration_seed) + int(fold_idx))
                    fewshot_calibration_mask = sample_fewshot_calibration_mask(
                        test_df,
                        context_cols=fewshot_context_calibration_cols,
                        k=k_shot,
                        rng=rng,
                        allow_backoff=fewshot_context_calibration_backoff,
                    )
                    if np.any(fewshot_calibration_mask):
                        calib_df = test_df.iloc[np.where(fewshot_calibration_mask)[0]]
                        calib_y_true = y_true[fewshot_calibration_mask]
                        calib_y_pred = y_pred[fewshot_calibration_mask]
                context_calibrator = fit_context_prediction_calibrator(
                    calib_df,
                    calib_y_true,
                    calib_y_pred,
                    context_cols=fewshot_context_calibration_cols,
                    std_eps=fewshot_context_calibration_std_eps,
                    min_group_size=fewshot_context_calibration_min_group_size,
                    allow_backoff=fewshot_context_calibration_backoff,
                )
                y_pred = apply_context_prediction_calibrator(
                    test_df,
                    y_pred,
                    context_calibrator,
                )
            clip_min = prediction_clip_min
            clip_max = prediction_clip_max
            if prediction_clip and model != "pairwise_rank":
                if clip_min is None:
                    clip_min = float(np.nanmin(train_df[target].to_numpy(dtype=float)))
                if clip_max is None:
                    clip_max = float(np.nanmax(train_df[target].to_numpy(dtype=float)))
                y_pred = np.clip(y_pred, clip_min, clip_max)

            eval_mask = ~fewshot_calibration_mask if int(fewshot_context_calibration_k) > 0 else np.ones(len(test_df), dtype=bool)
            if not np.any(eval_mask):
                fold_idx += 1
                continue
            test_eval_df = test_df.iloc[np.where(eval_mask)[0]]
            y_true_eval = y_true[eval_mask]
            y_pred_eval = y_pred[eval_mask]

            if task_prediction_rows_out is not None:
                option_col = task_prediction_option_col or pairwise_option_col
                task_cols = [c for c in (task_prediction_task_cols or ["benchmark"]) if c in fold_df.columns]
                holdout_group_col = (
                    task_prediction_holdout_group_col
                    if task_prediction_holdout_group_col in fold_df.columns
                    else None
                )
                if option_col and option_col in fold_df.columns and task_cols:
                    candidate_df = filter_complete_rows(fold_df, predictors, target)
                    if not candidate_df.empty:
                        task_keys = test_eval_df[task_cols].drop_duplicates()
                        candidate_df = candidate_df.merge(task_keys, on=task_cols, how="inner")
                        if not candidate_df.empty:
                            candidate_offsets = None
                            candidate_scales = None
                            if target_residualizer is not None:
                                candidate_df, candidate_offsets, candidate_scales = apply_context_target_residualizer(
                                    candidate_df,
                                    target,
                                    target_residualizer,
                                )
                            if model == "pairwise_rank":
                                candidate_pred = predict_pairwise_rank(
                                    candidate_df, predictors_fold, coef, mean, std, local_standardize
                                )
                            else:
                                candidate_pred = predict_linear_model(
                                    candidate_df, predictors_fold, coef, mean, std, local_standardize
                                )
                                if prediction_clip:
                                    candidate_pred = np.clip(candidate_pred, clip_min, clip_max)
                            if (
                                candidate_scales is not None
                                and str(target_context_eval or "absolute").strip().lower()
                                != "residual"
                            ):
                                candidate_pred = candidate_pred * candidate_scales
                            if (
                                candidate_offsets is not None
                                and str(target_context_eval or "absolute").strip().lower()
                                != "residual"
                            ):
                                candidate_pred = candidate_pred + candidate_offsets
                            if context_calibrator is not None:
                                candidate_pred = apply_context_prediction_calibrator(
                                    candidate_df,
                                    candidate_pred,
                                    context_calibrator,
                                )
                            for row, pred in zip(candidate_df.to_dict(orient="records"), candidate_pred):
                                is_holdout = 0
                                if holdout_group_col is not None:
                                    is_holdout = int(row.get(holdout_group_col) == train_group)
                                row.update(
                                    {
                                        "prediction": float(pred),
                                        "target": float(row[target]),
                                        "fold": holdout_name,
                                        "is_holdout_option": is_holdout,
                                    }
                                )
                                task_prediction_rows_out.append(row)

            pred_nan = int(np.isnan(y_pred_eval).sum())
            pred_inf = int(np.isinf(y_pred_eval).sum())
            target_nan = int(np.isnan(y_true_eval).sum())
            target_inf = int(np.isinf(y_true_eval).sum())
            pred_finite = y_pred_eval[np.isfinite(y_pred_eval)]
            target_finite = y_true_eval[np.isfinite(y_true_eval)]
            pred_min = float(pred_finite.min()) if pred_finite.size else np.nan
            pred_max = float(pred_finite.max()) if pred_finite.size else np.nan
            target_min = float(target_finite.min()) if target_finite.size else np.nan
            target_max = float(target_finite.max()) if target_finite.size else np.nan
            pred_std = safe_std(pred_finite)
            target_std = safe_std(target_finite)

            mae, rmse = mae_rmse(y_true_eval, y_pred_eval)
            pearson = pearson_corr(y_true_eval, y_pred_eval)
            spearman = spearman_corr(y_true_eval, y_pred_eval)

            results.append(
                {
                    "joint_holdout": holdout_name,
                    train_group_col: train_group,
                    benchmark_col: benchmark,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_eval_df)),
                    "n_calibration": int(fewshot_calibration_mask.sum()),
                    "mae": mae,
                    "rmse": rmse,
                    "pearson": pearson,
                    "spearman": spearman,
                    "target_min": target_min,
                    "target_max": target_max,
                    "pred_min": pred_min,
                    "pred_max": pred_max,
                    "target_std": target_std,
                    "pred_std": pred_std,
                    "max_abs_zscore_feature": max_abs_z,
                    "pred_nan": pred_nan,
                    "pred_inf": pred_inf,
                    "target_nan": target_nan,
                    "target_inf": target_inf,
                    "dropped_predictor_count": len(dropped),
                }
            )

            for row, pred in zip(test_eval_df.to_dict(orient="records"), y_pred_eval):
                row.update(
                    {
                        "prediction": float(pred),
                        "target": float(row[target]),
                        "fold": holdout_name,
                        "joint_holdout": holdout_name,
                        "is_calibration_row": 0,
                    }
                )
                pred_rows.append(row)
            fold_idx += 1

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(results)

    if not pred_df.empty:
        macro_mode = str(overall_aggregation or "micro").strip().lower() == "macro_fold"
        if macro_mode and not summary_df.empty:
            overall_mae = float(summary_df["mae"].mean())
            overall_rmse = float(summary_df["rmse"].mean())
            overall_pearson = float(summary_df["pearson"].mean())
            overall_spearman = float(summary_df["spearman"].mean())
        else:
            overall_mae, overall_rmse = mae_rmse(
                pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
            )
            overall_pearson = pearson_corr(
                pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
            )
            overall_spearman = spearman_corr(
                pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
            )
        overall = {
            "joint_holdout": "__overall__",
            train_group_col: "__overall__",
            benchmark_col: "__overall__",
            "n_train": int(len(df)),
            "n_test": int(len(pred_df)),
            "n_calibration": float(summary_df["n_calibration"].sum()) if "n_calibration" in summary_df.columns else np.nan,
            "mae": overall_mae,
            "rmse": overall_rmse,
            "pearson": overall_pearson,
            "spearman": overall_spearman,
            "target_min": float(pred_df["target"].min()) if not pred_df.empty else np.nan,
            "target_max": float(pred_df["target"].max()) if not pred_df.empty else np.nan,
            "pred_min": float(pred_df["prediction"].min()) if not pred_df.empty else np.nan,
            "pred_max": float(pred_df["prediction"].max()) if not pred_df.empty else np.nan,
            "target_std": safe_std(pred_df["target"].to_numpy()) if not pred_df.empty else np.nan,
            "pred_std": safe_std(pred_df["prediction"].to_numpy()) if not pred_df.empty else np.nan,
            "max_abs_zscore_feature": np.nan,
            "pred_nan": int(np.isnan(pred_df["prediction"]).sum()) if not pred_df.empty else 0,
            "pred_inf": int(np.isinf(pred_df["prediction"]).sum()) if not pred_df.empty else 0,
            "target_nan": int(np.isnan(pred_df["target"]).sum()) if not pred_df.empty else 0,
            "target_inf": int(np.isinf(pred_df["target"]).sum()) if not pred_df.empty else 0,
            "dropped_predictor_count": np.nan,
            "overall_aggregation": "macro_fold" if macro_mode else "micro",
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    return summary_df, pred_df


def _standardize_predictors(train_df, test_df, predictors, standardize, min_std=0.0):
    if not standardize:
        mapping = {col: col for col in predictors}
        return train_df.copy(), test_df.copy(), predictors, mapping

    train_df = train_df.copy()
    test_df = test_df.copy()
    mean = train_df[predictors].mean()
    std = train_df[predictors].std(ddof=0)
    if min_std > 0:
        std = std.where(std >= float(min_std), float(min_std))
    std = std.replace(0, 1.0)

    standardized_cols = []
    mapping = {}
    for col in predictors:
        z_col = f"{col}_z"
        train_df[z_col] = (train_df[col] - mean[col]) / std[col]
        test_df[z_col] = (test_df[col] - mean[col]) / std[col]
        standardized_cols.append(z_col)
        mapping[col] = z_col

    return train_df, test_df, standardized_cols, mapping


def _normalize_predictors_by_group(train_df, test_df, predictors, group_col, mode):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if group_col not in train_df.columns or group_col not in test_df.columns:
        return train_df, test_df

    group_means = train_df.groupby(group_col)[predictors].mean()
    global_means = train_df[predictors].mean()

    if mode == "center":
        for col in predictors:
            train_df[col] = train_df[col] - train_df[group_col].map(group_means[col])
            test_means = test_df[group_col].map(group_means[col]).fillna(global_means[col])
            test_df[col] = test_df[col] - test_means
        return train_df, test_df

    if mode == "zscore":
        group_stds = train_df.groupby(group_col)[predictors].std(ddof=0).replace(0, 1.0)
        global_stds = train_df[predictors].std(ddof=0).replace(0, 1.0)
        for col in predictors:
            train_means = train_df[group_col].map(group_means[col])
            train_stds = train_df[group_col].map(group_stds[col]).replace(0, 1.0)
            train_df[col] = (train_df[col] - train_means) / train_stds
            test_means = test_df[group_col].map(group_means[col]).fillna(global_means[col])
            test_stds = test_df[group_col].map(group_stds[col]).fillna(global_stds[col]).replace(0, 1.0)
            test_df[col] = (test_df[col] - test_means) / test_stds
        return train_df, test_df

    return train_df, test_df


def _random_intercept(result, group):
    try:
        re_dict = result.random_effects
    except Exception:
        return 0.0
    value = re_dict.get(group)
    if value is None:
        return 0.0
    try:
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
    except Exception:
        pass
    try:
        arr = np.asarray(value).ravel()
    except Exception:
        return 0.0
    return float(arr[0]) if arr.size else 0.0


def run_group_cv_mixedlm(
    df,
    holdout_col,
    predictors,
    target,
    random_group_col=None,
    random_slopes=None,
    standardize=True,
    standardize_mode="local",
    center_by_group=False,
    center_group_col=None,
    group_norm_mode="none",
    within_benchmark_norm="none",
    encoder_group_norm_mode="none",
    encoder_group_col=None,
    target_group_demean=False,
    target_group_col=None,
    min_predictor_std=0.0,
    prediction_clip=False,
    prediction_clip_min=None,
    prediction_clip_max=None,
    permute_target=False,
    permute_group_col=None,
    permute_seed=0,
):
    if not HAS_STATSMODELS:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    pred_rows = []
    group_col = random_group_col or holdout_col
    random_slopes = random_slopes or []

    # Global standardization: compute mean/std from ALL data before CV
    global_mean = None
    global_std = None
    df_standardized = df.copy()
    if standardize and standardize_mode == "global":
        complete_df = filter_complete_rows(df, predictors, target)
        if not complete_df.empty:
            X_all = complete_df[predictors].to_numpy(dtype=float)
            global_mean = X_all.mean(axis=0)
            global_std = X_all.std(axis=0)
            if min_predictor_std > 0:
                global_std = np.where(global_std < float(min_predictor_std), float(min_predictor_std), global_std)
            global_std[global_std == 0] = 1.0
            # Apply global standardization to all data
            for i, pred in enumerate(predictors):
                df_standardized[pred] = (df_standardized[pred] - global_mean[i]) / global_std[i]

    groups = sorted(df[holdout_col].dropna().unique())
    for idx, group in enumerate(groups):
        # Use pre-standardized data if global mode
        fold_df = df_standardized if (standardize and standardize_mode == "global") else df
        train_df = fold_df[fold_df[holdout_col] != group]
        test_df = fold_df[fold_df[holdout_col] == group]
        train_df = filter_complete_rows(train_df, predictors, target)
        test_df = filter_complete_rows(test_df, predictors, target)

        if train_df.empty or test_df.empty:
            continue
        if len(train_df) <= len(predictors):
            continue
        if train_df[group_col].nunique() < 2:
            continue

        if target_group_demean and target_group_col:
            # Support both single column (string) and multiple columns (list)
            if isinstance(target_group_col, str):
                train_df, test_df, _ = demean_target_by_group(
                    train_df, test_df, target, target_group_col
                )
            else:
                # Multiple groups
                train_df, test_df = demean_target_by_multiple_groups(
                    train_df, test_df, target, target_group_col
                )

        if permute_target:
            rng = np.random.RandomState(int(permute_seed) + idx)
            train_df = permute_target_within_group(
                train_df, target, permute_group_col, rng
            )

        if within_benchmark_norm != "none":
            train_df, test_df = normalize_predictors_within_benchmark(
                train_df, test_df, predictors, within_benchmark_norm
            )

        train_df, test_df, predictors_fold, _ = drop_low_variance_predictors(
            train_df, test_df, predictors, min_predictor_std
        )
        if not predictors_fold:
            continue
        if len(train_df) <= len(predictors_fold):
            continue

        if encoder_group_norm_mode != "none" and encoder_group_col:
            train_df, test_df = _normalize_predictors_by_group(
                train_df, test_df, predictors_fold, encoder_group_col, encoder_group_norm_mode
            )
        if group_norm_mode != "none" and center_group_col:
            train_df, test_df = _normalize_predictors_by_group(
                train_df, test_df, predictors_fold, center_group_col, group_norm_mode
            )
        elif center_by_group and center_group_col:
            train_df, test_df = _normalize_predictors_by_group(
                train_df, test_df, predictors_fold, center_group_col, "center"
            )

        # Disable local standardization if global mode is used
        local_standardize = standardize and standardize_mode == "local"
        train_df, test_df, pred_cols, mapping = _standardize_predictors(
            train_df, test_df, predictors_fold, local_standardize, min_std=min_predictor_std
        )

        random_cols = []
        for name in random_slopes:
            if name in mapping:
                random_cols.append(mapping[name])

        formula = f"{target} ~ " + " + ".join(pred_cols)
        try:
            re_formula = "1"
            if random_cols:
                re_formula = "1 + " + " + ".join(random_cols)
            result = smf.mixedlm(
                formula,
                data=train_df,
                groups=train_df[group_col],
                re_formula=re_formula,
            ).fit(reml=False, method="lbfgs")
        except Exception:
            continue

        fe_params = result.fe_params
        y_pred = fe_params.get("Intercept", 0.0)
        for col in pred_cols:
            y_pred = y_pred + fe_params.get(col, 0.0) * test_df[col].to_numpy(dtype=float)

        # Add random intercepts when available (skip if singular covariance)
        if group_col in test_df.columns:
            preds = np.asarray(y_pred, dtype=float)
            for group_name in test_df[group_col].unique():
                re = _random_intercept(result, group_name)
                if re != 0.0:
                    mask = test_df[group_col] == group_name
                    preds[mask.to_numpy()] += re
            y_pred = preds

        y_true = test_df[target].to_numpy(dtype=float)
        if prediction_clip:
            clip_min = prediction_clip_min
            clip_max = prediction_clip_max
            if clip_min is None:
                clip_min = float(np.nanmin(train_df[target].to_numpy(dtype=float)))
            if clip_max is None:
                clip_max = float(np.nanmax(train_df[target].to_numpy(dtype=float)))
            y_pred = np.clip(y_pred, clip_min, clip_max)

        mae, rmse = mae_rmse(y_true, y_pred)
        pearson = pearson_corr(y_true, y_pred)
        spearman = spearman_corr(y_true, y_pred)

        results.append({
            holdout_col: group,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
        })

        for row, pred in zip(test_df.to_dict(orient="records"), y_pred):
            row.update({
                "prediction": float(pred),
                "target": float(row[target]),
                "fold": group,
            })
            pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    summary_df = pd.DataFrame(results)

    if not pred_df.empty:
        overall_mae, overall_rmse = mae_rmse(
            pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()
        )
        overall = {
            holdout_col: "__overall__",
            "n_train": int(len(df)),
            "n_test": int(len(pred_df)),
            "mae": overall_mae,
            "rmse": overall_rmse,
            "pearson": pearson_corr(pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()),
            "spearman": spearman_corr(pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()),
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    return summary_df, pred_df


def _zscore(series):
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def compute_within_benchmark_slopes(df, predictors, target, output_path, min_rows=12):
    rows = []
    for benchmark, sub in df.groupby("benchmark"):
        sub = filter_complete_rows(sub, predictors, target)
        if len(sub) < min_rows:
            continue
        z_df = sub.copy()
        z_df[target] = _zscore(z_df[target])
        for col in predictors:
            z_df[col] = _zscore(z_df[col])

        row = {
            "benchmark": benchmark,
            "n": int(len(sub)),
            "r2": np.nan,
            "mode": "univariate",
        }
        if len(sub) >= max(min_rows, len(predictors) + 2):
            X = z_df[predictors].to_numpy(dtype=float)
            y = z_df[target].to_numpy(dtype=float)
            X = np.column_stack([np.ones(len(X)), X])
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X.dot(coef)
            denom = np.sum((y - np.mean(y)) ** 2)
            r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)
            row["r2"] = float(r2)
            row["mode"] = "multivariate"
            for name, value in zip(predictors, coef[1:]):
                row[name] = float(value)
        else:
            # Fallback: univariate standardized OLS per predictor (slope == correlation).
            y = z_df[target].to_numpy(dtype=float)
            for name in predictors:
                x = z_df[name].to_numpy(dtype=float)
                if len(x) < 2:
                    row[name] = np.nan
                    continue
                denom = np.dot(x - np.mean(x), x - np.mean(x))
                if denom == 0:
                    row[name] = np.nan
                else:
                    row[name] = float(np.dot(x - np.mean(x), y - np.mean(y)) / denom)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
    return df_out


def compute_within_benchmark_univariate_slopes(df, predictors, target, output_path, min_rows=12):
    rows = []
    for benchmark, sub in df.groupby("benchmark"):
        sub = filter_complete_rows(sub, predictors, target)
        if len(sub) < min_rows:
            continue
        z_df = sub.copy()
        z_df[target] = _zscore(z_df[target])
        for col in predictors:
            z_df[col] = _zscore(z_df[col])

        row = {
            "benchmark": benchmark,
            "n": int(len(sub)),
            "r2": np.nan,
            "mode": "univariate",
        }
        y = z_df[target].to_numpy(dtype=float)
        for name in predictors:
            x = z_df[name].to_numpy(dtype=float)
            if len(x) < 2:
                row[name] = np.nan
                continue
            denom = np.dot(x - np.mean(x), x - np.mean(x))
            if denom == 0:
                row[name] = np.nan
            else:
                slope = float(np.dot(x - np.mean(x), y - np.mean(y)) / denom)
                row[name] = slope
        if predictors:
            vals = np.array([row.get(p) for p in predictors], dtype=float)
            if np.isfinite(vals).any():
                # Approximate average univariate R2 via mean corr^2
                row["r2"] = float(np.nanmean(np.square(vals)))
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
    return df_out


def compute_within_train_dataset_univariate_slopes(
    df,
    predictors,
    target,
    group_col,
    output_path,
    min_rows=12,
):
    rows = []
    for group, sub in df.groupby(group_col):
        sub = filter_complete_rows(sub, predictors, target)
        if len(sub) < min_rows:
            continue
        z_df = sub.copy()
        z_df[target] = _zscore(z_df[target])
        for col in predictors:
            z_df[col] = _zscore(z_df[col])

        row = {
            group_col: group,
            "n": int(len(sub)),
            "r2": np.nan,
            "mode": "univariate",
        }
        y = z_df[target].to_numpy(dtype=float)
        for name in predictors:
            x = z_df[name].to_numpy(dtype=float)
            if len(x) < 2:
                row[name] = np.nan
                continue
            denom = np.dot(x - np.mean(x), x - np.mean(x))
            if denom == 0:
                row[name] = np.nan
            else:
                slope = float(np.dot(x - np.mean(x), y - np.mean(y)) / denom)
                row[name] = slope
        if predictors:
            vals = np.array([row.get(p) for p in predictors], dtype=float)
            if np.isfinite(vals).any():
                row["r2"] = float(np.nanmean(np.square(vals)))
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
    return df_out


def compute_ranking_dataframe(
    pred_df,
    target_col,
    option_col,
    benchmark_col="benchmark",
    context_cols=None,
    topk_frac=None,
    topk_min=1,
    require_single_fold_task=False,
):
    rows = []
    if pred_df.empty:
        return pd.DataFrame()

    context_cols = [c for c in (context_cols or []) if c and c != benchmark_col]
    if context_cols:
        pred_df, context_cols, _ = _ensure_context_columns(pred_df, context_cols)

    effective_option_col = option_col
    if effective_option_col == benchmark_col:
        fallback_candidates = [
            "train_dataset",
            "train_dataset_encoder",
            "train_dataset_model_family_encoder",
            "model_family_encoder",
            "run_name",
            "run_id",
        ]
        fallback = next((c for c in fallback_candidates if c in pred_df.columns), None)
        if fallback is not None:
            print(
                "Warning: ranking option column matches benchmark column "
                f"('{benchmark_col}'). Falling back to option grouping by '{fallback}'."
            )
            effective_option_col = fallback
        else:
            # Keep old behavior if no fallback exists, but avoid duplicate labels.
            effective_option_col = "__ranking_option_fallback__"
            pred_df = pred_df.copy()
            pred_df[effective_option_col] = pred_df[benchmark_col].astype(str)

    required_cols = list(dict.fromkeys([benchmark_col] + context_cols + [effective_option_col, "prediction", target_col]))
    missing_cols = [c for c in required_cols if c not in pred_df.columns]
    if missing_cols:
        print(
            "Warning: cannot compute ranking summary; missing columns: "
            + ", ".join(missing_cols)
        )
        return pd.DataFrame()
    df = pred_df.dropna(subset=required_cols).copy()
    if df.empty:
        return pd.DataFrame()

    group_cols = [benchmark_col] + context_cols
    if require_single_fold_task:
        if "fold" not in df.columns:
            print(
                "Warning: require_single_fold_task=True but 'fold' column is missing; "
                "continuing without fold-consistency filtering."
            )
        else:
            task_key = "__rank_task_key__"
            df[task_key] = list(zip(*[df[col] for col in group_cols]))
            fold_counts = (
                df.groupby(task_key, dropna=False)["fold"]
                .nunique(dropna=True)
                .astype(int)
            )
            valid_task_keys = set(fold_counts[fold_counts <= 1].index.tolist())
            skipped = int((fold_counts > 1).sum())
            if skipped > 0:
                print(
                    "Warning: skipped "
                    f"{skipped}/{int(len(fold_counts))} ranking tasks due to mixed-fold options."
                )
            if not valid_task_keys:
                return pd.DataFrame()
            df = df[df[task_key].isin(valid_task_keys)].copy()
            df.drop(columns=[task_key], inplace=True)

    for group_key, sub in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            group_key = (group_key,)
        group_map = {col: group_key[i] for i, col in enumerate(group_cols)}
        grouped = sub.groupby(effective_option_col).agg(
            pred_mean=("prediction", "mean"),
            true_mean=(target_col, "mean"),
            n=("prediction", "size"),
        )
        if len(grouped) < 2:
            continue

        grouped = grouped.sort_values("pred_mean", ascending=False)
        true_best = grouped["true_mean"].max()
        true_best_idx = grouped["true_mean"].idxmax()
        pred_best_idx = grouped["pred_mean"].idxmax()
        pred_best_true = grouped.loc[pred_best_idx, "true_mean"]
        pred_top3 = list(grouped.index[:3])

        rank_true = grouped["true_mean"].rank(ascending=False, method="min")
        rank_pred = grouped["pred_mean"].rank(ascending=False, method="min")
        n_options = int(len(grouped))
        denom = float(max(n_options - 1, 1))
        true_rank_pct = (rank_true - 1.0) / denom
        pred_rank_pct = (rank_pred - 1.0) / denom
        abs_rank_error = (rank_pred - rank_true).abs()
        abs_rank_pct_error = (pred_rank_pct - true_rank_pct).abs()
        top1 = int(pred_best_idx == true_best_idx)
        top3 = int(rank_true.loc[pred_best_idx] <= 3)
        regret = float(true_best - pred_best_true)
        spearman = spearman_corr(grouped["true_mean"].to_numpy(), grouped["pred_mean"].to_numpy())
        kendall = kendall_tau_b(grouped["true_mean"].to_numpy(), grouped["pred_mean"].to_numpy())
        cindex = pairwise_cindex(grouped["true_mean"].to_numpy(), grouped["pred_mean"].to_numpy())

        topk = np.nan
        topk_k = np.nan
        topk_frac_out = np.nan
        if topk_frac is not None and topk_frac > 0:
            k = int(math.ceil(float(topk_frac) * n_options))
            if topk_min is not None:
                k = max(int(topk_min), k)
            k = min(k, n_options)
            topk = int(rank_true.loc[pred_best_idx] <= k)
            topk_k = int(k)
            topk_frac_out = float(topk_frac)

        rows.append({
            **group_map,
            "n_options": n_options,
            "top1": top1,
            "top3": top3,
            "topk": topk,
            "topk_k": topk_k,
            "topk_frac": topk_frac_out,
            "regret": regret,
            "spearman": spearman,
            "kendall_tau": kendall,
            "pairwise_cindex": cindex,
            "mean_abs_rank_error": float(abs_rank_error.mean()),
            "median_abs_rank_error": float(abs_rank_error.median()),
            "mean_abs_rank_pct_error": float(abs_rank_pct_error.mean()),
            "median_abs_rank_pct_error": float(abs_rank_pct_error.median()),
            "true_best_option": str(true_best_idx),
            "pred_best_option": str(pred_best_idx),
            "pred_top3_options": ",".join(str(x) for x in pred_top3),
            "pred_best_true_rank": int(rank_true.loc[pred_best_idx]),
            "pred_best_true_rank_pct": float(true_rank_pct.loc[pred_best_idx]),
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    overall = {
        benchmark_col: "__overall__",
        "n_options": int(df_out["n_options"].sum()),
        "top1": float(df_out["top1"].mean()),
        "top3": float(df_out["top3"].mean()),
        "topk": float(df_out["topk"].mean()) if "topk" in df_out.columns else np.nan,
        "topk_k": float(df_out["topk_k"].mean()) if "topk_k" in df_out.columns else np.nan,
        "topk_frac": float(df_out["topk_frac"].mean()) if "topk_frac" in df_out.columns else np.nan,
        "regret": float(df_out["regret"].mean()),
        "spearman": float(df_out["spearman"].mean()),
        "kendall_tau": float(df_out["kendall_tau"].mean())
        if "kendall_tau" in df_out.columns
        else np.nan,
        "pairwise_cindex": float(df_out["pairwise_cindex"].mean())
        if "pairwise_cindex" in df_out.columns
        else np.nan,
        "mean_abs_rank_error": float(df_out["mean_abs_rank_error"].mean())
        if "mean_abs_rank_error" in df_out.columns
        else np.nan,
        "median_abs_rank_error": float(df_out["median_abs_rank_error"].mean())
        if "median_abs_rank_error" in df_out.columns
        else np.nan,
        "mean_abs_rank_pct_error": float(df_out["mean_abs_rank_pct_error"].mean())
        if "mean_abs_rank_pct_error" in df_out.columns
        else np.nan,
        "median_abs_rank_pct_error": float(df_out["median_abs_rank_pct_error"].mean())
        if "median_abs_rank_pct_error" in df_out.columns
        else np.nan,
        "true_best_option": "n/a",
        "pred_best_option": "n/a",
    }
    for col in context_cols:
        overall[col] = "__overall__"
    df_out = pd.concat([df_out, pd.DataFrame([overall])], ignore_index=True)
    return df_out


def compute_ranking_summary(
    pred_df,
    target_col,
    option_col,
    output_path,
    benchmark_col="benchmark",
    context_cols=None,
    topk_frac=None,
    topk_min=1,
    require_single_fold_task=False,
):
    df_out = compute_ranking_dataframe(
        pred_df,
        target_col,
        option_col,
        benchmark_col,
        context_cols=context_cols,
        topk_frac=topk_frac,
        topk_min=topk_min,
        require_single_fold_task=require_single_fold_task,
    )
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
        return df_out.to_dict(orient="records")
    out_path = Path(output_path)
    if out_path.exists():
        out_path.unlink()
    return []


def compute_rank_detail_rows(
    pred_df,
    target_col,
    option_col,
    benchmark_col="benchmark",
    context_cols=None,
    require_single_fold_task=False,
):
    rows = []
    if pred_df.empty:
        return pd.DataFrame()

    context_cols = [c for c in (context_cols or []) if c and c != benchmark_col]
    if context_cols:
        pred_df, context_cols, _ = _ensure_context_columns(pred_df, context_cols)

    effective_option_col = option_col
    if effective_option_col == benchmark_col:
        fallback_candidates = [
            "train_dataset",
            "train_dataset_encoder",
            "train_dataset_model_family_encoder",
            "model_family_encoder",
            "run_name",
            "run_id",
        ]
        fallback = next((c for c in fallback_candidates if c in pred_df.columns), None)
        if fallback is not None:
            print(
                "Warning: rank-detail option column matches benchmark column "
                f"('{benchmark_col}'). Falling back to '{fallback}'."
            )
            effective_option_col = fallback
        else:
            effective_option_col = "__ranking_option_fallback__"
            pred_df = pred_df.copy()
            pred_df[effective_option_col] = pred_df[benchmark_col].astype(str)

    required_cols = list(dict.fromkeys([benchmark_col] + context_cols + [effective_option_col, "prediction", target_col]))
    missing_cols = [c for c in required_cols if c not in pred_df.columns]
    if missing_cols:
        print(
            "Warning: cannot compute rank-detail rows; missing columns: "
            + ", ".join(missing_cols)
        )
        return pd.DataFrame()
    df = pred_df.dropna(subset=required_cols).copy()
    if df.empty:
        return pd.DataFrame()

    output_option_col = effective_option_col if effective_option_col != benchmark_col else "option"
    group_cols = [benchmark_col] + context_cols
    if require_single_fold_task:
        if "fold" not in df.columns:
            print(
                "Warning: require_single_fold_task=True but 'fold' column is missing; "
                "continuing without fold-consistency filtering for rank-detail rows."
            )
        else:
            task_key = "__rank_task_key__"
            df[task_key] = list(zip(*[df[col] for col in group_cols]))
            fold_counts = (
                df.groupby(task_key, dropna=False)["fold"]
                .nunique(dropna=True)
                .astype(int)
            )
            valid_task_keys = set(fold_counts[fold_counts <= 1].index.tolist())
            skipped = int((fold_counts > 1).sum())
            if skipped > 0:
                print(
                    "Warning: skipped "
                    f"{skipped}/{int(len(fold_counts))} rank-detail tasks due to mixed-fold options."
                )
            if not valid_task_keys:
                return pd.DataFrame()
            df = df[df[task_key].isin(valid_task_keys)].copy()
            df.drop(columns=[task_key], inplace=True)

    for group_key, sub in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            group_key = (group_key,)
        group_map = {col: group_key[i] for i, col in enumerate(group_cols)}
        grouped = sub.groupby(effective_option_col).agg(
            pred_mean=("prediction", "mean"),
            true_mean=(target_col, "mean"),
            n=("prediction", "size"),
        )
        if len(grouped) < 2:
            continue

        grouped["true_rank"] = grouped["true_mean"].rank(ascending=False, method="min")
        grouped["pred_rank"] = grouped["pred_mean"].rank(ascending=False, method="min")
        n_options = int(len(grouped))
        denom = float(max(n_options - 1, 1))
        grouped["true_rank_pct"] = (grouped["true_rank"] - 1.0) / denom
        grouped["pred_rank_pct"] = (grouped["pred_rank"] - 1.0) / denom
        grouped["rank_error"] = grouped["pred_rank"] - grouped["true_rank"]
        grouped["abs_rank_error"] = grouped["rank_error"].abs()
        grouped["rank_pct_error"] = grouped["pred_rank_pct"] - grouped["true_rank_pct"]
        grouped["abs_rank_pct_error"] = grouped["rank_pct_error"].abs()

        for option, row in grouped.iterrows():
            rows.append({
                **group_map,
                output_option_col: option,
                "n_options": n_options,
                "true_mean": float(row["true_mean"]),
                "pred_mean": float(row["pred_mean"]),
                "true_rank": int(row["true_rank"]),
                "pred_rank": int(row["pred_rank"]),
                "true_rank_pct": float(row["true_rank_pct"]),
                "pred_rank_pct": float(row["pred_rank_pct"]),
                "rank_error": float(row["rank_error"]),
                "abs_rank_error": float(row["abs_rank_error"]),
                "rank_pct_error": float(row["rank_pct_error"]),
                "abs_rank_pct_error": float(row["abs_rank_pct_error"]),
                "n": int(row["n"]),
            })

    return pd.DataFrame(rows)


def write_rank_detail_rows(
    pred_df,
    target_col,
    option_col,
    output_path,
    benchmark_col="benchmark",
    context_cols=None,
    require_single_fold_task=False,
):
    df_out = compute_rank_detail_rows(
        pred_df,
        target_col,
        option_col,
        benchmark_col,
        context_cols=context_cols,
        require_single_fold_task=require_single_fold_task,
    )
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
    else:
        out_path = Path(output_path)
        if out_path.exists():
            out_path.unlink()
    return df_out


def compute_holdout_placement_rows(
    pred_df,
    reference_df,
    target_col,
    option_col,
    task_cols=None,
    fold_col="fold",
    holdout_group_col=None,
):
    if pred_df is None or pred_df.empty:
        return pd.DataFrame()
    if reference_df is None or reference_df.empty:
        return pd.DataFrame()
    # Defensive: merged tables may contain duplicate column labels.
    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()].copy()
    reference_df = reference_df.loc[:, ~reference_df.columns.duplicated()].copy()

    task_cols = [c for c in (task_cols or ["benchmark"]) if c]
    pred_target_col = target_col if target_col in pred_df.columns else "target"
    if pred_target_col not in pred_df.columns:
        return pd.DataFrame()

    pred_required = [fold_col, option_col, "prediction", pred_target_col] + task_cols
    pred_missing = [c for c in pred_required if c not in pred_df.columns]
    if pred_missing:
        print(
            "Warning: cannot compute holdout placement rows; missing prediction columns: "
            + ", ".join(pred_missing)
        )
        return pd.DataFrame()

    ref_required = [option_col, target_col] + task_cols
    ref_missing = [c for c in ref_required if c not in reference_df.columns]
    if ref_missing:
        print(
            "Warning: cannot compute holdout placement rows; missing reference columns: "
            + ", ".join(ref_missing)
        )
        return pd.DataFrame()

    holdout_group_enabled = (
        holdout_group_col is not None
        and holdout_group_col in pred_df.columns
        and holdout_group_col in reference_df.columns
    )

    pred_cols = pred_required + ([holdout_group_col] if holdout_group_enabled else [])
    ref_cols = ref_required + ([holdout_group_col] if holdout_group_enabled else [])
    pred_cols = list(dict.fromkeys(pred_cols))
    ref_cols = list(dict.fromkeys(ref_cols))
    pred_work = pred_df[pred_cols].dropna(
        subset=[fold_col, option_col, "prediction", pred_target_col] + task_cols
    )
    ref_work = reference_df[ref_cols].dropna(subset=[option_col, target_col] + task_cols)
    if pred_work.empty or ref_work.empty:
        return pd.DataFrame()

    heldout_group_cols = [fold_col] + task_cols + [option_col]
    if holdout_group_enabled:
        heldout_group_cols.append(holdout_group_col)
    heldout_group_cols = list(dict.fromkeys(heldout_group_cols))
    heldout_df = (
        pred_work.groupby(heldout_group_cols, dropna=False)
        .agg(
            heldout_true=(pred_target_col, "mean"),
            heldout_pred=("prediction", "mean"),
            heldout_n=(pred_target_col, "size"),
        )
        .reset_index()
    )
    if heldout_df.empty:
        return pd.DataFrame()

    anchor_group_cols = task_cols + [option_col]
    if holdout_group_enabled:
        anchor_group_cols.append(holdout_group_col)
    anchor_group_cols = list(dict.fromkeys(anchor_group_cols))
    anchor_df = (
        ref_work.groupby(anchor_group_cols, dropna=False)
        .agg(anchor_true=(target_col, "mean"), anchor_n=(target_col, "size"))
        .reset_index()
    )
    if anchor_df.empty:
        return pd.DataFrame()

    anchor_by_task = {}
    for task_key, sub in anchor_df.groupby(task_cols, dropna=False):
        if not isinstance(task_key, tuple):
            task_key = (task_key,)
        anchor_by_task[task_key] = sub.reset_index(drop=True)

    rows = []
    for _, h in heldout_df.iterrows():
        task_key = tuple(h[col] for col in task_cols)
        anchors = anchor_by_task.get(task_key)
        if anchors is None or anchors.empty:
            continue

        anchors = anchors[anchors[option_col] != h[option_col]]
        if holdout_group_enabled:
            anchors = anchors[anchors[holdout_group_col] != h[holdout_group_col]]
        if anchors.empty:
            continue

        holdout_true = float(h["heldout_true"])
        holdout_pred = float(h["heldout_pred"])
        anchor_true = anchors["anchor_true"].to_numpy(dtype=float)
        n_options = int(len(anchor_true) + 1)
        denom = float(max(n_options - 1, 1))

        true_rank = 1 + int(np.sum(anchor_true > holdout_true))
        pred_rank = 1 + int(np.sum(anchor_true > holdout_pred))
        true_rank_pct = float((true_rank - 1) / denom)
        pred_rank_pct = float((pred_rank - 1) / denom)
        rank_error = float(pred_rank - true_rank)
        abs_rank_error = float(abs(rank_error))
        rank_pct_error = float(pred_rank_pct - true_rank_pct)
        abs_rank_pct_error = float(abs(rank_pct_error))

        true_delta = holdout_true - anchor_true
        pred_delta = holdout_pred - anchor_true
        valid_pairs = true_delta != 0.0
        if np.any(valid_pairs):
            pred_sign = np.sign(pred_delta[valid_pairs])
            true_sign = np.sign(true_delta[valid_pairs])
            pairwise = (pred_sign == true_sign).astype(float)
            pairwise[pred_sign == 0.0] = 0.5
            pairwise_win_rate = float(np.mean(pairwise))
            pairwise_n = int(valid_pairs.sum())
        else:
            pairwise_win_rate = np.nan
            pairwise_n = 0

        all_true = np.concatenate(([holdout_true], anchor_true))
        all_pred_for_select = np.concatenate(([holdout_pred], anchor_true))
        selected_idx = int(np.argmax(all_pred_for_select))
        selected_true = float(all_true[selected_idx])
        true_best = float(np.max(all_true))
        regret = float(true_best - selected_true)

        out = {
            fold_col: h[fold_col],
            option_col: h[option_col],
            "heldout_true": holdout_true,
            "heldout_pred": holdout_pred,
            "heldout_n": int(h["heldout_n"]),
            "n_anchors": int(len(anchor_true)),
            "n_options": n_options,
            "true_rank": int(true_rank),
            "pred_rank": int(pred_rank),
            "true_rank_pct": true_rank_pct,
            "pred_rank_pct": pred_rank_pct,
            "rank_error": rank_error,
            "abs_rank_error": abs_rank_error,
            "rank_pct_error": rank_pct_error,
            "abs_rank_pct_error": abs_rank_pct_error,
            "pairwise_win_rate": pairwise_win_rate,
            "pairwise_n": pairwise_n,
            "regret": regret,
            "selected_is_holdout": int(selected_idx == 0),
            "holdout_true_top1": int(true_rank == 1),
            "holdout_pred_top1": int(pred_rank == 1),
        }
        if holdout_group_enabled:
            out[holdout_group_col] = h[holdout_group_col]
        for col in task_cols:
            out[col] = h[col]
        rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def compute_insertion_rows_from_task_predictions(
    task_pred_df,
    option_col,
    task_cols=None,
    fold_col="fold",
    holdout_flag_col="is_holdout_option",
):
    if task_pred_df is None or task_pred_df.empty:
        return pd.DataFrame()
    task_cols = [c for c in (task_cols or ["benchmark"]) if c in task_pred_df.columns]
    required = [fold_col, option_col, "prediction", "target", holdout_flag_col] + task_cols
    missing = [c for c in required if c not in task_pred_df.columns]
    if missing:
        print(
            "Warning: cannot compute insertion rows; missing columns: "
            + ", ".join(missing)
        )
        return pd.DataFrame()

    df = task_pred_df.dropna(subset=required).copy()
    if df.empty:
        return pd.DataFrame()

    group_cols = [fold_col] + task_cols
    rows = []
    for group_key, sub in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            group_key = (group_key,)
        group_map = {col: group_key[i] for i, col in enumerate(group_cols)}
        grouped = (
            sub.groupby(option_col, dropna=False)
            .agg(
                pred_mean=("prediction", "mean"),
                true_mean=("target", "mean"),
                is_holdout=(holdout_flag_col, "max"),
                n=("prediction", "size"),
            )
            .reset_index()
        )
        holdouts = grouped[grouped["is_holdout"] > 0.5]
        anchors = grouped[grouped["is_holdout"] <= 0.5]
        if holdouts.empty or anchors.empty:
            continue

        anchor_true = anchors["true_mean"].to_numpy(dtype=float)
        anchor_pred = anchors["pred_mean"].to_numpy(dtype=float)
        n_options = int(len(anchor_true) + 1)
        denom = float(max(n_options - 1, 1))

        for _, h in holdouts.iterrows():
            holdout_true = float(h["true_mean"])
            holdout_pred = float(h["pred_mean"])

            true_rank = 1 + int(np.sum(anchor_true > holdout_true))
            pred_rank = 1 + int(np.sum(anchor_pred > holdout_pred))
            true_rank_pct = float((true_rank - 1) / denom)
            pred_rank_pct = float((pred_rank - 1) / denom)
            rank_error = float(pred_rank - true_rank)
            abs_rank_error = float(abs(rank_error))
            rank_pct_error = float(pred_rank_pct - true_rank_pct)
            abs_rank_pct_error = float(abs(rank_pct_error))

            true_delta = holdout_true - anchor_true
            pred_delta = holdout_pred - anchor_pred
            valid_pairs = true_delta != 0.0
            if np.any(valid_pairs):
                pred_sign = np.sign(pred_delta[valid_pairs])
                true_sign = np.sign(true_delta[valid_pairs])
                pairwise = (pred_sign == true_sign).astype(float)
                pairwise[pred_sign == 0.0] = 0.5
                pairwise_win_rate = float(np.mean(pairwise))
                pairwise_n = int(valid_pairs.sum())
            else:
                pairwise_win_rate = np.nan
                pairwise_n = 0

            all_true = np.concatenate(([holdout_true], anchor_true))
            all_pred = np.concatenate(([holdout_pred], anchor_pred))
            selected_idx = int(np.argmax(all_pred))
            selected_true = float(all_true[selected_idx])
            true_best = float(np.max(all_true))
            regret = float(true_best - selected_true)

            rows.append(
                {
                    **group_map,
                    option_col: h[option_col],
                    "heldout_true": holdout_true,
                    "heldout_pred": holdout_pred,
                    "heldout_n": int(h["n"]),
                    "n_anchors": int(len(anchor_true)),
                    "n_options": n_options,
                    "true_rank": int(true_rank),
                    "pred_rank": int(pred_rank),
                    "true_rank_pct": true_rank_pct,
                    "pred_rank_pct": pred_rank_pct,
                    "rank_error": rank_error,
                    "abs_rank_error": abs_rank_error,
                    "rank_pct_error": rank_pct_error,
                    "abs_rank_pct_error": abs_rank_pct_error,
                    "pairwise_win_rate": pairwise_win_rate,
                    "pairwise_n": pairwise_n,
                    "regret": regret,
                    "selected_is_holdout": int(selected_idx == 0),
                    "holdout_true_top1": int(true_rank == 1),
                    "holdout_pred_top1": int(pred_rank == 1),
                }
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def summarize_holdout_placement_rows(detail_df, fold_col="fold"):
    if detail_df is None or detail_df.empty:
        return pd.DataFrame()
    if fold_col not in detail_df.columns:
        return pd.DataFrame()

    per_fold_rows = []
    for fold, sub in detail_df.groupby(fold_col, dropna=False):
        per_fold_rows.append(
            {
                fold_col: fold,
                "n_tasks": int(len(sub)),
                "abs_rank_error": float(sub["abs_rank_error"].mean()),
                "abs_rank_pct_error": float(sub["abs_rank_pct_error"].mean()),
                "pairwise_win_rate": float(sub["pairwise_win_rate"].mean()),
                "regret": float(sub["regret"].mean()),
                "holdout_true_top1": float(sub["holdout_true_top1"].mean()),
                "holdout_pred_top1": float(sub["holdout_pred_top1"].mean()),
                "rank_spearman": spearman_corr(
                    sub["true_rank_pct"].to_numpy(dtype=float),
                    sub["pred_rank_pct"].to_numpy(dtype=float),
                ),
                "rank_kendall": kendall_tau_b(
                    sub["true_rank_pct"].to_numpy(dtype=float),
                    sub["pred_rank_pct"].to_numpy(dtype=float),
                ),
            }
        )
    per_fold = pd.DataFrame(per_fold_rows)
    if per_fold.empty:
        return pd.DataFrame()

    overall = {
        fold_col: "__overall__",
        "n_folds": int(len(per_fold)),
        "n_tasks": int(len(detail_df)),
        "abs_rank_error": float(per_fold["abs_rank_error"].mean()),
        "abs_rank_pct_error": float(per_fold["abs_rank_pct_error"].mean()),
        "pairwise_win_rate": float(per_fold["pairwise_win_rate"].mean()),
        "regret": float(per_fold["regret"].mean()),
        "holdout_true_top1": float(per_fold["holdout_true_top1"].mean()),
        "holdout_pred_top1": float(per_fold["holdout_pred_top1"].mean()),
        "rank_spearman": float(per_fold["rank_spearman"].mean()),
        "rank_kendall": float(per_fold["rank_kendall"].mean()),
        "abs_rank_error_micro": float(detail_df["abs_rank_error"].mean()),
        "abs_rank_pct_error_micro": float(detail_df["abs_rank_pct_error"].mean()),
        "pairwise_win_rate_micro": float(detail_df["pairwise_win_rate"].mean()),
        "regret_micro": float(detail_df["regret"].mean()),
        "holdout_true_top1_micro": float(detail_df["holdout_true_top1"].mean()),
        "holdout_pred_top1_micro": float(detail_df["holdout_pred_top1"].mean()),
        "rank_spearman_micro": spearman_corr(
            detail_df["true_rank_pct"].to_numpy(dtype=float),
            detail_df["pred_rank_pct"].to_numpy(dtype=float),
        ),
        "rank_kendall_micro": kendall_tau_b(
            detail_df["true_rank_pct"].to_numpy(dtype=float),
            detail_df["pred_rank_pct"].to_numpy(dtype=float),
        ),
    }

    valid = per_fold["rank_spearman"].to_numpy(dtype=float)
    valid = valid[np.isfinite(valid)]
    if valid.size:
        clipped = np.clip(valid, -0.999999, 0.999999)
        overall["rank_spearman_fisher"] = float(np.tanh(np.mean(np.arctanh(clipped))))
    else:
        overall["rank_spearman_fisher"] = np.nan

    return pd.concat([per_fold, pd.DataFrame([overall])], ignore_index=True)


def write_holdout_placement_outputs(
    pred_df,
    reference_df,
    target_col,
    option_col,
    detail_path,
    summary_path,
    task_cols=None,
    fold_col="fold",
    holdout_group_col=None,
):
    detail_df = compute_holdout_placement_rows(
        pred_df=pred_df,
        reference_df=reference_df,
        target_col=target_col,
        option_col=option_col,
        task_cols=task_cols,
        fold_col=fold_col,
        holdout_group_col=holdout_group_col,
    )
    if detail_df.empty:
        for p in (Path(detail_path), Path(summary_path)):
            if p.exists():
                p.unlink()
        return pd.DataFrame(), pd.DataFrame()

    detail_df.to_csv(detail_path, index=False)
    summary_df = summarize_holdout_placement_rows(detail_df, fold_col=fold_col)
    if summary_df.empty:
        p = Path(summary_path)
        if p.exists():
            p.unlink()
    else:
        summary_df.to_csv(summary_path, index=False)
    return summary_df, detail_df


def write_insertion_outputs_from_task_predictions(
    task_pred_df,
    option_col,
    detail_path,
    summary_path,
    task_cols=None,
    fold_col="fold",
):
    detail_df = compute_insertion_rows_from_task_predictions(
        task_pred_df=task_pred_df,
        option_col=option_col,
        task_cols=task_cols,
        fold_col=fold_col,
    )
    if detail_df.empty:
        for p in (Path(detail_path), Path(summary_path)):
            if p.exists():
                p.unlink()
        return pd.DataFrame(), pd.DataFrame()

    detail_df.to_csv(detail_path, index=False)
    summary_df = summarize_holdout_placement_rows(detail_df, fold_col=fold_col)
    if summary_df.empty:
        p = Path(summary_path)
        if p.exists():
            p.unlink()
    else:
        summary_df.to_csv(summary_path, index=False)
    return summary_df, detail_df


def write_direction_audit(pred_df, target_col, option_col, output_path, benchmark_col="benchmark"):
    if pred_df.empty:
        output_path.write_text("No predictions available for direction audit.")
        return

    if target_col not in pred_df.columns and "target" in pred_df.columns:
        target_col = "target"

    effective_option_col = option_col
    if effective_option_col == benchmark_col:
        fallback_candidates = [
            "train_dataset",
            "train_dataset_encoder",
            "train_dataset_model_family_encoder",
            "model_family_encoder",
            "run_name",
            "run_id",
        ]
        fallback = next((c for c in fallback_candidates if c in pred_df.columns), None)
        if fallback is not None:
            print(
                "Warning: direction-audit option column matches benchmark column "
                f"('{benchmark_col}'). Falling back to '{fallback}'."
            )
            effective_option_col = fallback
        else:
            effective_option_col = "__ranking_option_fallback__"
            pred_df = pred_df.copy()
            pred_df[effective_option_col] = pred_df[benchmark_col].astype(str)

    for benchmark in sorted(pred_df[benchmark_col].dropna().unique()):
        sub = pred_df[pred_df[benchmark_col] == benchmark]
        grouped = sub.groupby(effective_option_col).agg(
            true_mean=(target_col, "mean"),
            pred_mean=("prediction", "mean"),
        )
        if len(grouped) < 2:
            continue

        grouped = grouped.copy()
        grouped["true_rank_desc"] = grouped["true_mean"].rank(ascending=False, method="min")
        grouped["pred_rank_desc"] = grouped["pred_mean"].rank(ascending=False, method="min")
        grouped["pred_rank_asc"] = grouped["pred_mean"].rank(ascending=True, method="min")

        true_best = grouped["true_mean"].idxmax()
        pred_best_max = grouped["pred_mean"].idxmax()
        pred_best_min = grouped["pred_mean"].idxmin()
        spearman_max = spearman_corr(grouped["true_mean"].to_numpy(), grouped["pred_mean"].to_numpy())
        spearman_min = spearman_corr(grouped["true_mean"].to_numpy(), -grouped["pred_mean"].to_numpy())

        lines = [
            f"Benchmark: {benchmark}",
            f"Options: {len(grouped)}",
            f"True best (max): {true_best}",
            f"Pred best (max): {pred_best_max} (top1={int(pred_best_max == true_best)})",
            f"Pred best (min): {pred_best_min} (top1={int(pred_best_min == true_best)})",
            f"Spearman(pred, true): {spearman_max:.4f}",
            f"Spearman(-pred, true): {spearman_min:.4f}",
            "",
            "option,true_mean,pred_mean,true_rank_desc,pred_rank_desc,pred_rank_asc",
        ]
        for option, row in grouped.sort_values("pred_mean", ascending=False).iterrows():
            lines.append(
                f"{option},{row['true_mean']:.6f},{row['pred_mean']:.6f},"
                f"{int(row['true_rank_desc'])},{int(row['pred_rank_desc'])},"
                f"{int(row['pred_rank_asc'])}"
            )
        output_path.write_text("\n".join(lines))
        return

    output_path.write_text("No benchmark with >=2 options found for direction audit.")


def compute_constant_selector(
    df,
    target_col,
    option_col,
    chosen_option,
    benchmark_col="benchmark",
    context_cols=None,
    topk_frac=None,
    topk_min=1,
):
    rows = []
    if df.empty:
        return pd.DataFrame()
    context_cols = [c for c in (context_cols or []) if c and c != benchmark_col]
    if context_cols:
        df, context_cols, _ = _ensure_context_columns(df, context_cols)
    effective_option_col = option_col
    if effective_option_col == benchmark_col:
        fallback_candidates = [
            "train_dataset",
            "train_dataset_encoder",
            "train_dataset_model_family_encoder",
            "model_family_encoder",
            "run_name",
            "run_id",
        ]
        fallback = next((c for c in fallback_candidates if c in df.columns), None)
        if fallback is not None:
            print(
                "Warning: constant-selector option column matches benchmark column "
                f"('{benchmark_col}'). Falling back to '{fallback}'."
            )
            effective_option_col = fallback
        else:
            effective_option_col = "__ranking_option_fallback__"
            df = df.copy()
            df[effective_option_col] = df[benchmark_col].astype(str)

    required_cols = list(dict.fromkeys([benchmark_col] + context_cols + [effective_option_col, target_col]))
    df = df.dropna(subset=required_cols)
    if df.empty:
        return pd.DataFrame()

    group_cols = [benchmark_col] + context_cols
    for group_key, sub in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            group_key = (group_key,)
        group_map = {col: group_key[i] for i, col in enumerate(group_cols)}
        grouped = sub.groupby(effective_option_col).agg(true_mean=(target_col, "mean"))
        if len(grouped) < 2:
            continue

        true_best = grouped["true_mean"].max()
        true_best_idx = grouped["true_mean"].idxmax()
        if chosen_option not in grouped.index:
            continue

        pred_best_true = grouped.loc[chosen_option, "true_mean"]
        rank_true = grouped["true_mean"].rank(ascending=False, method="min")
        top1 = int(chosen_option == true_best_idx)
        top3 = int(rank_true.loc[chosen_option] <= 3)
        regret = float(true_best - pred_best_true)

        topk = np.nan
        topk_k = np.nan
        topk_frac_out = np.nan
        if topk_frac is not None and topk_frac > 0:
            n_options = int(len(grouped))
            k = int(math.ceil(float(topk_frac) * n_options))
            if topk_min is not None:
                k = max(int(topk_min), k)
            k = min(k, n_options)
            topk = int(rank_true.loc[chosen_option] <= k)
            topk_k = int(k)
            topk_frac_out = float(topk_frac)

        rows.append({
            **group_map,
            "n_options": int(len(grouped)),
            "top1": top1,
            "top3": top3,
            "topk": topk,
            "topk_k": topk_k,
            "topk_frac": topk_frac_out,
            "regret": regret,
            "spearman": np.nan,
            "true_best_option": str(true_best_idx),
            "pred_best_option": str(chosen_option),
            "pred_top3_options": str(chosen_option),
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    overall = {
        benchmark_col: "__overall__",
        "n_options": int(df_out["n_options"].sum()),
        "top1": float(df_out["top1"].mean()),
        "top3": float(df_out["top3"].mean()),
        "topk": float(df_out["topk"].mean()) if "topk" in df_out.columns else np.nan,
        "topk_k": float(df_out["topk_k"].mean()) if "topk_k" in df_out.columns else np.nan,
        "topk_frac": float(df_out["topk_frac"].mean()) if "topk_frac" in df_out.columns else np.nan,
        "regret": float(df_out["regret"].mean()),
        "spearman": np.nan,
        "true_best_option": "n/a",
        "pred_best_option": "n/a",
    }
    for col in context_cols:
        overall[col] = "__overall__"
    df_out = pd.concat([df_out, pd.DataFrame([overall])], ignore_index=True)
    return df_out


def compute_baseline_rankings(
    df,
    target_col,
    option_col,
    output_path,
    selectors,
    benchmark_col="benchmark",
    context_cols=None,
    topk_frac=None,
    topk_min=1,
):
    baseline_frames = []
    if df.empty:
        return pd.DataFrame()

    for selector in selectors:
        sel_type = selector.get("type", "metric")
        name = selector.get("name")
        if sel_type == "metric":
            col = selector.get("column")
            if not col or col not in df.columns:
                continue
            # Keep full row context so ranking-context group columns can still be derived.
            sub = df.copy()
            sub = sub.dropna(subset=[benchmark_col, option_col, target_col, col])
            if sub.empty:
                continue
            pred_df = sub.copy()
            pred_df["prediction"] = pred_df[col]
            if selector.get("direction", 1) < 0:
                pred_df["prediction"] = -pred_df["prediction"]
            ranking_df = compute_ranking_dataframe(
                pred_df,
                target_col,
                option_col,
                benchmark_col,
                context_cols=context_cols,
                topk_frac=topk_frac,
                topk_min=topk_min,
            )
        elif sel_type == "constant":
            option = selector.get("option")
            if not option:
                continue
            ranking_df = compute_constant_selector(
                df,
                target_col,
                option_col,
                option,
                benchmark_col,
                context_cols=context_cols,
                topk_frac=topk_frac,
                topk_min=topk_min,
            )
        elif sel_type == "best_avg":
            valid = df.dropna(subset=[option_col, target_col])
            if valid.empty:
                continue
            option = valid.groupby(option_col)[target_col].mean().idxmax()
            ranking_df = compute_constant_selector(
                df,
                target_col,
                option_col,
                option,
                benchmark_col,
                context_cols=context_cols,
                topk_frac=topk_frac,
                topk_min=topk_min,
            )
        else:
            continue

        if ranking_df is None or ranking_df.empty:
            continue
        ranking_df.insert(0, "selector", name)
        baseline_frames.append(ranking_df)

    if not baseline_frames:
        return pd.DataFrame()

    df_out = pd.concat(baseline_frames, ignore_index=True)
    df_out.to_csv(output_path, index=False)
    return df_out


def compute_baseline_insertion_summaries(
    task_pred_df,
    option_col,
    output_path,
    selectors,
    task_cols=None,
    fold_col="fold",
):
    if task_pred_df is None or task_pred_df.empty:
        return pd.DataFrame()
    if not selectors:
        return pd.DataFrame()
    required = [option_col, fold_col, "target", "prediction", "is_holdout_option"]
    for col in required:
        if col not in task_pred_df.columns:
            return pd.DataFrame()

    baseline_frames = []
    for selector in selectors:
        sel_type = selector.get("type", "metric")
        name = selector.get("name")
        work = task_pred_df.copy()

        if sel_type == "metric":
            col = selector.get("column")
            if not col or col not in work.columns:
                continue
            work = work.dropna(subset=[col]).copy()
            if work.empty:
                continue
            work["prediction"] = pd.to_numeric(work[col], errors="coerce")
            if selector.get("direction", 1) < 0:
                work["prediction"] = -work["prediction"]
        elif sel_type == "constant":
            chosen_option = selector.get("option")
            if not chosen_option:
                continue
            work["prediction"] = (work[option_col].astype(str) == str(chosen_option)).astype(float)
        elif sel_type == "best_avg":
            valid = work.dropna(subset=[option_col, "target"])
            if valid.empty:
                continue
            chosen_option = valid.groupby(option_col, dropna=False)["target"].mean().idxmax()
            work["prediction"] = (work[option_col] == chosen_option).astype(float)
        else:
            continue

        work = work.dropna(subset=[fold_col, option_col, "target", "prediction", "is_holdout_option"])
        if work.empty:
            continue

        detail_df = compute_insertion_rows_from_task_predictions(
            task_pred_df=work,
            option_col=option_col,
            task_cols=task_cols,
            fold_col=fold_col,
        )
        if detail_df.empty:
            continue
        summary_df = summarize_holdout_placement_rows(detail_df, fold_col=fold_col)
        if summary_df.empty:
            continue
        summary_df.insert(0, "selector", str(name))
        baseline_frames.append(summary_df)

    if not baseline_frames:
        return pd.DataFrame()

    df_out = pd.concat(baseline_frames, ignore_index=True)
    df_out.to_csv(output_path, index=False)
    return df_out


def _format_bool(value):
    if pd.isna(value):
        return "unknown"
    if isinstance(value, bool):
        return "True" if value else "False"
    lower = str(value).strip().lower()
    if lower in ("true", "1", "yes"):
        return "True"
    if lower in ("false", "0", "no"):
        return "False"
    return "unknown"


ENCODER_CONFIG_ORDER = ("FF", "FT", "TF", "TT")


def _bool_to_char(value):
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return "T" if value else "F"
    lower = str(value).strip().lower()
    if lower in ("true", "1", "yes"):
        return "T"
    if lower in ("false", "0", "no"):
        return "F"
    return None


def ensure_encoder_config(df):
    if "encoder_config" in df.columns:
        return df
    if "pretrained" not in df.columns or "freeze" not in df.columns:
        return df
    df = df.copy()
    pre = df["pretrained"].apply(_bool_to_char)
    frz = df["freeze"].apply(_bool_to_char)
    config = pre.fillna("U") + frz.fillna("U")
    config = config.where(pre.notna() & frz.notna(), "unknown")
    df["encoder_config"] = config
    return df


def ensure_model_family(df):
    if "model_family" in df.columns:
        return df
    if "run_id" not in df.columns:
        return df
    df = df.copy()
    df["model_family"] = df["run_id"].apply(lambda value: derive_model_family(Path(str(value))))
    return df


def create_model_family_encoder_column(df):
    """
    Create a composite 'model_family_encoder' column for proper demeaning.
    
    For models like RAFT that don't have encoder variants (encoder_config is empty/NaN),
    use just the model_family. For models like CatsPP that do have encoder variants,
    combine model_family and encoder_config.
    
    This ensures that when we demean by 'encoder', RAFT models are grouped separately
    and don't get mixed with CatsPP encoder groups.
    """
    df = ensure_model_family(df)
    df = ensure_encoder_config(df)
    
    if "model_family" not in df.columns:
        return df
    
    df = df.copy()
    
    # Check if encoder_config exists and has meaningful values
    has_encoder = "encoder_config" in df.columns
    
    if has_encoder:
        # For rows with valid encoder_config, combine model_family + encoder_config
        # For rows without (e.g., RAFT), use just model_family
        df["model_family_encoder"] = df.apply(
            lambda row: (
                f"{row['model_family']}_{row['encoder_config']}"
                if pd.notna(row.get('encoder_config')) and row.get('encoder_config') not in ['', 'unknown']
                else str(row['model_family'])
            ),
            axis=1
        )
    else:
        # No encoder_config column, just use model_family
        df["model_family_encoder"] = df["model_family"].astype(str)
    
    return df


def _safe_group_tag(value):
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return "unknown"
    safe = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)


def _iter_per_encoder_groups(report_df, cv_df):
    if "model_family_encoder" in report_df.columns:
        group_col = "model_family_encoder"
        if cv_df is not None and "model_family_encoder" not in cv_df.columns:
            cv_df = create_model_family_encoder_column(cv_df)
        for group_name, group in report_df.groupby(group_col, dropna=False):
            tag = _safe_group_tag(group_name)
            cv_group = (
                cv_df[cv_df[group_col] == group_name].copy()
                if cv_df is not None
                else None
            )
            yield tag, group, cv_group
        return
    for (pretrained, freeze), group in report_df.groupby(
        ["pretrained", "freeze"], dropna=False
    ):
        tag = f"pretrained{_format_bool(pretrained)}_freeze{_format_bool(freeze)}"
        cv_group = None
        if cv_df is not None:
            cv_group = cv_df[
                (cv_df["pretrained"] == pretrained) & (cv_df["freeze"] == freeze)
            ].copy()
        yield tag, group, cv_group


def ensure_train_dataset_encoder_column(df, output_col="train_dataset_encoder"):
    if output_col in df.columns:
        return df
    if "train_dataset" not in df.columns:
        return df
    df = ensure_encoder_config(df)
    if "encoder_config" not in df.columns:
        return df
    df = df.copy()
    df[output_col] = (
        df["train_dataset"].astype(str) + "__" + df["encoder_config"].astype(str)
    )
    return df


def ensure_train_dataset_model_family_encoder_column(
    df, output_col="train_dataset_model_family_encoder"
):
    if output_col in df.columns:
        return df
    if "train_dataset" not in df.columns:
        return df
    df = create_model_family_encoder_column(df)
    if "model_family_encoder" not in df.columns:
        return df
    df = df.copy()
    df[output_col] = (
        df["train_dataset"].astype(str) + "__" + df["model_family_encoder"].astype(str)
    )
    return df


def _parse_csv_group_cols(value: str) -> List[str]:
    return [c.strip() for c in str(value or "").split(",") if c.strip()]


def _ensure_context_columns(df, cols):
    out = df
    available = []
    missing = []
    for col in cols:
        if col in out.columns:
            available.append(col)
            continue
        if col == "encoder_config":
            out = ensure_encoder_config(out)
        elif col == "model_family":
            out = ensure_model_family(out)
        elif col == "model_family_encoder":
            out = create_model_family_encoder_column(out)
        elif col == "train_dataset_encoder":
            out = ensure_train_dataset_encoder_column(out)
        elif col == "train_dataset_model_family_encoder":
            out = ensure_train_dataset_model_family_encoder_column(out)
        if col in out.columns:
            available.append(col)
        else:
            missing.append(col)
    return out, available, missing


def add_rank_target(df, source_col, group_cols, output_col):
    if source_col not in df.columns:
        return df
    df = df.copy()
    if not group_cols:
        values = df[source_col]
        ranks = values.rank(method="average", ascending=True)
        df[output_col] = ranks
        return df

    group_cols = [col for col in group_cols if col in df.columns]
    if not group_cols:
        return add_rank_target(df, source_col, [], output_col)

    df[output_col] = (
        df.groupby(group_cols, dropna=False)[source_col]
        .rank(method="average", ascending=True)
    )
    return df


def _parse_encoder_config_list(value):
    if not value:
        return []
    items = []
    for raw in str(value).split(","):
        token = raw.strip().upper()
        if not token:
            continue
        items.append(token)
    return items


def _parse_model_family_list(value):
    if not value:
        return []
    items = []
    for raw in str(value).split(","):
        token = raw.strip().lower()
        if not token:
            continue
        items.append(token)
    return items


def _parse_benchmark_list(value):
    if not value:
        return []
    items = []
    for raw in str(value).split(","):
        token = normalize_dataset_name(raw)
        if not token:
            continue
        items.append(token)
    return items


def _parse_dataset_list(value):
    if not value:
        return []
    items = []
    for raw in str(value).split(","):
        token = normalize_dataset_name(raw)
        if not token:
            continue
        items.append(token)
    return items


def _parse_custom_interaction_specs(value):
    specs = []
    if not value:
        return specs
    for raw in str(value).split(","):
        token = raw.strip()
        if not token:
            continue
        scale = 1.0
        if "@" in token:
            token, scale_raw = token.rsplit("@", 1)
            token = token.strip()
            scale_raw = scale_raw.strip()
            if scale_raw:
                try:
                    scale = float(scale_raw)
                except ValueError:
                    print(
                        "Warning: invalid custom interaction scale "
                        f"'{scale_raw}' in '{raw}'. Using 1.0."
                    )
        if "*" in token:
            left, right = token.split("*", 1)
        elif ":" in token:
            left, right = token.split(":", 1)
        else:
            print(
                "Warning: invalid custom interaction spec "
                f"'{raw}'. Use col1*col2 (or col1*col2@scale)."
            )
            continue
        left = left.strip()
        right = right.strip()
        if not left or not right:
            print(
                "Warning: invalid custom interaction spec "
                f"'{raw}'. Empty column name."
            )
            continue
        specs.append((left, right, scale))
    return specs


def add_custom_interaction_features(df, raw_specs):
    specs = _parse_custom_interaction_specs(raw_specs)
    if df.empty or not specs:
        return df, [], []

    out = df.copy()
    created = []
    skipped = []
    for left, right, scale in specs:
        if left not in out.columns or right not in out.columns:
            skipped.append((left, right))
            continue
        col = f"{left}_x_{right}"
        left_vals = pd.to_numeric(out[left], errors="coerce")
        right_vals = pd.to_numeric(out[right], errors="coerce")
        product = left_vals * right_vals
        if scale != 1.0:
            product = product * float(scale)
        out[col] = product
        created.append(col)
    return out, created, skipped


def fit_context_target_residualizer(
    train_df,
    target,
    context_cols,
    transform="residual",
    std_eps=1e-9,
):
    if train_df is None or train_df.empty:
        return None
    cols = [c for c in (context_cols or []) if c in train_df.columns]
    if not cols:
        return None
    mode = str(transform or "residual").strip().lower()
    if mode not in {"residual", "zscore"}:
        mode = "residual"
    mean_col = "__target_context_mean__"
    means = (
        train_df.groupby(cols, dropna=False)[target]
        .mean()
        .reset_index()
        .rename(columns={target: mean_col})
    )
    std_col = "__target_context_std__"
    stds = None
    global_std = float(np.nanstd(train_df[target].to_numpy(dtype=float), ddof=0))
    if not np.isfinite(global_std) or global_std <= float(std_eps):
        global_std = 1.0
    if mode == "zscore":
        stds = (
            train_df.groupby(cols, dropna=False)[target]
            .std(ddof=0)
            .reset_index()
            .rename(columns={target: std_col})
        )
    global_mean = float(np.nanmean(train_df[target].to_numpy(dtype=float)))
    return {
        "context_cols": cols,
        "transform": mode,
        "std_eps": float(std_eps),
        "mean_col": mean_col,
        "std_col": std_col,
        "means": means,
        "stds": stds,
        "global_mean": global_mean,
        "global_std": global_std,
    }


def apply_context_target_residualizer(df, target, residualizer):
    if df is None:
        return df, None, None
    if residualizer is None:
        out = df.copy()
        return out, np.zeros(len(out), dtype=float), np.ones(len(out), dtype=float)

    cols = [c for c in residualizer.get("context_cols", []) if c in df.columns]
    if not cols:
        out = df.copy()
        return out, np.zeros(len(out), dtype=float), np.ones(len(out), dtype=float)

    means = residualizer.get("means")
    mean_col = str(residualizer.get("mean_col") or "__target_context_mean__")
    stds = residualizer.get("stds")
    std_col = str(residualizer.get("std_col") or "__target_context_std__")
    global_mean = float(residualizer.get("global_mean", np.nan))
    global_std = float(residualizer.get("global_std", 1.0))
    transform = str(residualizer.get("transform") or "residual").strip().lower()
    std_eps = float(residualizer.get("std_eps", 1e-9))
    if not np.isfinite(global_std) or global_std <= std_eps:
        global_std = 1.0

    if means is None or means.empty:
        out = df.copy()
        offsets = np.full(len(out), global_mean, dtype=float)
        scales = np.full(len(out), global_std if transform == "zscore" else 1.0, dtype=float)
        if transform == "zscore":
            out[target] = (out[target].to_numpy(dtype=float) - offsets) / scales
        else:
            out[target] = out[target].to_numpy(dtype=float) - offsets
        return out, offsets, scales

    out = df.copy()
    out["__target_resid_row__"] = np.arange(len(out))
    out = out.merge(means[cols + [mean_col]], on=cols, how="left")
    if transform == "zscore" and stds is not None and not stds.empty:
        out = out.merge(stds[cols + [std_col]], on=cols, how="left")
    out = out.sort_values("__target_resid_row__", kind="mergesort")
    offsets = out[mean_col].fillna(global_mean).to_numpy(dtype=float)
    scales = np.ones(len(out), dtype=float)
    drop_cols = ["__target_resid_row__", mean_col]
    if transform == "zscore":
        if std_col in out.columns:
            scales = out[std_col].fillna(global_std).to_numpy(dtype=float)
        else:
            scales = np.full(len(out), global_std, dtype=float)
        scales = np.where((~np.isfinite(scales)) | (scales <= std_eps), global_std, scales)
        scales = np.where((~np.isfinite(scales)) | (scales <= std_eps), 1.0, scales)
        out[target] = (out[target].to_numpy(dtype=float) - offsets) / scales
        if std_col in out.columns:
            drop_cols.append(std_col)
    else:
        out[target] = out[target].to_numpy(dtype=float) - offsets
    out = out.drop(columns=drop_cols)
    return out, offsets, scales


def residualize_target_by_context(
    train_df,
    test_df,
    target,
    context_cols,
    transform="residual",
    std_eps=1e-9,
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    residualizer = fit_context_target_residualizer(
        train_df,
        target,
        context_cols,
        transform=transform,
        std_eps=std_eps,
    )
    if residualizer is None:
        return train_df, test_df, None, None, None, None, None
    train_out, train_offsets, train_scales = apply_context_target_residualizer(
        train_df, target, residualizer
    )
    test_out, test_offsets, test_scales = apply_context_target_residualizer(
        test_df, target, residualizer
    )
    return (
        train_out,
        test_out,
        train_offsets,
        test_offsets,
        train_scales,
        test_scales,
        residualizer,
    )


def sample_fewshot_calibration_mask(df, context_cols=None, k=0, rng=None, allow_backoff=True):
    n_rows = int(len(df)) if df is not None else 0
    mask = np.zeros(n_rows, dtype=bool)
    if n_rows <= 1:
        return mask
    k_int = int(k)
    if k_int <= 0:
        return mask
    rng = rng or np.random.RandomState(0)
    cols = [c for c in (context_cols or []) if c in df.columns]
    if cols and bool(allow_backoff):
        for cand_cols in [cols[:kk] for kk in range(len(cols), 0, -1)]:
            counts = (
                df[cand_cols]
                .assign(__row__=1)
                .groupby(cand_cols, dropna=False)["__row__"]
                .sum()
            )
            if not counts.empty and int(counts.max()) > 1:
                cols = cand_cols
                break
    if not cols:
        k_eff = min(k_int, n_rows - 1)
        if k_eff <= 0:
            return mask
        chosen = rng.choice(np.arange(n_rows), size=k_eff, replace=False)
        mask[chosen] = True
        return mask

    work = df[cols].copy()
    work["__row__"] = np.arange(n_rows)
    for _, grp in work.groupby(cols, dropna=False, sort=False):
        rows = grp["__row__"].to_numpy(dtype=int)
        if rows.size <= 1:
            continue
        k_eff = min(k_int, rows.size - 1)
        if k_eff <= 0:
            continue
        chosen = rng.choice(rows, size=k_eff, replace=False)
        mask[chosen] = True
    return mask


def fit_context_prediction_calibrator(
    df,
    y_true,
    y_pred,
    context_cols=None,
    std_eps=1e-9,
    min_group_size=2,
    allow_backoff=True,
):
    if df is None or len(df) == 0:
        return None
    cols = [c for c in (context_cols or []) if c in df.columns]
    work = df[cols].copy() if cols else pd.DataFrame(index=df.index)
    work["__pred__"] = np.asarray(y_pred, dtype=float)
    work["__true__"] = np.asarray(y_true, dtype=float)
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["__pred__", "__true__"])
    if work.empty:
        return None

    global_pred_mean = float(np.nanmean(work["__pred__"].to_numpy(dtype=float)))
    global_true_mean = float(np.nanmean(work["__true__"].to_numpy(dtype=float)))
    global_pred_std = float(np.nanstd(work["__pred__"].to_numpy(dtype=float), ddof=0))
    global_true_std = float(np.nanstd(work["__true__"].to_numpy(dtype=float), ddof=0))
    if not np.isfinite(global_pred_std) or global_pred_std <= float(std_eps):
        global_pred_std = 1.0
    if not np.isfinite(global_true_std) or global_true_std <= float(std_eps):
        global_true_std = 1.0

    min_n = max(int(min_group_size), 1)
    levels = []
    if cols:
        candidate_cols = [cols]
        if bool(allow_backoff):
            candidate_cols = [cols[:k] for k in range(len(cols), 0, -1)]
        for level_cols in candidate_cols:
            stats = (
                work.groupby(level_cols, dropna=False)
                .agg(
                    pred_mean=("__pred__", "mean"),
                    pred_std=("__pred__", lambda s: float(np.nanstd(s.to_numpy(dtype=float), ddof=0))),
                    true_mean=("__true__", "mean"),
                    true_std=("__true__", lambda s: float(np.nanstd(s.to_numpy(dtype=float), ddof=0))),
                    group_n=("__pred__", "size"),
                )
                .reset_index()
            )
            stats = stats[stats["group_n"] >= min_n].copy()
            if stats.empty:
                continue
            levels.append(
                {
                    "context_cols": list(level_cols),
                    "stats": stats[level_cols + ["pred_mean", "pred_std", "true_mean", "true_std"]],
                }
            )
            if not bool(allow_backoff):
                break
    stats = levels[0]["stats"] if levels else None
    effective_context_cols = levels[0]["context_cols"] if levels else []
    return {
        "context_cols": cols,
        "effective_context_cols": effective_context_cols,
        "levels": levels,
        "stats": stats,
        "global_pred_mean": global_pred_mean,
        "global_true_mean": global_true_mean,
        "global_pred_std": global_pred_std,
        "global_true_std": global_true_std,
        "std_eps": float(std_eps),
        "min_group_size": int(min_n),
        "allow_backoff": bool(allow_backoff),
    }


def apply_context_prediction_calibrator(df, y_pred, calibrator):
    pred = np.asarray(y_pred, dtype=float)
    if calibrator is None or df is None or len(pred) == 0:
        return pred

    cols = [c for c in (calibrator.get("context_cols") or []) if c in df.columns]
    std_eps = float(calibrator.get("std_eps", 1e-9))
    gpm = float(calibrator.get("global_pred_mean", 0.0))
    gtm = float(calibrator.get("global_true_mean", 0.0))
    gps = float(calibrator.get("global_pred_std", 1.0))
    gts = float(calibrator.get("global_true_std", 1.0))
    if not np.isfinite(gps) or gps <= std_eps:
        gps = 1.0
    if not np.isfinite(gts) or gts <= std_eps:
        gts = 1.0

    levels = calibrator.get("levels") or []
    n_rows = len(pred)
    pred_mean = np.full(n_rows, gpm, dtype=float)
    true_mean = np.full(n_rows, gtm, dtype=float)
    pred_std = np.full(n_rows, gps, dtype=float)
    true_std = np.full(n_rows, gts, dtype=float)
    assigned = np.zeros(n_rows, dtype=bool)

    # Backward compatibility for calibrators saved before hierarchical levels existed.
    if not levels:
        stats = calibrator.get("stats")
        if cols and stats is not None and len(stats) > 0:
            levels = [{"context_cols": cols, "stats": stats}]

    for level in levels:
        level_cols = [c for c in (level.get("context_cols") or []) if c in df.columns]
        stats = level.get("stats")
        if not level_cols or stats is None or len(stats) == 0:
            continue
        work = df[level_cols].copy()
        work["__pred_row__"] = np.arange(len(work))
        work = work.merge(stats, on=level_cols, how="left")
        work = work.sort_values("__pred_row__", kind="mergesort")
        cand_pred_mean = work["pred_mean"].to_numpy(dtype=float)
        cand_true_mean = work["true_mean"].to_numpy(dtype=float)
        cand_pred_std = work["pred_std"].to_numpy(dtype=float)
        cand_true_std = work["true_std"].to_numpy(dtype=float)
        valid = np.isfinite(cand_pred_mean) & np.isfinite(cand_true_mean)
        take = (~assigned) & valid
        if not np.any(take):
            continue
        pred_mean[take] = cand_pred_mean[take]
        true_mean[take] = cand_true_mean[take]
        pred_std[take] = cand_pred_std[take]
        true_std[take] = cand_true_std[take]
        assigned[take] = True

    pred_std = np.where((~np.isfinite(pred_std)) | (pred_std <= std_eps), gps, pred_std)
    true_std = np.where((~np.isfinite(true_std)) | (true_std <= std_eps), gts, true_std)
    return ((pred - pred_mean) / pred_std) * true_std + true_mean


def demean_target_by_multiple_groups(train_df, test_df, target, group_cols):
    """
    Sequentially demean target by multiple grouping columns.
    Order matters: typically do encoder first, then benchmark.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    for group_col in group_cols:
        if group_col not in train_df.columns or group_col not in test_df.columns:
            continue
        
        # Compute group means from training data only
        group_means = train_df.groupby(group_col)[target].mean()
        global_mean = train_df[target].mean()
        
        # Apply to both train and test
        train_offsets = train_df[group_col].map(group_means).fillna(global_mean)
        test_offsets = test_df[group_col].map(group_means).fillna(global_mean)
        
        train_df[target] = train_df[target] - train_offsets
        test_df[target] = test_df[target] - test_offsets
    
    return train_df, test_df


def _select_target_demean_groups(args):
    """Select which groups to demean target by (returns list)."""
    demean_groups = []
    
    # Order matters: encoder first (main effect), then benchmark (difficulty)
    # Use model_family_encoder to properly handle models like RAFT that don't have encoder variants
    if args.cv_demean_target_by_encoder:
        demean_groups.append("model_family_encoder")
    
    if args.cv_demean_target_by_benchmark:
        demean_groups.append("benchmark")
    
    return demean_groups


def _select_target_demean_group(args):
    """Legacy function for backward compatibility - returns first group or None."""
    groups = _select_target_demean_groups(args)
    return groups[0] if groups else None


def filter_encoder_configs(df, exclude):
    if not exclude:
        return df, 0
    df = ensure_encoder_config(df)
    if "encoder_config" not in df.columns:
        return df, 0
    exclude_set = {item.upper() for item in exclude}
    if not exclude_set:
        return df, 0
    before = len(df)
    filtered = df[~df["encoder_config"].astype(str).str.upper().isin(exclude_set)].copy()
    return filtered, before - len(filtered)


def filter_model_families(df, exclude):
    if not exclude:
        return df, 0
    df = ensure_model_family(df)
    if "model_family" not in df.columns:
        return df, 0
    exclude_set = {item.lower() for item in exclude}
    if not exclude_set:
        return df, 0
    before = len(df)
    filtered = df[~df["model_family"].astype(str).str.lower().isin(exclude_set)].copy()
    return filtered, before - len(filtered)


def _collect_encoder_configs(series):
    configs = [c for c in ENCODER_CONFIG_ORDER if c in set(series.dropna().unique())]
    return configs


def _model_family_token(value):
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    token = token.strip("_")
    return token or "unknown"


def _collect_model_families(series):
    unique = list(pd.Series(series).dropna().unique())
    ordered = [m for m in MODEL_FAMILY_ORDER if m in unique]
    remainder = [m for m in unique if m not in ordered]
    remainder = sorted(remainder)
    return ordered + remainder


def add_model_family_effects(df, predictors, families, baseline, add_interactions):
    if not families or len(families) < 2:
        return df, [], []
    df = df.copy()
    baseline = baseline or families[-1]
    if baseline not in families:
        baseline = families[-1]

    dummy_cols = []
    family_tokens = {fam: _model_family_token(fam) for fam in families}
    for fam in families:
        if fam == baseline:
            continue
        token = family_tokens[fam]
        col = f"mf_{token}"
        df[col] = (df["model_family"] == fam).astype(float)
        dummy_cols.append(col)

    interaction_cols = []
    if add_interactions:
        for pred in predictors:
            if pred not in df.columns:
                continue
            for fam in families:
                if fam == baseline:
                    continue
                token = family_tokens[fam]
                inter_col = f"{pred}__mf_{token}"
                df[inter_col] = df[pred] * df[f"mf_{token}"]
                interaction_cols.append(inter_col)
    return df, dummy_cols, interaction_cols


def add_spair_indicator_interactions(df, predictors, benchmark_col="benchmark"):
    if df.empty or benchmark_col not in df.columns or not predictors:
        return df, predictors, [], None
    df = df.copy()
    normalized = df[benchmark_col].astype(str).map(normalize_dataset_name)
    indicator_col = "spair_indicator"
    df[indicator_col] = (normalized == "spair").astype(float)
    interaction_cols = []
    for pred in predictors:
        if pred not in df.columns:
            continue
        inter_col = f"{indicator_col}_x_{pred}"
        df[inter_col] = df[indicator_col] * df[pred]
        interaction_cols.append(inter_col)
    updated = list(predictors)
    if indicator_col not in updated:
        updated.append(indicator_col)
    updated.extend(interaction_cols)
    return df, updated, interaction_cols, indicator_col


def prepare_model_family_pooled_frames(
    report_df,
    cv_df,
    predictors,
    base_predictors,
    baseline,
    add_interactions,
    include_main_effects,
):
    if not add_interactions and not include_main_effects:
        return report_df, cv_df, predictors

    report_df = ensure_model_family(report_df)
    cv_df = ensure_model_family(cv_df)
    if "model_family" not in report_df.columns:
        return report_df, cv_df, predictors

    families = _collect_model_families(report_df["model_family"])
    if len(families) < 2:
        print("Warning: model_family effects requested but insufficient families.")
        return report_df, cv_df, predictors

    report_df, dummy_cols, interaction_cols = add_model_family_effects(
        report_df, base_predictors, families, baseline, add_interactions
    )
    cv_df, _, _ = add_model_family_effects(
        cv_df, base_predictors, families, baseline, add_interactions
    )
    if include_main_effects:
        predictors = predictors + dummy_cols
    if add_interactions:
        predictors = predictors + interaction_cols
    return report_df, cv_df, predictors


def add_encoder_interactions(df, predictors, configs, baseline):
    if not configs or len(configs) < 2:
        return df, [], []
    df = df.copy()
    if baseline not in configs:
        baseline = configs[-1]

    dummy_cols = []
    for cfg in configs:
        if cfg == baseline:
            continue
        col = f"enc_{cfg}"
        df[col] = (df["encoder_config"] == cfg).astype(float)
        dummy_cols.append(col)

    interaction_cols = []
    for pred in predictors:
        if pred not in df.columns:
            continue
        for cfg in configs:
            if cfg == baseline:
                continue
            inter_col = f"{pred}__enc_{cfg}"
            df[inter_col] = df[pred] * df[f"enc_{cfg}"]
            interaction_cols.append(inter_col)
    return df, dummy_cols, interaction_cols


def prepare_encoder_pooled_frames(
    report_df,
    cv_df,
    predictors,
    baseline,
    add_interactions,
    include_main_effects,
    encoder_norm_mode,
):
    if not add_interactions and not include_main_effects and encoder_norm_mode == "none":
        return report_df, cv_df, predictors

    report_df = ensure_encoder_config(report_df)
    cv_df = ensure_encoder_config(cv_df)
    if "encoder_config" not in report_df.columns:
        return report_df, cv_df, predictors

    configs = _collect_encoder_configs(report_df["encoder_config"])
    if len(configs) < 2:
        print("Warning: encoder effects requested but insufficient encoder configs.")
        return report_df, cv_df, predictors

    baseline = (baseline or "").strip().upper()
    if baseline and baseline not in configs:
        baseline = configs[-1]
    dummy_cols = []
    if include_main_effects or add_interactions:
        for cfg in configs:
            if cfg == baseline:
                continue
            col = f"enc_{cfg}"
            report_df[col] = (report_df["encoder_config"] == cfg).astype(float)
            cv_df[col] = (cv_df["encoder_config"] == cfg).astype(float)
            dummy_cols.append(col)

    interaction_cols = []
    if add_interactions:
        for pred in predictors:
            if pred not in report_df.columns:
                continue
            for cfg in configs:
                if cfg == baseline:
                    continue
                inter_col = f"{pred}__enc_{cfg}"
                report_df[inter_col] = report_df[pred] * report_df[f"enc_{cfg}"]
                cv_df[inter_col] = cv_df[pred] * cv_df[f"enc_{cfg}"]
                interaction_cols.append(inter_col)
    if include_main_effects:
        predictors = predictors + dummy_cols
    predictors = predictors + interaction_cols
    return report_df, cv_df, predictors


def demean_target_by_group(train_df, test_df, target, group_col):
    train_df = train_df.copy()
    test_df = test_df.copy()
    if group_col not in train_df.columns or group_col not in test_df.columns:
        return train_df, test_df, None
    group_means = train_df.groupby(group_col)[target].mean()
    global_mean = train_df[target].mean()
    train_offsets = train_df[group_col].map(group_means).fillna(global_mean)
    test_offsets = test_df[group_col].map(group_means).fillna(global_mean)
    train_df[target] = train_df[target] - train_offsets
    test_df[target] = test_df[target] - test_offsets
    return train_df, test_df, test_offsets.to_numpy(dtype=float)


def add_relative_target(df, baseline_dataset, target_col):
    baseline_name = normalize_dataset_name(baseline_dataset)
    group_cols = ["benchmark"]
    for col in ("pretrained", "freeze"):
        if col in df.columns:
            group_cols.append(col)

    baseline_df = df[df["train_dataset"] == baseline_name]
    if baseline_df.empty:
        return df, len(df), baseline_name

    baseline = (
        baseline_df.groupby(group_cols)[target_col]
        .mean()
        .reset_index()
        .rename(columns={target_col: "baseline_value"})
    )
    merged = df.merge(baseline, on=group_cols, how="left")
    merged["auc_delta"] = merged[target_col] - merged["baseline_value"]
    merged["baseline_train_dataset"] = baseline_name
    missing = int(merged["baseline_value"].isna().sum())
    return merged, missing, baseline_name


def _select_random_slopes(args, predictors):
    if not args.mixedlm_random_slopes:
        return []
    requested = [p.strip() for p in args.mixedlm_random_slopes.split(",") if p.strip()]
    return [p for p in requested if p in predictors]


def _drop_algebraic_redundancies(predictors):
    redundant = []
    for prefix in ("flow", "resnet", "dino", "hof"):
        cov = f"{prefix}_eval_to_train_coverage"
        cov_explicit = f"{prefix}_eval_to_train_over_train_precision"
        outside = f"{prefix}_outside_mass"
        cov_logit = f"{cov}_logit"
        cov_logit_explicit = f"{cov_explicit}_logit"
        outside_logit = f"{outside}_logit"
        if (cov in predictors or cov_explicit in predictors) and outside in predictors:
            redundant.append(outside)
        if (cov_logit in predictors or cov_logit_explicit in predictors) and outside_logit in predictors:
            redundant.append(outside_logit)
    if not redundant:
        return predictors, []
    filtered = [p for p in predictors if p not in redundant]
    return filtered, redundant


def _dedupe_predictors(predictors):
    seen = set()
    ordered = []
    for pred in predictors:
        if pred in seen:
            continue
        ordered.append(pred)
        seen.add(pred)
    return ordered


def _extend_predictors_with_kl(predictors, feature_df):
    if not predictors or feature_df is None or feature_df.empty:
        return predictors
    prefixes = set()
    for pred in predictors:
        if pred.startswith("flow_"):
            prefixes.add("flow")
        if pred.startswith("resnet_"):
            prefixes.add("resnet")
        if pred.startswith("dino_"):
            prefixes.add("dino")
        if pred.startswith("hof_"):
            prefixes.add("hof")
    if not prefixes:
        return predictors
    extended = list(predictors)
    for prefix in sorted(prefixes):
        for suffix in (
            "train_to_eval_kl_div_hist",
            "eval_to_train_kl_div_hist",
            "train_to_eval_kl_div_hist_log1p_linear",
            "eval_to_train_kl_div_hist_log1p_linear",
            "train_to_eval_kl_div",
            "eval_to_train_kl_div",
        ):
            col = f"{prefix}_{suffix}"
            if col in feature_df.columns and col not in extended:
                extended.append(col)
    return extended


def _build_baseline_selectors(feature_df, predictors=None, use_logit=True):
    selectors = []

    def add_metric(name, col, direction=1):
        if col in feature_df.columns:
            selectors.append({
                "name": name,
                "type": "metric",
                "column": col,
                "direction": direction,
            })

    def infer_direction(name):
        lower = name.lower()
        if "coverage" in lower or "precision" in lower or "recall" in lower:
            return 1
        if "kl_div" in lower or "divergence" in lower:
            return -1
        if "mmd" in lower:
            return -1
        if "outside" in lower:
            return -1
        if "dist" in lower:
            return -1
        if "radius" in lower:
            return -1
        return 1

    if predictors:
        seen = set()
        for pred in predictors:
            if not pred or pred in seen:
                continue
            seen.add(pred)
            add_metric(pred, pred, direction=infer_direction(pred))
    else:
        distance_cols = [
            "flow_train_to_eval_mean_dist",
            "flow_eval_to_train_mean_dist",
            "resnet_train_to_eval_mean_dist",
            "resnet_eval_to_train_mean_dist",
            "dino_train_to_eval_mean_dist",
            "dino_eval_to_train_mean_dist",
            "hof_train_to_eval_mean_dist",
            "hof_eval_to_train_mean_dist",
        ]
        has_distance = any(col in feature_df.columns for col in distance_cols)
        if has_distance:
            add_metric(
                "flow_train_to_eval_mean_dist",
                "flow_train_to_eval_mean_dist",
                direction=-1,
            )
            add_metric(
                "flow_eval_to_train_mean_dist",
                "flow_eval_to_train_mean_dist",
                direction=-1,
            )
            add_metric(
                "resnet_train_to_eval_mean_dist",
                "resnet_train_to_eval_mean_dist",
                direction=-1,
            )
            add_metric(
                "resnet_eval_to_train_mean_dist",
                "resnet_eval_to_train_mean_dist",
                direction=-1,
            )
            add_metric(
                "dino_train_to_eval_mean_dist",
                "dino_train_to_eval_mean_dist",
                direction=-1,
            )
            add_metric(
                "dino_eval_to_train_mean_dist",
                "dino_eval_to_train_mean_dist",
                direction=-1,
            )
        elif use_logit:
            flow_cov_logit = (
                "flow_train_to_eval_over_eval_recall_logit"
                if "flow_train_to_eval_over_eval_recall_logit" in feature_df.columns
                else "flow_train_to_eval_coverage_logit"
            )
            resnet_cov_logit = (
                "resnet_train_to_eval_over_eval_recall_logit"
                if "resnet_train_to_eval_over_eval_recall_logit" in feature_df.columns
                else "resnet_train_to_eval_coverage_logit"
            )
            dino_cov_logit = (
                "dino_train_to_eval_over_eval_recall_logit"
                if "dino_train_to_eval_over_eval_recall_logit" in feature_df.columns
                else "dino_train_to_eval_coverage_logit"
            )
            hof_cov_logit = (
                "hof_train_to_eval_over_eval_recall_logit"
                if "hof_train_to_eval_over_eval_recall_logit" in feature_df.columns
                else "hof_train_to_eval_coverage_logit"
            )
            add_metric(flow_cov_logit, flow_cov_logit, direction=1)
            add_metric(resnet_cov_logit, resnet_cov_logit, direction=1)
            add_metric(dino_cov_logit, dino_cov_logit, direction=1)
            add_metric(hof_cov_logit, hof_cov_logit, direction=1)
        else:
            flow_cov = (
                "flow_train_to_eval_over_eval_recall"
                if "flow_train_to_eval_over_eval_recall" in feature_df.columns
                else "flow_train_to_eval_coverage"
            )
            resnet_cov = (
                "resnet_train_to_eval_over_eval_recall"
                if "resnet_train_to_eval_over_eval_recall" in feature_df.columns
                else "resnet_train_to_eval_coverage"
            )
            dino_cov = (
                "dino_train_to_eval_over_eval_recall"
                if "dino_train_to_eval_over_eval_recall" in feature_df.columns
                else "dino_train_to_eval_coverage"
            )
            hof_cov = (
                "hof_train_to_eval_over_eval_recall"
                if "hof_train_to_eval_over_eval_recall" in feature_df.columns
                else "hof_train_to_eval_coverage"
            )
            add_metric(flow_cov, flow_cov, direction=1)
            add_metric(resnet_cov, resnet_cov, direction=1)
            add_metric(dino_cov, dino_cov, direction=1)
            add_metric(hof_cov, hof_cov, direction=1)

        add_metric("flow_mmd", "flow_mmd", direction=-1)
        add_metric("feature_mmd", "feature_mmd", direction=-1)
        add_metric("dino_mmd", "dino_mmd", direction=-1)

    selectors.append({
        "name": "always_flyingthings",
        "type": "constant",
        "option": "flyingthings",
    })
    selectors.append({
        "name": "always_best_avg",
        "type": "best_avg",
    })
    return selectors


def run_analysis_bundle(feature_df, out_dir, predictors, args, cv_df=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    if feature_df.empty:
        return

    if args.target not in feature_df.columns:
        print(f"Target '{args.target}' not found in {out_dir / 'auc_with_features.csv'}.")
        return

    # Run stability selection if requested
    if args.run_stability_selection:
        stability_path = out_dir / "stability_selection_results.csv"
        print(f"Running stability selection with {args.stability_n_bootstrap} bootstrap iterations...")
        run_stability_selection(
            feature_df,
            predictors,
            args.target,
            n_bootstrap=args.stability_n_bootstrap,
            threshold=args.stability_threshold,
            output_path=stability_path,
        )
        print(f"Stability selection results saved to {stability_path}")

    # Run family comparison if requested
    if args.run_family_comparison:
        # Define predictor families
        flow_predictors = [p for p in predictors if p.startswith("flow_")]
        dino_predictors = [p for p in predictors if p.startswith("dino_")]
        hof_predictors = [p for p in predictors if p.startswith("hof_")]
        predictor_families = {}
        if flow_predictors:
            predictor_families['flow'] = flow_predictors
        if dino_predictors:
            predictor_families['dino'] = dino_predictors
        if hof_predictors:
            predictor_families['hof'] = hof_predictors
        
        if len(predictor_families) >= 2:
            family_path = out_dir / "predictor_family_comparison.csv"
            print(f"Running predictor family comparison...")
            compute_predictor_family_comparison(
                feature_df,
                predictor_families,
                args.target,
                family_path,
                standardize=args.standardize,
                ridge_alpha=args.ridge_alpha,
                cv_standardize_mode=args.cv_standardize_mode,
            )
            print(f"Family comparison results saved to {family_path}")
    
    # Run univariate comparison if requested
    if args.run_univariate_comparison:
        univariate_path = out_dir / "univariate_predictor_comparison.csv"
        print(f"Running univariate predictor comparison ({len(predictors)} predictors)...")
        compute_univariate_predictor_comparison(
            feature_df,
            predictors,
            args.target,
            univariate_path,
            standardize=args.standardize,
            ridge_alpha=args.ridge_alpha,
            cv_standardize_mode=args.cv_standardize_mode,
        )
        print(f"Univariate comparison results saved to {univariate_path}")
        loto_uni_df = feature_df.copy()
        if args.loto_collapse_mixed:
            loto_uni_df["train_dataset_group"] = loto_uni_df["train_dataset"].apply(collapse_mixed_dataset)
            loto_group_col = "train_dataset_group"
        else:
            loto_group_col = "train_dataset"
        loto_univariate_path = out_dir / "univariate_predictor_comparison_loto.csv"
        print(f"Running LOTO univariate predictor comparison ({len(predictors)} predictors)...")
        compute_univariate_predictor_comparison_by_group(
            loto_uni_df,
            predictors,
            args.target,
            loto_group_col,
            loto_univariate_path,
            standardize=args.standardize,
            ridge_alpha=args.ridge_alpha,
            cv_standardize_mode=args.cv_standardize_mode,
        )
        print(f"LOTO univariate comparison results saved to {loto_univariate_path}")

    within_path = out_dir / "within_benchmark_slopes.csv"
    compute_within_benchmark_slopes(feature_df, predictors, args.target, within_path)
    within_uni_path = out_dir / "within_benchmark_slopes_univariate.csv"
    compute_within_benchmark_univariate_slopes(feature_df, predictors, args.target, within_uni_path)
    within_train_uni_path = out_dir / "within_train_dataset_slopes_univariate.csv"
    train_uni_df = feature_df.copy()
    train_group_col = "train_dataset"
    if args.loto_collapse_mixed:
        train_uni_df["train_dataset_group"] = train_uni_df["train_dataset"].apply(collapse_mixed_dataset)
        train_group_col = "train_dataset_group"
    compute_within_train_dataset_univariate_slopes(
        train_uni_df,
        predictors,
        args.target,
        train_group_col,
        within_train_uni_path,
    )

    if not args.skip_regression:
        regression_path = out_dir / "regression_summary.txt"
        run_regression(
            feature_df,
            predictors,
            args.target,
            regression_path,
            linear_model=args.linear_model,
            ridge_alpha=args.ridge_alpha,
            use_mixedlm=args.regression_mixedlm,
        )

    if args.skip_prediction:
        return

    pred_target = args.prediction_target or args.target
    pred_model = args.prediction_model or args.linear_model
    cv_df = cv_df if cv_df is not None else feature_df
    repeat_agg_mode = str(args.cv_repeat_aggregation or "none").strip().lower()
    if repeat_agg_mode != "none":
        repeat_group_cols_req = ["train_dataset", "benchmark"]
        if args.ranking_group:
            repeat_group_cols_req.append(str(args.ranking_group))
        repeat_group_cols_req.extend(_parse_csv_group_cols(args.ranking_context_cols))
        repeat_group_cols_req.extend(_parse_csv_group_cols(args.pairwise_group_cols))
        if args.cv_residualize_target_by_context:
            repeat_group_cols_req.extend(_parse_csv_group_cols(args.cv_residual_context_cols))
        if "model_family_encoder" in cv_df.columns:
            repeat_group_cols_req.append("model_family_encoder")
        repeat_group_cols_req = list(dict.fromkeys([c for c in repeat_group_cols_req if c]))
        cv_df, repeat_group_cols, repeat_missing_cols = _ensure_context_columns(
            cv_df, repeat_group_cols_req
        )
        if repeat_missing_cols:
            print(
                "Warning: missing repeat-aggregation columns (ignored): "
                + ", ".join(repeat_missing_cols)
            )
        repeat_group_cols = [c for c in repeat_group_cols if c in cv_df.columns]
        if len(repeat_group_cols) < 2:
            print(
                "Warning: insufficient columns for --cv-repeat-aggregation; "
                "skipping repeat aggregation."
            )
        else:
            before_rows = len(cv_df)
            cv_df = collapse_cv_rows_to_cells(
                cv_df, repeat_group_cols, numeric_agg=repeat_agg_mode
            )
            after_rows = len(cv_df)
            if before_rows != after_rows:
                print(
                    "Aggregated repeated CV rows by "
                    f"{repeat_group_cols} ({repeat_agg_mode}): {before_rows} -> {after_rows}"
                )
    if args.collapse_cv_cells:
        collapse_cols = ["train_dataset", "benchmark"]
        available = [c for c in collapse_cols if c in cv_df.columns]
        if len(available) == len(collapse_cols):
            before_rows = len(cv_df)
            cv_df = collapse_cv_rows_to_cells(cv_df, collapse_cols)
            after_rows = len(cv_df)
            if before_rows != after_rows:
                print(
                    "Collapsed CV rows to cells by "
                    f"{collapse_cols}: {before_rows} -> {after_rows}"
                )
        else:
            print(
                "Warning: requested collapse_cv_cells but missing columns: "
                + ", ".join([c for c in collapse_cols if c not in cv_df.columns])
            )
    exclude_benchmarks = _parse_benchmark_list(args.exclude_benchmarks)
    if exclude_benchmarks and "benchmark" in cv_df.columns:
        cv_df = cv_df[
            ~cv_df["benchmark"].astype(str).str.lower().isin(exclude_benchmarks)
        ].copy()
        if cv_df.empty:
            print(
                "Warning: excluded all rows for prediction; skipping LOBO/LOTO/joint-OOD runs."
            )
            return
    target_demean_groups = _select_target_demean_groups(args)
    target_group_demean = len(target_demean_groups) > 0

    pairwise_group_cols_req = _parse_csv_group_cols(args.pairwise_group_cols)
    if not pairwise_group_cols_req:
        pairwise_group_cols_req = ["benchmark"]
    cv_df, pairwise_group_cols, pairwise_missing_cols = _ensure_context_columns(
        cv_df, pairwise_group_cols_req
    )
    if pairwise_missing_cols:
        print(
            "Warning: missing pairwise group columns (ignored): "
            + ", ".join(pairwise_missing_cols)
        )
    if not pairwise_group_cols:
        pairwise_group_cols = ["benchmark"]

    ranking_context_cols_req = _parse_csv_group_cols(args.ranking_context_cols)
    cv_df, ranking_context_cols, ranking_context_missing_cols = _ensure_context_columns(
        cv_df, ranking_context_cols_req
    )
    if ranking_context_missing_cols:
        print(
            "Warning: missing ranking context columns (ignored): "
            + ", ".join(ranking_context_missing_cols)
        )
    residual_context_cols = []
    residual_eval_space = str(args.cv_residual_eval_space or "residual").strip().lower()
    fewshot_calibration_cols = []
    if args.cv_fewshot_context_calibration:
        fewshot_cols_req = _parse_csv_group_cols(args.cv_fewshot_context_calibration_cols)
        if not fewshot_cols_req:
            fewshot_cols_req = ["benchmark", "model_family_encoder"]
        cv_df, fewshot_calibration_cols, fewshot_missing_cols = _ensure_context_columns(
            cv_df, fewshot_cols_req
        )
        if fewshot_missing_cols:
            print(
                "Warning: missing few-shot calibration context columns (ignored): "
                + ", ".join(fewshot_missing_cols)
            )
        if not fewshot_calibration_cols:
            print(
                "Warning: no valid few-shot calibration context columns found; disabling "
                "--cv-fewshot-context-calibration."
            )
            args.cv_fewshot_context_calibration = False
    if args.cv_residualize_target_by_context:
        residual_context_cols_req = _parse_csv_group_cols(args.cv_residual_context_cols)
        if not residual_context_cols_req:
            residual_context_cols_req = ["benchmark"] + list(ranking_context_cols or [])
        cv_df, residual_context_cols, residual_context_missing_cols = _ensure_context_columns(
            cv_df, residual_context_cols_req
        )
        if residual_context_missing_cols:
            print(
                "Warning: missing residual context columns (ignored): "
                + ", ".join(residual_context_missing_cols)
            )
        if not residual_context_cols:
            print(
                "Warning: no valid residual context columns found; disabling "
                "--cv-residualize-target-by-context."
            )
            args.cv_residualize_target_by_context = False

    lobo_group_norm_mode = args.cv_normalize_predictors_by_benchmark
    lobo_center_by_group = args.center_predictors_by_benchmark
    lobo_center_group_col = "benchmark"
    lobo_target_demean_groups = target_demean_groups
    lobo_target_group_demean = target_group_demean
    lobo_encoder_group_norm_mode = args.cv_normalize_predictors_by_encoder
    lobo_encoder_group_col = "encoder_config"
    if args.lobo_model_centered:
        if "model_family_encoder" not in cv_df.columns:
            cv_df = create_model_family_encoder_column(cv_df)
        lobo_group_norm_mode = "center"
        lobo_center_by_group = False
        lobo_center_group_col = "model_family_encoder"
        lobo_target_demean_groups = ["model_family_encoder"]
        lobo_target_group_demean = True

    loto_group_norm_mode = args.cv_normalize_predictors_by_benchmark
    loto_center_by_group = args.center_predictors_by_benchmark
    loto_center_group_col = "benchmark"
    loto_target_demean_groups = target_demean_groups
    loto_target_group_demean = target_group_demean
    loto_encoder_group_norm_mode = args.cv_normalize_predictors_by_encoder
    loto_encoder_group_col = "encoder_config"
    if args.loto_benchmark_centered:
        if "model_family_encoder" not in cv_df.columns:
            cv_df = create_model_family_encoder_column(cv_df)
        loto_group_norm_mode = "center"
        loto_center_by_group = False
        loto_center_group_col = "benchmark"
        loto_target_demean_groups = ["benchmark", "model_family_encoder"]
        loto_target_group_demean = True
    if args.cv_residualize_target_by_context:
        if lobo_target_group_demean or loto_target_group_demean:
            raise SystemExit(
                "Cannot combine --cv-residualize-target-by-context with target demeaning "
                "(--cv-demean-target-by-*, --lobo-model-centered, or --loto-benchmark-centered)."
            )
        if args.prediction_mixedlm and HAS_STATSMODELS:
            print(
                "Warning: MixedLM prediction outputs are skipped when "
                "--cv-residualize-target-by-context is enabled."
            )

    lobo_summary, lobo_preds = run_group_cv(
        cv_df,
        "benchmark",
        predictors,
        pred_target,
        standardize=args.standardize,
        standardize_mode=args.cv_standardize_mode,
        center_by_group=lobo_center_by_group,
        center_group_col=lobo_center_group_col,
        group_norm_mode=lobo_group_norm_mode,
        within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
        encoder_group_norm_mode=lobo_encoder_group_norm_mode,
        encoder_group_col=lobo_encoder_group_col,
        target_group_demean=lobo_target_group_demean,
        target_group_col=lobo_target_demean_groups,
        target_context_residualize_cols=(
            residual_context_cols if args.cv_residualize_target_by_context else None
        ),
        target_context_transform=args.cv_residual_target_transform,
        target_context_std_eps=args.cv_residual_target_std_eps,
        target_context_eval=residual_eval_space,
        fewshot_context_calibration=args.cv_fewshot_context_calibration,
        fewshot_context_calibration_cols=fewshot_calibration_cols,
        fewshot_context_calibration_std_eps=args.cv_fewshot_context_calibration_std_eps,
        fewshot_context_calibration_min_group_size=args.cv_fewshot_context_calibration_min_group_size,
        fewshot_context_calibration_backoff=args.cv_fewshot_context_calibration_backoff,
        fewshot_context_calibration_k=args.cv_fewshot_context_calibration_k,
        fewshot_context_calibration_seed=args.cv_fewshot_context_calibration_seed,
        min_predictor_std=args.cv_min_predictor_std,
        prediction_clip=args.prediction_clip,
        prediction_clip_min=args.prediction_clip_min,
        prediction_clip_max=args.prediction_clip_max,
        model=pred_model,
        ridge_alpha=args.ridge_alpha,
        pairwise_option_col=args.ranking_group,
        pairwise_group_cols=pairwise_group_cols,
        fit_sample_weighting=args.fit_sample_weighting,
        fit_balance_real_synth=args.fit_balance_real_synth,
        overall_aggregation=args.overall_aggregation,
    )
    if not lobo_summary.empty:
        lobo_summary.to_csv(out_dir / "prediction_lobo_summary.csv", index=False)
    if not lobo_preds.empty:
        lobo_preds.to_csv(out_dir / "prediction_lobo_rows.csv", index=False)
        compute_ranking_summary(
            lobo_preds,
            pred_target,
            args.ranking_group,
            out_dir / "prediction_lobo_rank_summary.csv",
            context_cols=ranking_context_cols,
            topk_frac=args.ranking_topk_frac,
            topk_min=args.ranking_topk_min,
        )
        write_rank_detail_rows(
            lobo_preds,
            pred_target,
            args.ranking_group,
            out_dir / "prediction_lobo_rank_detail.csv",
            context_cols=ranking_context_cols,
        )
        if args.sanity_direction_audit:
            write_direction_audit(
                lobo_preds,
                pred_target,
                args.ranking_group,
                out_dir / "prediction_lobo_direction_audit.txt",
            )

    baseline_selectors = _build_baseline_selectors(
        feature_df,
        predictors=predictors,
        use_logit=args.logit_coverage,
    )
    compute_baseline_rankings(
        cv_df,
        pred_target,
        args.ranking_group,
        out_dir / "prediction_lobo_rank_baselines.csv",
        baseline_selectors,
        context_cols=ranking_context_cols,
        topk_frac=args.ranking_topk_frac,
        topk_min=args.ranking_topk_min,
    )

    if (
        args.prediction_mixedlm
        and HAS_STATSMODELS
        and not args.cv_residualize_target_by_context
    ):
        random_slopes = _select_random_slopes(args, predictors)
        lobo_mixed_summary, lobo_mixed_preds = run_group_cv_mixedlm(
            cv_df,
            holdout_col="benchmark",
            predictors=predictors,
            target=pred_target,
            random_group_col="benchmark",
            random_slopes=random_slopes,
            standardize=args.standardize,
            standardize_mode=args.cv_standardize_mode,
            center_by_group=lobo_center_by_group,
            center_group_col=lobo_center_group_col,
            group_norm_mode=lobo_group_norm_mode,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=lobo_encoder_group_norm_mode,
            encoder_group_col=lobo_encoder_group_col,
            target_group_demean=lobo_target_group_demean,
            target_group_col=lobo_target_demean_groups,
            min_predictor_std=args.cv_min_predictor_std,
            prediction_clip=args.prediction_clip,
            prediction_clip_min=args.prediction_clip_min,
            prediction_clip_max=args.prediction_clip_max,
        )
        if not lobo_mixed_summary.empty:
            lobo_mixed_summary.to_csv(
                out_dir / "prediction_lobo_mixed_summary.csv", index=False
            )
        if not lobo_mixed_preds.empty:
            lobo_mixed_preds.to_csv(
                out_dir / "prediction_lobo_mixed_rows.csv", index=False
            )
            compute_ranking_summary(
                lobo_mixed_preds,
                args.target,
                args.ranking_group,
                out_dir / "prediction_lobo_mixed_rank_summary.csv",
                context_cols=ranking_context_cols,
                topk_frac=args.ranking_topk_frac,
                topk_min=args.ranking_topk_min,
            )

    if args.loto_collapse_mixed:
        loto_df = cv_df.copy()
        loto_df["train_dataset_group"] = loto_df["train_dataset"].apply(collapse_mixed_dataset)
        group_col = "train_dataset_group"
    else:
        loto_df = cv_df
        group_col = "train_dataset"

    placement_task_cols = ["benchmark"] + list(ranking_context_cols or [])
    loto_task_pred_rows = []
    loto_summary, loto_preds = run_group_cv(
        loto_df,
        group_col,
        predictors,
        pred_target,
        standardize=args.standardize,
        standardize_mode=args.cv_standardize_mode,
        center_by_group=loto_center_by_group,
        center_group_col=loto_center_group_col,
        group_norm_mode=loto_group_norm_mode,
        within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
        encoder_group_norm_mode=loto_encoder_group_norm_mode,
        encoder_group_col=loto_encoder_group_col,
        target_group_demean=loto_target_group_demean,
        target_group_col=loto_target_demean_groups,
        target_context_residualize_cols=(
            residual_context_cols if args.cv_residualize_target_by_context else None
        ),
        target_context_transform=args.cv_residual_target_transform,
        target_context_std_eps=args.cv_residual_target_std_eps,
        target_context_eval=residual_eval_space,
        fewshot_context_calibration=args.cv_fewshot_context_calibration,
        fewshot_context_calibration_cols=fewshot_calibration_cols,
        fewshot_context_calibration_std_eps=args.cv_fewshot_context_calibration_std_eps,
        fewshot_context_calibration_min_group_size=args.cv_fewshot_context_calibration_min_group_size,
        fewshot_context_calibration_backoff=args.cv_fewshot_context_calibration_backoff,
        fewshot_context_calibration_k=args.cv_fewshot_context_calibration_k,
        fewshot_context_calibration_seed=args.cv_fewshot_context_calibration_seed,
        min_predictor_std=args.cv_min_predictor_std,
        prediction_clip=args.prediction_clip,
        prediction_clip_min=args.prediction_clip_min,
        prediction_clip_max=args.prediction_clip_max,
        model=pred_model,
        ridge_alpha=args.ridge_alpha,
        pairwise_option_col=args.ranking_group,
        pairwise_group_cols=pairwise_group_cols,
        fit_sample_weighting=args.fit_sample_weighting,
        fit_balance_real_synth=args.fit_balance_real_synth,
        overall_aggregation=args.overall_aggregation,
        task_prediction_rows_out=loto_task_pred_rows,
        task_prediction_option_col=args.ranking_group,
        task_prediction_task_cols=placement_task_cols,
        task_prediction_holdout_group_col=group_col,
    )
    if not loto_summary.empty:
        loto_summary.to_csv(out_dir / "prediction_loto_summary.csv", index=False)
    if not loto_preds.empty:
        loto_preds.to_csv(out_dir / "prediction_loto_rows.csv", index=False)
    for stale in (
        out_dir / "prediction_loto_rank_summary.csv",
        out_dir / "prediction_loto_rank_detail.csv",
    ):
        if stale.exists():
            stale.unlink()
    write_insertion_outputs_from_task_predictions(
            task_pred_df=pd.DataFrame(loto_task_pred_rows),
            option_col=args.ranking_group,
            detail_path=out_dir / "prediction_loto_holdout_placement_detail.csv",
            summary_path=out_dir / "prediction_loto_holdout_placement_summary.csv",
            task_cols=placement_task_cols,
            fold_col="fold",
    )
    loto_baseline_df = pd.DataFrame()
    if args.loto_single_predictor_baselines:
        loto_baseline_df = compute_baseline_insertion_summaries(
            task_pred_df=pd.DataFrame(loto_task_pred_rows),
            option_col=args.ranking_group,
            output_path=out_dir / "prediction_loto_holdout_placement_baselines.csv",
            selectors=baseline_selectors,
            task_cols=placement_task_cols,
            fold_col="fold",
        )
    if (not args.loto_single_predictor_baselines) or loto_baseline_df.empty:
        stale = out_dir / "prediction_loto_holdout_placement_baselines.csv"
        if stale.exists():
            stale.unlink()

    if args.joint_ood_holdout:
        joint_train_col = group_col
        benchmark_col = "benchmark" if "benchmark" in loto_df.columns else None
        if benchmark_col:
            joint_task_pred_rows = []
            joint_summary, joint_preds = run_joint_holdout_cv(
                loto_df,
                joint_train_col,
                benchmark_col,
                predictors,
                pred_target,
                standardize=args.standardize,
                standardize_mode=args.cv_standardize_mode,
                center_by_group=loto_center_by_group,
                center_group_col=loto_center_group_col,
                group_norm_mode=loto_group_norm_mode,
                within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
                encoder_group_norm_mode=loto_encoder_group_norm_mode,
                encoder_group_col=loto_encoder_group_col,
                target_group_demean=loto_target_group_demean,
                target_group_col=loto_target_demean_groups,
                target_context_residualize_cols=(
                    residual_context_cols if args.cv_residualize_target_by_context else None
                ),
                target_context_transform=args.cv_residual_target_transform,
                target_context_std_eps=args.cv_residual_target_std_eps,
                target_context_eval=residual_eval_space,
                fewshot_context_calibration=args.cv_fewshot_context_calibration,
                fewshot_context_calibration_cols=fewshot_calibration_cols,
                fewshot_context_calibration_std_eps=args.cv_fewshot_context_calibration_std_eps,
                fewshot_context_calibration_min_group_size=args.cv_fewshot_context_calibration_min_group_size,
                fewshot_context_calibration_backoff=args.cv_fewshot_context_calibration_backoff,
                fewshot_context_calibration_k=args.cv_fewshot_context_calibration_k,
                fewshot_context_calibration_seed=args.cv_fewshot_context_calibration_seed,
                min_predictor_std=args.cv_min_predictor_std,
                prediction_clip=args.prediction_clip,
                prediction_clip_min=args.prediction_clip_min,
                prediction_clip_max=args.prediction_clip_max,
                model=pred_model,
                ridge_alpha=args.ridge_alpha,
                pairwise_option_col=args.ranking_group,
                pairwise_group_cols=pairwise_group_cols,
                fit_sample_weighting=args.fit_sample_weighting,
                fit_balance_real_synth=args.fit_balance_real_synth,
                overall_aggregation=args.overall_aggregation,
                task_prediction_rows_out=joint_task_pred_rows,
                task_prediction_option_col=args.ranking_group,
                task_prediction_task_cols=placement_task_cols,
                task_prediction_holdout_group_col=joint_train_col,
            )
            if not joint_summary.empty:
                joint_summary.to_csv(out_dir / "prediction_jointood_summary.csv", index=False)
            if not joint_preds.empty:
                joint_preds.to_csv(out_dir / "prediction_jointood_rows.csv", index=False)
            joint_task_pred_df = pd.DataFrame(joint_task_pred_rows)
            if not joint_task_pred_df.empty:
                joint_rank_context_cols = ["fold"] + list(ranking_context_cols or [])
                compute_ranking_summary(
                    joint_task_pred_df,
                    pred_target,
                    args.ranking_group,
                    out_dir / "prediction_jointood_rank_summary.csv",
                    context_cols=joint_rank_context_cols,
                    topk_frac=args.ranking_topk_frac,
                    topk_min=args.ranking_topk_min,
                    require_single_fold_task=True,
                )
                write_rank_detail_rows(
                    joint_task_pred_df,
                    pred_target,
                    args.ranking_group,
                    out_dir / "prediction_jointood_rank_detail.csv",
                    context_cols=joint_rank_context_cols,
                    require_single_fold_task=True,
                )
                write_insertion_outputs_from_task_predictions(
                    task_pred_df=joint_task_pred_df,
                    option_col=args.ranking_group,
                    detail_path=out_dir / "prediction_jointood_holdout_placement_detail.csv",
                    summary_path=out_dir / "prediction_jointood_holdout_placement_summary.csv",
                    task_cols=placement_task_cols,
                    fold_col="fold",
                )
                joint_baseline_df = pd.DataFrame()
                if args.jointood_single_predictor_baselines:
                    joint_baseline_df = compute_baseline_insertion_summaries(
                        task_pred_df=joint_task_pred_df,
                        option_col=args.ranking_group,
                        output_path=out_dir / "prediction_jointood_holdout_placement_baselines.csv",
                        selectors=baseline_selectors,
                        task_cols=placement_task_cols,
                        fold_col="fold",
                    )
                if (not args.jointood_single_predictor_baselines) or joint_baseline_df.empty:
                    stale = out_dir / "prediction_jointood_holdout_placement_baselines.csv"
                    if stale.exists():
                        stale.unlink()
            else:
                for stale in (
                    out_dir / "prediction_jointood_rank_summary.csv",
                    out_dir / "prediction_jointood_rank_detail.csv",
                    out_dir / "prediction_jointood_holdout_placement_summary.csv",
                    out_dir / "prediction_jointood_holdout_placement_detail.csv",
                    out_dir / "prediction_jointood_holdout_placement_baselines.csv",
                ):
                    if stale.exists():
                        stale.unlink()

    if args.sanity_permutation:
        perm_group = args.sanity_permute_group or "benchmark"
        perm_seed = args.sanity_permute_seed

        perm_lobo_summary, perm_lobo_preds = run_group_cv(
            cv_df,
            "benchmark",
            predictors,
            pred_target,
            standardize=args.standardize,
            standardize_mode=args.cv_standardize_mode,
            center_by_group=lobo_center_by_group,
            center_group_col=lobo_center_group_col,
            group_norm_mode=lobo_group_norm_mode,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=lobo_encoder_group_norm_mode,
            encoder_group_col=lobo_encoder_group_col,
            target_group_demean=lobo_target_group_demean,
            target_group_col=lobo_target_demean_groups,
            target_context_residualize_cols=(
                residual_context_cols if args.cv_residualize_target_by_context else None
            ),
            target_context_transform=args.cv_residual_target_transform,
            target_context_std_eps=args.cv_residual_target_std_eps,
            target_context_eval=residual_eval_space,
            fewshot_context_calibration=args.cv_fewshot_context_calibration,
            fewshot_context_calibration_cols=fewshot_calibration_cols,
            fewshot_context_calibration_std_eps=args.cv_fewshot_context_calibration_std_eps,
            fewshot_context_calibration_min_group_size=args.cv_fewshot_context_calibration_min_group_size,
            fewshot_context_calibration_backoff=args.cv_fewshot_context_calibration_backoff,
            fewshot_context_calibration_k=args.cv_fewshot_context_calibration_k,
            fewshot_context_calibration_seed=args.cv_fewshot_context_calibration_seed,
            min_predictor_std=args.cv_min_predictor_std,
            prediction_clip=args.prediction_clip,
            prediction_clip_min=args.prediction_clip_min,
            prediction_clip_max=args.prediction_clip_max,
            model=pred_model,
            ridge_alpha=args.ridge_alpha,
            permute_target=True,
            permute_group_col=perm_group,
            permute_seed=perm_seed,
            pairwise_option_col=args.ranking_group,
            pairwise_group_cols=pairwise_group_cols,
            fit_sample_weighting=args.fit_sample_weighting,
            fit_balance_real_synth=args.fit_balance_real_synth,
            overall_aggregation=args.overall_aggregation,
        )
        if not perm_lobo_summary.empty:
            perm_lobo_summary.to_csv(
                out_dir / "prediction_lobo_permutation_summary.csv", index=False
            )
        if not perm_lobo_preds.empty:
            perm_lobo_preds.to_csv(
                out_dir / "prediction_lobo_permutation_rows.csv", index=False
            )
            compute_ranking_summary(
                perm_lobo_preds,
                pred_target,
                args.ranking_group,
                out_dir / "prediction_lobo_permutation_rank_summary.csv",
                context_cols=ranking_context_cols,
                topk_frac=args.ranking_topk_frac,
                topk_min=args.ranking_topk_min,
            )

        perm_loto_summary, perm_loto_preds = run_group_cv(
            loto_df,
            group_col,
            predictors,
            pred_target,
            standardize=args.standardize,
            standardize_mode=args.cv_standardize_mode,
            center_by_group=loto_center_by_group,
            center_group_col=loto_center_group_col,
            group_norm_mode=loto_group_norm_mode,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=loto_encoder_group_norm_mode,
            encoder_group_col=loto_encoder_group_col,
            target_group_demean=loto_target_group_demean,
            target_group_col=loto_target_demean_groups,
            target_context_residualize_cols=(
                residual_context_cols if args.cv_residualize_target_by_context else None
            ),
            target_context_transform=args.cv_residual_target_transform,
            target_context_std_eps=args.cv_residual_target_std_eps,
            target_context_eval=residual_eval_space,
            fewshot_context_calibration=args.cv_fewshot_context_calibration,
            fewshot_context_calibration_cols=fewshot_calibration_cols,
            fewshot_context_calibration_std_eps=args.cv_fewshot_context_calibration_std_eps,
            fewshot_context_calibration_min_group_size=args.cv_fewshot_context_calibration_min_group_size,
            fewshot_context_calibration_backoff=args.cv_fewshot_context_calibration_backoff,
            fewshot_context_calibration_k=args.cv_fewshot_context_calibration_k,
            fewshot_context_calibration_seed=args.cv_fewshot_context_calibration_seed,
            min_predictor_std=args.cv_min_predictor_std,
            prediction_clip=args.prediction_clip,
            prediction_clip_min=args.prediction_clip_min,
            prediction_clip_max=args.prediction_clip_max,
            model=pred_model,
            ridge_alpha=args.ridge_alpha,
            permute_target=True,
            permute_group_col=perm_group,
            permute_seed=perm_seed,
            pairwise_option_col=args.ranking_group,
            pairwise_group_cols=pairwise_group_cols,
            fit_sample_weighting=args.fit_sample_weighting,
            fit_balance_real_synth=args.fit_balance_real_synth,
            overall_aggregation=args.overall_aggregation,
        )
        if not perm_loto_summary.empty:
            perm_loto_summary.to_csv(
                out_dir / "prediction_loto_permutation_summary.csv", index=False
            )
        if not perm_loto_preds.empty:
            perm_loto_preds.to_csv(
                out_dir / "prediction_loto_permutation_rows.csv", index=False
            )

    if (
        args.prediction_mixedlm
        and HAS_STATSMODELS
        and not args.cv_residualize_target_by_context
    ):
        random_slopes = _select_random_slopes(args, predictors)
        loto_mixed_summary, loto_mixed_preds = run_group_cv_mixedlm(
            loto_df,
            holdout_col=group_col,
            predictors=predictors,
            target=pred_target,
            random_group_col="benchmark",
            random_slopes=random_slopes,
            standardize=args.standardize,
            standardize_mode=args.cv_standardize_mode,
            center_by_group=loto_center_by_group,
            center_group_col=loto_center_group_col,
            group_norm_mode=loto_group_norm_mode,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=loto_encoder_group_norm_mode,
            encoder_group_col=loto_encoder_group_col,
            target_group_demean=loto_target_group_demean,
            target_group_col=loto_target_demean_groups,
            min_predictor_std=args.cv_min_predictor_std,
            prediction_clip=args.prediction_clip,
            prediction_clip_min=args.prediction_clip_min,
            prediction_clip_max=args.prediction_clip_max,
        )
        if not loto_mixed_summary.empty:
            loto_mixed_summary.to_csv(
                out_dir / "prediction_loto_mixed_summary.csv", index=False
            )
        if not loto_mixed_preds.empty:
            loto_mixed_preds.to_csv(
                out_dir / "prediction_loto_mixed_rows.csv", index=False
            )


def run_summary_report(out_dir, predictors, args):
    script_path = Path(__file__).resolve().parent / "summarize_leakage_free_results.py"
    if not script_path.exists():
        print(f"Warning: summary script not found at {script_path}")
        return
    output_path = out_dir / "summary_report.txt"
    cmd = [
        sys.executable,
        str(script_path),
        "--output-file",
        str(output_path),
        "--auc-table",
        str(out_dir / "auc_with_features.csv"),
        "--lobo-summary",
        str(out_dir / "prediction_lobo_summary.csv"),
        "--lobo-rank-summary",
        str(out_dir / "prediction_lobo_rank_summary.csv"),
        "--lobo-rank-baselines",
        str(out_dir / "prediction_lobo_rank_baselines.csv"),
        "--loto-summary",
        str(out_dir / "prediction_loto_summary.csv"),
        "--loto-rank-summary",
        str(out_dir / "prediction_loto_rank_summary.csv"),
        "--loto-holdout-placement-summary",
        str(out_dir / "prediction_loto_holdout_placement_summary.csv"),
        "--loto-holdout-placement-baselines",
        str(out_dir / "prediction_loto_holdout_placement_baselines.csv"),
        "--jointood-summary",
        str(out_dir / "prediction_jointood_summary.csv"),
        "--jointood-rank-summary",
        str(out_dir / "prediction_jointood_rank_summary.csv"),
        "--jointood-holdout-placement-summary",
        str(out_dir / "prediction_jointood_holdout_placement_summary.csv"),
        "--jointood-holdout-placement-baselines",
        str(out_dir / "prediction_jointood_holdout_placement_baselines.csv"),
        "--lobo-mixed-summary",
        str(out_dir / "prediction_lobo_mixed_summary.csv"),
        "--loto-mixed-summary",
        str(out_dir / "prediction_loto_mixed_summary.csv"),
        "--within-benchmark-slopes",
        str(out_dir / "within_benchmark_slopes.csv"),
        "--within-benchmark-slopes-univariate",
        str(out_dir / "within_benchmark_slopes_univariate.csv"),
        "--within-train-dataset-slopes-univariate",
        str(out_dir / "within_train_dataset_slopes_univariate.csv"),
        "--target",
        str(args.target),
        "--predictors",
        ",".join(predictors),
        "--linear-model",
        str(args.linear_model),
        "--ridge-alpha",
        str(args.ridge_alpha),
        "--prediction-model",
        str(args.prediction_model or args.linear_model),
        "--standardize",
        str(args.standardize),
        "--fit-sample-weighting",
        str(args.fit_sample_weighting),
        "--fit-balance-real-synth",
        str(args.fit_balance_real_synth),
        "--overall-aggregation",
        str(args.overall_aggregation),
        "--cv-standardize-mode",
        str(args.cv_standardize_mode),
        "--per-encoder",
        str(args.per_encoder),
        "--encoder-main-effects",
        str(args.encoder_main_effects),
        "--encoder-interactions",
        str(args.encoder_interactions),
        "--model-family-main-effects",
        str(args.model_family_main_effects),
        "--model-family-interactions",
        str(args.model_family_interactions),
    ]
    if args.prediction_target:
        cmd.extend(["--prediction-target", str(args.prediction_target)])
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Warning: summary generation failed for {out_dir}: {exc}")


def run_regression(df, predictors, target, output_path, linear_model="ols", ridge_alpha=1.0, use_mixedlm=True):
    lines = []
    df_model = filter_complete_rows(df, predictors, target)
    lines.append(f"Rows: {len(df_model)}")
    lines.append(f"Predictors: {', '.join(predictors)}")
    lines.append(f"Target: {target}")
    lines.append("")

    if df_model.empty:
        lines.append("No complete rows available for regression.")
        output_path.write_text("\n".join(lines))
        return

    if linear_model == "pairwise_rank":
        lines.append("Pairwise rank model selected; regression summary skipped.")
        output_path.write_text("\n".join(lines))
        return

    if linear_model == "ridge":
        z_df = df_model.copy()
        z_df[target] = _zscore(z_df[target])
        for col in predictors:
            z_df[col] = _zscore(z_df[col])
        X = z_df[predictors].to_numpy(dtype=float)
        y = z_df[target].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        penalty = np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(X.T @ X + float(ridge_alpha) * penalty, X.T @ y)
        y_pred = X.dot(coef)
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)
        lines.append(f"Ridge (standardized, alpha={ridge_alpha}):")
        lines.append(f"Intercept: {coef[0]:.6f}")
        for name, value in zip(predictors, coef[1:]):
            lines.append(f"{name}: {value:.6f}")
        lines.append(f"R2: {r2:.4f}")
        if use_mixedlm:
            lines.append("MixedLM skipped (ridge selected).")
    elif HAS_STATSMODELS:
        formula = f"{target} ~ " + " + ".join(predictors)
        lines.append("OLS:")
        try:
            result = smf.ols(formula, data=df_model).fit()
            lines.append(result.summary().as_text())
        except Exception as exc:
            lines.append(f"OLS failed: {exc}")

        if use_mixedlm:
            lines.append("")
            lines.append("MixedLM (random intercept by benchmark):")
            try:
                if df_model["benchmark"].nunique() < 2:
                    lines.append("Skipped: need at least 2 benchmarks")
                else:
                    mixed = smf.mixedlm(formula, data=df_model, groups=df_model["benchmark"]).fit(
                        reml=False, method="lbfgs"
                    )
                    lines.append(mixed.summary().as_text())
            except Exception as exc:
                lines.append(f"MixedLM failed: {exc}")
    else:
        X = df_model[predictors].to_numpy(dtype=float)
        y = df_model[target].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X.dot(coef)
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)

        lines.append("OLS (numpy):")
        lines.append(f"Intercept: {coef[0]:.6f}")
        for name, value in zip(predictors, coef[1:]):
            lines.append(f"{name}: {value:.6f}")
        lines.append(f"R2: {r2:.4f}")
        lines.append("Statsmodels not installed; mixedlm skipped.")

    output_path.write_text("\n".join(lines))


def collapse_mixed_dataset(name):
    if name is None:
        return None
    if name.startswith("spair_synthetic_"):
        return "spair_synthetic"
    return name


def is_mixed_dataset(name: str | None) -> bool:
    if not name:
        return False
    name = normalize_dataset_name(name)
    if name.startswith("synthetic"):
        return False
    return "_synthetic" in name


def filter_train_datasets_by_mode(df: pd.DataFrame, mode: str):
    if df.empty or "train_dataset" not in df.columns or mode == "all":
        return df, 0
    df = df.copy()
    mask = df["train_dataset"].apply(is_mixed_dataset)
    if mode == "base_only":
        filtered = int(mask.sum())
        return df[~mask].copy(), filtered
    if mode == "mixed_only":
        filtered = int((~mask).sum())
        return df[mask].copy(), filtered
    return df, 0


def main():
    parser = argparse.ArgumentParser(description="Leakage-free analysis pipeline.")
    parser.add_argument(
        "--snapshots-dir",
        nargs="+",
        default=["snapshots"],
        help="One or more snapshot root directories to scan.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/leakage_free",
        help="Directory to write output CSV files.",
    )
    parser.add_argument(
        "--mode",
        choices=["auc", "all"],
        default="auc",
        help="auc: only AUC/curve/prediction/regression. all: also dev/fixed tables.",
    )
    parser.add_argument(
        "--dev-benchmarks",
        default="middlebury,spair,pointodyssey",
        help="Comma-separated list of dev benchmarks for checkpoint selection.",
    )
    parser.add_argument(
        "--metric",
        default="pck",
        help="Metric column to use for selection and AUC.",
    )
    parser.add_argument(
        "--train-datasets-mode",
        choices=["all", "base_only", "mixed_only"],
        default="all",
        help="Filter train datasets before analysis (default: all).",
    )
    parser.add_argument(
        "--fixed-steps",
        type=int,
        default=None,
        help="Fixed training_steps for leakage-free evaluation (optional).",
    )
    parser.add_argument(
        "--fixed-policy",
        choices=["nearest", "exact"],
        default="nearest",
        help="How to handle missing fixed steps.",
    )
    parser.add_argument(
        "--auc-steps",
        type=int,
        default=5000,
        help="Max training_steps for AUC (default: 5000).",
    )
    parser.add_argument(
        "--auc-pad",
        action="store_true",
        help="Pad last value to auc-steps when runs end early.",
    )
    parser.add_argument(
        "--no-auc-pad",
        dest="auc_pad",
        action="store_false",
        help="Disable padding last value to auc-steps.",
    )
    parser.add_argument(
        "--coverage-csv",
        default="coverage_results.csv",
        help="Coverage CSV for flow/label metrics.",
    )
    parser.add_argument(
        "--variogram-csv",
        default=None,
        help="Variogram CSV for train/eval curve distances (optional).",
    )
    parser.add_argument(
        "--coverage-resnet-csv",
        default="coverage_resnet_results.csv",
        help="Coverage CSV for feature metrics.",
    )
    parser.add_argument(
        "--coverage-dino-csv",
        default=None,
        help="Coverage CSV for DINO feature metrics (optional).",
    )
    parser.add_argument(
        "--coverage-hof-csv",
        default=None,
        help="Coverage CSV for HOF motion metrics (optional).",
    )
    parser.add_argument(
        "--flow-stats-dir",
        default=None,
        help="Directory of flow extraction stats JSON files (optional).",
    )
    parser.add_argument(
        "--flow-stats-log-eps",
        type=float,
        default=1e-6,
        help="Epsilon for log density features from flow stats.",
    )
    parser.add_argument(
        "--use-flow-density-predictors",
        dest="use_flow_density_predictors",
        action="store_true",
        help="Include log density predictors from flow stats when available.",
    )
    parser.add_argument(
        "--no-use-flow-density-predictors",
        dest="use_flow_density_predictors",
        action="store_false",
        help="Disable flow density predictors.",
    )
    parser.set_defaults(use_flow_density_predictors=False)
    parser.add_argument(
        "--flow-density-interactions",
        dest="flow_density_interactions",
        action="store_true",
        help="Add coverage x log_avg_flows_eval interaction predictors.",
    )
    parser.add_argument(
        "--no-flow-density-interactions",
        dest="flow_density_interactions",
        action="store_false",
        help="Disable coverage x log_avg_flows_eval interactions.",
    )
    parser.set_defaults(flow_density_interactions=False)
    parser.add_argument(
        "--custom-interactions",
        default="",
        help=(
            "Comma-separated custom interaction specs (e.g., "
            "flow_train_to_eval_eps1px*log_n_samples_eval,dino_eval_to_train_mean_dist*log_n_samples_eval). "
            "Use @scale suffix to scale interaction terms."
        ),
    )
    parser.add_argument(
        "--flow-eps-values",
        default="1,2,4,8,16,32,64",
        help="Comma-separated flow epsilon values to use as predictors.",
    )
    parser.add_argument(
        "--use-flow-eps-predictors",
        dest="use_flow_eps_predictors",
        action="store_true",
        help="Include flow epsilon coverage predictors when available.",
    )
    parser.add_argument(
        "--no-flow-eps-predictors",
        dest="use_flow_eps_predictors",
        action="store_false",
        help="Disable flow epsilon coverage predictors.",
    )
    parser.add_argument(
        "--use-flow-eps-weighted-predictors",
        dest="use_flow_eps_weighted_predictors",
        action="store_true",
        help="Include weighted flow epsilon coverage predictors when available.",
    )
    parser.add_argument(
        "--no-flow-eps-weighted-predictors",
        dest="use_flow_eps_weighted_predictors",
        action="store_false",
        help="Disable weighted flow epsilon coverage predictors.",
    )
    parser.add_argument(
        "--flow-eps-rings",
        dest="flow_eps_rings",
        action="store_true",
        help="Convert cumulative flow epsilon ladder predictors into ring/delta bins.",
    )
    parser.add_argument(
        "--no-flow-eps-rings",
        dest="flow_eps_rings",
        action="store_false",
        help="Keep cumulative flow epsilon ladder predictors (default).",
    )
    parser.add_argument(
        "--flow-mmd-csv",
        default="flow_mmd_results.csv",
        help="Flow MMD CSV.",
    )
    parser.add_argument(
        "--feature-mmd-csv",
        default="feature_mmd_results.csv",
        help="Feature MMD CSV.",
    )
    parser.add_argument(
        "--dino-mmd-csv",
        default=None,
        help="DINO MMD CSV (optional).",
    )
    parser.add_argument(
        "--logit-coverage",
        dest="logit_coverage",
        action="store_true",
        help="Apply logit transform to coverage metrics.",
    )
    parser.add_argument(
        "--no-logit-coverage",
        dest="logit_coverage",
        action="store_false",
        help="Disable logit transform on coverage metrics.",
    )
    parser.set_defaults(logit_coverage=True)
    parser.set_defaults(use_flow_eps_predictors=True)
    parser.set_defaults(use_flow_eps_weighted_predictors=False)
    parser.set_defaults(flow_eps_rings=False)
    parser.add_argument(
        "--rename-coverage",
        action="store_true",
        help="Add explicit precision/recall coverage column names in outputs.",
    )
    parser.add_argument(
        "--distance-radius-norm",
        choices=["none", "divide"],
        default="none",
        help="Normalize distance predictors by the corresponding train radius.",
    )
    parser.add_argument(
        "--distance-ratio-transform",
        choices=["none", "log1p"],
        default="none",
        help="Transform distance/radius ratio predictors (e.g., log1p).",
    )
    parser.add_argument(
        "--radius-transform",
        choices=["keep", "log", "drop"],
        default="keep",
        help="Transform radius predictors (default: keep).",
    )
    parser.add_argument(
        "--radius-eps",
        type=float,
        default=1e-6,
        help="Epsilon for radius normalization/log transform.",
    )
    parser.add_argument(
        "--distance-radius-floor",
        type=float,
        default=0.0,
        help="Minimum radius when normalizing distances.",
    )
    parser.add_argument(
        "--predictors",
        default=None,
        help="Comma-separated predictors for regression/prediction.",
    )
    parser.add_argument(
        "--include-kl",
        dest="include_kl",
        action="store_true",
        help="Append KL-divergence predictors when available.",
    )
    parser.add_argument(
        "--no-include-kl",
        dest="include_kl",
        action="store_false",
        help="Disable automatic KL-divergence predictors.",
    )
    parser.set_defaults(include_kl=False)
    parser.add_argument(
        "--target",
        default=None,
        help="Target column for regression/prediction.",
    )
    parser.add_argument(
        "--prediction-target",
        default=None,
        help="Target column for LOBO/LOTO predictions (optional).",
    )
    parser.add_argument(
        "--linear-model",
        choices=["ols", "ridge", "pairwise_rank"],
        default="ols",
        help="Linear model for regression/prediction (default: ols).",
    )
    parser.add_argument(
        "--prediction-model",
        choices=["ols", "ridge", "pairwise_rank"],
        default=None,
        help="Model for LOBO/LOTO predictions (defaults to linear-model).",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="Ridge penalty strength when linear-model=ridge.",
    )
    parser.add_argument(
        "--run-stability-selection",
        action="store_true",
        help="Run stability selection to identify robust predictors.",
    )
    parser.add_argument(
        "--stability-n-bootstrap",
        type=int,
        default=100,
        help="Number of bootstrap iterations for stability selection.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.7,
        help="Stability threshold (fraction of bootstraps predictor must be selected).",
    )
    parser.add_argument(
        "--run-family-comparison",
        action="store_true",
        help="Compare predictor families (flow vs dino vs mmd) separately.",
    )
    parser.add_argument(
        "--run-univariate-comparison",
        action="store_true",
        help="Fit each predictor individually in separate LOBO runs.",
    )
    parser.add_argument(
        "--relative-target-baseline",
        default=None,
        help="Training dataset name to use as per-benchmark baseline for delta target.",
    )
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        help="Standardize predictors in prediction models.",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Disable predictor standardization.",
    )
    parser.set_defaults(standardize=True)
    parser.add_argument(
        "--cv-standardize-mode",
        choices=["local", "global"],
        default="local",
        help="Standardization mode: 'local' = within each CV fold (removes between-group variance), "
             "'global' = once using all data before CV (preserves between-group variance). Default: local.",
    )
    parser.add_argument(
        "--collapse-cv-cells",
        dest="collapse_cv_cells",
        action="store_true",
        help=(
            "Collapse CV rows to one row per (train_dataset, benchmark) cell "
            "before LOBO/LOTO/joint-OOD; numeric columns use mean."
        ),
    )
    parser.add_argument(
        "--no-collapse-cv-cells",
        dest="collapse_cv_cells",
        action="store_false",
        help="Disable CV cell collapsing and use raw row-level CV.",
    )
    parser.set_defaults(collapse_cv_cells=True)
    parser.add_argument(
        "--cv-repeat-aggregation",
        choices=["none", "mean", "median"],
        default="none",
        help=(
            "Aggregate repeated CV rows before LOBO/LOTO/Joint-OOD using context-aware "
            "grouping; numeric columns use the selected reducer (default: none)."
        ),
    )
    parser.add_argument(
        "--fit-sample-weighting",
        choices=["none", "inverse_benchmark", "inverse_train_dataset", "inverse_task"],
        default="none",
        help=(
            "Training-loss weighting mode for OLS/Ridge in CV folds. "
            "'inverse_task' uses benchmark x model-family-encoder when available."
        ),
    )
    parser.add_argument(
        "--fit-balance-real-synth",
        dest="fit_balance_real_synth",
        action="store_true",
        help="Rebalance fit loss so real and synthetic train-dataset rows carry equal total weight.",
    )
    parser.add_argument(
        "--no-fit-balance-real-synth",
        dest="fit_balance_real_synth",
        action="store_false",
        help="Disable real-vs-synthetic fit balancing.",
    )
    parser.set_defaults(fit_balance_real_synth=False)
    parser.add_argument(
        "--overall-aggregation",
        choices=["micro", "macro_fold"],
        default="micro",
        help=(
            "How to compute __overall__ prediction metrics in LOBO/LOTO/Joint summaries: "
            "micro pools all rows; macro_fold averages per-fold metrics."
        ),
    )
    parser.add_argument(
        "--skip-prediction",
        action="store_true",
        help="Skip LOBO/LOTO prediction validation.",
    )
    parser.add_argument(
        "--joint-ood-holdout",
        dest="joint_ood_holdout",
        action="store_true",
        help="Run joint OOD CV: hold out both train_dataset and benchmark per fold.",
    )
    parser.add_argument(
        "--no-joint-ood-holdout",
        dest="joint_ood_holdout",
        action="store_false",
        help="Disable joint OOD CV.",
    )
    parser.set_defaults(joint_ood_holdout=False)
    parser.add_argument(
        "--center-predictors-by-benchmark",
        dest="center_predictors_by_benchmark",
        action="store_true",
        help="Center predictors by benchmark (train-only stats) during CV.",
    )
    parser.add_argument(
        "--no-center-predictors-by-benchmark",
        dest="center_predictors_by_benchmark",
        action="store_false",
        help="Disable benchmark-centering of predictors during CV.",
    )
    parser.set_defaults(center_predictors_by_benchmark=False)
    parser.add_argument(
        "--benchmark-normalize-predictors",
        choices=["auto", "none", "center", "zscore"],
        default="none",
        help="Normalize predictors within each benchmark before regression/prediction.",
    )
    parser.add_argument(
        "--benchmark-normalize-scope",
        choices=["all", "report_only", "none"],
        default="none",
        help="Where to apply benchmark normalization (default: none).",
    )
    parser.add_argument(
        "--benchmark-normalize-target",
        choices=["none", "center", "zscore"],
        default="none",
        help="Normalize target within each benchmark before regression/prediction.",
    )
    parser.add_argument(
        "--cv-normalize-predictors-by-benchmark",
        choices=["none", "center", "zscore"],
        default="none",
        help="Apply train-only benchmark normalization inside LOBO/LOTO folds.",
    )
    parser.add_argument(
        "--cv-within-benchmark-predictor-norm",
        choices=["none", "rank", "zscore"],
        default="none",
        help="Normalize predictors within each benchmark using all fold rows (rank/zscore).",
    )
    parser.add_argument(
        "--cv-normalize-predictors-by-encoder",
        choices=["none", "center", "zscore"],
        default="none",
        help="Apply train-only encoder-config normalization inside LOBO/LOTO folds.",
    )
    parser.add_argument(
        "--lobo-model-centered",
        action="store_true",
        help="Center predictors and demean target by model_family_encoder for LOBO.",
    )
    parser.add_argument(
        "--loto-benchmark-centered",
        action="store_true",
        help="Center predictors by benchmark and demean target by benchmark + model_family_encoder for LOTO.",
    )
    parser.add_argument(
        "--cv-demean-target-by-encoder",
        dest="cv_demean_target_by_encoder",
        action="store_true",
        help="Demean target by encoder config inside LOBO/LOTO folds.",
    )
    parser.add_argument(
        "--cv-demean-target-by-benchmark",
        dest="cv_demean_target_by_benchmark",
        action="store_true",
        help="Demean target by benchmark inside LOBO/LOTO folds (train-only means).",
    )
    parser.add_argument(
        "--no-cv-demean-target-by-encoder",
        dest="cv_demean_target_by_encoder",
        action="store_false",
        help="Disable encoder-config target demeaning inside LOBO/LOTO folds.",
    )
    parser.add_argument(
        "--no-cv-demean-target-by-benchmark",
        dest="cv_demean_target_by_benchmark",
        action="store_false",
        help="Disable benchmark target demeaning inside LOBO/LOTO folds.",
    )
    parser.set_defaults(cv_demean_target_by_encoder=False)
    parser.set_defaults(cv_demean_target_by_benchmark=False)
    parser.add_argument(
        "--cv-residualize-target-by-context",
        dest="cv_residualize_target_by_context",
        action="store_true",
        help=(
            "Residualize target by joint context means computed from training rows "
            "inside each CV fold."
        ),
    )
    parser.add_argument(
        "--no-cv-residualize-target-by-context",
        dest="cv_residualize_target_by_context",
        action="store_false",
        help="Disable fold-safe joint-context target residualization.",
    )
    parser.set_defaults(cv_residualize_target_by_context=False)
    parser.add_argument(
        "--cv-residual-context-cols",
        default="benchmark,model_family_encoder",
        help=(
            "CSV context columns for fold-safe target residualization "
            "(default: benchmark,model_family_encoder)."
        ),
    )
    parser.add_argument(
        "--cv-residual-eval-space",
        choices=["absolute", "residual"],
        default="residual",
        help=(
            "When residualizing target by context: evaluate predictions in residual "
            "space (default) or map predictions back to absolute space."
        ),
    )
    parser.add_argument(
        "--cv-residual-target-transform",
        choices=["residual", "zscore"],
        default="residual",
        help=(
            "Target transform within residual contexts: "
            "'residual' subtracts train-context mean; "
            "'zscore' subtracts train-context mean and divides by train-context std."
        ),
    )
    parser.add_argument(
        "--cv-residual-target-std-eps",
        type=float,
        default=1e-9,
        help="Epsilon floor for context std when --cv-residual-target-transform=zscore.",
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration",
        dest="cv_fewshot_context_calibration",
        action="store_true",
        help=(
            "Apply leaky per-fold context calibration (mean+std affine) on held-out rows "
            "after prediction, to estimate few-shot calibration upside."
        ),
    )
    parser.add_argument(
        "--no-cv-fewshot-context-calibration",
        dest="cv_fewshot_context_calibration",
        action="store_false",
        help="Disable leaky per-fold context calibration (default).",
    )
    parser.set_defaults(cv_fewshot_context_calibration=False)
    parser.add_argument(
        "--cv-fewshot-context-calibration-cols",
        default="benchmark,model_family_encoder",
        help=(
            "CSV context columns for leaky few-shot calibration stats "
            "(default: benchmark,model_family_encoder)."
        ),
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-std-eps",
        type=float,
        default=1e-9,
        help="Epsilon floor for prediction/target std in few-shot calibration.",
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-min-group-size",
        type=int,
        default=2,
        help=(
            "Minimum rows per calibration context group; smaller groups back off to coarser "
            "contexts (or global if none)."
        ),
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-backoff",
        dest="cv_fewshot_context_calibration_backoff",
        action="store_true",
        help=(
            "Enable hierarchical backoff for few-shot calibration contexts "
            "(e.g., benchmark,model_family_encoder -> benchmark -> global)."
        ),
    )
    parser.add_argument(
        "--no-cv-fewshot-context-calibration-backoff",
        dest="cv_fewshot_context_calibration_backoff",
        action="store_false",
        help="Disable hierarchical backoff for few-shot calibration contexts.",
    )
    parser.set_defaults(cv_fewshot_context_calibration_backoff=True)
    parser.add_argument(
        "--cv-fewshot-context-calibration-k",
        type=int,
        default=0,
        help=(
            "True few-shot mode: sample up to K calibration rows per context group in each held-out "
            "fold, fit calibrator on those rows only, and evaluate on remaining rows. "
            "Use 0 to keep full-fold calibration behavior."
        ),
    )
    parser.add_argument(
        "--cv-fewshot-context-calibration-seed",
        type=int,
        default=0,
        help="Random seed for fold-level K-shot calibration row sampling.",
    )
    parser.add_argument(
        "--loto-single-predictor-baselines",
        dest="loto_single_predictor_baselines",
        action="store_true",
        help="Compute LOTO holdout-placement single-predictor baseline block (default: on).",
    )
    parser.add_argument(
        "--no-loto-single-predictor-baselines",
        dest="loto_single_predictor_baselines",
        action="store_false",
        help="Skip LOTO holdout-placement single-predictor baseline block.",
    )
    parser.set_defaults(loto_single_predictor_baselines=True)
    parser.add_argument(
        "--jointood-single-predictor-baselines",
        dest="jointood_single_predictor_baselines",
        action="store_true",
        help="Compute Joint-OOD holdout-placement single-predictor baseline block (default: on).",
    )
    parser.add_argument(
        "--no-jointood-single-predictor-baselines",
        dest="jointood_single_predictor_baselines",
        action="store_false",
        help="Skip Joint-OOD holdout-placement single-predictor baseline block.",
    )
    parser.set_defaults(jointood_single_predictor_baselines=True)
    parser.add_argument(
        "--cv-min-predictor-std",
        type=float,
        default=0.0,
        help="Drop per-fold predictors with std below this threshold.",
    )
    parser.add_argument(
        "--sanity-permutation",
        dest="sanity_permutation",
        action="store_true",
        help="Run permutation sanity checks for LOBO/LOTO.",
    )
    parser.add_argument(
        "--no-sanity-permutation",
        dest="sanity_permutation",
        action="store_false",
        help="Disable permutation sanity checks.",
    )
    parser.set_defaults(sanity_permutation=False)
    parser.add_argument(
        "--sanity-permute-group",
        default="benchmark",
        help="Group column for permutation shuffles (default: benchmark).",
    )
    parser.add_argument(
        "--sanity-permute-seed",
        type=int,
        default=0,
        help="Random seed base for permutation shuffles.",
    )
    parser.add_argument(
        "--sanity-direction-audit",
        dest="sanity_direction_audit",
        action="store_true",
        help="Emit a direction-audit file for one LOBO fold.",
    )
    parser.add_argument(
        "--no-sanity-direction-audit",
        dest="sanity_direction_audit",
        action="store_false",
        help="Disable direction-audit output.",
    )
    parser.set_defaults(sanity_direction_audit=False)
    parser.add_argument(
        "--prediction-clip",
        dest="prediction_clip",
        action="store_true",
        help="Clip predictions to training target range within each fold.",
    )
    parser.add_argument(
        "--no-prediction-clip",
        dest="prediction_clip",
        action="store_false",
        help="Disable prediction clipping.",
    )
    parser.set_defaults(prediction_clip=False)
    parser.add_argument(
        "--prediction-clip-min",
        type=float,
        default=None,
        help="Override minimum clip value (defaults to fold min target).",
    )
    parser.add_argument(
        "--prediction-clip-max",
        type=float,
        default=None,
        help="Override maximum clip value (defaults to fold max target).",
    )
    parser.add_argument(
        "--encoder-interactions",
        dest="encoder_interactions",
        action="store_true",
        help="Add encoder-config indicators and predictor interactions for pooled analysis.",
    )
    parser.add_argument(
        "--no-encoder-interactions",
        dest="encoder_interactions",
        action="store_false",
        help="Disable encoder-config interactions for pooled analysis.",
    )
    parser.set_defaults(encoder_interactions=False)
    parser.add_argument(
        "--encoder-interaction-baseline",
        default="TT",
        help="Baseline encoder config for pooled interactions (FF, FT, TF, TT).",
    )
    parser.add_argument(
        "--encoder-main-effects",
        dest="encoder_main_effects",
        action="store_true",
        help="Include encoder-config main effects in pooled analysis.",
    )
    parser.add_argument(
        "--no-encoder-main-effects",
        dest="encoder_main_effects",
        action="store_false",
        help="Exclude encoder-config main effects in pooled analysis.",
    )
    parser.set_defaults(encoder_main_effects=True)
    parser.add_argument(
        "--model-family-interactions",
        dest="model_family_interactions",
        action="store_true",
        help="Add model-family indicators and predictor interactions for pooled analysis.",
    )
    parser.add_argument(
        "--no-model-family-interactions",
        dest="model_family_interactions",
        action="store_false",
        help="Disable model-family interactions for pooled analysis.",
    )
    parser.set_defaults(model_family_interactions=False)
    parser.add_argument(
        "--model-family-interaction-baseline",
        default=MODEL_FAMILY_DEFAULT,
        help="Baseline model_family for pooled interactions.",
    )
    parser.add_argument(
        "--spair-indicator-interactions",
        dest="spair_indicator_interactions",
        action="store_true",
        help="Add a spair indicator and predictor interactions for pooled analysis.",
    )
    parser.add_argument(
        "--no-spair-indicator-interactions",
        dest="spair_indicator_interactions",
        action="store_false",
        help="Disable spair indicator interactions.",
    )
    parser.set_defaults(spair_indicator_interactions=False)
    parser.add_argument(
        "--model-family-main-effects",
        dest="model_family_main_effects",
        action="store_true",
        help="Include model-family main effects in pooled analysis.",
    )
    parser.add_argument(
        "--no-model-family-main-effects",
        dest="model_family_main_effects",
        action="store_false",
        help="Exclude model-family main effects in pooled analysis.",
    )
    parser.set_defaults(model_family_main_effects=False)
    parser.add_argument(
        "--exclude-encoder-configs",
        default="",
        help="Comma-separated encoder configs to exclude (FF, FT, TF, TT).",
    )
    parser.add_argument(
        "--exclude-model-families",
        default="",
        help="Comma-separated model_family values to exclude (e.g., raft,catspp).",
    )
    parser.add_argument(
        "--rank-target",
        dest="rank_target",
        action="store_true",
        help="Replace target with within-group rank of rank-target-source.",
    )
    parser.add_argument(
        "--no-rank-target",
        dest="rank_target",
        action="store_false",
        help="Disable rank-based target transformation.",
    )
    parser.set_defaults(rank_target=False)
    parser.add_argument(
        "--rank-target-source",
        default=None,
        help="Source column to rank (defaults to target).",
    )
    parser.add_argument(
        "--rank-target-col",
        default=None,
        help="Output column name for rank target (default: <source>_rank).",
    )
    parser.add_argument(
        "--rank-target-group",
        default="benchmark",
        help="Column to group by when ranking target (default: benchmark).",
    )
    parser.add_argument(
        "--rank-target-with-encoder",
        action="store_true",
        help="Include encoder_config when ranking target.",
    )
    parser.add_argument(
        "--no-rank-target-with-encoder",
        dest="rank_target_with_encoder",
        action="store_false",
        help="Do not include encoder_config when ranking target.",
    )
    parser.set_defaults(rank_target_with_encoder=False)
    parser.add_argument(
        "--rank-target-with-model",
        action="store_true",
        help="Include model_family when ranking target.",
    )
    parser.add_argument(
        "--no-rank-target-with-model",
        dest="rank_target_with_model",
        action="store_false",
        help="Do not include model_family when ranking target.",
    )
    parser.set_defaults(rank_target_with_model=False)
    parser.add_argument(
        "--exclude-benchmarks",
        default="",
        help="Comma-separated benchmarks to exclude from analysis (including LOBO/LOTO).",
    )
    parser.add_argument(
        "--exclude-train-datasets",
        default="",
        help="Comma-separated train_dataset values to exclude from analysis.",
    )
    parser.add_argument(
        "--ranking-group",
        default="train_dataset",
        help=(
            "Column to rank options within each benchmark "
            "(default: train_dataset). "
            "Special values: train_dataset_encoder, "
            "train_dataset_model_family_encoder."
        ),
    )
    parser.add_argument(
        "--ranking-context-cols",
        default="",
        help=(
            "Optional CSV of extra grouping columns for rank evaluation "
            "(e.g., model_family_encoder)."
        ),
    )
    parser.add_argument(
        "--pairwise-group-cols",
        default="benchmark",
        help=(
            "CSV columns that define independent pairwise training groups "
            "(default: benchmark; example: benchmark,model_family_encoder)."
        ),
    )
    parser.add_argument(
        "--allow-benchmark-ranking-group",
        "--allow-non-benchmark-ranking-group",
        dest="allow_benchmark_ranking_group",
        action="store_true",
        help=(
            "Allow --ranking-group benchmark for option ranking. "
            "Default behavior refuses benchmark as an option group because it is the evaluation context."
        ),
    )
    parser.add_argument(
        "--ranking-topk-frac",
        type=float,
        default=0.2,
        help="Top-k fraction for rank summary (default: 0.2 for top-20%%).",
    )
    parser.add_argument(
        "--ranking-topk-min",
        type=int,
        default=1,
        help="Minimum k for top-k evaluation (default: 1).",
    )
    parser.add_argument(
        "--dual-target",
        action="store_true",
        help="Run analysis twice (delta and absolute targets) into subdirectories.",
    )
    parser.add_argument(
        "--dual-target-dirs",
        default="delta,absolute",
        help="Comma-separated subdirs for dual-target outputs.",
    )
    parser.add_argument(
        "--additional-targets",
        default=None,
        help="Comma-separated list of extra target columns to analyze (optional).",
    )
    parser.add_argument(
        "--additional-target-dirs",
        default=None,
        help="Comma-separated subdirs for additional targets (defaults to target names).",
    )
    parser.add_argument(
        "--strict-dataset-match",
        action="store_true",
        help="Disable dataset name fallbacks when matching coverage/MMD pairs.",
    )
    parser.add_argument(
        "--allow-unsplit-coverage",
        dest="allow_unsplit_coverage",
        action="store_true",
        help="Allow coverage matches without split tags.",
    )
    parser.add_argument(
        "--no-allow-unsplit-coverage",
        dest="allow_unsplit_coverage",
        action="store_false",
        help="Disable coverage matches without split tags.",
    )
    parser.set_defaults(allow_unsplit_coverage=True)
    parser.add_argument(
        "--allow-unsplit-flow-stats",
        dest="allow_unsplit_flow_stats",
        action="store_true",
        help="Allow flow stats matches without split tags.",
    )
    parser.add_argument(
        "--no-allow-unsplit-flow-stats",
        dest="allow_unsplit_flow_stats",
        action="store_false",
        help="Disable flow stats matches without split tags.",
    )
    parser.set_defaults(allow_unsplit_flow_stats=True)
    parser.add_argument(
        "--allow-unsplit-mmd",
        dest="allow_unsplit_mmd",
        action="store_true",
        help="Allow MMD matches without split tags.",
    )
    parser.add_argument(
        "--no-allow-unsplit-mmd",
        dest="allow_unsplit_mmd",
        action="store_false",
        help="Disable MMD matches without split tags.",
    )
    parser.set_defaults(allow_unsplit_mmd=True)
    parser.add_argument(
        "--prediction-mixedlm",
        dest="prediction_mixedlm",
        action="store_true",
        help="Run MixedLM-based LOBO/LOTO (random intercept by benchmark).",
    )
    parser.add_argument(
        "--no-prediction-mixedlm",
        dest="prediction_mixedlm",
        action="store_false",
        help="Disable MixedLM-based LOBO/LOTO.",
    )
    parser.set_defaults(prediction_mixedlm=HAS_STATSMODELS)
    parser.add_argument(
        "--regression-mixedlm",
        dest="regression_mixedlm",
        action="store_true",
        help="Include MixedLM fit in regression summary.",
    )
    parser.add_argument(
        "--no-regression-mixedlm",
        dest="regression_mixedlm",
        action="store_false",
        help="Disable MixedLM fit in regression summary.",
    )
    parser.set_defaults(regression_mixedlm=HAS_STATSMODELS)
    parser.add_argument(
        "--mixedlm-random-slopes",
        default=None,
        help="Comma-separated predictors to include as random slopes in MixedLM.",
    )
    parser.add_argument(
        "--skip-regression",
        action="store_true",
        help="Skip regression summary.",
    )
    parser.add_argument(
        "--per-encoder",
        dest="per_encoder",
        action="store_true",
        help="Run regression/prediction per encoder config (pretrained/freeze).",
    )
    parser.add_argument(
        "--no-per-encoder",
        dest="per_encoder",
        action="store_false",
        help="Disable per-encoder regression/prediction outputs.",
    )
    parser.set_defaults(per_encoder=True)
    parser.add_argument(
        "--run-summary",
        dest="run_summary",
        action="store_true",
        help="Generate summary_report.txt for overall and per-encoder outputs.",
    )
    parser.add_argument(
        "--no-run-summary",
        dest="run_summary",
        action="store_false",
        help="Skip summary_report.txt generation.",
    )
    parser.set_defaults(run_summary=True)
    parser.add_argument(
        "--loto-collapse-mixed",
        action="store_true",
        help="Collapse spair_synthetic_* to spair_synthetic for LOTO grouping.",
    )
    args = parser.parse_args()
    if int(args.cv_fewshot_context_calibration_min_group_size) < 1:
        raise SystemExit(
            "--cv-fewshot-context-calibration-min-group-size must be >= 1."
        )
    if int(args.cv_fewshot_context_calibration_k) < 0:
        raise SystemExit(
            "--cv-fewshot-context-calibration-k must be >= 0."
        )
    if args.ranking_group == "benchmark" and not args.allow_benchmark_ranking_group:
        raise SystemExit(
            "Refusing to run with --ranking-group "
            f"'{args.ranking_group}'. "
            "Use --ranking-group train_dataset (recommended), or pass "
            "--allow-benchmark-ranking-group to override intentionally."
        )
    target_default = args.target is None
    if args.target is None:
        args.target = "auc_normalized"

    dev_benchmarks = [b.strip() for b in args.dev_benchmarks.split(",") if b.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_rows = []
    dev_summary_rows = []
    fixed_rows = []
    auc_rows = []
    curve_rows = []

    snapshot_dirs = list_snapshot_dirs(args.snapshots_dir)
    for snapshot_dir in snapshot_dirs:
        csv_path = snapshot_dir / "validation_results.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Skipping {snapshot_dir}: could not read CSV ({exc})")
            continue
        if args.metric not in df.columns:
            print(f"Skipping {snapshot_dir}: metric '{args.metric}' not found")
            continue

        df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
        df = df.dropna(subset=["training_steps", "benchmark", args.metric])
        if df.empty:
            print(f"Skipping {snapshot_dir}: no usable rows")
            continue

        df["benchmark"] = df["benchmark"].str.lower()

        run_info = parse_run_name(snapshot_dir)
        run_info["run_id"] = str(snapshot_dir)
        run_info["train_dataset"] = normalize_dataset_name(run_info.get("train_dataset"))

        fixed_step = None
        fixed_df = None
        fixed_step_map = {}
        fixed_step_col = None
        if args.fixed_steps is not None:
            fixed_step_col = f"{args.metric}_at_{int(args.fixed_steps)}"
            fixed_step = args.fixed_steps
            if args.fixed_policy == "nearest":
                fixed_step = find_nearest_step(df, args.fixed_steps)
            if fixed_step is None:
                print(f"Skipping {snapshot_dir}: no steps for fixed selection")
                continue
            fixed_df = df[df["training_steps"] == fixed_step].copy()
            if fixed_df.empty:
                print(f"Skipping {snapshot_dir}: fixed step {fixed_step} missing")
                if args.fixed_policy == "exact":
                    continue
            else:
                if args.fixed_policy == "exact":
                    expected_benchmarks = set(df["benchmark"].unique())
                    found_benchmarks = set(fixed_df["benchmark"].unique())
                    missing_benchmarks = expected_benchmarks - found_benchmarks
                    if missing_benchmarks:
                        missing_str = ", ".join(sorted(missing_benchmarks))
                        print(
                            f"Skipping {snapshot_dir}: fixed step {fixed_step} missing benchmarks {missing_str}"
                        )
                        continue
                fixed_step_map = (
                    fixed_df.groupby("benchmark")[args.metric].mean().to_dict()
                )

        if args.mode == "all":
            dev_step, dev_score, dev_count = select_dev_step(df, dev_benchmarks, args.metric)
            if dev_step is None:
                print(f"Skipping {snapshot_dir}: no dev benchmarks found")
            else:
                sel_df = df[df["training_steps"] == dev_step].copy()
                if sel_df.empty:
                    print(f"Skipping {snapshot_dir}: dev step {dev_step} missing")
                else:
                    epoch_val = sel_df["epoch"].iloc[0] if "epoch" in sel_df.columns else np.nan
                    dev_summary_rows.append(
                        {
                            **run_info,
                            "selected_training_steps": int(dev_step),
                            "selected_epoch": int(epoch_val) if pd.notna(epoch_val) else np.nan,
                            "dev_score": float(dev_score),
                            "dev_benchmarks": ",".join(dev_benchmarks),
                            "dev_count": int(dev_count),
                        }
                    )
                    for _, row in sel_df.iterrows():
                        dev_rows.append(
                            {
                                **run_info,
                                "selected_training_steps": int(dev_step),
                                "selected_epoch": int(row["epoch"]) if "epoch" in sel_df.columns else np.nan,
                                "dev_score": float(dev_score),
                                "dev_benchmarks": ",".join(dev_benchmarks),
                                "dev_count": int(dev_count),
                                "benchmark": row["benchmark"],
                                args.metric: float(row[args.metric]),
                            }
                        )

            if args.fixed_steps is not None and fixed_df is not None and not fixed_df.empty:
                for _, row in fixed_df.iterrows():
                    fixed_rows.append(
                        {
                            **run_info,
                            "fixed_training_steps": int(fixed_step),
                            "requested_training_steps": int(args.fixed_steps),
                            "benchmark": row["benchmark"],
                            args.metric: float(row[args.metric]),
                        }
                    )

        curve_stats = compute_curve_stats(df, args.metric)
        curve_map = {row["benchmark"]: row for row in curve_stats}
        for bench, sub in df.groupby("benchmark"):
            auc, n_points, last_step = compute_auc(sub, args.metric, args.auc_steps, args.auc_pad)
            if last_step is not None and last_step < args.auc_steps:
                print(
                    f"Warning: {snapshot_dir.name} {bench} ends at {last_step} < {args.auc_steps} steps"
                )
            curve_info = curve_map.get(bench, {})
            curve_extra = {k: v for k, v in curve_info.items() if k != "benchmark"}
            row = {
                **run_info,
                "benchmark": bench,
                "auc_steps": int(args.auc_steps),
                "auc": auc,
                "auc_normalized": auc / args.auc_steps if args.auc_steps else np.nan,
                "auc_last_step": int(last_step) if last_step is not None else np.nan,
                "auc_normalized_observed": (auc / last_step) if last_step else np.nan,
                "auc_points": int(n_points),
                **curve_extra,
            }
            if fixed_step_col is not None:
                row[fixed_step_col] = (
                    float(fixed_step_map.get(bench, np.nan)) if fixed_step_map else np.nan
                )
                row["fixed_training_steps"] = int(fixed_step) if fixed_step is not None else np.nan
                row["requested_training_steps"] = int(args.fixed_steps)
            auc_rows.append(row)

        for row in curve_stats:
            curve_rows.append({**run_info, **row})

    if dev_rows:
        pd.DataFrame(dev_rows).to_csv(out_dir / "dev_selected_results.csv", index=False)
    if dev_summary_rows:
        pd.DataFrame(dev_summary_rows).to_csv(out_dir / "dev_selected_summary.csv", index=False)
    if fixed_rows:
        pd.DataFrame(fixed_rows).to_csv(out_dir / "fixed_step_results.csv", index=False)
    if auc_rows:
        pd.DataFrame(auc_rows).to_csv(out_dir / "auc_results.csv", index=False)
    if curve_rows:
        pd.DataFrame(curve_rows).to_csv(out_dir / "curve_stats.csv", index=False)

    if not auc_rows:
        print(f"No AUC rows produced; stopping early.")
        return

    auc_df = pd.DataFrame(auc_rows)

    flow_lookup = load_coverage_lookup(args.coverage_csv, allow_unsplit=args.allow_unsplit_coverage)
    variogram_lookup = (
        load_variogram_lookup(args.variogram_csv, allow_unsplit=args.allow_unsplit_coverage)
        if args.variogram_csv
        else {}
    )
    resnet_lookup = load_coverage_lookup(
        args.coverage_resnet_csv, allow_unsplit=args.allow_unsplit_coverage
    )
    dino_lookup = (
        load_coverage_lookup(args.coverage_dino_csv, allow_unsplit=args.allow_unsplit_coverage)
        if args.coverage_dino_csv
        else None
    )
    hof_lookup = (
        load_coverage_lookup(args.coverage_hof_csv, allow_unsplit=args.allow_unsplit_coverage)
        if args.coverage_hof_csv
        else None
    )
    flow_mmd_lookup = load_mmd_lookup(args.flow_mmd_csv, allow_unsplit=args.allow_unsplit_mmd)
    feature_mmd_lookup = load_mmd_lookup(
        args.feature_mmd_csv, allow_unsplit=args.allow_unsplit_mmd
    )
    dino_mmd_lookup = (
        load_mmd_lookup(args.dino_mmd_csv, allow_unsplit=args.allow_unsplit_mmd)
        if args.dino_mmd_csv
        else None
    )
    flow_stats_lookup = load_flow_stats(args.flow_stats_dir)

    feature_df, missing = build_auc_feature_table(
        auc_df,
        flow_lookup,
        resnet_lookup,
        hof_lookup,
        variogram_lookup,
        flow_mmd_lookup,
        feature_mmd_lookup,
        logit_coverage=args.logit_coverage,
        dino_lookup=dino_lookup,
        dino_mmd_lookup=dino_mmd_lookup,
        strict_dataset_match=args.strict_dataset_match,
        allow_unsplit_coverage=args.allow_unsplit_coverage,
        allow_unsplit_mmd=args.allow_unsplit_mmd,
        distance_radius_norm=args.distance_radius_norm,
        distance_ratio_transform=args.distance_ratio_transform,
        radius_transform=args.radius_transform,
        radius_eps=args.radius_eps,
        radius_floor=args.distance_radius_floor,
        rename_coverage=args.rename_coverage,
    )
    feature_df, filtered = filter_train_datasets_by_mode(
        feature_df, args.train_datasets_mode
    )
    if filtered:
        print(
            f"Filtered {filtered} rows using train_datasets_mode={args.train_datasets_mode}."
        )
    if args.flow_stats_dir:
        feature_df, density_missing = add_flow_density_features(
            feature_df,
            flow_stats_lookup,
            log_eps=args.flow_stats_log_eps,
            allow_unsplit=args.allow_unsplit_flow_stats,
            add_interactions=args.flow_density_interactions,
        )
        if density_missing:
            density_path = out_dir / "missing_flow_stats.txt"
            lines = ["Missing flow stats (sample):"]
            for key, count in sorted(density_missing.items(), key=lambda x: x[1], reverse=True)[:20]:
                lines.append(f"{key}: {count}")
            density_path.write_text("\n".join(lines))
    if args.relative_target_baseline and not feature_df.empty:
        feature_df, missing_baseline, baseline_name = add_relative_target(
            feature_df, args.relative_target_baseline, args.target
        )
        if target_default:
            args.target = "auc_delta"
        if missing_baseline:
            baseline_path = out_dir / "missing_baseline.txt"
            baseline_path.write_text(
                f"Missing baseline rows: {missing_baseline} (baseline={baseline_name})\n"
            )

    exclude_benchmarks = _parse_benchmark_list(args.exclude_benchmarks)
    if exclude_benchmarks and "benchmark" in feature_df.columns and not feature_df.empty:
        before = len(feature_df)
        feature_df = feature_df[
            ~feature_df["benchmark"].astype(str).str.lower().isin(exclude_benchmarks)
        ].copy()
        dropped = before - len(feature_df)
        if dropped:
            print(
                "Dropped rows for benchmarks: "
                + ", ".join(exclude_benchmarks)
                + f" ({dropped} rows)"
            )

    exclude_train_datasets = _parse_dataset_list(args.exclude_train_datasets)
    if exclude_train_datasets and "train_dataset" in feature_df.columns and not feature_df.empty:
        before = len(feature_df)
        feature_df = feature_df[
            ~feature_df["train_dataset"].astype(str).str.lower().isin(exclude_train_datasets)
        ].copy()
        dropped = before - len(feature_df)
        if dropped:
            print(
                "Dropped rows for train datasets: "
                + ", ".join(exclude_train_datasets)
                + f" ({dropped} rows)"
            )

    if args.rank_target and not feature_df.empty:
        feature_df = ensure_encoder_config(feature_df)
        if args.rank_target_with_model:
            feature_df = ensure_model_family(feature_df)
        rank_source = args.rank_target_source or args.target
        rank_col = args.rank_target_col or f"{rank_source}_rank"
        rank_groups = []
        if args.rank_target_group:
            rank_groups.append(args.rank_target_group)
        if args.rank_target_with_encoder:
            rank_groups.append("encoder_config")
        if args.rank_target_with_model:
            rank_groups.append("model_family")
        feature_df = add_rank_target(feature_df, rank_source, rank_groups, rank_col)
        if args.target == rank_source:
            args.target = rank_col
        if args.prediction_target == rank_source:
            args.prediction_target = rank_col
    
    # Create composite model_family_encoder column for proper demeaning
    # This must happen before any target demeaning operations
    if not feature_df.empty:
        feature_df = create_model_family_encoder_column(feature_df)
        if args.ranking_group == "train_dataset_encoder":
            feature_df = ensure_train_dataset_encoder_column(feature_df)
        elif args.ranking_group == "train_dataset_model_family_encoder":
            feature_df = ensure_train_dataset_model_family_encoder_column(feature_df)

    exclude_encoder_configs = _parse_encoder_config_list(args.exclude_encoder_configs)
    if exclude_encoder_configs:
        feature_df, dropped = filter_encoder_configs(feature_df, exclude_encoder_configs)
        if dropped:
            print(
                "Dropped rows for encoder configs: "
                + ", ".join(exclude_encoder_configs)
                + f" ({dropped} rows)"
            )
    exclude_model_families = _parse_model_family_list(args.exclude_model_families)
    if exclude_model_families:
        feature_df, dropped = filter_model_families(feature_df, exclude_model_families)
        if dropped:
            print(
                "Dropped rows for model families: "
                + ", ".join(exclude_model_families)
                + f" ({dropped} rows)"
            )

    if not feature_df.empty:
        write_distance_diagnostics(feature_df, out_dir / "distance_diagnostics.txt")

    if missing:
        missing_path = out_dir / "missing_coverage.txt"
        lines = ["Missing coverage/MMD lookups (sample):"]
        for key, count in sorted(missing.items(), key=lambda x: x[1], reverse=True)[:20]:
            lines.append(f"{key}: {count}")
        missing_path.write_text("\n".join(lines))

    if args.predictors:
        predictors = [p.strip() for p in args.predictors.split(",") if p.strip()]
    else:
        if args.rename_coverage:
            if args.logit_coverage:
                predictors = [
                    "flow_train_to_eval_over_eval_recall_logit",
                    "flow_eval_to_train_over_train_precision_logit",
                    "resnet_train_to_eval_over_eval_recall_logit",
                    "resnet_eval_to_train_over_train_precision_logit",
                    "flow_mmd",
                    "feature_mmd",
                ]
            else:
                predictors = [
                    "flow_train_to_eval_over_eval_recall",
                    "flow_eval_to_train_over_train_precision",
                    "resnet_train_to_eval_over_eval_recall",
                    "resnet_eval_to_train_over_train_precision",
                    "flow_mmd",
                    "feature_mmd",
                ]
        else:
            if args.logit_coverage:
                predictors = [
                    "flow_train_to_eval_coverage_logit",
                    "flow_eval_to_train_coverage_logit",
                    "resnet_train_to_eval_coverage_logit",
                    "resnet_eval_to_train_coverage_logit",
                    "flow_mmd",
                    "feature_mmd",
                ]
            else:
                predictors = [
                    "flow_train_to_eval_coverage",
                    "flow_eval_to_train_coverage",
                    "resnet_train_to_eval_coverage",
                    "resnet_eval_to_train_coverage",
                    "flow_mmd",
                    "feature_mmd",
                ]

    if args.use_flow_eps_predictors:
        eps_values = parse_eps_values(args.flow_eps_values)
        for eps in eps_values:
            predictors.append(f"flow_train_to_eval_eps{eps}px")
            predictors.append(f"flow_eval_to_train_eps{eps}px")
    if args.use_flow_eps_weighted_predictors:
        eps_values = parse_eps_values(args.flow_eps_values)
        for eps in eps_values:
            predictors.append(f"flow_train_to_eval_eps{eps}px_weighted")
            predictors.append(f"flow_eval_to_train_eps{eps}px_weighted")

    if args.use_flow_density_predictors:
        density_cols = [
            "log_n_samples_eval",
            "log_avg_flows_eval",
            "log_n_samples_train",
            "log_avg_flows_train",
        ]
        for col in density_cols:
            if col in feature_df.columns:
                predictors.append(col)

    if args.flow_density_interactions:
        base_cov = [p for p in predictors if _is_coverage_column(p)]
        for col in base_cov:
            interaction = f"{col}_x_log_avg_flows_eval"
            if interaction in feature_df.columns:
                predictors.append(interaction)

    if args.include_kl:
        predictors = _extend_predictors_with_kl(predictors, feature_df)

    if args.spair_indicator_interactions:
        feature_df, predictors, interaction_cols, indicator_col = add_spair_indicator_interactions(
            feature_df, predictors
        )
        if indicator_col:
            print(
                f"Added spair indicator interactions: {indicator_col} "
                f"({len(interaction_cols)} interactions)"
            )
    if args.custom_interactions:
        custom_specs = _parse_custom_interaction_specs(args.custom_interactions)
        spec_by_col = {f"{left}_x_{right}": (left, right, scale) for left, right, scale in custom_specs}
        feature_df, custom_cols, custom_skipped = add_custom_interaction_features(
            feature_df, args.custom_interactions
        )
        added_custom = []
        gated_custom = []
        if custom_cols:
            for col in custom_cols:
                spec = spec_by_col.get(col)
                # Keep custom interactions only when anchored to a selected
                # base predictor (left term in col1*col2).
                if spec is not None and spec[0] not in predictors:
                    gated_custom.append(col)
                    continue
                if col not in predictors:
                    predictors.append(col)
                added_custom.append(col)
            if added_custom:
                print(
                    "Added custom interactions: "
                    + ", ".join(added_custom)
                )
            if gated_custom:
                print(
                    "Skipped custom interactions (base predictor not selected): "
                    + ", ".join(gated_custom)
                )
        if custom_skipped:
            skipped_tokens = [f"{l}*{r}" for l, r in custom_skipped]
            print(
                "Warning: skipped custom interactions (missing columns): "
                + ", ".join(skipped_tokens)
            )

    if args.target not in feature_df.columns:
        print(f"Target '{args.target}' not found in auc_with_features table.")
        return

    if args.flow_eps_rings:
        eps_values = parse_eps_values(args.flow_eps_values)
        feature_df = apply_flow_eps_ring_features(feature_df, eps_values, weighted=False)
        feature_df = apply_flow_eps_ring_features(feature_df, eps_values, weighted=True)

    missing_predictors = [p for p in predictors if p not in feature_df.columns]
    if missing_predictors:
        print(
            "Warning: predictors missing from auc_with_features table: "
            + ", ".join(missing_predictors)
        )
        predictors = [p for p in predictors if p in feature_df.columns]
    if not predictors:
        print("No valid predictors found in auc_with_features table.")
        return
    all_nan = [p for p in predictors if feature_df[p].isna().all()]
    if all_nan:
        print(
            "Warning: predictors contain only NaNs and will be dropped: "
            + ", ".join(all_nan)
        )
        predictors = [p for p in predictors if p not in all_nan]
    if not predictors:
        print("No valid predictors found after dropping NaN-only columns.")
        return
    constant = [p for p in predictors if feature_df[p].nunique(dropna=True) < 2]
    if constant:
        print(
            "Warning: predictors are constant and will be dropped: "
            + ", ".join(constant)
        )
        predictors = [p for p in predictors if p not in constant]
    if not predictors:
        print("No valid predictors found after dropping constant columns.")
        return
    predictors, redundant = _drop_algebraic_redundancies(predictors)
    if redundant:
        print(
            "Warning: predictors are algebraically redundant and were dropped: "
            + ", ".join(redundant)
        )
    if not predictors:
        print("No valid predictors found after dropping redundant columns.")
        return

    write_predictor_colinearity(
        feature_df,
        predictors,
        out_dir / "predictor_colinearity_triangle.csv",
    )

    predictor_norm = args.benchmark_normalize_predictors
    if predictor_norm == "auto":
        predictor_norm = "none"
    target_norm = args.benchmark_normalize_target

    report_df = feature_df
    cv_df = feature_df
    apply_report_norm = args.benchmark_normalize_scope in ("all", "report_only")
    apply_cv_norm = args.benchmark_normalize_scope == "all"

    if not feature_df.empty:
        if apply_report_norm or apply_cv_norm:
            feature_df.to_csv(out_dir / "auc_with_features_raw.csv", index=False)
        if apply_report_norm:
            report_df = normalize_by_group(report_df, predictors, "benchmark", predictor_norm)
        if apply_cv_norm:
            cv_df = normalize_by_group(cv_df, predictors, "benchmark", predictor_norm)
        if apply_report_norm or apply_cv_norm:
            meta_path = out_dir / "analysis_normalization.txt"
            meta_path.write_text(
                "benchmark_normalize_predictors: "
                + str(predictor_norm)
                + "\nbenchmark_normalize_target: "
                + str(target_norm)
                + "\nbenchmark_normalize_scope: "
                + str(args.benchmark_normalize_scope)
                + "\ncv_normalize_predictors_by_benchmark: "
                + str(args.cv_normalize_predictors_by_benchmark)
                + "\ncv_normalize_predictors_by_encoder: "
                + str(args.cv_normalize_predictors_by_encoder)
                + "\n"
            )
        report_df.to_csv(out_dir / "auc_with_features.csv", index=False)

    def run_target_mode(mode_target, mode_out_dir):
        if mode_target not in feature_df.columns:
            print(f"Skipping {mode_out_dir.name}: target '{mode_target}' not in table.")
            return
        mode_out_dir.mkdir(parents=True, exist_ok=True)
        mode_args = argparse.Namespace(**vars(args))
        mode_args.target = mode_target
        mode_args.prediction_target = mode_target
        base_predictors = predictors
        mode_report_df = report_df.copy()
        mode_cv_df = cv_df.copy()
        if target_norm != "none":
            if apply_report_norm:
                mode_report_df = normalize_by_group(
                    mode_report_df, [mode_target], "benchmark", target_norm
                )
            if apply_cv_norm:
                mode_cv_df = normalize_by_group(
                    mode_cv_df, [mode_target], "benchmark", target_norm
                )
        if mode_args.rank_target:
            rank_source = mode_target
            rank_col = mode_args.rank_target_col or f"{rank_source}_rank"
            rank_groups = []
            if mode_args.rank_target_group:
                rank_groups.append(mode_args.rank_target_group)
            if mode_args.rank_target_with_encoder:
                mode_report_df = ensure_encoder_config(mode_report_df)
                mode_cv_df = ensure_encoder_config(mode_cv_df)
                rank_groups.append("encoder_config")
            if mode_args.rank_target_with_model:
                mode_report_df = ensure_model_family(mode_report_df)
                mode_cv_df = ensure_model_family(mode_cv_df)
                rank_groups.append("model_family")
            mode_report_df = add_rank_target(mode_report_df, rank_source, rank_groups, rank_col)
            mode_cv_df = add_rank_target(mode_cv_df, rank_source, rank_groups, rank_col)
            mode_args.target = rank_col
            mode_args.prediction_target = rank_col
        pooled_report_df, pooled_cv_df, pooled_predictors = prepare_encoder_pooled_frames(
            mode_report_df.copy(),
            mode_cv_df.copy(),
            base_predictors,
            mode_args.encoder_interaction_baseline,
            mode_args.encoder_interactions,
            mode_args.encoder_main_effects,
            mode_args.cv_normalize_predictors_by_encoder,
        )
        pooled_report_df, pooled_cv_df, pooled_predictors = prepare_model_family_pooled_frames(
            pooled_report_df,
            pooled_cv_df,
            pooled_predictors,
            base_predictors,
            mode_args.model_family_interaction_baseline,
            mode_args.model_family_interactions,
            mode_args.model_family_main_effects,
        )
        pooled_report_df.to_csv(mode_out_dir / "auc_with_features.csv", index=False)
        run_analysis_bundle(
            pooled_report_df,
            mode_out_dir,
            pooled_predictors,
            mode_args,
            cv_df=pooled_cv_df,
        )
        _write_run_metadata(mode_out_dir, pooled_predictors, mode_args)
        if mode_args.run_summary:
            run_summary_report(mode_out_dir, pooled_predictors, mode_args)
        if mode_args.per_encoder:
            per_args = argparse.Namespace(**vars(mode_args))
            per_args.encoder_interactions = False
            per_args.encoder_main_effects = True
            per_args.cv_normalize_predictors_by_encoder = "none"
            per_args.cv_demean_target_by_encoder = False
            per_dir = mode_out_dir / "by_encoder"
            per_dir.mkdir(parents=True, exist_ok=True)
            for tag, group, cv_group in _iter_per_encoder_groups(
                mode_report_df, mode_cv_df
            ):
                group_dir = per_dir / tag
                group_dir.mkdir(parents=True, exist_ok=True)
                group.to_csv(group_dir / "auc_with_features.csv", index=False)
                run_analysis_bundle(
                    group,
                    group_dir,
                    predictors,
                    per_args,
                    cv_df=cv_group,
                )
                _write_run_metadata(group_dir, predictors, per_args)
                if mode_args.run_summary:
                    run_summary_report(group_dir, predictors, per_args)

    if args.dual_target:
        mode_dirs = [m.strip() for m in args.dual_target_dirs.split(",") if m.strip()]
        if len(mode_dirs) != 2:
            raise ValueError("--dual-target-dirs must provide two subdirectory names.")
        modes = [
            (mode_dirs[0], "auc_delta"),
            (mode_dirs[1], "auc_normalized"),
        ]
        for mode_dir, mode_target in modes:
            run_target_mode(mode_target, out_dir / mode_dir)
    else:
        if target_norm != "none":
            if apply_report_norm:
                report_df = normalize_by_group(
                    report_df, [args.target], "benchmark", target_norm
                )
            if apply_cv_norm:
                cv_df = normalize_by_group(
                    cv_df, [args.target], "benchmark", target_norm
                )
        base_predictors = predictors
        pooled_report_df, pooled_cv_df, pooled_predictors = prepare_encoder_pooled_frames(
            report_df.copy(),
            cv_df.copy(),
            base_predictors,
            args.encoder_interaction_baseline,
            args.encoder_interactions,
            args.encoder_main_effects,
            args.cv_normalize_predictors_by_encoder,
        )
        pooled_report_df, pooled_cv_df, pooled_predictors = prepare_model_family_pooled_frames(
            pooled_report_df,
            pooled_cv_df,
            pooled_predictors,
            base_predictors,
            args.model_family_interaction_baseline,
            args.model_family_interactions,
            args.model_family_main_effects,
        )
        pooled_report_df.to_csv(out_dir / "auc_with_features.csv", index=False)
        run_analysis_bundle(
            pooled_report_df,
            out_dir,
            pooled_predictors,
            args,
            cv_df=pooled_cv_df,
        )
        _write_run_metadata(out_dir, pooled_predictors, args)
        if args.run_summary:
            run_summary_report(out_dir, pooled_predictors, args)

    if args.additional_targets:
        targets = [t.strip() for t in args.additional_targets.split(",") if t.strip()]
        dir_names = []
        if args.additional_target_dirs:
            dir_names = [d.strip() for d in args.additional_target_dirs.split(",") if d.strip()]
            if len(dir_names) != len(targets):
                raise ValueError("--additional-target-dirs must match --additional-targets length.")
        for idx, target in enumerate(targets):
            if target == args.target:
                continue
            mode_dir = dir_names[idx] if dir_names else target
            run_target_mode(target, out_dir / mode_dir)

    if args.per_encoder:
        per_args = argparse.Namespace(**vars(args))
        per_args.encoder_interactions = False
        per_args.encoder_main_effects = True
        per_args.cv_normalize_predictors_by_encoder = "none"
        per_args.cv_demean_target_by_encoder = False
        per_dir = out_dir / "by_encoder"
        per_dir.mkdir(parents=True, exist_ok=True)
        for tag, group, cv_group in _iter_per_encoder_groups(report_df, cv_df):
            group_dir = per_dir / tag
            group_dir.mkdir(parents=True, exist_ok=True)
            group.to_csv(group_dir / "auc_with_features.csv", index=False)
            run_analysis_bundle(group, group_dir, predictors, per_args, cv_df=cv_group)
            if args.run_summary:
                run_summary_report(group_dir, predictors, per_args)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
