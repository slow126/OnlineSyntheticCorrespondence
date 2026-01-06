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
  - regression_summary.txt: OLS/mixedlm regression summary

Optional (mode=all):
  - dev_selected_results.csv, dev_selected_summary.csv
  - fixed_step_results.csv
"""

import argparse
import math
import csv
import re
import subprocess
import sys
from collections import defaultdict
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


def normalize_dataset_name(name):
    if name is None:
        return None
    name = str(name).strip().lower()
    return name.replace('+', '_')


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
    for part in snapshot_dir.parts:
        if part == "snapshots":
            return MODEL_FAMILY_DEFAULT
        if part.startswith("snapshots_"):
            suffix = part.split("snapshots_", 1)[1].strip().lower()
            if not suffix:
                return MODEL_FAMILY_DEFAULT
            return MODEL_FAMILY_ALIASES.get(suffix, suffix)
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
            for row in reader:
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

                if not allow_unsplit and (not train_split or not eval_split):
                    continue

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
                    }
    except Exception as exc:
        print(f"Warning: could not read coverage CSV {csv_path}: {exc}")

    return coverage_lookup


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
    prefixes = ("flow", "resnet", "dino")
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
    prefixes = ("flow", "resnet", "dino")
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
            if radius_train is not None and eval_col in df.columns:
                df[f"{eval_col}_over_radius_train"] = df[eval_col].astype(float) / radius_train
            if train_col in df.columns and eval_col in df.columns:
                denom = df[train_col].astype(float) + float(eps)
                df[f"{prefix}_{stat}_dist_asymmetry"] = (
                    df[eval_col].astype(float) + float(eps)
                ) / denom
    return df


def transform_radius_features(df, mode, eps):
    if mode == "keep":
        return df
    df = df.copy()
    prefixes = ("flow", "resnet", "dino")
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


def write_distance_diagnostics(df, out_path):
    lines = []
    prefixes = ("flow", "resnet", "dino")
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
            lines.append(
                f"  max_abs(mean-median)={np.nanmax(np.abs(mean_vals - median_vals)):.6f}"
            )
            lines.append(
                f"  max_abs(mean-p90)={np.nanmax(np.abs(mean_vals - p90_vals)):.6f}"
            )
            lines.append(
                f"  max_abs(median-p90)={np.nanmax(np.abs(median_vals - p90_vals)):.6f}"
            )
            lines.append(f"  corr(mean,median)={_safe_corr(mean_vals, median_vals):.6f}")
            lines.append(f"  corr(mean,p90)={_safe_corr(mean_vals, p90_vals):.6f}")
            lines.append(f"  corr(median,p90)={_safe_corr(median_vals, p90_vals):.6f}")
    if not lines:
        lines.append("No distance diagnostics available.")
    out_path.write_text("\n".join(lines))


def build_auc_feature_table(
    auc_df,
    flow_lookup,
    resnet_lookup,
    flow_mmd_lookup,
    feature_mmd_lookup,
    logit_coverage=False,
    dino_lookup=None,
    dino_mmd_lookup=None,
    strict_dataset_match=False,
    allow_unsplit_coverage=True,
    allow_unsplit_mmd=True,
    distance_radius_norm="none",
    radius_transform="keep",
    radius_eps=1e-6,
    radius_floor=0.0,
):
    rows = []
    missing = defaultdict(int)
    known_datasets = gather_known_datasets(
        flow_lookup, resnet_lookup, flow_mmd_lookup, feature_mmd_lookup, dino_lookup, dino_mmd_lookup
    )

    for row in auc_df.to_dict(orient="records"):
        train_dataset = normalize_dataset_name(row.get("train_dataset"))
        benchmark = normalize_dataset_name(row.get("benchmark"))
        if not train_dataset or not benchmark:
            continue

        flow_metrics = None
        resnet_metrics = None
        dino_metrics = None
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
            if dino_lookup:
                dino_metrics, _ = lookup_pair(
                    dino_lookup, candidate, benchmark, allow_unsplit=allow_unsplit_coverage
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
            if flow_metrics is not None or resnet_metrics is not None:
                resolved_train = candidate
                break

        if flow_metrics is None:
            missing[("flow", train_dataset, benchmark)] += 1
        if resnet_metrics is None:
            missing[("resnet", train_dataset, benchmark)] += 1
        if dino_lookup is not None and dino_metrics is None:
            missing[("dino", train_dataset, benchmark)] += 1
        if flow_mmd is None:
            missing[("flow_mmd", train_dataset, benchmark)] += 1
        if feature_mmd is None:
            missing[("feature_mmd", train_dataset, benchmark)] += 1
        if dino_mmd_lookup is not None and dino_mmd is None:
            missing[("dino_mmd", train_dataset, benchmark)] += 1

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
            "flow_mmd": flow_mmd,
            "feature_mmd": feature_mmd,
            "dino_mmd": dino_mmd,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    df = add_distance_ratio_features(df, radius_eps, radius_floor)
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
            ],
        )
    return df, missing


def pearson_corr(x, y):
    if len(x) < 2:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return np.nan
    return float(np.dot(x, y) / denom)


def spearman_corr(x, y):
    if len(x) < 2:
        return np.nan
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def fit_linear_model(
    train_df,
    predictors,
    target,
    standardize=True,
    model="ols",
    ridge_alpha=1.0,
    min_std=0.0,
):
    X = train_df[predictors].to_numpy(dtype=float)
    y = train_df[target].to_numpy(dtype=float)
    mean = np.zeros(X.shape[1])
    std = np.ones(X.shape[1])
    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        if min_std > 0:
            std = np.where(std < min_std, float(min_std), std)
        std[std == 0] = 1.0
        X = (X - mean) / std
    X = np.column_stack([np.ones(len(X)), X])
    if model == "ridge":
        alpha = float(ridge_alpha)
        penalty = np.eye(X.shape[1])
        penalty[0, 0] = 0.0
        coef = np.linalg.solve(X.T @ X + alpha * penalty, X.T @ y)
    else:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef, mean, std


def fit_pairwise_rank_model(
    train_df,
    predictors,
    target,
    group_col,
    option_col,
    standardize=True,
    ridge_alpha=1.0,
    min_std=0.0,
    max_iter=200,
    lr=0.1,
):
    agg_cols = [group_col, option_col] if option_col else [group_col]
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
    if group_col in grouped.columns:
        for _, sub in grouped.groupby(group_col):
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
    else:
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if y[i] == y[j]:
                    continue
                label = 1.0 if y[i] > y[j] else -1.0
                diffs.append(X[i] - X[j])
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
        grad = -(labels[:, None] * diffs) * (1.0 / (1.0 + np.exp(margins)))[:, None]
        grad = grad.mean(axis=0)
        if reg > 0:
            grad = grad + reg * w
        w = w - float(lr) * grad

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
    model="ols",
    ridge_alpha=1.0,
    permute_target=False,
    permute_group_col=None,
    permute_seed=0,
    pairwise_option_col=None,
    pairwise_group_col="benchmark",
):
    results = []
    pred_rows = []

    groups = sorted(df[group_col].dropna().unique())
    for idx, group in enumerate(groups):
        train_df = df[df[group_col] != group]
        test_df = df[df[group_col] == group]
        train_df = filter_complete_rows(train_df, predictors, target)
        test_df = filter_complete_rows(test_df, predictors, target)

        if train_df.empty or test_df.empty:
            continue
        if len(train_df) <= len(predictors):
            continue

        y_true = test_df[target].to_numpy(dtype=float)
        target_offsets = None
        if target_group_demean and target_group_col:
            train_df, test_df, target_offsets = demean_target_by_group(
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

        if model == "pairwise_rank":
            coef, mean, std = fit_pairwise_rank_model(
                train_df,
                predictors_fold,
                target,
                pairwise_group_col,
                pairwise_option_col,
                standardize=standardize,
                ridge_alpha=ridge_alpha,
                min_std=min_predictor_std,
            )
            y_pred = predict_pairwise_rank(test_df, predictors_fold, coef, mean, std, standardize)
        else:
            coef, mean, std = fit_linear_model(
                train_df,
                predictors_fold,
                target,
                standardize=standardize,
                model=model,
                ridge_alpha=ridge_alpha,
                min_std=min_predictor_std,
            )
            y_pred = predict_linear_model(test_df, predictors_fold, coef, mean, std, standardize)
        if standardize:
            X_test = (test_df[predictors_fold].to_numpy(dtype=float) - mean) / std
            max_abs_z = float(np.nanmax(np.abs(X_test))) if X_test.size else np.nan
        else:
            max_abs_z = np.nan
        if target_offsets is not None:
            y_pred = y_pred + target_offsets
        if prediction_clip and model != "pairwise_rank":
            clip_min = prediction_clip_min
            clip_max = prediction_clip_max
            if clip_min is None:
                clip_min = float(np.nanmin(train_df[target].to_numpy(dtype=float)))
            if clip_max is None:
                clip_max = float(np.nanmax(train_df[target].to_numpy(dtype=float)))
            y_pred = np.clip(y_pred, clip_min, clip_max)

        pred_nan = int(np.isnan(y_pred).sum())
        pred_inf = int(np.isinf(y_pred).sum())
        target_nan = int(np.isnan(y_true).sum())
        target_inf = int(np.isinf(y_true).sum())
        pred_finite = y_pred[np.isfinite(y_pred)]
        target_finite = y_true[np.isfinite(y_true)]
        pred_min = float(pred_finite.min()) if pred_finite.size else np.nan
        pred_max = float(pred_finite.max()) if pred_finite.size else np.nan
        target_min = float(target_finite.min()) if target_finite.size else np.nan
        target_max = float(target_finite.max()) if target_finite.size else np.nan
        pred_std = float(np.std(pred_finite)) if pred_finite.size else np.nan
        target_std = float(np.std(target_finite)) if target_finite.size else np.nan

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        pearson = pearson_corr(y_true, y_pred)
        spearman = spearman_corr(y_true, y_pred)

        results.append({
            group_col: group,
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
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
        overall = {
            group_col: "__overall__",
            "n_train": int(len(df)),
            "n_test": int(len(pred_df)),
            "mae": float(np.mean(np.abs(pred_df["target"] - pred_df["prediction"]))),
            "rmse": float(np.sqrt(np.mean((pred_df["target"] - pred_df["prediction"]) ** 2))),
            "pearson": pearson_corr(pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()),
            "spearman": spearman_corr(pred_df["target"].to_numpy(), pred_df["prediction"].to_numpy()),
            "target_min": float(pred_df["target"].min()) if not pred_df.empty else np.nan,
            "target_max": float(pred_df["target"].max()) if not pred_df.empty else np.nan,
            "pred_min": float(pred_df["prediction"].min()) if not pred_df.empty else np.nan,
            "pred_max": float(pred_df["prediction"].max()) if not pred_df.empty else np.nan,
            "target_std": float(pred_df["target"].std(ddof=0)) if not pred_df.empty else np.nan,
            "pred_std": float(pred_df["prediction"].std(ddof=0)) if not pred_df.empty else np.nan,
            "max_abs_zscore_feature": np.nan,
            "pred_nan": int(np.isnan(pred_df["prediction"]).sum()) if not pred_df.empty else 0,
            "pred_inf": int(np.isinf(pred_df["prediction"]).sum()) if not pred_df.empty else 0,
            "target_nan": int(np.isnan(pred_df["target"]).sum()) if not pred_df.empty else 0,
            "target_inf": int(np.isinf(pred_df["target"]).sum()) if not pred_df.empty else 0,
            "dropped_predictor_count": np.nan,
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

    groups = sorted(df[holdout_col].dropna().unique())
    for idx, group in enumerate(groups):
        train_df = df[df[holdout_col] != group]
        test_df = df[df[holdout_col] == group]
        train_df = filter_complete_rows(train_df, predictors, target)
        test_df = filter_complete_rows(test_df, predictors, target)

        if train_df.empty or test_df.empty:
            continue
        if len(train_df) <= len(predictors):
            continue
        if train_df[group_col].nunique() < 2:
            continue

        if target_group_demean and target_group_col:
            train_df, test_df, _ = demean_target_by_group(
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

        train_df, test_df, pred_cols, mapping = _standardize_predictors(
            train_df, test_df, predictors_fold, standardize, min_std=min_predictor_std
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

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
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
        overall = {
            holdout_col: "__overall__",
            "n_train": int(len(df)),
            "n_test": int(len(pred_df)),
            "mae": float(np.mean(np.abs(pred_df["target"] - pred_df["prediction"]))),
            "rmse": float(np.sqrt(np.mean((pred_df["target"] - pred_df["prediction"]) ** 2))),
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


def compute_ranking_dataframe(
    pred_df,
    target_col,
    option_col,
    benchmark_col="benchmark",
    topk_frac=None,
    topk_min=1,
):
    rows = []
    if pred_df.empty:
        return pd.DataFrame()

    required_cols = [benchmark_col, option_col, "prediction", target_col]
    df = pred_df.dropna(subset=required_cols).copy()
    if df.empty:
        return pd.DataFrame()

    for benchmark, sub in df.groupby(benchmark_col):
        grouped = sub.groupby(option_col).agg(
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
            benchmark_col: benchmark,
            "n_options": n_options,
            "top1": top1,
            "top3": top3,
            "topk": topk,
            "topk_k": topk_k,
            "topk_frac": topk_frac_out,
            "regret": regret,
            "spearman": spearman,
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
    df_out = pd.concat([df_out, pd.DataFrame([overall])], ignore_index=True)
    return df_out


def compute_ranking_summary(
    pred_df,
    target_col,
    option_col,
    output_path,
    benchmark_col="benchmark",
    topk_frac=None,
    topk_min=1,
):
    df_out = compute_ranking_dataframe(
        pred_df,
        target_col,
        option_col,
        benchmark_col,
        topk_frac=topk_frac,
        topk_min=topk_min,
    )
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
        return df_out.to_dict(orient="records")
    return []


def compute_rank_detail_rows(pred_df, target_col, option_col, benchmark_col="benchmark"):
    rows = []
    if pred_df.empty:
        return pd.DataFrame()

    required_cols = [benchmark_col, option_col, "prediction", target_col]
    df = pred_df.dropna(subset=required_cols).copy()
    if df.empty:
        return pd.DataFrame()

    for benchmark, sub in df.groupby(benchmark_col):
        grouped = sub.groupby(option_col).agg(
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
                benchmark_col: benchmark,
                option_col: option,
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
):
    df_out = compute_rank_detail_rows(pred_df, target_col, option_col, benchmark_col)
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
    return df_out


def write_direction_audit(pred_df, target_col, option_col, output_path, benchmark_col="benchmark"):
    if pred_df.empty:
        output_path.write_text("No predictions available for direction audit.")
        return

    if target_col not in pred_df.columns and "target" in pred_df.columns:
        target_col = "target"

    for benchmark in sorted(pred_df[benchmark_col].dropna().unique()):
        sub = pred_df[pred_df[benchmark_col] == benchmark]
        grouped = sub.groupby(option_col).agg(
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
    topk_frac=None,
    topk_min=1,
):
    rows = []
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=[benchmark_col, option_col, target_col])
    if df.empty:
        return pd.DataFrame()

    for benchmark, sub in df.groupby(benchmark_col):
        grouped = sub.groupby(option_col).agg(true_mean=(target_col, "mean"))
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
            benchmark_col: benchmark,
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
    df_out = pd.concat([df_out, pd.DataFrame([overall])], ignore_index=True)
    return df_out


def compute_baseline_rankings(
    df,
    target_col,
    option_col,
    output_path,
    selectors,
    benchmark_col="benchmark",
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
            sub = df[[benchmark_col, option_col, target_col, col]].dropna()
            if sub.empty:
                continue
            pred_df = sub.rename(columns={col: "prediction"}).copy()
            if selector.get("direction", 1) < 0:
                pred_df["prediction"] = -pred_df["prediction"]
            ranking_df = compute_ranking_dataframe(
                pred_df,
                target_col,
                option_col,
                benchmark_col,
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


def _select_target_demean_group(args):
    if args.cv_demean_target_by_benchmark and args.cv_demean_target_by_encoder:
        print(
            "Warning: both cv_demean_target_by_benchmark and "
            "cv_demean_target_by_encoder are set; using benchmark demeaning."
        )
    if args.cv_demean_target_by_benchmark:
        return "benchmark"
    if args.cv_demean_target_by_encoder:
        return "encoder_config"
    return None


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
    if not add_interactions and encoder_norm_mode == "none":
        return report_df, cv_df, predictors

    report_df = ensure_encoder_config(report_df)
    cv_df = ensure_encoder_config(cv_df)
    if "encoder_config" not in report_df.columns:
        return report_df, cv_df, predictors

    if not add_interactions:
        return report_df, cv_df, predictors

    configs = _collect_encoder_configs(report_df["encoder_config"])
    if len(configs) < 2:
        print("Warning: encoder_interactions requested but insufficient encoder configs.")
        return report_df, cv_df, predictors

    baseline = (baseline or "").strip().upper()
    report_df, dummy_cols, interaction_cols = add_encoder_interactions(
        report_df, predictors, configs, baseline
    )
    cv_df, _, _ = add_encoder_interactions(cv_df, predictors, configs, baseline)
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
    for prefix in ("flow", "resnet", "dino"):
        cov = f"{prefix}_eval_to_train_coverage"
        outside = f"{prefix}_outside_mass"
        cov_logit = f"{cov}_logit"
        outside_logit = f"{outside}_logit"
        if cov in predictors and outside in predictors:
            redundant.append(outside)
        if cov_logit in predictors and outside_logit in predictors:
            redundant.append(outside_logit)
    if not redundant:
        return predictors, []
    filtered = [p for p in predictors if p not in redundant]
    return filtered, redundant


def _build_baseline_selectors(feature_df, use_logit=True):
    selectors = []

    def add_metric(name, col, direction=1):
        if col in feature_df.columns:
            selectors.append({
                "name": name,
                "type": "metric",
                "column": col,
                "direction": direction,
            })

    distance_cols = [
        "flow_train_to_eval_mean_dist",
        "flow_eval_to_train_mean_dist",
        "resnet_train_to_eval_mean_dist",
        "resnet_eval_to_train_mean_dist",
        "dino_train_to_eval_mean_dist",
        "dino_eval_to_train_mean_dist",
    ]
    has_distance = any(col in feature_df.columns for col in distance_cols)
    if has_distance:
        add_metric("flow_train_to_eval_mean_dist", "flow_train_to_eval_mean_dist", direction=-1)
        add_metric("flow_eval_to_train_mean_dist", "flow_eval_to_train_mean_dist", direction=-1)
        add_metric("resnet_train_to_eval_mean_dist", "resnet_train_to_eval_mean_dist", direction=-1)
        add_metric("resnet_eval_to_train_mean_dist", "resnet_eval_to_train_mean_dist", direction=-1)
        add_metric("dino_train_to_eval_mean_dist", "dino_train_to_eval_mean_dist", direction=-1)
        add_metric("dino_eval_to_train_mean_dist", "dino_eval_to_train_mean_dist", direction=-1)
    elif use_logit:
        add_metric(
            "flow_train_to_eval_coverage_logit",
            "flow_train_to_eval_coverage_logit",
            direction=1,
        )
        add_metric(
            "resnet_train_to_eval_coverage_logit",
            "resnet_train_to_eval_coverage_logit",
            direction=1,
        )
        add_metric(
            "dino_train_to_eval_coverage_logit",
            "dino_train_to_eval_coverage_logit",
            direction=1,
        )
    else:
        add_metric("flow_train_to_eval_coverage", "flow_train_to_eval_coverage", direction=1)
        add_metric(
            "resnet_train_to_eval_coverage",
            "resnet_train_to_eval_coverage",
            direction=1,
        )
        add_metric("dino_train_to_eval_coverage", "dino_train_to_eval_coverage", direction=1)

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

    within_path = out_dir / "within_benchmark_slopes.csv"
    compute_within_benchmark_slopes(feature_df, predictors, args.target, within_path)

    if not args.skip_regression:
        regression_path = out_dir / "regression_summary.txt"
        run_regression(
            feature_df,
            predictors,
            args.target,
            regression_path,
            linear_model=args.linear_model,
            ridge_alpha=args.ridge_alpha,
            use_mixedlm=True,
        )

    if args.skip_prediction:
        return

    pred_target = args.prediction_target or args.target
    pred_model = args.prediction_model or args.linear_model
    cv_df = cv_df if cv_df is not None else feature_df
    exclude_benchmarks = _parse_benchmark_list(args.exclude_benchmarks)
    if exclude_benchmarks and "benchmark" in cv_df.columns:
        cv_df = cv_df[
            ~cv_df["benchmark"].astype(str).str.lower().isin(exclude_benchmarks)
        ].copy()
        if cv_df.empty:
            print(
                "Warning: excluded all rows for prediction; skipping LOBO/LOTO runs."
            )
            return
    target_demean_group = _select_target_demean_group(args)
    target_group_demean = target_demean_group is not None

    lobo_summary, lobo_preds = run_group_cv(
        cv_df,
        "benchmark",
        predictors,
        pred_target,
        standardize=args.standardize,
        center_by_group=args.center_predictors_by_benchmark,
        center_group_col="benchmark",
        group_norm_mode=args.cv_normalize_predictors_by_benchmark,
        within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
        encoder_group_norm_mode=args.cv_normalize_predictors_by_encoder,
        encoder_group_col="encoder_config",
        target_group_demean=target_group_demean,
        target_group_col=target_demean_group,
        min_predictor_std=args.cv_min_predictor_std,
        prediction_clip=args.prediction_clip,
        prediction_clip_min=args.prediction_clip_min,
        prediction_clip_max=args.prediction_clip_max,
        model=pred_model,
        ridge_alpha=args.ridge_alpha,
        pairwise_option_col=args.ranking_group,
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
            topk_frac=args.ranking_topk_frac,
            topk_min=args.ranking_topk_min,
        )
        write_rank_detail_rows(
            lobo_preds,
            pred_target,
            args.ranking_group,
            out_dir / "prediction_lobo_rank_detail.csv",
        )
        if args.sanity_direction_audit:
            write_direction_audit(
                lobo_preds,
                pred_target,
                args.ranking_group,
                out_dir / "prediction_lobo_direction_audit.txt",
            )

    baseline_selectors = _build_baseline_selectors(feature_df, use_logit=args.logit_coverage)
    compute_baseline_rankings(
        cv_df,
        pred_target,
        args.ranking_group,
        out_dir / "prediction_lobo_rank_baselines.csv",
        baseline_selectors,
        topk_frac=args.ranking_topk_frac,
        topk_min=args.ranking_topk_min,
    )

    if args.prediction_mixedlm and HAS_STATSMODELS:
        random_slopes = _select_random_slopes(args, predictors)
        lobo_mixed_summary, lobo_mixed_preds = run_group_cv_mixedlm(
            cv_df,
            holdout_col="benchmark",
            predictors=predictors,
            target=pred_target,
            random_group_col="benchmark",
            random_slopes=random_slopes,
            standardize=args.standardize,
            center_by_group=args.center_predictors_by_benchmark,
            center_group_col="benchmark",
            group_norm_mode=args.cv_normalize_predictors_by_benchmark,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=args.cv_normalize_predictors_by_encoder,
            encoder_group_col="encoder_config",
            target_group_demean=target_group_demean,
            target_group_col=target_demean_group,
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

    loto_summary, loto_preds = run_group_cv(
        loto_df,
        group_col,
        predictors,
        pred_target,
        standardize=args.standardize,
        center_by_group=args.center_predictors_by_benchmark,
        center_group_col="benchmark",
        group_norm_mode=args.cv_normalize_predictors_by_benchmark,
        within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
        encoder_group_norm_mode=args.cv_normalize_predictors_by_encoder,
        encoder_group_col="encoder_config",
        target_group_demean=target_group_demean,
        target_group_col=target_demean_group,
        min_predictor_std=args.cv_min_predictor_std,
        prediction_clip=args.prediction_clip,
        prediction_clip_min=args.prediction_clip_min,
        prediction_clip_max=args.prediction_clip_max,
        model=pred_model,
        ridge_alpha=args.ridge_alpha,
        pairwise_option_col=args.ranking_group,
    )
    if not loto_summary.empty:
        loto_summary.to_csv(out_dir / "prediction_loto_summary.csv", index=False)
    if not loto_preds.empty:
        loto_preds.to_csv(out_dir / "prediction_loto_rows.csv", index=False)
        compute_ranking_summary(
            loto_preds,
            pred_target,
            args.ranking_group,
            out_dir / "prediction_loto_rank_summary.csv",
            topk_frac=args.ranking_topk_frac,
            topk_min=args.ranking_topk_min,
        )
        write_rank_detail_rows(
            loto_preds,
            pred_target,
            args.ranking_group,
            out_dir / "prediction_loto_rank_detail.csv",
        )

    if args.sanity_permutation:
        perm_group = args.sanity_permute_group or "benchmark"
        perm_seed = args.sanity_permute_seed

        perm_lobo_summary, perm_lobo_preds = run_group_cv(
            cv_df,
            "benchmark",
            predictors,
            pred_target,
            standardize=args.standardize,
            center_by_group=args.center_predictors_by_benchmark,
            center_group_col="benchmark",
            group_norm_mode=args.cv_normalize_predictors_by_benchmark,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=args.cv_normalize_predictors_by_encoder,
            encoder_group_col="encoder_config",
            target_group_demean=target_group_demean,
            target_group_col=target_demean_group,
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
                topk_frac=args.ranking_topk_frac,
                topk_min=args.ranking_topk_min,
            )

        perm_loto_summary, perm_loto_preds = run_group_cv(
            loto_df,
            group_col,
            predictors,
            pred_target,
            standardize=args.standardize,
            center_by_group=args.center_predictors_by_benchmark,
            center_group_col="benchmark",
            group_norm_mode=args.cv_normalize_predictors_by_benchmark,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=args.cv_normalize_predictors_by_encoder,
            encoder_group_col="encoder_config",
            target_group_demean=target_group_demean,
            target_group_col=target_demean_group,
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
        )
        if not perm_loto_summary.empty:
            perm_loto_summary.to_csv(
                out_dir / "prediction_loto_permutation_summary.csv", index=False
            )
        if not perm_loto_preds.empty:
            perm_loto_preds.to_csv(
                out_dir / "prediction_loto_permutation_rows.csv", index=False
            )

    if args.prediction_mixedlm and HAS_STATSMODELS:
        random_slopes = _select_random_slopes(args, predictors)
        loto_mixed_summary, loto_mixed_preds = run_group_cv_mixedlm(
            loto_df,
            holdout_col=group_col,
            predictors=predictors,
            target=pred_target,
            random_group_col="benchmark",
            random_slopes=random_slopes,
            standardize=args.standardize,
            center_by_group=args.center_predictors_by_benchmark,
            center_group_col="benchmark",
            group_norm_mode=args.cv_normalize_predictors_by_benchmark,
            within_benchmark_norm=args.cv_within_benchmark_predictor_norm,
            encoder_group_norm_mode=args.cv_normalize_predictors_by_encoder,
            encoder_group_col="encoder_config",
            target_group_demean=target_group_demean,
            target_group_col=target_demean_group,
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
        "--lobo-mixed-summary",
        str(out_dir / "prediction_lobo_mixed_summary.csv"),
        "--loto-mixed-summary",
        str(out_dir / "prediction_loto_mixed_summary.csv"),
        "--within-benchmark-slopes",
        str(out_dir / "within_benchmark_slopes.csv"),
        "--target",
        str(args.target),
        "--predictors",
        ",".join(predictors),
        "--linear-model",
        str(args.linear_model),
        "--ridge-alpha",
        str(args.ridge_alpha),
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
        "--coverage-csv",
        default="coverage_results.csv",
        help="Coverage CSV for flow/label metrics.",
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
    parser.add_argument(
        "--distance-radius-norm",
        choices=["none", "divide"],
        default="none",
        help="Normalize distance predictors by the corresponding train radius.",
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
        "--skip-prediction",
        action="store_true",
        help="Skip LOBO/LOTO prediction validation.",
    )
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
        help="Comma-separated benchmarks to exclude from LOBO/LOTO prediction runs.",
    )
    parser.add_argument(
        "--ranking-group",
        default="train_dataset",
        help="Column to rank options within each benchmark (default: train_dataset).",
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

            if args.fixed_steps is not None:
                fixed_step = args.fixed_steps
                if args.fixed_policy == "nearest":
                    fixed_step = find_nearest_step(df, args.fixed_steps)
                if fixed_step is None:
                    print(f"Skipping {snapshot_dir}: no steps for fixed selection")
                else:
                    fixed_df = df[df["training_steps"] == fixed_step].copy()
                    if fixed_df.empty:
                        print(f"Skipping {snapshot_dir}: fixed step {fixed_step} missing")
                    else:
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
            auc_rows.append(
                {
                    **run_info,
                    "benchmark": bench,
                    "auc_steps": int(args.auc_steps),
                    "auc": auc,
                    "auc_normalized": auc / args.auc_steps if args.auc_steps else np.nan,
                    "auc_points": int(n_points),
                    **curve_extra,
                }
            )

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
    resnet_lookup = load_coverage_lookup(
        args.coverage_resnet_csv, allow_unsplit=args.allow_unsplit_coverage
    )
    dino_lookup = (
        load_coverage_lookup(args.coverage_dino_csv, allow_unsplit=args.allow_unsplit_coverage)
        if args.coverage_dino_csv
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

    feature_df, missing = build_auc_feature_table(
        auc_df,
        flow_lookup,
        resnet_lookup,
        flow_mmd_lookup,
        feature_mmd_lookup,
        logit_coverage=args.logit_coverage,
        dino_lookup=dino_lookup,
        dino_mmd_lookup=dino_mmd_lookup,
        strict_dataset_match=args.strict_dataset_match,
        allow_unsplit_coverage=args.allow_unsplit_coverage,
        allow_unsplit_mmd=args.allow_unsplit_mmd,
        distance_radius_norm=args.distance_radius_norm,
        radius_transform=args.radius_transform,
        radius_eps=args.radius_eps,
        radius_floor=args.distance_radius_floor,
    )
    feature_df, filtered = filter_train_datasets_by_mode(
        feature_df, args.train_datasets_mode
    )
    if filtered:
        print(
            f"Filtered {filtered} rows using train_datasets_mode={args.train_datasets_mode}."
        )
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

    if args.target not in feature_df.columns:
        print(f"Target '{args.target}' not found in auc_with_features table.")
        return

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
            for (pretrained, freeze), group in mode_report_df.groupby(
                ["pretrained", "freeze"], dropna=False
            ):
                tag = f"pretrained{_format_bool(pretrained)}_freeze{_format_bool(freeze)}"
                group_dir = per_dir / tag
                group_dir.mkdir(parents=True, exist_ok=True)
                group.to_csv(group_dir / "auc_with_features.csv", index=False)
                cv_group = mode_cv_df[
                    (mode_cv_df["pretrained"] == pretrained)
                    & (mode_cv_df["freeze"] == freeze)
                ].copy()
                run_analysis_bundle(
                    group,
                    group_dir,
                    predictors,
                    per_args,
                    cv_df=cv_group,
                )
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
        for (pretrained, freeze), group in report_df.groupby(
            ["pretrained", "freeze"], dropna=False
        ):
            tag = f"pretrained{_format_bool(pretrained)}_freeze{_format_bool(freeze)}"
            group_dir = per_dir / tag
            group_dir.mkdir(parents=True, exist_ok=True)
            group.to_csv(group_dir / "auc_with_features.csv", index=False)
            cv_group = cv_df[
                (cv_df["pretrained"] == pretrained) & (cv_df["freeze"] == freeze)
            ].copy()
            run_analysis_bundle(group, group_dir, predictors, per_args, cv_df=cv_group)
            if args.run_summary:
                run_summary_report(group_dir, predictors, per_args)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
