#!/usr/bin/env python3
"""
One-stop analysis pipeline for leakage-free evaluation.

Outputs (under --output-dir):
  - auc_results.csv: per-run, per-benchmark AUC up to N steps
  - auc_with_features.csv: AUC table joined with coverage/MMD predictors
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


def load_coverage_lookup(csv_path):
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

                recall_val = row.get("recall", np.nan)
                try:
                    recall_val = float(recall_val) if recall_val not in (None, "") else np.nan
                except (ValueError, TypeError):
                    recall_val = np.nan

                precision_val = row.get("precision", np.nan)
                try:
                    precision_val = float(precision_val) if precision_val not in (None, "") else np.nan
                except (ValueError, TypeError):
                    precision_val = np.nan

                outside_val = row.get("outside", np.nan)
                try:
                    outside_val = float(outside_val) if outside_val not in (None, "") else np.nan
                except (ValueError, TypeError):
                    outside_val = np.nan

                coverage_lookup[(train_id, eval_id)] = {
                    "recall": recall_val,
                    "precision": precision_val,
                    "outside": outside_val,
                }
                coverage_lookup[(train_dataset, eval_dataset)] = {
                    "recall": recall_val,
                    "precision": precision_val,
                    "outside": outside_val,
                }
    except Exception as exc:
        print(f"Warning: could not read coverage CSV {csv_path}: {exc}")

    return coverage_lookup


def load_mmd_lookup(csv_path):
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
                if dataset1 == dataset2 and split1 == split2:
                    continue
                dataset1_id = f"{dataset1}_{split1}" if split1 else dataset1
                dataset2_id = f"{dataset2}_{split2}" if split2 else dataset2
                mmd_lookup[(dataset1_id, dataset2_id)] = mmd2
                mmd_lookup[(dataset2_id, dataset1_id)] = mmd2
            else:
                if dataset1 == dataset2:
                    continue
                mmd_lookup[(dataset1, dataset2)] = mmd2
                mmd_lookup[(dataset2, dataset1)] = mmd2

            mmd_lookup[(dataset1, dataset2)] = mmd2
            mmd_lookup[(dataset2, dataset1)] = mmd2
    except Exception as exc:
        print(f"Warning: could not read MMD CSV {csv_path}: {exc}")

    return mmd_lookup


def lookup_pair(lookup, train_dataset, benchmark, train_split="train"):
    candidates = [
        (f"{train_dataset}_{train_split}", f"{benchmark}_val"),
        (f"{train_dataset}_{train_split}", f"{benchmark}_test"),
        (f"{train_dataset}_{train_split}", benchmark),
        (train_dataset, benchmark),
        (train_dataset, f"{benchmark}_val"),
        (train_dataset, f"{benchmark}_test"),
    ]
    for key in candidates:
        if key in lookup:
            return lookup[key], key
    return None, candidates[-1]


def lookup_mmd(lookup, train_dataset, benchmark, train_split="train"):
    candidates = [
        (f"{train_dataset}_{train_split}", f"{benchmark}_val"),
        (f"{train_dataset}_{train_split}", f"{benchmark}_test"),
        (f"{train_dataset}_{train_split}", benchmark),
        (train_dataset, benchmark),
        (train_dataset, f"{benchmark}_val"),
        (train_dataset, f"{benchmark}_test"),
    ]
    for key in candidates:
        if key in lookup:
            return lookup[key], key
    return None, candidates[-1]


def resolve_train_dataset(train_dataset, known_datasets=None):
    if train_dataset is None:
        return []
    candidates = [train_dataset]
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


def build_auc_feature_table(
    auc_df,
    flow_lookup,
    resnet_lookup,
    flow_mmd_lookup,
    feature_mmd_lookup,
    logit_coverage=False,
    dino_lookup=None,
    dino_mmd_lookup=None,
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

        for candidate in resolve_train_dataset(train_dataset, known_datasets):
            flow_metrics, flow_key = lookup_pair(flow_lookup, candidate, benchmark)
            resnet_metrics, resnet_key = lookup_pair(resnet_lookup, candidate, benchmark)
            if dino_lookup:
                dino_metrics, _ = lookup_pair(dino_lookup, candidate, benchmark)
            flow_mmd, _ = lookup_mmd(flow_mmd_lookup, candidate, benchmark)
            feature_mmd, _ = lookup_mmd(feature_mmd_lookup, candidate, benchmark)
            if dino_mmd_lookup:
                dino_mmd, _ = lookup_mmd(dino_mmd_lookup, candidate, benchmark)
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
            "flow_recall": flow_metrics["recall"] if flow_metrics else np.nan,
            "flow_precision": flow_metrics["precision"] if flow_metrics else np.nan,
            "flow_outside": flow_metrics["outside"] if flow_metrics else np.nan,
            "resnet_recall": resnet_metrics["recall"] if resnet_metrics else np.nan,
            "resnet_precision": resnet_metrics["precision"] if resnet_metrics else np.nan,
            "resnet_outside": resnet_metrics["outside"] if resnet_metrics else np.nan,
            "dino_recall": dino_metrics["recall"] if dino_metrics else np.nan,
            "dino_precision": dino_metrics["precision"] if dino_metrics else np.nan,
            "dino_outside": dino_metrics["outside"] if dino_metrics else np.nan,
            "flow_mmd": flow_mmd,
            "feature_mmd": feature_mmd,
            "dino_mmd": dino_mmd,
        })
        rows.append(row)

    df = pd.DataFrame(rows)
    if logit_coverage:
        df = add_logit_columns(
            df,
            [
                "flow_recall",
                "flow_precision",
                "flow_outside",
                "resnet_recall",
                "resnet_precision",
                "resnet_outside",
                "dino_recall",
                "dino_precision",
                "dino_outside",
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


def fit_linear_model(train_df, predictors, target, standardize=True):
    X = train_df[predictors].to_numpy(dtype=float)
    y = train_df[target].to_numpy(dtype=float)
    mean = np.zeros(X.shape[1])
    std = np.ones(X.shape[1])
    if standardize:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        X = (X - mean) / std
    X = np.column_stack([np.ones(len(X)), X])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return coef, mean, std


def predict_linear_model(df, predictors, coef, mean, std, standardize=True):
    X = df[predictors].to_numpy(dtype=float)
    if standardize:
        X = (X - mean) / std
    X = np.column_stack([np.ones(len(X)), X])
    return X.dot(coef)


def filter_complete_rows(df, predictors, target):
    mask = df[predictors + [target]].notna().all(axis=1)
    return df[mask].copy()


def run_group_cv(
    df,
    group_col,
    predictors,
    target,
    standardize=True,
    center_by_group=False,
    center_group_col=None,
):
    results = []
    pred_rows = []

    groups = sorted(df[group_col].dropna().unique())
    for group in groups:
        train_df = df[df[group_col] != group]
        test_df = df[df[group_col] == group]
        train_df = filter_complete_rows(train_df, predictors, target)
        test_df = filter_complete_rows(test_df, predictors, target)

        if train_df.empty or test_df.empty:
            continue
        if len(train_df) <= len(predictors):
            continue

        if center_by_group and center_group_col:
            train_df, test_df = _center_predictors_by_group(
                train_df, test_df, predictors, center_group_col
            )

        coef, mean, std = fit_linear_model(train_df, predictors, target, standardize)
        y_pred = predict_linear_model(test_df, predictors, coef, mean, std, standardize)
        y_true = test_df[target].to_numpy(dtype=float)

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
        }
        summary_df = pd.concat([summary_df, pd.DataFrame([overall])], ignore_index=True)

    return summary_df, pred_df


def _standardize_predictors(train_df, test_df, predictors, standardize):
    if not standardize:
        mapping = {col: col for col in predictors}
        return train_df.copy(), test_df.copy(), predictors, mapping

    train_df = train_df.copy()
    test_df = test_df.copy()
    mean = train_df[predictors].mean()
    std = train_df[predictors].std(ddof=0).replace(0, 1.0)

    standardized_cols = []
    mapping = {}
    for col in predictors:
        z_col = f"{col}_z"
        train_df[z_col] = (train_df[col] - mean[col]) / std[col]
        test_df[z_col] = (test_df[col] - mean[col]) / std[col]
        standardized_cols.append(z_col)
        mapping[col] = z_col

    return train_df, test_df, standardized_cols, mapping


def _center_predictors_by_group(train_df, test_df, predictors, group_col):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if group_col not in train_df.columns or group_col not in test_df.columns:
        return train_df, test_df

    group_means = train_df.groupby(group_col)[predictors].mean()
    global_means = train_df[predictors].mean()

    for col in predictors:
        train_df[col] = train_df[col] - train_df[group_col].map(group_means[col])
        test_means = test_df[group_col].map(group_means[col]).fillna(global_means[col])
        test_df[col] = test_df[col] - test_means

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
):
    if not HAS_STATSMODELS:
        return pd.DataFrame(), pd.DataFrame()

    results = []
    pred_rows = []
    group_col = random_group_col or holdout_col
    random_slopes = random_slopes or []

    groups = sorted(df[holdout_col].dropna().unique())
    for group in groups:
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

        if center_by_group and center_group_col:
            train_df, test_df = _center_predictors_by_group(
                train_df, test_df, predictors, center_group_col
            )

        train_df, test_df, pred_cols, mapping = _standardize_predictors(
            train_df, test_df, predictors, standardize
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
        if len(sub) < max(min_rows, len(predictors) + 2):
            continue
        z_df = sub.copy()
        z_df[target] = _zscore(z_df[target])
        for col in predictors:
            z_df[col] = _zscore(z_df[col])

        X = z_df[predictors].to_numpy(dtype=float)
        y = z_df[target].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(X)), X])
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X.dot(coef)
        denom = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - (np.sum((y - y_pred) ** 2) / denom if denom != 0 else np.nan)
        row = {
            "benchmark": benchmark,
            "n": int(len(sub)),
            "r2": float(r2),
        }
        for name, value in zip(predictors, coef[1:]):
            row[name] = float(value)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
    return df_out


def compute_ranking_dataframe(pred_df, target_col, option_col, benchmark_col="benchmark"):
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
        top1 = int(pred_best_idx == true_best_idx)
        top3 = int(rank_true.loc[pred_best_idx] <= 3)
        regret = float(true_best - pred_best_true)
        spearman = spearman_corr(grouped["true_mean"].to_numpy(), grouped["pred_mean"].to_numpy())

        rows.append({
            benchmark_col: benchmark,
            "n_options": int(len(grouped)),
            "top1": top1,
            "top3": top3,
            "regret": regret,
            "spearman": spearman,
            "true_best_option": str(true_best_idx),
            "pred_best_option": str(pred_best_idx),
            "pred_top3_options": ",".join(str(x) for x in pred_top3),
        })

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    overall = {
        benchmark_col: "__overall__",
        "n_options": int(df_out["n_options"].sum()),
        "top1": float(df_out["top1"].mean()),
        "top3": float(df_out["top3"].mean()),
        "regret": float(df_out["regret"].mean()),
        "spearman": float(df_out["spearman"].mean()),
        "true_best_option": "n/a",
        "pred_best_option": "n/a",
    }
    df_out = pd.concat([df_out, pd.DataFrame([overall])], ignore_index=True)
    return df_out


def compute_ranking_summary(pred_df, target_col, option_col, output_path, benchmark_col="benchmark"):
    df_out = compute_ranking_dataframe(pred_df, target_col, option_col, benchmark_col)
    if not df_out.empty:
        df_out.to_csv(output_path, index=False)
        return df_out.to_dict(orient="records")
    return []


def compute_constant_selector(df, target_col, option_col, chosen_option, benchmark_col="benchmark"):
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

        rows.append({
            benchmark_col: benchmark,
            "n_options": int(len(grouped)),
            "top1": top1,
            "top3": top3,
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
            ranking_df = compute_ranking_dataframe(pred_df, target_col, option_col, benchmark_col)
        elif sel_type == "constant":
            option = selector.get("option")
            if not option:
                continue
            ranking_df = compute_constant_selector(df, target_col, option_col, option, benchmark_col)
        elif sel_type == "best_avg":
            valid = df.dropna(subset=[option_col, target_col])
            if valid.empty:
                continue
            option = valid.groupby(option_col)[target_col].mean().idxmax()
            ranking_df = compute_constant_selector(df, target_col, option_col, option, benchmark_col)
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
    if args.mixedlm_random_slopes:
        return [p.strip() for p in args.mixedlm_random_slopes.split(",") if p.strip()]

    slopes = [p for p in predictors if p in ("flow_recall_logit", "flow_mmd")]
    if not slopes:
        slopes = [p for p in predictors if p in ("flow_recall", "flow_mmd")]
    return slopes


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

    if use_logit:
        add_metric("flow_recall_logit", "flow_recall_logit", direction=1)
        add_metric("resnet_recall_logit", "resnet_recall_logit", direction=1)
        add_metric("dino_recall_logit", "dino_recall_logit", direction=1)
    else:
        add_metric("flow_recall", "flow_recall", direction=1)
        add_metric("resnet_recall", "resnet_recall", direction=1)
        add_metric("dino_recall", "dino_recall", direction=1)

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


def run_analysis_bundle(feature_df, out_dir, predictors, args):
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
        run_regression(feature_df, predictors, args.target, regression_path, use_mixedlm=True)

    if args.skip_prediction:
        return

    lobo_summary, lobo_preds = run_group_cv(
        feature_df,
        "benchmark",
        predictors,
        args.target,
        standardize=args.standardize,
        center_by_group=args.center_predictors_by_benchmark,
        center_group_col="benchmark",
    )
    if not lobo_summary.empty:
        lobo_summary.to_csv(out_dir / "prediction_lobo_summary.csv", index=False)
    if not lobo_preds.empty:
        lobo_preds.to_csv(out_dir / "prediction_lobo_rows.csv", index=False)
        compute_ranking_summary(
            lobo_preds,
            args.target,
            args.ranking_group,
            out_dir / "prediction_lobo_rank_summary.csv",
        )

    baseline_selectors = _build_baseline_selectors(feature_df, use_logit=args.logit_coverage)
    compute_baseline_rankings(
        feature_df,
        args.target,
        args.ranking_group,
        out_dir / "prediction_lobo_rank_baselines.csv",
        baseline_selectors,
    )

    if args.prediction_mixedlm and HAS_STATSMODELS:
        random_slopes = _select_random_slopes(args, predictors)
        lobo_mixed_summary, lobo_mixed_preds = run_group_cv_mixedlm(
            feature_df,
            holdout_col="benchmark",
            predictors=predictors,
            target=args.target,
            random_group_col="benchmark",
            random_slopes=random_slopes,
            standardize=args.standardize,
            center_by_group=args.center_predictors_by_benchmark,
            center_group_col="benchmark",
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
            )

    if args.loto_collapse_mixed:
        loto_df = feature_df.copy()
        loto_df["train_dataset_group"] = loto_df["train_dataset"].apply(collapse_mixed_dataset)
        group_col = "train_dataset_group"
    else:
        loto_df = feature_df
        group_col = "train_dataset"

    loto_summary, loto_preds = run_group_cv(
        loto_df,
        group_col,
        predictors,
        args.target,
        standardize=args.standardize,
        center_by_group=args.center_predictors_by_benchmark,
        center_group_col="benchmark",
    )
    if not loto_summary.empty:
        loto_summary.to_csv(out_dir / "prediction_loto_summary.csv", index=False)
    if not loto_preds.empty:
        loto_preds.to_csv(out_dir / "prediction_loto_rows.csv", index=False)

    if args.prediction_mixedlm and HAS_STATSMODELS:
        random_slopes = _select_random_slopes(args, predictors)
        loto_mixed_summary, loto_mixed_preds = run_group_cv_mixedlm(
            loto_df,
            holdout_col=group_col,
            predictors=predictors,
            target=args.target,
            random_group_col="benchmark",
            random_slopes=random_slopes,
            standardize=args.standardize,
            center_by_group=args.center_predictors_by_benchmark,
            center_group_col="benchmark",
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
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Warning: summary generation failed for {out_dir}: {exc}")


def run_regression(df, predictors, target, output_path, use_mixedlm=True):
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

    if HAS_STATSMODELS:
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
        "--ranking-group",
        default="train_dataset",
        help="Column to rank options within each benchmark (default: train_dataset).",
    )
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

        for bench, sub in df.groupby("benchmark"):
            auc, n_points, last_step = compute_auc(sub, args.metric, args.auc_steps, args.auc_pad)
            if last_step is not None and last_step < args.auc_steps:
                print(
                    f"Warning: {snapshot_dir.name} {bench} ends at {last_step} < {args.auc_steps} steps"
                )
            auc_rows.append(
                {
                    **run_info,
                    "benchmark": bench,
                    "auc_steps": int(args.auc_steps),
                    "auc": auc,
                    "auc_normalized": auc / args.auc_steps if args.auc_steps else np.nan,
                    "auc_points": int(n_points),
                }
            )

        for row in compute_curve_stats(df, args.metric):
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

    flow_lookup = load_coverage_lookup(args.coverage_csv)
    resnet_lookup = load_coverage_lookup(args.coverage_resnet_csv)
    dino_lookup = load_coverage_lookup(args.coverage_dino_csv) if args.coverage_dino_csv else None
    flow_mmd_lookup = load_mmd_lookup(args.flow_mmd_csv)
    feature_mmd_lookup = load_mmd_lookup(args.feature_mmd_csv)
    dino_mmd_lookup = load_mmd_lookup(args.dino_mmd_csv) if args.dino_mmd_csv else None

    feature_df, missing = build_auc_feature_table(
        auc_df,
        flow_lookup,
        resnet_lookup,
        flow_mmd_lookup,
        feature_mmd_lookup,
        logit_coverage=args.logit_coverage,
        dino_lookup=dino_lookup,
        dino_mmd_lookup=dino_mmd_lookup,
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
    if not feature_df.empty:
        feature_df.to_csv(out_dir / "auc_with_features.csv", index=False)

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
                "flow_recall_logit",
                "flow_precision_logit",
                "resnet_recall_logit",
                "resnet_precision_logit",
                "flow_mmd",
                "feature_mmd",
            ]
        else:
            predictors = [
                "flow_recall",
                "flow_precision",
                "resnet_recall",
                "resnet_precision",
                "flow_mmd",
                "feature_mmd",
            ]

    if args.target not in feature_df.columns:
        print(f"Target '{args.target}' not found in auc_with_features table.")
        return

    run_analysis_bundle(feature_df, out_dir, predictors, args)
    if args.run_summary:
        run_summary_report(out_dir, predictors, args)

    if args.per_encoder:
        per_dir = out_dir / "by_encoder"
        per_dir.mkdir(parents=True, exist_ok=True)
        for (pretrained, freeze), group in feature_df.groupby(
            ["pretrained", "freeze"], dropna=False
        ):
            tag = f"pretrained{_format_bool(pretrained)}_freeze{_format_bool(freeze)}"
            group_dir = per_dir / tag
            group_dir.mkdir(parents=True, exist_ok=True)
            group.to_csv(group_dir / "auc_with_features.csv", index=False)
            run_analysis_bundle(group, group_dir, predictors, args)
            if args.run_summary:
                run_summary_report(group_dir, predictors, args)

    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
