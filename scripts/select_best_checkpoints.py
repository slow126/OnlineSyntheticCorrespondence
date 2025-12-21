#!/usr/bin/env python3
"""
Select best checkpoints from validation_results.csv using configurable policies.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


RUN_NAME_RE = re.compile(
    r"^(?P<train_dataset>.+)_(?P<step_tag>logsteps|steps)(?P<logsteps>\d+)"
    r"_pretrained(?P<pretrained>True|False)_freeze(?P<freeze>True|False)"
    r"_(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$"
)


def normalize_dataset_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return str(name).strip().lower().replace("+", "_")


def parse_training_summary(summary_path: Path) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {}
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


def parse_run_name(snapshot_dir: Path) -> Dict[str, Optional[str]]:
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
            info[key] = summary_info[key]
        if info.get(key) is None and config_info.get(key) is not None:
            info[key] = config_info[key]
    return info


def parse_run_config(config_path: Path) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {}
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


def list_snapshot_dirs(root_dirs: List[str]) -> List[Path]:
    snapshot_dirs = []
    for root_dir in root_dirs:
        root = Path(root_dir)
        if not root.exists():
            continue
        for csv_path in root.rglob("validation_results.csv"):
            snapshot_dirs.append(csv_path.parent)
    return sorted(set(snapshot_dirs))


def _find_nearest_step(steps: np.ndarray, target_step: int) -> Optional[int]:
    if len(steps) == 0:
        return None
    steps = np.array(sorted(steps))
    return int(steps[np.abs(steps - target_step).argmin()])


def _prepare_df(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.copy()
    df["training_steps"] = pd.to_numeric(df["training_steps"], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=["training_steps", "benchmark", metric])
    df["training_steps"] = df["training_steps"].astype(int)
    df["benchmark"] = df["benchmark"].astype(str).str.lower()
    return df


def _filter_benchmarks(df: pd.DataFrame, benchmarks: Optional[List[str]]) -> pd.DataFrame:
    if not benchmarks:
        return df
    bset = {b.strip().lower() for b in benchmarks if b.strip()}
    return df[df["benchmark"].isin(bset)]


def _mean_metric_per_step(df: pd.DataFrame, metric: str) -> pd.Series:
    return df.groupby("training_steps")[metric].mean()


def _select_best_metric(
    df: pd.DataFrame,
    metric: str,
    higher_is_better: bool
) -> Tuple[Optional[int], Optional[float]]:
    mean_by_step = _mean_metric_per_step(df, metric)
    if mean_by_step.empty:
        return None, None
    if higher_is_better:
        step = int(mean_by_step.idxmax())
    else:
        step = int(mean_by_step.idxmin())
    return step, float(mean_by_step.loc[step])


def _compute_auc_curve(
    df: pd.DataFrame,
    metric: str,
    auc_steps: Optional[int],
    auc_pad: bool,
    metric_sign: float,
) -> Dict[int, float]:
    sub = df.copy()
    if auc_steps is not None:
        sub = sub[sub["training_steps"] <= auc_steps]
    sub = sub.groupby("training_steps", as_index=False)[metric].mean()
    sub = sub.sort_values("training_steps")
    if len(sub) < 2:
        return {}

    steps = sub["training_steps"].to_numpy(dtype=int)
    values = sub[metric].to_numpy(dtype=float) * metric_sign
    auc_by_step: Dict[int, float] = {}

    for i in range(1, len(steps)):
        auc = float(np.trapz(values[: i + 1], steps[: i + 1]))
        if auc_pad and auc_steps is not None and i == len(steps) - 1 and steps[-1] < auc_steps:
            auc += float((auc_steps - steps[-1]) * values[-1])
        auc_by_step[int(steps[i])] = auc
    return auc_by_step


def _select_by_auc(
    df: pd.DataFrame,
    metric: str,
    auc_steps: Optional[int],
    auc_pad: bool,
    higher_is_better: bool,
) -> Tuple[Optional[int], Optional[float]]:
    metric_sign = 1.0 if higher_is_better else -1.0
    auc_curves = []
    for _, sub in df.groupby("benchmark"):
        curve = _compute_auc_curve(sub, metric, auc_steps, auc_pad, metric_sign)
        if curve:
            auc_curves.append(curve)

    if not auc_curves:
        return None, None

    all_steps = sorted({step for curve in auc_curves for step in curve})
    best_step = None
    best_score = None
    for step in all_steps:
        values = [curve.get(step) for curve in auc_curves if step in curve]
        if not values:
            continue
        avg_auc = float(np.mean(values))
        if best_score is None or avg_auc > best_score:
            best_score = avg_auc
            best_step = step
    return best_step, best_score


def _select_fixed_step(
    df: pd.DataFrame,
    metric: str,
    fixed_steps: int,
    fixed_policy: str,
) -> Tuple[Optional[int], Optional[float]]:
    steps = df["training_steps"].dropna().unique()
    if fixed_policy == "nearest":
        selected = _find_nearest_step(steps, fixed_steps)
    else:
        selected = int(fixed_steps) if fixed_steps in steps else None
    if selected is None:
        return None, None
    mean_by_step = _mean_metric_per_step(df, metric)
    if selected not in mean_by_step.index:
        return None, None
    return int(selected), float(mean_by_step.loc[selected])


def main() -> None:
    parser = argparse.ArgumentParser(description="Select best checkpoints from validation curves.")
    parser.add_argument(
        "--snapshots-dir",
        nargs="+",
        default=["snapshots"],
        help="One or more snapshot root directories to scan.",
    )
    parser.add_argument(
        "--output",
        default="analysis/selected_checkpoints.csv",
        help="CSV output file with selected checkpoints.",
    )
    parser.add_argument(
        "--metric",
        default="pck",
        help="Metric column to use for selection.",
    )
    parser.add_argument(
        "--metric-mode",
        choices=["higher", "lower"],
        default="higher",
        help="Whether higher or lower metric values are better.",
    )
    parser.add_argument(
        "--select",
        choices=["auc", "best_metric", "fixed"],
        default="auc",
        help="Selection policy for checkpoints.",
    )
    parser.add_argument(
        "--auc-steps",
        type=int,
        default=None,
        help="Max training_steps for AUC selection (optional).",
    )
    parser.add_argument(
        "--auc-pad",
        action="store_true",
        help="Pad last value to auc-steps when runs end early.",
    )
    parser.add_argument(
        "--fixed-steps",
        type=int,
        default=None,
        help="Fixed training_steps for selection (required for select=fixed).",
    )
    parser.add_argument(
        "--fixed-policy",
        choices=["nearest", "exact"],
        default="nearest",
        help="How to handle missing fixed steps.",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated benchmarks to use for selection (optional).",
    )
    args = parser.parse_args()

    benchmarks = None
    if args.benchmarks:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    higher_is_better = args.metric_mode == "higher"
    snapshot_dirs = list_snapshot_dirs(args.snapshots_dir)

    rows = []
    for snapshot_dir in snapshot_dirs:
        csv_path = snapshot_dir / "validation_results.csv"
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if args.metric not in df.columns:
            continue

        df = _prepare_df(df, args.metric)
        if df.empty:
            continue
        df = _filter_benchmarks(df, benchmarks)
        if df.empty:
            continue

        if args.select == "fixed":
            if args.fixed_steps is None:
                raise ValueError("--fixed-steps is required when select=fixed")
            step, score = _select_fixed_step(df, args.metric, args.fixed_steps, args.fixed_policy)
        elif args.select == "best_metric":
            step, score = _select_best_metric(df, args.metric, higher_is_better)
        else:
            step, score = _select_by_auc(df, args.metric, args.auc_steps, args.auc_pad, higher_is_better)
            if step is None:
                step, score = _select_best_metric(df, args.metric, higher_is_better)

        if step is None:
            continue

        step_rows = df[df["training_steps"] == step]
        epoch = None
        if "epoch" in step_rows.columns and not step_rows.empty:
            try:
                epoch = int(step_rows["epoch"].iloc[0])
            except Exception:
                epoch = None

        run_info = parse_run_name(snapshot_dir)
        checkpoint_path = None
        if epoch is not None:
            checkpoint_path = str(snapshot_dir / f"epoch_{epoch}.pth")

        rows.append(
            {
                **run_info,
                "selected_training_steps": int(step),
                "selected_epoch": epoch,
                "selected_metric": float(score) if score is not None else np.nan,
                "selection_policy": args.select,
                "metric": args.metric,
                "auc_steps": args.auc_steps,
                "benchmarks": ",".join(benchmarks) if benchmarks else "__all__",
                "checkpoint_path": checkpoint_path,
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
