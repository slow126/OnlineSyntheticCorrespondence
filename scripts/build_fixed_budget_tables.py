#!/usr/bin/env python3
"""
Build fixed-budget tables from validation_results.csv across snapshot directories.

Outputs:
  - Raw table with snapshot_path per row.
  - Filtered table with duplicate snapshot names removed and short runs dropped.
  - Grouped summary table with common-step stats and limiting snapshots.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


MODEL_TOKENS = ("raft", "flowformer", "cats")
DEFAULT_MODEL = "cats"


def normalize_dataset_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return str(name).strip().lower().replace("+", "_")


def _parse_bool(text: str) -> Optional[bool]:
    value = text.strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def parse_training_summary(summary_path: Path) -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {
        "train_dataset": None,
        "backbone": None,
        "pretrained": None,
        "freeze": None,
    }
    if not summary_path.exists():
        return info
    try:
        with summary_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Train dataset:"):
                    info["train_dataset"] = normalize_dataset_name(line.split(":", 1)[1])
                elif line.startswith("Backbone:"):
                    info["backbone"] = line.split(":", 1)[1].strip().lower()
                elif line.startswith("Pretrained backbone:"):
                    info["pretrained"] = _parse_bool(line.split(":", 1)[1])
                elif line.startswith("Freeze backbone:"):
                    info["freeze"] = _parse_bool(line.split(":", 1)[1])
    except OSError:
        return info
    return info


def parse_name_flags(snapshot_name: str) -> Dict[str, Optional[str]]:
    name = snapshot_name.lower()
    info: Dict[str, Optional[str]] = {}
    if "pretrainedtrue" in name:
        info["pretrained"] = True
    elif "pretrainedfalse" in name:
        info["pretrained"] = False
    if "freezetrue" in name:
        info["freeze"] = True
    elif "freezefalse" in name:
        info["freeze"] = False
    backbone_match = re.search(r"(resnet\d+)", name)
    if backbone_match:
        info["backbone"] = backbone_match.group(1)
    return info


def parse_dataset_from_name(snapshot_name: str) -> Optional[str]:
    base = snapshot_name
    stamp_match = re.match(r"^(?P<base>.+)_(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$", base)
    if stamp_match:
        base = stamp_match.group("base")

    stop_prefixes = (
        "steps",
        "logsteps",
        "pretrained",
        "freeze",
        "stride",
        "sequence",
        "eval",
        "full",
        "size",
    )
    parts = base.split("_")
    dataset_parts: List[str] = []
    for part in parts:
        lower = part.lower()
        if lower in MODEL_TOKENS:
            break
        if lower.startswith(stop_prefixes):
            break
        if re.fullmatch(r"\d+", lower):
            if dataset_parts:
                dataset_parts.append(part)
                continue
            break
        dataset_parts.append(part)
    if not dataset_parts:
        return None
    return normalize_dataset_name("_".join(dataset_parts))


def choose_train_dataset(
    summary_dataset: Optional[str], name_dataset: Optional[str]
) -> Optional[str]:
    if summary_dataset and name_dataset:
        if summary_dataset == name_dataset:
            return summary_dataset
        if summary_dataset in name_dataset:
            return name_dataset
        if name_dataset in summary_dataset:
            return summary_dataset
        return name_dataset if len(name_dataset) > len(summary_dataset) else summary_dataset
    return summary_dataset or name_dataset


def detect_model_tag(snapshot_dir: Path, default_model: str) -> str:
    for part in snapshot_dir.parts:
        lower = part.lower()
        for token in MODEL_TOKENS:
            if token in lower:
                return token
    return default_model


def build_encoder_regime(backbone: Optional[str], pretrained: Optional[bool], freeze: Optional[bool]) -> str:
    tokens = []
    if backbone:
        tokens.append(backbone)
    if pretrained is not None:
        tokens.append(f"pretrained{pretrained}")
    if freeze is not None:
        tokens.append(f"freeze{freeze}")
    return "_".join(tokens) if tokens else "unknown"


def collect_snapshot_dirs(roots: Iterable[str]) -> List[Path]:
    snapshot_dirs = []
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.exists():
            print(f"Warning: Snapshot directory not found: {root}")
            continue
        for csv_path in root_path.rglob("validation_results.csv"):
            snapshot_dirs.append(csv_path.parent)
    return sorted(set(snapshot_dirs))


def _snapshot_name(path_str: str) -> str:
    return Path(path_str).name


def _row_max_step(df: pd.DataFrame) -> pd.Series:
    step_cols = [c for c in df.columns if c.startswith("step_")]
    if not step_cols:
        return pd.Series([None] * len(df), index=df.index)
    step_numbers = [int(c.split("_", 1)[1]) for c in step_cols]
    step_df = df[step_cols].copy()
    step_df.columns = step_numbers
    return step_df.apply(
        lambda row: row.dropna().index.max() if row.notna().any() else None,
        axis=1,
    )


def _max_common_step(df: pd.DataFrame) -> Optional[int]:
    if df.empty:
        return None
    step_cols = [c for c in df.columns if c.startswith("step_")]
    if not step_cols:
        return None
    common_mask = df[step_cols].notna().all(axis=0)
    if not common_mask.any():
        return None
    common_steps = [int(col.split("_", 1)[1]) for col in common_mask.index[common_mask]]
    return max(common_steps) if common_steps else None


def _format_paths(paths: Iterable[str]) -> str:
    return ";".join(sorted(set(paths)))


def _format_step_list(steps: Iterable[Optional[float]]) -> str:
    values = [int(step) for step in steps if pd.notna(step)]
    values.sort()
    return "[" + ",".join(str(v) for v in values) + "]"


def load_validation_csv(
    csv_path: Path,
    metric: str,
    step_col: str,
    benchmarks: Optional[Iterable[str]],
    min_step: Optional[int],
    max_step: Optional[int],
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if metric not in df.columns:
        return pd.DataFrame()
    if step_col not in df.columns:
        if "training_steps" in df.columns:
            step_col = "training_steps"
        elif "epoch" in df.columns:
            step_col = "epoch"
        else:
            return pd.DataFrame()

    df = df.copy()
    df[step_col] = pd.to_numeric(df[step_col], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    if "benchmark" not in df.columns:
        return pd.DataFrame()
    df["benchmark"] = df["benchmark"]
    df = df.dropna(subset=[step_col, metric, "benchmark"])
    if df.empty:
        return df

    df[step_col] = df[step_col].astype(int)
    df["benchmark"] = df["benchmark"].astype(str).str.lower()

    if benchmarks:
        bench_set = {b.strip().lower() for b in benchmarks if b.strip()}
        df = df[df["benchmark"].isin(bench_set)]

    if min_step is not None:
        df = df[df[step_col] >= min_step]
    if max_step is not None:
        df = df[df[step_col] <= max_step]
    df = df.rename(columns={step_col: "eval_step"})
    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build fixed-budget eval tables from validation_results.csv files."
    )
    parser.add_argument(
        "snapshot_dirs",
        nargs="+",
        help="Root directories that contain snapshot subdirectories.",
    )
    parser.add_argument("--metric", default="pck", help="Metric column to extract (default: pck).")
    parser.add_argument(
        "--step-col",
        default="training_steps",
        help="Column to use as eval step (default: training_steps).",
    )
    parser.add_argument(
        "--output",
        default="analysis/fixed_budget_eval_tables.csv",
        help="Output CSV path for the raw wide table.",
    )
    parser.add_argument(
        "--output-filtered",
        default="analysis/fixed_budget_eval_tables_filtered.csv",
        help="Output CSV path for the filtered wide table.",
    )
    parser.add_argument(
        "--output-grouped",
        default="analysis/fixed_budget_eval_tables_grouped.csv",
        help="Output CSV path for the grouped summary table.",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Comma-separated list of benchmarks to include.",
    )
    parser.add_argument(
        "--agg",
        choices=["mean", "median", "max"],
        default="mean",
        help="Aggregation for duplicate runs at the same step.",
    )
    parser.add_argument(
        "--min-step",
        type=int,
        default=None,
        help="Minimum eval step to include.",
    )
    parser.add_argument(
        "--max-step",
        type=int,
        default=None,
        help="Maximum eval step to include.",
    )
    parser.add_argument(
        "--per-group-dir",
        default=None,
        help="Optional directory to write per-group tables.",
    )
    parser.add_argument(
        "--default-model",
        default=DEFAULT_MODEL,
        help="Model tag to use when none is detected (default: cats).",
    )
    parser.add_argument(
        "--min-step-threshold",
        type=int,
        default=1000,
        help="Drop rows with max eval step <= threshold (default: 1000).",
    )
    parser.add_argument(
        "--no-dedupe-snapshot-name",
        action="store_true",
        help="Disable dedupe by snapshot directory name.",
    )

    args = parser.parse_args()
    benchmarks = args.benchmarks.split(",") if args.benchmarks else None

    snapshot_dirs = collect_snapshot_dirs(args.snapshot_dirs)
    if not snapshot_dirs:
        print("No snapshot directories found.")
        return 1

    rows = []
    for snapshot_dir in snapshot_dirs:
        csv_path = snapshot_dir / "validation_results.csv"
        if not csv_path.exists():
            continue

        summary_info = parse_training_summary(snapshot_dir / "training_summary.txt")
        name_info = parse_name_flags(snapshot_dir.name)
        train_dataset = choose_train_dataset(
            summary_info.get("train_dataset"),
            parse_dataset_from_name(snapshot_dir.name),
        )
        backbone = summary_info.get("backbone") or name_info.get("backbone")
        pretrained = summary_info.get("pretrained")
        if pretrained is None:
            pretrained = name_info.get("pretrained")
        freeze = summary_info.get("freeze")
        if freeze is None:
            freeze = name_info.get("freeze")

        model_tag = detect_model_tag(snapshot_dir, args.default_model)
        encoder_regime = build_encoder_regime(backbone, pretrained, freeze)

        df = load_validation_csv(
            csv_path=csv_path,
            metric=args.metric,
            step_col=args.step_col,
            benchmarks=benchmarks,
            min_step=args.min_step,
            max_step=args.max_step,
        )
        if df.empty:
            continue

        for _, row in df.iterrows():
            rows.append(
                {
                    "train_dataset": train_dataset or "unknown",
                    "model": model_tag,
                    "encoder_regime": encoder_regime,
                    "benchmark": row["benchmark"],
                    "eval_step": int(row["eval_step"]),
                    args.metric: float(row[args.metric]),
                    "snapshot_path": str(snapshot_dir),
                }
            )

    if not rows:
        print("No validation rows found.")
        return 1

    data = pd.DataFrame(rows)
    group_cols = [
        "train_dataset",
        "model",
        "encoder_regime",
        "benchmark",
        "snapshot_path",
        "eval_step",
    ]
    grouped = (
        data.groupby(group_cols, as_index=False)[args.metric]
        .agg(args.agg)
        .rename(columns={args.metric: "metric"})
    )

    wide = grouped.pivot_table(
        index=["train_dataset", "model", "encoder_regime", "benchmark", "snapshot_path"],
        columns="eval_step",
        values="metric",
        aggfunc="first",
    )
    wide = wide.sort_index(axis=1)
    wide.columns = [f"step_{col}" for col in wide.columns]
    wide = wide.reset_index()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(output_path, index=False)
    print(f"Wrote {len(wide)} rows to {output_path}")

    filtered = wide.copy()
    filtered["snapshot_name"] = filtered["snapshot_path"].apply(_snapshot_name)
    if not args.no_dedupe_snapshot_name:
        chosen = (
            filtered.sort_values("snapshot_path")
            .groupby("snapshot_name", as_index=False)["snapshot_path"]
            .first()
        )
        keep_paths = set(chosen["snapshot_path"].tolist())
        filtered = filtered[filtered["snapshot_path"].isin(keep_paths)]

    if args.min_step_threshold is not None:
        row_max = _row_max_step(filtered)
        filtered = filtered[row_max.fillna(0) > args.min_step_threshold]

    filtered = filtered.drop(columns=["snapshot_name"])
    filtered_output = Path(args.output_filtered)
    filtered_output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(filtered_output, index=False)
    print(f"Wrote {len(filtered)} rows to {filtered_output}")
    filtered_common_step = _max_common_step(filtered)
    print(f"Max common training step (filtered): {filtered_common_step}")

    grouped_rows = []
    if not filtered.empty:
        group_keys = ["train_dataset", "model", "encoder_regime", "benchmark"]
        row_max = _row_max_step(filtered)
        filtered_with_max = filtered.assign(row_max_step=row_max)
        for group, group_df in filtered_with_max.groupby(group_keys):
            max_common_step = _max_common_step(group_df)
            max_steps = group_df["row_max_step"].dropna()
            min_max_step = int(max_steps.min()) if not max_steps.empty else None
            max_max_step = int(max_steps.max()) if not max_steps.empty else None
            limiting_paths = []
            if min_max_step is not None:
                limiting_paths = group_df.loc[
                    group_df["row_max_step"] == min_max_step, "snapshot_path"
                ].tolist()
            snapshot_max_steps = _format_step_list(group_df["row_max_step"].tolist())
            grouped_rows.append(
                {
                    "train_dataset": group[0],
                    "model": group[1],
                    "encoder_regime": group[2],
                    "benchmark": group[3],
                    "snapshot_count": int(len(group_df)),
                    "max_common_step": max_common_step,
                    "min_snapshot_max_step": min_max_step,
                    "max_snapshot_max_step": max_max_step,
                    "snapshot_max_steps": snapshot_max_steps,
                    "limiting_snapshot_paths": _format_paths(limiting_paths),
                }
            )

    grouped_columns = [
        "train_dataset",
        "model",
        "encoder_regime",
        "benchmark",
        "snapshot_count",
        "max_common_step",
        "min_snapshot_max_step",
        "max_snapshot_max_step",
        "snapshot_max_steps",
        "limiting_snapshot_paths",
    ]
    grouped_df = pd.DataFrame(grouped_rows, columns=grouped_columns)
    grouped_output = Path(args.output_grouped)
    grouped_output.parent.mkdir(parents=True, exist_ok=True)
    grouped_df.to_csv(grouped_output, index=False)
    print(f"Wrote {len(grouped_df)} rows to {grouped_output}")

    if args.per_group_dir:
        per_group_root = Path(args.per_group_dir)
        per_group_root.mkdir(parents=True, exist_ok=True)
        for (train_dataset, model, encoder_regime), sub in grouped.groupby(
            ["train_dataset", "model", "encoder_regime"]
        ):
            per_table = sub.pivot_table(
                index="benchmark",
                columns="eval_step",
                values="metric",
                aggfunc="first",
            ).sort_index(axis=1)
            per_table.columns = [f"step_{col}" for col in per_table.columns]
            safe_name = f"{train_dataset}__{model}__{encoder_regime}".replace("/", "_")
            per_path = per_group_root / f"{safe_name}.csv"
            per_table.to_csv(per_path)
        print(f"Wrote per-group tables to {per_group_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
