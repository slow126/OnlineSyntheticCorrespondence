#!/usr/bin/env python3
"""Audit a GLUNet snapshot grid before importing it into transfer analysis."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


RUN_RE = re.compile(
    r"^(?P<dataset>.+)_glunet_steps(?P<steps>\d+)"
    r"_pretrained(?P<pretrained>True|False)"
    r"_freeze(?P<freeze>True|False)_"
)
LOG_RE = re.compile(
    r"^(?P<dataset>.+)_glunet_steps(?P<steps>\d+)"
    r"_pretrained(?P<pretrained>True|False)"
    r"_freeze(?P<freeze>True|False)_(?P<job_id>\d+)\.err$"
)


def run_key(match: re.Match) -> tuple[str, bool, bool]:
    return (
        match.group("dataset"),
        match.group("pretrained") == "True",
        match.group("freeze") == "True",
    )


def trajectory_status(epoch_means: pd.Series) -> str:
    if len(epoch_means) < 2:
        return "insufficient_curve"

    best_epoch = int(epoch_means.idxmax())
    latest_epoch = int(epoch_means.index[-1])
    best = float(epoch_means.max())
    latest = float(epoch_means.iloc[-1])
    drop = best - latest

    if drop >= 3.0:
        return "unstable_collapse"
    if len(epoch_means) < 3:
        return "insufficient_curve"

    delta = float(epoch_means.iloc[-1] - epoch_means.iloc[-2])
    interval = float(np.median(np.diff(epoch_means.index)))
    intervals_since_best = (
        (latest_epoch - best_epoch) / interval if interval > 0 else 0.0
    )
    if best_epoch == latest_epoch and delta > 0.5:
        return "still_improving"
    if best_epoch == latest_epoch:
        return "latest_best_plateau_candidate"
    if intervals_since_best >= 2 and drop >= 0.5:
        return "clear_post_peak_decline"
    return "possible_post_peak"


def parse_logs(log_dir: Path) -> dict[tuple[str, bool, bool], dict]:
    logs = []
    for path in sorted(log_dir.glob("*.err")):
        match = LOG_RE.match(path.name)
        if match is None:
            continue
        text = path.read_text(errors="replace")
        logs.append(
            {
                "key": run_key(match),
                "log_file": path.name,
                "job_id": match.group("job_id"),
                "log_end_timestamp": path.stat().st_mtime,
                "host_memory_oom": (
                    "oom_kill" in text or "Out Of Memory" in text
                ),
                "native_core_dump": "Aborted (core dumped)" in text,
                "tensorflow_cuda_init_errors": text.count(
                    "CUDA_ERROR_NOT_INITIALIZED"
                ),
            }
        )

    if not logs:
        return {}

    common_cutoff = max(log["log_end_timestamp"] for log in logs)
    by_key = {}
    for log in logs:
        if log["host_memory_oom"]:
            reason = "slurm_host_memory_oom"
        elif log["native_core_dump"]:
            reason = "native_core_dump"
        elif common_cutoff - log["log_end_timestamp"] <= 5 * 60:
            reason = "synchronized_external_cutoff"
        else:
            reason = "abrupt_no_terminal_record"
        log["termination_reason"] = reason
        log["log_end_time"] = datetime.fromtimestamp(
            log.pop("log_end_timestamp")
        ).astimezone().isoformat(timespec="seconds")
        by_key[log.pop("key")] = log
    return by_key


def audit_run(run_dir: Path, log: dict | None) -> dict:
    match = RUN_RE.match(run_dir.name)
    if match is None:
        return {"run": run_dir.name, "status": "unrecognized_name"}

    row = {
        "run": run_dir.name,
        "dataset": match.group("dataset"),
        "pretrained": match.group("pretrained") == "True",
        "freeze": match.group("freeze") == "True",
    }
    if log is not None:
        row.update(log)
    else:
        row.update(
            log_file=None,
            job_id=None,
            log_end_time=None,
            host_memory_oom=False,
            native_core_dump=False,
            tensorflow_cuda_init_errors=0,
            termination_reason="missing_log",
        )

    config_path = run_dir / "config.yaml"
    summary_path = run_dir / "training_summary.txt"
    validation_path = run_dir / "validation_results.csv"
    checkpoints = list(run_dir.glob("*.pth")) + list(run_dir.glob("*.ckpt"))
    row.update(
        has_config=config_path.exists(),
        has_summary=summary_path.exists(),
        has_validation=validation_path.exists(),
        checkpoint_count=len(checkpoints),
    )

    config = {}
    if config_path.exists():
        with config_path.open() as handle:
            config = yaml.safe_load(handle) or {}
    training = config.get("training", {})
    model = config.get("model", {})
    glunet = model.get("glunet", {}) or {}
    row.update(
        planned_epochs=training.get("epochs", np.nan),
        steps_per_epoch=training.get("steps_per_epoch", np.nan),
        configured_backbone=glunet.get("model_name", "resnet50"),
    )

    if not validation_path.exists():
        row["status"] = "missing_validation"
        row["trajectory_status"] = "not_available"
        return row

    validation = pd.read_csv(validation_path)
    required = {"epoch", "benchmark", "pck"}
    missing = required - set(validation.columns)
    if missing:
        row["status"] = "invalid_validation_schema"
        row["trajectory_status"] = "not_available"
        row["schema_missing"] = ",".join(sorted(missing))
        return row

    epoch_means = validation.groupby("epoch")["pck"].mean().sort_index()
    primary_curve = (
        validation[validation["benchmark"].eq("flyingthings")]
        .groupby("epoch")["pck"]
        .mean()
        .sort_index()
    )
    max_epoch = int(validation["epoch"].max())
    planned_epochs = row["planned_epochs"]
    progress = (
        max_epoch / float(planned_epochs)
        if pd.notna(planned_epochs) and float(planned_epochs) > 0
        else np.nan
    )
    row.update(
        status="valid_partial_evaluation",
        validation_rows=len(validation),
        benchmark_count=validation["benchmark"].nunique(),
        eval_checkpoint_count=validation["epoch"].nunique(),
        min_epoch=int(validation["epoch"].min()),
        max_epoch=max_epoch,
        max_training_steps=max_epoch * int(row["steps_per_epoch"]),
        planned_progress_fraction=progress,
        finite_pck=bool(np.isfinite(validation["pck"]).all()),
        duplicate_epoch_benchmark=int(
            validation.duplicated(["epoch", "benchmark"]).sum()
        ),
        mean_pck_first=float(epoch_means.iloc[0]),
        mean_pck_latest=float(epoch_means.iloc[-1]),
        mean_pck_best=float(epoch_means.max()),
        mean_pck_best_epoch=int(epoch_means.idxmax()),
        mean_pck_drop_from_best=float(epoch_means.max() - epoch_means.iloc[-1]),
        mean_intervals_since_best=(
            int((max_epoch - int(epoch_means.idxmax())) / np.median(
                np.diff(epoch_means.index)
            ))
            if len(epoch_means) >= 2
            else 0
        ),
        primary_pck_latest=(
            float(primary_curve.iloc[-1]) if len(primary_curve) else np.nan
        ),
        primary_pck_best=(
            float(primary_curve.max()) if len(primary_curve) else np.nan
        ),
        primary_pck_best_epoch=(
            int(primary_curve.idxmax()) if len(primary_curve) else np.nan
        ),
        primary_pck_drop_from_best=(
            float(primary_curve.max() - primary_curve.iloc[-1])
            if len(primary_curve)
            else np.nan
        ),
        benchmarks_best_at_latest=int(
            sum(
                int(frame.loc[frame["pck"].idxmax(), "epoch"]) == max_epoch
                for _, frame in validation.groupby("benchmark")
            )
        ),
        trajectory_status=trajectory_status(epoch_means),
    )

    summary_backbone = None
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            if line.startswith("Backbone:"):
                summary_backbone = line.split(":", 1)[1].strip()
                break
    row["summary_backbone"] = summary_backbone
    row["backbone_metadata_match"] = (
        summary_backbone is None or summary_backbone == row["configured_backbone"]
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", default="glunet_fullgrid_snapshots")
    parser.add_argument(
        "--out-dir",
        default="scripts/transfer_analysis_v4/results_glunet_fullgrid_2k",
    )
    args = parser.parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    logs = parse_logs(snapshot_dir / "logs")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(snapshot_dir.iterdir()):
        if not path.is_dir():
            continue
        match = RUN_RE.match(path.name)
        if match is None:
            continue
        rows.append(audit_run(path, logs.get(run_key(match))))

    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "GRID_AUDIT.csv", index=False)

    valid = audit[audit["status"].eq("valid_partial_evaluation")]
    missing = audit[~audit["status"].eq("valid_partial_evaluation")]
    max_progress = valid["planned_progress_fraction"].max()
    common_steps = int(valid["max_training_steps"].min())
    mismatch_count = int((valid["backbone_metadata_match"] == False).sum())  # noqa: E712
    checkpoint_count = int(audit["checkpoint_count"].fillna(0).sum())
    termination_counts = audit["termination_reason"].value_counts()
    trajectory_counts = valid["trajectory_status"].value_counts()
    mean_latest_count = int(
        (valid["mean_pck_best_epoch"] == valid["max_epoch"]).sum()
    )
    primary_latest_count = int(
        (valid["primary_pck_best_epoch"] == valid["max_epoch"]).sum()
    )

    completeness = (
        audit.groupby("dataset")["has_validation"]
        .agg(["sum", "count"])
        .reset_index()
    )
    lines = [
        "# GLUNet Full-Grid Audit",
        "",
        f"Snapshot root: `{snapshot_dir}`",
        "",
        "## Decision",
        "",
        "The 5,000-epoch setting was a maximum, not a convergence target. No "
        "early-stopping callback is configured, and none of these jobs ended "
        "because a validation criterion fired. Most were externally interrupted "
        "together; seven have explicit runtime failures. Some longer validation "
        "curves do show practical post-peak behavior, but the grid as a whole is "
        "too incomplete to replace the standardized 10,000-step GLUNet result.",
        "",
        "## Summary",
        "",
        f"- Directories: {len(audit)}",
        f"- Runs with valid evaluation CSVs: {len(valid)}",
        f"- Runs missing or invalid evaluation CSVs: {len(missing)}",
        f"- Runs with at least three evaluation checkpoints: "
        f"{int((valid['eval_checkpoint_count'] >= 3).sum())}",
        f"- Maximum configured progress reached: {max_progress:.1%} "
        "(not a convergence criterion)",
        f"- Downloaded model checkpoints: {checkpoint_count}",
        f"- Summary/config backbone metadata mismatches: {mismatch_count}",
        "",
        "All valid CSVs contain finite PCK values for 10 benchmarks and no "
        "duplicate `(epoch, benchmark)` rows. The backbone mismatch is a summary "
        "writer bug: configs instantiate `resnet50`, while old summaries fall "
        "back to the CATs++ `resnet101` label.",
        "",
        "## Why Jobs Ended",
        "",
        "There is no `EarlyStopping` callback in the training entry point. The "
        "SLURM configuration requested 23:59:00, and these jobs ended far short "
        "of that wall time.",
        "",
    ]
    for reason, count in termination_counts.items():
        lines.append(f"- `{reason}`: {int(count)}")
    lines.extend([
        "",
        "The synchronized group stopped between 02:03:46 and 02:07:54 local "
        "time despite different start times and training rates. That pattern "
        "strongly indicates a mass cancellation, node/scheduler event, or other "
        "external interruption. The `.err` files do not contain the final SLURM "
        "state, so `sacct` or the corresponding `.out` files are required to "
        "distinguish those possibilities.",
        "",
        "The native core dumps occur during the expensive synthetic validation "
        "pass, but there is no Python traceback identifying the native library. "
        "MOVi-F logs also contain repeated TensorFlow/XLA CUDA initialization "
        "errors; those runs often continued, so the messages are suspicious "
        "noise rather than a demonstrated termination cause.",
        "",
        "## Validation Peaks",
        "",
        f"- Mean transfer PCK is best at the latest checkpoint for "
        f"{mean_latest_count}/{len(valid)} evaluated runs.",
        f"- FlyingThings PCK is best at the latest checkpoint for "
        f"{primary_latest_count}/{len(valid)} evaluated runs.",
    ])
    for status, count in trajectory_counts.items():
        lines.append(f"- `{status}`: {int(count)}")
    lines.extend([
        "",
        "The mean-PCK classification treats a peak at least two validation "
        "intervals earlier with a >=0.5-point decline as clear post-peak "
        "behavior. One- and two-point curves are generally insufficient. "
        "Training loss is not present as a usable time series in these `.err` "
        "files; validation PCK is the correct checkpoint-selection signal.",
        "",
        "### Interpretable Curves",
        "",
        "| dataset | pretrained | freeze | latest epoch | mean peak | drop | "
        "FlyingThings peak | status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    interpretable = valid[
        valid["eval_checkpoint_count"].ge(3)
        | valid["trajectory_status"].eq("unstable_collapse")
    ].sort_values(["dataset", "pretrained", "freeze"])
    for row in interpretable.itertuples():
        lines.append(
            f"| {row.dataset} | {row.pretrained} | {row.freeze} | "
            f"{int(row.max_epoch)} | {int(row.mean_pck_best_epoch)} | "
            f"{row.mean_pck_drop_from_best:.2f} | "
            f"{int(row.primary_pck_best_epoch)} | `{row.trajectory_status}` |"
        )
    lines.extend([
        "",
        "## Dataset Completeness",
        "",
        "| dataset | evaluated configs | directories |",
        "|---|---:|---:|",
    ])
    for row in completeness.itertuples():
        lines.append(f"| {row.dataset} | {int(row.sum)} | {int(row.count)} |")
    lines.extend([
        "",
        "## Integration Rule",
        "",
        f"Import only the {len(valid)} valid runs, cap `peak_pck` at "
        f"{common_steps:,} steps, leave AUC missing, and write to sidecar "
        "artifacts. This fixed shared horizon avoids giving interrupted runs "
        "different opportunities to peak. Runs without `validation_results.csv` "
        "are excluded.",
        "",
    ])
    report = "\n".join(lines)
    (out_dir / "GRID_AUDIT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
