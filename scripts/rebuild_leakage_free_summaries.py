#!/usr/bin/env python3
"""
Rebuild summary_report.txt files for leakage-free runs using run_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional


def _load_metadata(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _find_run_dirs(root: Path) -> List[Path]:
    dirs = []
    for auc_path in root.rglob("auc_with_features.csv"):
        run_dir = auc_path.parent
        dirs.append(run_dir)
    return sorted(set(dirs))


def _build_command(run_dir: Path, meta: Dict) -> List[str]:
    predictors = meta.get("predictors") or []
    predictors_str = ",".join(predictors) if isinstance(predictors, list) else str(predictors)
    cmd = [
        "python",
        "scripts/summarize_leakage_free_results.py",
        "--output-dir",
        str(run_dir),
        "--output-file",
        str(run_dir / "summary_report.txt"),
        "--auc-table",
        str(run_dir / "auc_with_features.csv"),
        "--lobo-summary",
        str(run_dir / "prediction_lobo_summary.csv"),
        "--lobo-rank-summary",
        str(run_dir / "prediction_lobo_rank_summary.csv"),
        "--lobo-rank-baselines",
        str(run_dir / "prediction_lobo_rank_baselines.csv"),
        "--loto-summary",
        str(run_dir / "prediction_loto_summary.csv"),
        "--loto-rank-summary",
        str(run_dir / "prediction_loto_rank_summary.csv"),
        "--within-benchmark-slopes",
        str(run_dir / "within_benchmark_slopes.csv"),
        "--within-benchmark-slopes-univariate",
        str(run_dir / "within_benchmark_slopes_univariate.csv"),
        "--within-train-dataset-slopes-univariate",
        str(run_dir / "within_train_dataset_slopes_univariate.csv"),
        "--target",
        str(meta.get("target") or "auc_normalized_observed"),
        "--predictors",
        predictors_str,
        "--linear-model",
        str(meta.get("linear_model") or meta.get("model") or "ols"),
        "--prediction-model",
        str(meta.get("prediction_model") or meta.get("model") or "ols"),
        "--ridge-alpha",
        str(meta.get("ridge_alpha", 1.0)),
        "--standardize",
        str(meta.get("standardize", True)),
    ]
    prediction_target = meta.get("prediction_target")
    if prediction_target:
        cmd += ["--prediction-target", str(prediction_target)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild summary reports for leakage-free runs.")
    parser.add_argument("--root", required=True, help="Root directory to scan.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if summary_report.txt exists.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    run_dirs = _find_run_dirs(root)
    if not run_dirs:
        raise SystemExit(f"No run directories found under: {root}")

    for run_dir in run_dirs:
        summary_path = run_dir / "summary_report.txt"
        if summary_path.exists() and not args.force:
            continue
        meta = _load_metadata(run_dir / "run_metadata.json") or {}
        cmd = _build_command(run_dir, meta)
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"Failed summary for {run_dir}: {exc}")


if __name__ == "__main__":
    main()
