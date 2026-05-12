#!/usr/bin/env python3
"""
Rebuild LOBO/LOTO ranking summaries using a different ranking group,
without re-running the full sweep.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import build_leakage_free_eval as eval_utils


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except Exception:
        return None


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Some legacy row exports can contain duplicate column names (e.g., benchmark),
    # which breaks groupby with "not 1-dimensional".
    if df.columns.duplicated().any():
        return df.loc[:, ~df.columns.duplicated()].copy()
    return df


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _ensure_option_col(df: pd.DataFrame, ranking_group: str) -> Tuple[pd.DataFrame, str]:
    if ranking_group in df.columns:
        return df, ranking_group
    if ranking_group == "train_dataset_encoder":
        if "train_dataset" in df.columns and "encoder_config" in df.columns:
            df = df.copy()
            df["train_dataset_encoder"] = (
                df["train_dataset"].astype(str) + "__" + df["encoder_config"].astype(str)
            )
            return df, "train_dataset_encoder"
    if ranking_group == "train_dataset_model_family_encoder":
        if "train_dataset" in df.columns and "model_family" in df.columns and "encoder_config" in df.columns:
            df = df.copy()
            df["train_dataset_model_family_encoder"] = (
                df["train_dataset"].astype(str)
                + "__"
                + df["model_family"].astype(str)
                + "_"
                + df["encoder_config"].astype(str)
            )
            return df, "train_dataset_model_family_encoder"
    return df, ranking_group


def _output_paths(run_dir: Path, suffix: str, overwrite: bool) -> Tuple[Path, Path, Path]:
    if overwrite:
        return (
            run_dir / "prediction_lobo_rank_summary.csv",
            run_dir / "prediction_lobo_rank_detail.csv",
            run_dir / "prediction_lobo_rank_baselines.csv",
        )
    return (
        run_dir / f"prediction_lobo_rank_summary.{suffix}.csv",
        run_dir / f"prediction_lobo_rank_detail.{suffix}.csv",
        run_dir / f"prediction_lobo_rank_baselines.{suffix}.csv",
    )


def _output_paths_loto(run_dir: Path, suffix: str, overwrite: bool) -> Tuple[Path, Path]:
    if overwrite:
        return (
            run_dir / "prediction_loto_rank_summary.csv",
            run_dir / "prediction_loto_rank_detail.csv",
        )
    return (
        run_dir / f"prediction_loto_rank_summary.{suffix}.csv",
        run_dir / f"prediction_loto_rank_detail.{suffix}.csv",
    )


def _output_paths_jointood(run_dir: Path, suffix: str, overwrite: bool) -> Tuple[Path, Path]:
    if overwrite:
        return (
            run_dir / "prediction_jointood_rank_summary.csv",
            run_dir / "prediction_jointood_rank_detail.csv",
        )
    return (
        run_dir / f"prediction_jointood_rank_summary.{suffix}.csv",
        run_dir / f"prediction_jointood_rank_detail.{suffix}.csv",
    )


def _parse_csv_cols(value: str) -> list[str]:
    return [c.strip() for c in str(value or "").split(",") if c.strip()]


def rebuild_for_run(
    run_dir: Path,
    ranking_group: str,
    ranking_context_cols: list[str],
    topk_frac: float,
    topk_min: int,
    overwrite: bool,
) -> bool:
    lobo_rows = _load_csv(run_dir / "prediction_lobo_rows.csv")
    loto_rows = _load_csv(run_dir / "prediction_loto_rows.csv")
    jointood_rows = _load_csv(run_dir / "prediction_jointood_rows.csv")
    if lobo_rows is None and loto_rows is None and jointood_rows is None:
        return False

    if lobo_rows is not None:
        lobo_rows = _dedupe_columns(lobo_rows)
    if loto_rows is not None:
        loto_rows = _dedupe_columns(loto_rows)
    if jointood_rows is not None:
        jointood_rows = _dedupe_columns(jointood_rows)

    meta = _load_metadata(run_dir / "run_metadata.json")
    predictors = meta.get("predictors") or []
    use_logit = bool(meta.get("logit_coverage", False))

    suffix = ranking_group.replace("/", "_")

    if lobo_rows is not None:
        lobo_rows, option_col = _ensure_option_col(lobo_rows, ranking_group)
        lobo_summary_path, lobo_detail_path, lobo_baseline_path = _output_paths(
            run_dir, suffix, overwrite
        )
        eval_utils.compute_ranking_summary(
            lobo_rows,
            "target",
            option_col,
            lobo_summary_path,
            context_cols=ranking_context_cols,
            topk_frac=topk_frac,
            topk_min=topk_min,
        )
        eval_utils.write_rank_detail_rows(
            lobo_rows,
            "target",
            option_col,
            lobo_detail_path,
            context_cols=ranking_context_cols,
        )
        selectors = eval_utils._build_baseline_selectors(
            lobo_rows,
            predictors=predictors,
            use_logit=use_logit,
        )
        eval_utils.compute_baseline_rankings(
            lobo_rows,
            "target",
            option_col,
            lobo_baseline_path,
            selectors,
            context_cols=ranking_context_cols,
            topk_frac=topk_frac,
            topk_min=topk_min,
        )

    if loto_rows is not None:
        loto_rows, option_col = _ensure_option_col(loto_rows, ranking_group)
        loto_summary_path, loto_detail_path = _output_paths_loto(run_dir, suffix, overwrite)
        eval_utils.compute_ranking_summary(
            loto_rows,
            "target",
            option_col,
            loto_summary_path,
            context_cols=ranking_context_cols,
            topk_frac=topk_frac,
            topk_min=topk_min,
        )
        eval_utils.write_rank_detail_rows(
            loto_rows,
            "target",
            option_col,
            loto_detail_path,
            context_cols=ranking_context_cols,
        )

    if jointood_rows is not None:
        jointood_rows, option_col = _ensure_option_col(jointood_rows, ranking_group)
        jointood_summary_path, jointood_detail_path = _output_paths_jointood(run_dir, suffix, overwrite)
        eval_utils.compute_ranking_summary(
            jointood_rows,
            "target",
            option_col,
            jointood_summary_path,
            context_cols=ranking_context_cols,
            topk_frac=topk_frac,
            topk_min=topk_min,
        )
        eval_utils.write_rank_detail_rows(
            jointood_rows,
            "target",
            option_col,
            jointood_detail_path,
            context_cols=ranking_context_cols,
        )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild LOBO/LOTO/Joint-OOD ranking summaries with a new ranking group."
    )
    parser.add_argument("--root", required=True, help="Root directory to scan.")
    parser.add_argument(
        "--ranking-group",
        required=True,
        help="Ranking group column (e.g., train_dataset, train_dataset_encoder, train_dataset_model_family_encoder).",
    )
    parser.add_argument(
        "--ranking-context-cols",
        default="",
        help="Comma-separated context columns for ranking summaries (e.g., model_family_encoder).",
    )
    parser.add_argument("--topk-frac", type=float, default=0.2)
    parser.add_argument("--topk-min", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prediction_*_rank_summary.csv files.",
    )
    args = parser.parse_args()
    ranking_context_cols = _parse_csv_cols(args.ranking_context_cols)

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"Missing root: {root}")

    run_dirs = sorted(
        set(p.parent for p in root.rglob("prediction_lobo_rows.csv"))
        | set(p.parent for p in root.rglob("prediction_loto_rows.csv"))
        | set(p.parent for p in root.rglob("prediction_jointood_rows.csv"))
    )
    if not run_dirs:
        raise SystemExit(f"No runs found under: {root}")

    rebuilt = 0
    total = len(run_dirs)
    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{total}] Rebuilding: {run_dir}")
        if rebuild_for_run(
            run_dir,
            args.ranking_group,
            ranking_context_cols,
            args.topk_frac,
            args.topk_min,
            args.overwrite,
        ):
            rebuilt += 1

    print(f"Rebuilt ranking summaries for {rebuilt} runs using ranking group: {args.ranking_group}")


if __name__ == "__main__":
    main()
