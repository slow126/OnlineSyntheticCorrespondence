#!/usr/bin/env python3
"""
Build focused comparison tables from method_summary.csv.

Outputs a small set of CSV tables that align with the asymmetry and
motion-vs-appearance hypotheses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd


BASE_COLS = [
    "method",
    "family",
    "symmetry",
    "notes",
    "target",
    "prediction_target",
    "predictors",
    "n_predictors",
    "n_predictors_base",
    "n_predictors_encoder_main_effects",
    "n_predictors_model_family_main_effects",
    "n_predictors_encoder_interactions",
    "n_predictors_model_family_interactions",
    "model",
]

METRIC_COLS = [
    "lobo_mae",
    "lobo_rmse",
    "lobo_spearman",
    "loto_mae",
    "loto_rmse",
    "loto_spearman",
    "lobo_regret",
    "lobo_rank_spearman",
    "loto_regret",
    "loto_rank_spearman",
]


def _pick_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in BASE_COLS + METRIC_COLS if c in df.columns]
    return cols if cols else list(df.columns)


def _select_methods(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    names = list(names)
    if not names or "method" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    mask = df["method"].isin(names)
    return df[mask].copy()


def _select_prefixes(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    prefixes = list(prefixes)
    if not prefixes or "method" not in df.columns:
        return pd.DataFrame(columns=df.columns)
    mask = df["method"].astype(str).str.startswith(tuple(prefixes))
    return df[mask].copy()


def _write_table(df: pd.DataFrame, out_path: Path, sort_col: str | None = None) -> None:
    if df.empty:
        return
    if sort_col and sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[_pick_cols(df)].to_csv(out_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hypothesis comparison tables.")
    parser.add_argument("--summary", required=True, help="Path to method_summary.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for tables")
    parser.add_argument(
        "--sort-metric",
        default="loto_spearman",
        help="Metric to sort tables by when present.",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise SystemExit(f"Missing method summary: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"No rows in method summary: {summary_path}")

    out_dir = Path(args.output_dir)

    # Primary asymmetry check for HOF motion.
    hof_primary = _select_methods(
        df,
        [
            "hof_motion_k1",
            "hof_motion_k1_eval_only",
            "hof_motion_k1_train_only",
            "hof_motion_k1_plus_density_l2",
            "hof_density_l2",
        ],
    )
    _write_table(hof_primary, out_dir / "asymmetry_hof_primary.csv", args.sort_metric)

    # Flow asymmetry (eps ladder / summary variants).
    flow_asym = _select_prefixes(
        df,
        [
            "flow_eps_raw_joint",
            "flow_eps_raw_joint_eval_only",
            "flow_eps_raw_joint_train_only",
            "flow_eps_raw_joint_eps_at50",
            "flow_eps_raw_joint_auc_at95",
            "flow_kmeans_weighted_all",
            "flow_kmeans_manifold",
            "flow_kl_k",
        ],
    )
    _write_table(flow_asym, out_dir / "asymmetry_flow.csv", args.sort_metric)

    # DINO asymmetry.
    dino_asym = _select_prefixes(
        df,
        [
            "dino_rnorm_k5",
            "dino_rnorm_k5_eval_only",
            "dino_rnorm_k5_train_only",
            "dino_kl_k",
        ],
    )
    _write_table(dino_asym, out_dir / "asymmetry_dino.csv", args.sort_metric)

    # Symmetric vs asymmetric baselines.
    mmd_vs_asym = _select_methods(df, ["mmd_only", "asym_and_mmd"])
    _write_table(mmd_vs_asym, out_dir / "mmd_vs_asym.csv", args.sort_metric)

    # Motion vs appearance overview.
    motion_vs_app = pd.concat(
        [
            _select_prefixes(df, ["hof_", "flow_"]),
            _select_prefixes(df, ["dino_"]),
        ],
        ignore_index=True,
    )
    if not motion_vs_app.empty:
        # Drop pairwise variants for clarity.
        if "method" in motion_vs_app.columns:
            motion_vs_app = motion_vs_app[
                ~motion_vs_app["method"].astype(str).str.endswith("_pairwise")
            ]
        _write_table(motion_vs_app, out_dir / "motion_vs_appearance.csv", args.sort_metric)


if __name__ == "__main__":
    main()
