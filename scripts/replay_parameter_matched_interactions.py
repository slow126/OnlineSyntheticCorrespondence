#!/usr/bin/env python3
"""
Replay parameter-matched selections with interaction-augmented predictors.

This script uses an existing parameter_matched_selection.csv as source-of-truth
for bucket ids and base predictor sets. It reruns build_leakage_free_eval.py per
bucket and writes outputs to:
  <output_root>/density_joint/leakage_free_<bucket><run_suffix>
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DEFAULT_SNAPSHOTS = [
    "/mnt/nvme_1tb_b/snapshots_ptody_fix",
    "/mnt/nvme_1tb_b/snapshots_synth_2d",
    "/mnt/nvme_1tb_b/snapshots_synthetic_long",
    "./snapshots_2d_warps",
    "/home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_mixed",
    "/home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft",
    "/home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_raft_2d_mix",
    "/home/spencer/Projects/OnlineSyntheticCorrespondence/snapshots_spair_only",
]


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _pick_flow_csv(predictors_csv: str, *, input_root: Path, baseline_root: Path | None) -> str:
    preds = predictors_csv or ""
    density_root = (baseline_root / "density_joint") if baseline_root else None

    def _density_or_fallback(name: str, fallback: Path) -> str:
        if density_root:
            candidate = density_root / name
            if candidate.exists():
                return str(candidate)
        return str(fallback)

    if "flow_train_to_eval_auc" in preds or "flow_eval_to_train_auc" in preds:
        return _density_or_fallback(
            "coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv",
            input_root / "coverage_v2_flow_only_raw_joint_curve_summary_q90_95.csv",
        )
    if "flow_train_to_eval_eps_at50" in preds or "flow_eval_to_train_eps_at50" in preds:
        return _density_or_fallback(
            "coverage_v2_flow_only_raw_joint_curve_summary_q50.csv",
            input_root / "coverage_v2_flow_only_raw_joint_curve_summary_q50.csv",
        )
    if "mean_dist_over_radius" in preds:
        return str(input_root / "coverage_v2_flow_only_raw_joint_kmeans_manifold_full.csv")
    if "_weighted" in preds:
        return str(input_root / "coverage_v2_flow_only_raw_joint_kmeans_full.csv")
    return str(input_root / "coverage_v2_flow_only_raw_joint_full.csv")


def _require_path_exists(path: str, *, label: str) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"{label} not found: {p}\n"
            "Check --baseline-root/--input-root resolution or provide an explicit CSV path."
        )


def _infer_baseline_root(selection_csv: Path) -> Path | None:
    # Typical layout:
    #   <baseline_root>/paper_plots_.../parameter_matched_selection.csv
    candidate = selection_csv.resolve().parents[1]
    if (candidate / "density_joint").exists():
        return candidate
    return None


def _dynamic_custom_interactions(
    predictors_csv: str,
    controls: Iterable[str],
) -> str:
    preds = _parse_csv_list(predictors_csv)
    controls = [c.strip() for c in controls if str(c).strip()]
    bases = [
        p
        for p in preds
        if (not p.endswith("_mmd")) and (not p.startswith("log_"))
    ]
    tokens = [f"{b}*{c}" for b in bases for c in controls]
    return ",".join(tokens)


def _default_density_controls(mode: str) -> List[str]:
    mode = str(mode).strip().lower()
    if mode == "samples_only":
        return ["log_n_samples_eval", "log_n_samples_train"]
    return [
        "log_n_samples_eval",
        "log_avg_flows_eval",
        "log_n_samples_train",
        "log_avg_flows_train",
    ]


def _resolve_hof_csv(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    fallback = Path("analysis/coverage_v2_hof_full_occ.csv")
    if fallback.exists():
        return str(fallback)
    return str(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay parameter-matched selection with interactions.")
    parser.add_argument("--selection-csv", required=True, help="Source parameter_matched_selection.csv")
    parser.add_argument("--output-root", required=True, help="Output sweep root")
    parser.add_argument(
        "--baseline-root",
        default="",
        help=(
            "Optional baseline run root used to resolve coverage/MMD CSVs "
            "(defaults to parent of selection-csv when available)."
        ),
    )
    parser.add_argument("--input-root", default="analysis", help="Input root for coverage CSV fallbacks.")
    parser.add_argument(
        "--run-suffix",
        default="__density_as_interactions",
        help="Suffix appended to leakage_free_<bucket>",
    )
    parser.add_argument(
        "--snapshots-dirs",
        default=",".join(DEFAULT_SNAPSHOTS),
        help="CSV list of snapshots dirs.",
    )
    parser.add_argument("--coverage-dino-csv", default="")
    parser.add_argument("--coverage-hof-csv", default="")
    parser.add_argument("--flow-mmd-csv", default="")
    parser.add_argument("--feature-mmd-csv", default="")
    parser.add_argument("--dino-mmd-csv", default="")
    parser.add_argument("--flow-stats-dir", default="/mnt/nvme_1tb_b/coverage_vectors/stats")
    parser.add_argument(
        "--cv-residual-target-transform",
        choices=["zscore", "residual"],
        default="zscore",
    )
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument(
        "--density-controls-mode",
        choices=["all", "samples_only"],
        default="all",
        help=(
            "Preset for density controls used in replay. "
            "'all' uses n_samples + avg_flows controls; "
            "'samples_only' uses only log_n_samples_{eval,train}."
        ),
    )
    parser.add_argument(
        "--dynamic-controls",
        default="",
        help=(
            "Controls used for dynamic interaction generation. "
            "If empty, derived from --density-controls-mode."
        ),
    )
    parser.add_argument(
        "--density-main-controls",
        default="",
        help=(
            "CSV density controls to include as direct main predictors in every replayed run. "
            "If empty, derived from --density-controls-mode. "
            "Set to explicit empty string to disable by passing --density-main-controls \" \"."
        ),
    )
    parser.add_argument(
        "--custom-interactions",
        default="",
        help="Override dynamic interactions with explicit CSV string.",
    )
    parser.add_argument(
        "--bucket-filter",
        default="",
        help="Optional CSV of bucket substrings to keep.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selection_csv = Path(args.selection_csv)
    if not selection_csv.exists():
        raise FileNotFoundError(f"selection csv not found: {selection_csv}")
    out_root = Path(args.output_root)
    density_root = out_root / "density_joint"
    density_root.mkdir(parents=True, exist_ok=True)

    input_root = Path(args.input_root)
    baseline_root = Path(args.baseline_root) if args.baseline_root else _infer_baseline_root(selection_csv)

    if baseline_root and not baseline_root.exists():
        raise FileNotFoundError(f"baseline root not found: {baseline_root}")

    default_dino_cov = (
        (baseline_root / "dino_coverage_rnorm_k5.csv")
        if baseline_root and (baseline_root / "dino_coverage_rnorm_k5.csv").exists()
        else (input_root / "coverage_v2_dino_full_fast.csv")
    )
    default_hof_cov = (
        (baseline_root / "hof_coverage_rnorm_k5.csv")
        if baseline_root and (baseline_root / "hof_coverage_rnorm_k5.csv").exists()
        else (input_root / "coverage_v2_hof_full.csv")
    )
    default_flow_mmd = (
        (baseline_root / "mmd" / "mmd_v2_flow_joint_v1.csv")
        if baseline_root and (baseline_root / "mmd" / "mmd_v2_flow_joint_v1.csv").exists()
        else Path("flow_mmd_results_fast.csv")
    )
    default_dino_mmd = (
        (baseline_root / "mmd" / "mmd_v2_dino_v1.csv")
        if baseline_root and (baseline_root / "mmd" / "mmd_v2_dino_v1.csv").exists()
        else Path("dino_mmd_results_fast.csv")
    )
    default_feature_mmd = (
        (baseline_root / "mmd" / "mmd_v2_feature_v1.csv")
        if baseline_root and (baseline_root / "mmd" / "mmd_v2_feature_v1.csv").exists()
        else (
            (baseline_root / "mmd" / "mmd_v2_dino_v1.csv")
            if baseline_root and (baseline_root / "mmd" / "mmd_v2_dino_v1.csv").exists()
            else Path("feature_mmd_results_fast.csv")
        )
    )

    coverage_dino_csv = args.coverage_dino_csv.strip() or str(default_dino_cov)
    coverage_hof_csv = args.coverage_hof_csv.strip() or str(default_hof_cov)
    flow_mmd_csv = args.flow_mmd_csv.strip() or str(default_flow_mmd)
    feature_mmd_csv = args.feature_mmd_csv.strip() or str(default_feature_mmd)
    dino_mmd_csv = args.dino_mmd_csv.strip() or str(default_dino_mmd)

    hof_csv = _resolve_hof_csv(coverage_hof_csv)
    snapshots = _parse_csv_list(args.snapshots_dirs)
    default_controls = _default_density_controls(args.density_controls_mode)
    controls = _parse_csv_list(args.dynamic_controls)
    if not controls and not args.dynamic_controls.strip():
        controls = list(default_controls)
    density_main_controls = _parse_csv_list(args.density_main_controls)
    if not density_main_controls and not args.density_main_controls.strip():
        density_main_controls = list(default_controls)
    filters = _parse_csv_list(args.bucket_filter)

    selected = pd.read_csv(selection_csv)
    required_cols = {"bucket", "predictors"}
    if not required_cols.issubset(set(selected.columns)):
        missing = sorted(required_cols - set(selected.columns))
        raise ValueError(f"selection csv missing required columns: {missing}")

    if filters:
        mask = pd.Series(False, index=selected.index)
        bucket_col = selected["bucket"].astype(str)
        for f in filters:
            mask = mask | bucket_col.str.contains(f, regex=False)
        selected = selected[mask].copy()

    if selected.empty:
        raise SystemExit("No rows selected after filters.")

    manifest_rows = []
    for _, row in selected.iterrows():
        bucket = str(row["bucket"])
        predictors_csv = str(row["predictors"])
        base_predictors = _parse_csv_list(predictors_csv)
        augmented_predictors = list(dict.fromkeys(base_predictors + density_main_controls))
        augmented_predictors_csv = ",".join(augmented_predictors)
        flow_csv = _pick_flow_csv(predictors_csv, input_root=input_root, baseline_root=baseline_root)
        _require_path_exists(
            flow_csv,
            label=f"flow coverage CSV for bucket '{bucket}'",
        )
        custom = args.custom_interactions.strip() or _dynamic_custom_interactions(predictors_csv, controls)

        out_dir = density_root / f"leakage_free_{bucket}{args.run_suffix}"
        cmd = [
            "python",
            "scripts/build_leakage_free_eval.py",
            "--snapshots-dir",
            *snapshots,
            "--output-dir",
            str(out_dir),
            "--coverage-csv",
            flow_csv,
            "--coverage-dino-csv",
            coverage_dino_csv,
            "--coverage-hof-csv",
            hof_csv,
            "--flow-mmd-csv",
            flow_mmd_csv,
            "--feature-mmd-csv",
            feature_mmd_csv,
            "--dino-mmd-csv",
            dino_mmd_csv,
            "--flow-stats-dir",
            args.flow_stats_dir,
            "--target",
            "auc_normalized_observed",
            "--predictors",
            augmented_predictors_csv,
            "--linear-model",
            "ridge",
            "--prediction-model",
            "ridge",
            "--ridge-alpha",
            str(args.ridge_alpha),
            "--ranking-group",
            "train_dataset",
            "--ranking-context-cols",
            "model_family_encoder",
            "--pairwise-group-cols",
            "benchmark,model_family_encoder",
            "--cv-residualize-target-by-context",
            "--cv-residual-context-cols",
            "benchmark,model_family_encoder",
            "--cv-residual-target-transform",
            args.cv_residual_target_transform,
            "--cv-residual-eval-space",
            "residual",
            "--cv-repeat-aggregation",
            "median",
            "--fit-sample-weighting",
            "inverse_task",
            "--fit-balance-real-synth",
            "--overall-aggregation",
            "macro_fold",
            "--joint-ood-holdout",
            "--no-loto-single-predictor-baselines",
            "--no-jointood-single-predictor-baselines",
            "--no-per-encoder",
            "--no-encoder-main-effects",
            "--no-encoder-interactions",
            "--no-model-family-main-effects",
            "--no-model-family-interactions",
            "--no-use-flow-density-predictors",
            "--no-flow-eps-predictors",
            "--no-flow-eps-weighted-predictors",
            "--no-flow-density-interactions",
            "--custom-interactions",
            custom,
            "--strict-dataset-match",
            "--no-allow-unsplit-coverage",
            "--no-allow-unsplit-flow-stats",
            "--no-allow-unsplit-mmd",
            "--no-logit-coverage",
            "--no-collapse-cv-cells",
            "--no-prediction-mixedlm",
            "--no-regression-mixedlm",
        ]

        print(f"[bucket {bucket}] -> {out_dir}")
        if args.dry_run:
            print(" ".join(cmd))
            status = "dry_run"
        else:
            subprocess.run(cmd, check=True)
            status = "ok"

        manifest_rows.append(
            {
                "bucket": bucket,
                "base_predictors": predictors_csv,
                "augmented_predictors": augmented_predictors_csv,
                "run_dir": str(out_dir),
                "flow_csv": flow_csv,
                "custom_interactions": custom,
                "status": status,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = out_root / "interaction_replay_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"Wrote: {manifest_path}")

    readme = out_root / "README_interaction_replay.txt"
    readme.write_text(
        "\n".join(
            [
                "Interaction replay from fixed parameter-matched selection",
                "",
                f"selection_csv: {selection_csv}",
                f"output_root: {out_root}",
                f"baseline_root: {baseline_root}",
                f"input_root: {input_root}",
                f"run_suffix: {args.run_suffix}",
                f"cv_residual_target_transform: {args.cv_residual_target_transform}",
                f"ridge_alpha: {args.ridge_alpha}",
                f"coverage_dino_csv: {coverage_dino_csv}",
                f"coverage_hof_csv: {hof_csv}",
                f"flow_mmd_csv: {flow_mmd_csv}",
                f"feature_mmd_csv: {feature_mmd_csv}",
                f"dino_mmd_csv: {dino_mmd_csv}",
                f"density_controls_mode: {args.density_controls_mode}",
                f"dynamic_controls: {','.join(controls)}",
                f"density_main_controls: {','.join(density_main_controls)}",
                f"custom_interactions_override: {bool(args.custom_interactions.strip())}",
                f"manifest: {manifest_path}",
            ]
        )
        + "\n"
    )
    print(f"Wrote: {readme}")


if __name__ == "__main__":
    main()
