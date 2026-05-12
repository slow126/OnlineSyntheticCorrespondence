#!/usr/bin/env python3
"""
Build final paper-facing summary tables with ranking-first selection.

Outputs:
  1) CatsPP train-vs-eval zero-shot tables split by pretrained True/False.
  2) Tiered ranking tables for LOBO, LOTO, Joint-OOD.
  3) Tier-4 heldout model-family protocol tables from heldout CV outputs.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


RANK_TIER_SPECS: Tuple[Tuple[str, str, str, str], ...] = (
    ("tier1_lobo", "lobo_rank_spearman", "lobo_rank_pairwise_cindex", "lobo_regret"),
    ("tier2_loto", "loto_rank_spearman", "loto_rank_pairwise_cindex", "loto_regret"),
    ("tier3_jointood", "jointood_rank_spearman", "jointood_rank_pairwise_cindex", "jointood_regret"),
)

HELDOUT_PROTOCOL_ORDER = {
    "model_train_benchmark": 1,
    "model_benchmark": 2,
    "model_train_benchmark_disjoint": 3,
    "model_benchmark_trainset_disjoint": 4,
    "model_only": 5,
}

_DATASET_DISPLAY_NAMES: Dict[str, str] = {
    "flyingthings": "FlyingThings",
    "pointodyssey": "PointOdyssey",
    "spair": "SPair",
    "sintel": "Sintel",
    "imagenet2dwarp": "ImageNet 2D Warp",
    "kitti2012": "KITTI-2012",
    "kitti2015": "KITTI-2015",
    "middlebury": "Middlebury",
    "pfpascal": "PF-PASCAL",
    "pfwillow": "PF-WILLOW",
    "tss": "TSS",
}

_SYNTHETIC_VARIANT_DISPLAY_NAMES: Dict[str, str] = {
    "2d_warp": "2D Warp",
    "small_zoom": "Small Zoom",
    "large_zoom": "Zoom",
    "random_flipping": "Random Flipping",
}

_TRAIN_DATASET_TYPE_LABELS: Dict[str, str] = {
    "sintel": "Flow",
    "flyingthings": "Flow",
    "spair": "Semantic",
    "imagenet2dwarp": "Flow",
    "pointodyssey": "Tracking",
    "synthetic": "Flow",
    "synthetic_2d_warp": "Flow",
    "synthetic_large_zoom": "Flow",
    "synthetic_small_zoom": "Flow",
    "synthetic_random_flipping": "Flow",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return pd.read_csv(path)


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _split_csv_arg(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _coerce_bool(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series.astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    )
    return mapped


def _parse_bool_text(value: str, default: bool) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    return bool(default)


def _write_csv(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _nonempty_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    keep: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        series = df[c]
        if series.notna().any():
            keep.append(c)
    return keep


def _rename_cols(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    rename = {k: v for k, v in mapping.items() if k in out.columns}
    if rename:
        out = out.rename(columns=rename)
    return out


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _titleize_identifier(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return raw
    parts = re.split(r"[_\s]+", raw)
    token_map = {
        "2d": "2D",
        "3d": "3D",
        "sdf": "SDF",
        "mmd": "MMD",
    }
    out: List[str] = []
    for part in parts:
        if not part:
            continue
        low = part.lower()
        if low in token_map:
            out.append(token_map[low])
        elif part.isupper():
            out.append(part)
        else:
            out.append(part[:1].upper() + part[1:])
    return " ".join(out)


def _format_dataset_display_name(raw_value: str, synthetic_label: str) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return raw
    if raw == "synthetic":
        return synthetic_label
    if raw.startswith("synthetic_"):
        suffix = raw[len("synthetic_") :]
        suffix_label = _SYNTHETIC_VARIANT_DISPLAY_NAMES.get(suffix, _titleize_identifier(suffix))
        return f"{synthetic_label} ({suffix_label})"
    match = re.fullmatch(r"([a-z0-9]+)_synthetic_(\d{1,3})_(\d{1,3})", raw)
    if match:
        base_raw = match.group(1)
        pct_a = int(match.group(2))
        pct_b = int(match.group(3))
        base_label = _DATASET_DISPLAY_NAMES.get(base_raw, _titleize_identifier(base_raw))
        return f"{base_label}/{synthetic_label} ({pct_a}%/{pct_b}% mix)"
    if raw in _DATASET_DISPLAY_NAMES:
        return _DATASET_DISPLAY_NAMES[raw]
    return _titleize_identifier(raw)


def _infer_train_dataset_type(raw_value: str) -> str:
    raw = str(raw_value or "").strip()
    if not raw:
        return "--"
    if raw in _TRAIN_DATASET_TYPE_LABELS:
        return _TRAIN_DATASET_TYPE_LABELS[raw]
    if raw.startswith("synthetic_"):
        return "Flow"
    if "_synthetic_" in raw:
        return "Mixed"
    return "--"


def _resolve_summary_paths(summary: str, method_summaries: str, run_roots: str) -> List[Path]:
    out: List[Path] = []
    if summary.strip():
        out.append(Path(summary))
    for item in _split_csv_arg(method_summaries):
        out.append(Path(item))
    for root in _split_csv_arg(run_roots):
        out.append(Path(root) / "method_summary.csv")
    dedup: List[Path] = []
    seen = set()
    for p in out:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        dedup.append(p)
    return dedup


def _load_method_summaries(
    paths: Sequence[Path],
    target: str,
    allowed_models: Sequence[str],
) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames: List[pd.DataFrame] = []
    for p in paths:
        df = _read_csv(p)
        if df.empty:
            continue
        work = df.copy()
        work["source_summary"] = str(p)
        work["source_run"] = p.parent.name
        if "target" in work.columns:
            work = work[work["target"].astype(str) == target]
        if "model" in work.columns and allowed_models:
            work = work[work["model"].astype(str).isin([str(m) for m in allowed_models])]
        frames.append(work)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    if "method" in merged.columns:
        merged["method"] = merged["method"].astype(str)
    return merged


def _dedupe_method_candidates(
    df: pd.DataFrame,
    primary: str,
    tie: str,
    lower_is_better_col: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    for col in (primary, tie, lower_is_better_col):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if primary not in work.columns:
        return pd.DataFrame(columns=work.columns)
    work = work[work[primary].notna()].copy()
    if work.empty:
        return work
    for col in ("predictors", "n_predictors_base", "n_predictors", "family", "model", "method"):
        if col not in work.columns:
            work[col] = ""
    sort_cols = [primary]
    ascending = [False]
    if tie in work.columns:
        sort_cols.append(tie)
        ascending.append(False)
    if lower_is_better_col in work.columns:
        sort_cols.append(lower_is_better_col)
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending)
    key_cols = ["model", "method", "predictors", "n_predictors_base", "target"]
    key_cols = [c for c in key_cols if c in work.columns]
    if key_cols:
        work = work.drop_duplicates(subset=key_cols, keep="first")
    return work


def _build_rank_tier_tables(
    method_df: pd.DataFrame,
    out_dir: Path,
    top_overall: int,
    top_per_family: int,
) -> List[Tuple[str, str]]:
    produced: List[Tuple[str, str]] = []
    if method_df.empty:
        return produced

    for tier_name, primary, tie, low_col in RANK_TIER_SPECS:
        if primary not in method_df.columns:
            continue
        ranked = _dedupe_method_candidates(method_df, primary=primary, tie=tie, lower_is_better_col=low_col)
        if ranked.empty:
            continue

        wanted = [
            "source_run",
            "model",
            "family",
            "symmetry",
            "method",
            "n_predictors_base",
            "n_predictors",
            primary,
            tie,
            low_col,
            "lobo_rank_spearman",
            "lobo_rank_pairwise_cindex",
            "loto_rank_spearman",
            "loto_rank_pairwise_cindex",
            "jointood_rank_spearman",
            "jointood_rank_pairwise_cindex",
            "jointood_rank_pct_err",
        ]
        cols: List[str] = []
        seen = set()
        for c in wanted:
            if c in ranked.columns and c not in seen:
                cols.append(c)
                seen.add(c)

        top_all = ranked.head(top_overall)[cols].reset_index(drop=True)
        _write_csv(top_all, out_dir / f"{tier_name}_top_overall.csv")
        produced.append((tier_name, f"{tier_name}_top_overall.csv"))

        cross_metric_candidates = [
            "lobo_rank_spearman",
            "lobo_rank_pairwise_cindex",
            "loto_rank_spearman",
            "loto_rank_pairwise_cindex",
            "jointood_rank_spearman",
            "jointood_rank_pairwise_cindex",
            "jointood_rank_pct_err",
        ]
        cross_metric_candidates = [c for c in cross_metric_candidates if c not in {primary, tie, low_col}]
        paper_metric_cols = _dedupe_keep_order(_nonempty_cols(
            top_all,
            [primary, tie, low_col] + cross_metric_candidates,
        ))
        paper_cols = [c for c in ("model", "family", "symmetry", "method", "n_predictors") if c in top_all.columns]
        paper_cols += [c for c in paper_metric_cols if c not in paper_cols]
        if paper_cols:
            top_all_paper = top_all[paper_cols].copy()
            top_all_paper = _rename_cols(
                top_all_paper,
                {
                    primary: "primary_rank_spearman",
                    tie: "primary_pairwise_cindex",
                    low_col: "primary_regret",
                    "lobo_rank_spearman": "lobo_spearman",
                    "lobo_rank_pairwise_cindex": "lobo_cindex",
                    "loto_rank_spearman": "loto_spearman",
                    "loto_rank_pairwise_cindex": "loto_cindex",
                    "jointood_rank_spearman": "jointood_spearman",
                    "jointood_rank_pairwise_cindex": "jointood_cindex",
                },
            )
            paper_name = f"{tier_name}_top_overall_paper.csv"
            _write_csv(top_all_paper, out_dir / paper_name)
            produced.append((tier_name, paper_name))

        if "family" in ranked.columns:
            top_family = ranked.groupby("family", dropna=False).head(top_per_family)
            top_family = top_family[cols].reset_index(drop=True)
            _write_csv(top_family, out_dir / f"{tier_name}_top_by_family.csv")
            produced.append((tier_name, f"{tier_name}_top_by_family.csv"))

            fam_metric_cols = _dedupe_keep_order(_nonempty_cols(
                top_family,
                [primary, tie, low_col] + cross_metric_candidates,
            ))
            fam_paper_cols = [c for c in ("model", "family", "symmetry", "method", "n_predictors") if c in top_family.columns]
            fam_paper_cols += [c for c in fam_metric_cols if c not in fam_paper_cols]
            if fam_paper_cols:
                top_family_paper = top_family[fam_paper_cols].copy()
                top_family_paper = _rename_cols(
                    top_family_paper,
                    {
                        primary: "primary_rank_spearman",
                        tie: "primary_pairwise_cindex",
                        low_col: "primary_regret",
                        "lobo_rank_spearman": "lobo_spearman",
                        "lobo_rank_pairwise_cindex": "lobo_cindex",
                        "loto_rank_spearman": "loto_spearman",
                        "loto_rank_pairwise_cindex": "loto_cindex",
                        "jointood_rank_spearman": "jointood_spearman",
                        "jointood_rank_pairwise_cindex": "jointood_cindex",
                    },
                )
                paper_name = f"{tier_name}_top_by_family_paper.csv"
                _write_csv(top_family_paper, out_dir / paper_name)
                produced.append((tier_name, paper_name))

    return produced


def _pivot_pretrained(
    grouped: pd.DataFrame,
    value_col: str,
    index_cols: Sequence[str],
    label: str,
) -> pd.DataFrame:
    piv = grouped.pivot_table(index=list(index_cols), columns="pretrained_bool", values=value_col, aggfunc="first")
    piv = piv.rename(columns={False: f"{label}_pretrained_false", True: f"{label}_pretrained_true"}).reset_index()
    return piv


def _build_catspp_zero_shot_tables(
    auc_path: Path,
    out_dir: Path,
    score_col: str,
    peak_col: str,
    catspp_pretrained: bool,
    catspp_freeze: bool,
    raft_pretrained: bool,
    raft_freeze: bool,
    base_train_datasets: Sequence[str],
    include_synthetic_only: bool,
    synthetic_train_datasets: Sequence[str],
    eval_benchmarks: Sequence[str],
    pair_precision: int,
    cell_mode: str,
    pretty_dataset_labels: bool,
    synthetic_label: str,
    aggregate_all_model_encoders: bool,
    add_size_columns: bool,
    size_column_mode: str,
    add_dataset_type_column: bool,
    size_precision: int,
) -> List[str]:
    produced: List[str] = []
    df = _read_csv(auc_path)
    needed = {"model_family", "train_dataset", "benchmark", "pretrained", "freeze", score_col, peak_col}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns in {auc_path}: {', '.join(missing)}")

    work = df.copy()
    work["pretrained_bool"] = _coerce_bool(work["pretrained"])
    work["freeze_bool"] = _coerce_bool(work["freeze"])
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work[peak_col] = pd.to_numeric(work[peak_col], errors="coerce")
    work = work[work["pretrained_bool"].notna() & work["freeze_bool"].notna()].copy()

    train_keep = {x.strip() for x in base_train_datasets if str(x).strip()}
    synth_keep = {x.strip() for x in synthetic_train_datasets if str(x).strip()}

    def _is_synthetic_only_name(name: str) -> bool:
        # Keep pure synthetic variants (synthetic, synthetic_large_zoom, etc.)
        # Exclude mixed datasets like pointodyssey_synthetic_30_70.
        n = str(name)
        return n.startswith("synthetic") and "_synthetic_" not in n

    all_train_names = set(work["train_dataset"].astype(str).unique())
    if include_synthetic_only:
        synth_auto = {n for n in all_train_names if _is_synthetic_only_name(n)}
    else:
        synth_auto = set()
    keep_train = train_keep.union(synth_keep).union(synth_auto)
    if keep_train:
        work = work[work["train_dataset"].astype(str).isin(keep_train)].copy()

    if eval_benchmarks:
        eval_set = {x.strip() for x in eval_benchmarks if str(x).strip()}
        work = work[work["benchmark"].astype(str).isin(eval_set)].copy()

    if work.empty:
        return produced

    def _best_cells(block: pd.DataFrame, model_label: str) -> pd.DataFrame:
        tmp = block.copy()
        tmp = tmp[tmp[score_col].notna() & tmp[peak_col].notna()].copy()
        if tmp.empty:
            return pd.DataFrame(columns=["train_dataset", "benchmark", "auc_obs_norm", "peak_pck", "n_runs", "model_slice"])
        tmp = tmp.sort_values([score_col], ascending=[False])
        best = tmp.groupby(["train_dataset", "benchmark"], dropna=False).head(1).copy()
        counts = (
            tmp.groupby(["train_dataset", "benchmark"], dropna=False)
            .size()
            .rename("n_runs")
            .reset_index()
        )
        best = best.merge(counts, on=["train_dataset", "benchmark"], how="left")
        best = best.rename(columns={score_col: "auc_obs_norm", peak_col: "peak_pck"})
        best["model_slice"] = model_label
        return best[["train_dataset", "benchmark", "auc_obs_norm", "peak_pck", "n_runs", "model_slice"]]

    def _mean_cells_all_encoders(block: pd.DataFrame, model_label: str) -> pd.DataFrame:
        tmp = block.copy()
        tmp = tmp[tmp[score_col].notna() & tmp[peak_col].notna()].copy()
        if tmp.empty:
            return pd.DataFrame(columns=["train_dataset", "benchmark", "auc_obs_norm", "peak_pck", "n_runs", "n_models", "model_slice"])
        encoder_col = "model_family_encoder" if "model_family_encoder" in tmp.columns else "model_family"
        tmp[encoder_col] = tmp[encoder_col].astype(str)
        enc_means = (
            tmp.groupby(["train_dataset", "benchmark", encoder_col], dropna=False)[[score_col, peak_col]]
            .mean()
            .reset_index()
        )
        agg = (
            enc_means.groupby(["train_dataset", "benchmark"], dropna=False)[[score_col, peak_col]]
            .mean()
            .reset_index()
        )
        n_models = (
            enc_means.groupby(["train_dataset", "benchmark"], dropna=False)[encoder_col]
            .nunique()
            .rename("n_models")
            .reset_index()
        )
        n_runs = (
            tmp.groupby(["train_dataset", "benchmark"], dropna=False)
            .size()
            .rename("n_runs")
            .reset_index()
        )
        agg = agg.merge(n_runs, on=["train_dataset", "benchmark"], how="left")
        agg = agg.merge(n_models, on=["train_dataset", "benchmark"], how="left")
        agg = agg.rename(columns={score_col: "auc_obs_norm", peak_col: "peak_pck"})
        agg["model_slice"] = model_label
        return agg[["train_dataset", "benchmark", "auc_obs_norm", "peak_pck", "n_runs", "n_models", "model_slice"]]

    def _build_train_size_map(block: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
        if "n_samples_train" not in block.columns and "avg_flows_train" not in block.columns:
            return {}
        tmp = block.copy()
        if "n_samples_train" in tmp.columns:
            tmp["n_samples_train"] = pd.to_numeric(tmp["n_samples_train"], errors="coerce")
        if "avg_flows_train" in tmp.columns:
            tmp["avg_flows_train"] = pd.to_numeric(tmp["avg_flows_train"], errors="coerce")
        out: Dict[str, Tuple[float, float]] = {}
        for train, grp in tmp.groupby("train_dataset", dropna=False):
            n_samples = float("nan")
            avg_flows = float("nan")
            if "n_samples_train" in grp.columns and grp["n_samples_train"].notna().any():
                n_samples = float(grp["n_samples_train"].median())
            if "avg_flows_train" in grp.columns and grp["avg_flows_train"].notna().any():
                avg_flows = float(grp["avg_flows_train"].median())
            out[str(train)] = (n_samples, avg_flows)
        return out

    size_map = _build_train_size_map(work)

    cats_label = "catspp_pretrained_true_freeze_false"
    cats = work[
        (work["model_family"].astype(str) == "catspp")
        & (work["pretrained_bool"] == bool(catspp_pretrained))
        & (work["freeze_bool"] == bool(catspp_freeze))
    ].copy()
    raft = work[
        (work["model_family"].astype(str) == "raft")
        & (work["pretrained_bool"] == bool(raft_pretrained))
        & (work["freeze_bool"] == bool(raft_freeze))
    ].copy()
    cats_cells = _best_cells(cats, cats_label)
    raft_cells = _best_cells(raft, "raft")
    all_cells = _mean_cells_all_encoders(work, "all_model_encoders_mean")

    # Detailed selected rows (audit-friendly).
    if aggregate_all_model_encoders:
        detail = all_cells.copy()
    else:
        detail = pd.concat([cats_cells, raft_cells], ignore_index=True)
    detail = detail.sort_values(["model_slice", "train_dataset", "benchmark"]).reset_index(drop=True)
    detail_name = "tier0_representative_selected_cells_long.csv"
    _write_csv(detail, out_dir / detail_name)
    produced.append(detail_name)

    if not aggregate_all_model_encoders:
        # Pairwise compare at matched cells.
        compare = cats_cells.merge(
            raft_cells,
            on=["train_dataset", "benchmark"],
            how="outer",
            suffixes=(f"_{cats_label}", "_raft"),
        )
        compare["delta_auc_obs_norm_catspp_minus_raft"] = (
            pd.to_numeric(compare.get(f"auc_obs_norm_{cats_label}"), errors="coerce")
            - pd.to_numeric(compare.get("auc_obs_norm_raft"), errors="coerce")
        )
        compare["delta_peak_pck_catspp_minus_raft"] = (
            pd.to_numeric(compare.get(f"peak_pck_{cats_label}"), errors="coerce")
            - pd.to_numeric(compare.get("peak_pck_raft"), errors="coerce")
        )
        compare = compare.sort_values(["train_dataset", "benchmark"]).reset_index(drop=True)
        compare_name = "tier0_catspp_vs_raft_cell_compare.csv"
        _write_csv(compare, out_dir / compare_name)
        produced.append(compare_name)

    row_order_seed: List[str] = list(base_train_datasets)
    # Always honor explicitly requested tier0 synthetic rows even if naming is non-synthetic_*.
    row_order_seed += [x for x in synthetic_train_datasets if str(x).strip()]
    row_order_seed += sorted([x for x in detail["train_dataset"].astype(str).unique() if _is_synthetic_only_name(x)])
    seen_row = set()
    row_order_seed = [x for x in row_order_seed if (x not in seen_row and not seen_row.add(x))]

    if eval_benchmarks:
        col_order_seed = list(eval_benchmarks)
    else:
        col_order_seed = sorted(detail["benchmark"].astype(str).unique())

    def _build_pair_grid(
        cells: pd.DataFrame,
        model_name: str,
        row_order_in: Sequence[str],
        col_order_in: Sequence[str],
        size_map_in: Dict[str, Tuple[float, float]],
        add_size_cols: bool,
        size_mode: str,
        add_dataset_type_col: bool,
        include_model_col: bool,
    ) -> pd.DataFrame:
        if cells.empty:
            return pd.DataFrame()
        row_order = list(row_order_in)
        row_order = [x for x in row_order if x in set(cells["train_dataset"].astype(str))]
        if not row_order:
            row_order = sorted(cells["train_dataset"].astype(str).unique())

        col_order = [x for x in col_order_in if x in set(cells["benchmark"].astype(str))]
        if not col_order:
            col_order = sorted(cells["benchmark"].astype(str).unique())

        lookup: Dict[Tuple[str, str], Tuple[float, float]] = {}
        for _, r in cells.iterrows():
            lookup[(str(r["train_dataset"]), str(r["benchmark"]))] = (_as_float(r["peak_pck"]), _as_float(r["auc_obs_norm"]))

        rows: List[Dict[str, object]] = []
        for train in row_order:
            row: Dict[str, object] = {"train_dataset": train}
            if add_dataset_type_col:
                row["dataset_type"] = _infer_train_dataset_type(train)
            if add_size_cols:
                size_vals = size_map_in.get(str(train), (float("nan"), float("nan")))
                train_n = size_vals[0]
                train_flows = size_vals[1]
                if np.isfinite(train_n):
                    row["train_samples"] = f"{train_n:.0f}"
                else:
                    row["train_samples"] = "--"
                if size_mode == "samples_and_flows":
                    if np.isfinite(train_flows):
                        row["train_avg_flows"] = f"{train_flows:.{size_precision}f}"
                    else:
                        row["train_avg_flows"] = "--"
            for bench in col_order:
                val = lookup.get((train, bench))
                if val is None or not np.isfinite(val[0]) or not np.isfinite(val[1]):
                    row[bench] = "--"
                else:
                    if cell_mode == "peak":
                        row[bench] = f"{val[0]:.{pair_precision}f}"
                    else:
                        row[bench] = f"({val[0]:.{pair_precision}f}, {val[1]:.{pair_precision}f})"
            rows.append(row)
        out = pd.DataFrame(rows)
        if add_dataset_type_col and "dataset_type" in out.columns:
            cols = list(out.columns)
            cols.remove("dataset_type")
            train_idx = cols.index("train_dataset")
            cols.insert(train_idx + 1, "dataset_type")
            out = out[cols]
        if include_model_col:
            model_insert_idx = 2 if add_dataset_type_col and "dataset_type" in out.columns else 1
            out.insert(model_insert_idx, "model", model_name)
        if pretty_dataset_labels:
            out["train_dataset"] = out["train_dataset"].map(
                lambda x: _format_dataset_display_name(str(x), synthetic_label=synthetic_label)
            )
            rename_cols: Dict[str, str] = {}
            for bench in col_order:
                rename_cols[bench] = _format_dataset_display_name(bench, synthetic_label=synthetic_label)
            if add_dataset_type_col:
                rename_cols["dataset_type"] = "Dataset Type"
            if add_size_cols:
                rename_cols["train_samples"] = "Train Samples"
                if size_mode == "samples_and_flows":
                    rename_cols["train_avg_flows"] = "Train Avg Flows"
            out = out.rename(columns=rename_cols)
        return out

    if aggregate_all_model_encoders:
        all_pair_grid = _build_pair_grid(
            all_cells,
            "all_model_encoders_mean",
            row_order_seed,
            col_order_seed,
            size_map,
            add_size_columns,
            size_column_mode,
            add_dataset_type_column,
            include_model_col=False,
        )
        all_name = "tier0_all_model_encoders_representative_pair_grid.csv"
        _write_csv(all_pair_grid, out_dir / all_name)
        produced.append(all_name)
        combined_pair = all_pair_grid.copy()
    else:
        cats_pair_grid = _build_pair_grid(
            cats_cells,
            "catspp",
            row_order_seed,
            col_order_seed,
            size_map,
            add_size_columns,
            size_column_mode,
            add_dataset_type_column,
            include_model_col=True,
        )
        raft_pair_grid = _build_pair_grid(
            raft_cells,
            "raft",
            row_order_seed,
            col_order_seed,
            size_map,
            add_size_columns,
            size_column_mode,
            add_dataset_type_column,
            include_model_col=True,
        )
        cats_name = "tier0_catspp_representative_pair_grid.csv"
        raft_name = "tier0_raft_representative_pair_grid.csv"
        _write_csv(cats_pair_grid, out_dir / cats_name)
        _write_csv(raft_pair_grid, out_dir / raft_name)
        produced.extend([cats_name, raft_name])
        combined_pair = pd.concat([cats_pair_grid, raft_pair_grid], ignore_index=True)
    if not combined_pair.empty:
        combined_pair = combined_pair.reset_index(drop=True)
    combined_name = "tier0_representative_pair_grid_combined.csv"
    _write_csv(combined_pair, out_dir / combined_name)
    produced.append(combined_name)

    return produced


def _sort_heldout(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ("rank_spearman", "rank_pairwise_cindex", "rank_regret"):
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["protocol_order"] = work["protocol"].astype(str).map(HELDOUT_PROTOCOL_ORDER).fillna(999).astype(int)
    sort_cols = ["protocol_order", "protocol", "head", "rank_spearman"]
    ascending = [True, True, True, False]
    if "rank_pairwise_cindex" in work.columns:
        sort_cols.append("rank_pairwise_cindex")
        ascending.append(False)
    if "rank_regret" in work.columns:
        sort_cols.append("rank_regret")
        ascending.append(True)
    return work.sort_values(sort_cols, ascending=ascending)


def _build_heldout_tier_tables(
    heldout_ranked_path: Path,
    out_dir: Path,
    top_per_group: int,
) -> List[str]:
    produced: List[str] = []
    df = _read_csv(heldout_ranked_path)
    if df.empty:
        return produced

    needed = {"protocol", "head", "lane", "method", "rank_spearman"}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Missing required heldout columns in {heldout_ranked_path}: {', '.join(missing)}"
        )

    ranked = _sort_heldout(df)
    cols = [
        c
        for c in (
            "protocol",
            "head",
            "lane",
            "variant",
            "method",
            "source_run",
            "signal_k",
            "n_predictors_total",
            "rank_spearman",
            "rank_pairwise_cindex",
            "rank_kendall_tau",
            "rank_regret",
            "n_folds_scored",
            "n_rows_scored",
        )
        if c in ranked.columns
    ]

    top_blocks = ranked.groupby(["protocol", "head", "lane"], dropna=False).head(top_per_group)
    top_blocks = top_blocks[cols].reset_index(drop=True)
    top_name = "tier4_model_holdout_top_by_protocol_head_lane.csv"
    _write_csv(top_blocks, out_dir / top_name)
    produced.append(top_name)

    top_paper_cols = _nonempty_cols(
        top_blocks,
        [
            "protocol",
            "head",
            "lane",
            "method",
            "signal_k",
            "n_predictors_total",
            "rank_spearman",
            "rank_pairwise_cindex",
            "rank_kendall_tau",
            "rank_regret",
        ],
    )
    if top_paper_cols:
        top_paper = top_blocks[top_paper_cols].copy()
        top_paper = _rename_cols(
            top_paper,
            {
                "rank_spearman": "spearman",
                "rank_pairwise_cindex": "cindex",
                "rank_kendall_tau": "kendall_tau",
                "rank_regret": "regret",
            },
        )
        top_paper_name = "tier4_model_holdout_top_by_protocol_head_lane_paper.csv"
        _write_csv(top_paper, out_dir / top_paper_name)
        produced.append(top_paper_name)

    best_by_protocol_head = ranked.groupby(["protocol", "head"], dropna=False).head(1)
    best_by_protocol_head = best_by_protocol_head[cols].reset_index(drop=True)
    best_name = "tier4_model_holdout_best_by_protocol_head.csv"
    _write_csv(best_by_protocol_head, out_dir / best_name)
    produced.append(best_name)

    best_paper_cols = _nonempty_cols(
        best_by_protocol_head,
        [
            "protocol",
            "head",
            "lane",
            "method",
            "signal_k",
            "n_predictors_total",
            "rank_spearman",
            "rank_pairwise_cindex",
            "rank_kendall_tau",
            "rank_regret",
        ],
    )
    if best_paper_cols:
        best_paper = best_by_protocol_head[best_paper_cols].copy()
        best_paper = _rename_cols(
            best_paper,
            {
                "rank_spearman": "spearman",
                "rank_pairwise_cindex": "cindex",
                "rank_kendall_tau": "kendall_tau",
                "rank_regret": "regret",
            },
        )
        best_paper_name = "tier4_model_holdout_best_by_protocol_head_paper.csv"
        _write_csv(best_paper, out_dir / best_paper_name)
        produced.append(best_paper_name)

    return produced


def _write_manifest(out_dir: Path, files: Iterable[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# Final Table Manifest", ""]
    for name in files:
        lines.append(f"- {name}")
    (out_dir / "final_tables_manifest.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final paper summary tables (ranking-first).")
    parser.add_argument(
        "--summary",
        default="",
        help="Legacy single method_summary.csv path (optional).",
    )
    parser.add_argument(
        "--method-summaries",
        default="",
        help="Comma-separated method_summary.csv paths to combine.",
    )
    parser.add_argument(
        "--run-roots",
        default="",
        help="Comma-separated run roots; each contributes <root>/method_summary.csv.",
    )
    parser.add_argument(
        "--heldout-ranked",
        default="",
        help="Optional heldout_model_cv_ranked_all.csv path for tier-4 tables.",
    )
    parser.add_argument(
        "--catspp-auc",
        default="",
        help="Optional auc_with_features.csv path for CatsPP zero-shot tables.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for all tables.",
    )
    parser.add_argument(
        "--target",
        default="auc_normalized_observed",
        help="Target column to filter method summaries.",
    )
    parser.add_argument(
        "--models",
        default="pairwise_rank,ols,ridge",
        help="Comma-separated model heads to retain from method summary tables.",
    )
    parser.add_argument(
        "--score-col",
        default="auc_normalized_observed",
        help="Performance column for zero-shot CatsPP tables.",
    )
    parser.add_argument(
        "--peak-col",
        default="peak_pck",
        help="Peak metric column for zero-shot grid tables.",
    )
    parser.add_argument(
        "--tier0-catspp-pretrained",
        default="true",
        help="Bool filter for CatsPP pretrained in tier-0 tables (default: true).",
    )
    parser.add_argument(
        "--tier0-catspp-freeze",
        default="false",
        help="Bool filter for CatsPP freeze in tier-0 tables (default: false).",
    )
    parser.add_argument(
        "--tier0-raft-pretrained",
        default="true",
        help="Bool filter for RAFT pretrained in tier-0 tables (default: true).",
    )
    parser.add_argument(
        "--tier0-raft-freeze",
        default="false",
        help="Bool filter for RAFT freeze in tier-0 tables (default: false).",
    )
    parser.add_argument(
        "--tier0-base-train-datasets",
        default="sintel,flyingthings,spair,pointodyssey",
        help="Comma-separated base training datasets to keep as row anchors for tier-0 representative grids.",
    )
    parser.add_argument(
        "--tier0-include-synthetic-only",
        default="true",
        help="Whether to include pure synthetic train variants (default: true).",
    )
    parser.add_argument(
        "--tier0-synthetic-train-datasets",
        default="",
        help="Optional comma-separated synthetic train datasets to include explicitly.",
    )
    parser.add_argument(
        "--tier0-eval-benchmarks",
        default="",
        help="Optional comma-separated benchmark column order/filter for tier-0 grids.",
    )
    parser.add_argument(
        "--tier0-pair-precision",
        type=int,
        default=1,
        help="Decimal precision for pair cells '(peak, auc)' in tier-0 representative grids.",
    )
    parser.add_argument(
        "--tier0-cell-mode",
        choices=("pair", "peak"),
        default="pair",
        help="Tier-0 grid cell format: 'pair' => '(peak, auc_norm)', 'peak' => peak PCK only.",
    )
    parser.add_argument(
        "--tier0-pretty-dataset-labels",
        action="store_true",
        help="Use cleaned paper-style dataset labels in tier-0 grid rows/columns.",
    )
    parser.add_argument(
        "--tier0-synthetic-label",
        default="SDF-Fractal3D",
        help="Display label for the base synthetic dataset when pretty labels are enabled.",
    )
    parser.add_argument(
        "--tier0-aggregate-all-model-encoders",
        action="store_true",
        help="Average tier-0 cells across all model encoder families present in the input CSV.",
    )
    parser.add_argument(
        "--tier0-add-size-columns",
        action="store_true",
        help="Add train dataset size columns to tier-0 grids.",
    )
    parser.add_argument(
        "--tier0-size-column-mode",
        choices=("samples_only", "samples_and_flows"),
        default="samples_and_flows",
        help="Tier-0 size column mode when --tier0-add-size-columns is enabled.",
    )
    parser.add_argument(
        "--tier0-add-dataset-type-column",
        action="store_true",
        help="Add a train-row Dataset Type column to tier-0 representative grids.",
    )
    parser.add_argument(
        "--tier0-size-precision",
        type=int,
        default=0,
        help="Decimal precision for Train Avg Flows when --tier0-add-size-columns is enabled.",
    )
    parser.add_argument(
        "--top-overall",
        type=int,
        default=20,
        help="Top rows for each tier overall table.",
    )
    parser.add_argument(
        "--top-per-family",
        type=int,
        default=6,
        help="Top rows per family for each tier table.",
    )
    parser.add_argument(
        "--top-per-heldout-group",
        type=int,
        default=3,
        help="Top rows per (protocol, head, lane) for heldout tier-4 table.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    produced: List[str] = []

    summary_paths = _resolve_summary_paths(args.summary, args.method_summaries, args.run_roots)
    model_list = _split_csv_arg(args.models)
    if summary_paths:
        method_df = _load_method_summaries(
            paths=summary_paths,
            target=args.target,
            allowed_models=model_list,
        )
        if not method_df.empty:
            produced_tiers = _build_rank_tier_tables(
                method_df=method_df,
                out_dir=out_dir,
                top_overall=max(1, int(args.top_overall)),
                top_per_family=max(1, int(args.top_per_family)),
            )
            produced.extend([name for _, name in produced_tiers])

    if args.catspp_auc.strip():
        produced.extend(
            _build_catspp_zero_shot_tables(
                auc_path=Path(args.catspp_auc),
                out_dir=out_dir,
                score_col=args.score_col,
                peak_col=args.peak_col,
                catspp_pretrained=_parse_bool_text(args.tier0_catspp_pretrained, True),
                catspp_freeze=_parse_bool_text(args.tier0_catspp_freeze, False),
                raft_pretrained=_parse_bool_text(args.tier0_raft_pretrained, True),
                raft_freeze=_parse_bool_text(args.tier0_raft_freeze, False),
                base_train_datasets=_split_csv_arg(args.tier0_base_train_datasets),
                include_synthetic_only=_parse_bool_text(args.tier0_include_synthetic_only, True),
                synthetic_train_datasets=_split_csv_arg(args.tier0_synthetic_train_datasets),
                eval_benchmarks=_split_csv_arg(args.tier0_eval_benchmarks),
                pair_precision=max(0, int(args.tier0_pair_precision)),
                cell_mode=str(args.tier0_cell_mode),
                pretty_dataset_labels=bool(args.tier0_pretty_dataset_labels),
                synthetic_label=str(args.tier0_synthetic_label),
                aggregate_all_model_encoders=bool(args.tier0_aggregate_all_model_encoders),
                add_size_columns=bool(args.tier0_add_size_columns),
                size_column_mode=str(args.tier0_size_column_mode),
                add_dataset_type_column=bool(args.tier0_add_dataset_type_column),
                size_precision=max(0, int(args.tier0_size_precision)),
            )
        )

    if args.heldout_ranked.strip():
        produced.extend(
            _build_heldout_tier_tables(
                heldout_ranked_path=Path(args.heldout_ranked),
                out_dir=out_dir,
                top_per_group=max(1, int(args.top_per_heldout_group)),
            )
        )

    _write_manifest(out_dir, produced)
    print(f"Wrote {len(produced)} tables to {out_dir}")
    for item in produced:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
