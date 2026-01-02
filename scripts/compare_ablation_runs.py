#!/usr/bin/env python3
"""
Compare ablation runs (combined/flow_only/semantic_only) across targets.

Reads prediction CSVs from each run directory and writes a single summary file
with tables for easy comparison.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

FLOW_FAMILY = ["kitti2012", "kitti2015", "middlebury", "flyingthings", "pointodyssey"]
SEMANTIC_FAMILY = ["spair", "pfpascal", "pfwillow", "tss"]


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().empty:
        return float("nan")
    return float(values.dropna().mean())


def _read_rank_summary(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    overall = df[df["benchmark"] == "__overall__"]
    if overall.empty:
        return None
    row = overall.iloc[0].to_dict()
    return row


def _rank_family_summary(df: pd.DataFrame, benchmarks: List[str]) -> Optional[Dict[str, float]]:
    if df.empty:
        return None
    sub = df[df["benchmark"].isin(benchmarks)].copy()
    if sub.empty:
        return None
    summary = {
        "top1": _safe_mean(sub.get("top1")),
        "top3": _safe_mean(sub.get("top3")),
        "regret": _safe_mean(sub.get("regret")),
        "spearman": _safe_mean(sub.get("spearman")),
        "mean_abs_rank_error": _safe_mean(sub.get("mean_abs_rank_error")),
        "mean_abs_rank_pct_error": _safe_mean(sub.get("mean_abs_rank_pct_error")),
        "n_benchmarks": int(sub["benchmark"].nunique()),
    }
    if "topk" in sub.columns:
        summary["topk"] = _safe_mean(sub.get("topk"))
    if "topk_k" in sub.columns:
        summary["topk_k"] = _safe_mean(sub.get("topk_k"))
    if "topk_frac" in sub.columns:
        summary["topk_frac"] = _safe_mean(sub.get("topk_frac"))
    return summary


def _read_pred_summary(path: Path, label_col: str) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    overall = df[df[label_col] == "__overall__"]
    if overall.empty:
        return None
    row = overall.iloc[0].to_dict()
    return row


def _format_rank_row(name: str, row: Dict[str, float]) -> str:
    parts = [
        f"{name:<14}",
        f"{row.get('top1', np.nan):>6.2f}",
        f"{row.get('top3', np.nan):>6.2f}",
    ]
    if "topk" in row and not pd.isna(row.get("topk")):
        parts.append(f"{row.get('topk', np.nan):>6.2f}")
    else:
        parts.append(f"{np.nan:>6}")
    parts.extend([
        f"{row.get('regret', np.nan):>7.2f}",
        f"{row.get('spearman', np.nan):>8.2f}",
    ])
    return "  ".join(parts)


def _format_pred_row(name: str, row: Dict[str, float]) -> str:
    pearson = row.get("pearson", np.nan)
    spearman = row.get("spearman", np.nan)
    pearson_ci = row.get("pearson_ci", (np.nan, np.nan))
    spearman_ci = row.get("spearman_ci", (np.nan, np.nan))
    parts = [
        f"{name:<14}",
        f"{row.get('mae', np.nan):>7.2f}",
        f"{row.get('rmse', np.nan):>7.2f}",
        f"{pearson:>8.2f}",
        f"[{pearson_ci[0]:.2f},{pearson_ci[1]:.2f}]",
        f"{spearman:>8.2f}",
        f"[{spearman_ci[0]:.2f},{spearman_ci[1]:.2f}]",
    ]
    return "  ".join(parts)


def _load_family_table(rank_path: Path) -> Dict[str, Optional[Dict[str, float]]]:
    if not rank_path.exists():
        return {}
    df = pd.read_csv(rank_path)
    if df.empty:
        return {}
    return {
        "flow": _rank_family_summary(df, FLOW_FAMILY),
        "semantic": _rank_family_summary(df, SEMANTIC_FAMILY),
    }


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return _pearson(rx, ry)


def _bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_boot: int = 200,
    seed: int = 17,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 3:
        return (float("nan"), float("nan"))
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xs = x[idx]
        ys = y[idx]
        if method == "pearson":
            val = _pearson(xs, ys)
        else:
            val = _spearman(xs, ys)
        if np.isnan(val):
            continue
        stats.append(val)
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return (float(lo), float(hi))


def _read_pred_rows(path: Path) -> Optional[Dict[str, float]]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "prediction" not in df.columns or "target" not in df.columns:
        return None
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["prediction", "target"])
    if df.empty:
        return None
    pred = df["prediction"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(target - pred)))
    rmse = float(np.sqrt(np.mean((target - pred) ** 2)))
    pearson = _pearson(pred, target)
    spearman = _spearman(pred, target)
    pearson_ci = _bootstrap_corr(pred, target, "pearson")
    spearman_ci = _bootstrap_corr(pred, target, "spearman")
    return {
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
        "spearman": spearman,
        "pearson_ci": pearson_ci,
        "spearman_ci": spearman_ci,
    }


def _read_predictor_family_importance(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    lines = path.read_text().splitlines()
    out: Dict[str, Dict[str, str]] = {}
    in_block = False
    for line in lines:
        if line.strip().startswith("Predictor family importance"):
            in_block = True
            continue
        if in_block:
            if not line.strip():
                break
            if ":" not in line:
                continue
            name, rest = line.strip().split(":", 1)
            out[name.strip()] = {"raw": rest.strip()}
    return out


def build_tables_for_run(run_dir: Path) -> Dict[str, Dict[str, Optional[Dict[str, float]]]]:
    tables = {}
    lobo_rank_path = run_dir / "prediction_lobo_rank_summary.csv"
    loto_rank_path = run_dir / "prediction_loto_rank_summary.csv"
    tables["lobo_rank"] = {
        "overall": _read_rank_summary(lobo_rank_path),
    }
    tables["lobo_family"] = _load_family_table(lobo_rank_path)
    tables["loto_rank"] = {
        "overall": _read_rank_summary(loto_rank_path),
    }
    tables["loto_family"] = _load_family_table(loto_rank_path)
    tables["lobo_pred"] = {
        "overall": _read_pred_rows(run_dir / "prediction_lobo_rows.csv"),
    }
    tables["loto_pred"] = {
        "overall": _read_pred_rows(run_dir / "prediction_loto_rows.csv"),
    }
    tables["family_importance"] = _read_predictor_family_importance(
        run_dir / "summary_report.txt"
    )
    return tables


def write_summary(output_path: Path, target: str, rows: Dict[str, Dict[str, Dict[str, Optional[Dict[str, float]]]]]) -> None:
    out_lines = []
    out_lines.append(f"ABLATION SUMMARY: {target}")
    out_lines.append("=" * 80)

    def write_rank_table(title: str, key: str, family: bool = False) -> None:
        out_lines.append("")
        out_lines.append(title)
        if family:
            header = "Variant         top1   top3   topk  regret  spearman  n_bench"
        else:
            header = "Variant         top1   top3   topk  regret  spearman"
        out_lines.append(header)
        out_lines.append("-" * len(header))
        for variant, tables in rows.items():
            row = tables.get(key, {}).get("overall")
            if family:
                family_row = tables.get(key, {}).get("flow")
                if family_row:
                    row = dict(family_row)
                else:
                    row = None
            if row is None:
                line = f"{variant:<14}  n/a"
            else:
                line = _format_rank_row(variant, row)
                if family:
                    line = f"{line}  {int(row.get('n_benchmarks', 0)):>7}"
            out_lines.append(line)

    def write_rank_family_table(title: str, key: str, family_name: str) -> None:
        out_lines.append("")
        out_lines.append(f"{title} ({family_name})")
        header = "Variant         top1   top3   topk  regret  spearman  n_bench"
        out_lines.append(header)
        out_lines.append("-" * len(header))
        for variant, tables in rows.items():
            row = tables.get(key, {}).get(family_name)
            if row is None:
                out_lines.append(f"{variant:<14}  n/a")
                continue
            line = _format_rank_row(variant, row)
            line = f"{line}  {int(row.get('n_benchmarks', 0)):>7}"
            out_lines.append(line)

    def write_pred_table(title: str, key: str) -> None:
        out_lines.append("")
        out_lines.append(title)
        header = "Variant          MAE    RMSE  Pearson     95% CI  Spearman    95% CI"
        out_lines.append(header)
        out_lines.append("-" * len(header))
        for variant, tables in rows.items():
            row = tables.get(key, {}).get("overall")
            if row is None:
                out_lines.append(f"{variant:<14}  n/a")
                continue
            out_lines.append(_format_pred_row(variant, row))

    def write_importance_table() -> None:
        out_lines.append("")
        out_lines.append("Predictor family importance (from summary_report.txt)")
        header = "Variant         flow                               semantic"
        out_lines.append(header)
        out_lines.append("-" * len(header))
        for variant, tables in rows.items():
            fam = tables.get("family_importance", {})
            flow = fam.get("flow", {}).get("raw", "n/a")
            sem = fam.get("semantic", {}).get("raw", "n/a")
            out_lines.append(f"{variant:<14}  {flow:<34}  {sem}")

    write_rank_table("LOBO rank metrics (overall)", "lobo_rank", family=False)
    write_rank_family_table("LOBO rank metrics", "lobo_family", "flow")
    write_rank_family_table("LOBO rank metrics", "lobo_family", "semantic")
    write_rank_table("LOTO rank metrics (overall)", "loto_rank", family=False)
    write_rank_family_table("LOTO rank metrics", "loto_family", "flow")
    write_rank_family_table("LOTO rank metrics", "loto_family", "semantic")
    write_pred_table("LOBO prediction metrics (overall)", "lobo_pred")
    write_pred_table("LOTO prediction metrics (overall)", "loto_pred")
    write_importance_table()

    output_path.write_text("\n".join(out_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ablation runs across targets.")
    parser.add_argument(
        "--base-dir",
        default="analysis/leakage_free_local_fast_dino_faiss",
        help="Base analysis directory containing target subfolders.",
    )
    parser.add_argument(
        "--targets",
        default="auc_delta,auc_delta_rank,peak_pck,peak_pck_rank",
        help="Comma-separated list of target subdirectories.",
    )
    parser.add_argument(
        "--variants",
        default="combined,flow_only,semantic_only",
        help="Comma-separated list of ablation variants.",
    )
    parser.add_argument(
        "--output-name",
        default="ablation_summary.txt",
        help="Summary filename to write within each target directory.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    for target in targets:
        target_dir = base_dir / target
        if not target_dir.exists():
            print(f"Skipping {target}: directory not found ({target_dir})")
            continue
        rows = {}
        for variant in variants:
            run_dir = target_dir / variant
            if not run_dir.exists():
                rows[variant] = {}
                continue
            rows[variant] = build_tables_for_run(run_dir)
        output_path = target_dir / args.output_name
        write_summary(output_path, target, rows)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
