#!/usr/bin/env python3
"""
Aggregate snapshot runs into motion-realism conditions for paper figures/tables.

Default conditions:
  - synthetic_2d  : Train dataset starts with synthetic_2d_warp
  - imagenet_2d   : Train dataset starts with imagenet2dwarp
  - synthetic_3d  : Train dataset starts with synthetic (except synthetic_2d_warp)

Outputs:
  - run_level_benchmark.csv
  - run_level_macro.csv
  - condition_summary.csv
  - benchmark_condition_summary.csv
  - final_table.csv
  - final_table.tex
  - bar_overall_macro.png
  - bar_by_benchmark.png (optional)
  - stratum_condition_scores.csv
  - stratum_condition_scores_complete.csv
  - stratified_condition_summary.csv
  - stratified_pairwise_deltas.csv
  - stratified_ordering_summary.txt
  - benchmark_condition_summary_stratified.csv
  - final_table_stratified.csv
  - final_table_stratified.tex
  - bar_overall_macro_stratified.png
  - bar_by_benchmark_stratified.png (optional)

Notes:
  - Confidence intervals use bootstrap percentile bounds and are clipped to [0, 100]
    because PCK is bounded.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 17


@dataclass(frozen=True)
class ConditionSpec:
    label: str
    patterns: List[re.Pattern[str]]


DEFAULT_CONDITIONS = [
    "synthetic_2d=^synthetic_2d_warp(?:$|_)",
    "imagenet_2d=^imagenet2dwarp(?:$|_)",
    "synthetic_3d=^synthetic(?:$|_(?!2d_warp).+)",
]

CONDITION_DISPLAY_MAP = {
    "synthetic_3d": "SDF-Fractal3D",
    "synthetic_2d": "SDF-Fractal2D-Warp",
    "imagenet_2d": "ImageNet2D-Warp",
}

BENCHMARK_DISPLAY_MAP = {
    "__overall_macro__": "Macro Avg",
    "flyingthings": "FlyingThings",
    "kitti2012": "KITTI-2012",
    "kitti2015": "KITTI-2015",
    "middlebury": "Middlebury",
    "pfpascal": "PF-PASCAL",
    "pfwillow": "PF-WILLOW",
    "pointodyssey": "PointOdyssey",
    "spair": "SPair-71k",
    "synthetic": "SDF-Fractal3D",
    "tss": "TSS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate snapshot performance across synthetic-vs-2D-warp conditions."
    )
    parser.add_argument(
        "--snapshot-dirs",
        nargs="+",
        required=True,
        help="Snapshot root directories to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/motion_realism_comparison",
        help="Output directory for tables/plots.",
    )
    parser.add_argument(
        "--condition",
        action="append",
        default=None,
        help=(
            "Condition spec as 'label=regex1,regex2'. "
            "If omitted, defaults to synthetic_3d/synthetic_2d/imagenet_2d."
        ),
    )
    parser.add_argument(
        "--benchmarks",
        default="",
        help="Optional comma-separated benchmark list to use for macro score.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum recursion depth while scanning roots.",
    )
    parser.add_argument(
        "--require-all-benchmarks",
        action="store_true",
        help="Drop a run if any requested benchmark is missing.",
    )
    parser.add_argument(
        "--include-name-regex",
        action="append",
        default=[],
        help="Only include runs whose snapshot directory name matches any given regex.",
    )
    parser.add_argument(
        "--exclude-name-regex",
        action="append",
        default=[],
        help="Exclude runs whose snapshot directory name matches any given regex.",
    )
    parser.add_argument(
        "--no-benchmark-plot",
        action="store_true",
        help="Skip grouped bar plot by benchmark.",
    )
    parser.add_argument(
        "--dedup-by",
        choices=["name", "path", "none"],
        default="name",
        help=(
            "De-duplicate runs before aggregation. "
            "'name' removes duplicate snapshot directory names across roots (default)."
        ),
    )
    parser.add_argument(
        "--stratified-keys",
        default="architecture,pretrained,freeze",
        help=(
            "Comma-separated keys for model-adjusted stratified analysis. "
            "Supported: architecture,pretrained,freeze"
        ),
    )
    parser.add_argument(
        "--no-stratified-report",
        action="store_true",
        help="Disable model-adjusted stratified outputs.",
    )
    parser.add_argument(
        "--pck-alpha-percent",
        type=float,
        default=5.0,
        help="Alpha threshold percentage used for PCK labeling (default: 5.0).",
    )
    return parser.parse_args()


def parse_condition_specs(condition_args: Sequence[str] | None) -> List[ConditionSpec]:
    specs = condition_args if condition_args else DEFAULT_CONDITIONS
    out: List[ConditionSpec] = []
    for text in specs:
        if "=" not in text:
            raise ValueError(f"Invalid --condition '{text}'. Expected 'label=regex1,regex2'.")
        label, rhs = text.split("=", 1)
        label = label.strip()
        pats = [p.strip() for p in rhs.split(",") if p.strip()]
        if not label or not pats:
            raise ValueError(f"Invalid --condition '{text}'.")
        out.append(ConditionSpec(label=label, patterns=[re.compile(p, re.IGNORECASE) for p in pats]))
    return out


def parse_benchmark_list(text: str) -> List[str]:
    if not text.strip():
        return []
    return [b.strip() for b in text.split(",") if b.strip()]


def parse_stratified_keys(text: str) -> List[str]:
    keys = [k.strip() for k in text.split(",") if k.strip()]
    supported = {"architecture", "pretrained", "freeze"}
    for key in keys:
        if key not in supported:
            raise ValueError(f"Unsupported --stratified-keys value '{key}'. Supported: {sorted(supported)}")
    return keys


def parse_dir_dataset(directory_name: str) -> str | None:
    param_keywords = {"stride", "sequence_length", "freeze", "pretrained", "eval", "steps", "logsteps"}
    parts = directory_name.split("_")
    if not parts:
        return None

    if parts[0].lower() == "synthetic" and len(parts) >= 2:
        dataset_parts = [parts[0]]
        for i in range(1, len(parts)):
            part = parts[i].lower()
            if (
                part in param_keywords
                or part.startswith("stride")
                or part.startswith("sequence")
                or part.startswith("freeze")
                or part.startswith("pretrained")
                or part.startswith("steps")
                or part.startswith("logsteps")
                or (part.isdigit() and i > 1)
            ):
                break
            dataset_parts.append(parts[i])
        return "_".join(dataset_parts)

    mixed_with_percent = re.match(r"^([a-zA-Z]+)_([a-zA-Z]+)_(\d+)_(\d+)(?:_|$)", directory_name)
    if mixed_with_percent:
        dataset1, dataset2, p1, p2 = mixed_with_percent.groups()
        return f"{dataset1}_{dataset2}_{p1}_{p2}"

    if len(parts) >= 2:
        part2 = parts[1].lower()
        is_param = (
            part2 in param_keywords
            or part2.startswith("stride")
            or part2.startswith("sequence")
            or part2.startswith("freeze")
            or part2.startswith("pretrained")
            or part2.startswith("steps")
            or part2.startswith("logsteps")
            or part2.isdigit()
        )
        if not is_param:
            return f"{parts[0]}_{parts[1]}"

    return parts[0]


def parse_training_dataset(summary_path: Path, snapshot_name: str) -> str | None:
    dataset = None
    if summary_path.exists():
        try:
            for line in summary_path.read_text().splitlines():
                if line.startswith("Train dataset:"):
                    dataset = line.split("Train dataset:", 1)[1].strip().lower().replace("+", "_")
                    break
        except Exception:
            dataset = None

    dir_dataset = parse_dir_dataset(snapshot_name)
    if dir_dataset:
        dir_dataset = dir_dataset.lower()

    if dataset is None and dir_dataset:
        return dir_dataset

    if dataset and dir_dataset:
        if dir_dataset.startswith("synthetic_") and dataset == "synthetic":
            return dir_dataset
        if (
            "_synthetic_" in dir_dataset
            and re.search(r"_\d+_\d+$", dir_dataset)
            and ("synthetic" in dataset or "+" in dataset)
        ):
            return dir_dataset

    return dataset or dir_dataset


def parse_snapshot_metadata(snapshot_name: str) -> Dict[str, str]:
    low = snapshot_name.lower()

    if "raft" in low:
        architecture = "raft"
    elif "cats" in low:
        architecture = "cats"
    else:
        architecture = "unknown"

    if "pretrainedtrue" in low:
        pretrained = "true"
    elif "pretrainedfalse" in low:
        pretrained = "false"
    else:
        pretrained = "unknown"

    if "freezetrue" in low:
        freeze = "true"
    elif "freezefalse" in low:
        freeze = "false"
    else:
        freeze = "unknown"

    return {
        "architecture": architecture,
        "pretrained": pretrained,
        "freeze": freeze,
    }


def parse_best_performance_from_summary(summary_path: Path) -> Dict[str, float]:
    if not summary_path.exists():
        return {}

    best_performance: Dict[str, float] = {}
    try:
        lines = summary_path.read_text().splitlines()
    except Exception:
        return {}

    in_best_section = False
    for line in lines:
        text = line.strip()
        if "BEST PERFORMANCE PER BENCHMARK:" in text:
            in_best_section = True
            continue
        if in_best_section:
            if text.startswith("-") and len(text) > 10:
                continue
            if text.startswith("MOTION-AWARE") or text.startswith("TRAINING CONFIGURATION"):
                break
            if ":" in text and "%" in text:
                benchmark, rest = text.split(":", 1)
                match = re.search(r"(\d+\.?\d*)%", rest)
                if match:
                    best_performance[benchmark.strip()] = float(match.group(1))
    return best_performance


def parse_best_performance_from_validation_csv(csv_path: Path) -> Dict[str, float]:
    if not csv_path.exists():
        return {}
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}
    if df.empty or "benchmark" not in df.columns or "pck" not in df.columns:
        return {}
    df["pck"] = pd.to_numeric(df["pck"], errors="coerce")
    df = df.dropna(subset=["benchmark", "pck"])
    if df.empty:
        return {}
    best = df.groupby("benchmark", dropna=False)["pck"].max().reset_index()
    return {str(r["benchmark"]): float(r["pck"]) for _, r in best.iterrows()}


def is_snapshot_dir(path: Path) -> bool:
    return (path / "training_summary.txt").exists() or (path / "validation_results.csv").exists()


def collect_snapshot_dirs(roots: Iterable[str], max_depth: int) -> List[Path]:
    found: List[Path] = []

    def walk(root: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for sub in sorted(root.iterdir()):
                if not sub.is_dir():
                    continue
                if is_snapshot_dir(sub):
                    found.append(sub)
                else:
                    walk(sub, depth + 1)
        except PermissionError:
            return

    for root in roots:
        rp = Path(root).expanduser()
        if rp.exists() and rp.is_dir():
            walk(rp, 0)

    seen = set()
    unique: List[Path] = []
    for p in found:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def infer_condition(dataset: str | None, snapshot_name: str, specs: Sequence[ConditionSpec]) -> str | None:
    candidates = [snapshot_name.lower()]
    if dataset:
        candidates.insert(0, dataset.lower())

    for spec in specs:
        for cand in candidates:
            if any(p.search(cand) for p in spec.patterns):
                return spec.label
    return None


def _name_allowed(
    name: str,
    include_patterns: Sequence[re.Pattern[str]],
    exclude_patterns: Sequence[re.Pattern[str]],
) -> bool:
    if include_patterns and not any(p.search(name) for p in include_patterns):
        return False
    if exclude_patterns and any(p.search(name) for p in exclude_patterns):
        return False
    return True


def summarize_values(values: pd.Series, bounds: tuple[float, float] | None = None) -> Dict[str, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    n = int(vals.shape[0])
    mean = float(vals.mean()) if n else float("nan")
    std = float(vals.std(ddof=1)) if n > 1 else float("nan")
    sem = float(std / math.sqrt(n)) if n > 1 and math.isfinite(std) else float("nan")
    if n == 0:
        return {
            "n_runs": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    if n == 1:
        ci_low = mean
        ci_high = mean
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        arr = vals.to_numpy(dtype=float)
        # Bootstrap CI on the mean avoids symmetric tails that can go negative on bounded metrics.
        boot_means = []
        for _ in range(BOOTSTRAP_ITERS):
            idx = rng.integers(0, n, size=n)
            boot_means.append(float(np.mean(arr[idx])))
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5]).tolist()

    if bounds is not None:
        low_b, high_b = bounds
        ci_low = float(np.clip(ci_low, low_b, high_b)) if math.isfinite(ci_low) else float("nan")
        ci_high = float(np.clip(ci_high, low_b, high_b)) if math.isfinite(ci_high) else float("nan")
    ci95 = max(mean - ci_low, ci_high - mean) if math.isfinite(mean) else float("nan")

    return {
        "n_runs": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95": float(ci95),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def summarize_series(values: pd.Series) -> Dict[str, float]:
    # PCK is a percentage and should remain in [0, 100].
    return summarize_values(values, bounds=(0.0, 100.0))


def _default_pretty_label(text: str) -> str:
    return text.replace("_", " ").strip()


def _pck_label(alpha_percent: float) -> str:
    if float(alpha_percent).is_integer():
        a = str(int(alpha_percent))
    else:
        a = f"{alpha_percent:g}"
    return f"PCK@{a}%"


def _pretty_condition_map(condition_order: Sequence[str]) -> Dict[str, str]:
    out = {}
    for c in condition_order:
        out[c] = CONDITION_DISPLAY_MAP.get(c, _default_pretty_label(c))
    return out


def _pretty_benchmark_name(name: str) -> str:
    return BENCHMARK_DISPLAY_MAP.get(name, _default_pretty_label(name))


def _prettify_table(
    table_df: pd.DataFrame,
    condition_order: Sequence[str],
    benchmark_col_label: str,
) -> tuple[pd.DataFrame, List[str]]:
    df = table_df.copy()
    if "benchmark" in df.columns:
        df["benchmark"] = df["benchmark"].astype(str).map(_pretty_benchmark_name)
        df = df.rename(columns={"benchmark": benchmark_col_label})

    cond_map = _pretty_condition_map(condition_order)
    present_raw = [c for c in condition_order if c in table_df.columns]
    rename_map = {c: cond_map[c] for c in present_raw}
    df = df.rename(columns=rename_map)
    condition_cols_pretty = [cond_map[c] for c in present_raw]
    ordered_cols = [benchmark_col_label] + condition_cols_pretty
    df = df[ordered_cols]
    return df, condition_cols_pretty


def _prettify_summary_for_plots(
    condition_summary: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    condition_order: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cond_map = _pretty_condition_map(condition_order)
    pretty_order = [cond_map[c] for c in condition_order]

    cond = condition_summary.copy()
    cond["condition"] = cond["condition"].astype(str).map(lambda x: cond_map.get(x, x))

    bench = benchmark_summary.copy()
    bench["condition"] = bench["condition"].astype(str).map(lambda x: cond_map.get(x, x))
    bench["benchmark"] = bench["benchmark"].astype(str).map(_pretty_benchmark_name)
    return cond, bench, pretty_order


def write_latex_table(
    table_df: pd.DataFrame,
    condition_order: Sequence[str],
    out_path: Path,
    benchmark_col: str = "benchmark",
) -> None:
    lines = [
        r"\begin{tabular}{l" + "r" * len(condition_order) + "}",
        r"\toprule",
        f"{benchmark_col} & " + " & ".join(c.replace("_", r"\_") for c in condition_order) + r" \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        bench = str(row[benchmark_col]).replace("_", r"\_")
        vals = []
        for c in condition_order:
            v = row.get(c, "--")
            vals.append(str(v).replace("%", r"\%"))
        lines.append(f"{bench} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def fmt_mean_std(mean: float, std: float) -> str:
    if not math.isfinite(mean):
        return "--"
    if math.isfinite(std):
        return f"{mean:.2f} ± {std:.2f}"
    return f"{mean:.2f}"


def make_overall_plot(
    condition_summary: pd.DataFrame,
    condition_order: Sequence[str],
    out_path: Path,
    pck_label: str,
) -> None:
    df = condition_summary.copy()
    df = df[df["condition"].isin(condition_order)]
    df["order"] = df["condition"].apply(lambda x: condition_order.index(x))
    df = df.sort_values("order")
    if df.empty:
        return

    x = np.arange(len(df))
    y = df["mean_macro_pck"].to_numpy(dtype=float)
    lo = pd.to_numeric(df["ci95_low_macro_pck"], errors="coerce").to_numpy(dtype=float)
    hi = pd.to_numeric(df["ci95_high_macro_pck"], errors="coerce").to_numpy(dtype=float)
    lower_err = np.clip(y - np.nan_to_num(lo, nan=y), 0.0, None)
    upper_err = np.clip(np.nan_to_num(hi, nan=y) - y, 0.0, None)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        x,
        y,
        yerr=np.vstack([lower_err, upper_err]),
        capsize=5,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"][: len(df)],
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=15, ha="right")
    ax.set_ylabel(f"Macro Best {pck_label}")
    ax.set_title("Training Set Comparison (Macro over Benchmarks)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0.0)

    for i, bar in enumerate(bars):
        n = int(df.iloc[i]["n_runs"])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(upper_err[i], 0.0) + 0.2,
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_benchmark_plot(
    benchmark_summary: pd.DataFrame,
    condition_order: Sequence[str],
    out_path: Path,
    pck_label: str,
) -> None:
    if benchmark_summary.empty:
        return

    benchmarks = sorted(benchmark_summary["benchmark"].unique().tolist())
    width = 0.8 / max(len(condition_order), 1)
    x = np.arange(len(benchmarks))

    fig, ax = plt.subplots(figsize=(max(10, len(benchmarks) * 1.1), 4.8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, condition in enumerate(condition_order):
        sub = benchmark_summary[benchmark_summary["condition"] == condition]
        means = []
        lower_errs = []
        upper_errs = []
        for b in benchmarks:
            row = sub[sub["benchmark"] == b]
            if row.empty:
                means.append(np.nan)
                lower_errs.append(0.0)
                upper_errs.append(0.0)
            else:
                m = float(row.iloc[0]["mean_pck"])
                lo = row.iloc[0].get("ci95_low_pck", np.nan)
                hi = row.iloc[0].get("ci95_high_pck", np.nan)
                means.append(m)
                lo_err = m - float(lo) if pd.notna(lo) else 0.0
                hi_err = float(hi) - m if pd.notna(hi) else 0.0
                lower_errs.append(max(lo_err, 0.0))
                upper_errs.append(max(hi_err, 0.0))
        ax.bar(
            x + (idx - (len(condition_order) - 1) / 2.0) * width,
            means,
            width=width,
            yerr=np.vstack([lower_errs, upper_errs]),
            capsize=3,
            label=condition,
            color=colors[idx % len(colors)],
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks, rotation=35, ha="right")
    ax.set_ylabel(f"Best {pck_label}")
    ax.set_title(f"Best {pck_label} by Benchmark and Training Condition")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _build_display_table(
    condition_summary: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    condition_order: Sequence[str],
    mean_macro_col: str,
    std_macro_col: str,
    mean_bench_col: str,
    std_bench_col: str,
) -> pd.DataFrame:
    overall_rows = []
    for condition in condition_order:
        row = condition_summary[condition_summary["condition"] == condition]
        if row.empty:
            overall_rows.append({"benchmark": "__overall_macro__", "condition": condition, "display": "--"})
        else:
            m = float(row.iloc[0][mean_macro_col])
            s = float(row.iloc[0][std_macro_col]) if pd.notna(row.iloc[0][std_macro_col]) else float("nan")
            overall_rows.append(
                {"benchmark": "__overall_macro__", "condition": condition, "display": fmt_mean_std(m, s)}
            )

    bench_rows = []
    for benchmark in sorted(benchmark_summary["benchmark"].unique().tolist()):
        sub = benchmark_summary[benchmark_summary["benchmark"] == benchmark]
        for condition in condition_order:
            row = sub[sub["condition"] == condition]
            if row.empty:
                bench_rows.append({"benchmark": benchmark, "condition": condition, "display": "--"})
            else:
                m = float(row.iloc[0][mean_bench_col])
                s = float(row.iloc[0][std_bench_col]) if pd.notna(row.iloc[0][std_bench_col]) else float("nan")
                bench_rows.append({"benchmark": benchmark, "condition": condition, "display": fmt_mean_std(m, s)})

    table_long = pd.DataFrame(overall_rows + bench_rows)
    final_table = table_long.pivot(index="benchmark", columns="condition", values="display").reset_index()
    keep_cols = ["benchmark"] + [c for c in condition_order if c in final_table.columns]
    return final_table[keep_cols]


def _build_stratum_id(df: pd.DataFrame, keys: Sequence[str]) -> pd.Series:
    work = df.copy()
    for key in keys:
        if key not in work.columns:
            work[key] = "unknown"
    return work[list(keys)].fillna("unknown").astype(str).agg("|".join, axis=1)


def build_stratified_outputs(
    macro_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    condition_order: Sequence[str],
    stratum_keys: Sequence[str],
    out_dir: Path,
    make_benchmark_plot_flag: bool,
    pck_label: str,
    benchmark_col_label: str,
) -> None:
    if not stratum_keys:
        return

    macro = macro_df.copy()
    bench = bench_df.copy()
    macro["stratum_id"] = _build_stratum_id(macro, stratum_keys)
    bench["stratum_id"] = _build_stratum_id(bench, stratum_keys)

    stratum_cond = (
        macro.groupby(list(stratum_keys) + ["stratum_id", "condition"], dropna=False)
        .agg(n_runs=("macro_best_pck", "count"), mean_macro_pck=("macro_best_pck", "mean"))
        .reset_index()
    )
    stratum_cond.to_csv(out_dir / "stratum_condition_scores.csv", index=False)

    stratum_pivot = stratum_cond.pivot_table(
        index=["stratum_id"] + list(stratum_keys),
        columns="condition",
        values="mean_macro_pck",
        aggfunc="mean",
    ).reset_index()
    needed = [c for c in condition_order if c in stratum_pivot.columns]
    complete = stratum_pivot.dropna(subset=needed) if needed else stratum_pivot.copy()
    complete.to_csv(out_dir / "stratum_condition_scores_complete.csv", index=False)

    if complete.empty:
        print("Stratified report skipped: no complete strata with all requested conditions.")
        return

    strat_rows = []
    for condition in condition_order:
        if condition not in complete.columns:
            continue
        stats = summarize_values(complete[condition], bounds=(0.0, 100.0))
        strat_rows.append(
            {
                "condition": condition,
                "n_strata": stats["n_runs"],
                "n_runs": stats["n_runs"],  # for plotting reuse
                "mean_macro_pck": stats["mean"],
                "std_macro_pck": stats["std"],
                "sem_macro_pck": stats["sem"],
                "ci95_macro_pck": stats["ci95"],
                "ci95_low_macro_pck": stats["ci95_low"],
                "ci95_high_macro_pck": stats["ci95_high"],
            }
        )
    strat_summary = pd.DataFrame(strat_rows)
    if not strat_summary.empty:
        strat_summary["order"] = strat_summary["condition"].apply(
            lambda c: condition_order.index(c) if c in condition_order else 10_000
        )
        strat_summary = strat_summary.sort_values("order").drop(columns=["order"])
    strat_summary.to_csv(out_dir / "stratified_condition_summary.csv", index=False)
    strat_summary_pretty_csv = strat_summary.copy()
    cond_map_csv = _pretty_condition_map(condition_order)
    if not strat_summary_pretty_csv.empty:
        strat_summary_pretty_csv["condition"] = strat_summary_pretty_csv["condition"].astype(str).map(
            lambda x: cond_map_csv.get(x, x)
        )
    strat_summary_pretty_csv.to_csv(out_dir / "stratified_condition_summary_paper.csv", index=False)

    pairs = [
        ("synthetic_3d", "synthetic_2d"),
        ("synthetic_2d", "imagenet_2d"),
        ("synthetic_3d", "imagenet_2d"),
    ]
    pair_rows = []
    for left, right in pairs:
        if left not in complete.columns or right not in complete.columns:
            continue
        delta = complete[left] - complete[right]
        stats = summarize_values(delta, bounds=None)
        pair_rows.append(
            {
                "comparison": f"{left}_minus_{right}",
                "n_strata": stats["n_runs"],
                "mean_delta": stats["mean"],
                "std_delta": stats["std"],
                "sem_delta": stats["sem"],
                "ci95_delta": stats["ci95"],
                "ci95_low_delta": stats["ci95_low"],
                "ci95_high_delta": stats["ci95_high"],
                "win_rate_left_gt_right": float((delta > 0).mean()),
            }
        )
    pd.DataFrame(pair_rows).to_csv(out_dir / "stratified_pairwise_deltas.csv", index=False)

    order_lines = []
    if all(c in complete.columns for c in ["synthetic_3d", "synthetic_2d", "imagenet_2d"]):
        cond3 = complete["synthetic_3d"]
        cond2 = complete["synthetic_2d"]
        condi = complete["imagenet_2d"]
        strict = ((cond3 > cond2) & (cond2 > condi)).mean()
        weak = ((cond3 >= cond2) & (cond2 >= condi)).mean()
        s2_better_s3 = (cond2 > cond3).mean()
        order_lines.append(f"n_complete_strata: {len(complete)}")
        order_lines.append(f"strict_order_rate(s3>s2>img): {strict:.4f}")
        order_lines.append(f"weak_order_rate(s3>=s2>=img): {weak:.4f}")
        order_lines.append(f"rate_s2_gt_s3: {s2_better_s3:.4f}")
    else:
        order_lines.append("ordering summary unavailable: missing one or more conditions in complete strata")
    (out_dir / "stratified_ordering_summary.txt").write_text("\n".join(order_lines) + "\n")

    bench_rows = []
    for benchmark, sub in bench.groupby("benchmark", dropna=False):
        sub_sc = (
            sub.groupby(list(stratum_keys) + ["stratum_id", "condition"], dropna=False)
            .agg(mean_pck=("best_pck", "mean"))
            .reset_index()
        )
        sub_pivot = sub_sc.pivot_table(
            index=["stratum_id"] + list(stratum_keys),
            columns="condition",
            values="mean_pck",
            aggfunc="mean",
        ).reset_index()
        sub_needed = [c for c in condition_order if c in sub_pivot.columns]
        sub_complete = sub_pivot.dropna(subset=sub_needed) if sub_needed else sub_pivot.copy()
        if sub_complete.empty:
            continue
        for condition in condition_order:
            if condition not in sub_complete.columns:
                continue
            stats = summarize_values(sub_complete[condition], bounds=(0.0, 100.0))
            bench_rows.append(
                {
                    "benchmark": benchmark,
                    "condition": condition,
                    "n_strata": stats["n_runs"],
                    "n_runs": stats["n_runs"],  # for plotting reuse
                    "mean_pck": stats["mean"],
                    "std_pck": stats["std"],
                    "sem_pck": stats["sem"],
                    "ci95_pck": stats["ci95"],
                    "ci95_low_pck": stats["ci95_low"],
                    "ci95_high_pck": stats["ci95_high"],
                }
            )
    bench_strat = pd.DataFrame(bench_rows)
    if not bench_strat.empty:
        bench_strat["condition_order"] = bench_strat["condition"].apply(
            lambda c: condition_order.index(c) if c in condition_order else 10_000
        )
        bench_strat = bench_strat.sort_values(["benchmark", "condition_order"]).drop(columns=["condition_order"])
    bench_strat.to_csv(out_dir / "benchmark_condition_summary_stratified.csv", index=False)
    bench_strat_pretty_csv = bench_strat.copy()
    if not bench_strat_pretty_csv.empty:
        cond_map_csv = _pretty_condition_map(condition_order)
        bench_strat_pretty_csv["condition"] = bench_strat_pretty_csv["condition"].astype(str).map(
            lambda x: cond_map_csv.get(x, x)
        )
        bench_strat_pretty_csv["benchmark"] = bench_strat_pretty_csv["benchmark"].astype(str).map(
            _pretty_benchmark_name
        )
    bench_strat_pretty_csv.to_csv(out_dir / "benchmark_condition_summary_stratified_paper.csv", index=False)

    if not strat_summary.empty and not bench_strat.empty:
        final_table = _build_display_table(
            condition_summary=strat_summary,
            benchmark_summary=bench_strat,
            condition_order=condition_order,
            mean_macro_col="mean_macro_pck",
            std_macro_col="std_macro_pck",
            mean_bench_col="mean_pck",
            std_bench_col="std_pck",
        )
        final_table.to_csv(out_dir / "final_table_stratified_raw.csv", index=False)
        final_table_pretty, cond_cols_pretty = _prettify_table(
            final_table,
            condition_order=condition_order,
            benchmark_col_label=benchmark_col_label,
        )
        final_table_pretty.to_csv(out_dir / "final_table_stratified.csv", index=False)
        write_latex_table(
            final_table_pretty,
            cond_cols_pretty,
            out_dir / "final_table_stratified.tex",
            benchmark_col=benchmark_col_label,
        )
        strat_summary_pretty, bench_strat_pretty, order_pretty = _prettify_summary_for_plots(
            strat_summary,
            bench_strat,
            condition_order,
        )
        make_overall_plot(
            strat_summary_pretty,
            order_pretty,
            out_dir / "bar_overall_macro_stratified.png",
            pck_label=pck_label,
        )
        if make_benchmark_plot_flag:
            make_benchmark_plot(
                bench_strat_pretty,
                order_pretty,
                out_dir / "bar_by_benchmark_stratified.png",
                pck_label=pck_label,
            )

    print(f"Stratified complete strata: {len(complete)}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pck_label = _pck_label(args.pck_alpha_percent)
    benchmark_col_label = f"Benchmark ({pck_label})"

    condition_specs = parse_condition_specs(args.condition)
    condition_order_raw = [c.label for c in condition_specs]
    canonical_order = ["synthetic_3d", "synthetic_2d", "imagenet_2d"]
    condition_order = [c for c in canonical_order if c in condition_order_raw] + [
        c for c in condition_order_raw if c not in canonical_order
    ]
    benchmark_filter = parse_benchmark_list(args.benchmarks)
    stratum_keys = parse_stratified_keys(args.stratified_keys)
    include_patterns = [re.compile(p, re.IGNORECASE) for p in args.include_name_regex]
    exclude_patterns = [re.compile(p, re.IGNORECASE) for p in args.exclude_name_regex]

    snapshots = collect_snapshot_dirs(args.snapshot_dirs, max_depth=args.max_depth)
    if not snapshots:
        raise SystemExit("No snapshot directories found.")

    benchmark_rows = []
    macro_rows = []
    seen_run_keys = set()

    for snap in snapshots:
        snap_name = snap.name
        if not _name_allowed(snap_name, include_patterns, exclude_patterns):
            continue

        summary_path = snap / "training_summary.txt"
        csv_path = snap / "validation_results.csv"
        metadata = parse_snapshot_metadata(snap_name)

        dataset = parse_training_dataset(summary_path, snap_name)
        condition = infer_condition(dataset, snap_name, condition_specs)
        if condition is None:
            continue

        if args.dedup_by != "none":
            if args.dedup_by == "name":
                run_key = snap_name
            else:
                run_key = str(snap.resolve())
            if run_key in seen_run_keys:
                continue
            seen_run_keys.add(run_key)

        best_map = parse_best_performance_from_summary(summary_path)
        if not best_map:
            best_map = parse_best_performance_from_validation_csv(csv_path)
        if not best_map:
            continue

        if benchmark_filter:
            best_map = {k: v for k, v in best_map.items() if k in benchmark_filter}
            if not best_map:
                continue

        if args.require_all_benchmarks and benchmark_filter:
            if any(b not in best_map for b in benchmark_filter):
                continue

        for benchmark, best_pck in best_map.items():
            benchmark_rows.append(
                {
                    "snapshot": snap_name,
                    "snapshot_path": str(snap),
                    "dataset": dataset,
                    "condition": condition,
                    "benchmark": benchmark,
                    "best_pck": float(best_pck),
                    **metadata,
                }
            )

        macro_vals = list(best_map.values())
        macro_rows.append(
            {
                "snapshot": snap_name,
                "snapshot_path": str(snap),
                "dataset": dataset,
                "condition": condition,
                "macro_best_pck": float(np.mean(macro_vals)),
                "n_benchmarks_used": len(macro_vals),
                **metadata,
            }
        )

    if not benchmark_rows or not macro_rows:
        raise SystemExit(
            "No matching runs found after filters/condition mapping. "
            "Check --condition and --include/--exclude name regex settings."
        )

    bench_df = pd.DataFrame(benchmark_rows)
    macro_df = pd.DataFrame(macro_rows)

    bench_df.to_csv(out_dir / "run_level_benchmark.csv", index=False)
    macro_df.to_csv(out_dir / "run_level_macro.csv", index=False)

    cond_summary_rows = []
    for condition, sub in macro_df.groupby("condition"):
        stats = summarize_series(sub["macro_best_pck"])
        cond_summary_rows.append(
            {
                "condition": condition,
                "n_runs": stats["n_runs"],
                "mean_macro_pck": stats["mean"],
                "std_macro_pck": stats["std"],
                "sem_macro_pck": stats["sem"],
                "ci95_macro_pck": stats["ci95"],
                "ci95_low_macro_pck": stats["ci95_low"],
                "ci95_high_macro_pck": stats["ci95_high"],
            }
        )
    condition_summary = pd.DataFrame(cond_summary_rows)
    if not condition_summary.empty:
        condition_summary["order"] = condition_summary["condition"].apply(
            lambda c: condition_order.index(c) if c in condition_order else 10_000
        )
        condition_summary = condition_summary.sort_values("order").drop(columns=["order"])
    condition_summary.to_csv(out_dir / "condition_summary.csv", index=False)

    bench_summary_rows = []
    for (benchmark, condition), sub in bench_df.groupby(["benchmark", "condition"]):
        stats = summarize_series(sub["best_pck"])
        bench_summary_rows.append(
            {
                "benchmark": benchmark,
                "condition": condition,
                "n_runs": stats["n_runs"],
                "mean_pck": stats["mean"],
                "std_pck": stats["std"],
                "sem_pck": stats["sem"],
                "ci95_pck": stats["ci95"],
                "ci95_low_pck": stats["ci95_low"],
                "ci95_high_pck": stats["ci95_high"],
            }
        )
    benchmark_summary = pd.DataFrame(bench_summary_rows)
    if not benchmark_summary.empty:
        benchmark_summary["condition_order"] = benchmark_summary["condition"].apply(
            lambda c: condition_order.index(c) if c in condition_order else 10_000
        )
        benchmark_summary = benchmark_summary.sort_values(["benchmark", "condition_order"]).drop(
            columns=["condition_order"]
        )
    benchmark_summary.to_csv(out_dir / "benchmark_condition_summary.csv", index=False)
    benchmark_summary_pretty_csv = benchmark_summary.copy()
    cond_map_csv = _pretty_condition_map(condition_order)
    benchmark_summary_pretty_csv["condition"] = benchmark_summary_pretty_csv["condition"].astype(str).map(
        lambda x: cond_map_csv.get(x, x)
    )
    benchmark_summary_pretty_csv["benchmark"] = benchmark_summary_pretty_csv["benchmark"].astype(str).map(
        _pretty_benchmark_name
    )
    benchmark_summary_pretty_csv.to_csv(out_dir / "benchmark_condition_summary_paper.csv", index=False)

    final_table = _build_display_table(
        condition_summary=condition_summary,
        benchmark_summary=benchmark_summary,
        condition_order=condition_order,
        mean_macro_col="mean_macro_pck",
        std_macro_col="std_macro_pck",
        mean_bench_col="mean_pck",
        std_bench_col="std_pck",
    )
    final_table.to_csv(out_dir / "final_table_raw.csv", index=False)
    final_table_pretty, cond_cols_pretty = _prettify_table(
        final_table,
        condition_order=condition_order,
        benchmark_col_label=benchmark_col_label,
    )
    final_table_pretty.to_csv(out_dir / "final_table.csv", index=False)
    write_latex_table(
        final_table_pretty,
        cond_cols_pretty,
        out_dir / "final_table.tex",
        benchmark_col=benchmark_col_label,
    )

    condition_summary_pretty, benchmark_summary_pretty, condition_order_pretty = _prettify_summary_for_plots(
        condition_summary,
        benchmark_summary,
        condition_order,
    )
    condition_summary_pretty.to_csv(out_dir / "condition_summary_paper.csv", index=False)
    make_overall_plot(
        condition_summary_pretty,
        condition_order_pretty,
        out_dir / "bar_overall_macro.png",
        pck_label=pck_label,
    )
    if not args.no_benchmark_plot:
        make_benchmark_plot(
            benchmark_summary_pretty,
            condition_order_pretty,
            out_dir / "bar_by_benchmark.png",
            pck_label=pck_label,
        )

    if not args.no_stratified_report:
        build_stratified_outputs(
            macro_df=macro_df,
            bench_df=bench_df,
            condition_order=condition_order,
            stratum_keys=stratum_keys,
            out_dir=out_dir,
            make_benchmark_plot_flag=not args.no_benchmark_plot,
            pck_label=pck_label,
            benchmark_col_label=benchmark_col_label,
        )

    print(f"Scanned snapshot dirs: {len(snapshots)}")
    print(f"Matched runs: {len(macro_df)}")
    for condition in condition_order:
        n_runs = int((macro_df["condition"] == condition).sum())
        print(f"  {condition}: {n_runs} runs")
    print(f"Wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
