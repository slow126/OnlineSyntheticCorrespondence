#!/usr/bin/env python3
"""
Build the final parameter-matched LOTO win-rate table for the paper.

Reads the parameter_matched_selection.csv produced by the sweep plots script
and emits a clean LaTeX table + CSV for inclusion in the paper.

Usage:
    python scripts/build_final_results_table.py \
        --sweep-csv  <path>/parameter_matched_selection.csv \
        --output-dir <path>/final_tables/paper_results

Row selection is declared explicitly in ROW_DEFS below — edit that list to
change which configurations appear in the table and in what order.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Row definitions: which configurations to show and how to label them.
#
# Each entry maps a `bucket` value (from parameter_matched_selection.csv) to
# a human-readable label and a display group.  Rows are emitted in the order
# listed; a \midrule + italic group header is inserted whenever the group
# changes.
#
# Edit this list to add/remove rows or change groupings.
# ---------------------------------------------------------------------------
ROW_DEFS: List[Dict[str, str]] = [
    # --- Symmetric MMD baselines -------------------------------------------
    dict(
        bucket="k01__mmd_appearance_only",
        label="Appearance MMD (symmetric)",
        group="Symmetric baselines",
    ),
    dict(
        bucket="k01__mmd_flow_only",
        label="Flow MMD (symmetric)",
        group="Symmetric baselines",
    ),
    # --- Single directed predictor per modality ----------------------------
    dict(
        bucket="k01__appearance_only_a1",
        label="Appearance, 1 directed",
        group="Single directed predictor",
    ),
    dict(
        bucket="k01__flow_only_f1",
        label="Flow, 1 directed",
        group="Single directed predictor",
    ),
    # --- Bidirectional (both directions, same modality) --------------------
    dict(
        bucket="k02__appearance_only_a2",
        label=r"Appearance, bidirectional (train$\rightarrow$eval + eval$\rightarrow$train)",
        group="Bidirectional single modality",
    ),
    dict(
        bucket="k02__hof_only_h2",
        label="HOF, bidirectional",
        group="Bidirectional single modality",
    ),
    dict(
        bucket="k02__flow_only_f2",
        label="Flow, bidirectional",
        group="Bidirectional single modality",
    ),
    # --- Combined motion + appearance --------------------------------------
    dict(
        bucket="k03__hybrid_f2_a1",
        label="Flow (bidir.) + Appearance (1 dir.)",
        group="Motion + appearance",
    ),
    dict(
        bucket="k04__hybrid_h2_a2",
        label="HOF (bidir.) + Appearance (bidir.)",
        group="Motion + appearance",
    ),
    dict(
        bucket="k04__hybrid_f3_a1",
        label="Flow (bidir.) + Appearance (1 dir.) + HOF (1 dir.)",
        group="Motion + appearance",
    ),
    dict(
        bucket="k04__hybrid_f2_a2",
        label="Flow (bidir.) + Appearance (bidir.)",
        group="Motion + appearance",
        bold_row=True,
    ),
    dict(
        bucket="k04__flow_only_f4",
        label="Flow (bidir.) + HOF (bidir.)",
        group="Motion + motion",
    ),
]

# Optional add-on row: density-only baseline from a separate sweep export.
KNOWN_DENSITY_CONTROLS = {
    "log_n_samples_eval",
    "log_n_samples_train",
    "log_avg_flows_eval",
    "log_avg_flows_train",
}

DENSITY_ONLY_DEFAULT = dict(
    bucket="k04__other_only__other4",
    label="Density-only (z-scored stats)",
    group="Density-only baseline",
)


def _predictor_short(predictors: str) -> str:
    """Compact representation of the predictor list for the table."""
    preds = [p.strip() for p in predictors.split(",") if p.strip()]
    flow = [p for p in preds if p.startswith("flow_") and "mmd" not in p]
    appear = [p for p in preds if p.startswith("dino_") and "mmd" not in p]
    hof = [p for p in preds if p.startswith("hof_") and "mmd" not in p]
    mmd = [p for p in preds if "mmd" in p]
    used = set(flow + appear + hof + mmd)
    other = [p for p in preds if p not in used]
    parts: List[str] = []
    if flow:
        parts.append(rf"F$\times${len(flow)}")
    if appear:
        parts.append(rf"A$\times${len(appear)}")
    if hof:
        parts.append(rf"H$\times${len(hof)}")
    if other:
        parts.append(rf"O$\times${len(other)}")
    if mmd:
        parts.append(rf"MMD$\times${len(mmd)}")
    return " + ".join(parts) if parts else "—"


def _safe_float(value: object) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _is_density_control_predictor(predictor: str) -> bool:
    p = predictor.strip()
    return p in KNOWN_DENSITY_CONTROLS or p.startswith("log_")


def _base_predictors(predictors: str) -> List[str]:
    preds = [p.strip() for p in predictors.split(',') if p.strip()]
    out = []
    for p in preds:
        if _is_density_control_predictor(p):
            continue
        if p.endswith("_mmd") or "mmd" in p:
            continue
        out.append(p)
    return out


def _direction_label_for_predictor(predictor: str) -> str:
    """Return a compact direction label if the predictor is directional."""
    p = predictor.strip()
    if "train_to_eval" in p and "eval_to_train" not in p:
        return "train$\\to$eval"
    if "eval_to_train" in p and "train_to_eval" not in p:
        return "eval$\\to$train"
    return ""


def _interactions_row_uses_avg_flows(row: pd.Series) -> bool:
    """Check whether a selected interaction run uses flow-count density interactions."""
    run_dir = Path(str(row.get("run_dir", "")))
    meta_path = run_dir / "run_metadata.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False

    custom = meta.get("custom_interactions", "") or ""
    if isinstance(custom, str) and "log_avg_flows" in custom:
        return True

    # Keep compatibility with older metadata layouts where this may be stored in args.
    args_ci = meta.get("custom_interactions_override", "")
    if isinstance(args_ci, str) and "log_avg_flows" in args_ci:
        return True

    if meta.get("flow_density_interactions", False):
        return True

    return False


def _interaction_selection_row_complete(row: pd.Series, *, require_summary_files: bool = True) -> bool:
    """Basic sanity check that an interaction selection row points to a finished run."""
    run_dir = Path(str(row.get("run_dir", "")))
    if not run_dir or not run_dir.exists():
        return False

    metric = row.get("metric_value")
    if not math.isfinite(_safe_float(metric)):
        return False

    if not require_summary_files:
        return True

    required = [
        "prediction_loto_holdout_placement_summary.csv",
        "prediction_lobo_rank_summary.csv",
        "prediction_jointood_holdout_placement_summary.csv",
    ]
    for fname in required:
        if not (run_dir / fname).exists():
            return False
    return True


def _infer_interaction_suffix_from_row(row: pd.Series) -> str:
    run_dir = Path(str(row.get("run_dir", "")))
    bucket = str(row.get("bucket", ""))
    if not bucket:
        return ""
    name = run_dir.name
    prefix = f"leakage_free_{bucket}"
    if name.startswith(prefix):
        return name[len(prefix) :]
    return ""


def _read_interaction_replay_readme(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    out: Dict[str, str] = {}
    for raw in path.read_text().splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _run_command(
    cmd: List[str],
    *,
    dry_run: bool = False,
    label: str = "",
) -> int:
    """Run a subprocess command with a compact debug print."""
    if label:
        print(label)
    pretty = " ".join(shlex.quote(x) for x in cmd)
    if dry_run:
        print(f"[dry-run] {pretty}")
        return 0
    print(pretty)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {pretty}")
    return result.returncode


def _direction_key_for_predictor(predictor: str) -> str:
    p = predictor.strip()
    if "train_to_eval" in p and "eval_to_train" not in p:
        return "train_to_eval"
    if "eval_to_train" in p and "train_to_eval" not in p:
        return "eval_to_train"
    return ""


def _single_predictor_label(predictor: str) -> str:
    if predictor.startswith("dino_") and "mmd" not in predictor:
        return "Appearance, 1 directed"
    if predictor.startswith("flow_") and "mmd" not in predictor:
        return "Flow, 1 directed"
    if predictor.startswith("hof_") and "mmd" not in predictor:
        return "HOF, 1 directed"
    return "Single directed"


def _single_predictor_sort_key(predictor: str) -> tuple:
    direction_order = {"eval_to_train": 0, "train_to_eval": 1, "": 2}
    if predictor.startswith("dino_") and "mmd" not in predictor:
        family_order = 0
    elif predictor.startswith("hof_") and "mmd" not in predictor:
        family_order = 1
    elif predictor.startswith("flow_") and "mmd" not in predictor:
        family_order = 2
    else:
        family_order = 9
    direction = _direction_key_for_predictor(predictor)
    return family_order, direction_order.get(direction, 2), predictor


def _safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _predictors_from_metadata(meta: Dict[str, Any]) -> List[str]:
    if not isinstance(meta, dict):
        return []
    preds = meta.get("predictors", [])
    if isinstance(preds, str):
        return [p.strip() for p in preds.split(",") if p.strip()]
    if isinstance(preds, (list, tuple)):
        return [str(p).strip() for p in preds if str(p).strip()]
    return []


def _parse_predictor_line(report_text: str) -> List[str]:
    marker = "Predictors:"
    for line in report_text.splitlines():
        if marker not in line:
            continue
        part = line.split(marker, 1)[1].strip()
        if not part:
            return []
        return [p.strip() for p in part.split(",") if p.strip()]
    return []


def _normalise_predictors(raw_predictors: object) -> List[str]:
    if raw_predictors is None:
        return []
    try:
        if pd.isna(raw_predictors):
            return []
    except Exception:
        pass
    if isinstance(raw_predictors, str):
        if not raw_predictors.strip() or raw_predictors.strip().lower() == "nan":
            return []
        return [p.strip() for p in raw_predictors.split(",") if p.strip()]
    if isinstance(raw_predictors, (list, tuple)):
        return [str(p).strip() for p in raw_predictors if str(p).strip()]
    return []


def _signal_predictor_signature(predictors: object) -> str:
    """Return a canonical predictor tuple string for non-density, non-MMD signals."""
    sig = []
    for p in _normalise_predictors(predictors):
        if _is_density_control_predictor(p):
            continue
        pl = p.lower()
        if "mmd" in pl:
            continue
        sig.append(p)
    return ";".join(sig)


def _row_signal_signature(row: pd.Series) -> str:
    """Pull the signal predictor signature from a selection row."""
    predictors = str(row.get("predictors", ""))
    sig = _signal_predictor_signature(predictors)
    if sig:
        return sig

    # Some rows may miss explicit predictor text but still have summary output.
    run_dir = Path(str(row.get("run_dir", "")))
    summary_path = run_dir / "summary_report.txt"
    if summary_path.exists():
        report_text = summary_path.read_text(errors="ignore")
        preds = _parse_predictor_line(report_text)
        sig = _signal_predictor_signature(preds)
        if sig:
            return sig

    return ""


def _row_key_from_defn(defn: Dict[str, Any], row: pd.Series) -> tuple[str, str]:
    predictor_key = str(defn.get("predictor_key", row.get("predictors", "")))
    signature = _signal_predictor_signature(predictor_key)
    if not signature:
        signature = _row_signal_signature(row)
    return str(row.get("bucket", defn.get("bucket", ""))), signature


def _infer_density_joint_root(sweep_csv: Path, df: pd.DataFrame) -> Optional[Path]:
    sweep_root = sweep_csv.resolve().parent
    candidate = sweep_root.parent / "density_joint"
    if candidate.exists():
        return candidate

    if sweep_root.name.startswith("paper_plots_") and sweep_root.parent.exists():
        alt = sweep_root.parent / "density_joint"
        if alt.exists():
            return alt

    for raw in df.get("run_dir", []):
        run_dir = Path(str(raw))
        if run_dir.name.startswith("leakage_free_") and run_dir.parent.name == "density_joint":
            return run_dir.parent

    # Fallback to common root style used in this repo if present.
    if "analysis_comprehensive_runs" in str(sweep_root):
        for part in sweep_root.parents:
            if part.name == "analysis_comprehensive_runs":
                alt = part / "density_joint"
                if alt.exists():
                    return alt
                break

    return None


def _read_best_overall_metric(run_dir: Path) -> float:
    summary_candidates = [
        ("prediction_loto_holdout_placement_summary.csv", ["pairwise_win_rate", "pairwise_win_rate_micro", "pairwise_cindex"]),
    ]
    for name, cols in summary_candidates:
        path = run_dir / name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        for col in cols:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            return float(vals.mean())

    # Final fallback: if summaries are malformed, derive from top-line report if possible.
    report = run_dir / "summary_report.txt"
    if not report.exists():
        return float("nan")
    for line in report.read_text(errors="ignore").splitlines():
        if "LOTO holdout placement" in line and "pair_win=" in line:
            # keep reading style for legacy reports if present
            marker = "pair_win="
            idx = line.find(marker)
            if idx >= 0:
                frag = line[idx + len(marker):]
                token = frag.split(",", 1)[0].strip()
                try:
                    return float(token)
                except ValueError:
                    continue
    return float("nan")


def _build_interaction_lookup(df: pd.DataFrame) -> tuple[Dict[tuple[str, str], pd.Series], Dict[str, List[pd.Series]]]:
    """Build interaction lookup tables keyed by (bucket, signal_signature) and bucket."""
    by_key: Dict[tuple[str, str], pd.Series] = {}
    by_bucket: Dict[str, List[pd.Series]] = defaultdict(list)
    if df is None or df.empty:
        return by_key, by_bucket

    for _, row in df.iterrows():
        bucket = str(row.get("bucket", ""))
        signature = _row_signal_signature(row)
        if not bucket:
            continue
        by_key[(bucket, signature)] = row
        by_bucket[bucket].append(row)
    return by_key, by_bucket


def _discover_single_hof_candidates(
    density_joint_root: Optional[Path],
) -> Dict[str, Dict[str, Any]]:
    """Discover one-directional HOF-only leakage-free runs.

    Returns a mapping bucket -> synthetic selection row dict.
    """
    if density_joint_root is None or not density_joint_root.exists():
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for run_dir in sorted(density_joint_root.glob("leakage_free_*")):
        if not run_dir.is_dir():
            continue

        meta = _safe_read_json(run_dir / "run_metadata.json")
        preds = _predictors_from_metadata(meta)

        # Fallback to the summary report where metadata is missing.
        if len(preds) != 1:
            summary_path = run_dir / "summary_report.txt"
            if summary_path.exists():
                preds = _parse_predictor_line(summary_path.read_text(errors="ignore"))

        if len(preds) != 1:
            continue

        predictor = _normalise_predictors(preds)[0]
        if not predictor.startswith("hof_"):
            continue

        direction = _direction_key_for_predictor(predictor)
        if direction not in {"train_to_eval", "eval_to_train"}:
            continue

        bucket = f"k01__hof_only_h1_{direction}"
        if bucket in out:
            continue

        out[bucket] = {
            "run_dir": str(run_dir),
            "summary_path": str(run_dir / "summary_report.txt"),
            "metric_value": _read_best_overall_metric(run_dir),
            "metric_col_used": "pairwise_win_rate",
            "predictors": predictor,
            "bucket": bucket,
            "k": 1,
            "n_flow": 0,
            "n_appearance": 0,
            "n_other": 0,
            "has_flow_mmd": False,
            "has_appearance_mmd": False,
            "has_other_mmd": False,
        }
    return out


def _append_direction_to_label(label: str, direction: str) -> str:
    if not direction:
        return label
    return rf"{label} ({direction})"


def _read_overall_metric(
    summary_path: Path,
    *,
    id_col: str,
    overall_token: str,
    metric_cols: List[str],
) -> float:
    if not summary_path.exists():
        return float("nan")
    try:
        df = pd.read_csv(summary_path)
    except Exception:
        return float("nan")
    if df.empty:
        return float("nan")
    work = df
    if id_col in work.columns:
        sub = work[work[id_col].astype(str) == overall_token]
        if not sub.empty:
            work = sub
    for col in metric_cols:
        if col not in work.columns:
            continue
        vals = pd.to_numeric(work[col], errors="coerce").dropna()
        if vals.empty:
            continue
        return float(vals.iloc[0])
    return float("nan")


def _format_pct(v: object) -> str:
    x = _safe_float(v)
    return "--" if not math.isfinite(x) else f"{x:.1f}\\%"


def _format_delta_pp(v: object) -> str:
    x = _safe_float(v)
    return "--" if not math.isfinite(x) else f"{x:+.1f}"


def _maybe_bold_cell(text: str, bold: bool) -> str:
    if not bold:
        return text
    return rf"\textbf{{{text}}}"


def _write_latex(
    rows: pd.DataFrame,
    out_path: Path,
    caption: str,
    label: str,
    *,
    include_heldout_columns: bool,
    include_interaction_uplift: bool,
) -> None:
    if include_heldout_columns:
        if include_interaction_uplift:
            tabular_spec = "lcrrrrr"
            header = (
                r"\textbf{Configuration} & \textbf{Predictors} & \textbf{$k$} & "
                r"\textbf{Held-out $T$} & \textbf{Held-out $B$} & \textbf{Held-out $(T,B)$} & "
                r"\textbf{$\Delta$ Int.-only (pp)} \\"
            )
            ncols = 7
        else:
            tabular_spec = "lcrrrr"
            header = (
                r"\textbf{Configuration} & \textbf{Predictors} & \textbf{$k$} & "
                r"\textbf{Held-out $T$} & \textbf{Held-out $B$} & \textbf{Held-out $(T,B)$} \\"
            )
            ncols = 6
    else:
        if include_interaction_uplift:
            tabular_spec = "lcrrr"
            header = (
                r"\textbf{Configuration} & \textbf{Predictors} & \textbf{$k$} & "
                r"\textbf{Win rate} & \textbf{$\Delta$ Int.-only (pp)} \\"
            )
            ncols = 5
        else:
            tabular_spec = "lcrr"
            header = r"\textbf{Configuration} & \textbf{Predictors} & \textbf{$k$} & \textbf{Win rate} \\"
            ncols = 4

    lines: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{tabular_spec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    prev_group = None
    for _, r in rows.iterrows():
        group = str(r["group"])
        if group != prev_group:
            if prev_group is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{group}}}}} \\")
            prev_group = group

        k_val = int(r["k"])
        pred_str = str(r["predictor_short"])
        label_str = str(r["label"])
        bold_row = bool(r.get("bold_row", False))
        if include_heldout_columns:
            loto = _maybe_bold_cell(_format_pct(r.get("loto_win_rate")), bold_row)
            lobo = _maybe_bold_cell(_format_pct(r.get("lobo_win_rate")), bold_row)
            joint = _maybe_bold_cell(_format_pct(r.get("joint_win_rate")), bold_row)
            k_txt = _maybe_bold_cell(str(k_val), bold_row)
            pred_txt = _maybe_bold_cell(pred_str, bold_row)
            label_txt = _maybe_bold_cell(label_str, bold_row)
            if include_interaction_uplift:
                uplift = _maybe_bold_cell(_format_delta_pp(r.get("interaction_uplift_pp")), bold_row)
                lines.append(
                    rf"{label_txt} & {pred_txt} & {k_txt} & {loto} & {lobo} & {joint} & {uplift} \\"
                )
            else:
                lines.append(rf"{label_txt} & {pred_txt} & {k_txt} & {loto} & {lobo} & {joint} \\")
        else:
            win_pct = _maybe_bold_cell(_format_pct(r.get("win_rate")), bold_row)
            k_txt = _maybe_bold_cell(str(k_val), bold_row)
            pred_txt = _maybe_bold_cell(pred_str, bold_row)
            label_txt = _maybe_bold_cell(label_str, bold_row)
            if include_interaction_uplift:
                uplift = _maybe_bold_cell(_format_delta_pp(r.get("interaction_uplift_pp")), bold_row)
                lines.append(rf"{label_txt} & {pred_txt} & {k_txt} & {win_pct} & {uplift} \\")
            else:
                lines.append(rf"{label_txt} & {pred_txt} & {k_txt} & {win_pct} \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build final paper results table")
    parser.add_argument(
        "--sweep-csv",
        required=True,
        help="Path to parameter_matched_selection.csv from the sweep plot script",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    parser.add_argument(
        "--win-rate-scale",
        choices=["fraction", "percent"],
        default="percent",
        help="Report win rate as fraction [0,1] or percentage [0,100]. Default: percent.",
    )
    parser.add_argument(
        "--caption",
        default=(
            r"\textbf{Parameter-matched pairwise ranking performance across held-out protocols (macro averages).} "
            r"Rows use a fixed predictor budget $k$ with one representative configuration per bucket. "
            r"Columns report within-context pairwise concordance (c-index; higher is better; chance = 0.5) when "
            r"holding out an entire training dataset (unseen candidate $T$), holding out an entire benchmark "
            r"(unseen target $B$), or holding out a training--benchmark pair (unseen combination $(T,B)$)."
        ),
        help="LaTeX caption string.",
    )
    parser.add_argument(
        "--include-heldout-columns",
        action="store_true",
        help=(
            "Add LOBO and Joint-OOD columns (plus LOTO) for each selected configuration. "
            "Without this flag, emit the original single Win rate column."
        ),
    )
    parser.add_argument(
        "--heldout-win-rate-mode",
        choices=["micro", "macro"],
        default="micro",
        help=(
            "When --include-heldout-columns is enabled, choose whether LOTO/Joint-OOD "
            "columns read pairwise win rate from micro or macro overall fields. "
            "LOBO remains pairwise_cindex from ranking summary."
        ),
    )
    parser.add_argument(
        "--interactions-sweep-csv",
        default="",
        help=(
            "Optional parameter_matched_selection.csv from interaction-only runs. "
            "When provided, add interaction uplift columns in CSV and (by default) LaTeX."
        ),
    )
    parser.add_argument(
        "--auto-generate-missing-interactions",
        action="store_true",
        help=(
            "If set, attempt to regenerate missing interaction rows before building the table "
            "using replay_parameter_matched_interactions.py + run_plot_residual_rank_param_matched.py."
        ),
    )
    parser.add_argument(
        "--interactions-output-root",
        default="",
        help=(
            "Root directory for interaction runs (where to write density_joint/...)."
        ),
    )
    parser.add_argument(
        "--interactions-baseline-root",
        default="",
        help=(
            "Baseline run root passed through to replay_parameter_matched_interactions.py."
            " If not provided, infer from --interactions-output-root metadata when possible."
        ),
    )
    parser.add_argument(
        "--interactions-run-suffix",
        default="__density_as_interactions",
        help=(
            "Suffix for replayed interaction directories and for selection-run-root mapping."
        ),
    )
    parser.add_argument(
        "--interactions-density-controls-mode",
        default="all",
        choices=["all", "samples_only"],
        help=(
            "Density-control preset for replayed interaction runs: 'all' (n_samples + avg_flows) "
            "or 'samples_only' (log_n_samples only)."
        ),
    )
    parser.add_argument(
        "--interactions-ridge-alpha",
        type=float,
        default=10.0,
        help="Ridge alpha passed to interaction replay.",
    )
    parser.add_argument(
        "--interactions-cv-residual-target-transform",
        default="zscore",
        choices=["zscore", "residual"],
        help="Residual transform passed to interaction replay.",
    )
    parser.add_argument(
        "--reject-avg-flow-interactions",
        action="store_true",
        help=(
            "If set, skip interaction offset rows whose run metadata shows density interactions "
            "using log_avg_flows terms. This is useful to enforce n-samples-only interactions "
            "for scale modulation ablations."
        ),
    )
    parser.add_argument(
        "--no-interaction-uplift-column",
        action="store_true",
        help=(
            "Disable the LaTeX interaction uplift column even when --interactions-sweep-csv is provided. "
            "CSV deltas are still exported."
        ),
    )
    parser.add_argument(
        "--label-single-directed-directions",
        action="store_true",
        help=(
            "Append explicit direction labels (train->eval vs eval->train) to single-directed rows "
            "using direction names found in predictor identifiers."
        ),
    )
    parser.add_argument(
        "--expand-single-directed-rows",
        action="store_true",
        help=(
            "Discover and include all single-predictor directional rows from the base sweep CSV "
            "in place of the two hardcoded single-directed rows."
        ),
    )
    parser.add_argument(
        "--density-only-sweep-csv",
        default="",
        help=(
            "Optional parameter_matched_selection.csv that contains a density-only row "
            "(default bucket: k04__other_only__other4)."
        ),
    )
    parser.add_argument(
        "--density-only-bucket",
        default=DENSITY_ONLY_DEFAULT["bucket"],
        help="Bucket key used for the optional density-only row.",
    )
    parser.add_argument(
        "--density-only-label",
        default=DENSITY_ONLY_DEFAULT["label"],
        help="Display label for the optional density-only row.",
    )
    parser.add_argument(
        "--density-only-group",
        default=DENSITY_ONLY_DEFAULT["group"],
        help="Display group for the optional density-only row.",
    )
    args = parser.parse_args()

    sweep_csv = Path(args.sweep_csv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(sweep_csv)

    density_df: Optional[pd.DataFrame] = None
    if args.density_only_sweep_csv:
        density_path = Path(args.density_only_sweep_csv)
        if not density_path.exists():
            raise SystemExit(f"--density-only-sweep-csv not found: {density_path}")
        density_df = pd.read_csv(density_path)

    density_joint_root = _infer_density_joint_root(sweep_csv, df)

    row_defs: List[Dict] = list(ROW_DEFS)
    if args.expand_single_directed_rows:
        discovered_single_rows: List[Dict] = []
        for _, r in df.iterrows():
            bucket = str(r["bucket"])
            try:
                n_flow = _safe_int(r.get("n_flow"), default=0)
                n_appearance = _safe_int(r.get("n_appearance"), default=0)
                n_other = _safe_int(r.get("n_other"), default=0)
                n_hof = _safe_int(r.get("n_hof"), default=0)
            except Exception:
                continue
            signal_predictors = _base_predictors(str(r.get("predictors", "")))
            if len(signal_predictors) != 1:
                continue
            # fallback for legacy CSVs where only aggregate n_* counts are available:
            # require only one non-density, non-density-control signal predictor.
            if n_flow + n_appearance + n_hof + n_other > 1 and n_flow + n_appearance + n_other <= 1:
                continue
            predictor = signal_predictors[0]
            if not (
                predictor.startswith("dino_")
                or predictor.startswith("flow_")
                or predictor.startswith("hof_")
            ):
                continue
            label = _single_predictor_label(predictor)
            discovered_single_rows.append(
                dict(
                    bucket=bucket,
                    label=label,
                    group="Single directed predictor",
                    predictor_key=predictor,
                )
            )

        if discovered_single_rows:
            synthetic_hof = _discover_single_hof_candidates(density_joint_root)
            for bucket, row_data in sorted(synthetic_hof.items(), key=lambda kv: kv[0]):
                if any(d["bucket"] == bucket for d in discovered_single_rows):
                    continue
                discovered_single_rows.append(
                    {
                        "bucket": bucket,
                        "label": _single_predictor_label(str(row_data["predictors"])),
                        "group": "Single directed predictor",
                        "predictor_key": str(row_data["predictors"]),
                        "_synthetic_row": row_data,
                    }
                )

            discovered_single_rows = sorted(
                discovered_single_rows,
                key=lambda d: _single_predictor_sort_key(str(d["predictor_key"])),
            )

            # Replace the single-directed section entirely with discovered rows.
            rebuilt_defs: List[Dict] = []
            inserted = False
            for defn in row_defs:
                if defn.get("group") == "Single directed predictor":
                    if not inserted:
                        rebuilt_defs.extend(discovered_single_rows)
                        inserted = True
                    continue
                rebuilt_defs.append(defn)
            if not inserted:
                rebuilt_defs.extend(discovered_single_rows)
            row_defs = rebuilt_defs
        else:
            print(
                "WARNING: no additional single-directional rows discovered from --sweep-csv.\n"
                "         If you expected HOF 1-direction rows, check for buckets such as\n"
                "         k01__hof_only_h1[_train_to_eval] in the source selection CSV."
            )

    else:
        # Sweep-based discovery found nothing; still add synthetic HOF singles if
        # discoverable from completed runs.
        synthetic_hof = _discover_single_hof_candidates(density_joint_root)
        discovered_single_rows = []
        for bucket, row_data in sorted(synthetic_hof.items(), key=lambda kv: kv[0]):
            discovered_single_rows.append(
                {
                    "bucket": bucket,
                    "label": _single_predictor_label(str(row_data["predictors"])),
                    "group": "Single directed predictor",
                    "predictor_key": str(row_data["predictors"]),
                    "_synthetic_row": row_data,
                }
            )

        if discovered_single_rows:
            row_defs = row_defs[:2] + discovered_single_rows + row_defs[2:]

    if density_df is not None:
        row_defs.append(
            dict(
                bucket=args.density_only_bucket,
                label=args.density_only_label,
                group=args.density_only_group,
                source="density",
            )
        )

    # Inject synthetic base rows into the in-memory base sweep to support rows
    # that are only materialized as standalone completed runs on disk (e.g. one-
    # direction HOF predictors).
    if args.expand_single_directed_rows:
        synthetic_base_rows = [
            dict(defn["_synthetic_row"])
            for defn in row_defs
            if defn.get("_synthetic_row") and defn.get("_synthetic_row") is not None
        ]
        if synthetic_base_rows:
            synthetic_df = pd.DataFrame(synthetic_base_rows)
            # Preserve all existing CSV rows; synthetic rows are only added when
            # missing from the source selection.
            existing_buckets = set(df["bucket"].astype(str))
            synthetic_df = synthetic_df[~synthetic_df["bucket"].astype(str).isin(existing_buckets)]
            if not synthetic_df.empty:
                df = pd.concat([df, synthetic_df], ignore_index=True)

    # Build lookup from bucket → row in the base/density selection CSVs.
    bucket_to_row: Dict[str, pd.Series] = {
        str(r["bucket"]): r for _, r in df.iterrows()
    }
    density_bucket_to_row: Dict[str, pd.Series] = {}
    if density_df is not None:
        density_bucket_to_row = {
            str(r["bucket"]): r for _, r in density_df.iterrows()
        }

    required_interaction_reqs: List[tuple[str, str, pd.Series, Dict[str, str]]] = []
    for defn in row_defs:
        if defn.get("source", "base") == "density":
            continue
        bucket = str(defn["bucket"])
        if bucket not in bucket_to_row:
            continue
        row = bucket_to_row[bucket]
        required_bucket, sig = _row_key_from_defn(defn, row)
        required_interaction_reqs.append((required_bucket, sig, row, defn))

    interactions_df: Optional[pd.DataFrame] = None
    interactions_path: Optional[Path] = None
    interactions_key_to_row: Dict[tuple[str, str], pd.Series] = {}
    interactions_bucket_to_rows: Dict[str, List[pd.Series]] = {}
    if args.interactions_sweep_csv:
        interactions_path = Path(args.interactions_sweep_csv)
        if interactions_path.exists():
            interactions_df = pd.read_csv(interactions_path)
            interactions_key_to_row, interactions_bucket_to_rows = _build_interaction_lookup(interactions_df)
        elif not args.auto_generate_missing_interactions:
            raise SystemExit(f"--interactions-sweep-csv not found: {interactions_path}")

    interaction_output_root: Optional[Path] = (
        Path(args.interactions_output_root).expanduser()
        if str(args.interactions_output_root).strip()
        else None
    )
    if interaction_output_root is None and interactions_path is not None:
        if interactions_path.parent.name.startswith("paper_plots_") and interactions_path.parent.parent.exists():
            interaction_output_root = interactions_path.parent.parent
        else:
            interaction_output_root = interactions_path.parent

    if (
        args.auto_generate_missing_interactions
        and args.interactions_sweep_csv
        and required_interaction_reqs
        and interaction_output_root is not None
    ):
        from collections import Counter

        existing_keys = {
            (str(row["bucket"]), _row_signal_signature(row))
            for _, row in (interactions_df.iterrows() if interactions_df is not None else [])
            if _interaction_selection_row_complete(row)
        }

        if args.reject_avg_flow_interactions:
            existing_keys = {
                key
                for key in existing_keys
                if not _interactions_row_uses_avg_flows(interactions_key_to_row.get(key, pd.Series(dtype=object)))
            }

        required_keys = {(bucket, sig) for bucket, sig, _, _ in required_interaction_reqs}
        missing_interaction_keys = required_keys - existing_keys

        if missing_interaction_keys:
            print(
                "INFO: missing interaction rows for "
                f"{len(missing_interaction_keys)} row key(s): "
                + ", ".join(f"{b}|{s}" for b, s in sorted(missing_interaction_keys))
            )

            inferred_baseline = args.interactions_baseline_root.strip()
            if not inferred_baseline:
                # Try to recover baseline root from a prior replay README.
                readme_path = interaction_output_root / "README_interaction_replay.txt"
                if readme_path.exists():
                    meta = _read_interaction_replay_readme(readme_path)
                    inferred_baseline = meta.get("baseline_root", "")
                if not inferred_baseline and interaction_output_root.parent.exists():
                    inferred_baseline = str(interaction_output_root.parent)

            interaction_run_suffix = args.interactions_run_suffix
            if not interaction_run_suffix and interactions_df is not None:
                sfx_counts = Counter(
                    _infer_interaction_suffix_from_row(row)
                    for row in interactions_df.itertuples(index=False)
                    if str(getattr(row, "bucket", "")) in {k for k, _ in required_keys}
                )
                if sfx_counts:
                    interaction_run_suffix = sfx_counts.most_common(1)[0][0]

            if not interaction_run_suffix:
                interaction_run_suffix = "__density_as_interactions"

            with tempfile.NamedTemporaryFile(
                "w", suffix=".csv", prefix="required_interactions_", delete=False
            ) as tmp:
                missing_selection_path = Path(tmp.name)
                missing_rows = [
                    dict(base_row)
                    for req_bucket, req_sig, base_row, _ in required_interaction_reqs
                    if (req_bucket, req_sig) in missing_interaction_keys
                ]
                if not missing_rows:
                    raise RuntimeError(
                        "Missing interaction keys were computed, but no matching base rows were found "
                        "to rebuild them."
                    )
                pd.DataFrame(missing_rows).to_csv(missing_selection_path, index=False)

            replay_cmd = [
                "python",
                "scripts/replay_parameter_matched_interactions.py",
                "--selection-csv",
                str(missing_selection_path),
                "--output-root",
                str(interaction_output_root),
                "--baseline-root",
                str(inferred_baseline),
                "--density-controls-mode",
                args.interactions_density_controls_mode,
                "--run-suffix",
                interaction_run_suffix,
                "--ridge-alpha",
                str(args.interactions_ridge_alpha),
                "--cv-residual-target-transform",
                args.interactions_cv_residual_target_transform,
            ]
            _run_command(replay_cmd, label="Running replay for missing interaction buckets")

            plot_cmd = [
                "python",
                "scripts/run_plot_residual_rank_param_matched.py",
                "--run-root",
                str(interaction_output_root / "density_joint"),
                "--selection-csv",
                str(missing_selection_path),
                "--selection-run-root",
                str(interaction_output_root),
                "--selection-run-suffix",
                interaction_run_suffix,
                "--best-cv-metric",
                "loto_pair_win",
                "--output-dir",
                str(interactions_path.parent),
            ]
            _run_command(plot_cmd, label="Rebuilding interaction selection CSV")

            missing_interactions_df = pd.read_csv(interactions_path)
            if interactions_df is not None and not interactions_df.empty:
                keep_existing = interactions_df[
                    ~interactions_df.apply(
                        lambda r: (str(r["bucket"]), _row_signal_signature(r)) in missing_interaction_keys,
                        axis=1,
                    )
                ].copy()
                interactions_df = (
                    pd.concat([keep_existing, missing_interactions_df], ignore_index=True)
                    if not missing_interactions_df.empty
                    else keep_existing
                )
            else:
                interactions_df = missing_interactions_df
            interactions_key_to_row, interactions_bucket_to_rows = _build_interaction_lookup(interactions_df)

            try:
                missing_selection_path.unlink()
            except Exception:
                pass

    if (
        interactions_df is None
        and args.interactions_sweep_csv
        and not args.auto_generate_missing_interactions
    ):
        if interactions_path and interactions_path.exists():
            interactions_df = pd.read_csv(interactions_path)
            interactions_key_to_row, interactions_bucket_to_rows = _build_interaction_lookup(interactions_df)

    include_interaction_uplift = (
        interactions_df is not None and not bool(args.no_interaction_uplift_column)
    )

    # Assemble output rows in declaration order.
    selected: List[Dict] = []
    missing: List[str] = []
    rejected_interactions: List[str] = []
    def maybe_override_label(group: str, defn_label: str, row_data: pd.Series) -> str:
        if not args.label_single_directed_directions:
            return defn_label
        if group != "Single directed predictor":
            return defn_label
        predictors = str(row_data.get("predictors", ""))
        preds = [p.strip() for p in predictors.split(",") if p.strip()]
        if len(preds) != 1:
            return defn_label
        direction = _direction_label_for_predictor(preds[0])
        return _append_direction_to_label(defn_label, direction)

    for defn in row_defs:
        bucket = defn["bucket"]
        source = defn.get("source", "base")
        source_rows = density_bucket_to_row if source == "density" else bucket_to_row
        if bucket not in source_rows:
            missing.append(f"{bucket} [{source}]")
            continue
        row = source_rows[bucket]
        label = maybe_override_label(defn["group"], defn["label"], row)
        win_rate = float(row["metric_value"])
        if args.win_rate_scale == "percent":
            win_rate *= 100.0

        run_dir = Path(str(row["run_dir"]))
        if args.heldout_win_rate_mode == "macro":
            loto_metric_cols = ["pairwise_win_rate", "pairwise_win_rate_micro"]
            joint_metric_cols = ["pairwise_win_rate", "pairwise_win_rate_micro"]
        else:
            loto_metric_cols = ["pairwise_win_rate_micro", "pairwise_win_rate"]
            joint_metric_cols = ["pairwise_win_rate_micro", "pairwise_win_rate"]

        loto_win = _read_overall_metric(
            run_dir / "prediction_loto_holdout_placement_summary.csv",
            id_col="fold",
            overall_token="__overall__",
            metric_cols=loto_metric_cols,
        )
        lobo_win = _read_overall_metric(
            run_dir / "prediction_lobo_rank_summary.csv",
            id_col="benchmark",
            overall_token="__overall__",
            metric_cols=["pairwise_cindex", "top1"],
        )
        joint_win = _read_overall_metric(
            run_dir / "prediction_jointood_holdout_placement_summary.csv",
            id_col="fold",
            overall_token="__overall__",
            metric_cols=joint_metric_cols,
        )
        if args.win_rate_scale == "percent":
            if math.isfinite(loto_win):
                loto_win *= 100.0
            if math.isfinite(lobo_win):
                lobo_win *= 100.0
            if math.isfinite(joint_win):
                joint_win *= 100.0

        interaction_win_rate = float("nan")
        interaction_loto_win = float("nan")
        interaction_lobo_win = float("nan")
        interaction_joint_win = float("nan")
        delta_win_rate = float("nan")
        delta_loto_win = float("nan")
        delta_lobo_win = float("nan")
        delta_joint_win = float("nan")
        interaction_uplift_pp = float("nan")
        defn_signal_signature = _row_key_from_defn(defn, row)[1]
        interaction_lookup_row = None
        if interactions_df is not None:
            interaction_lookup_row = interactions_key_to_row.get((bucket, defn_signal_signature))
            if interaction_lookup_row is None and not defn_signal_signature:
                # Fallback: if exact signature missing (e.g. old legacy file), use any
                # complete row from same bucket. This is best-effort only.
                bucket_rows = interactions_bucket_to_rows.get(bucket, [])
                if bucket_rows:
                    for cand in bucket_rows:
                        if _interaction_selection_row_complete(cand):
                            interaction_lookup_row = cand
                            break
            if interaction_lookup_row is not None:
                if defn_signal_signature and _signal_predictor_signature(interaction_lookup_row.get("predictors", "")) != defn_signal_signature:
                    # Preserve strict matching for directional/single predictor rows.
                    interaction_lookup_row = None
        interaction_source_ok = interaction_lookup_row is not None
        if args.reject_avg_flow_interactions and interaction_source_ok:
            int_row = interaction_lookup_row
            if _interactions_row_uses_avg_flows(int_row):
                rejected_interactions.append(bucket)
                interaction_source_ok = False

        if interaction_source_ok:
            int_row = interaction_lookup_row
            interaction_win_rate = float(int_row["metric_value"])
            if args.win_rate_scale == "percent":
                interaction_win_rate *= 100.0

            int_run_dir = Path(str(int_row["run_dir"]))
            interaction_loto_win = _read_overall_metric(
                int_run_dir / "prediction_loto_holdout_placement_summary.csv",
                id_col="fold",
                overall_token="__overall__",
                metric_cols=loto_metric_cols,
            )
            interaction_lobo_win = _read_overall_metric(
                int_run_dir / "prediction_lobo_rank_summary.csv",
                id_col="benchmark",
                overall_token="__overall__",
                metric_cols=["pairwise_cindex", "top1"],
            )
            interaction_joint_win = _read_overall_metric(
                int_run_dir / "prediction_jointood_holdout_placement_summary.csv",
                id_col="fold",
                overall_token="__overall__",
                metric_cols=joint_metric_cols,
            )
            if args.win_rate_scale == "percent":
                if math.isfinite(interaction_loto_win):
                    interaction_loto_win *= 100.0
                if math.isfinite(interaction_lobo_win):
                    interaction_lobo_win *= 100.0
                if math.isfinite(interaction_joint_win):
                    interaction_joint_win *= 100.0

            if math.isfinite(interaction_win_rate) and math.isfinite(win_rate):
                delta_win_rate = interaction_win_rate - win_rate
            if math.isfinite(interaction_loto_win) and math.isfinite(loto_win):
                delta_loto_win = interaction_loto_win - loto_win
            if math.isfinite(interaction_lobo_win) and math.isfinite(lobo_win):
                delta_lobo_win = interaction_lobo_win - lobo_win
            if math.isfinite(interaction_joint_win) and math.isfinite(joint_win):
                delta_joint_win = interaction_joint_win - joint_win

            heldout_deltas = [delta_loto_win, delta_lobo_win, delta_joint_win]
            finite_heldout_deltas = [d for d in heldout_deltas if math.isfinite(d)]
            if finite_heldout_deltas:
                interaction_uplift_pp = float(sum(finite_heldout_deltas) / len(finite_heldout_deltas))
            elif math.isfinite(delta_win_rate):
                interaction_uplift_pp = delta_win_rate

        selected.append(
            {
                "group": defn["group"],
                "label": label,
                "bold_row": bool(defn.get("bold_row", False)),
                "bucket": bucket,
                "source": source,
                "k": int(row["k"]),
                "n_flow": int(row["n_flow"]),
                "n_appearance": int(row["n_appearance"]),
                "predictor_short": _predictor_short(str(row["predictors"])),
                "predictors": str(row["predictors"]),
                "win_rate": win_rate,
                "loto_win_rate": loto_win,
                "lobo_win_rate": lobo_win,
                "joint_win_rate": joint_win,
                "interaction_win_rate": interaction_win_rate,
                "interaction_loto_win_rate": interaction_loto_win,
                "interaction_lobo_win_rate": interaction_lobo_win,
                "interaction_joint_win_rate": interaction_joint_win,
                "delta_win_rate_interaction_minus_base": delta_win_rate,
                "delta_loto_interaction_minus_base": delta_loto_win,
                "delta_lobo_interaction_minus_base": delta_lobo_win,
                "delta_joint_interaction_minus_base": delta_joint_win,
                "interaction_uplift_pp": interaction_uplift_pp,
            }
        )

    if missing:
        print(f"WARNING: {len(missing)} bucket(s) not found in sweep CSV and were skipped:")
        for b in missing:
            print(f"  {b}")

    if not selected:
        raise SystemExit("No rows matched — check --sweep-csv path and ROW_DEFS buckets.")

    if args.reject_avg_flow_interactions and rejected_interactions:
        uniq = sorted(set(rejected_interactions))
        print(
            f"WARNING: {len(uniq)} interaction bucket(s) skipped due to log_avg_flows leakage:"
        )
        for b in uniq:
            print(f"  {b}")

    out_df = pd.DataFrame(selected)

    # Write CSV.
    csv_path = out_dir / "paper_results_table.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    # Write LaTeX.
    tex_path = out_dir / "paper_results_table.tex"
    caption = args.caption
    if include_interaction_uplift:
        caption = caption.rstrip()
        if not caption.endswith("."):
            caption += "."
        caption += " The uplift column reports interaction-only minus baseline (percentage points), averaged across reported held-out protocols."
    _write_latex(
        out_df,
        tex_path,
        caption=caption,
        label="tab:param_matched_results",
        include_heldout_columns=bool(args.include_heldout_columns),
        include_interaction_uplift=include_interaction_uplift,
    )
    print(f"Wrote {tex_path}")

    # Print a quick ASCII preview.
    print("\n--- Preview ---")
    if args.include_heldout_columns:
        if include_interaction_uplift:
            print(f"{'Configuration':<45} {'k':>4}  {'Held T':>7}  {'Held B':>7}  {'Held TB':>7}  {'Δ Int.':>7}")
            print("-" * 98)
        else:
            print(f"{'Configuration':<45} {'k':>4}  {'Held T':>7}  {'Held B':>7}  {'Held TB':>7}")
            print("-" * 85)
    else:
        if include_interaction_uplift:
            print(f"{'Configuration':<45} {'k':>4}  {'Win rate':>9}  {'Δ Int.':>7}")
            print("-" * 78)
        else:
            print(f"{'Configuration':<45} {'k':>4}  {'Win rate':>9}")
            print("-" * 65)
    prev_group = None
    for _, r in out_df.iterrows():
        if r["group"] != prev_group:
            print(f"\n  [{r['group']}]")
            prev_group = r["group"]
        label = str(r["label"]).replace(r"\textbf{", "").replace("}", "")
        scale = "%" if args.win_rate_scale == "percent" else ""
        if args.include_heldout_columns:
            def _fmt(v: object) -> str:
                x = _safe_float(v)
                return "--" if not math.isfinite(x) else f"{x:>6.1f}{scale}"
            if include_interaction_uplift:
                print(
                    f"  {label:<43} {int(r['k']):>4}  "
                    f"{_fmt(r['loto_win_rate'])}  {_fmt(r['lobo_win_rate'])}  {_fmt(r['joint_win_rate'])}  "
                    f"{_fmt(r['interaction_uplift_pp'])}"
                )
            else:
                print(
                    f"  {label:<43} {int(r['k']):>4}  "
                    f"{_fmt(r['loto_win_rate'])}  {_fmt(r['lobo_win_rate'])}  {_fmt(r['joint_win_rate'])}"
                )
        else:
            if include_interaction_uplift:
                uplift = _safe_float(r["interaction_uplift_pp"])
                uplift_s = "--" if not math.isfinite(uplift) else f"{uplift:>+6.1f}{scale}"
                print(f"  {label:<43} {int(r['k']):>4}  {r['win_rate']:>8.1f}{scale}  {uplift_s}")
            else:
                print(f"  {label:<43} {int(r['k']):>4}  {r['win_rate']:>8.1f}{scale}")


if __name__ == "__main__":
    main()
