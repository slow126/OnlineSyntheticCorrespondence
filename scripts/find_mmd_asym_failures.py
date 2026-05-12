#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _match_any(name: str, patterns: List[str]) -> bool:
    for pat in patterns:
        if not pat:
            continue
        if re.search(pat, name):
            return True
    return False


def _filter_methods(
    methods: pd.DataFrame,
    patterns: List[str],
    target_filter: List[str],
    symmetry_allow: Optional[List[str]] = None,
    exclude_pairwise: bool = False,
    exclude_combo: bool = False,
) -> pd.DataFrame:
    rows = methods.copy()
    if target_filter and "target" in rows.columns:
        rows = rows[rows["target"].astype(str).isin(target_filter)]
    if symmetry_allow and "symmetry" in rows.columns:
        rows = rows[rows["symmetry"].astype(str).isin(symmetry_allow)]
    if exclude_pairwise:
        rows = rows[~rows["method"].astype(str).str.contains("_pairwise", regex=False)]
    if exclude_combo:
        rows = rows[~rows["method"].astype(str).str.startswith("combo_")]
    if patterns:
        mask = rows["method"].astype(str).apply(lambda m: _match_any(m, patterns))
        rows = rows[mask]
    return rows


def _collect_summaries(
    methods: pd.DataFrame,
    patterns: List[str],
    target_filter: List[str],
    symmetry_allow: Optional[List[str]] = None,
    summary_name: str = "prediction_loto_rank_summary.csv",
    exclude_pairwise: bool = False,
    exclude_combo: bool = False,
    allowed_methods: Optional[set] = None,
) -> Dict[str, pd.DataFrame]:
    rows = _filter_methods(
        methods,
        patterns,
        target_filter,
        symmetry_allow,
        exclude_pairwise=exclude_pairwise,
        exclude_combo=exclude_combo,
    )
    if allowed_methods is not None:
        rows = rows[rows["method"].astype(str).isin(allowed_methods)]
    selected: Dict[str, pd.DataFrame] = {}
    for _, row in rows.iterrows():
        method = str(row.get("method", ""))
        path = Path(row.get("path", ""))
        summary = _read_csv(path / summary_name)
        if summary is None:
            continue
        selected[method] = summary
    return selected


def _infer_group_col(df: pd.DataFrame) -> Optional[str]:
    for col in ["benchmark", "train_dataset", "eval_dataset", "dataset", "group"]:
        if col in df.columns:
            return col
    return None


def _best_per_group(
    summaries: Dict[str, pd.DataFrame],
    metric: str,
    lower_better: bool,
) -> Tuple[Dict[str, Tuple[str, pd.Series]], Optional[str]]:
    best: Dict[str, Tuple[str, pd.Series]] = {}
    group_col: Optional[str] = None
    for method, df in summaries.items():
        if metric not in df.columns:
            continue
        if group_col is None:
            group_col = _infer_group_col(df)
        if group_col is None or group_col not in df.columns:
            continue
        for _, row in df.iterrows():
            group = str(row[group_col])
            if group == "__overall__":
                continue
            val = row.get(metric)
            if pd.isna(val):
                continue
            if group not in best:
                best[group] = (method, row)
                continue
            _, best_row = best[group]
            best_val = best_row.get(metric)
            if pd.isna(best_val):
                best[group] = (method, row)
                continue
            if lower_better:
                if val < best_val:
                    best[group] = (method, row)
            else:
                if val > best_val:
                    best[group] = (method, row)
    return best, group_col


def _fmt(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "NA"
    try:
        return f"{float(val):.3f}"
    except Exception:
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find MMD vs asymmetric failures from LOTO rank summaries."
    )
    parser.add_argument("--root", required=True, help="Output root (e.g., analysis_comprehensive_runs/hof_motion_v3)")
    parser.add_argument(
        "--mmd-pattern",
        default="mmd",
        help="Regex to match MMD methods (default: mmd)",
    )
    parser.add_argument(
        "--mmd-symmetry",
        default="sym",
        help="Comma-separated symmetry labels for MMD group (default: sym).",
    )
    parser.add_argument(
        "--asym-pattern",
        default="eval_to_train|train_to_eval|coverage|flow_",
        help="Regex to match asymmetric methods (combined).",
    )
    parser.add_argument(
        "--asym-symmetry",
        default="asym",
        help="Comma-separated symmetry labels for asymmetric group (default: asym).",
    )
    parser.add_argument(
        "--asym-eval-pattern",
        default="eval_only",
        help="Regex to match eval->train asymmetric methods (default: eval_only).",
    )
    parser.add_argument(
        "--asym-train-pattern",
        default="train_only",
        help="Regex to match train->eval asymmetric methods (default: train_only).",
    )
    parser.add_argument(
        "--target-filter",
        default="",
        help="Comma-separated targets to include (e.g., auc_normalized_observed).",
    )
    parser.add_argument(
        "--metric",
        default="pred_best_true_rank",
        help="Metric to compare (default: pred_best_true_rank for rank summaries).",
    )
    parser.add_argument(
        "--metric-source",
        choices=["rank", "summary"],
        default="rank",
        help="Use rank summaries or standard prediction summaries (default: rank).",
    )
    parser.add_argument(
        "--fold",
        choices=["loto", "lobo"],
        default="loto",
        help="Which fold to use for summaries (default: loto).",
    )
    parser.add_argument(
        "--exclude-pairwise",
        action="store_true",
        help="Exclude *_pairwise methods (recommended for MAE summaries).",
    )
    parser.add_argument(
        "--exclude-combo",
        action="store_true",
        help="Exclude combo_* methods (for single-family comparisons).",
    )
    parser.add_argument(
        "--match-field",
        choices=["n_predictors", "n_predictors_base"],
        default="n_predictors",
        help="Field to match for parameter parity (default: n_predictors).",
    )
    parser.add_argument(
        "--match-window",
        type=int,
        default=0,
        help="Allowed absolute difference for parameter parity (default: 0 = exact match).",
    )
    parser.add_argument(
        "--exclude-mmd-from-asym",
        action="store_true",
        help="Exclude methods containing 'mmd' from asymmetric pool.",
    )
    parser.add_argument(
        "--lower-better",
        action="store_true",
        help="Treat metric as lower-is-better (recommended for rank error / regret).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of examples to print for each direction.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if args.metric_source == "summary" and args.metric == "pred_best_true_rank":
        args.metric = "mae"
    method_summary = _read_csv(root / "method_summary.csv")
    if method_summary is None:
        raise SystemExit("Missing method_summary.csv")

    target_filter = [t.strip() for t in args.target_filter.split(",") if t.strip()]
    mmd_patterns = [p.strip() for p in args.mmd_pattern.split(",") if p.strip()]
    asym_patterns = [p.strip() for p in args.asym_pattern.split(",") if p.strip()]
    mmd_sym = [s.strip() for s in args.mmd_symmetry.split(",") if s.strip()]
    asym_sym = [s.strip() for s in args.asym_symmetry.split(",") if s.strip()]

    if args.metric_source == "rank":
        summary_name = f"prediction_{args.fold}_rank_summary.csv"
    else:
        summary_name = f"prediction_{args.fold}_summary.csv"

    exclude_pairwise = args.exclude_pairwise or (args.metric_source == "summary")

    mmd_summaries = _collect_summaries(
        method_summary,
        mmd_patterns,
        target_filter,
        symmetry_allow=mmd_sym,
        summary_name=summary_name,
        exclude_pairwise=exclude_pairwise,
        exclude_combo=args.exclude_combo,
    )

    # Optional parameter parity: restrict asym methods to match MMD parameter counts.
    allowed_asym_methods = None
    if args.match_window is not None and args.match_window >= 0:
        meta = method_summary.set_index("method")
        mmd_vals = []
        for m in mmd_summaries.keys():
            if m in meta.index and args.match_field in meta.columns:
                val = meta.loc[m, args.match_field]
                if pd.notna(val):
                    mmd_vals.append(int(val))
        if mmd_vals:
            def _match_val(val: float) -> bool:
                try:
                    v = int(val)
                except Exception:
                    return False
                return any(abs(v - mv) <= args.match_window for mv in mmd_vals)

            candidates = method_summary.copy()
            if args.match_field in candidates.columns:
                candidates = candidates[candidates[args.match_field].apply(_match_val)]
            if args.exclude_mmd_from_asym:
                candidates = candidates[~candidates["method"].astype(str).str.contains("mmd", regex=False)]
            allowed_asym_methods = set(candidates["method"].astype(str).tolist())

    asym_summaries = _collect_summaries(
        method_summary,
        asym_patterns,
        target_filter,
        symmetry_allow=asym_sym,
        summary_name=summary_name,
        exclude_pairwise=exclude_pairwise,
        exclude_combo=args.exclude_combo,
        allowed_methods=allowed_asym_methods,
    )
    asym_eval_summaries = _collect_summaries(
        method_summary,
        [args.asym_eval_pattern],
        target_filter,
        symmetry_allow=asym_sym,
        summary_name=summary_name,
        exclude_pairwise=exclude_pairwise,
        exclude_combo=args.exclude_combo,
        allowed_methods=allowed_asym_methods,
    )
    asym_train_summaries = _collect_summaries(
        method_summary,
        [args.asym_train_pattern],
        target_filter,
        symmetry_allow=asym_sym,
        summary_name=summary_name,
        exclude_pairwise=exclude_pairwise,
        exclude_combo=args.exclude_combo,
        allowed_methods=allowed_asym_methods,
    )

    match_info = ""
    if allowed_asym_methods is not None:
        match_info = f", matched_on={args.match_field}±{args.match_window}"
    print(
        f"Selected methods: mmd={len(mmd_summaries)}, asym={len(asym_summaries)}, "
        f"eval_only={len(asym_eval_summaries)}, train_only={len(asym_train_summaries)} "
        f"(summary={summary_name}{match_info})"
    )

    if not mmd_summaries:
        raise SystemExit("No MMD summaries found. Check --mmd-pattern.")
    if not asym_summaries:
        raise SystemExit(
            "No asymmetric summaries found. Check --asym-pattern or set it to empty to use symmetry filter only."
        )

    mmd_best, mmd_group_col = _best_per_group(mmd_summaries, args.metric, args.lower_better)
    asym_best, asym_group_col = _best_per_group(asym_summaries, args.metric, args.lower_better)
    asym_eval_best, _ = _best_per_group(asym_eval_summaries, args.metric, args.lower_better)
    asym_train_best, _ = _best_per_group(asym_train_summaries, args.metric, args.lower_better)

    rows = []
    group_label = mmd_group_col or asym_group_col or "group"
    groups = set(mmd_best.keys()) | set(asym_best.keys()) | set(asym_eval_best.keys()) | set(asym_train_best.keys())
    for group in sorted(groups):
        mmd = mmd_best.get(group)
        asym = asym_best.get(group)
        asym_eval = asym_eval_best.get(group)
        asym_train = asym_train_best.get(group)
        if mmd is None or asym is None:
            continue
        mmd_method, mmd_row = mmd
        asym_method, asym_row = asym
        mmd_val = mmd_row.get(args.metric)
        asym_val = asym_row.get(args.metric)
        if pd.isna(mmd_val) or pd.isna(asym_val):
            continue
        # Positive delta => asym better when lower-better; else negative means asym worse.
        if args.lower_better:
            delta = float(mmd_val) - float(asym_val)
        else:
            delta = float(asym_val) - float(mmd_val)
        # Eval->train / train->eval deltas (if available)
        eval_val = None
        train_val = None
        eval_method = None
        train_method = None
        if asym_eval is not None:
            eval_method, eval_row = asym_eval
            eval_val = eval_row.get(args.metric)
        if asym_train is not None:
            train_method, train_row = asym_train
            train_val = train_row.get(args.metric)
        row_out = {
            "group": group,
            "mmd_method": mmd_method,
            "asym_method": asym_method,
            "mmd_val": float(mmd_val),
            "asym_val": float(asym_val),
            "delta": delta,
            "eval_method": eval_method,
            "eval_val": float(eval_val) if eval_val is not None and not pd.isna(eval_val) else float("nan"),
            "train_method": train_method,
            "train_val": float(train_val) if train_val is not None and not pd.isna(train_val) else float("nan"),
        }
        # Optional rank-only fields
        if args.metric_source == "rank":
            row_out.update(
                {
                    "mmd_pred_best": mmd_row.get("pred_best_option", "NA"),
                    "asym_pred_best": asym_row.get("pred_best_option", "NA"),
                    "mmd_true_rank": mmd_row.get("pred_best_true_rank", "NA"),
                    "asym_true_rank": asym_row.get("pred_best_true_rank", "NA"),
                    "mmd_regret": mmd_row.get("regret", "NA"),
                    "asym_regret": asym_row.get("regret", "NA"),
                }
            )
        rows.append(row_out)

    if not rows:
        raise SystemExit("No overlapping benchmarks found between MMD and asymmetric methods.")

    df = pd.DataFrame(rows)
    # Split into asym better / mmd better / ties.
    df_asym = df[df["delta"] > 0].sort_values("delta", ascending=False).head(args.top_k)
    df_mmd = df[df["delta"] < 0].sort_values("delta", ascending=True).head(args.top_k)
    df_tie = df[df["delta"] == 0].head(args.top_k)

    print("=== MMD vs Asymmetric (Asym Better) ===")
    if df_asym.empty:
        print("- None")
    for _, r in df_asym.iterrows():
        print(
            f"- {group_label}={r['group']}: "
            f"mmd={_fmt(r['mmd_val'])} ({r['mmd_method']}), "
            f"asym={_fmt(r['asym_val'])} ({r['asym_method']}), "
            f"delta={_fmt(r['delta'])}"
        )
        if args.metric_source == "rank":
            print(
                f"  pred_best: mmd={r['mmd_pred_best']} (true_rank={_fmt(r['mmd_true_rank'])}, regret={_fmt(r['mmd_regret'])}), "
                f"asym={r['asym_pred_best']} (true_rank={_fmt(r['asym_true_rank'])}, regret={_fmt(r['asym_regret'])})"
            )
        if pd.notna(r.get("eval_val")) or pd.notna(r.get("train_val")):
            print(
                f"  eval->train={_fmt(r.get('eval_val'))} ({r.get('eval_method','NA')}), "
                f"train->eval={_fmt(r.get('train_val'))} ({r.get('train_method','NA')})"
            )

    print("\n=== MMD vs Asymmetric (MMD Better) ===")
    if df_mmd.empty:
        print("- None")
    for _, r in df_mmd.iterrows():
        print(
            f"- {group_label}={r['group']}: "
            f"mmd={_fmt(r['mmd_val'])} ({r['mmd_method']}), "
            f"asym={_fmt(r['asym_val'])} ({r['asym_method']}), "
            f"delta={_fmt(r['delta'])}"
        )
        if args.metric_source == "rank":
            print(
                f"  pred_best: mmd={r['mmd_pred_best']} (true_rank={_fmt(r['mmd_true_rank'])}, regret={_fmt(r['mmd_regret'])}), "
                f"asym={r['asym_pred_best']} (true_rank={_fmt(r['asym_true_rank'])}, regret={_fmt(r['asym_regret'])})"
            )
        if pd.notna(r.get("eval_val")) or pd.notna(r.get("train_val")):
            print(
                f"  eval->train={_fmt(r.get('eval_val'))} ({r.get('eval_method','NA')}), "
                f"train->eval={_fmt(r.get('train_val'))} ({r.get('train_method','NA')})"
            )

    if not df_tie.empty:
        print("\n=== MMD vs Asymmetric (Ties) ===")
        for _, r in df_tie.iterrows():
            print(
                f"- {group_label}={r['group']}: "
                f"mmd={_fmt(r['mmd_val'])} ({r['mmd_method']}), "
                f"asym={_fmt(r['asym_val'])} ({r['asym_method']}), "
                f"delta={_fmt(r['delta'])}"
            )

    # Summary counts
    better_asym = (df["delta"] > 0).sum()
    better_mmd = (df["delta"] < 0).sum()
    print(
        f"\nSummary: asym better in {better_asym}/{len(df)} benchmarks, mmd better in {better_mmd}/{len(df)} benchmarks."
    )


if __name__ == "__main__":
    main()
