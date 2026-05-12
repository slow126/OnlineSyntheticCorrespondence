#!/usr/bin/env python3
"""Validate anchor-based calibration for deployment-style use cases.

This script simulates the user workflow:
1) Start from zero-shot predictions for a target context.
2) Collect K anchor evaluations (predicted + observed scores).
3) Fit a calibration map from anchors.
4) Rank remaining options and report macro metrics.

It supports protocol-specific inputs (LOBO, LOTO, LOMO/model_only) and
writes per-group and aggregated summaries.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _parse_csv_list(text: str | None) -> list[str]:
    if not text:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_csv(text: str | None) -> list[int]:
    out: list[int] = []
    for tok in _parse_csv_list(text):
        try:
            out.append(int(tok))
        except Exception as ex:
            raise ValueError(f"Failed to parse integer token '{tok}' in CSV '{text}'.") from ex
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)
    denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(x0, y0) / denom)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return _pearson(xr, yr)


def _top1_hit(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 1:
        return float("nan")
    tmax = float(np.max(y_true))
    pmax = float(np.max(y_pred))
    true_set = set(np.where(y_true >= (tmax - 1e-12))[0].tolist())
    pred_set = set(np.where(y_pred >= (pmax - 1e-12))[0].tolist())
    return 1.0 if true_set.intersection(pred_set) else 0.0


def _zsafe(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0))
    if not np.isfinite(sd) or sd <= float(eps):
        return x - mu
    return (x - mu) / sd


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2:
        return float("nan")
    sst = float(np.sum((yt - np.mean(yt)) ** 2))
    if sst <= 0.0:
        return float("nan")
    sse = float(np.sum((yp - yt) ** 2))
    return float(1.0 - (sse / sst))


def _resolve_column(df: pd.DataFrame, requested: str | None, candidates: Iterable[str], label: str) -> str:
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Requested {label} column '{requested}' not found.")
        return str(requested)
    for c in candidates:
        if c in df.columns:
            return str(c)
    raise ValueError(f"Could not resolve {label} column from candidates={list(candidates)}.")


def _select_anchor_indices(
    y_pred: np.ndarray,
    k: int,
    mode: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    n = int(len(y_pred))
    if n <= 1 or int(k) <= 0:
        return np.zeros(0, dtype=int)
    k_eff = int(min(max(int(k), 0), n - 1))
    if k_eff <= 0:
        return np.zeros(0, dtype=int)
    if mode == "random":
        return np.sort(rng.choice(np.arange(n), size=k_eff, replace=False).astype(int))

    # Quantile spread over predicted scores for deterministic coverage.
    order = np.argsort(y_pred, kind="mergesort")
    if k_eff == 1:
        return np.array([int(order[n // 2])], dtype=int)
    qpos = np.clip((np.linspace(0.0, 1.0, k_eff) * (n - 1)).round().astype(int), 0, n - 1)
    qpos = np.unique(qpos)
    if len(qpos) < k_eff:
        for p in range(n):
            if p not in qpos:
                qpos = np.append(qpos, p)
            if len(qpos) >= k_eff:
                break
    return np.sort(order[qpos[:k_eff]].astype(int))


@dataclass
class Calibrator:
    mode: str
    a: float
    b: float
    raw_slope: float | None

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.a * x + self.b


def _fit_calibrator(
    y_pred_anchor: np.ndarray,
    y_true_anchor: np.ndarray,
    policy: str,
    affine_min_points: int,
    nonnegative_slope: bool,
) -> Calibrator:
    n = int(len(y_pred_anchor))
    if n <= 0:
        return Calibrator(mode="none", a=1.0, b=0.0, raw_slope=None)

    use_affine = False
    if policy == "affine":
        use_affine = True
    elif policy == "auto":
        use_affine = n >= int(affine_min_points)
    elif policy == "offset":
        use_affine = False
    else:
        raise ValueError(f"Unsupported calibration policy: {policy}")

    if use_affine and n >= 2:
        A = np.column_stack([y_pred_anchor, np.ones(n, dtype=float)])
        try:
            raw_a, b = np.linalg.lstsq(A, y_true_anchor, rcond=None)[0]
            a = float(max(float(raw_a), 0.0)) if bool(nonnegative_slope) else float(raw_a)
            return Calibrator(mode="affine", a=a, b=float(b), raw_slope=float(raw_a))
        except Exception:
            pass

    b = float(np.mean(y_true_anchor - y_pred_anchor))
    return Calibrator(mode="offset", a=1.0, b=b, raw_slope=None)


def _load_protocol_df(
    csv_path: Path,
    protocol_name: str,
    true_col: str | None,
    pred_col: str | None,
    group_cols: list[str],
    option_col: str,
    exclude_existing_calibration_rows: bool,
) -> tuple[pd.DataFrame, str, str, list[str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"{protocol_name}: file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{protocol_name}: empty file: {csv_path}")

    if exclude_existing_calibration_rows and "is_calibration_row" in df.columns:
        mask = pd.to_numeric(df["is_calibration_row"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 0.0
        df = df.loc[mask].copy()

    tcol = _resolve_column(
        df,
        requested=true_col,
        candidates=["y_true", "target_eval", "target_residual", "target"],
        label=f"{protocol_name} true",
    )
    pcol = _resolve_column(
        df,
        requested=pred_col,
        candidates=["y_pred", "prediction", "prediction_residual"],
        label=f"{protocol_name} pred",
    )
    if option_col not in df.columns:
        raise ValueError(f"{protocol_name}: option column '{option_col}' not found.")

    gcols = [c for c in group_cols if c in df.columns]
    if not gcols:
        raise ValueError(f"{protocol_name}: none of requested group columns found: {group_cols}")
    return df, tcol, pcol, gcols


def _load_lomo_df(
    csv_path: Path,
    best_csv: Path | None,
    candidate_id: str | None,
    protocol_filter: str,
    head_filter: str,
    true_col: str | None,
    pred_col: str | None,
    group_cols: list[str],
    option_col: str,
    exclude_existing_calibration_rows: bool,
) -> tuple[pd.DataFrame, str, str, list[str], str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"lomo: file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"lomo: empty file: {csv_path}")

    if "protocol" in df.columns:
        df = df[df["protocol"].astype(str) == str(protocol_filter)].copy()
    if "head" in df.columns:
        df = df[df["head"].astype(str) == str(head_filter)].copy()
    if df.empty:
        raise ValueError("lomo: no rows left after protocol/head filtering.")

    if exclude_existing_calibration_rows and "is_calibration_row" in df.columns:
        mask = pd.to_numeric(df["is_calibration_row"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 0.0
        df = df.loc[mask].copy()

    chosen_candidate = str(candidate_id or "").strip()
    if not chosen_candidate and best_csv is not None and best_csv.exists():
        best = pd.read_csv(best_csv)
        m = (
            (best["protocol"].astype(str) == str(protocol_filter))
            & (best["head"].astype(str) == str(head_filter))
        )
        sub = best.loc[m]
        if not sub.empty and "candidate_id" in sub.columns:
            chosen_candidate = str(sub["candidate_id"].iloc[0])
    if "candidate_id" in df.columns:
        if not chosen_candidate:
            uniq = sorted(df["candidate_id"].astype(str).dropna().unique().tolist())
            if len(uniq) != 1:
                raise ValueError(
                    "lomo: multiple candidate_id values present; pass --lomo-candidate-id "
                    "or --lomo-best-by-protocol-head-csv."
                )
            chosen_candidate = str(uniq[0])
        df = df[df["candidate_id"].astype(str) == str(chosen_candidate)].copy()
    if df.empty:
        raise ValueError("lomo: no rows left after candidate filtering.")

    tcol = _resolve_column(
        df,
        requested=true_col,
        candidates=["y_true", "target_eval", "target_residual", "target"],
        label="lomo true",
    )
    pcol = _resolve_column(
        df,
        requested=pred_col,
        candidates=["y_pred", "prediction", "prediction_residual"],
        label="lomo pred",
    )
    if option_col not in df.columns:
        raise ValueError(f"lomo: option column '{option_col}' not found.")
    gcols = [c for c in group_cols if c in df.columns]
    if not gcols:
        raise ValueError(f"lomo: none of requested group columns found: {group_cols}")
    return df, tcol, pcol, gcols, chosen_candidate


def _evaluate_protocol(
    protocol: str,
    df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    group_cols: list[str],
    option_col: str,
    k_values: list[int],
    seeds: list[int],
    anchor_selection: str,
    calibration_policy: str,
    affine_min_points: int,
    nonnegative_slope: bool,
    min_group_size: int,
    min_eval_size: int,
    max_calibration_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_group_rows: list[dict[str, object]] = []
    by_seed_rows: list[dict[str, object]] = []

    if anchor_selection == "quantile":
        seeds_eval = [int(seeds[0])]
    else:
        seeds_eval = [int(s) for s in seeds]

    for seed in seeds_eval:
        rng = np.random.RandomState(int(seed))
        for k in k_values:
            for gkey, gdf in df.groupby(group_cols, dropna=False, sort=False):
                work = gdf.copy()
                work[true_col] = pd.to_numeric(work[true_col], errors="coerce")
                work[pred_col] = pd.to_numeric(work[pred_col], errors="coerce")
                valid = np.isfinite(work[true_col].to_numpy(dtype=float)) & np.isfinite(
                    work[pred_col].to_numpy(dtype=float)
                )
                work = work.loc[valid].copy()
                if work.empty:
                    continue

                # Collapse potential repeated rows per option.
                if option_col in work.columns:
                    work = (
                        work.groupby(option_col, dropna=False)
                        .agg({true_col: "mean", pred_col: "mean"})
                        .reset_index()
                    )
                n_total = int(len(work))
                if n_total < int(min_group_size):
                    per_group_rows.append(
                        {
                            "protocol": protocol,
                            "seed": int(seed),
                            "k": int(k),
                            "group_key": "|".join(str(x) for x in (gkey if isinstance(gkey, tuple) else (gkey,))),
                            "n_total": int(n_total),
                            "status": "skip_small_group",
                        }
                    )
                    continue

                max_by_eval = max(int(n_total - int(min_eval_size)), 0)
                max_by_frac = int(np.floor(float(max_calibration_fraction) * float(n_total)))
                max_allowed = max(0, min(int(n_total - 1), max_by_eval, max_by_frac))
                k_eff = int(min(max(int(k), 0), max_allowed))

                y_true = work[true_col].to_numpy(dtype=float)
                y_pred = work[pred_col].to_numpy(dtype=float)
                anchor_idx = _select_anchor_indices(y_pred, k_eff, mode=anchor_selection, rng=rng)
                eval_mask = np.ones(n_total, dtype=bool)
                eval_mask[anchor_idx] = False
                n_eval = int(eval_mask.sum())
                if n_eval < int(min_eval_size):
                    per_group_rows.append(
                        {
                            "protocol": protocol,
                            "seed": int(seed),
                            "k": int(k),
                            "group_key": "|".join(str(x) for x in (gkey if isinstance(gkey, tuple) else (gkey,))),
                            "n_total": int(n_total),
                            "n_anchor": int(len(anchor_idx)),
                            "n_eval": int(n_eval),
                            "status": "skip_small_eval",
                        }
                    )
                    continue

                calibrator = _fit_calibrator(
                    y_pred_anchor=y_pred[anchor_idx],
                    y_true_anchor=y_true[anchor_idx],
                    policy=calibration_policy,
                    affine_min_points=int(affine_min_points),
                    nonnegative_slope=bool(nonnegative_slope),
                )
                y_cal = calibrator.apply(y_pred)
                y_true_eval = y_true[eval_mask]
                y_cal_eval = y_cal[eval_mask]

                # Shape-only views: centered removes offset; z-scored removes offset+scale.
                y_true_eval_center = y_true_eval - float(np.mean(y_true_eval))
                y_cal_eval_center = y_cal_eval - float(np.mean(y_cal_eval))
                y_true_eval_z = _zsafe(y_true_eval)
                y_cal_eval_z = _zsafe(y_cal_eval)

                rmse = float(np.sqrt(np.mean((y_cal_eval - y_true_eval) ** 2)))
                mae = float(np.mean(np.abs(y_cal_eval - y_true_eval)))
                pear = _pearson(y_true_eval, y_cal_eval)
                spear = _spearman(y_true_eval, y_cal_eval)
                top1 = _top1_hit(y_true_eval, y_cal_eval)
                r2 = _r2(y_true_eval, y_cal_eval)

                rmse_center = float(np.sqrt(np.mean((y_cal_eval_center - y_true_eval_center) ** 2)))
                mae_center = float(np.mean(np.abs(y_cal_eval_center - y_true_eval_center)))
                pear_center = _pearson(y_true_eval_center, y_cal_eval_center)
                spear_center = _spearman(y_true_eval_center, y_cal_eval_center)
                r2_center = _r2(y_true_eval_center, y_cal_eval_center)

                rmse_z = float(np.sqrt(np.mean((y_cal_eval_z - y_true_eval_z) ** 2)))
                mae_z = float(np.mean(np.abs(y_cal_eval_z - y_true_eval_z)))
                pear_z = _pearson(y_true_eval_z, y_cal_eval_z)
                spear_z = _spearman(y_true_eval_z, y_cal_eval_z)
                r2_z = _r2(y_true_eval_z, y_cal_eval_z)

                sd_true = float(np.std(y_true_eval, ddof=0))
                sd_pred = float(np.std(y_cal_eval, ddof=0))
                dispersion_ratio = float(sd_pred / sd_true) if sd_true > 0.0 else float("nan")
                per_group_rows.append(
                    {
                        "protocol": protocol,
                        "seed": int(seed),
                        "k": int(k),
                        "group_key": "|".join(str(x) for x in (gkey if isinstance(gkey, tuple) else (gkey,))),
                        "n_total": int(n_total),
                        "n_anchor": int(len(anchor_idx)),
                        "n_eval": int(n_eval),
                        "calibration_fraction": float(len(anchor_idx) / n_total),
                        "calibrator_mode": calibrator.mode,
                        "raw_slope": calibrator.raw_slope,
                        "slope": float(calibrator.a),
                        "intercept": float(calibrator.b),
                        "rmse": rmse,
                        "mae": mae,
                        "pearson": pear,
                        "spearman": spear,
                        "r2": r2,
                        "rmse_centered": rmse_center,
                        "mae_centered": mae_center,
                        "pearson_centered": pear_center,
                        "spearman_centered": spear_center,
                        "r2_centered": r2_center,
                        "rmse_z": rmse_z,
                        "mae_z": mae_z,
                        "pearson_z": pear_z,
                        "spearman_z": spear_z,
                        "r2_z": r2_z,
                        "std_true_eval": sd_true,
                        "std_pred_eval": sd_pred,
                        "dispersion_ratio": dispersion_ratio,
                        "top1_hit": top1,
                        "status": "ok",
                    }
                )

            tmp = pd.DataFrame(
                [
                    r
                    for r in per_group_rows
                    if r.get("protocol") == protocol and int(r.get("seed", -1)) == int(seed) and int(r.get("k", -1)) == int(k)
                ]
            )
            ok = tmp[tmp["status"] == "ok"].copy() if not tmp.empty else pd.DataFrame()
            n_skipped = int((tmp["status"] != "ok").sum()) if not tmp.empty else 0
            row = {
                "protocol": protocol,
                "seed": int(seed),
                "k": int(k),
                "n_groups_total": int(len(tmp)),
                "n_groups_used": int(len(ok)),
                "n_groups_skipped": int(n_skipped),
                "spearman_macro": float(ok["spearman"].mean()) if not ok.empty else float("nan"),
                "top1_macro": float(ok["top1_hit"].mean()) if not ok.empty else float("nan"),
                "pearson_macro": float(ok["pearson"].mean()) if not ok.empty else float("nan"),
                "rmse_macro": float(ok["rmse"].mean()) if not ok.empty else float("nan"),
                "mae_macro": float(ok["mae"].mean()) if not ok.empty else float("nan"),
                "r2_macro": float(ok["r2"].mean()) if not ok.empty else float("nan"),
                "spearman_centered_macro": float(ok["spearman_centered"].mean()) if not ok.empty else float("nan"),
                "pearson_centered_macro": float(ok["pearson_centered"].mean()) if not ok.empty else float("nan"),
                "rmse_centered_macro": float(ok["rmse_centered"].mean()) if not ok.empty else float("nan"),
                "mae_centered_macro": float(ok["mae_centered"].mean()) if not ok.empty else float("nan"),
                "r2_centered_macro": float(ok["r2_centered"].mean()) if not ok.empty else float("nan"),
                "spearman_z_macro": float(ok["spearman_z"].mean()) if not ok.empty else float("nan"),
                "pearson_z_macro": float(ok["pearson_z"].mean()) if not ok.empty else float("nan"),
                "rmse_z_macro": float(ok["rmse_z"].mean()) if not ok.empty else float("nan"),
                "mae_z_macro": float(ok["mae_z"].mean()) if not ok.empty else float("nan"),
                "r2_z_macro": float(ok["r2_z"].mean()) if not ok.empty else float("nan"),
                "dispersion_ratio_macro": float(ok["dispersion_ratio"].mean()) if not ok.empty else float("nan"),
                "calibration_fraction_mean": float(ok["calibration_fraction"].mean()) if not ok.empty else float("nan"),
                "n_eval_mean": float(ok["n_eval"].mean()) if not ok.empty else float("nan"),
                "slope_nonpositive_frac_raw": (
                    float((pd.to_numeric(ok["raw_slope"], errors="coerce") <= 0.0).mean())
                    if ("raw_slope" in ok.columns and not ok.empty)
                    else float("nan")
                ),
            }
            by_seed_rows.append(row)

    return pd.DataFrame(per_group_rows), pd.DataFrame(by_seed_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate deployment-style anchor calibration on heldout prediction rows.")
    ap.add_argument("--output-dir", default="analysis_comprehensive_runs/calibration_usecase_validation")
    ap.add_argument("--k-values", default="0,1,3,5", help="CSV list of anchor budgets.")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="CSV list of random seeds.")
    ap.add_argument("--anchor-selection", choices=["quantile", "random"], default="quantile")
    ap.add_argument("--calibration-policy", choices=["auto", "offset", "affine"], default="auto")
    ap.add_argument("--affine-min-points", type=int, default=3)
    ap.add_argument("--allow-negative-slope", action="store_true")
    ap.add_argument("--min-group-size", type=int, default=6)
    ap.add_argument("--min-eval-size", type=int, default=3)
    ap.add_argument("--max-calibration-fraction", type=float, default=1.0)
    ap.add_argument("--option-col", default="train_dataset")
    ap.add_argument("--exclude-existing-calibration-rows", action="store_true")

    # LOBO / LOTO inputs.
    ap.add_argument("--lobo-csv", default="")
    ap.add_argument("--lobo-true-col", default="")
    ap.add_argument("--lobo-pred-col", default="")
    ap.add_argument("--lobo-group-cols", default="benchmark,model_family_encoder")

    ap.add_argument("--loto-csv", default="")
    ap.add_argument("--loto-true-col", default="")
    ap.add_argument("--loto-pred-col", default="")
    ap.add_argument("--loto-group-cols", default="benchmark,model_family_encoder")

    # LOMO/model_only inputs from heldout_model_cv_pred_rows.
    ap.add_argument("--lomo-csv", default="")
    ap.add_argument("--lomo-best-by-protocol-head-csv", default="")
    ap.add_argument("--lomo-candidate-id", default="")
    ap.add_argument("--lomo-protocol", default="model_only")
    ap.add_argument("--lomo-head", default="ridge")
    ap.add_argument("--lomo-true-col", default="")
    ap.add_argument("--lomo-pred-col", default="")
    ap.add_argument("--lomo-group-cols", default="benchmark,model_family_encoder")

    args = ap.parse_args()

    k_values = sorted(set(_parse_int_csv(args.k_values)))
    seeds = sorted(set(_parse_int_csv(args.seeds)))
    if not k_values:
        raise ValueError("No K values provided.")
    if not seeds:
        seeds = [0]
    if any(k < 0 for k in k_values):
        raise ValueError("All K values must be >= 0.")
    if int(args.min_group_size) < 2:
        raise ValueError("--min-group-size must be >= 2.")
    if int(args.min_eval_size) < 1:
        raise ValueError("--min-eval-size must be >= 1.")
    if float(args.max_calibration_fraction) <= 0.0 or float(args.max_calibration_fraction) > 1.0:
        raise ValueError("--max-calibration-fraction must be in (0, 1].")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_groups: list[pd.DataFrame] = []
    all_seed_rows: list[pd.DataFrame] = []
    run_meta: dict[str, object] = {
        "k_values": [int(x) for x in k_values],
        "seeds": [int(x) for x in seeds],
        "anchor_selection": str(args.anchor_selection),
        "calibration_policy": str(args.calibration_policy),
        "affine_min_points": int(args.affine_min_points),
        "allow_negative_slope": bool(args.allow_negative_slope),
        "min_group_size": int(args.min_group_size),
        "min_eval_size": int(args.min_eval_size),
        "max_calibration_fraction": float(args.max_calibration_fraction),
        "option_col": str(args.option_col),
        "exclude_existing_calibration_rows": bool(args.exclude_existing_calibration_rows),
        "protocols": {},
    }

    def _run_protocol(
        protocol: str,
        df: pd.DataFrame,
        true_col: str,
        pred_col: str,
        group_cols: list[str],
    ) -> None:
        gdf, sdf = _evaluate_protocol(
            protocol=protocol,
            df=df,
            true_col=true_col,
            pred_col=pred_col,
            group_cols=group_cols,
            option_col=str(args.option_col),
            k_values=k_values,
            seeds=seeds,
            anchor_selection=str(args.anchor_selection),
            calibration_policy=str(args.calibration_policy),
            affine_min_points=int(args.affine_min_points),
            nonnegative_slope=(not bool(args.allow_negative_slope)),
            min_group_size=int(args.min_group_size),
            min_eval_size=int(args.min_eval_size),
            max_calibration_fraction=float(args.max_calibration_fraction),
        )
        all_groups.append(gdf)
        all_seed_rows.append(sdf)

    if str(args.lobo_csv).strip():
        lobo_df, lobo_t, lobo_p, lobo_g = _load_protocol_df(
            csv_path=Path(args.lobo_csv),
            protocol_name="lobo",
            true_col=str(args.lobo_true_col).strip() or None,
            pred_col=str(args.lobo_pred_col).strip() or None,
            group_cols=_parse_csv_list(args.lobo_group_cols),
            option_col=str(args.option_col),
            exclude_existing_calibration_rows=bool(args.exclude_existing_calibration_rows),
        )
        run_meta["protocols"]["lobo"] = {
            "csv": str(args.lobo_csv),
            "true_col": lobo_t,
            "pred_col": lobo_p,
            "group_cols": lobo_g,
            "n_rows": int(len(lobo_df)),
        }
        _run_protocol("lobo", lobo_df, lobo_t, lobo_p, lobo_g)

    if str(args.loto_csv).strip():
        loto_df, loto_t, loto_p, loto_g = _load_protocol_df(
            csv_path=Path(args.loto_csv),
            protocol_name="loto",
            true_col=str(args.loto_true_col).strip() or None,
            pred_col=str(args.loto_pred_col).strip() or None,
            group_cols=_parse_csv_list(args.loto_group_cols),
            option_col=str(args.option_col),
            exclude_existing_calibration_rows=bool(args.exclude_existing_calibration_rows),
        )
        run_meta["protocols"]["loto"] = {
            "csv": str(args.loto_csv),
            "true_col": loto_t,
            "pred_col": loto_p,
            "group_cols": loto_g,
            "n_rows": int(len(loto_df)),
        }
        _run_protocol("loto", loto_df, loto_t, loto_p, loto_g)

    if str(args.lomo_csv).strip():
        lomo_df, lomo_t, lomo_p, lomo_g, lomo_cid = _load_lomo_df(
            csv_path=Path(args.lomo_csv),
            best_csv=Path(args.lomo_best_by_protocol_head_csv) if str(args.lomo_best_by_protocol_head_csv).strip() else None,
            candidate_id=str(args.lomo_candidate_id).strip() or None,
            protocol_filter=str(args.lomo_protocol),
            head_filter=str(args.lomo_head),
            true_col=str(args.lomo_true_col).strip() or None,
            pred_col=str(args.lomo_pred_col).strip() or None,
            group_cols=_parse_csv_list(args.lomo_group_cols),
            option_col=str(args.option_col),
            exclude_existing_calibration_rows=bool(args.exclude_existing_calibration_rows),
        )
        run_meta["protocols"]["lomo"] = {
            "csv": str(args.lomo_csv),
            "protocol_filter": str(args.lomo_protocol),
            "head_filter": str(args.lomo_head),
            "candidate_id": str(lomo_cid),
            "true_col": lomo_t,
            "pred_col": lomo_p,
            "group_cols": lomo_g,
            "n_rows": int(len(lomo_df)),
        }
        _run_protocol("lomo", lomo_df, lomo_t, lomo_p, lomo_g)

    if not all_groups:
        raise ValueError("No protocol inputs provided. Pass --lobo-csv and/or --loto-csv and/or --lomo-csv.")

    per_group = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
    by_seed = pd.concat(all_seed_rows, ignore_index=True) if all_seed_rows else pd.DataFrame()
    summary = (
        by_seed.groupby(["protocol", "k"], dropna=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_groups_used_mean=("n_groups_used", "mean"),
            n_groups_used_min=("n_groups_used", "min"),
            spearman_macro_mean=("spearman_macro", "mean"),
            spearman_macro_std=("spearman_macro", "std"),
            top1_macro_mean=("top1_macro", "mean"),
            top1_macro_std=("top1_macro", "std"),
            rmse_macro_mean=("rmse_macro", "mean"),
            rmse_macro_std=("rmse_macro", "std"),
            mae_macro_mean=("mae_macro", "mean"),
            mae_macro_std=("mae_macro", "std"),
            r2_macro_mean=("r2_macro", "mean"),
            r2_macro_std=("r2_macro", "std"),
            spearman_centered_macro_mean=("spearman_centered_macro", "mean"),
            spearman_centered_macro_std=("spearman_centered_macro", "std"),
            rmse_centered_macro_mean=("rmse_centered_macro", "mean"),
            rmse_centered_macro_std=("rmse_centered_macro", "std"),
            r2_centered_macro_mean=("r2_centered_macro", "mean"),
            r2_centered_macro_std=("r2_centered_macro", "std"),
            spearman_z_macro_mean=("spearman_z_macro", "mean"),
            spearman_z_macro_std=("spearman_z_macro", "std"),
            rmse_z_macro_mean=("rmse_z_macro", "mean"),
            rmse_z_macro_std=("rmse_z_macro", "std"),
            r2_z_macro_mean=("r2_z_macro", "mean"),
            r2_z_macro_std=("r2_z_macro", "std"),
            dispersion_ratio_macro_mean=("dispersion_ratio_macro", "mean"),
            dispersion_ratio_macro_std=("dispersion_ratio_macro", "std"),
            calibration_fraction_mean=("calibration_fraction_mean", "mean"),
            n_eval_mean=("n_eval_mean", "mean"),
        )
        .reset_index()
        if not by_seed.empty
        else pd.DataFrame()
    )

    per_group_csv = out_dir / "anchor_calibration_per_group.csv"
    by_seed_csv = out_dir / "anchor_calibration_summary_by_seed.csv"
    summary_csv = out_dir / "anchor_calibration_summary.csv"
    config_json = out_dir / "anchor_calibration_config.json"
    per_group.to_csv(per_group_csv, index=False)
    by_seed.to_csv(by_seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    config_json.write_text(json.dumps(run_meta, indent=2, sort_keys=True))

    print(f"Wrote: {per_group_csv}")
    print(f"Wrote: {by_seed_csv}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {config_json}")
    if not summary.empty:
        print("\nHeadline (macro over groups):")
        show_cols = [
            "protocol",
            "k",
            "spearman_macro_mean",
            "spearman_z_macro_mean",
            "top1_macro_mean",
            "rmse_macro_mean",
            "rmse_z_macro_mean",
            "dispersion_ratio_macro_mean",
            "calibration_fraction_mean",
            "n_groups_used_mean",
        ]
        print(summary[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
