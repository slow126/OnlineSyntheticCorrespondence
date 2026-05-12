#!/usr/bin/env python3
"""
Build reviewer-oriented LOTO claim-evidence tables from raw per-method rows.

Evidence unit:
  - Atomic: paired method deltas within (fold, context).
  - Main-paper robust: per-(fold, context) median delta across matched method pairs.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _safe_num(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _context_id(df: pd.DataFrame) -> pd.Series:
    return df["benchmark"].astype(str) + "::" + df["model_family_encoder"].astype(str)


def _pairwise_cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    conc = 0.0
    ties = 0.0
    comparable = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dt = y_true[i] - y_true[j]
            if dt == 0:
                continue
            comparable += 1.0
            dp = y_pred[i] - y_pred[j]
            if dp == 0:
                ties += 1.0
                continue
            if (dt > 0 and dp > 0) or (dt < 0 and dp < 0):
                conc += 1.0
    if comparable == 0:
        return float("nan")
    return float((conc + 0.5 * ties) / comparable)


def _trimmed_mean(vals: pd.Series, trim_frac: float = 0.1) -> float:
    v = _safe_num(vals).dropna().sort_values().to_numpy()
    if len(v) == 0:
        return float("nan")
    k = int(math.floor(len(v) * trim_frac))
    if 2 * k >= len(v):
        return float(v.mean())
    return float(v[k : len(v) - k].mean())


def _distance_type(method_name: str) -> str:
    m = str(method_name)
    if "kmeans" in m:
        return "kmeans"
    if "eps_raw" in m:
        return "eps_raw"
    if "kl_k" in m:
        return "kl"
    if "mmd" in m:
        return "mmd"
    return "other"


def _split_predictors(text: object) -> List[str]:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    return [t.strip() for t in str(text).split(",") if str(t).strip()]


def _parse_signatures_from_predictors(text: object) -> Dict[str, object]:
    preds = _split_predictors(text)
    flow = sorted([p for p in preds if p.startswith("flow_") and "mmd" not in p])
    appearance = sorted([p for p in preds if p.startswith("dino_") and "mmd" not in p])
    hof = sorted([p for p in preds if p.startswith("hof_") and "mmd" not in p])
    flow_mmd = sorted([p for p in preds if p == "flow_mmd" or (p.startswith("flow_") and "mmd" in p)])
    appearance_mmd = sorted(
        [p for p in preds if p in {"dino_mmd", "feature_mmd"} or (p.startswith("dino_") and "mmd" in p)]
    )
    other_mmd = sorted([p for p in preds if "mmd" in p and p not in set(flow_mmd + appearance_mmd)])
    return {
        "predictor_set": frozenset(preds),
        "flow_set": frozenset(flow),
        "appearance_set": frozenset(appearance),
        "hof_set": frozenset(hof),
        "flow_mmd_set": frozenset(flow_mmd),
        "appearance_mmd_set": frozenset(appearance_mmd),
        "other_mmd_set": frozenset(other_mmd),
        "n_flow_sig": len(flow),
        "n_appearance_sig": len(appearance),
        "n_hof_sig": len(hof),
        "n_flow_mmd_sig": len(flow_mmd),
        "n_appearance_mmd_sig": len(appearance_mmd),
        "n_other_mmd_sig": len(other_mmd),
    }


def _base_signature(rec: Dict[str, object]) -> str:
    return (
        f"F{int(rec['n_flow_sig'])}_A{int(rec['n_appearance_sig'])}_H{int(rec['n_hof_sig'])}"
        f"_FM{int(rec['n_flow_mmd_sig'])}_AM{int(rec['n_appearance_mmd_sig'])}_OM{int(rec['n_other_mmd_sig'])}"
    )


def _has_any_mmd(rec: Dict[str, object]) -> bool:
    return int(rec["n_flow_mmd_sig"]) + int(rec["n_appearance_mmd_sig"]) + int(rec["n_other_mmd_sig"]) > 0


def _base_modalities(rec: Dict[str, object]) -> str:
    mods: List[str] = []
    if int(rec["n_flow_sig"]) > 0:
        mods.append("FLOW")
    if int(rec["n_appearance_sig"]) > 0:
        mods.append("APPEARANCE")
    if int(rec["n_hof_sig"]) > 0:
        mods.append("HOF")
    return "+".join(mods) if mods else "none"


def _predictor_short_name(name: str) -> str:
    m = {
        "dino_train_to_eval_mean_dist": "A_t2e_dist",
        "dino_eval_to_train_mean_dist": "A_e2t_dist",
        "dino_train_to_eval_kl_div": "A_t2e_kl",
        "dino_eval_to_train_kl_div": "A_e2t_kl",
        "hof_train_to_eval_mean_dist": "H_t2e_dist",
        "hof_eval_to_train_mean_dist": "H_e2t_dist",
        "hof_train_to_eval_kl_div": "H_t2e_kl",
        "hof_eval_to_train_kl_div": "H_e2t_kl",
        "flow_train_to_eval_kl_div": "F_t2e_kl",
        "flow_eval_to_train_kl_div": "F_e2t_kl",
        "flow_mmd": "F_mmd",
        "dino_mmd": "A_mmd",
        "feature_mmd": "feat_mmd",
    }
    if name in m:
        return m[name]
    s = str(name)
    if s.startswith("flow_"):
        return "F:" + s[len("flow_") :]
    if s.startswith("dino_"):
        return "A:" + s[len("dino_") :]
    if s.startswith("hof_"):
        return "H:" + s[len("hof_") :]
    return s


def _added_predictor_label(text: str) -> str:
    toks = [t for t in str(text).split(";") if t]
    if not toks:
        return ""
    return " + ".join(sorted(_predictor_short_name(t) for t in toks))


def _base_recipe_from_counts(
    n_flow: object,
    n_app: object,
    n_hof: object,
    n_flow_mmd: object,
    n_app_mmd: object,
    n_other_mmd: object,
) -> str:
    def _i(x: object) -> int:
        try:
            return int(x)
        except Exception:
            return 0

    return (
        f"FLOW={_i(n_flow)}, APPEAR={_i(n_app)}, HOF={_i(n_hof)}"
        f" | F_MMD={_i(n_flow_mmd)}, A_MMD={_i(n_app_mmd)}, O_MMD={_i(n_other_mmd)}"
    )


def _base_recipe_display(row: pd.Series) -> str:
    n_flow = int(row.get("base_n_flow", 0))
    n_app = int(row.get("base_n_appearance", 0))
    n_hof = int(row.get("base_n_hof", 0))
    n_fm = int(row.get("base_n_flow_mmd", 0))
    n_am = int(row.get("base_n_appearance_mmd", 0))
    n_om = int(row.get("base_n_other_mmd", 0))
    dist = str(row.get("distance_type", ""))
    comp = str(row.get("comparison", ""))

    flow_part = f"FLOW={n_flow}"
    extra = ""
    if n_flow > 0 and dist:
        flow_part = f"FLOW={n_flow}({dist})"
    elif ("FLOW directional predictor" in comp) and dist:
        extra = f", added_FLOW_dist={dist}"

    return (
        f"{flow_part}, APPEAR={n_app}, HOF={n_hof}"
        f" | F_MMD={n_fm}, A_MMD={n_am}, O_MMD={n_om}{extra}"
    )


def _added_predictors(a: Dict[str, object], b: Dict[str, object]) -> str:
    added = sorted(list(set(a["predictor_set"]) - set(b["predictor_set"])))
    if len(added) == 0:
        return ""
    return ";".join(added)


def _regime_tag(method_name: str) -> str:
    m = str(method_name)
    has_train = "train_only" in m
    has_eval = "eval_only" in m
    if has_train and not has_eval:
        return "train_only"
    if has_eval and not has_train:
        return "eval_only"
    if has_train and has_eval:
        return "train_and_eval_only"
    return "joint"


def _added_exactly_one(a: frozenset, b: frozenset) -> bool:
    return len(a) == len(b) + 1 and b.issubset(a)


def _added_by_steps(a: frozenset, b: frozenset, steps: Sequence[int]) -> int:
    if not b.issubset(a):
        return 0
    d = len(a) - len(b)
    if d <= 0:
        return 0
    return d if d in set(steps) else 0


def _load_method_registry(method_summary_path: Path, exclude_substr: str) -> pd.DataFrame:
    df = pd.read_csv(method_summary_path)
    if exclude_substr:
        df = df[~df["path"].astype(str).str.contains(exclude_substr, regex=False)].copy()
    needed = [
        "method",
        "path",
        "symmetry",
        "derived_group",
        "n_predictors",
        "k_modal",
        "k_total",
        "predictors",
        "notes",
        "n_flow",
        "n_appearance",
        "n_hof",
        "n_flow_mmd",
        "n_appearance_mmd",
        "n_other_mmd",
    ]
    keep = [c for c in needed if c in df.columns]
    out = df[keep].drop_duplicates("method").reset_index(drop=True)
    parsed = out["predictors"].apply(_parse_signatures_from_predictors).apply(pd.Series)
    out = pd.concat([out, parsed], axis=1)
    out["distance_type"] = out["method"].astype(str).map(_distance_type)
    out["regime_tag"] = out["method"].astype(str).map(_regime_tag)
    return out


def _resolve_method_dir(root: Path, path_str: str) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p
    cwd_p = Path.cwd() / p
    if cwd_p.exists():
        return cwd_p
    root_p = root / p
    if root_p.exists():
        return root_p
    return cwd_p


def _build_comparison_pairs(
    pdir: Path, registry: pd.DataFrame, steps: Sequence[int]
) -> pd.DataFrame:
    methods_available = set(registry["method"].astype(str).tolist())
    rows: List[Dict[str, str]] = []

    # Strict incremental pairs directly from method metadata (not best-by-composition edges).
    recs = registry.to_dict("records")
    for a in recs:
        for b in recs:
            if a["method"] == b["method"]:
                continue
            # Shared matching key: train/eval regime only.
            # NOTE: We intentionally do not require same distance_type here because
            # incremental comparisons may add a modality whose distance family is
            # absent in the baseline method name (e.g., adding FLOW eps_raw to a
            # baseline APPEAR-only method). Enforcing distance_type equality would
            # incorrectly drop those valid pairs.
            if a["regime_tag"] != b["regime_tag"]:
                continue

            # +APPEARANCE
            add_n = _added_by_steps(a["appearance_set"], b["appearance_set"], steps)
            if (
                a["flow_set"] == b["flow_set"]
                and a["hof_set"] == b["hof_set"]
                and a["flow_mmd_set"] == b["flow_mmd_set"]
                and a["appearance_mmd_set"] == b["appearance_mmd_set"]
                and a["other_mmd_set"] == b["other_mmd_set"]
                and add_n > 0
            ):
                rows.append(
                    {
                        "comparison": f"+{add_n} APPEARANCE directional predictor",
                        "group": "incremental",
                        "method_a": str(a["method"]),
                        "method_b": str(b["method"]),
                        "distance_type": str(a["distance_type"]),
                        "base_flow_distance_type": str(a["distance_type"]),
                        "base_regime_tag": str(a["regime_tag"]),
                        "base_modalities": _base_modalities(b),
                        "base_mmd_status": "with_mmd" if _has_any_mmd(b) else "no_mmd",
                        "base_signature": _base_signature(b),
                        "added_predictor": _added_predictors(a, b),
                        "base_n_flow": int(b["n_flow_sig"]),
                        "base_n_appearance": int(b["n_appearance_sig"]),
                        "base_n_hof": int(b["n_hof_sig"]),
                        "base_n_flow_mmd": int(b["n_flow_mmd_sig"]),
                        "base_n_appearance_mmd": int(b["n_appearance_mmd_sig"]),
                        "base_n_other_mmd": int(b["n_other_mmd_sig"]),
                    }
                )
            # +HOF
            add_n = _added_by_steps(a["hof_set"], b["hof_set"], steps)
            if (
                a["flow_set"] == b["flow_set"]
                and a["appearance_set"] == b["appearance_set"]
                and a["flow_mmd_set"] == b["flow_mmd_set"]
                and a["appearance_mmd_set"] == b["appearance_mmd_set"]
                and a["other_mmd_set"] == b["other_mmd_set"]
                and add_n > 0
            ):
                rows.append(
                    {
                        "comparison": f"+{add_n} HOF directional predictor",
                        "group": "incremental",
                        "method_a": str(a["method"]),
                        "method_b": str(b["method"]),
                        "distance_type": str(a["distance_type"]),
                        "base_flow_distance_type": str(a["distance_type"]),
                        "base_regime_tag": str(a["regime_tag"]),
                        "base_modalities": _base_modalities(b),
                        "base_mmd_status": "with_mmd" if _has_any_mmd(b) else "no_mmd",
                        "base_signature": _base_signature(b),
                        "added_predictor": _added_predictors(a, b),
                        "base_n_flow": int(b["n_flow_sig"]),
                        "base_n_appearance": int(b["n_appearance_sig"]),
                        "base_n_hof": int(b["n_hof_sig"]),
                        "base_n_flow_mmd": int(b["n_flow_mmd_sig"]),
                        "base_n_appearance_mmd": int(b["n_appearance_mmd_sig"]),
                        "base_n_other_mmd": int(b["n_other_mmd_sig"]),
                    }
                )
            # +FLOW
            add_n = _added_by_steps(a["flow_set"], b["flow_set"], steps)
            if (
                a["appearance_set"] == b["appearance_set"]
                and a["hof_set"] == b["hof_set"]
                and a["flow_mmd_set"] == b["flow_mmd_set"]
                and a["appearance_mmd_set"] == b["appearance_mmd_set"]
                and a["other_mmd_set"] == b["other_mmd_set"]
                and add_n > 0
            ):
                rows.append(
                    {
                        "comparison": f"+{add_n} FLOW directional predictor",
                        "group": "incremental",
                        "method_a": str(a["method"]),
                        "method_b": str(b["method"]),
                        "distance_type": str(a["distance_type"]),
                        "base_flow_distance_type": str(a["distance_type"]),
                        "base_regime_tag": str(a["regime_tag"]),
                        "base_modalities": _base_modalities(b),
                        "base_mmd_status": "with_mmd" if _has_any_mmd(b) else "no_mmd",
                        "base_signature": _base_signature(b),
                        "added_predictor": _added_predictors(a, b),
                        "base_n_flow": int(b["n_flow_sig"]),
                        "base_n_appearance": int(b["n_appearance_sig"]),
                        "base_n_hof": int(b["n_hof_sig"]),
                        "base_n_flow_mmd": int(b["n_flow_mmd_sig"]),
                        "base_n_appearance_mmd": int(b["n_appearance_mmd_sig"]),
                        "base_n_other_mmd": int(b["n_other_mmd_sig"]),
                    }
                )
            # +APPEARANCE_MMD
            add_n = _added_by_steps(a["appearance_mmd_set"], b["appearance_mmd_set"], steps)
            if (
                a["flow_set"] == b["flow_set"]
                and a["appearance_set"] == b["appearance_set"]
                and a["hof_set"] == b["hof_set"]
                and a["flow_mmd_set"] == b["flow_mmd_set"]
                and a["other_mmd_set"] == b["other_mmd_set"]
                and add_n > 0
            ):
                rows.append(
                    {
                        "comparison": f"+{add_n} APPEARANCE_MMD predictor",
                        "group": "incremental",
                        "method_a": str(a["method"]),
                        "method_b": str(b["method"]),
                        "distance_type": str(a["distance_type"]),
                        "base_flow_distance_type": str(a["distance_type"]),
                        "base_regime_tag": str(a["regime_tag"]),
                        "base_modalities": _base_modalities(b),
                        "base_mmd_status": "with_mmd" if _has_any_mmd(b) else "no_mmd",
                        "base_signature": _base_signature(b),
                        "added_predictor": _added_predictors(a, b),
                        "base_n_flow": int(b["n_flow_sig"]),
                        "base_n_appearance": int(b["n_appearance_sig"]),
                        "base_n_hof": int(b["n_hof_sig"]),
                        "base_n_flow_mmd": int(b["n_flow_mmd_sig"]),
                        "base_n_appearance_mmd": int(b["n_appearance_mmd_sig"]),
                        "base_n_other_mmd": int(b["n_other_mmd_sig"]),
                    }
                )
            # +FLOW_MMD
            add_n = _added_by_steps(a["flow_mmd_set"], b["flow_mmd_set"], steps)
            if (
                a["flow_set"] == b["flow_set"]
                and a["appearance_set"] == b["appearance_set"]
                and a["hof_set"] == b["hof_set"]
                and a["appearance_mmd_set"] == b["appearance_mmd_set"]
                and a["other_mmd_set"] == b["other_mmd_set"]
                and add_n > 0
            ):
                rows.append(
                    {
                        "comparison": f"+{add_n} FLOW_MMD predictor",
                        "group": "incremental",
                        "method_a": str(a["method"]),
                        "method_b": str(b["method"]),
                        "distance_type": str(a["distance_type"]),
                        "base_flow_distance_type": str(a["distance_type"]),
                        "base_regime_tag": str(a["regime_tag"]),
                        "base_modalities": _base_modalities(b),
                        "base_mmd_status": "with_mmd" if _has_any_mmd(b) else "no_mmd",
                        "base_signature": _base_signature(b),
                        "added_predictor": _added_predictors(a, b),
                        "base_n_flow": int(b["n_flow_sig"]),
                        "base_n_appearance": int(b["n_appearance_sig"]),
                        "base_n_hof": int(b["n_hof_sig"]),
                        "base_n_flow_mmd": int(b["n_flow_mmd_sig"]),
                        "base_n_appearance_mmd": int(b["n_appearance_mmd_sig"]),
                        "base_n_other_mmd": int(b["n_other_mmd_sig"]),
                    }
                )

    pure = pd.read_csv(pdir / "pure_modalities_matched_k_deltas.csv")
    for _, r in pure.iterrows():
        rows.extend(
            [
                {
                    "comparison": "FLOW vs APPEARANCE (pure, matched-k)",
                    "group": "pure_modality",
                    "method_a": str(r["flow_method"]),
                    "method_b": str(r["appearance_method"]),
                    "distance_type": _distance_type(str(r["flow_method"])),
                    "base_flow_distance_type": _distance_type(str(r["flow_method"])),
                    "base_regime_tag": "",
                    "base_modalities": "",
                    "base_mmd_status": "",
                    "base_signature": "",
                    "added_predictor": "",
                    "base_n_flow": 0,
                    "base_n_appearance": 0,
                    "base_n_hof": 0,
                    "base_n_flow_mmd": 0,
                    "base_n_appearance_mmd": 0,
                    "base_n_other_mmd": 0,
                },
                {
                    "comparison": "FLOW vs HOF (pure, matched-k)",
                    "group": "pure_modality",
                    "method_a": str(r["flow_method"]),
                    "method_b": str(r["hof_method"]),
                    "distance_type": _distance_type(str(r["flow_method"])),
                    "base_flow_distance_type": _distance_type(str(r["flow_method"])),
                    "base_regime_tag": "",
                    "base_modalities": "",
                    "base_mmd_status": "",
                    "base_signature": "",
                    "added_predictor": "",
                    "base_n_flow": 0,
                    "base_n_appearance": 0,
                    "base_n_hof": 0,
                    "base_n_flow_mmd": 0,
                    "base_n_appearance_mmd": 0,
                    "base_n_other_mmd": 0,
                },
                {
                    "comparison": "HOF vs APPEARANCE (pure, matched-k)",
                    "group": "pure_modality",
                    "method_a": str(r["hof_method"]),
                    "method_b": str(r["appearance_method"]),
                    "distance_type": _distance_type(str(r["hof_method"])),
                    "base_flow_distance_type": _distance_type(str(r["hof_method"])),
                    "base_regime_tag": "",
                    "base_modalities": "",
                    "base_mmd_status": "",
                    "base_signature": "",
                    "added_predictor": "",
                    "base_n_flow": 0,
                    "base_n_appearance": 0,
                    "base_n_hof": 0,
                    "base_n_flow_mmd": 0,
                    "base_n_appearance_mmd": 0,
                    "base_n_other_mmd": 0,
                },
            ]
        )

    # Explicit symmetric-vs-directed rows for reviewer-readable table claims.
    explicit_noninc_pairs = [
        ("FLOW_MMD vs 1 FLOW (eval-only)", "mmd_flow_only", "flow_eps_raw_joint_auc_at95_eval_only"),
        ("FLOW_MMD vs 1 HOF (eval-only)", "mmd_flow_only", "hof_motion_k1_eval_only"),
        ("DINO_MMD vs 1 DINO (eval-only)", "mmd_dino_only", "dino_rnorm_k5_eval_only"),
        (
            "FLOW+DINO_MMD vs (1 FLOW + 1 DINO) (eval-only)",
            "mmd_only",
            "combo_flow_eps_raw__dino_rnorm_k5_eval_only",
        ),
    ]
    for label, a, b in explicit_noninc_pairs:
        rows.append(
            {
                "comparison": label,
                "group": "explicit_nonincremental",
                "method_a": a,
                "method_b": b,
                "distance_type": _distance_type(b),
                "base_flow_distance_type": _distance_type(b),
                "base_regime_tag": "",
                "base_modalities": "",
                "base_mmd_status": "",
                "base_signature": "",
                "added_predictor": "",
                "base_n_flow": 0,
                "base_n_appearance": 0,
                "base_n_hof": 0,
                "base_n_flow_mmd": 0,
                "base_n_appearance_mmd": 0,
                "base_n_other_mmd": 0,
            }
        )

    out = pd.DataFrame(rows).drop_duplicates()
    out = out[out["method_a"].isin(methods_available) & out["method_b"].isin(methods_available)].copy()
    out["match_group_id"] = out.apply(
        lambda r: hashlib.sha1(f"{r['comparison']}::{r['method_a']}::{r['method_b']}".encode("utf-8")).hexdigest()[:12],
        axis=1,
    )
    return out.reset_index(drop=True)


def _method_cell_metrics(method_path: Path) -> pd.DataFrame:
    # LOTO per-cell atomic file: one row per (fold, benchmark, model_family).
    p = method_path / "prediction_loto_holdout_placement_detail.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    required = {"fold", "benchmark", "model_family_encoder", "heldout_true", "heldout_pred", "pairwise_win_rate"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame()

    df["heldout_true"] = _safe_num(df["heldout_true"])
    df["heldout_pred"] = _safe_num(df["heldout_pred"])
    df["pairwise_win_rate"] = _safe_num(df["pairwise_win_rate"])
    if "regret" in df.columns:
        df["regret"] = _safe_num(df["regret"])
    if "abs_rank_pct_error" in df.columns:
        df["abs_rank_pct_error"] = _safe_num(df["abs_rank_pct_error"])
    df = df[df["heldout_true"].notna() & df["heldout_pred"].notna() & df["pairwise_win_rate"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Guard against accidental duplicates.
    df["context_id"] = _context_id(df)
    gcols = ["fold", "context_id", "benchmark", "model_family_encoder"]
    agg = (
        df.groupby(gcols, dropna=False)
        .agg(
            cindex=("pairwise_win_rate", "mean"),
            heldout_true=("heldout_true", "mean"),
            heldout_pred=("heldout_pred", "mean"),
            regret=("regret", "mean") if "regret" in df.columns else ("pairwise_win_rate", "size"),
            abs_rank_pct_error=("abs_rank_pct_error", "mean") if "abs_rank_pct_error" in df.columns else ("pairwise_win_rate", "size"),
            n_rows=("pairwise_win_rate", "size"),
        )
        .reset_index()
    )
    agg["mae"] = (agg["heldout_pred"] - agg["heldout_true"]).abs()
    agg["rmse"] = np.sqrt((agg["heldout_pred"] - agg["heldout_true"]) ** 2)
    out = agg.rename(columns={"fold": "fold_id"})
    return out[
        [
            "fold_id",
            "context_id",
            "benchmark",
            "model_family_encoder",
            "cindex",
            "mae",
            "rmse",
            "regret",
            "abs_rank_pct_error",
            "n_rows",
        ]
    ]


def _compute_pair_deltas(
    pairs: pd.DataFrame, method_to_path: Dict[str, Path]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cache: Dict[str, pd.DataFrame] = {}
    paired_rows: List[Dict[str, object]] = []

    for _, p in pairs.iterrows():
        ma = str(p["method_a"])
        mb = str(p["method_b"])
        pa = method_to_path.get(ma)
        pb = method_to_path.get(mb)
        if pa is None or pb is None:
            continue
        if ma not in cache:
            cache[ma] = _method_cell_metrics(pa)
        if mb not in cache:
            cache[mb] = _method_cell_metrics(pb)
        a_df = cache[ma]
        b_df = cache[mb]
        if a_df.empty or b_df.empty:
            continue

        merged = a_df.merge(
            b_df,
            on=["fold_id", "context_id", "benchmark", "model_family_encoder"],
            suffixes=("_a", "_b"),
            how="inner",
        )
        if merged.empty:
            continue
        for _, r in merged.iterrows():
            paired_rows.append(
                {
                    "comparison": p["comparison"],
                    "group": p["group"],
                    "distance_type": p["distance_type"],
                    "base_flow_distance_type": p.get("base_flow_distance_type", p["distance_type"]),
                    "base_regime_tag": p.get("base_regime_tag", ""),
                    "base_modalities": p.get("base_modalities", ""),
                    "base_mmd_status": p.get("base_mmd_status", ""),
                    "base_signature": p.get("base_signature", ""),
                    "added_predictor": p.get("added_predictor", ""),
                    "base_n_flow": int(p.get("base_n_flow", 0)),
                    "base_n_appearance": int(p.get("base_n_appearance", 0)),
                    "base_n_hof": int(p.get("base_n_hof", 0)),
                    "base_n_flow_mmd": int(p.get("base_n_flow_mmd", 0)),
                    "base_n_appearance_mmd": int(p.get("base_n_appearance_mmd", 0)),
                    "base_n_other_mmd": int(p.get("base_n_other_mmd", 0)),
                    "match_group_id": p["match_group_id"],
                    "method_a": ma,
                    "method_b": mb,
                    "fold_id": r["fold_id"],
                    "context_id": r["context_id"],
                    "benchmark": r["benchmark"],
                    "model_family_encoder": r["model_family_encoder"],
                    "delta_cindex": float(r["cindex_a"] - r["cindex_b"]),
                    "delta_mae": float(r["mae_b"] - r["mae_a"]),   # improvement positive
                    "delta_rmse": float(r["rmse_b"] - r["rmse_a"]), # improvement positive
                    "delta_regret": float(r["regret_b"] - r["regret_a"]),
                    "delta_abs_rank_pct_error": float(r["abs_rank_pct_error_b"] - r["abs_rank_pct_error_a"]),
                }
            )

    paired = pd.DataFrame(paired_rows)
    if paired.empty:
        return paired, pd.DataFrame()

    cell = (
        paired.groupby(["comparison", "group", "fold_id", "context_id"], dropna=False)
        .agg(
            delta_cindex=("delta_cindex", "median"),
            delta_mae=("delta_mae", "median"),
            delta_rmse=("delta_rmse", "median"),
            delta_regret=("delta_regret", "median"),
            delta_abs_rank_pct_error=("delta_abs_rank_pct_error", "median"),
            cell_variant_count=("match_group_id", "nunique"),
        )
        .reset_index()
    )
    return paired, cell


def _summarize_from_paired(
    paired: pd.DataFrame,
    group_cols: List[str],
    min_cells_standard: int,
) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()

    cell = (
        paired.groupby(group_cols + ["fold_id", "context_id"], dropna=False)
        .agg(
            delta_cindex=("delta_cindex", "median"),
            delta_mae=("delta_mae", "median"),
            delta_rmse=("delta_rmse", "median"),
            delta_abs_rank_pct_error=("delta_abs_rank_pct_error", "median"),
            cell_variant_count=("match_group_id", "nunique"),
        )
        .reset_index()
    )

    rows: List[Dict[str, object]] = []
    for keys, g in cell.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            keys = (keys[0],) if isinstance(keys, tuple) else (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        n_cells = int(len(g))
        row.update(
            {
                "n_cells": n_cells,
                "n_folds": int(g["fold_id"].nunique()),
                "n_contexts": int(g["context_id"].nunique()),
                "delta_cindex_median": float(_safe_num(g["delta_cindex"]).median()),
                "delta_mae_median": float(_safe_num(g["delta_mae"]).median()),
                "delta_abs_rank_pct_error_median": float(_safe_num(g["delta_abs_rank_pct_error"]).median()),
                "pos_frac_cindex": float((_safe_num(g["delta_cindex"]) > 0).mean()),
                "pos_frac_mae": float((_safe_num(g["delta_mae"]) > 0).mean()),
                "median_cell_variant_count": float(_safe_num(g["cell_variant_count"]).median()),
                "evidence": "standard" if n_cells >= int(min_cells_standard) else "sparse",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_cell_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if len(group_cols) == 1:
            keys = (keys[0],) if isinstance(keys, tuple) else (keys,)
        row = {c: v for c, v in zip(group_cols, keys)}
        c = _safe_num(g["delta_cindex"]).dropna()
        m = _safe_num(g["delta_mae"]).dropna()
        s = _safe_num(g["delta_rmse"]).dropna()
        r = _safe_num(g["delta_abs_rank_pct_error"]).dropna()
        row.update(
            {
                "n_cells": int(len(g)),
                "n_folds": int(g["fold_id"].nunique()),
                "n_contexts": int(g["context_id"].nunique()),
                "delta_cindex_mean": float(c.mean()) if len(c) else float("nan"),
                "delta_cindex_median": float(c.median()) if len(c) else float("nan"),
                "delta_mae_mean": float(m.mean()) if len(m) else float("nan"),
                "delta_mae_median": float(m.median()) if len(m) else float("nan"),
                "delta_rmse_mean": float(s.mean()) if len(s) else float("nan"),
                "delta_rmse_median": float(s.median()) if len(s) else float("nan"),
                "delta_abs_rank_pct_error_mean": float(r.mean()) if len(r) else float("nan"),
                "delta_abs_rank_pct_error_median": float(r.median()) if len(r) else float("nan"),
                "pos_frac_cindex": float((c > 1e-12).mean()) if len(c) else float("nan"),
                "tie_frac_cindex": float((c.abs() <= 1e-12).mean()) if len(c) else float("nan"),
                "neg_frac_cindex": float((c < -1e-12).mean()) if len(c) else float("nan"),
                "pos_frac_mae": float((m > 1e-12).mean()) if len(m) else float("nan"),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_claim_rows(cell_df: pd.DataFrame, min_cells_standard: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for comp, g in cell_df.groupby("comparison", dropna=False):
        n_cells = int(len(g))
        n_folds = int(g["fold_id"].nunique())
        n_contexts = int(g["context_id"].nunique())
        row = {
            "comparison": comp,
            "n_cells": n_cells,
            "n_folds": n_folds,
            "n_contexts": n_contexts,
            "delta_cindex_median": float(_safe_num(g["delta_cindex"]).median()),
            "delta_mae_median": float(_safe_num(g["delta_mae"]).median()),
            "delta_rmse_median": float(_safe_num(g["delta_rmse"]).median()),
            "delta_regret_median": float(_safe_num(g["delta_regret"]).median()),
            "delta_abs_rank_pct_error_median": float(_safe_num(g["delta_abs_rank_pct_error"]).median()),
            "delta_cindex_trimmed_mean": _trimmed_mean(g["delta_cindex"]),
            "delta_mae_trimmed_mean": _trimmed_mean(g["delta_mae"]),
            "delta_rmse_trimmed_mean": _trimmed_mean(g["delta_rmse"]),
            "delta_regret_trimmed_mean": _trimmed_mean(g["delta_regret"]),
            "delta_abs_rank_pct_error_trimmed_mean": _trimmed_mean(g["delta_abs_rank_pct_error"]),
            "pos_frac_cindex": float((_safe_num(g["delta_cindex"]) > 0).mean()),
            "pos_frac_mae": float((_safe_num(g["delta_mae"]) > 0).mean()),
            "pos_frac_rmse": float((_safe_num(g["delta_rmse"]) > 0).mean()),
            "pos_frac_regret": float((_safe_num(g["delta_regret"]) > 0).mean()),
            "pos_frac_abs_rank_pct_error": float((_safe_num(g["delta_abs_rank_pct_error"]) > 0).mean()),
            "median_cell_variant_count": float(_safe_num(g["cell_variant_count"]).median()),
            "evidence": "standard" if n_cells >= int(min_cells_standard) else "sparse",
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    def _sort_key(comp: str) -> Tuple[int, int, str]:
        c = str(comp)
        m = re.match(r"^\+(\d+)\s+", c)
        k = int(m.group(1)) if m else 0
        if c.startswith("Directional vs Symmetric"):
            return (0, 0, c)
        if c.startswith("FLOW vs APPEARANCE"):
            return (1, 0, c)
        if c.startswith("FLOW vs HOF"):
            return (2, 0, c)
        if c.startswith("HOF vs APPEARANCE"):
            return (3, 0, c)
        if "APPEARANCE directional predictor" in c and "_MMD" not in c:
            return (4, k, c)
        if "HOF directional predictor" in c and "_MMD" not in c:
            return (5, k, c)
        if "FLOW directional predictor" in c and "_MMD" not in c:
            return (6, k, c)
        if "APPEARANCE_MMD predictor" in c:
            return (7, k, c)
        if "FLOW_MMD predictor" in c:
            return (8, k, c)
        return (1000, k, c)

    keys = out["comparison"].map(_sort_key).tolist()
    out["__cat"] = [k[0] for k in keys]
    out["__k"] = [k[1] for k in keys]
    out["__name"] = [k[2] for k in keys]
    out = out.sort_values(["__cat", "__k", "__name"]).drop(columns=["__cat", "__k", "__name"]).reset_index(drop=True)
    return out


def _distance_robustness(paired: pd.DataFrame) -> pd.DataFrame:
    if paired.empty:
        return pd.DataFrame()
    # Per (comparison, base composition, distance type), collapse first by cell, then summarize.
    cell = (
        paired.groupby(
            ["group", "comparison", "base_modalities", "base_mmd_status", "base_flow_distance_type", "fold_id", "context_id"],
            dropna=False,
        )
        .agg(
            delta_cindex=("delta_cindex", "median"),
            delta_mae=("delta_mae", "median"),
            delta_rmse=("delta_rmse", "median"),
            delta_regret=("delta_regret", "median"),
            delta_abs_rank_pct_error=("delta_abs_rank_pct_error", "median"),
        )
        .reset_index()
    )
    rows: List[Dict[str, object]] = []
    for (grp, comp, mods, mmd_s, dist), g in cell.groupby(
        ["group", "comparison", "base_modalities", "base_mmd_status", "base_flow_distance_type"], dropna=False
    ):
        rows.append(
            {
                "group": grp,
                "comparison": comp,
                "base_modalities": mods,
                "base_mmd_status": mmd_s,
                "distance_type": dist,
                "n_cells": int(len(g)),
                "delta_cindex_median": float(_safe_num(g["delta_cindex"]).median()),
                "delta_mae_median": float(_safe_num(g["delta_mae"]).median()),
                "delta_rmse_median": float(_safe_num(g["delta_rmse"]).median()),
                "delta_regret_median": float(_safe_num(g["delta_regret"]).median()),
                "delta_abs_rank_pct_error_median": float(_safe_num(g["delta_abs_rank_pct_error"]).median()),
                "pos_frac_cindex": float((_safe_num(g["delta_cindex"]) > 0).mean()),
                "pos_frac_mae": float((_safe_num(g["delta_mae"]) > 0).mean()),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["group", "comparison", "base_modalities", "base_mmd_status", "distance_type"])
        .reset_index(drop=True)
    )


def _write_tex_simple(df: pd.DataFrame, out_path: Path, caption: str, label: str) -> None:
    if df.empty:
        out_path.write_text("% Empty table\n", encoding="utf-8")
        return
    fmt = df.copy()
    for c in fmt.columns:
        if c.startswith("delta_"):
            fmt[c] = _safe_num(fmt[c]).map(lambda v: "--" if pd.isna(v) else f"{float(v):.4f}")
        if c.startswith("pos_frac_") or c.startswith("tie_frac_") or c.startswith("neg_frac_"):
            fmt[c] = _safe_num(fmt[c]).map(lambda v: "--" if pd.isna(v) else f"{100.0*float(v):.1f}\\%")
    tab = fmt.to_latex(index=False, escape=True)
    tex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\small",
            r"\resizebox{\linewidth}{!}{%",
            tab.strip(),
            r"}",
            r"\end{table}",
            "",
        ]
    )
    out_path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build robust LOTO claim-evidence tables")
    parser.add_argument("--root", required=True, help="Run root (ridge_resid... directory)")
    parser.add_argument(
        "--exclude-substr",
        default="/asym_and_mmd",
        help="Exclude methods whose path contains this substring",
    )
    parser.add_argument(
        "--increment-steps",
        default="1",
        help="Comma-separated allowed increment sizes for incremental comparisons (e.g., 1,2,4)",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--min-cells-standard", type=int, default=20)
    parser.add_argument(
        "--canonical-distance-types",
        default="eps_raw,mmd",
        help=(
            "Comma-separated distance families to show in the main incremental table. "
            "Each canonical type becomes one row per (comparison, added_predictor_label). "
            "Use 'all' to include every distance family. "
            "Default: 'eps_raw,mmd' (eps-radius coverage + symmetric MMD baseline)."
        ),
    )
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdir = root / "final_tables" / "predictor_group_tables"
    method_summary_path = pdir / "method_summary_with_derived_groups.csv"
    registry = _load_method_registry(method_summary_path, args.exclude_substr)
    method_to_path = {str(r["method"]): _resolve_method_dir(root, str(r["path"])) for _, r in registry.iterrows()}
    steps = sorted({int(s.strip()) for s in str(args.increment_steps).split(",") if s.strip()})
    steps = [s for s in steps if s > 0]
    if not steps:
        raise ValueError("--increment-steps must include at least one positive integer")

    pairs = _build_comparison_pairs(pdir, registry, steps=steps)
    paired, cell = _compute_pair_deltas(pairs, method_to_path)
    non_inc = paired[paired["group"] != "incremental"].copy()
    inc = paired[paired["group"] == "incremental"].copy()
    if not inc.empty:
        inc["added_predictor_label"] = inc["added_predictor"].map(_added_predictor_label)
        inc["base_recipe"] = inc.apply(_base_recipe_display, axis=1)

    agg_metrics = {
        "delta_cindex": ("delta_cindex", "median"),
        "delta_mae": ("delta_mae", "median"),
        "delta_rmse": ("delta_rmse", "median"),
        "delta_abs_rank_pct_error": ("delta_abs_rank_pct_error", "median"),
    }

    # Non-incremental claim-level summary (cell = fold x context).
    non_inc_cell = (
        non_inc.groupby(["comparison", "fold_id", "context_id"], dropna=False).agg(**agg_metrics).reset_index()
        if not non_inc.empty
        else pd.DataFrame()
    )
    summary_non_inc = _summarize_cell_table(non_inc_cell, ["comparison"])
    if not summary_non_inc.empty:
        n_recipe_non = (
            non_inc.groupby(["comparison"], dropna=False)["match_group_id"].nunique().reset_index(name="n_recipes")
        )
        summary_non_inc = summary_non_inc.merge(n_recipe_non, on=["comparison"], how="left")
        summary_non_inc.insert(0, "group", "non_incremental")
        summary_non_inc.insert(2, "added_predictor_label", "none")

    # Incremental summaries:
    # Stage A: recipe-level cell deltas.
    inc_recipe_cell = (
        inc.groupby(
            ["comparison", "added_predictor_label", "distance_type", "base_recipe", "fold_id", "context_id"],
            dropna=False,
        )
        .agg(**agg_metrics)
        .reset_index()
        if not inc.empty
        else pd.DataFrame()
    )
    # Stage B: claim-level cell deltas, collapsing across recipes.
    inc_cell = (
        inc_recipe_cell.groupby(["comparison", "added_predictor_label", "fold_id", "context_id"], dropna=False)
        .agg(**agg_metrics)
        .reset_index()
        if not inc_recipe_cell.empty
        else pd.DataFrame()
    )
    summary_inc = _summarize_cell_table(inc_cell, ["comparison", "added_predictor_label"])
    if not summary_inc.empty:
        n_recipe_inc = (
            inc_recipe_cell.groupby(["comparison", "added_predictor_label"], dropna=False)["base_recipe"]
            .nunique()
            .reset_index(name="n_recipes")
        )
        summary_inc = summary_inc.merge(n_recipe_inc, on=["comparison", "added_predictor_label"], how="left")
        summary_inc.insert(0, "group", "incremental")

    # Robustness by distance family (still collapsed across recipes within each cell).
    robust_cell = (
        inc_recipe_cell.groupby(
            ["comparison", "added_predictor_label", "distance_type", "fold_id", "context_id"], dropna=False
        )
        .agg(**agg_metrics)
        .reset_index()
        if not inc_recipe_cell.empty
        else pd.DataFrame()
    )
    robustness = _summarize_cell_table(robust_cell, ["comparison", "added_predictor_label", "distance_type"])
    if not robustness.empty:
        n_recipe_dist = (
            inc_recipe_cell.groupby(["comparison", "added_predictor_label", "distance_type"], dropna=False)["base_recipe"]
            .nunique()
            .reset_index(name="n_recipes")
        )
        robustness = robustness.merge(
            n_recipe_dist, on=["comparison", "added_predictor_label", "distance_type"], how="left"
        )

    # Debug table: recipe-specific results (supplementary / audit trail).
    top_recipes_debug = _summarize_cell_table(
        inc_recipe_cell,
        ["comparison", "added_predictor_label", "base_recipe", "distance_type"],
    )

    order = [
        "FLOW_MMD vs 1 FLOW (eval-only)",
        "FLOW_MMD vs 1 HOF (eval-only)",
        "DINO_MMD vs 1 DINO (eval-only)",
        "FLOW+DINO_MMD vs (1 FLOW + 1 DINO) (eval-only)",
        "FLOW vs APPEARANCE (pure, matched-k)",
        "FLOW vs HOF (pure, matched-k)",
        "HOF vs APPEARANCE (pure, matched-k)",
    ]
    omap = {k: i for i, k in enumerate(order)}
    if not summary_non_inc.empty:
        summary_non_inc["__o"] = summary_non_inc["comparison"].map(lambda x: omap.get(str(x), 999))
        summary_non_inc = summary_non_inc.sort_values(["__o", "comparison"]).drop(columns=["__o"]).reset_index(drop=True)
    if not summary_inc.empty:
        summary_inc = summary_inc.sort_values(["comparison", "added_predictor_label"]).reset_index(drop=True)
    if not robustness.empty:
        robustness = robustness.sort_values(["comparison", "added_predictor_label", "distance_type"]).reset_index(drop=True)
    if not top_recipes_debug.empty:
        top_recipes_debug = top_recipes_debug.sort_values(
            ["comparison", "added_predictor_label", "base_recipe", "distance_type"]
        ).reset_index(drop=True)

    claim_table_main = pd.concat([summary_non_inc, summary_inc], ignore_index=True, sort=False)
    if not claim_table_main.empty:
        claim_table_main["added_predictor_label"] = claim_table_main["added_predictor_label"].fillna("")
        claim_table_main["n_recipes"] = _safe_num(claim_table_main["n_recipes"]).fillna(0).astype(int)

    non_inc_main = summary_non_inc[
        [
            "comparison",
            "n_cells",
            "n_folds",
            "n_contexts",
            "delta_cindex_mean",
            "delta_cindex_median",
            "pos_frac_cindex",
            "tie_frac_cindex",
            "neg_frac_cindex",
            "delta_mae_mean",
            "delta_mae_median",
        ]
    ].copy()

    # Build the main incremental table: one row per (comparison, added_predictor_label,
    # distance_type) restricted to canonical distance families.  No performance-based
    # selection — distance families are fixed a priori to represent the operators used
    # in the main pipeline.
    canonical_types_arg = str(args.canonical_distance_types).strip()
    if canonical_types_arg.lower() == "all":
        canonical_set: set = set()  # empty = no filter
    else:
        canonical_set = {t.strip() for t in canonical_types_arg.split(",") if t.strip()}

    if robustness.empty:
        inc_main = summary_inc[
            [
                "comparison",
                "added_predictor_label",
                "n_recipes",
                "n_cells",
                "n_folds",
                "n_contexts",
                "delta_cindex_mean",
                "delta_cindex_median",
                "pos_frac_cindex",
                "tie_frac_cindex",
                "neg_frac_cindex",
                "delta_mae_mean",
                "delta_mae_median",
            ]
        ].copy()
    else:
        filtered = robustness.copy()
        if canonical_set:
            filtered = filtered[filtered["distance_type"].isin(canonical_set)].copy()
        filtered = filtered.sort_values(
            ["comparison", "added_predictor_label", "distance_type"]
        ).reset_index(drop=True)
        inc_main = filtered[
            [
                "comparison",
                "added_predictor_label",
                "distance_type",
                "n_recipes",
                "n_cells",
                "n_folds",
                "n_contexts",
                "delta_cindex_mean",
                "delta_cindex_median",
                "pos_frac_cindex",
                "tie_frac_cindex",
                "neg_frac_cindex",
                "delta_mae_mean",
                "delta_mae_median",
            ]
        ].copy()
    robustness_main = robustness[
        [
            "comparison",
            "added_predictor_label",
            "distance_type",
            "n_recipes",
            "n_cells",
            "n_folds",
            "n_contexts",
            "delta_cindex_mean",
            "delta_cindex_median",
            "pos_frac_cindex",
            "tie_frac_cindex",
            "delta_mae_mean",
            "delta_mae_median",
        ]
    ].copy()

    pairs.to_csv(out_dir / "comparison_pairs.csv", index=False)
    paired.to_csv(out_dir / "paired_method_cell_deltas.csv", index=False)
    cell.to_csv(out_dir / "cell_level_deltas.csv", index=False)
    claim_table_main.to_csv(out_dir / "claim_table_main.csv", index=False)
    robustness.to_csv(out_dir / "robustness_by_distance_family.csv", index=False)
    top_recipes_debug.to_csv(out_dir / "top_recipes_debug.csv", index=False)
    # Backward-compatible filenames.
    summary_non_inc.to_csv(out_dir / "claim_evidence_loto.csv", index=False)
    summary_inc.to_csv(out_dir / "claim_evidence_incremental.csv", index=False)
    robustness.to_csv(out_dir / "claim_evidence_by_distance_type.csv", index=False)
    non_inc_main.to_csv(out_dir / "claim_evidence_loto_main.csv", index=False)
    inc_main.to_csv(out_dir / "claim_evidence_incremental_main.csv", index=False)
    robustness_main.to_csv(out_dir / "claim_evidence_by_distance_type_main.csv", index=False)

    _write_tex_simple(
        non_inc_main,
        out_dir / "claim_evidence_loto.tex",
        caption=(
            "Non-incremental claim evidence under LOTO evaluation. "
            "Each cell = one (fold, context) pair; $\\Delta$c-index = method\\_a $-$ method\\_b "
            "(positive = method\\_a ranks better); $\\Delta$MAE = baseline $-$ method "
            "(positive = error reduction). "
            "Ties in c-index arise from the discrete set of candidate datasets per cell."
        ),
        label="tab:claim_evidence_loto",
    )
    canonical_note = (
        f"Distance family fixed to canonical operators ({canonical_types_arg}); "
        "no performance-based selection is performed."
        if canonical_types_arg.lower() != "all"
        else "All distance families shown."
    )
    _write_tex_simple(
        inc_main,
        out_dir / "claim_evidence_incremental.tex",
        caption=(
            "Incremental claim evidence under LOTO evaluation. "
            "Paired $\\Delta$c-index after adding one predictor type, "
            "collapsed across parameter-matched baseline recipes by within-cell median "
            "(positive = improvement in ranking). "
            + canonical_note
            + " Full robustness across distance families is in the supplementary."
        ),
        label="tab:claim_evidence_incremental",
    )
    _write_tex_simple(
        robustness_main,
        out_dir / "claim_evidence_by_distance_type.tex",
        caption="LOTO Robustness by Distance Family (collapsed across recipes within each fold-context cell)",
        label="tab:claim_evidence_distance_family",
    )
    include = "\n\n".join(
        [
            (out_dir / "claim_evidence_loto.tex").read_text(encoding="utf-8").strip(),
            (out_dir / "claim_evidence_incremental.tex").read_text(encoding="utf-8").strip(),
        ]
    ) + "\n"
    (out_dir / "include_all_tables.tex").write_text(include, encoding="utf-8")

    print(f"Wrote {out_dir / 'comparison_pairs.csv'}")
    print(f"Wrote {out_dir / 'paired_method_cell_deltas.csv'}")
    print(f"Wrote {out_dir / 'cell_level_deltas.csv'}")
    print(f"Wrote {out_dir / 'claim_table_main.csv'}")
    print(f"Wrote {out_dir / 'robustness_by_distance_family.csv'}")
    print(f"Wrote {out_dir / 'top_recipes_debug.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_loto.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_incremental.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_by_distance_type.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_loto_main.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_incremental_main.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_by_distance_type_main.csv'}")
    print(f"Wrote {out_dir / 'claim_evidence_loto.tex'}")
    print(f"Wrote {out_dir / 'claim_evidence_incremental.tex'}")
    print(f"Wrote {out_dir / 'claim_evidence_by_distance_type.tex'}")
    print(f"Wrote {out_dir / 'include_all_tables.tex'}")


if __name__ == "__main__":
    main()
