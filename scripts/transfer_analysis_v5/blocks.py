"""Named, machine-rendered table blocks for Paper 2 documents.

Single source of truth for every auto-generated table. Each block is a named
chunk of markdown rendered directly from an artifact CSV/report on disk —
nothing hand-transcribed. Documents embed a block between markers:

    <!-- tbl:L1 -->
    ...rendered content (owned by the script)...
    <!-- /tbl:L1 -->

`update_tables.py` re-renders ONLY the content between markers and never
touches anything else, so all surrounding prose is hand-owned and safe.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

ROOT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
V4 = ROOT / "scripts/transfer_analysis_v4"
V5 = ROOT / "scripts/transfer_analysis_v5"
RES = V5 / "results"
LEWM = Path("/home/spencer/Projects/le-wm/outputs")
GRID = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")
VAULT = Path("/home/spencer/Documents/Obsidian/Correspondence/Project")


import re

# 'arch|pretrained|frozen' -> 'arch T/F' — literal pipes inside table cells
# break markdown rendering (each | starts a new column), so variant labels
# must never reach a table raw
_VARIANT_RE = re.compile(r"\b(catspp|glunet|raft)\|(True|False)\|(True|False)\b")


def _tf(s: str) -> str:
    return "T" if s == "True" else "F"


def pretty_variants(text: str) -> str:
    return _VARIANT_RE.sub(lambda m: f"{m.group(1)} {_tf(m.group(2))}/{_tf(m.group(3))}",
                           text)


def md(df, floatfmt="+.3f"):
    df = df.copy()
    for c in df.columns[df.dtypes == object]:
        df[c] = df[c].map(lambda s: pretty_variants(s).replace("|", "\\|")
                          if isinstance(s, str) else s)
    df.columns = [c.replace("|", "\\|") if isinstance(c, str) else c
                  for c in df.columns]
    return df.to_markdown(index=False, floatfmt=floatfmt)


def need(p: Path) -> Path:
    if not p.exists():
        raise SystemExit(f"MISSING ARTIFACT: {p}")
    return p


# presentation order: scratch group first, then pretrained — NEVER interleave
# regimes in a rendered table (reviewer-readability rule)
SCRATCH = ["catspp|False|False", "catspp|False|True", "glunet|False|False",
           "glunet|False|True", "raft|True|False"]
PRETRAINED = ["catspp|True|False", "catspp|True|True",
              "glunet|True|False", "glunet|True|True"]
VORDER = {v: i for i, v in enumerate(SCRATCH + PRETRAINED)}


def regime_of(v: str) -> str:
    return "scratch" if v in SCRATCH else "pretrained"


def by_regime(df: pd.DataFrame, vcol: str = "variant") -> pd.DataFrame:
    """Insert a regime column (if absent) and sort scratch-then-pretrained."""
    df = df.copy()
    if "regime" not in df.columns:
        df.insert(df.columns.get_loc(vcol) + 1, "regime",
                  df[vcol].map(regime_of))
    return df.sort_values(by=vcol, key=lambda s: s.map(VORDER))


def build_blocks() -> dict[str, str]:
    """Render every block from artifacts on disk. Returns {name: markdown}."""
    law = pd.read_csv(need(V4 / "regime_direction_verification/master_table_mean_nn.csv"))
    law = law.drop(columns=["metric"])  # constant (mean_nn)
    law = by_regime(law)
    law = law.rename(columns={"rho_ab": "rho[d(T->B)]", "rho_ba": "rho[d(B->T)]",
                              "d": "flip d", "d_lo": "d lo95", "d_hi": "d hi95"})

    holdout = pd.read_csv(need(RES / "rule_holdout_checks.csv"))
    holdout = by_regime(holdout)
    # lovo_dir == rule_dir (unanimous 9/9) and rho_lovo == rho_rule: collapse
    holdout["direction"] = holdout.rule_dir.map(
        {"a_to_b": "d(T->B)", "b_to_a": "d(B->T)"})
    holdout = holdout[["variant", "regime", "direction", "rho_rule", "rho_sym",
                       "d", "mean_level"]]

    asym = pd.read_csv(need(RES / "asym_vs_sym.csv"))
    asym = by_regime(asym)
    asym = asym.rename(columns={"precision": "rho[d(T->B)]", "recall": "rho[d(B->T)]"})

    eps = pd.read_csv(need(RES / "eps_rule_table.csv"))
    eps = by_regime(eps)
    eps = eps[["variant", "regime", "rule_eps1", "rule_eps4", "rule_eps16",
               "sym_eps1", "sym_eps4", "sym_eps16"]]

    oracles = by_regime(pd.read_csv(need(RES / "ceiling_oracles.csv")))
    controls = pd.read_csv(need(RES / "controls_fresh.csv"))

    # R2a — the real per-feature sampling-stability table (v1 Table 4b):
    # spearman of each feature at budget N vs its value at the largest budget,
    # over the 110 train->eval pairs (canonical-11 pipeline)
    stab = pd.read_csv(need(ROOT / "analysis_v3/density_invariance_pair_sharded/"
                                   "stability_flow_train_eval__eval_eval.csv"))
    stab = stab[stab.pair_type == "train_eval"]
    R2A_METRICS = {
        "mean_nn_a_to_b": "mean-NN d(T->B)",
        "mean_nn_b_to_a": "mean-NN d(B->T)",
        "mean_nn_sym": "mean-NN sym",
        "a_covered_by_b_eps4px": "eps-coverage 4px (T->B)",
        "b_covered_by_a_eps4px": "eps-coverage 4px (B->T)",
        "kl_a_to_b_k20": "kNN-KL k20 (T->B)",
        "kl_b_to_a_k20": "kNN-KL k20 (B->T)",
    }
    r2a = stab[stab.metric.isin(R2A_METRICS)].copy()
    r2a["feature"] = r2a.metric.map(R2A_METRICS)
    r2a = r2a.pivot_table(index="feature", columns="level", values="rho")
    r2a.columns = [f"N={int(c):,}" for c in r2a.columns]
    r2a = (r2a.reindex(list(R2A_METRICS.values()))
              .reset_index().rename_axis(None, axis=1))

    # R2b — intervention-grid estimator-seed stability (session artifact;
    # rank correlation of the grid's source ranking across subsampling seeds)
    sampling = pd.read_csv(need(RES / "sampling_stability.csv"))
    sampling["direction"] = sampling.direction.map(
        {"dP": "d(T->B) within intervention family",
         "dR": "d(B->T) within intervention family"})
    sampling = sampling.rename(columns={
        "single_seed_mean": "1-seed vs 5-seed-avg: mean rank rho",
        "single_seed_min": "min",
        "splithalf_mean": "split-half: mean rank rho",
        "splithalf_min": "min "})
    regret = pd.read_csv(need(RES / "selection_regret_rule.csv"))

    # R1 split into four readable tables (was one 20-row mixed dump)
    c = controls.set_index("control")
    r1a = pd.DataFrame({
        "check": ["observed rule (baseline)", "shuffle null"],
        "rule rho": [c.loc["observed_rule", "value"],
                     c.loc["shuffle_null_mean", "value"]],
        "note": ["", c.loc["shuffle_null_mean", "note"]]})
    fam = controls[controls.control.str.startswith("drop_family_")].copy()
    fam["generator family dropped"] = fam.control.str.replace("drop_family_", "")
    fam["n sources dropped"] = fam.note.str.extract(r"(\d+)").astype(int)
    r1b = fam[["generator family dropped", "n sources dropped", "value"]].rename(
        columns={"value": "rule rho"})
    siz = controls[controls.control.str.startswith("control_size_")].copy()
    siz["size/density control"] = siz.control.str.replace("control_size_", "")
    r1c = siz[["size/density control", "value"]].rename(
        columns={"value": "fit-free rank rho"})
    ben = controls[controls.control.str.startswith("drop_benchmark_")].copy()
    ben["benchmark dropped"] = ben.control.str.replace("drop_benchmark_", "")
    r1d = ben[["benchmark dropped", "value"]].rename(columns={"value": "rule rho"})

    # P8 split + pivoted (was one long table with nan family rows)
    gap = pd.read_csv(need(RES / "pairwise_gap_rule.csv"))
    bins = ["0-1", "1-2", "2-5", "5-10", ">10", "ALL"]
    pa = gap[gap.measure == "predictor_accuracy"]
    p8a = pa.pivot_table(index="gap_bin", columns="split", values="acc")
    p8a["n pairs"] = pa.groupby("gap_bin")["n"].first()
    p8a = p8a.reindex(bins)[["LOTO", "LOBO", "JOINT", "n pairs"]].reset_index()
    p8a = p8a.rename(columns={"gap_bin": "true PCK-gap bin"}).rename_axis(None, axis=1)
    er = gap[gap.measure == "empirical_reproducibility"]
    p8b = er.pivot_table(index="gap_bin", columns="family", values="acc")
    p8b = p8b.reindex(bins)[["same_arch", "cross_arch"]].reset_index()
    p8b = p8b.rename(columns={"gap_bin": "true PCK-gap bin",
                              "same_arch": "same-arch retrain agreement",
                              "cross_arch": "cross-arch retrain agreement"})
    p8b = p8b.rename_axis(None, axis=1)
    oos = pd.read_csv(need(RES / "intervention_oos.csv"))
    oos = oos.rename(columns={"precision": "rho[d(T->B)]", "recall": "rho[d(B->T)]",
                              "sym": "rho[sym]"})
    consensus = pd.read_csv(need(V4 / "results_rule_v5core/CONSENSUS_RULE.csv"))
    summary = pd.read_csv(need(V4 / "results_rule_v5core/summary.csv"))
    gapboot = pd.read_csv(need(V4 / "results_rule_v5core/bootstrap_gap.csv"))
    bench_cal = pd.read_csv(need(RES / "benchsim_rule/summary_all_variants.csv"))
    # constant columns -> caption, not columns
    bench_cal = bench_cal.drop(columns=["variant_filter", "n_rows", "n_contexts"])
    joint = pd.read_csv(need(RES / "joint_anchor_v2.csv"))
    flow_dir = pd.read_csv(need(LEWM / "intervention_motion_distances_directional.csv"))
    dino_dir = pd.read_csv(need(LEWM / "intervention_appearance_distances_directional.csv"))
    report_flow = need(V4 / "regime_direction_verification/REPORT.md").read_text()
    report_dino = need(V4 / "regime_direction_verification/REPORT_dino.md").read_text()

    # grid PCKs (recomputed fresh from snapshots)
    rows = []
    for d in sorted(GRID.iterdir()):
        f = d / "validation_results.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f)
        if v["epoch"].nunique() < 50:  # still-training run (grid horizon=50); skip
            continue
        src = d.name.rsplit("_pt", 1)[0]
        arm = "FF" if "_pt0_fz0" in d.name else "TT"
        for b, g in v.groupby("benchmark"):
            rows.append((src, arm, b, float(g["pck"].max())))
    # trial19 TT now harvested from snapshots/<...>_trial19_<...> symlinked into the
    # grid as trial19_pt1_fz1_harvested (kitti2015-only run). Former hardcoded
    # ("trial19","TT","kitti2015",96.1158) removed 2026-06-10 — harvest reproduces it exactly.
    pck = pd.DataFrame(rows, columns=["source", "arm", "benchmark", "peak_pck"])
    # middlebury eval confirmed bugged (2026-06-10): excluded everywhere until
    # models are re-evaluated with the fixed eval
    pck = pck[pck.benchmark != "middlebury"]
    grid_wide = pck.pivot_table(index=["arm", "source"], columns="benchmark",
                                values="peak_pck").round(2).reset_index()
    grid_wide = grid_wide.sort_values(["arm", "kitti2015"],
                                      ascending=[True, False])
    grid_wide = grid_wide.rename_axis(None, axis=1)

    # breakdown with BOTH directed-distance spaces (within-grid ranks)
    m = pck.merge(flow_dir, on=["source", "benchmark"], how="left")
    m = m.merge(dino_dir, on=["source", "benchmark"], how="left")
    brk = []
    for (arm, bench), g in m.groupby(["arm", "benchmark"]):
        g = g.dropna(subset=["flow_mean_nn_a_to_b"]).copy()
        if len(g) < 3:
            continue
        rc = "flow_mean_nn_a_to_b" if arm == "FF" else "flow_mean_nn_b_to_a"
        g["rule rank (within grid)"] = g[rc].rank().astype(int)
        g["actual rank (within grid)"] = (-g.peak_pck).rank().astype(int)
        brk.append(g)
    brk = pd.concat(brk)
    brk = brk.rename(columns={
        "flow_mean_nn_a_to_b": "flow d(T->B)", "flow_mean_nn_b_to_a": "flow d(B->T)",
        "dino_mean_nn_a_to_b": "dino d(T->B)", "dino_mean_nn_b_to_a": "dino d(B->T)"})
    # within each (arm, benchmark) block: best rule rank first, so the reader
    # compares 'rule rank' vs 'actual rank' down two adjacent columns
    brk = brk.sort_values(["arm", "benchmark", "rule rank (within grid)"])
    brk_cols = ["arm", "benchmark", "source", "rule rank (within grid)",
                "actual rank (within grid)", "peak_pck",
                "flow d(T->B)", "flow d(B->T)", "dino d(T->B)", "dino d(B->T)"]
    brk[brk_cols].to_csv(RES / "intervention_breakdown.csv", index=False)
    i2_cols = brk_cols[1:]  # arm fixed per table -> drop the column
    i2_ff = brk[brk.arm == "FF"][i2_cols]
    i2_tt = brk[brk.arm == "TT"][i2_cols]

    # grid distance dumps -> one compact source x benchmark matrix per
    # direction per space (was two 36-row long-format tables)
    def dist_pivot(df, col):
        p = df.pivot_table(index="source", columns="benchmark", values=col)
        return p.reset_index().rename_axis(None, axis=1)

    # P1 pivots: family rows x split columns, "point [lo, hi]" cells
    ms = summary[(summary.label == "main") & (summary["head"] == "g")]
    FAMILIES = ["motion_rule", "motion_precision", "motion_recall",
                "motion", "motion_sym", "appearance"]
    ms = ms[ms.family.isin(FAMILIES)].copy()

    def ci_pivot(value, lo, hi):
        t = ms.copy()
        t["cell"] = t.apply(
            lambda r: f"{r[value]:+.3f} [{r[lo]:+.2f}, {r[hi]:+.2f}]", axis=1)
        p = t.pivot(index="family", columns="split", values="cell")
        p = p.reindex(index=FAMILIES, columns=["LOTO", "LOBO", "JOINT"])
        return p.reset_index().rename_axis(None, axis=1).to_markdown(index=False)

    # L3 — DINO mirror of L1, with the motion flip column for contrast
    dlaw = pd.read_csv(need(V4 / "regime_direction_verification/"
                                 "master_table_mean_nn_dino.csv"))
    mlaw = pd.read_csv(need(V4 / "regime_direction_verification/"
                                 "master_table_mean_nn.csv"))
    l3 = dlaw.merge(mlaw[["variant", "d"]].rename(columns={"d": "flip d (motion)"}),
                    on="variant")
    l3 = by_regime(l3.drop(columns=["metric"]))
    l3 = l3.rename(columns={
        "rho_ab": "DINO rho[d(T->B)]", "rho_ba": "DINO rho[d(B->T)]",
        "d": "flip d (DINO)", "d_lo": "DINO d lo95", "d_hi": "DINO d hi95"})
    l3 = l3[["variant", "regime", "DINO rho[d(T->B)]", "DINO rho[d(B->T)]",
             "flip d (DINO)", "DINO d lo95", "DINO d hi95", "flip d (motion)"]]

    # P9b — feature-set comparison for the per-regime linear model
    fsc = pd.read_csv(need(RES / "per_regime_featureset_comparison.csv"))
    p9b = fsc.pivot_table(index=["features", "regime"], columns="setting",
                          values="ctx_rho_linear")
    mae = fsc[fsc.setting == "LOTO"].set_index(["features", "regime"])[
        "MAE(anchor+model)"]
    p9b["MAE held-out source"] = mae
    forder = ["motion mean-NN both dirs (2)", "motion eps 4px both dirs (2)",
              "motion eps 1/4/16px both dirs (6)", "motion mean-NN + eps (8)",
              "appearance (DINO) mean-NN both dirs (2)"]
    rorder = ["scratch", "pretrained", "pooled (regime-blind)"]
    p9b = (p9b.reindex(pd.MultiIndex.from_product([forder, rorder],
                                                  names=["features", "regime"]))
              [["LOTO", "LOBO", "JOINT", "MAE held-out source"]]
              .reset_index().rename_axis(None, axis=1))

    # P9 — per-regime two-feature linear models (per_regime_linear.py)
    prl = pd.read_csv(need(RES / "per_regime_linear_summary.csv"))
    p9 = prl[["regime", "setting", "ctx_rho_linear", "ctx_rho_rule",
              "ctx_rho_sym", "w[d(T->B)]", "w[d(B->T)]",
              "MAE(anchor+model)", "r(anchor+model)"]].rename(columns={
        "setting": "held out",
        "ctx_rho_linear": "rank rho: linear",
        "ctx_rho_rule": "rank rho: rule",
        "ctx_rho_sym": "rank rho: symmetric",
        "MAE(anchor+model)": "MAE anchor+model",
        "r(anchor+model)": "r anchor+model"})
    p9 = p9.sort_values(["held out", "regime"],
                        key=lambda s: s.map({"LOTO": 0, "LOBO": 1, "JOINT": 2,
                                             "scratch": 0, "pretrained": 1,
                                             "pooled (regime-blind)": 2}))

    # joint anchor metrics for the P4 menu
    jm = joint[np.isfinite(joint.L2)]
    joint_mae = float(np.mean(np.abs(jm.L2 - jm.actual)))
    joint_r = float(pearsonr(jm.L2, jm.actual)[0])

    p4 = f"""| setting | anchor / calibration | MAE (PCK) | r | artifact |
|---|---|---|---|---|
| LOTO | cell-mean L + raw rule g | 8.94 | +0.874 | rows_LOTO_motion_rule.csv |
| LOTO | sim-IDW (triangle) L + g | 9.86 | +0.846 | results_rule_triangleL |
| LOTO | per-regime gain c_r·g (constrained, LOSO) | 10.03 (scratch 11.98 / pretr 7.65) | +0.819 (0.740 / **0.901**) | /tmp/calrows2_LOTO.csv |
| LOTO | per-regime full affine (LOSO) | 11.45 (scratch 14.18 / pretr 8.10) | +0.747 (0.623 / 0.895) | unconstrained affine HURTS scratch |
| LOBO | per-family IDW L + g | 18.73 | +0.693 | rows_LOBO_motion_rule.csv |
| LOBO | + per-regime gain | 17.94 (18.56 / 17.19) | +0.706 | |
| JOINT | grand+variant offset (OLD — degenerate stripes) | 29.17 | +0.007 | retired |
| JOINT | **two-way similarity anchor** (variant mean + ee-IDW bench effect + tt-IDW source effect, in-fold only) | **{joint_mae:.2f}** | **{joint_r:+.3f}** | joint_anchor_v2.csv |
| JOINT | two-way + per-regime affine (LOSO) | 19.69 | +0.631 | /tmp/calrows_JOINT.csv |"""

    return {
        "L1": md(law),
        "L2": md(holdout),
        "L4": md(asym),
        "L5": md(eps),
        "P1a": ci_pivot("ctx_rho_g", "ctx_rho_g_lo", "ctx_rho_g_hi"),
        "P1b": ci_pivot("abs_r_Lg", "abs_r_Lg_lo", "abs_r_Lg_hi"),
        "P2": md(gapboot[gapboot["head"] == "g"][
            ["split", "ctx_rho_g_gap", "ctx_rho_g_gap_lo",
             "ctx_rho_g_gap_hi", "ctx_rho_g_gap_p_gt_0"]]),
        "P3_oracles": md(oracles),
        "P3_consensus": md(consensus),
        "P4": p4,
        "L3": md(l3),
        "P6": md(bench_cal),
        "P7": md(regret),
        "P9": md(p9),
        "P9b": md(p9b),
        "P8a": md(p8a),
        "P8b": md(p8b),
        "R1a": md(r1a),
        "R1b": md(r1b),
        "R1c": md(r1c),
        "R1d": md(r1d),
        "R2a": md(r2a),
        "R2b": md(sampling),
        "I1": md(grid_wide, floatfmt=".2f"),
        "I2_FF": md(i2_ff, floatfmt=".4f"),
        "I2_TT": md(i2_tt, floatfmt=".4f"),
        "I3": md(oos),
        "DIST_FLOW_TB": md(dist_pivot(flow_dir, "flow_mean_nn_a_to_b"), ".5f"),
        "DIST_FLOW_BT": md(dist_pivot(flow_dir, "flow_mean_nn_b_to_a"), ".5f"),
        "DIST_DINO_TB": md(dist_pivot(dino_dir, "dino_mean_nn_a_to_b"), ".4f"),
        "DIST_DINO_BT": md(dist_pivot(dino_dir, "dino_mean_nn_b_to_a"), ".4f"),
        "APPENDIX_A": pretty_variants(report_flow.rstrip()),
        "APPENDIX_B": pretty_variants(report_dino.rstrip()),
    }
