"""Render the ACCV 2026 paper tables as LaTeX (booktabs) from the artifact
CSVs — nothing hand-transcribed. Output: ACCV_2026/tables/*.tex, \\input from
main.tex / supp_main.tex.

Bolding conventions:
  - law table: the regime-matched direction's rho per row
  - predictor table: best value per column
  - OOS table: the matched direction column
  - per-regime linear: dominant coefficient + better of (linear, rule)

    python scripts/transfer_analysis_v5/make_paper_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from blocks import (RES, V4, SCRATCH, PRETRAINED, by_regime, need,  # noqa
                    regime_of)

OUT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence/ACCV_2026/tables")
OUT.mkdir(parents=True, exist_ok=True)

ARCH = {"catspp": "CATs++", "glunet": "GLU-Net", "raft": "RAFT"}


def vtex(v: str) -> str:
    a, p, f = v.split("|")
    tf = lambda s: "\\cmark" if s == "True" else "--"
    return f"{ARCH[a]} & {tf(p)} & {tf(f)}"


def fmt(x, bold=False, prec=2):
    s = f"{x:+.{prec}f}"
    return f"\\textbf{{{s}}}" if bold else s


def best(x, prec=2):
    """Green-highlighted cell for the genuine best value (honest emphasis)."""
    return f"\\best{{{x:+.{prec}f}}}"


def write(name: str, content: str):
    (OUT / name).write_text(content + "\n")
    print(f"wrote tables/{name}")


# ----------------- Table 1: law (a: motion, with symmetric contrast; b: DINO)
# directed rho from the verification artifact; symmetric columns from
# asym_vs_sym.csv (same within-context machinery)
law = pd.read_csv(need(V4 / "regime_direction_verification/master_table_mean_nn.csv"))
law = by_regime(law)
asym_m = pd.read_csv(need(RES / "asym_vs_sym.csv")).set_index("variant")


def law_rows(df, sym_lookup, with_fid=True, mark=True):
    rows, prev = [], None
    for _, r in df.iterrows():
        if prev is not None and r.regime != prev:
            rows.append(r"\midrule")
        prev = r.regime
        scr = r.regime == "scratch"
        sym, w2 = sym_lookup(r.variant)
        vals = {"ab": r.rho_ab, "ba": r.rho_ba, "sym": sym, "w2": w2}
        avail = {k: v for k, v in vals.items()
                 if v == v and (with_fid or k != "w2")}
        bestk = max(avail, key=avail.get) if (mark and avail) else None
        matched = "ab" if scr else "ba"

        def cell(k, _v=vals):
            v = _v[k]
            if v != v:
                return "--"
            s = f"{v:+.2f}"
            # two INDEPENDENT encodings: green = genuine best (any column);
            # bold = the rule's column (precision for scratch, recall for
            # pretrained). bold+green = rule aligns with best; bold only = miss.
            if mark and k == bestk:
                s = f"\\textcolor{{bestgreen}}{{{s}}}"
            if mark and k == matched:
                s = f"\\textbf{{{s}}}"
            return s
        tail = f" & {cell('w2')}" if with_fid else ""
        rows.append(f"{vtex(r.variant)} & {cell('ab')} & {cell('ba')} & "
                    f"{cell('sym')}{tail} " + r"\\")
    return rows


HEADER = ("Architecture & Backbone & Frozen & $\\rho[\\dtb]$ & "
          "$\\rho[\\dbt]$ & $\\rho[\\mathrm{sym}]$ & "
          "$\\rho[\\mathrm{W2}]$ \\\\")
write("tab_law.tex", "\n".join([
    "\\begin{tabular}{l cc rr rr}", "\\toprule",
    " & & & \\multicolumn{2}{c}{directed} & "
    "\\multicolumn{2}{c}{symmetric (contrast)} \\\\",
    "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}", HEADER, "\\midrule"]
    + law_rows(law, lambda v: (asym_m.loc[v, "sym"], asym_m.loc[v, "w2"]))
    + ["\\bottomrule", "\\end{tabular}"]))

# DINO mirror with its own symmetric column, computed from raw artifacts
dlaw = by_regime(pd.read_csv(need(
    V4 / "regime_direction_verification/master_table_mean_nn_dino.csv")))
tt = pd.read_csv(need(Path("scripts/transfer_analysis_v3/transfer_table_nomid.csv")))
PURE = ["flyingthings", "imagenet2dwarp", "movi_f", "pointodyssey", "sintel",
        "spair", "synthetic", "synthetic_2d_warp", "synthetic_large_zoom",
        "synthetic_random_flipping", "synthetic_small_zoom"]
tt = tt[tt.train_dataset.isin(PURE)].dropna(subset=["peak_pck"]).copy()
tt["variant"] = (tt.model_family.astype(str) + "|" + tt.pretrained.astype(str)
                 + "|" + tt.freeze.astype(str))
tt["cv"] = tt.benchmark + "|" + tt.variant
dd = pd.read_csv(need(Path("analysis_v3/pairwise_self_distances.csv")))
dd = dd[(dd.pair_type == "train_eval") & (dd.space == "dino")]
tt = tt.join(dd.set_index(["dataset_a", "dataset_b"])["mean_nn_sym"],
             on=["train_dataset", "benchmark"], how="left")
from scipy.stats import spearmanr  # noqa: E402


def dino_sym_rho(variant):
    g = tt[tt.variant == variant]
    out = []
    for _, c in g.groupby("cv"):
        if c.train_dataset.nunique() >= 3 and c.mean_nn_sym.std() > 1e-15:
            r = spearmanr(c.peak_pck, -c.mean_nn_sym).statistic
            if np.isfinite(r):
                out.append(r)
    return (float(np.mean(out)) if out else float("nan"), float("nan"))


write("tab_law_dino.tex", "\n".join([
    "\\begin{tabular}{l cc rr r}", "\\toprule",
    " & & & \\multicolumn{2}{c}{directed} & symmetric \\\\",
    "\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}",
    "Architecture & Backbone & Frozen & $\\rho[\\dtb]$ & "
    "$\\rho[\\dbt]$ & $\\rho[\\mathrm{sym}]$ \\\\", "\\midrule"]
    + law_rows(dlaw, lambda v: (dino_sym_rho(v)[0], float("nan")), with_fid=False)
    + ["\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Table 2: predictor families
s = pd.read_csv(need(V4 / "results_rule_v5core/summary.csv"))
ms = s[(s.label == "main") & (s["head"] == "g")]
FAMS = [("motion_policy", "Regime-aware policy (ours, fit-free)"),
        ("motion_rule", "Matched direction everywhere"),
        ("motion_precision", "$d_{T\\to B}$ only"),
        ("motion_recall", "$d_{B\\to T}$ only"),
        ("motion_meannn_sym", "Summed distance everywhere"),
        ("motion_meannn_both", "Mean-NN, both dir.\\ (fit)"),
        ("motion_sym", "FID/SW2/MMD"),
        ("motion", "Motion ridge (13-feat.)"),
        ("appearance", "Appearance ridge (DINO)")]
ms = ms.set_index(["family", "split"])
from scipy.stats import spearmanr as _sp  # noqa: E402
_PRED = V4 / "results_rule_v5core/predictions/peak_pck"


def rho_g(fam, sp):
    """ctx_rho_g from summary if present, else computed from the rows file."""
    if (fam, sp) in ms.index:
        return float(ms.loc[(fam, sp), "ctx_rho_g"])
    r = pd.read_csv(need(_PRED / f"rows_{sp}_{fam}.csv"))
    vals = []
    for _, c in r.groupby("context_id"):
        if c.train_dataset.nunique() >= 3 and c.g.std() > 1e-12:
            v = _sp(c.actual, c.g).statistic
            if v == v:
                vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


colbest = {sp: max(rho_g(f, sp) for f, _ in FAMS)
           for sp in ["LOTO", "LOBO", "JOINT"]}
rows = []
for f, label in FAMS:
    cells = []
    for sp in ["LOTO", "LOBO", "JOINT"]:
        g = rho_g(f, sp)
        isbest = abs(g - colbest[sp]) < 1e-9
        cells.append(best(g) if isbest else f"{g:+.2f}")
    rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    if f == "motion_policy":
        rows.append(r"\midrule")
write("tab_predictors.tex", "\n".join([
    "\\begin{tabular}{l ccc}",
    "\\toprule",
    "Predictor & \\makecell{Held-out\\\\source} & \\makecell{Held-out\\\\bench.} & \\makecell{Both\\\\held} \\\\",
    "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]))

# --------------------- Table 3a: per-regime predictor ablation (the key one)
import numpy as np  # noqa
av = pd.read_csv(need(RES / "asym_vs_sym.csv"))
av["regime"] = av.variant.map(regime_of)
prl = pd.read_csv(need(RES / "per_regime_linear_summary.csv"))


def _rm(col, reg):
    s = av[av.regime == reg][col]
    return s.mean(), s.std() / np.sqrt(len(s))


def _cell(col, reg, bold=False):
    m, e = _rm(col, reg)
    s = f"{m:+.2f}\\,{{\\scriptsize$\\pm${e:.2f}}}"
    return f"\\textbf{{{s}}}" if bold else s


# best fit-free predictor per regime: symmetric for scratch, recall for pretrained
rows = [
    ("off-target mass $\\dtb$", "precision", {"scratch": False, "pretrained": False}),
    ("missing support $\\dbt$", "recall", {"scratch": False, "pretrained": True}),
    ("symmetric (mean of the two)", "sym", {"scratch": True, "pretrained": False}),
]
body = [f"{name} & {_cell(col, 'scratch', b['scratch'])} & "
        f"{_cell(col, 'pretrained', b['pretrained'])} \\\\"
        for name, col, b in rows]
# 2-coef linear: held-out (LOTO), reported for contrast
lin = prl.set_index(["regime", "setting"]).ctx_rho_linear
body.append("\\addlinespace")
body.append(f"two-coefficient fit (held out) & {lin[('scratch','LOTO')]:+.2f} & "
            f"{lin[('pretrained','LOTO')]:+.2f} \\\\")
write("tab_perregime_ablation.tex", "\n".join([
    "\\begin{tabular}{l cc}", "\\toprule",
    "within-context $\\rho$ & from scratch & pretrained backbone \\\\",
    "\\midrule"] + body + ["\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------ Table 3: per-regime linear models
p9 = pd.read_csv(need(RES / "per_regime_linear_summary.csv"))
# fit-free policy arm per regime (from the per-variant fit-free table):
# scratch arm = summed distance (the symmetric mean), pretrained arm = recall
_avp = pd.read_csv(need(RES / "asym_vs_sym.csv"))
_avp["regime"] = _avp.variant.map(regime_of)
POLICY_RHO = {
    "scratch": _avp[_avp.regime == "scratch"]["sym"].mean(),
    "pretrained": _avp[_avp.regime == "pretrained"]["recall"].mean(),
    "pooled (regime-blind)": pd.concat(
        [_avp[_avp.regime == "scratch"]["sym"],
         _avp[_avp.regime == "pretrained"]["recall"]]).mean(),
}
rows = []
for reg, label in [("scratch", "from scratch"), ("pretrained", "pretrained"),
                   ("pooled (regime-blind)", "pooled (regime-blind)")]:
    r = p9[(p9.regime == reg)].set_index("setting")
    w_ab, w_ba = r.loc["LOTO", "w[d(T->B)]"], r.loc["LOTO", "w[d(B->T)]"]
    # bold = the cost(s) the regime pays: both from scratch, recall pretrained
    bold_ab = reg == "scratch"
    bold_ba = reg in ("scratch", "pretrained")
    lin = [r.loc[sp, "ctx_rho_linear"] for sp in ["LOTO", "LOBO", "JOINT"]]
    pol = POLICY_RHO[reg]
    # bold the fit only if it beats the fit-free arm by a real margin;
    # ties go to fit-free (nothing to overfit)
    lin_better = np.mean(lin) > pol + 0.02
    rows.append(
        f"{label} & {fmt(w_ab, bold=bold_ab)} & "
        f"{fmt(w_ba, bold=bold_ba)} & "
        + " / ".join(fmt(v, bold=lin_better) for v in lin)
        + f" & {fmt(pol, bold=not lin_better)} \\\\")
write("tab_regime_linear.tex", "\n".join([
    "\\begin{tabular}{l rr c c}",
    "\\toprule",
    " & \\multicolumn{2}{c}{coefficients} & ranking $\\rho$ & policy $\\rho$ \\\\",
    "\\cmidrule(lr){2-3}",
    "Model & $w[d_{T\\to B}]$ & $w[d_{B\\to T}]$ & "
    "(src / bench / both held out) & (fit-free)$^{*}$ \\\\",
    "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]))

# --------- Table 4: OOS test — each correlation next to the variation that
# licenses it (a rho only carries causal weight where its factor has dynamic
# range). FlyingThings is the discriminative cell (off-target varies 3.1x,
# coverage ~flat); the KITTI cells are precision-matched by family design
# (off-target spread <= 1.04x) and reported as sign-consistent only.
oos = pd.read_csv(need(RES / "intervention_oos.csv"))
ff = oos[oos.arm == "FF"]
rows = []
for _, r in ff.iterrows():
    disc = r.prec_spread > 1.5  # off-target has real dynamic range here
    rows.append(
        f"{r.benchmark} & {int(r.n)} & "
        f"{r.prec_spread:.2f}$\\times$ & {fmt(r.precision, bold=disc)} & "
        f"{r.rec_spread:.2f}$\\times$ & {fmt(r.recall)} \\\\")
# no pooled mean row: it blends the one discriminative cell (FlyingThings)
# with the appearance-confounded KITTI cells and reads as a clean +0.66 when
# it is not. The per-benchmark rows + spread columns are the honest view.
write("tab_oos.tex", "\n".join([
    "\\begin{tabular}{l c cc cc}",
    "\\toprule",
    " & & \\multicolumn{2}{c}{off-target mass $\\dtb$} & "
    "\\multicolumn{2}{c}{missing support $\\dbt$} \\\\",
    "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
    "Benchmark & $n$ & \\makecell{varies\\\\(max/min)} & "
    "\\makecell{ranks transfer?\\\\$\\rho$} & \\makecell{varies\\\\(max/min)} & "
    "\\makecell{ranks transfer?\\\\$\\rho$} \\\\",
    "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]))

# ------------------------------------------------- Table 5: decision utility
reg = pd.read_csv(need(RES / "selection_regret_rule.csv"))
gap = pd.read_csv(need(RES / "pairwise_gap_rule.csv"))
SPLITS = ["LOTO", "LOBO", "JOINT"]


def _med_regret(fam, sp):
    return reg[(reg.split == sp) & (reg.family == fam)].median_regret.iloc[0]


def _acc10(fam, sp):
    return gap[(gap.split == sp) & (gap.family == fam)
               & (gap.gap_bin == ">10")].acc.iloc[0]


# rows = selection strategies; the recommended regime-aware policy
# (symmetric from scratch, matched direction d_B->T with a pretrained
# backbone; built by make_policy_rows.py) leads.
PRED_ROWS = [
    ("regime-aware policy (ours)", "motion_policy", True),
    ("matched direction everywhere", "motion_rule", False),
    ("summed distance everywhere", "motion_meannn_sym", False),
    ("appearance similarity (DINO)", "appearance", False),
]
rows = []
for name, fam, bold in PRED_ROWS:
    cells = [f"{_med_regret(fam, sp):.1f}" for sp in SPLITS] + \
            [f"{_acc10(fam, sp):.2f}" for sp in SPLITS]
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    rows.append(f"{name} & " + " & ".join(cells) + " \\\\")
rnd = reg[reg.family == "motion_policy"].random_mean_regret.iloc[0]
rows.append(f"random pick (expected) & {rnd:.1f} & {rnd:.1f} & {rnd:.1f} & "
            "0.50 & 0.50 & 0.50 \\\\")
sa = gap[(gap.family == "same_arch") & (gap.gap_bin == ">10")].acc.iloc[0]
ca = gap[(gap.family == "cross_arch") & (gap.gap_bin == ">10")].acc.iloc[0]
rows += ["\\midrule",
         f"\\emph{{retraining, same architecture}} & --- & --- & --- "
         f"& \\multicolumn{{3}}{{c}}{{{sa:.2f}}} \\\\",
         f"\\emph{{retraining, different architecture}} & --- & --- & --- "
         f"& \\multicolumn{{3}}{{c}}{{{ca:.2f}}} \\\\"]
write("tab_utility.tex", "\n".join([
    "\\begin{tabular}{l ccc ccc}",
    "\\toprule",
    " & \\multicolumn{3}{c}{median top-1 regret (PCK) $\\downarrow$} & "
    "\\multicolumn{3}{c}{pairwise acc., gap $>$10 $\\uparrow$} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
    "Selection strategy \\hfill (held out $\\rightarrow$) & source & "
    "bench. & both & source & bench. & both \\\\",
    "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]))

# --------- COMPREHENSIVE per-regime direction test (definitive) -> by Table 1
oosd = pd.read_csv(need(RES / "intervention_oos.csv"))
ff = oosd[oosd.arm == "FF"]
tt_ = oosd[oosd.arm == "TT"]
av2 = pd.read_csv(need(RES / "asym_vs_sym.csv"))
av2["regime"] = av2.variant.map(regime_of)
_M = {"off": ("precision", "precision"), "miss": ("recall", "recall"),
      "sym": ("sym", "sym")}


def _canon(metric, reg):
    return av2[av2.regime == reg][_M[metric][0]].mean()


def _grid(metric, frame):
    return frame[_M[metric][1]].mean()


PRED = [("off", r"off-target mass $\dtb$"),
        ("miss", r"missing support $\dbt$"),
        ("sym", "symmetric (mean of the two)")]
# every column rendered at full contrast, green = best per column; the grid
# pretrained arm (dagger) is seed-unstable but NOT greyed — the nominal
# failure mode is shown openly and argued in the caption.
colv = {"sc": {m: _canon(m, "scratch") for m, _ in PRED},
        "ct": {m: _grid(m, ff) for m, _ in PRED},
        "pc": {m: _canon(m, "pretrained") for m, _ in PRED},
        "tt": {m: _grid(m, tt_) for m, _ in PRED}}
cmax = {k: max(v.values()) for k, v in colv.items()}
body = []
for m, lab in PRED:
    cells = []
    for k in ["sc", "ct", "pc", "tt"]:
        v = colv[k][m]
        cells.append(best(v) if abs(v - cmax[k]) < 1e-9 else f"{v:+.2f}")
    body.append(f"{lab} & " + " & ".join(cells) + r" \\")
# composite scores, read off the rows above (regime decides which arm each
# column uses): the design objective tracks the binding direction everywhere;
# the selection policy's scratch arm (the sum) goes blind on the grid BY
# CONSTRUCTION (coverage held fixed -> its recall term is noise) — shown, not
# hidden, because it demarcates selection (natural pools) from design.
_comp = [
    (r"\emph{matched direction (design objective)}",
     [colv["sc"]["off"], colv["ct"]["off"],
      colv["pc"]["miss"], colv["tt"]["miss"]]),
    (r"\emph{regime-aware policy (selection score)}",
     [colv["sc"]["sym"], colv["ct"]["sym"],
      colv["pc"]["miss"], colv["tt"]["miss"]]),
]
body.append(r"\midrule")
for lab, vals in _comp:
    body.append(f"{lab} & " + " & ".join(f"{v:+.2f}" for v in vals) + r" \\")
write("tab_dirtest.tex", "\n".join([
    r"\begin{tabular}{l cc cc}", r"\toprule",
    r" & \multicolumn{2}{c}{\textbf{from scratch}} & "
    r"\multicolumn{2}{c}{\textbf{pretrained backbone}} \\",
    r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
    r"within-context $\rho$ & canonical & controlled & canonical & "
    r"controlled$^{\dagger}$ \\",
    r" & (11 src) & (grid) & (11 src) & (grid) \\", r"\midrule"] + body
    + [r"\bottomrule", r"\end{tabular}"]))

# ============================ SUPPLEMENTARY TABLES ============================
def simple_table(df, colspec, header, name, prec=2, bold_mask=None):
    body = []
    for i, (_, r) in enumerate(df.iterrows()):
        cells = []
        for j, v in enumerate(r):
            if isinstance(v, (int, np.integer)):
                cells.append(str(v))
            elif isinstance(v, float):
                b = bold_mask is not None and bold_mask[i][j]
                cells.append(fmt(v, bold=b, prec=prec))
            else:
                cells.append(str(v).replace("_", "\\_"))
        body.append(" & ".join(cells) + " \\\\")
    write(name, "\n".join(
        [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
        + body + ["\\bottomrule", "\\end{tabular}"]))


# supp: asym vs sym (L4)
asym = by_regime(pd.read_csv(need(RES / "asym_vs_sym.csv")))
asym["variant"] = asym.variant.map(lambda v: vtex(v).replace(" & ", " "))
asym = asym[["variant", "regime", "rule", "precision", "recall", "sym",
             "fid", "w2", "mmd"]]
simple_table(asym, "l l rrrrrrr",
             "Variant & Regime & rule & $\\rho[d_{T\\to B}]$ & "
             "$\\rho[d_{B\\to T}]$ & sym & FID & SW2 & MMD \\\\",
             "tab_supp_asym.tex")

# supp: eps replication (L5)
eps = by_regime(pd.read_csv(need(RES / "eps_rule_table.csv")))
eps = eps[["variant", "regime", "rule_eps1", "rule_eps4", "rule_eps16",
           "sym_eps1", "sym_eps4", "sym_eps16"]]
eps["variant"] = eps.variant.map(lambda v: vtex(v).replace(" & ", " "))
simple_table(eps, "l l rrr rrr",
             "Variant & Regime & \\multicolumn{3}{c}{rule "
             "($\\epsilon{=}1/4/16$\\,px)} & \\multicolumn{3}{c}{symmetric} \\\\",
             "tab_supp_eps.tex")

# supp: oracles (P3)
oo = by_regime(pd.read_csv(need(RES / "ceiling_oracles.csv")))
oo = oo[["variant", "regime", "rule_rho", "O2_same_regime",
         "O3_same_regime_consensus", "frac_O2", "frac_O3"]]
oo["variant"] = oo.variant.map(lambda v: vtex(v).replace(" & ", " "))
simple_table(oo, "l l rrrrr",
             "Variant & Regime & rule $\\rho$ & oracle $O_2$ & oracle $O_3$ & "
             "frac.\\ $O_2$ & frac.\\ $O_3$ \\\\",
             "tab_supp_oracles.tex")

# supp: gap-stratified (P8a/b merged)
bins = ["0-1", "1-2", "2-5", "5-10", ">10", "ALL"]
pa = gap[gap.measure == "predictor_accuracy"].pivot_table(
    index="gap_bin", columns="split", values="acc").reindex(bins)
er = gap[gap.measure == "empirical_reproducibility"].pivot_table(
    index="gap_bin", columns="family", values="acc").reindex(bins)
rows = []
for b in bins:
    rows.append(f"{b} & {pa.loc[b,'LOTO']:.3f} & {pa.loc[b,'LOBO']:.3f} & "
                f"{pa.loc[b,'JOINT']:.3f} & {er.loc[b,'same_arch']:.3f} & "
                f"{er.loc[b,'cross_arch']:.3f} \\\\")
write("tab_supp_gap.tex", "\n".join([
    "\\begin{tabular}{l ccc cc}", "\\toprule",
    " & \\multicolumn{3}{c}{rule accuracy} & "
    "\\multicolumn{2}{c}{retraining agreement} \\\\",
    "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}",
    "True $|$PCK gap$|$ & src.\\ held & bench.\\ held & both & "
    "same arch. & cross arch. \\\\", "\\midrule"]
    + rows + ["\\bottomrule", "\\end{tabular}"]))

# supp: controls (R1)
c = pd.read_csv(need(RES / "controls_fresh.csv")).set_index("control")
rows = [f"shuffle null (200 perms) & ${c.loc['shuffle_null_mean','value']:+.3f}"
        f"$ \\\\"]
for k in [k for k in c.index if k.startswith("drop_family_")]:
    rows.append(f"drop family \\emph{{{k.replace('drop_family_','')}}} & "
                f"${c.loc[k,'value']:+.3f}$ \\\\")
for k in [k for k in c.index if k.startswith("control_size_")]:
    rows.append(f"size control: {k.replace('control_size_','').replace('_',' ')}"
                f" & ${c.loc[k,'value']:+.3f}$ \\\\")
bmin = min(c.loc[k, "value"] for k in c.index if k.startswith("drop_benchmark"))
bmax = max(c.loc[k, "value"] for k in c.index if k.startswith("drop_benchmark"))
rows.append(f"drop any single benchmark (range) & $[{bmin:+.3f},\\,{bmax:+.3f}]$ \\\\")
write("tab_supp_controls.tex", "\n".join([
    "\\begin{tabular}{l r}", "\\toprule",
    f"Control (baseline rule $\\rho = {c.loc['observed_rule','value']:+.3f}$) & "
    "$\\rho$ \\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabular}"]))

# supp: feature-set comparison (P9b)
fsc = pd.read_csv(need(RES / "per_regime_featureset_comparison.csv"))
piv = fsc.pivot_table(index=["features", "regime"], columns="setting",
                      values="ctx_rho_linear")
mae = fsc[fsc.setting == "LOTO"].set_index(["features", "regime"])[
    "MAE(anchor+model)"]
rows = []
for feat in ["motion mean-NN both dirs (2)", "motion eps 4px both dirs (2)",
             "motion eps 1/4/16px both dirs (6)", "motion mean-NN + eps (8)",
             "appearance (DINO) mean-NN both dirs (2)"]:
    for regm in ["scratch", "pretrained"]:
        r = piv.loc[(feat, regm)]
        feat_tex = feat.replace("eps", "$\\epsilon$")
        mae_v = mae.loc[(feat, regm)]
        rows.append(f"{feat_tex} & {regm} & "
                    f"{fmt(r.LOTO)} & {fmt(r.LOBO)} & {fmt(r.JOINT)} & "
                    f"{mae_v:.1f} " + "\\\\")
    rows.append("\\addlinespace")
write("tab_supp_featuresets.tex", "\n".join([
    "\\begin{tabular}{l l rrr r}", "\\toprule",
    "Features & Regime & \\multicolumn{3}{c}{ranking $\\rho$ "
    "(src/bench/both held)} & MAE \\\\", "\\midrule"]
    + rows + ["\\bottomrule", "\\end{tabular}"]))

# supp: DINO mirror (L3)
dlaw = pd.read_csv(need(V4 / "regime_direction_verification/"
                             "master_table_mean_nn_dino.csv"))
dlaw = by_regime(dlaw)
rows = []
prev = None
for _, r in dlaw.iterrows():
    if prev is not None and r.regime != prev:
        rows.append("\\midrule")
    prev = r.regime
    rows.append(f"{vtex(r.variant)} & {fmt(r.rho_ab)} & {fmt(r.rho_ba)} & "
                f"{fmt(r.d)}\\,[{r.d_lo:+.2f},\\,{r.d_hi:+.2f}] \\\\")
write("tab_supp_dino.tex", "\n".join([
    "\\begin{tabular}{l cc rr c}", "\\toprule",
    "Architecture & Pre. & Frz. & $\\rho[d_{T\\to B}]$ & $\\rho[d_{B\\to T}]$ &"
    " flip $\\Delta$ [95\\% CI] \\\\", "\\midrule"]
    + rows + ["\\bottomrule", "\\end{tabular}"]))

print("done")
