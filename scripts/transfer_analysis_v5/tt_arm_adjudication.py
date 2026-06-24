"""Adjudicate the TT-arm contradiction in the intervention-grid OOS test.

Context (results/intervention_oos.csv): in the pretrained (TT) arm, precision
rho NOMINALLY beats recall on all 3 benchmarks (+0.733/+0.800/+0.383 vs
+0.083/+0.650/-0.100), the opposite of the law's pretrained prediction. The
standing-but-unquantified excuse: the TT sources are matched-motion appearance
ablations whose motion distances are near-tied, so rankings on near-ties are
noise. This script quantifies that excuse.

(a) Tabulates the TT arm's 9 sources (d_tb, d_bt, transfer) per benchmark and
    prints the spread (max/min ratio, range, relative range) of each distance,
    vs the FF arm. NOTE: both arms comprise the SAME 9 sources (the 2 extra FF
    grid runs lack distance rows), so distance spreads are identical across
    arms; the meaningful contrast is per-benchmark, per-direction.
(b) Estimator noise: checks le-wm/outputs for per-seed distance files (none
    exist; only the final CSV + cached vectors), so quotes the recorded
    estimator-seed artifact results/sampling_stability.csv and quantifies
    TRANSFER-side noise instead: the EXACT permutation null of Spearman rho at
    n=9 (all 9! = 362880 permutations), null sd, and two-sided exact p for
    every observed rho.
(c) Rank stability of the distances: relative spread vs a 5% perturbation;
    number of adjacent pairs (of 8) that flip under a 5% (and 1%)
    multiplicative perturbation; Monte-Carlo (10k draws, d' = d*(1+N(0,0.05))):
    mean rank self-correlation of the perturbed vs recorded ranking, and the
    induced distribution of the headline rho. Transfer range vs the documented
    seed-noise floor (~0.4-0.9 PCK on KITTI, per task brief; calibration memory
    cites +/-1-2 PCK as the loose bound).
(d) Same quantities for the FF arm's flyingthings row (headline +0.67) so the
    paper can state WHY that cell is interpretable and the TT KITTI cells are
    not.

Inputs (same as intervention_oos_test.py):
  grid snapshots:  /mnt/nvme_1tb_a/snapshots/transfer_grid/*/validation_results.csv
  distances:       le-wm/outputs/intervention_motion_distances_directional.csv

Outputs (NEW files only):
  results/tt_arm_adjudication.csv          (per arm x benchmark x direction metrics)
  results/tt_arm_adjudication_sources.csv  (raw 9-source tabulation)
  results/tt_arm_adjudication_summary.txt

    nice -n 15 /home/spencer/miniconda3/envs/cuda/bin/python \
        scripts/transfer_analysis_v5/tt_arm_adjudication.py
"""
from __future__ import annotations

import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

GRID = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")
DIST = Path("/home/spencer/Projects/le-wm/outputs/"
            "intervention_motion_distances_directional.csv")
LEWM_OUT = Path("/home/spencer/Projects/le-wm/outputs")
RES = Path(__file__).parent / "results"
SAMPLING = RES / "sampling_stability.csv"

OUT_CSV = RES / "tt_arm_adjudication.csv"
OUT_SRC = RES / "tt_arm_adjudication_sources.csv"
OUT_TXT = RES / "tt_arm_adjudication_summary.txt"

DROP_BENCH = {"middlebury"}  # eval bug confirmed 2026-06-10; excluded everywhere
NOISE_FLOOR_LO, NOISE_FLOOR_HI = 0.4, 0.9  # documented KITTI seed-noise floor (PCK)
N_MC = 10_000
MC_SIGMA = 0.05
RNG = np.random.default_rng(0)

DIRS = {"precision": "flow_mean_nn_a_to_b", "recall": "flow_mean_nn_b_to_a"}
LAW = {"FF": "precision", "TT": "recall"}


def harvest():
    """Same harvest as intervention_oos_test.py (peak pck, finished runs only)."""
    rows = []
    for d in sorted(GRID.iterdir()):
        f = d / "validation_results.csv"
        if not f.exists():
            continue
        v = pd.read_csv(f)
        if v["epoch"].nunique() < 50:
            continue
        src = d.name.rsplit("_pt", 1)[0]
        arm = "FF" if "_pt0_fz0" in d.name else "TT"
        for b, g in v.groupby("benchmark"):
            rows.append((src, arm, b, float(g["pck"].max())))
    return pd.DataFrame(rows, columns=["source", "arm", "benchmark", "peak_pck"])


def ranks(x):
    """Average ranks (matches scipy spearman on ties)."""
    return pd.Series(x).rank().to_numpy()


def exact_null(n=9):
    """Exact permutation distribution of Spearman rho at n (no ties)."""
    base = np.arange(1, n + 1, dtype=np.float64)
    c = base - base.mean()
    den = float(c @ c)
    perms = np.array(list(itertools.permutations(range(n))), dtype=np.int8)
    null = (c[perms] @ c) / den
    return null


def exact_p(null, obs):
    return float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def adj_flips(d, eps):
    """Adjacent pairs (of n-1) whose sorted ratio <= 1+eps (a single eps-sized
    multiplicative perturbation can swap them)."""
    s = np.sort(np.asarray(d, dtype=float))
    r = s[1:] / s[:-1]
    return int(np.sum(r <= 1.0 + eps)), [float(x) for x in r]


def mc_perturb(d, pck, sigma=MC_SIGMA, n_mc=N_MC, rng=RNG):
    """d' = d*(1+N(0,sigma)). Returns (rank self-corr mean, rho' mean, rho' sd,
    P(rho' <= 0)) where rho' = spearman(pck, -d')."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dp = d[None, :] * (1.0 + rng.normal(0.0, sigma, size=(n_mc, n)))
    # ranks per row (ties measure-zero under continuous noise)
    idx = np.argsort(dp, axis=1)
    rd = np.empty_like(dp)
    np.put_along_axis(rd, idx, np.arange(1, n + 1, dtype=float)[None, :]
                      .repeat(n_mc, 0), axis=1)
    rd_c = rd - rd.mean(axis=1, keepdims=True)
    rd_norm = np.sqrt((rd_c ** 2).sum(axis=1))  # = sqrt(60) for n=9, no ties

    r0 = ranks(d); r0_c = r0 - r0.mean(); r0_n = math.sqrt(float(r0_c @ r0_c))
    selfcorr = (rd_c @ r0_c) / (rd_norm * r0_n)

    rp = ranks(pck); rp_c = rp - rp.mean(); rp_n = math.sqrt(float(rp_c @ rp_c))
    rho = (rd_c @ rp_c) / (rd_norm * rp_n) * -1.0  # spearman(pck, -d')
    return (float(selfcorr.mean()), float(rho.mean()), float(rho.std()),
            float(np.mean(rho <= 0.0)))


def main():
    RES.mkdir(parents=True, exist_ok=True)
    pck = harvest()
    pck = pck[~pck.benchmark.isin(DROP_BENCH)]
    dist = pd.read_csv(DIST)
    m = pck.merge(dist, on=["source", "benchmark"], how="inner")

    # ---- per-seed distance file check (part b, estimator side) -------------
    seed_files = sorted(p.name for p in LEWM_OUT.glob("*seed*")
                        if p.suffix in {".csv", ".npy"})
    sampling = pd.read_csv(SAMPLING) if SAMPLING.exists() else None

    # ---- raw source tabulation (part a) ------------------------------------
    src_tab = (m.pivot_table(index=["benchmark", "source",
                                    "flow_mean_nn_a_to_b", "flow_mean_nn_b_to_a"],
                             columns="arm", values="peak_pck")
               .reset_index()
               .rename(columns={"FF": "peak_pck_FF", "TT": "peak_pck_TT",
                                "flow_mean_nn_a_to_b": "d_tb",
                                "flow_mean_nn_b_to_a": "d_bt"}))
    src_tab = src_tab.sort_values(["benchmark", "d_tb"]).reset_index(drop=True)
    src_tab.to_csv(OUT_SRC, index=False)

    # ---- exact permutation null at n=9 (part b, transfer side) -------------
    null = exact_null(9)
    null_sd = float(null.std())

    recs = []
    for (arm, b), g in m.groupby(["arm", "benchmark"]):
        g = g.sort_values("source")
        assert g.source.nunique() == len(g)
        p_vals = g.peak_pck.to_numpy()
        t_range = float(p_vals.max() - p_vals.min())
        for dname, col in DIRS.items():
            d = g[col].to_numpy()
            assert len(np.unique(d)) == len(d), f"tied distances {arm}/{b}/{dname}"
            rho = float(spearmanr(g.peak_pck, -g[col]).statistic)
            f5, _ = adj_flips(d, 0.05)
            f1, _ = adj_flips(d, 0.01)
            sc, rmu, rsd, ple0 = mc_perturb(d, p_vals)
            recs.append(dict(
                arm=arm, benchmark=b, direction=dname,
                law_relevant=(LAW[arm] == dname), n=len(g),
                rho=rho, p_exact_2sided=exact_p(null, rho),
                d_min=float(d.min()), d_max=float(d.max()),
                spread_ratio=float(d.max() / d.min()),
                spread_range=float(d.max() - d.min()),
                rel_spread_pct=float(100 * (d.max() - d.min()) / np.median(d)),
                adj_flips_5pct_of_8=f5, adj_flips_1pct_of_8=f1,
                mc5_rank_selfcorr=sc, mc5_rho_mean=rmu, mc5_rho_sd=rsd,
                mc5_p_rho_le_0=ple0,
                transfer_range_pck=t_range,
                transfer_range_over_floor_hi=t_range / NOISE_FLOOR_HI,
                transfer_range_over_floor_lo=t_range / NOISE_FLOOR_LO,
            ))
    df = pd.DataFrame(recs).sort_values(["arm", "benchmark", "direction"],
                                        ascending=[True, True, False])
    df.to_csv(OUT_CSV, index=False)

    # ---- summary ------------------------------------------------------------
    L = []
    P = L.append
    P("TT-ARM ADJUDICATION — intervention grid OOS (generated by "
      "tt_arm_adjudication.py)")
    P(f"distances: {DIST}")
    P(f"grid harvest: {GRID} (peak pck, epoch-complete runs, middlebury "
      f"excluded)")
    P("")
    P("(a) RAW TABULATION — the same 9 sources back BOTH arms (the 2 extra FF "
      "runs,")
    P("    kitti_badmotion_ft_gso_matte and synthetic_fractal_trial76, have no "
      "distance")
    P("    rows), so FF and TT distance spreads are IDENTICAL by construction; "
      "the")
    P("    contrast is per-benchmark, per-direction.")
    for b, g in src_tab.groupby("benchmark"):
        P(f"\n  benchmark={b} (sorted by d_tb):")
        P(g.to_string(index=False,
                      float_format=lambda x: f"{x:.6f}" if abs(x) < 1 else f"{x:.2f}"))
    P("")
    P("  DISTANCE SPREADS (max/min ratio | range | range as % of median):")
    for _, r in df[df.arm == "TT"].iterrows():
        P(f"    {r.benchmark:12s} {r.direction:9s}: ratio {r.spread_ratio:.4f} | "
          f"range {r.spread_range:.6f} | rel {r.rel_spread_pct:.2f}%"
          f"{'   <- law-relevant for TT' if r.law_relevant else ''}")
    P("  (FF rows identical — same sources.)")
    P("")
    P("(b) ESTIMATOR NOISE — per-seed distance files in le-wm/outputs: "
      f"{seed_files if seed_files else 'NONE (only the final CSV + cached vectors)'}")
    if sampling is not None:
        P("    Recorded estimator-seed artifact (results/sampling_stability.csv,")
        P("    rank-corr of the source ranking across 40k-subsample seeds, "
          "1-seed vs 5-seed-avg / split-half):")
        for _, r in sampling.iterrows():
            nm = "precision(dP)" if r.direction == "dP" else "recall(dR)"
            P(f"      {nm}: single-seed mean {r.single_seed_mean:.3f} "
              f"min {r.single_seed_min:.3f} | split-half mean "
              f"{r.splithalf_mean:.3f} min {r.splithalf_min:.3f}")
    P("")
    P(f"    TRANSFER-side null: EXACT permutation distribution of Spearman at "
      f"n=9 (9!={math.factorial(9)} perms): sd = {null_sd:.4f} "
      f"(analytic 1/sqrt(8) = {1/math.sqrt(8):.4f})")
    P("    Two-sided exact p for every observed rho:")
    for _, r in df.iterrows():
        P(f"      {r.arm} {r.benchmark:12s} {r.direction:9s}: rho {r.rho:+.3f}  "
          f"p = {r.p_exact_2sided:.4f}"
          f"{'   <- law-relevant' if r.law_relevant else ''}")
    P("")
    P("(c) RANK STABILITY OF THE DISTANCES vs REALITY OF THE TRANSFER "
      "DIFFERENCES")
    P("    flips = adjacent sorted pairs (of 8) swappable by a single 5% (1%) ")
    P("    multiplicative perturbation; MC = 10k draws d'=d*(1+N(0,0.05)):")
    for _, r in df[df.arm == "TT"].iterrows():
        P(f"    {r.benchmark:12s} {r.direction:9s}: flips@5% {r.adj_flips_5pct_of_8}/8 "
          f"(@1% {r.adj_flips_1pct_of_8}/8) | MC rank self-corr "
          f"{r.mc5_rank_selfcorr:.3f} | TT rho under 5% noise: "
          f"{r.mc5_rho_mean:+.3f} +/- {r.mc5_rho_sd:.3f}, P(rho<=0) "
          f"{r.mc5_p_rho_le_0:.3f}")
    P("")
    P(f"    Transfer spread vs documented seed-noise floor "
      f"({NOISE_FLOOR_LO}-{NOISE_FLOOR_HI} PCK on KITTI, task brief; "
      f"calibration memory: +/-1-2 PCK):")
    for (arm, b), g in df.groupby(["arm", "benchmark"]):
        r = g.iloc[0]
        P(f"      {arm} {b:12s}: transfer range {r.transfer_range_pck:.2f} PCK "
          f"= {r.transfer_range_over_floor_hi:.1f}x the 0.9 floor "
          f"({r.transfer_range_over_floor_lo:.1f}x the 0.4 floor)")
    P("")
    P("(d) FF flyingthings (the headline) vs TT KITTI cells:")
    ffft = df[(df.arm == "FF") & (df.benchmark == "flyingthings")
              & (df.direction == "precision")].iloc[0]
    P(f"    FF flyingthings precision: rho {ffft.rho:+.3f} (exact p "
      f"{ffft.p_exact_2sided:.4f}), spread ratio {ffft.spread_ratio:.2f} "
      f"(rel {ffft.rel_spread_pct:.1f}%), flips@5% "
      f"{ffft.adj_flips_5pct_of_8}/8, MC rank self-corr "
      f"{ffft.mc5_rank_selfcorr:.3f}, rho under 5% noise {ffft.mc5_rho_mean:+.3f} "
      f"+/- {ffft.mc5_rho_sd:.3f}")
    for b in ["kitti2012", "kitti2015"]:
        r = df[(df.arm == "TT") & (df.benchmark == b)
               & (df.direction == "precision")].iloc[0]
        P(f"    TT {b} precision:      rho {r.rho:+.3f} (exact p "
          f"{r.p_exact_2sided:.4f}), spread ratio {r.spread_ratio:.4f} "
          f"(rel {r.rel_spread_pct:.1f}%), flips@5% {r.adj_flips_5pct_of_8}/8, "
          f"MC rank self-corr {r.mc5_rank_selfcorr:.3f}, rho under 5% noise "
          f"{r.mc5_rho_mean:+.3f} +/- {r.mc5_rho_sd:.3f}")
    ttft = df[(df.arm == "TT") & (df.benchmark == "flyingthings")
              & (df.direction == "precision")].iloc[0]
    P(f"    TT flyingthings precision: rho {ttft.rho:+.3f} (exact p "
      f"{ttft.p_exact_2sided:.4f}), spread ratio {ttft.spread_ratio:.2f} "
      f"(rel {ttft.rel_spread_pct:.1f}%), flips@5% "
      f"{ttft.adj_flips_5pct_of_8}/8, MC rank self-corr "
      f"{ttft.mc5_rank_selfcorr:.3f} — NOT excused by near-ties; flag "
      f"separately.")
    P("")
    txt = "\n".join(L)
    OUT_TXT.write_text(txt + "\n")
    print(txt)
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_SRC}\nwrote {OUT_TXT}")


if __name__ == "__main__":
    main()
