"""Density vs spatial-alignment vs motion-matching for SEMANTIC-from-scratch transfer.

Question (Spencer, 2026-06-12): does semantic-from-scratch transfer just like
DENSER LABELS (more supervised pixels -> stronger gradients / more
regularization for an encoder that has to learn features from nothing), rather
than motion *matching* or spatial *alignment*? And if we control / hold out
density, does motion or spatial still hold?

What the pre-probe showed and why this design:
  * SPATIAL coverage is SATURATED: every dense-flow source fills image space
    (xy-entropy 6.65..6.93 of max 6.93), so benchmark->source xy distance is
    ~tied across the candidate sources (dTB_xy max/min 1.4-1.7). Spatial
    *alignment* is therefore NOT a graded axis you could select on -> drop it
    as a "signal", keep it as a saturation control.
  * What VARIES is MOTION density: moving-fraction 39%..100%, flow-entropy
    1.7..6.0, and motion distances vary 35-540x. So "denser labels" is
    operationalized as effective MOTION supervision, not spatial spread.

Three explanations, now separable on the axis that actually varies:
  (D) source-only motion RICHNESS/density: moving_frac, flow_entropy, mean_mag,
      log_n_raw  -- benchmark-independent ("just more/denser gradient signal").
  (M) benchmark-specific motion MATCHING: dBT_flow (coverage), dTB_flow.
  (S) spatial alignment: dBT_xy / dTB_xy -- reported as saturation control only.

Controls:
  (1) STATISTICAL: partial-spearman -- does motion MATCHING (M) survive after
      removing source RICHNESS (D), and vice-versa, per regime x stratum?
  (2) PHYSICAL hold-out: histogram-match every source's |flow| magnitude to a
      common reference (equalizes motion density/amount), recompute dBT_flow,
      re-correlate. If real-motion matching survives magnitude-equalization it
      is about motion STRUCTURE, not amount; if semantic stays at floor it was
      never motion.

Predicted dissociation (the interesting outcome): on REAL-MOTION, matching (M)
predicts and survives controlling for richness; on SEMANTIC-from-scratch,
matching is at the floor and only source RICHNESS (D) predicts -- i.e. the
from-scratch encoder, unable to use motion it can't yet represent, benefits
only from generically richer supervision. Under a PRETRAINED backbone, semantic
matching (M) returns.

Outputs (scripts/transfer_analysis_v5/results/):
  bfv_density_source_features.csv
  bfv_density_matched_distances.csv      (magnitude-equalized dBT_flow)
  + printed report
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, rankdata

ROOT = "/home/spencer/Projects/OnlineSyntheticCorrespondence"
sys.path.insert(0, ROOT)
from scripts.coverage import spaces

VEC = "/mnt/nvme_1tb_b/coverage_vectors"
IMG = 512
N_SUB = 120_000
SEED = 0
WORKERS = 4                 # polite to the kubric render
MOVE_THR = 0.01             # |flow|_norm threshold for "moving" (>~2.5px)

PURE = ["flyingthings","imagenet2dwarp","movi_f","pointodyssey","sintel","spair",
        "synthetic","synthetic_2d_warp","synthetic_large_zoom",
        "synthetic_random_flipping","synthetic_small_zoom"]
BENCH = {"flyingthings":"test","kitti2012":"val","kitti2015":"val",
         "pointodyssey":"test","synthetic":"val",
         "spair":"test","pfpascal":"test","pfwillow":"test","tss":"val"}
REAL = ["kitti2012","kitti2015","flyingthings","pointodyssey","synthetic"]
SEM  = ["spair","pfpascal","pfwillow","tss"]
RES = os.path.join(ROOT, "scripts/transfer_analysis_v5/results")


# ----------------------------------------------------------------------------- loading
def load_raw(name, split):
    p = os.path.join(VEC, f"{name}_{split}_flow.npy")
    if not os.path.exists(p):
        return None, 0
    a = np.load(p, mmap_mode="r")
    nraw = len(a)
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(nraw, min(N_SUB, nraw), replace=False))
    v = spaces.normalize_flow_vectors(np.asarray(a[idx], dtype=np.float32), IMG, IMG)
    return v, nraw


def directed_mean_nn(a, b):
    d, _ = cKDTree(b).query(a, k=1, workers=WORKERS)
    return float(np.mean(d))


# ----------------------------------------------------------------------------- source features
def source_features(v, nraw):
    mag = np.hypot(v[:, 2], v[:, 3])
    # spatial coverage (saturation control)
    Hxy, _, _ = np.histogram2d(v[:, 0], v[:, 1], bins=32, range=[[-1, 1], [-1, 1]])
    pxy = Hxy[Hxy > 0] / Hxy.sum(); xy_ent = float(-(pxy * np.log(pxy)).sum())
    # motion richness / density
    Hf, _, _ = np.histogram2d(v[:, 2], v[:, 3], bins=48)
    pf = Hf[Hf > 0] / Hf.sum(); flow_ent = float(-(pf * np.log(pf)).sum())
    return dict(xy_entropy=xy_ent, flow_entropy=flow_ent,
                moving_frac=float((mag > MOVE_THR).mean()),
                mean_mag=float(mag.mean()), median_mag=float(np.median(mag)),
                log_n_raw=float(np.log10(nraw)))


# ----------------------------------------------------------------------------- magnitude equalization (physical control)
def magnitude_matched(v, ref_edges, ref_hist, rng):
    """Importance-resample v so its |flow| histogram matches a common reference.
    Equalizes MOTION DENSITY/AMOUNT across sources; structure/direction is kept."""
    mag = np.hypot(v[:, 2], v[:, 3])
    b = np.clip(np.digitize(mag, ref_edges) - 1, 0, len(ref_hist) - 1)
    src_hist = np.bincount(b, minlength=len(ref_hist)).astype(float)
    src_hist[src_hist == 0] = np.inf                     # never sample empty bins
    w = ref_hist[b] / src_hist[b]                        # target / source density
    w = w / w.sum()
    idx = rng.choice(len(v), size=len(v), replace=True, p=w)
    return v[idx]


# ----------------------------------------------------------------------------- stats helpers
def partial_spearman(y, x, z):
    """spearman(y, x | z) via rank-residualization."""
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    Z = np.c_[np.ones_like(rz), rz]
    def res(a):
        c, *_ = np.linalg.lstsq(Z, a, rcond=None); return a - Z @ c
    ey, ex = res(ry), res(rx)
    if ey.std() < 1e-9 or ex.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


def load_peak():
    t0 = pd.read_csv(os.path.join(ROOT, "scripts/transfer_analysis_v3/transfer_table.csv"))
    t = t0[t0.train_dataset.isin(PURE)].copy()
    def reg(r):
        if r.model_family == "raft": return "FF"
        if r.pretrained == False and r.freeze == False: return "FF"
        if r.pretrained == True and r.freeze == True: return "TT"
        return None
    t["regime"] = t.apply(reg, axis=1)
    return t[t.regime.notna()][["model_family","regime","train_dataset","benchmark","peak_pck"]]


# pooled-across-arch mean of per-(benchmark) within-source correlations
def pooled_cell(P, reg, bset, valcol, signed=-1.0, control=None):
    vals = []
    for arch, ga in P[P.regime == reg].groupby("model_family"):
        rs = []
        for b, g in ga[ga.benchmark.isin(bset)].groupby("benchmark"):
            g = g.dropna(subset=["peak_pck", valcol] + ([control] if control else []))
            if g.train_dataset.nunique() < 4 or g[valcol].std() < 1e-12:
                continue
            x = signed * g[valcol].values
            if control:
                if g[control].std() < 1e-12:
                    continue
                r = partial_spearman(g.peak_pck.values, x, g[control].values)
            else:
                r = spearmanr(g.peak_pck.values, x).statistic
            if np.isfinite(r):
                rs.append(r)
        if rs:
            vals.append(np.mean(rs))
    return np.mean(vals) if vals else np.nan


def main():
    os.makedirs(RES, exist_ok=True)
    # load all clouds once + source features
    clouds, feats = {}, {}
    for s in PURE:
        v, nraw = load_raw(s, "train")
        clouds[("src", s)] = v
        if v is not None:
            feats[s] = source_features(v, nraw)
    for b in BENCH:
        clouds[("bench", b)], _ = load_raw(b, BENCH[b])
    F = pd.DataFrame(feats).T.reset_index().rename(columns={"index": "source"})
    F.to_csv(os.path.join(RES, "bfv_density_source_features.csv"), index=False)
    print("=== SOURCE FEATURES (xy_entropy is the saturation control) ===")
    print(F.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # distances: reuse the probe CSV if present, else recompute full/xy/flow
    probe = os.path.join(RES, "bfv_spatial_vs_flow_distances.csv")
    D = pd.read_csv(probe)

    # ---- physical control: magnitude-matched dBT_flow ----
    # common reference = pooled |flow| histogram over all sources
    allmag = np.concatenate([np.hypot(clouds[("src", s)][:, 2], clouds[("src", s)][:, 3])
                             for s in PURE])
    ref_edges = np.quantile(allmag, np.linspace(0, 1, 41))   # 40 adaptive bins
    ref_edges[0], ref_edges[-1] = -np.inf, np.inf
    ref_hist = np.full(len(ref_edges) - 1, 1.0 / (len(ref_edges) - 1))  # uniform target
    rng = np.random.default_rng(SEED)
    eq_src = {s: magnitude_matched(clouds[("src", s)], ref_edges, ref_hist, rng) for s in PURE}
    rows = []
    for s in PURE:
        A = eq_src[s][:, [2, 3]]
        for b in BENCH:
            B = clouds[("bench", b)][:, [2, 3]]
            rows.append(dict(source=s, benchmark=b,
                             dBT_flow_eq=directed_mean_nn(B, A),
                             dTB_flow_eq=directed_mean_nn(A, B)))
        print(f"  mag-matched {s:>28s} done", flush=True)
    DEQ = pd.DataFrame(rows)
    DEQ.to_csv(os.path.join(RES, "bfv_density_matched_distances.csv"), index=False)

    # ---- assemble + analyze ----
    peak = load_peak()
    P = (peak.merge(D, left_on=["train_dataset","benchmark"], right_on=["source","benchmark"])
              .merge(DEQ, on=["source","benchmark"])
              .merge(F, on="source"))

    RICH = "flow_entropy"     # primary source-only richness proxy (also report moving_frac)
    def block(title, reg):
        print("\n" + "=" * 88); print(title); print("=" * 88)
        hdr = (f"{'stratum':>11s} | {'rich:flent':>10s} {'rich:mov%':>9s} {'rich:logN':>9s} "
               f"| {'M:dBTflow':>9s} {'M|rich':>8s} {'rich|M':>8s} | {'M_eq':>7s} {'xy(sat)':>8s}")
        print(hdr); print("-" * len(hdr))
        for sname, bset in [("real-motion", REAL), ("semantic", SEM)]:
            c = dict(
                rich_fe = pooled_cell(P, reg, bset, "flow_entropy", signed=+1.0),
                rich_mv = pooled_cell(P, reg, bset, "moving_frac",  signed=+1.0),
                rich_n  = pooled_cell(P, reg, bset, "log_n_raw",    signed=+1.0),
                M       = pooled_cell(P, reg, bset, "dBT_flow",     signed=-1.0),
                M_pr    = pooled_cell(P, reg, bset, "dBT_flow",     signed=-1.0, control=RICH),
                R_pr    = pooled_cell(P, reg, bset, RICH,           signed=+1.0, control="dBT_flow"),
                M_eq    = pooled_cell(P, reg, bset, "dBT_flow_eq",  signed=-1.0),
                xy      = pooled_cell(P, reg, bset, "dBT_xy",       signed=-1.0),
            )
            f = lambda v: f"{v:>+.2f}" if np.isfinite(v) else "  -- "
            print(f"{sname:>11s} | {f(c['rich_fe']):>10s} {f(c['rich_mv']):>9s} {f(c['rich_n']):>9s} "
                  f"| {f(c['M']):>9s} {f(c['M_pr']):>8s} {f(c['R_pr']):>8s} "
                  f"| {f(c['M_eq']):>7s} {f(c['xy']):>8s}")

    print("\nLegend: rich:* = source-only motion-RICHNESS -> transfer (benchmark-independent).")
    print("  M:dBTflow = motion MATCHING (coverage). M|rich = matching after partialling richness.")
    print("  rich|M = richness after partialling matching. M_eq = matching on |flow|-magnitude-")
    print("  equalized sources (density held out). xy(sat) = spatial-alignment (saturated control).")
    block("REGIME = SCRATCH (from-scratch encoder)", "FF")
    block("REGIME = PRETRAINED (frozen backbone)", "TT")
    print("\nREAD: if on SEMANTIC/scratch only rich:* is +ve while M and M|rich ~0, and on")
    print("REAL-MOTION M survives M|rich and M_eq -> from-scratch-semantic = generic label")
    print("density (gradient richness), real-motion = genuine motion matching.")


if __name__ == "__main__":
    main()
