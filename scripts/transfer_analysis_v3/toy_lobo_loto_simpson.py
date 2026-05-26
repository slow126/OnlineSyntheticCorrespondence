"""Toy illustration of the LOBO-vs-LOTO tension and the Simpson's-paradox trap.

NO real data.  One small generative model produces every panel, so the figure
is internally consistent: whatever drives the Simpson reversal is the same
thing that drives the see-saw.

Generative model (i = training source, k = benchmark):

    perf(i, k) = a_i            # source quality          (orders sources)
               + b_k            # benchmark difficulty    (dominates pooled var)
               - gamma * e_ik   # WITHIN-context motion signal (the real claim)
               + noise

    motion_distance(i, k) = mu_k + e_ik
        mu_k  correlated with b_k   -> the confound that creates Simpson's paradox
        e_ik  the within-context deviation -> the only honest, claimable signal

Why each panel falls out of this one model:
  A. Simpson: pooled corr(distance, perf) is driven by (mu_k, b_k) and reverses
     the sign of the within-benchmark slope (-gamma * e_ik).
  B. The (source x benchmark) grid: LOTO hides a ROW (source), LOBO hides a
     COLUMN (benchmark).  Different things are observable in each.
  C. See-saw: a single knob theta = weight on the borrowed LEVEL vs the pure
     (i->k) feature.  LOTO falls as theta rises; LOBO climbs.  Same start, because
     at theta=0 both use the identical feature-only score.
  D. The mechanism, discretely: own (i->k) feature wins LOTO; borrowed level wins
     LOBO.

Run:
    python scripts/transfer_analysis_v3/toy_lobo_loto_simpson.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import pearsonr, spearmanr

RNG = np.random.default_rng(7)

# ----------------------------------------------------------------------------
# Generative toy
# ----------------------------------------------------------------------------
N_SRC = 9          # training datasets (sources)
N_BENCH = 7        # benchmarks (targets)

SIGMA_A = 1.8      # source-quality spread  (orders sources within a benchmark)
SIGMA_B = 2.6      # benchmark-difficulty spread (dominates the POOLED variance)
GAMMA = 0.85       # strength of the true within-context motion effect
MU_FROM_B = 1.7    # how strongly mean motion distance tracks benchmark difficulty
SIGMA_MU = 0.4     # extra spread in per-benchmark mean distance
SIGMA_NOISE = 0.30


def simulate():
    a = RNG.normal(0, SIGMA_A, N_SRC)                      # source quality
    b = RNG.normal(0, SIGMA_B, N_BENCH)                    # benchmark difficulty
    bz = (b - b.mean()) / b.std()
    mu = MU_FROM_B * bz + RNG.normal(0, SIGMA_MU, N_BENCH)  # mean dist tracks difficulty
    e = RNG.normal(0, 1.0, (N_SRC, N_BENCH))               # within-context deviation

    dist = mu[None, :] + e                                  # motion distance(i,k)
    perf = (a[:, None] + b[None, :] - GAMMA * e
            + RNG.normal(0, SIGMA_NOISE, (N_SRC, N_BENCH)))
    return dict(a=a, b=b, mu=mu, e=e, dist=dist, perf=perf)


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x - x.mean()


# ----------------------------------------------------------------------------
# Estimators, as functions of the level-weight knob theta
# ----------------------------------------------------------------------------
def loto_ctx_rho(D, theta):
    """LOTO: hold out a SOURCE (row). Benchmark seen, so we can form the
    leave-self-out context mean; but it is mechanically anti-correlated with the
    held source's own perf. Rank sources WITHIN each benchmark."""
    perf, dist = D["perf"], D["dist"]
    rhos = []
    for k in range(N_BENCH):
        y = perf[:, k]
        own = -zscore(dist[:, k])                       # closer in motion -> higher
        S = y.sum()
        loo = np.array([(S - y[i]) / (N_SRC - 1) for i in range(N_SRC)])
        level = zscore(loo)                             # anti-correlated with y by construction
        score = (1 - theta) * own + theta * level
        rhos.append(spearmanr(score, y).statistic)
    return float(np.nanmean(rhos))


def lobo_ctx_rho(D, theta):
    """LOBO: hold out a BENCHMARK (column). Sources seen elsewhere, so the
    borrowed level carries each source's quality a_i (positively useful). Rank
    sources WITHIN the held-out benchmark."""
    perf, dist = D["perf"], D["dist"]
    rhos = []
    for k in range(N_BENCH):
        y = perf[:, k]
        own = -zscore(dist[:, k])
        others = [j for j in range(N_BENCH) if j != k]
        borrow = perf[:, others].mean(axis=1)           # carries a_i across sources
        level = zscore(borrow)
        score = (1 - theta) * own + theta * level
        rhos.append(spearmanr(score, y).statistic)
    return float(np.nanmean(rhos))


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def make_figure(D, out_dir: Path):
    perf, dist, b = D["perf"], D["dist"], D["b"]
    C_LOTO, C_LOBO = "#d1495b", "#1b6ca8"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.2), constrained_layout=True)
    fig.suptitle("Why pooled correlations lie, and why one knob can't win both splits\n"
                 "(toy data: perf = source_quality + benchmark_difficulty "
                 "− γ·within-context-motion + noise)",
                 fontsize=12, fontweight="bold")

    # --- A: Simpson's paradox -------------------------------------------------
    axA = axes[0, 0]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, N_BENCH))
    within_rhos = []
    for k in range(N_BENCH):
        x, y = dist[:, k], perf[:, k]
        axA.scatter(x, y, color=cmap[k], s=34, alpha=0.85, zorder=3)
        m, c = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        axA.plot(xs, m * xs + c, color=cmap[k], lw=1.4, alpha=0.8, zorder=2)
        within_rhos.append(spearmanr(x, y).statistic)
    # pooled regression across everything
    xf, yf = dist.ravel(), perf.ravel()
    m, c = np.polyfit(xf, yf, 1)
    xs = np.array([xf.min(), xf.max()])
    axA.plot(xs, m * xs + c, color="k", lw=3.0, ls="--", zorder=4,
             label="pooled fit")
    pooled_r = pearsonr(xf, yf)[0]
    mean_within = float(np.nanmean(within_rhos))
    axA.set_title("A.  Simpson's paradox: pooled trend flips the within-benchmark sign",
                  fontsize=10, fontweight="bold", pad=8)
    axA.set_xlabel("motion distance  (i → k)")
    axA.set_ylabel("transfer performance")
    axA.legend(loc="upper left", fontsize=9, frameon=False)
    axA.text(0.97, 0.04,
             f"pooled ρ = {pooled_r:+.2f}  (looks predictive, wrong sign)\n"
             f"mean within-benchmark ρ = {mean_within:+.2f}  (the real claim)",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round", fc="#fff3cd", ec="#cc9a06"))
    axA.text(0.03, 0.55, "each color = one benchmark\n(its own difficulty level)",
             transform=axA.transAxes, fontsize=8.5, color="#444", style="italic")

    # --- B: the grid + holdouts ----------------------------------------------
    axB = axes[0, 1]
    im = axB.imshow(perf, aspect="auto", cmap="RdYlGn", origin="upper")
    axB.set_title("B.  The grid: LOTO hides a row, LOBO hides a column",
                  fontsize=10, fontweight="bold", pad=8)
    axB.set_xlabel("benchmark  k")
    axB.set_ylabel("training source  i")
    axB.set_xticks(range(N_BENCH)); axB.set_yticks(range(N_SRC))
    held_src, held_bench = 3, 4
    axB.add_patch(Rectangle((-0.5, held_src - 0.5), N_BENCH, 1, fill=False,
                            edgecolor=C_LOTO, lw=3.5))
    axB.add_patch(Rectangle((held_bench - 0.5, -0.5), 1, N_SRC, fill=False,
                            edgecolor=C_LOBO, lw=3.5))
    axB.text(N_BENCH - 0.4, held_src, "  LOTO: source unseen\n  (benchmark still seen\n   → level observable,\n   rank by OWN i→k dist)",
             color=C_LOTO, va="center", ha="left", fontsize=8.2, fontweight="bold")
    axB.text(held_bench, N_SRC - 0.3, "LOBO: benchmark unseen\n(sources seen elsewhere\n→ borrow level, which\n carries source quality)",
             color=C_LOBO, va="top", ha="center", fontsize=8.2, fontweight="bold")
    fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04, label="transfer perf")

    # --- C: the see-saw -------------------------------------------------------
    axC = axes[1, 0]
    thetas = np.linspace(0, 1, 41)
    loto = [loto_ctx_rho(D, t) for t in thetas]
    lobo = [lobo_ctx_rho(D, t) for t in thetas]
    axC.plot(thetas, loto, color=C_LOTO, lw=2.6, label="LOTO  (unseen source)")
    axC.plot(thetas, lobo, color=C_LOBO, lw=2.6, label="LOBO  (unseen benchmark)")
    axC.axhline(0, color="#999", lw=0.8, ls=":")
    axC.scatter([0, 0], [loto[0], lobo[0]], color="k", zorder=5, s=28)
    axC.annotate("same start:\nfeature-only score", (0.02, loto[0]),
                 xytext=(0.22, loto[0] - 0.28), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", color="#555"))
    axC.set_title("C.  One knob, opposite effects: leaning on the borrowed level",
                  fontsize=10, fontweight="bold", pad=8)
    axC.set_xlabel("θ  =  weight on borrowed LEVEL  (0 = pure i→k feature, 1 = pure level)")
    axC.set_ylabel("within-context ranking  ρ")
    axC.legend(loc="center right", fontsize=9, frameon=False)
    axC.text(0.5, 0.06,
             "level term is anti-correlated with held source's perf (LOTO)\n"
             "but carries source quality for the unseen benchmark (LOBO)",
             transform=axC.transAxes, ha="center", va="bottom", fontsize=8,
             color="#444", style="italic")

    # --- D: the two mechanisms, discretely -----------------------------------
    axD = axes[1, 1]
    own_loto, lvl_loto = loto_ctx_rho(D, 0.0), loto_ctx_rho(D, 1.0)
    own_lobo, lvl_lobo = lobo_ctx_rho(D, 0.0), lobo_ctx_rho(D, 1.0)
    x = np.arange(2)
    w = 0.36
    axD.bar(x - w / 2, [own_loto, own_lobo], w, label="own (i→k) feature",
            color="#6a994e")
    axD.bar(x + w / 2, [lvl_loto, lvl_lobo], w, label="borrowed level",
            color="#bc6c25")
    axD.axhline(0, color="#444", lw=0.9)
    for xi, vals in zip(x, [(own_loto, lvl_loto), (own_lobo, lvl_lobo)]):
        for dx, v in zip((-w / 2, w / 2), vals):
            axD.text(xi + dx, v + (0.02 if v >= 0 else -0.05), f"{v:+.2f}",
                     ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    axD.set_xticks(x)
    axD.set_xticklabels(["LOTO\n(unseen source)", "LOBO\n(unseen benchmark)"])
    axD.set_ylabel("within-context ranking  ρ")
    axD.set_title("D.  Which mechanism each split needs",
                  fontsize=10, fontweight="bold", pad=8)
    axD.legend(loc="upper left", fontsize=9, frameon=False)
    axD.text(0.5, -0.18,
             "feature wins LOTO; level wins LOBO  →  no single mix is best for both",
             transform=axD.transAxes, ha="center", va="top", fontsize=8.5,
             color="#444", style="italic")

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "toy_lobo_loto_simpson.png"
    pdf = out_dir / "toy_lobo_loto_simpson.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf, dict(pooled_r=pooled_r, mean_within=mean_within,
                          own_loto=own_loto, lvl_loto=lvl_loto,
                          own_lobo=own_lobo, lvl_lobo=lvl_lobo)


def main():
    D = simulate()
    out_dir = Path(__file__).resolve().parent / "results" / "toy_illustration"
    png, pdf, stats = make_figure(D, out_dir)
    print("Toy illustration written:")
    print(f"  {png}")
    print(f"  {pdf}")
    print("\nKey numbers (toy):")
    print(f"  Simpson:  pooled rho = {stats['pooled_r']:+.2f}   "
          f"mean within-benchmark rho = {stats['mean_within']:+.2f}")
    print(f"  LOTO:  own-feature rho = {stats['own_loto']:+.2f}   "
          f"borrowed-level rho = {stats['lvl_loto']:+.2f}")
    print(f"  LOBO:  own-feature rho = {stats['own_lobo']:+.2f}   "
          f"borrowed-level rho = {stats['lvl_lobo']:+.2f}")


if __name__ == "__main__":
    main()
