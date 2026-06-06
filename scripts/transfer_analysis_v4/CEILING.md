# Predictability Ceiling

Replicate dimension: **variant** (6 levels). Source: `scripts/transfer_analysis_v4/results/predictions/peak_pck/rows_LOTO_motion.csv`.

**Ceiling = how reproducibly model variants agree on the source ranking within a benchmark.** Motion can't beat what the target itself doesn't reproduce. `frac_of_ceiling = motion_rho / ceiling`.

| benchmark | ceiling ρ | motion ρ | motion / ceiling | n_reps |
|---|---|---|---|---|
| kitti2012 | +0.844 | +0.827 | 98% | 5 |
| kitti2015 | +0.759 | +0.709 | 93% | 5 |
| synthetic | +0.667 | +0.518 | 78% | 5 |
| flyingthings | +0.634 | +0.455 | 72% | 5 |
| pfpascal | +0.628 | +0.364 | 58% | 5 |
| pfwillow | +0.606 | +0.100 | 17% | 5 |
| middlebury | +0.597 | +0.645 | 108% | 5 |
| tss | +0.517 | +0.100 | 19% | 5 |
| pointodyssey | +0.415 | +0.855 | 206% | 5 |
| spair | +0.174 | -0.036 | -21% | 5 |
| **POOLED** | **+0.584** | **+0.454** | **78%** | 6 |

**Source main-effect share:** 32.8% of within-cell variance is generic source quality (ceiling for any *source-level* feature); the rest is dyadic interaction + replicate noise.

> [!warning] Current variants are correlated (CATs++ toggles + RAFT), so the ceiling is an optimistic UPPER bound and motion's fraction a conservative FLOOR. Re-run with GLU-Net / FlowFormer++ / PWCNet (they enter as new `variant` rows automatically) for the unbiased number.
