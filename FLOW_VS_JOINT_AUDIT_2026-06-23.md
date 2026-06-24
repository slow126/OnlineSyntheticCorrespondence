# Flow-only vs Joint audit — ACCV 2026 coverage metric

**Date:** 2026-06-23
**Trigger:** Handoff claimed the headline coverage metric is *flow-only* `[dx,dy]`, not the
joint `[x,y,dx,dy]` we believed. This audit verifies which space every coverage
number/figure/table actually uses, and produces the flow-only-vs-joint comparison.

## BOTTOM LINE

**Everything in the paper is computed in the JOINT 4D space `[x, y, dx, dy]` (α=1.0).
The handoff is wrong.** The confusion is a naming trap: a CSV column / variable labelled
`"flow"` (or `flow_mean_nn_*`, or `f_flow`) means *the flow-derived representation*, which
the pipeline builds with `to_joint_space(alpha=1.0)` — **not** `to_flow_space` `[dx,dy]`.

The paper text already describes the metric correctly as 4D
(`03_setup.tex:80`: *"a normalized four-dimensional space of image position and
displacement, f_flow = [x̂, ŷ, Δx̂, Δŷ] ∈ [−1,1]⁴"*; `03_setup.tex:9`: *"joint flow
vectors"*). And the supplement (`supp_main.tex:487`) already discloses the flow-only
decomposition. **No numbers need to change.**

## PROOF (exact reproduction from raw caches)

The generator [compute_pairwise_self_distances.py:131-142](scripts/transfer_analysis_v3/compute_pairwise_self_distances.py#L131)
applies `spaces.to_joint_space(vecs, alpha=1.0)` to the `"flow"` space (docstring line 16:
*"Both flow (joint 4D, normalized) and DINO..."*).

Reproducing `movi_f → kitti2015` `mean_nn_b_to_a` from the raw caches:

| space | recomputed | CSV value (`pairwise_self_distances`, space=flow) | match |
|---|---|---|---|
| **JOINT** `[x,y,dx,dy]` | **0.000718** | 0.000718 | **1.00×** |
| flow-only `[dx,dy]` | 0.000021 | 0.000718 | 0.03× |

The CSV value the handoff called "flow-only" reproduces **exactly** in JOINT (1.00×).
The handoff's "table≈0.0007 ≈ flow-only(0.0018) not joint(0.034)" anchor was a botched
recompute (wrong direction/normalization).

## WHERE EACH ARTIFACT LIVES (all JOINT)

| Artifact | Source file | Space | Evidence |
|---|---|---|---|
| Table 1 / `tab:law` (coverage law +0.73) | `make_stratified_law_tables.py` → `pairwise_self_distances.csv` (space=flow) | **JOINT** | generator applies `to_joint_space`; movi_f anchor reproduces 1.00× |
| Fig 2 (coverage-law scatter) | `make_fig_coverage_law.py` → `transfer_table_nomid.csv` `flow_mean_nn_*` | **JOINT** | identical values to `pairwise_self_distances` space=flow |
| Fig 8 (magnitude ladder) | `make_ladder_fig.py` → `coverage_v2_flow_ladder.csv` (space=joint) | **JOINT** | CSV is joint-only; uses support-overlap estimator |
| §7 interventions distances | le-wm `intervention_motion_distances_directional.csv` | **JOINT** | `kitti_recovered_hq→kitti2015` 0.0065 ≈ joint recompute 0.0055, not flow-only 0.0002 |
| §7(iv) "dBT flat ≈0.0017" | same CSV, `kitti_recovered_hq→kitti2012` | **JOINT** | CSV joint value = 0.001756 ≈ 0.0017 |
| Supp. decomposition | `bfv_spatial_vs_flow_probe.py` | full(JOINT)/xy/flow-only | the only place flow-only is reported, by design |

## THE FLOW-ONLY vs JOINT COMPARISON (what was asked for)

### Table 1 (within-context Spearman of peak PCK vs −distance), by sub-space
From `bfv_spatial_vs_flow_probe.py` (120k subsample, ranking-stable, pooled over archs):

**Coverage `dBT` (the headline predictor):**
| regime | stratum | full = **JOINT (paper)** | spatial-only `[x,y]` | flow-only `[dx,dy]` |
|---|---|---|---|---|
| scratch | real-motion | **+0.60** | +0.20 | +0.14 |
| scratch | semantic | −0.15 | +0.36 | −0.19 |
| pretrained | real-motion | **+0.72** | +0.25 | +0.25 |
| pretrained | semantic | +0.47 | +0.48 | +0.40 |

The headline "+0.73 on real motion" = joint dBT pretrained-real = **+0.72**. In flow-only it
**collapses to +0.25** (and +0.14 scratch). Position and motion *together* carry the signal;
neither marginal alone reproduces it. → **JOINT is the correct, stronger choice and is what
the paper uses.** (Supplement quotes flow-only real-motion as +0.18/+0.26 — same story,
small diff from FlowFormer inclusion / subsample.)

### Fig 8 magnitude ladder (kitti2015 target), dBT = coverage distance (smaller = better)
From `.audit_scratch/ladder_space_compare.py` (joint recompute matches the paper CSV exactly):

| rung | flow-only dBT | **JOINT dBT (paper)** |
|---|---|---|
| 0.25× | 0.0072 | 0.0103 |
| 0.5× | 0.0047 | 0.0093 |
| 1× | 0.0021 | 0.0082 |
| 1.5× | 0.0011 | 0.0081 |
| 2× | 0.0006 | 0.0089 |

- **flow-only:** monotone — "bigger motion is always better", and off-target dTB ≈ 0 (invisible).
  This is the *misleading* reading.
- **JOINT (paper):** inverted-U — overshoot at 2× **hurts** coverage (min at ~1.5×). This is
  what matches the empirical inverted-U transfer PCK (2× trains *worse*).

`make_ladder_fig.py` docstring already says it switched to the joint support-overlap estimator
*"NOT the scale-normalized distance the old panel used, which falsely fell monotonically"* —
i.e. the author **already caught and fixed** the exact flow-only artifact the handoff feared.
(The `mot` column in `ladder_master_table.csv` is leftover flow-only and is **not** used by the
figure.)

## RECOMMENDATION

1. **No numbers change.** The paper is internally consistent: Table 1, Fig 2, Fig 8, and §7
   interventions are all joint.
2. The handoff's premise ("Fig 8 is the lone joint artifact; everything else is flow-only") is
   false — invert it: *everything* is joint; flow-only appears only in the supplement decomposition.
3. Optional clarity (not required): the paper could state once that "BFV / f_flow" is the 4D
   joint space, to prevent exactly this re-derivation. §3 already says "four-dimensional" and
   Fig 3 says "joint flow vectors", so this is minor.
4. Optional hygiene: drop/relabel the unused flow-only `mot` column in `ladder_master_table.csv`
   so no future reader wires it into a figure.

## Scratch artifacts
- `.audit_scratch/verify_space.py` — movi_f exact-reproduction test
- `.audit_scratch/compare_spaces_full.py` — full 16M flow/joint recompute (killed; redundant with probe)
- `.audit_scratch/ladder_space_compare.py` + `.csv` — Fig 8 flow vs joint
- `scripts/transfer_analysis_v5/results/bfv_spatial_vs_flow_distances.csv` — Table 1 full/xy/flow distances
