# GLU-Net Modeling — what adding it did to the transfer-prediction analysis

**Written 2026-06-09.** Handoff for a fresh agent. Context: we folded a 3rd architecture (GLU-Net) into the transfer-prediction pipeline (`scripts/transfer_analysis_v4`). It was supposed to *strengthen* the cross-architecture generality + predictability-ceiling story. Instead it **weakened the headline ρ_g for every architecture** and unsettled the ceiling/consensus claim. This doc captures exactly what we found, the numbers, the mechanism hypothesis, what's affected vs not, and the open decision.

---

## TL;DR
- The predictor headline is `ctx_rho_g` = within-context Spearman(actual transfer, `g`), where `g` is a **single shared RidgeCV** fit on within-context-demeaned motion features, pooled across **all contexts of all architectures**.
- Adding GLU-Net's 40 contexts (4 variants × 10 benchmarks) to that shared pool **dropped motion ρ_g for CATs++ and RAFT too**, by ~0.04–0.22 — not just added a weak GLU-Net row.
- Cause hypothesis: **GLU-Net's transfer responds to motion features with a different functional form**, so a *single shared ridge* across all 3 architectures is misspecified; the compromise fit hurts every architecture.
- Motion still beats appearance everywhere, but the headline is materially weaker pooled, and this interacts with the **cross-architecture consensus / predictability-ceiling** claim (the "disagreement").
- **Open decision (not yet made):** per-architecture `g` vs pooled `g` vs CATs++/RAFT-headline-with-GLU-Net-as-a-check. See §6.
- **→ RESOLVED 2026-06-09 (late session): two-pool fit. See §9.** CATs++/RAFT are mutually compatible (pooling them on the clean table reproduces the pre-GLU-Net headline exactly: 0.52/0.46/0.31); GLU-Net is the lone incompatible responder and gets its own fit. Consensus fraction recomputed under this structure: 0.63/0.71/0.39 of the unchanged 0.731 ceiling.

---

## 1. The pipeline & where `g` is fit
- `perf(i,k,v) = L(i,k,v) + g(features(i→k))`. **All ranking claims are on `g`.**
- A *context* = `(benchmark, model, pretrained, freeze)`. Within each context, features + target are **demeaned by context mean**, then ONE shared RidgeCV is fit on the pooled demeaned rows (α∈[0.01..1000]; winsorize 1/99 → median-impute → standardize). So `g` is **shared across all contexts/architectures**; only the demeaning is per-context.
- `ctx_rho_g` (the headline) = for each context compute Spearman(actual, g), then average over contexts. Reported per split: **LOTO** (leave-one-source), **LOBO** (leave-one-benchmark), **JOINT**.
- Code: `scripts/transfer_analysis_v4/experiments.py` (`_fit_ridge`, within-demean, `feature_cols`); driver `run_v4.sh`.
- Inputs: `scripts/transfer_analysis_v3/transfer_table.csv` (the perf table) + `analysis_v3/pairwise_self_distances.csv` (the motion/appearance distance features).

GLU-Net was ingested into the table this session (clean, verified): `transfer_table.csv` = **catspp 960 + glunet 440 + raft 220 = 1620 rows**. GLU-Net rows are **peak-only** (auc_normalized=NaN). See `glunet-v4-ingestion` notes / `compute_glunet_auc.py`.

---

## 2. THE FINDING — GLU-Net in the shared ridge weakens every architecture

### 2a. Pooled motion vs appearance ctx_rho_g (13-feat `motion` family, pure-11, peak_pck, L_MODE=mixed)
| split | motion **pre-GLU-Net** | motion **post-GLU-Net** | appearance pre | appearance post |
|---|---|---|---|---|
| LOTO | +0.508 | **+0.308** | −0.254 | −0.220 |
| LOBO | +0.448 | **+0.355** | +0.074 | +0.160 |
| JOINT | +0.321 | **+0.181** | −0.202 | −0.208 |

Motion≫appearance gap narrowed: LOTO 0.76→0.53, **LOBO 0.37→0.20**, JOINT 0.52→0.39. Motion still wins every split, but weaker.

**Post-GLU-Net CIs (N_BOOT=200, from `results_glunet_clean/summary.csv`) — and a casualty:**
| split | motion ρ_g [95% CI] | appearance ρ_g [95% CI] |
|---|---|---|
| LOTO | +0.308 **[+0.057, +0.498]** | −0.220 [−0.522, +0.134] |
| LOBO | +0.355 **[+0.272, +0.415]** | +0.160 [+0.063, +0.256] |
| JOINT | +0.181 **[−0.009, +0.290]** | −0.208 [−0.325, −0.033] |
- **LOBO** is the strong split: motion CI [+0.272,+0.415] does **not overlap** appearance [+0.063,+0.256] → motion>appearance holds (barely — 0.272 vs 0.256).
- **JOINT motion ρ_g now includes 0** ([−0.009,+0.290]) — i.e. **not significantly >0** with GLU-Net pooled in (pre-GLU-Net JOINT motion was +0.321, clearly significant). This is a real casualty.
- LOTO: motion CI excludes 0 but appearance CI is wide and crosses 0; the proper test is the **paired motion−appearance gap** (P(gap>0)) in `bootstrap_gap.csv`, not CI overlap — compute it for all 3 splits before claiming significance.
- pre = `results_mixed/summary_points_peak_pck.csv` (05-27, catspp+raft only, 590 rows, 11 sources)
- post = `results_glunet_clean/summary_points_peak_pck.csv` (06-09, +glunet, 980 rows, 11 sources)
- **Verified apples-to-apples:** both pure-only, 11 sources, same catspp/raft base; only difference = glunet added.

### 2b. Per-VARIANT motion ρ_g — the drop hits CATs++/RAFT, not just GLU-Net (LOTO shown)
Same CATs++/RAFT data, identical settings, only difference = GLU-Net in the shared pool:
| variant | pre-GLU-Net | post-GLU-Net | drop |
|---|---|---|---|
| catspp F/F | +0.587 | +0.401 | **−0.19** |
| catspp F/T | +0.622 | +0.398 | **−0.22** |
| catspp T/F | +0.293 | +0.250 | −0.04 |
| catspp T/T | +0.434 | +0.393 | −0.04 |
| raft T/F | +0.605 | +0.465 | −0.14 |

The **from-scratch CATs++ variants (F/F, F/T) drop the most** — exactly the cells the §3.1 mechanism story leans on ("no appearance floor → motion governs"). The existing briefing §3.1 numbers (F/F +0.605 etc.) are the **pre-GLU-Net** values.

### 2c. Full per-variant breakdown, POST-GLU-Net (clean run), motion / appearance
```
LOTO   catspp F/F +0.401/-0.468  F/T +0.398/-0.387  T/F +0.250/-0.153  T/T +0.393/-0.068
       glunet F/F +0.352/-0.180  F/T +0.315/-0.339  T/F -0.030/-0.105  T/T +0.227/+0.124
       raft   T/F +0.465/-0.401
LOBO   catspp F/F +0.392/-0.028  F/T +0.400/+0.025  T/F +0.275/+0.195  T/T +0.403/+0.296
       glunet F/F +0.399/+0.304  F/T +0.363/+0.070  T/F +0.138/-0.008  T/T +0.405/+0.410
       raft   T/F +0.422/+0.177
JOINT  catspp F/F +0.235/-0.383  F/T +0.210/-0.286  T/F +0.046/-0.190  T/T +0.228/-0.069
       glunet F/F +0.180/-0.180  F/T +0.207/-0.305  T/F +0.039/-0.143  T/T +0.254/+0.084
       raft   T/F +0.227/-0.404
```
Note GLU-Net **T/F** is the weak spot (LOTO −0.030, LOBO +0.138) and **T/T** appearance is sometimes positive/high (LOBO +0.410) — GLU-Net pretrained-trainable behaves oddly. The from-scratch GLU-Net (F/F, F/T) actually look fine (+0.35–0.40).

### 2d. Per-ARCHITECTURE (pooled across that arch's variants), POST
| split | CATs++ | RAFT | GLU-Net |
|---|---|---|---|
| LOTO motion | +0.360 | +0.465 | +0.216 |
| LOBO motion | +0.368 | +0.422 | +0.326 |
| JOINT motion | +0.180 | +0.227 | +0.170 |

RAFT (also flow-native) predicts **best**; GLU-Net (also flow-native) predicts **worst**. So it's NOT "flow architectures are unpredictable" — it's GLU-Net specifically.

---

## 3. Mechanism hypothesis (to test)
The shared ridge assumes one motion→transfer function across all architectures. If GLU-Net's function differs (e.g. it saturates, or weights different motion features, or its peak_pck is noisier because it's peak-only at 100 steps/epoch with 6 cos cells only reaching ep80), then forcing a shared ridge produces a compromise that fits **none** well — and the pooled `g` predictions degrade for CATs++/RAFT too. Evidence consistent with this:
- The drop is largest for the cells with the strongest pre-existing signal (catspp F/F/F/T), i.e. the shared fit is being pulled away from them.
- GLU-Net T/F is near-zero/negative — a genuinely different response, not noise around the same line.

**Things to check:** (a) fit `g` separately per architecture and compare; (b) inspect GLU-Net's within-context feature↔transfer scatter vs CATs++; (c) check whether the 6 ep-80 GLU-Net cells (synthetic_2d_warp, large_zoom, random_flipping) are adding noise; (d) check the peak-only vs AUC handling for GLU-Net (`build_table.py` keeps glunet rows with auc_points<3, auc_normalized=NaN).

---

## 4. The consensus / predictability-ceiling angle (the "disagreement")
The §6 claim uses **cross-architecture consensus**: how much of the empirical "predictable" signal (the agreement among architectures on source rankings, the oracle ceiling) does motion capture. Pre-GLU-Net (`results_glunet_observed_peak_all_splits/CROSS_ARCHITECTURE_CONSENSUS_ALL_SPLITS.csv`):
| split | motion_rho_held_arch | cross_arch_consensus_rho (ceiling) | fraction |
|---|---|---|---|
| LOTO | 0.477 | 0.619 | 0.77 |
| LOBO | 0.491 | 0.619 | 0.79 |
| JOINT | 0.351 | 0.619 | 0.57 |

The claim was "motion captures ~77–79% of the cross-architecture consensus." There are two consensus CSVs:
- `results_glunet_observed_peak_all_splits/...`: ceiling 0.619, fraction 0.77/0.79/0.57 — but this **already contains GLU-Net** (glunet **110** rows, a preliminary grid). NOT a no-GLU-Net baseline.
- `results_glunet_clean/...`: ceiling 0.731, fraction 0.52/0.56/0.30 — full GLU-Net grid (glunet **440**).

⚠️ **DO NOT read 0.619→0.731 as "adding GLU-Net raised the ceiling" — that comparison is confounded** (both have GLU-Net; the difference is preliminary-110 vs full-440 grid). An earlier version of this doc made that claim; it is **retracted**. We do **not** have a clean no-GLU-Net ceiling, because a 2-architecture (CATs++ + RAFT only) held-one-arch-out consensus is degenerate (one arch predicting one other) — so "what GLU-Net did to the ceiling" is not cleanly measurable with what's on disk.

What IS true: with the full grid, **motion captures only ~30–56% of the cross-arch consensus** (LOTO 0.52, LOBO 0.56, JOINT 0.30). Whether that fraction is "low because GLU-Net transfers by a non-motion rule" vs "low because the **shared ridge** mismeasures motion's predictive power across heterogeneous architectures" is **the open question** — and it's the same shared-ridge issue as §2. Resolve it by recomputing both ρ_g and the consensus fraction under a **per-architecture** fit before drawing any ceiling conclusion. Code: `regenerate_consensus_csv.py`, `ceiling_analysis.py`.

**Conceptual note** (why agreement-up and motion-down *can* coexist in principle, even though we can't demonstrate it cleanly here): consensus measures whether architectures agree *with each other*; motion ρ measures whether the *motion feature* explains that agreement. Extra agreement that isn't motion-shaped → higher consensus, lower motion fraction. But here the more likely driver of motion's low fraction is the shared-ridge measurement artifact, not a genuine non-motion agreement.

---

## 5. What is NOT affected (don't re-investigate these)
- **The search predictor (`interventional-study/full_fit/peak_pck/*.pkl`)** is a SEPARATE fit (full-table, in-sample). Refitting it with GLU-Net changed it negligibly: coefficients same sign, α unchanged (100), in-sample R² 0.091→0.107, and the predicted-score surface is **0.9986 rank-correlated** old-vs-new on real data with the same argmax. So the interventive search is unaffected; this issue is purely about the **v4 `ctx_rho_g` headline + consensus**.
- The from-scratch CATs++ intervention grid (§9.5) and the motion-vs-appearance distance check are a different analysis (intervention sources, not the canonical-11), independently confounded (frozen camera + appearance) — see `expressive_kubric_plan.md`. Don't conflate.

---

## 6. THE OPEN DECISION (what the new agent should resolve)
How to report `g` now that 3 heterogeneous architectures are in:
1. **Per-architecture `g` (likely correct).** Fit `g` separately per architecture; report each (CATs++ ~0.59, RAFT ~0.61, GLU-Net ~0.4 — its own honest number). Pros: each keeps real signal; heterogeneity becomes a finding; no misspecification. Cons: loses the single "one model across architectures" framing; needs a code path (experiments.py currently shares `g` — add a `--per-arch` / group-by-arch fit, or run experiments per architecture subset). ~16 min/run.
2. **Keep pooled.** Report the weaker shared-ridge number (LOTO +0.31 / LOBO +0.36 / JOINT +0.18). Honest but materially weaker and hides GLU-Net misspecification.
3. **CATs++/RAFT headline + GLU-Net as a generalization check.** Headline = strong pre-GLU-Net pooled CATs++/RAFT; GLU-Net presented separately (still motion>appearance, weaker). Keeps the strong headline; GLU-Net is supporting cross-arch evidence.

Recommendation leaning: **(1) or (3)** — both preserve the real per-architecture signal and turn the heterogeneity into an honest finding rather than a diluted average. Whichever is chosen, the **consensus/ceiling (§4) must be recomputed and framed consistently** with it.

---

## 7. Reproduce / provenance (all real, no synthetic values)
```
# clean end-to-end (what produced the post-GLU-Net numbers):
bash OnlineSyntheticCorrespondence/run_clean_pipeline.sh      # build_table -> full_fit -> run_v4 (N_BOOT=200) -> consensus
# pre-GLU-Net comparison run already on disk:
scripts/transfer_analysis_v4/results_mixed/                   # catspp+raft only (05-27)
scripts/transfer_analysis_v4/results_glunet_clean/            # +glunet (06-09)
# per-variant / per-arch ctx_rho_g computed directly from:
#   results_*/predictions/peak_pck/rows_{LOTO,LOBO,JOINT}_{motion,appearance}.csv
#   group by context_id -> spearman(actual,g) per context -> average by variant/arch
# table backup before glunet: scripts/transfer_analysis_v3/transfer_table.csv.bak_pre_glunet_*
# predictor backup: interventional-study/full_fit/_pre_glunet_bak_*
```
Faster reruns: `SKIP_GBM=1 SKIP_FIGURES=1` cuts the ~16-min experiments step to ~3–5 min (GBM head + figures aren't needed for ρ_g). Bootstrap is single-threaded and slow — N_BOOT=200 is plenty for CIs; point estimates don't need it.

## 8. Key numbers to carry forward
- Pooled motion ρ_g: pre **0.51/0.45/0.32** → post **0.31/0.36/0.18** (LOTO/LOBO/JOINT)
- Per-variant LOTO drops: catspp F/F 0.59→0.40, F/T 0.62→0.40, raft 0.61→0.47
- Per-arch (post): CATs++ 0.36/0.37/0.18, RAFT 0.47/0.42/0.23, GLU-Net 0.22/0.33/0.17
- Consensus (full GLU-Net grid): motion captures **0.52/0.56/0.30** of the 0.731 ceiling. ⚠️ the "0.619→0.731" pre/post is CONFOUNDED (both have GLU-Net; preliminary-110 vs full-440) — retracted; no clean no-GLU-Net ceiling exists. Recompute fraction under per-arch fit.
- ρ_g CIs (post, N=200): LOTO +0.31 [+0.06,+0.50], LOBO +0.36 [+0.27,+0.42], JOINT +0.18 **[−0.01,+0.29] (crosses 0)**
- Predictor surface stability (search, unaffected): 0.9986 rank corr, argmax unchanged

---

## 9. RESOLUTION (2026-06-09, late session) — two-pool fit + gap-stratified reframe

`experiments.py` gained `--keep-model` (filter table by `model_family` before fitting; the
shared-ridge pool then contains only those architectures' contexts). All runs below:
clean table (catspp 960 + glunet 440 + raft 220), `--targets peak_pck --pure-only
--no-gbm --families motion appearance`, L_MODE=mixed — identical to the clean run
except the pool restriction.

### 9a. Architecture-subset fits (motion ctx_rho_g, LOTO/LOBO/JOINT)
| pool | LOTO | LOBO | JOINT |
|---|---|---|---|
| CATs++ alone | +0.442 | +0.428 | +0.219 |
| RAFT alone (only 100 rows — data-starved) | +0.474 | +0.515 | +0.332 |
| **CATs++ + RAFT pooled** | **+0.519** | **+0.457** | **+0.306** |
| GLU-Net alone | +0.130 | +0.400 | +0.139 |
| GLU-Net alone, zridge head | +0.211 | +0.427 | +0.222 |
| (all 3 shared — the §2 dilution) | +0.308 | +0.355 | +0.181 |

- **CATs+++RAFT pooled REPRODUCES the pre-GLU-Net headline on the current clean
  table**: 0.52/0.46/0.31 vs pre 0.51/0.45/0.32; per-variant LOTO catspp F/F +0.594,
  F/T +0.629, raft +0.611 (pre: 0.587/0.622/0.605). The two are *compatible*; pooling
  them helps RAFT (alone 0.47 → pooled 0.61 per-variant, borrowing strength).
- **GLU-Net is the lone incompatible responder.** Its own honest fit: LOBO strong
  (+0.40; T/T variant +0.572!), LOTO weak (+0.13), dragged by T/F (−0.046).
- **ep-80 noise check:** dropping the 3 under-trained sources (synthetic_2d_warp,
  large_zoom, random_flipping; cos cells only reached ep80) lifts GLU-Net LOTO
  0.13 → **0.235** (zridge 0.239), LOBO 0.473 (zridge 0.557). Different source set
  (8 vs 11) so not strictly comparable, but consistent with part of GLU-Net's LOTO
  weakness being measurement noise from under-trained cells, not biology.
- Shuffle controls ~0 in every subset run (no leakage introduced by the filter).

**Paper framing:** headline = CATs+++RAFT pool (0.52/0.46/0.31, the old strong
numbers, now with 3-arch context); GLU-Net reported separately as the heterogeneity
finding (motion>appearance still holds for it on LOBO/JOINT; its response function
differs and a shared fit is misspecified — connect to Reviewer C's complaint that
architecture averaging obscures interpretation).

### 9b. Consensus / ceiling under the two-pool fit
Merged rows (catsppraft g for catspp/raft rows + glunet-own g for glunet rows) →
`regenerate_consensus_csv.py` (min-src 4, n-boot 500):
| split | motion_rho_held_arch | ceiling | fraction (was, shared fit) |
|---|---|---|---|
| LOTO | 0.458 | 0.731 | **0.63** (0.52) |
| LOBO | 0.519 | 0.731 | **0.71** (0.56) |
| JOINT | 0.288 | 0.731 | **0.39** (0.30) |
Ceiling unchanged (actuals only). The §4 open question is answered: a large part of
the "low fraction" was the shared-ridge measurement artifact. Honest claim now:
"motion captures ~63–71% (LOTO/LOBO) of the cross-architecture consensus."

### 9c. Gap-stratified pairwise accuracy (NEW: `pairwise_gap_analysis.py`)
Answers "0.70 pairwise accuracy is not useful" + "similar sets genuinely flip".
Predictor accuracy by true |peak_pck gap| (two-pool fit), vs the empirical
reproducibility of the SAME pair ordering across independently trained contexts:
```
                       0-1     1-2     2-5     5-10    >10     ALL
LOTO predictor         0.561   0.559   0.577   0.636   0.696   0.625
LOBO predictor         0.588   0.561   0.586   0.662   0.768   0.661
same-arch retrain      0.582   0.621   0.701   0.775   0.844   0.736
cross-arch retrain     0.564   0.619   0.662   0.710   0.774   0.689
```
- Pairs with gap <2 PCK are **irreducible**: even the same architecture retrained
  under a different regime agrees with itself only 58–62% there. The predictor's
  misses concentrate exactly on those pairs.
- At gaps >10 PCK (the decisions that matter), **LOBO predictor 0.77 ≈ cross-arch
  retraining 0.77**, approaching same-arch 0.84 — i.e. motion features rank
  well-separated source pairs as reliably as retraining a different architecture.
- Output CSVs: `results_glunet_clean/pairwise_gap_analysis.csv` (shared fit),
  `results_perarch_merged/pairwise_gap_analysis.csv` (two-pool fit).

### 9d. Provenance / repro
```
# per-pool fits (each ~2 min):
python scripts/transfer_analysis_v4/experiments.py --targets peak_pck --pure-only \
  --no-gbm --families motion appearance --keep-model catspp raft \
  --out scripts/transfer_analysis_v4/results_perarch_catsppraft
#   ... same with --keep-model glunet -> results_perarch_glunet
#   ... --keep-model glunet --drop-source synthetic_2d_warp synthetic_large_zoom \
#       synthetic_random_flipping -> results_perarch_glunet_noep80
# merged two-pool predictions + consensus + gap analysis:
#   results_perarch_merged/predictions/peak_pck/rows_*  (concat catsppraft+glunet)
#   regenerate_consensus_csv.py --rows-dir results_perarch_merged/predictions/peak_pck
#   pairwise_gap_analysis.py --rows-dir <same>
```
### 9e. Bootstrap CIs (N_BOOT=200, ridge head, run 2026-06-09 late)
**CATs+++RAFT pool (headline):**
| split | motion ρ_g [95% CI] | appearance ρ_g [95% CI] | motion−appearance gap | P(gap>0) |
|---|---|---|---|---|
| LOTO | +0.519 [+0.247, +0.693] | −0.243 [−0.499, +0.112] | +0.763 [+0.124, +0.987] | 0.980 |
| LOBO | +0.457 [+0.396, +0.533] | +0.078 [+0.002, +0.157] | +0.379 [+0.262, +0.478] | 1.000 |
| JOINT | +0.306 [+0.140, +0.420] | −0.187 [−0.302, −0.013] | +0.493 [+0.248, +0.707] | 1.000 |

- Motion CI excludes 0 in ALL THREE splits — the shared-fit JOINT "casualty"
  (CI crossing 0, §2a) is reversed under the two-pool fit.
- Motion>appearance significant in all three splits (P ≥ 0.98).

**GLU-Net pool (reported separately, the heterogeneity finding):**
| split | motion ρ_g [95% CI] | gap | P(gap>0) |
|---|---|---|---|
| LOTO | +0.130 [−0.206, +0.368] | +0.249 | 0.775 |
| LOBO | +0.400 [+0.316, +0.482] | +0.125 | 0.910 |
| JOINT | +0.139 [−0.082, +0.331] | +0.332 | 0.985 |

Honest GLU-Net claim: motion signal robust on LOBO (CI excludes 0) and the
motion−appearance gap significant on JOINT; LOTO is noise-limited (peak-only
measurement + ep-80 cells, §9a). Do not claim more than this for GLU-Net.
