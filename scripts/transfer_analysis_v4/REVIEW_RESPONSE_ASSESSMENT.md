# Review Response Assessment — v4 Positioning

This is a critical organization note comparing the ECCV 2026 reviews against
the current v4 evidence in `CLAIMS.md`, `ABLATION.md`, and
`INTERVENTIONAL_STUDY.md`.

## Executive read

The new version is materially stronger, but it is no longer the same paper.
The old submission was framed around **directed coverage** and
SDF-Fractal3D as an almost method-driven dataset-design story. The v4 evidence
supports a cleaner and more defensible claim:

> Across held-out dyadic transfer regimes, **motion-domain cross-distance**
> predicts correspondence transfer substantially better than appearance-domain
> distance, and this signal can potentially be used as a fast screening
> objective for controllable synthetic-data search.

That pivot fixes several reviewer concerns, especially the expert criticism
that directed coverage felt heuristic and over-claimed. But it also means the
paper should stop leading with directed coverage as the core contribution.
Directed/asymmetric coverage should become either:

- a diagnostic visualization / historical baseline,
- one member of a broader motion-distance feature family, or
- an ablated feature subset that is not necessary for the headline.

The strongest current evidence is statistical, not causal. The paper should
avoid claiming that motion is *the dominant causal factor* in general. The
defensible claim is that motion distance is the strongest measured predictor
of transfer in this study, while appearance distance is weak or negative.

## What is fixed

### 1. The old "0.70 pairwise accuracy is not useful" objection is mostly fixed

The old metric was pairwise ranking accuracy around 0.69--0.70. Reviewers read
that as only moderately useful. v4 replaces the headline with within-context
Spearman correlation of the learned transfer component `g` against
`peak_pck`.

Current headline:

| Regime | motion rho_g | appearance rho_g | gap |
|---|---:|---:|---:|
| LOTO | +0.508 | -0.254 | +0.762 |
| LOBO | +0.448 | +0.074 | +0.374 |
| JOINT | +0.321 | -0.202 | +0.523 |

The paired bootstrap gap is decisive: P(gap > 0) is 0.998 or 1.000 across the
three regimes. This is a better paper statistic than c-index because it states
the scientific contrast directly: motion beats appearance.

Recommended framing:

> We do not present the estimator as a high-precision absolute performance
> oracle. We test whether measurable dataset-pair distances contain a
> transfer-ranking signal. Motion distance does; appearance distance largely
> does not.

### 2. The "appearance realism" claim is now better grounded

The old paper argued from SDF-Fractal3D's non-photorealistic performance. That
was easy to attack because SDF differs from other data in many ways besides
realism.

v4 has a much stronger cross-dataset statement: DINO appearance distance fails
to predict transfer, while flow/motion distance predicts transfer in all three
held-out regimes. This is still not a proof that realism is irrelevant, but it
does directly weaken the "appearance similarity is the main transfer driver"
hypothesis.

Recommended language:

> Appearance similarity, as measured by DINO cross-distance, is not a reliable
> transfer predictor in our correspondence transfer grid. This does not imply
> that photorealism is useless; it implies that appearance similarity alone is
> insufficient to explain which training sources transfer.

Avoid:

> Photorealism is not primary.

That sentence invites the exact Reviewer C objection again.

### 3. Leakage and residualization concerns are much better handled

The new `L + g` decomposition is clearer than the old context-residual text.
The key improvement is that the claims explicitly separate:

- `L`: calibration / level setting,
- `g`: within-context ranking signal.

The L-mode ablation is strong: `rho_g` is identical across mixed,
symmetric-informed, symmetric-uninformed, and targeted-informed L mechanisms.
That directly answers the concern that the context residual machinery might be
creating the result.

Important paper move:

The main method section should say early that all scientific ranking claims are
on `g`, not on calibrated absolute `L + g`. Absolute prediction can be a
secondary engineering feature, not the main scientific claim.

### 4. Dataset size / density is no longer a soft spot

The old reviews asked whether dataset size was being treated as a nuisance
despite scale affecting representation learning. v4 gives a direct answer:
density alone is weak or anti-predictive, and adding density does not explain
away motion.

Current result:

| Family | LOTO | LOBO | JOINT |
|---|---:|---:|---:|
| motion | +0.508 | +0.448 | +0.321 |
| density | -0.282 | +0.153 | -0.213 |
| motion + density | +0.265 | +0.472 | +0.050 |

This should be reported as a robustness/control result, not as "size does not
matter for training." The careful statement is:

> Dataset size/density does not account for the observed motion-distance
> ranking signal in this transfer grid.

### 5. Target metric improved

Using `peak_pck` as the headline is a good correction. The old
`auc_normalized` target mixed final transfer quality with training speed. v4
shows the motion signal is stronger on `peak_pck`, while the motion-over-
appearance ordering survives on `auc_normalized`.

This is a clean response to reviewer unease about reported metrics. Lead with
`peak_pck`; put `auc_normalized` in robustness.

## What is only partly fixed

### 1. Directed coverage may not matter, and that changes the novelty story

The feature ablation says the strongest overall subset is `mean_nn`:

| subset | LOTO | LOBO | JOINT | mean |
|---|---:|---:|---:|---:|
| mean_nn | +0.419 | +0.501 | +0.478 | +0.466 |
| asym_only | +0.515 | +0.450 | +0.324 | +0.430 |
| all 13 | +0.508 | +0.448 | +0.321 | +0.426 |
| coverage | +0.321 | +0.387 | +0.240 | +0.316 |
| kl | -0.276 | +0.095 | -0.250 | -0.144 |

This is a big deal. It means the paper should not keep saying "directed
coverage" is the core technical mechanism unless a final ablation reverses
this. The current data say simpler mean nearest-neighbor motion distances are
at least as good, and often better.

There are two viable ways to organize this:

**Conservative version:**
Lead with all 13 motion cross-distance features. Present directed coverage,
mean-NN, KL, and symmetric terms as components. Then show mean-NN alone is a
surprisingly strong minimalist variant.

**Sharper version:**
Lead with the 3-feature `mean_nn` model because it is simpler and strongest on
LOBO/JOINT. Relegate directed coverage to diagnostics and ablations.

My recommendation: use the conservative version unless final ablations make
`mean_nn` clearly dominant and stable under drop-source robustness. Reviewers
are more likely to trust "all pre-specified features, then ablated" than "we
picked the strongest subset."

### 2. The interventional/practical story is not fixed yet

`INTERVENTIONAL_STUDY.md` has the right plan, but it is still a plan. The old
Reviewer A objection was that the metric remained an analysis tool, not a
practical construction method. That is not solved until the predictor-guided
search has actual predicted-vs-actual validation.

The mandatory validation experiment is the right gate:

1. Pick 5--10 deliberately diverse generated variants.
2. Predict their rankings using the full-fit predictor.
3. Train them.
4. Report Spearman rho between predicted and actual rankings.

If that rho is good, the practical contribution becomes real. If it is weak,
the practical claim should stay modest: fast diagnostic screening, not
automated dataset design.

The interventional note now uses the same settled `peak_pck` motion results as
`CLAIMS.md`: +0.508 / +0.448 / +0.321 for LOTO / LOBO / JOINT.

### 3. Causal language remains risky

The new evidence is much better for prediction than for causality. The motion
intervention story may help, but it remains within one synthetic family unless
the final ablations expand it.

Safe claim:

> Motion-distance features are the strongest measured predictors of transfer,
> and controlled SDF variants provide partial intervention evidence that
> changing motion statistics can change transfer in the predicted direction.

Unsafe claim:

> Motion support is the dominant causal driver of correspondence transfer.

Reviewer C will likely reject that stronger version again.

### 4. The old SDF-Fractal3D dataset narrative still needs cleanup

Reviewer A's confusion was not only about numbers. It was about paper order:
the dataset appeared first, the metric appeared later, and the relationship
felt retroactive.

The new organization should be:

1. Problem: predicting dataset transfer for correspondence.
2. Measurement: motion vs appearance cross-distance features.
3. Main result: motion predicts transfer, appearance does not.
4. SDF-Fractal3D: a controllable generator/case study used to test and exploit
   that signal.
5. Interventional/search result: optional practical demonstration if validated.

This avoids pretending SDF was originally designed by the learned metric.

## Still-dangerous review gaps

### 1. Baseline breadth

Reviewer C objected that the old baselines were narrow. v4 has stronger
internal ablations, but not necessarily stronger external transferability
baselines. If space permits, add at least one or two simple external baselines:

- symmetric MMD / Wasserstein / FID-style distances if now available,
- dataset-only and benchmark-only metadata baselines,
- stronger learned transferability features if cheap,
- random features with matched dimensionality, already partly present.

If not, be explicit that this is a controlled diagnostic study rather than a
survey of all transferability estimators.

### 2. Drop-one-source robustness

This is listed as TODO in `CLAIMS.md` and is high value. With only 11 pure
training sources, a skeptical reviewer can ask whether one source drives the
motion-vs-appearance gap. A drop-one-source sweep would directly answer that.

Priority: high.

### 3. Sparse vs dense task partition

The old paper mixed flow, tracking, stereo, and semantic correspondence. v4 can
answer this better by partitioning benchmarks and reporting whether the motion
signal survives in sparse semantic benchmarks separately from dense flow-like
benchmarks.

Priority: high because it addresses both Reviewer A's task-scope confusion and
Reviewer C's mixed-regime concern.

### 4. Architecture generalization

The old reviewers asked about point tracking, temporal models, and video
foundation models. Unless you add those experiments, scope the claim to the
evaluated RAFT/CATs++-style correspondence settings. The per-variant breakdown
helps, especially showing stronger signal when training data actually explains
variance, but it does not establish broader architecture generalization.

### 5. Absolute performance / performance ceiling

Do not let the paper imply the predictor is an absolute score oracle or that
SDF-Fractal3D is universally superior. The old transfer table showed SDF trails
some existing datasets. The new paper can turn this into a strength:

> SDF-Fractal3D is not claimed as a universal replacement for real or
> photorealistic training data. It is useful because it is controllable, cheap,
> and exposes motion factors that the transfer analysis identifies as
> predictive.

## Recommended paper structure

### Abstract

Use four claims, in this order:

1. Dataset transfer in correspondence is poorly explained by visual similarity
   alone.
2. A dyadic transfer analysis over training sources, benchmarks, and model
   variants shows motion cross-distance predicts transfer while DINO appearance
   distance does not.
3. The motion-over-appearance gap survives held-source, held-benchmark, and
   joint held-out regimes, plus leakage, density, L-mechanism, and target-metric
   controls.
4. In controllable synthetic generation, the predictor can be used as a fast
   screening objective; final wording depends on validation status.

### Introduction

Do not open with SDF-Fractal3D as the main novelty. Open with the transfer
selection problem. Introduce SDF later as a controllable testbed.

Avoid the old "Flow x2 + Appearance x2" shorthand. If needed, say "features
computed in both train-to-target and target-to-train directions."

### Method

Use the `L + g` decomposition as the conceptual backbone:

```text
performance(source, benchmark, model) = context level L + transfer signal g
```

Then state that the scientific comparisons rank by `g`.

Feature organization should be:

- motion cross-distance features,
- appearance cross-distance features,
- density / random controls,
- optional directed coverage subfeatures.

Directed coverage should not be the title of the method unless final results
show it is necessary.

### Results

Put the main result first:

1. Motion vs appearance rho_g table on `peak_pck`.
2. Paired bootstrap gap table.
3. Shuffle/leakage control.
4. Density control.
5. L-mechanism invariance.
6. Feature subset ablation, explicitly noting that `mean_nn` is strong and KL
   is bad.
7. Target metric robustness: `peak_pck` headline, `auc_normalized` secondary.
8. Optional per-variant breakdown.
9. Interventional/search validation, if complete.

### Discussion / Limitations

Be direct:

- not a causal proof of universal motion dominance,
- not validated for temporal foundation models or 4D tracking,
- appearance/photorealism may still matter for ceiling performance,
- only 11 pure training sources,
- directed coverage is diagnostic, not required by all successful variants.

## Reviewer-by-reviewer status

### Reviewer A, S1/C4

Mostly addressable by reorganization.

Fixed or improved:

- Practical value is clearer if the interventional search validates.
- Dataset sample counts and fairness can be handled in setup.
- Task scope can be stated upfront.
- The "analysis tool not construction method" complaint can be answered only
  if the search loop produces actual validated rankings.

Still needs writing:

- Explain SDF design motivation earlier.
- Be honest that SDF is a controllable testbed/counterexample, not always the
  best dataset.
- Remove confusing shorthand.

### Reviewer B, S3/C2

Likely much improved.

Fixed or improved:

- Outlier/normalization concern is partly answered by winsorization and target
  metric robustness, though BFV-specific clipping/magnitude handling still
  needs explicit method text.
- Dataset-size concern is answered by density controls.
- Absolute-vs-relative prediction is answered by `L + g`.
- Performance ceiling can be handled with careful SDF framing.

Still needs writing:

- Exact flow magnitude handling.
- HOF/DINO dimensionality/reduction details if HOF remains in the paper.
- Limit claims beyond evaluated architectures.

### Reviewer C, S2/C5

This remains the hardest reviewer. v4 helps, but only if claims are narrowed.

Fixed or improved:

- The evidence is now a robust dyadic prediction result rather than a broad
  causal story from one dataset.
- The directed coverage heuristic is less central if the paper pivots to
  motion cross-distance.
- Feature subset ablations and L-invariance make the estimator look less like
  fragile feature engineering.

Still risky:

- Novelty may still read as empirical rather than algorithmic.
- Causality remains limited.
- Baseline breadth may still be criticized.
- Only 11 pure training sources makes source robustness essential.

## Highest-priority action list

1. Decide the headline feature set: conservative `all 13` vs cleaner
   `mean_nn`. My recommendation is `all 13` as headline, `mean_nn` as a strong
   ablation, unless drop-one-source clearly favors `mean_nn`.
2. Run drop-one-source robustness.
3. Produce sparse-vs-dense benchmark partition results.
4. Update `INTERVENTIONAL_STUDY.md` numbers to match `CLAIMS.md`.
5. Finish the interventional validation before making any practical search
   claim.
6. Rewrite the paper outline around motion-vs-appearance transfer prediction,
   not directed coverage.
7. Move SDF-Fractal3D from "main proof" to "controllable case study /
   practical generator."
8. Add a blunt limitations paragraph that concedes remaining causality,
   architecture, and photorealism-ceiling limits.
