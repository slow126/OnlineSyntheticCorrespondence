# Interventional Study — Predictor-Guided Hyperparameter Search

**Briefing for a new agent.** This document describes the next phase of the
project: turning the transfer-prediction analysis (claim validated; see
`CLAIMS.md`) into an *actionable tool* that drives synthetic-data
hyperparameter search.

If you're picking up this work, read the documents below in order before
writing code, then come back here for the implementation plan.

---

## Read first (in order)

| Doc | Purpose | Time to read |
|---|---|---|
| `scripts/transfer_analysis_v4/README.md` | What v4 is, how to run the base pipeline | 5 min |
| `scripts/transfer_analysis_v4/HANDOFF.md` | Engineering log — what was tried, what stays, what to never do again | 20 min |
| `scripts/transfer_analysis_v4/CLAIMS.md` | Paper-prep doc: each scientific claim mapped to its evidence | 15 min |
| `scripts/transfer_analysis_v4/ABLATION.md` | Cross-mode ablation results (auto-generated; refresh with `compile_ablation_summary.py`) | 5 min |
| `scripts/transfer_analysis_v4/results_mixed/results.md` | The headline per-mode report. Has the bars, scatters, and tables. | 10 min |
| `scripts/transfer_analysis_v3/HANDOFF.md` | The v3 feature-generation pipeline (still alive; v4 reads its outputs) | optional, 15 min |

Together this is ~1 hour of context. Don't skip the HANDOFFs — they have
the "we tried this and it failed, don't redo" log.

---

## What's settled (you don't need to revisit any of this)

From `CLAIMS.md` Claims 1–11:

- **Motion (flow) cross-distance predicts transfer** with within-context
  Spearman ρ ≈ +0.47 / +0.48 / +0.28 on LOTO / LOBO / JOINT (`peak_pck`
  target). Paired bootstrap motion − appearance gap has P > 0.997.
- **Appearance (DINO) cross-distance does NOT predict transfer** — weakly
  anti-predicts on LOTO/JOINT, near-zero on LOBO.
- The result is **robust to L choice, regression head, target metric,
  feature subset, leakage controls, density confounds, and DINO outliers**.
- The **g** model (within-context ridge on demeaned 13-dim flow features)
  is the science. The **L** term (calibration band) is regime-specific:
  cell_mean on LOTO, eval-similarity IDW on LOBO, variant intercept on JOINT.
- **For a new candidate dataset, we are in the LOTO regime**: the candidate
  is the held source; benchmarks are observed. So L = cell_mean(k, v),
  which is a *constant* per (benchmark, variant) — the same for every
  candidate. The candidate's score is differentiated entirely by g.

---

## The goal

Build a **fast feedback loop** for synthetic-data hyperparameter search.

The pipeline:

```
[generate kubric/SDF variant with hyperparameters θ]
        ↓
[extract flow vectors from a small sample]
        ↓
[compute 13-dim (i_θ → k) features for each benchmark k]
        ↓
[predict peak_pck(i_θ, k, v) for each (benchmark, variant)]
        ↓
[aggregate to a single fitness score]
        ↓
[update search strategy; pick next θ; repeat]
        ↓
(after search converges)
[full training of best candidate for paper-quality numbers]
```

The user controls motion content via camera trajectory in their
kubric/SDF pipeline. This is the primary axis to search over.

**Paper framing for the practical claim:**
> "Our predictor enables hyperparameter search for synthetic data
> generation at ~1-2 minutes per candidate (vs ~10 hours per training
> run). We validate that predictor rankings match actual rankings on N
> training runs, then demonstrate the search loop discovers a variant
> that outperforms the baseline by X% on Y benchmarks."

---

## The hard requirement: validate before you trust

This is the critical gate. **Do not skip.**

The predictor was trained on 11 fairly diverse training sources. When you
generate 100 kubric variants that differ only in small camera trajectory
tweaks, those variants live in a tight cluster of feature space that the
predictor's training distribution may not cover. Two failure modes:

1. **Saturation** — all variants get ~cell_mean prediction; g ≈ 0 across
   the cluster; predictor outputs essentially the same fitness for
   everything. No signal to optimize against.
2. **Extrapolation** — variants land in feature regions far from any
   training source; ridge extrapolates linearly, could be wildly wrong.

**Validation experiment (mandatory before any reported result):**

1. Pick 5–10 candidate variants spanning a *wide* motion range (e.g.,
   stationary camera, slow pan, fast zoom, chaotic, etc. — deliberate
   diversity).
2. Generate a small dataset for each.
3. Compute features → predicted ranking.
4. Train the actual models for each variant (this is the expensive part —
   but you only do it once).
5. Report Spearman ρ between predicted and actual rankings.

If validation ρ > 0.4 → predictor is usable for search within this
generation distribution. If ρ < 0.2 → predictor is saturated or
extrapolating; need to retrain it on a more diverse set including
candidate samples, OR scope the search to a narrower region first.

This validation experiment is also the paper's strongest practical
demonstration: "the predictor's rankings transfer to the held-out
hyperparameter-search regime with ρ = X."

---

## Implementation tasks

In approximate order. Each is small (a few hours each, except #5 which is
overnight scale).

### Task 1: Full-fit predictor mode (~1 hour)

Add a `--mode full_fit` flag to `experiments.py` that:
- Skips the CV folds entirely
- Fits one ridge per (target, family) on all 540 rows
- Pickles `(imp, scl, reg, q_lo, q_hi)` for each (target, family)
- Pickles the cell_means lookup: `{(benchmark, variant): mean}`
- Saves to `scripts/transfer_analysis_v4/full_fit/<target>/<family>.pkl`

This is the predictor you'd actually use for scoring new candidates.

**Key files:**
- `experiments.py` — currently has `_fit_ridge` returning the model tuple;
  add a mode that calls this on the full table once.

### Task 2: Sample-size stability check (~2 hours, one-off)

Question: how small can a candidate's flow-vector sample be before the
13-dim feature vector becomes too noisy to rank reliably?

Procedure:
- Pick an existing source (`flyingthings` is dense, has ~250k vectors/pair)
- Subsample to N ∈ {50, 100, 500, 1000, 5000, all} pairs
- For each subsample, compute the 13-dim feature vector against each
  benchmark (110 features total)
- Compute Spearman ρ between the subsampled vector and the full-size vector
- Plot ρ vs N

**Goal:** find the smallest N where feature ρ > 0.9 (or whatever threshold
gives predictor-stable rankings). That's your search loop's per-candidate
sample size.

**Probable answer:** somewhere in [100, 1000] pairs. If much larger,
the search loop is too slow.

### Task 3: Candidate feature extractor (~3-4 hours)

A script that, given a path to a candidate dataset's flow vectors,
computes the 13-dim feature vector against each of the 10 benchmarks.

**Reuse aggressively:**
- `scripts/transfer_analysis_v3/compute_pairwise_self_distances.py` already
  knows how to compute the 13 metrics for a pair. Wrap it.
- Cached benchmark vector clouds live at `/mnt/nvme_1tb_b/coverage_vectors/`.
  The candidate's flow vectors should follow the same layout.

**Output:** `candidate_features.csv` with one row per benchmark (~10 rows,
13 feature cols).

**Time:** ~1-2 min per candidate if sample size is well-tuned (Task 2).

### Task 4: Scoring script (~1 hour)

`score_candidate.py`:
- Takes `candidate_features.csv` (output of Task 3) + path to a full-fit
  predictor pickle (output of Task 1)
- For each (benchmark, variant): predicts L + g
- Aggregates: by default, mean predicted peak_pck across benchmarks (use
  per-benchmark for "target a specific benchmark" mode)
- Outputs JSON: `{score: float, per_benchmark: {bench: pred}, ...}`

### Task 5: Search loop driver (~half a day to overnight scale)

Glue it together. Pseudocode:

```python
search = BayesianOptimization(  # or Optuna, random search, ...
    f=lambda theta: score_candidate(
        features=extract_features(
            generate(theta, sample_size=N_OPTIMAL)
        ),
        predictor=full_fit_model,
    ),
    space=theta_space,
)
search.run(n_iter=100)
best_theta = search.best()
```

Make sure each candidate's intermediate artifacts go to a unique dir so
you can audit the search trajectory afterwards.

**Reasonable starting search strategies:**
- Random search with 50-100 candidates — simplest, gives a Pareto baseline
- Bayesian optimization (skopt or optuna) — better at 50+ candidates
- Pre-defined grid over 2-3 key hyperparameters — easiest to interpret

### Task 6: Final validation training (paper section)

Once search converges:
1. Take the top-5 (or top-3) candidates from the search.
2. Generate full-size datasets for each.
3. Train full models (expensive — same compute as training one of the
   original 11 sources).
4. Report: actual peak_pck of top candidates vs baseline; predictor's
   ranking of those candidates vs actual ranking.

This is the paper's strongest practical claim — search-then-validate.

---

## Architecture sketch

```
[NEW]   scripts/transfer_analysis_v4/full_fit_predictor.py
        → CLI: trains predictor on all data, pickles to disk

[NEW]   scripts/transfer_analysis_v4/score_candidate.py
        → CLI: given candidate features + predictor, returns score(s)

[NEW]   scripts/transfer_analysis_v4/extract_candidate_features.py
        → CLI: given flow vector .npy path, computes 13-dim features
          against each cached benchmark

[NEW]   scripts/transfer_analysis_v4/search_loop.py  (or similar)
        → driver for whichever search strategy you pick

[REUSE] scripts/transfer_analysis_v3/compute_pairwise_self_distances.py
        → already knows how to compute the metrics; wrap don't rewrite

[REUSE] /mnt/nvme_1tb_b/coverage_vectors/
        → cached benchmark flow vectors are already here

[NEW]   /mnt/nvme_1tb_b/candidates/<run_id>/flow_vectors.npy
        → suggested path layout for generated candidate data
```

---

## Open decisions (need user input)

1. **Search target.** Are we optimizing for:
   - (a) General-purpose synthetic data → maximize mean peak_pck across all
     10 benchmarks?
   - (b) Targeted at a benchmark family → e.g., maximize peak_pck weighted
     toward KITTI variants?
   - (c) Pareto front across benchmarks → return multiple candidates with
     different trade-offs?

2. **Search algorithm.** Random / grid / Bayesian / something more
   sophisticated? Start with random for the validation step (lowest risk),
   then upgrade if needed.

3. **Hyperparameter space.** The user mentioned camera movement as the
   easiest axis. What's the parametrization? (e.g., translation magnitude,
   rotation rate, camera-to-object distance, trajectory shape...) The
   search space's dimensionality determines how many candidates you need.

4. **Generation budget.** How long does it take to generate one candidate
   dataset at the chosen sample size? This bounds the total search budget.

5. **Final-validation budget.** Top-K full training runs at the end —
   what's the K?

---

## Things to NOT do (lessons from previous sessions, see HANDOFF.md)

- **Don't add features for the candidate scoring** that aren't in the
  predictor's training distribution. The 13-dim flow self-distance set is
  what the predictor knows. Adding new features means retraining the
  predictor.
- **Don't fold L into the candidate ranking on LOTO.** L = cell_mean is
  constant across candidates within a context — it contributes zero to the
  ranking. The ranking comes from g. (This is the entire point of the
  within-estimator framework; see CLAIMS.md Claim 6.)
- **Don't trust the predictor before the validation step.** Predictor
  saturation in a narrow generation distribution is a real risk; skipping
  validation means you might be reporting noise.
- **Don't make this a 10-axis hyperparameter search on the first try.**
  Pick 1-2 axes, validate, then expand. Otherwise you'll burn compute on
  a search problem whose surface the predictor can't resolve.

---

## Suggested first day of work for a new agent

1. (~30 min) Read the docs listed at top.
2. (~1-2 hours) Implement Task 1 (full-fit mode). Verify by manually
   scoring an existing source held out — should match the LOTO numbers in
   `results_mixed/`.
3. (~2 hours) Implement Task 2 (sample-size stability). This bounds the
   search budget and is informative regardless of how the rest goes.
4. (Half day) Implement Task 3 (candidate feature extractor) and Task 4
   (scoring script).
5. (Half day) Implement Task 5 (search loop) using random search as the
   first algorithm.
6. Before reporting any search results, run the validation experiment
   (Task 6 abbreviated): 5-10 diverse candidates → predict → train →
   compare. If ρ < 0.2, stop and discuss with the user before proceeding.

---

## Communicate this back

Once tasks 1-5 are wired up and Task 2 has a recommended sample size, post
back to the user with:

- Selected hyperparameter axes
- Selected search algorithm
- Per-candidate cost (feature extraction time × predictor inference time)
- Predicted total search budget
- Validation experiment design (which candidates, how many)

Then proceed to validation. Don't run a 100-candidate search before
validation completes. The paper's credibility hinges on the
predicted-vs-actual ranking being well-supported.
