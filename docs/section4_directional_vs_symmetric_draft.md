# 4. Comparing Datasets: Directional vs Symmetric Distances

## 4.1 Motivation
Our goal is to quantify train/eval mismatch in a way that is useful for transfer. A single symmetric discrepancy value is often insufficient because transfer failures are directional:

- missing support: eval contains regions not represented in train
- extra mass: train contains regions irrelevant for eval

These two regimes can produce similar symmetric distances while implying different transfer behavior.

## 4.2 Directed kNN Discrepancy
Let train vectors be \(T=\{t_i\}_{i=1}^{n_T}\) and eval vectors be \(E=\{e_j\}_{j=1}^{n_E}\). We compute two directed nearest-neighbor distance sets:

\[
d_{E \to T}(j)=\min_i \|e_j-t_i\|,\quad
d_{T \to E}(i)=\min_j \|t_i-e_j\|.
\]

From these, we form directional summary statistics (mean, median, p90) and directional coverage over thresholds \(\epsilon\):

\[
C_{E \to T}(\epsilon)=\frac{1}{n_E}\sum_j \mathbf{1}[d_{E \to T}(j)\le\epsilon],\quad
C_{T \to E}(\epsilon)=\frac{1}{n_T}\sum_i \mathbf{1}[d_{T \to E}(i)\le\epsilon].
\]

Interpretation:

- \(E \to T\) probes missing support (eval regions unsupported by train)
- \(T \to E\) probes extra mass (train regions unsupported by eval)

In our flow-epsilon pipeline, we deliberately use raw fixed-\(\epsilon\) directional curves (no self-radius normalization), which directly parameterize the coverage-distance tradeoff.

## 4.3 Normalization Variants (qnorm/rnorm and radius ratios)
We also consider self-radius-normalized variants for representations where self-radii are available.

For train radius \(R_T\) and eval radius \(R_E\):

- qnorm (query-normalized): threshold by the query radius
- rnorm (reference-normalized): threshold by the reference radius

Operationally, this corresponds to features such as:

- `*_train_to_eval_mean_dist_over_radius_eval`
- `*_eval_to_train_mean_dist_over_radius_train`

plus the analogous median/p90 forms.

These are useful diagnostics, but empirically less stable in our setting when radii are very small. Ratio features can become heavy-tailed and sensitive to small denominator changes, which can dominate regression behavior.

## 4.4 Practical Metric Set Used in This Study
Our primary mixed-model runs use:

- flow directional epsilon features (`flow_eval_to_train_eps*`, `flow_train_to_eval_eps*`)
- directed DINO KL terms (`dino_eval_to_train_kl_div`, `dino_train_to_eval_kl_div`)
- motion-density summary (`hof_density_l2`)
- symmetric baselines (`flow_mmd`, `dino_mmd`)

This corresponds to the run:

- `analysis_comprehensive_runs/hof_motion_v3_nofamily/density_joint/leakage_free_combo_flow_eps_raw__dino_kl_k5__hof_density_l2__mmd`

which yields the section-level headline ranking metrics (LOTO rank Spearman \(\approx 0.23\), regret \(\approx 4.57\)).

## 4.5 Robustness Notes from Ablations
Across ablations, three patterns are consistent:

1. directional raw distance/coverage features are stable and interpretable
2. radius-normalized ratio variants can degrade rank behavior in some runs (especially when effective radii are small)
3. KL features are high-variance and direction-sensitive; sign and effect size can change under centering choices and family splits

This is consistent with known finite-sample behavior of kNN-density KL estimators and ratio-based statistics.

## 4.6 Symmetric Baseline and Failure-Mode Collapse
MMD is symmetric:

\[
\mathrm{MMD}(T,E)=\mathrm{MMD}(E,T).
\]

Therefore MMD cannot distinguish whether mismatch is caused by:

- eval having unsupported regions (under-coverage), or
- train having unsupported excess regions (extra mass).

Both can produce comparable symmetric discrepancy, despite opposite directional implications for transfer.

## 4.7 Figure 3 (Conceptual)
Figure 3 illustrates two toy two-mode regimes:

- Case A (under-coverage): high \(E \to T\), low/moderate \(T \to E\)
- Case B (extra mass): low/moderate \(E \to T\), high \(T \to E\)

The symmetric MMD value is nearly unchanged across cases, while directional distances swap. This visualizes why directional metrics are required to separate failure modes.

## 4.8 Recommendation
For model-selection and transfer prediction, we recommend:

- primary: directional flow epsilon curves and directional nearest-neighbor summaries
- secondary diagnostics: qnorm/rnorm and radius-ratio variants
- tertiary diagnostics: KL terms (reported with robustness caveats)
- always report both directions to avoid symmetric failure-mode collapse

