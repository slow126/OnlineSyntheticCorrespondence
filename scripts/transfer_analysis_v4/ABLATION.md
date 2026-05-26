# Transfer Analysis v4 — L / Feature Ablation Summary

Combining 12 result directories:

- **feature-subset=asym_only** — `results_fsub_asym_only`
- **feature-subset=coverage** — `results_fsub_coverage`
- **feature-subset=kl** — `results_fsub_kl`
- **feature-subset=mean_nn** — `results_fsub_mean_nn`
- **mixed** — `results_mixed`
- **symmetric_informed** — `results_symmetric_informed`
- **symmetric_uninformed** — `results_symmetric_uninformed`
- **targeted_informed (all features)** — `results_targeted_informed`
- **targeted_informed (targeted-subset=asym_only)** — `results_targeted_asym_only`
- **targeted_informed (targeted-subset=coverage)** — `results_targeted_coverage`
- **targeted_informed (targeted-subset=kl)** — `results_targeted_kl`
- **targeted_informed (targeted-subset=mean_nn)** — `results_targeted_mean_nn`

---

## Quick read

- **ctx_rho_g (ridge)** should be ~identical across L modes — g is L-invariant. Any drift is just rounding/sampling.
- **ctx_rho_g (ridge)** should change with `--feature-subset` if the dropped features carry signal. If it doesn't move much, ridge was already down-weighting them.
- **ctx_rho_L** is the *level-only* ranking score and changes a lot across L modes — that's the comparison the ablations are actually testing.
- **abs_r_Lg** is calibration. Lower under uniform L; comparable across informed variants.


---

# Target: `auc_normalized`


## 1. ridge ctx_rho_g — feature claim (should be L-invariant)

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | +0.278 | -0.393 | +0.019 | +0.356 | +0.196 | +0.034 | +0.181 | -0.249 | +0.028 |
| feature-subset=coverage | +0.278 | +0.159 | +0.019 | +0.370 | +0.102 | +0.034 | +0.172 | -0.253 | +0.028 |
| feature-subset=kl | +0.040 | -0.169 | +0.019 | +0.212 | +0.246 | +0.034 | -0.010 | -0.196 | +0.028 |
| feature-subset=mean_nn | +0.278 | -0.171 | +0.019 | +0.404 | -0.038 | +0.034 | +0.270 | -0.175 | +0.028 |
| mixed | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| symmetric_informed | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| symmetric_uninformed | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| targeted_informed (all features) | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| targeted_informed (targeted-subset=asym_only) | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| targeted_informed (targeted-subset=coverage) | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| targeted_informed (targeted-subset=kl) | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| targeted_informed (targeted-subset=mean_nn) | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |

## 2. symmetric / FID / W2 feature families — ridge ctx_rho_g

_No rows available yet. Re-run v4 after the corresponding feature columns land._

## 3. ctx_rho_L — level-only ranking ρ

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=coverage | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=kl | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=mean_nn | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| mixed | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| symmetric_informed | +0.020 | +0.087 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| symmetric_uninformed | -1.000 | -1.000 | -1.000 | +0.755 | +0.755 | +0.755 | -0.755 | -0.755 | -0.755 |
| targeted_informed (all features) | +0.047 | -0.056 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| targeted_informed (targeted-subset=asym_only) | +0.050 | -0.056 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| targeted_informed (targeted-subset=coverage) | +0.063 | -0.885 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| targeted_informed (targeted-subset=kl) | -0.079 | -0.066 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| targeted_informed (targeted-subset=mean_nn) | +0.107 | -0.569 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |

## 4. abs_r_Lg — pooled calibration

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | +0.797 | +0.674 | +0.766 | +0.744 | +0.777 | +0.459 | +0.222 | -0.073 | +0.147 |
| feature-subset=coverage | +0.793 | +0.766 | +0.766 | +0.757 | +0.791 | +0.459 | +0.282 | +0.153 | +0.147 |
| feature-subset=kl | +0.761 | +0.712 | +0.766 | +0.797 | +0.781 | +0.459 | +0.072 | +0.020 | +0.147 |
| feature-subset=mean_nn | +0.790 | +0.748 | +0.766 | +0.778 | +0.792 | +0.459 | +0.071 | +0.090 | +0.147 |
| mixed | +0.796 | +0.672 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| symmetric_informed | +0.782 | +0.723 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| symmetric_uninformed | +0.796 | +0.672 | +0.766 | +0.471 | +0.434 | +0.459 | +0.218 | -0.070 | +0.147 |
| targeted_informed (all features) | +0.717 | +0.630 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| targeted_informed (targeted-subset=asym_only) | +0.716 | +0.630 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| targeted_informed (targeted-subset=coverage) | +0.747 | +0.673 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| targeted_informed (targeted-subset=kl) | +0.708 | +0.629 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| targeted_informed (targeted-subset=mean_nn) | +0.720 | +0.686 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |

## 5. motion ridge ctx_rho_g — with 95% CIs

| ablation | LOTO | LOBO | JOINT |
|---|---|---|---|
| feature-subset=asym_only | +0.278 [-0.045, +0.537] | +0.356 [+0.250, +0.468] | +0.181 [+0.003, +0.312] |
| feature-subset=coverage | +0.278 [-0.061, +0.553] | +0.370 [+0.291, +0.456] | +0.172 [-0.002, +0.308] |
| feature-subset=kl | +0.040 [-0.272, +0.330] | +0.212 [+0.142, +0.298] | -0.010 [-0.163, +0.141] |
| feature-subset=mean_nn | +0.278 [-0.089, +0.517] | +0.404 [+0.324, +0.486] | +0.270 [+0.081, +0.386] |
| mixed | +0.271 [-0.079, +0.516] | +0.342 [+0.223, +0.471] | +0.166 [+0.005, +0.305] |
| symmetric_informed | +0.271 [-0.070, +0.533] | +0.342 [+0.221, +0.469] | +0.166 [+0.005, +0.322] |
| symmetric_uninformed | +0.271 [-0.057, +0.528] | +0.342 [+0.229, +0.465] | +0.166 [-0.003, +0.314] |
| targeted_informed (all features) | +0.271 [-0.061, +0.491] | +0.342 [+0.226, +0.471] | +0.166 [+0.008, +0.305] |
| targeted_informed (targeted-subset=asym_only) | +0.271 [-0.066, +0.521] | +0.342 [+0.214, +0.464] | +0.166 [-0.008, +0.308] |
| targeted_informed (targeted-subset=coverage) | +0.271 [-0.079, +0.515] | +0.342 [+0.227, +0.468] | +0.166 [+0.006, +0.303] |
| targeted_informed (targeted-subset=kl) | +0.271 [-0.061, +0.530] | +0.342 [+0.229, +0.475] | +0.166 [-0.017, +0.295] |
| targeted_informed (targeted-subset=mean_nn) | +0.271 [-0.031, +0.556] | +0.342 [+0.237, +0.469] | +0.166 [+0.014, +0.303] |

---

# Target: `peak_pck`


## 1. ridge ctx_rho_g — feature claim (should be L-invariant)

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | +0.515 | -0.271 | -0.126 | +0.450 | +0.077 | -0.077 | +0.324 | -0.191 | -0.097 |
| feature-subset=coverage | +0.321 | -0.151 | -0.126 | +0.387 | -0.216 | -0.077 | +0.240 | -0.287 | -0.097 |
| feature-subset=kl | -0.276 | -0.277 | -0.126 | +0.095 | +0.039 | -0.077 | -0.250 | -0.318 | -0.097 |
| feature-subset=mean_nn | +0.419 | -0.091 | -0.126 | +0.501 | +0.021 | -0.077 | +0.478 | -0.099 | -0.097 |
| mixed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| symmetric_informed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| symmetric_uninformed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| targeted_informed (all features) | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| targeted_informed (targeted-subset=asym_only) | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| targeted_informed (targeted-subset=coverage) | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| targeted_informed (targeted-subset=kl) | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| targeted_informed (targeted-subset=mean_nn) | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |

## 2. symmetric / FID / W2 feature families — ridge ctx_rho_g

_No rows available yet. Re-run v4 after the corresponding feature columns land._

## 3. ctx_rho_L — level-only ranking ρ

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=coverage | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=kl | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=mean_nn | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| mixed | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| symmetric_informed | +0.014 | +0.386 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| symmetric_uninformed | -1.000 | -1.000 | -1.000 | +0.540 | +0.540 | +0.540 | -0.540 | -0.540 | -0.540 |
| targeted_informed (all features) | +0.171 | +0.064 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| targeted_informed (targeted-subset=asym_only) | +0.147 | +0.064 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| targeted_informed (targeted-subset=coverage) | +0.214 | -0.850 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| targeted_informed (targeted-subset=kl) | +0.059 | +0.064 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| targeted_informed (targeted-subset=mean_nn) | +0.252 | -0.439 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |

## 4. abs_r_Lg — pooled calibration

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | +0.851 | +0.750 | +0.819 | +0.737 | +0.755 | +0.402 | +0.383 | -0.211 | +0.148 |
| feature-subset=coverage | +0.855 | +0.819 | +0.819 | +0.743 | +0.764 | +0.402 | +0.444 | +0.154 | +0.148 |
| feature-subset=kl | +0.806 | +0.759 | +0.819 | +0.771 | +0.765 | +0.402 | +0.008 | -0.045 | +0.148 |
| feature-subset=mean_nn | +0.822 | +0.805 | +0.819 | +0.758 | +0.769 | +0.402 | +0.059 | +0.037 | +0.148 |
| mixed | +0.851 | +0.751 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| symmetric_informed | +0.838 | +0.811 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| symmetric_uninformed | +0.851 | +0.751 | +0.819 | +0.417 | +0.368 | +0.402 | +0.379 | -0.216 | +0.148 |
| targeted_informed (all features) | +0.777 | +0.740 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| targeted_informed (targeted-subset=asym_only) | +0.774 | +0.740 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| targeted_informed (targeted-subset=coverage) | +0.837 | +0.752 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| targeted_informed (targeted-subset=kl) | +0.813 | +0.737 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| targeted_informed (targeted-subset=mean_nn) | +0.811 | +0.770 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |

## 5. motion ridge ctx_rho_g — with 95% CIs

| ablation | LOTO | LOBO | JOINT |
|---|---|---|---|
| feature-subset=asym_only | +0.515 [+0.232, +0.652] | +0.450 [+0.369, +0.530] | +0.324 [+0.137, +0.452] |
| feature-subset=coverage | +0.321 [+0.026, +0.556] | +0.387 [+0.278, +0.498] | +0.240 [+0.066, +0.390] |
| feature-subset=kl | -0.276 [-0.529, +0.056] | +0.095 [+0.002, +0.199] | -0.250 [-0.377, -0.036] |
| feature-subset=mean_nn | +0.419 [+0.194, +0.545] | +0.501 [+0.433, +0.574] | +0.478 [+0.302, +0.567] |
| mixed | +0.508 [+0.210, +0.652] | +0.448 [+0.368, +0.533] | +0.321 [+0.134, +0.453] |
| symmetric_informed | +0.508 [+0.214, +0.656] | +0.448 [+0.371, +0.536] | +0.321 [+0.111, +0.445] |
| symmetric_uninformed | +0.508 [+0.239, +0.646] | +0.448 [+0.366, +0.522] | +0.321 [+0.149, +0.444] |
| targeted_informed (all features) | +0.508 [+0.230, +0.666] | +0.448 [+0.368, +0.533] | +0.321 [+0.117, +0.452] |
| targeted_informed (targeted-subset=asym_only) | +0.508 [+0.197, +0.649] | +0.448 [+0.367, +0.531] | +0.321 [+0.135, +0.453] |
| targeted_informed (targeted-subset=coverage) | +0.508 [+0.223, +0.656] | +0.448 [+0.366, +0.534] | +0.321 [+0.138, +0.437] |
| targeted_informed (targeted-subset=kl) | +0.508 [+0.234, +0.656] | +0.448 [+0.362, +0.534] | +0.321 [+0.139, +0.435] |
| targeted_informed (targeted-subset=mean_nn) | +0.508 [+0.228, +0.650] | +0.448 [+0.367, +0.532] | +0.321 [+0.131, +0.462] |

---

## Files referenced

- `scripts/transfer_analysis_v4/results_fsub_asym_only/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_asym_only/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_coverage/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_coverage/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_kl/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_kl/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_mean_nn/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_mean_nn/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_symmetric_informed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_symmetric_informed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_symmetric_uninformed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_symmetric_uninformed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_targeted_asym_only/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_targeted_asym_only/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_targeted_coverage/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_targeted_coverage/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_targeted_informed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_targeted_informed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_targeted_kl/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_targeted_kl/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_targeted_mean_nn/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_targeted_mean_nn/summary.csv` — long-form metrics