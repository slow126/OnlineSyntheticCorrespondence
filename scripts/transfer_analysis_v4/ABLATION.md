# Transfer Analysis v4 — L / Feature Ablation Summary

Combining 17 result directories:

- **feature-subset=asym_only** — `results_fsub_asym_only`
- **feature-subset=coverage** — `results_fsub_coverage`
- **feature-subset=eps_16px** — `results_fsub_eps_16px`
- **feature-subset=eps_1px** — `results_fsub_eps_1px`
- **feature-subset=eps_4px** — `results_fsub_eps_4px`
- **feature-subset=kl** — `results_fsub_kl`
- **feature-subset=kl_k20** — `results_fsub_kl_k20`
- **feature-subset=kl_k5** — `results_fsub_kl_k5`
- **feature-subset=mean_nn** — `results_fsub_mean_nn`
- **feature-subset=mean_nn_asym** — `results_fsub_mean_nn_asym`
- **feature-subset=mean_nn_sym** — `results_fsub_mean_nn_sym`
- **mixed** — `results_mixed`
- **results_density_idw** — `results_density_idw`
- **results_eb_shrunk** — `results_eb_shrunk`
- **symmetric_informed** — `results_symmetric_informed`
- **symmetric_uninformed** — `results_symmetric_uninformed`
- **targeted_informed (all features)** — `results_targeted_informed`

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
| feature-subset=coverage | +0.278 | +0.159 | +0.019 | +0.370 | +0.102 | +0.034 | +0.172 | -0.253 | +0.028 |
| feature-subset=eps_1px | +0.157 | +0.123 | +0.019 | +0.337 | -0.238 | +0.034 | +0.159 | -0.262 | +0.028 |
| feature-subset=eps_4px | +0.178 | +0.364 | +0.019 | +0.282 | -0.181 | +0.034 | +0.169 | -0.161 | +0.028 |
| feature-subset=mean_nn | +0.278 | -0.171 | +0.019 | +0.404 | -0.038 | +0.034 | +0.270 | -0.175 | +0.028 |
| feature-subset=mean_nn_asym | +0.286 | -0.177 | +0.019 | +0.402 | -0.029 | +0.034 | +0.273 | -0.177 | +0.028 |
| feature-subset=mean_nn_sym | +0.373 | -0.139 | +0.019 | +0.414 | +0.012 | +0.034 | +0.322 | -0.165 | +0.028 |
| mixed | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| results_density_idw | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| results_eb_shrunk | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| symmetric_informed | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| symmetric_uninformed | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |
| targeted_informed (all features) | +0.271 | -0.393 | +0.019 | +0.342 | +0.196 | +0.034 | +0.166 | -0.244 | +0.028 |

## 2. symmetric / FID / W2 feature families — ridge ctx_rho_g

| ablation | split | motion_sym | motion_fid | motion_w2 | motion_mmd | appearance_sym | appearance_fid | appearance_w2 | appearance_mmd | appearance_nullk |
|---|---|---|---|---|---|---|---|---|---|---|
| mixed | LOTO | +0.201 [-0.041, +0.415] | +0.308 [-0.030, +0.486] | +0.337 [+0.016, +0.509] | -0.586 [-0.700, -0.377] | -0.194 [-0.466, +0.195] | -0.108 [-0.400, +0.253] | -0.080 [-0.408, +0.277] | -0.386 [-0.514, -0.069] | -0.055 [-0.366, +0.262] |
| mixed | LOBO | +0.395 [+0.363, +0.427] | +0.344 [+0.268, +0.425] | +0.365 [+0.289, +0.458] | +0.170 [+0.102, +0.235] | +0.058 [-0.055, +0.165] | +0.030 [-0.088, +0.129] | +0.012 [-0.093, +0.116] | +0.306 [+0.203, +0.406] | +0.109 [+0.012, +0.205] |
| mixed | JOINT | +0.285 [+0.131, +0.430] | +0.295 [+0.123, +0.413] | +0.357 [+0.192, +0.461] | -0.076 [-0.260, +0.105] | -0.202 [-0.338, -0.026] | -0.106 [-0.254, +0.049] | -0.080 [-0.232, +0.081] | -0.305 [-0.414, -0.135] | -0.003 [-0.175, +0.150] |
| results_density_idw | LOTO | +0.201 [-0.035, +0.394] | +0.308 [+0.021, +0.490] | +0.337 [+0.037, +0.516] | -0.586 [-0.676, -0.306] | -0.194 [-0.478, +0.175] | -0.108 [-0.434, +0.225] | -0.080 [-0.380, +0.233] | -0.386 [-0.509, -0.090] | -0.055 [-0.369, +0.232] |
| results_density_idw | LOBO | +0.395 [+0.361, +0.430] | +0.344 [+0.272, +0.427] | +0.365 [+0.275, +0.450] | +0.170 [+0.100, +0.233] | +0.058 [-0.044, +0.170] | +0.030 [-0.074, +0.127] | +0.012 [-0.105, +0.124] | +0.306 [+0.211, +0.421] | +0.109 [-0.006, +0.209] |
| results_density_idw | JOINT | +0.285 [+0.103, +0.410] | +0.295 [+0.123, +0.416] | +0.357 [+0.182, +0.460] | -0.076 [-0.221, +0.110] | -0.202 [-0.337, -0.031] | -0.106 [-0.266, +0.078] | -0.080 [-0.236, +0.092] | -0.305 [-0.388, -0.149] | -0.003 [-0.152, +0.158] |
| results_eb_shrunk | LOTO | +0.201 [-0.063, +0.394] | +0.308 [+0.034, +0.506] | +0.337 [+0.024, +0.517] | -0.586 [-0.688, -0.340] | -0.194 [-0.469, +0.150] | -0.108 [-0.394, +0.252] | -0.080 [-0.372, +0.320] | -0.386 [-0.508, -0.142] | -0.055 [-0.359, +0.270] |
| results_eb_shrunk | LOBO | +0.395 [+0.366, +0.429] | +0.344 [+0.267, +0.436] | +0.365 [+0.299, +0.447] | +0.170 [+0.109, +0.246] | +0.058 [-0.055, +0.148] | +0.030 [-0.077, +0.115] | +0.012 [-0.104, +0.121] | +0.306 [+0.209, +0.414] | +0.109 [-0.006, +0.212] |
| results_eb_shrunk | JOINT | +0.285 [+0.117, +0.437] | +0.295 [+0.120, +0.421] | +0.357 [+0.175, +0.465] | -0.076 [-0.250, +0.098] | -0.202 [-0.347, -0.023] | -0.106 [-0.279, +0.079] | -0.080 [-0.218, +0.085] | -0.305 [-0.412, -0.141] | -0.003 [-0.155, +0.143] |
| symmetric_informed | LOTO | +0.201 [-0.064, +0.412] | +0.308 [-0.004, +0.498] | +0.337 [+0.028, +0.522] | -0.586 [-0.677, -0.355] | -0.194 [-0.480, +0.156] | -0.108 [-0.438, +0.242] | -0.080 [-0.401, +0.269] | -0.386 [-0.527, -0.082] | -0.055 [-0.357, +0.277] |
| symmetric_informed | LOBO | +0.395 [+0.359, +0.434] | +0.344 [+0.272, +0.420] | +0.365 [+0.283, +0.456] | +0.170 [+0.110, +0.232] | +0.058 [-0.070, +0.175] | +0.030 [-0.075, +0.128] | +0.012 [-0.120, +0.123] | +0.306 [+0.213, +0.418] | +0.109 [+0.020, +0.220] |
| symmetric_informed | JOINT | +0.285 [+0.088, +0.402] | +0.295 [+0.137, +0.424] | +0.357 [+0.188, +0.472] | -0.076 [-0.253, +0.112] | -0.202 [-0.333, -0.013] | -0.106 [-0.252, +0.060] | -0.080 [-0.221, +0.090] | -0.305 [-0.399, -0.144] | -0.003 [-0.177, +0.169] |
| symmetric_uninformed | LOTO | +0.201 [-0.114, +0.408] | +0.308 [+0.023, +0.501] | +0.337 [+0.013, +0.513] | -0.586 [-0.692, -0.337] | -0.194 [-0.467, +0.192] | -0.108 [-0.420, +0.244] | -0.080 [-0.399, +0.310] | -0.386 [-0.515, -0.051] | -0.055 [-0.346, +0.289] |
| symmetric_uninformed | LOBO | +0.395 [+0.354, +0.431] | +0.344 [+0.260, +0.428] | +0.365 [+0.286, +0.459] | +0.170 [+0.105, +0.233] | +0.058 [-0.045, +0.152] | +0.030 [-0.080, +0.119] | +0.012 [-0.122, +0.124] | +0.306 [+0.211, +0.416] | +0.109 [-0.007, +0.227] |
| symmetric_uninformed | JOINT | +0.285 [+0.107, +0.406] | +0.295 [+0.127, +0.418] | +0.357 [+0.191, +0.489] | -0.076 [-0.261, +0.101] | -0.202 [-0.335, -0.042] | -0.106 [-0.266, +0.062] | -0.080 [-0.232, +0.092] | -0.305 [-0.398, -0.149] | -0.003 [-0.162, +0.157] |
| targeted_informed (all features) | LOTO | +0.201 [-0.058, +0.389] | +0.308 [+0.018, +0.501] | +0.337 [+0.023, +0.509] | -0.586 [-0.688, -0.335] | -0.194 [-0.496, +0.140] | -0.108 [-0.416, +0.250] | -0.080 [-0.411, +0.236] | -0.386 [-0.514, -0.103] | -0.055 [-0.365, +0.305] |
| targeted_informed (all features) | LOBO | +0.395 [+0.354, +0.431] | +0.344 [+0.275, +0.436] | +0.365 [+0.279, +0.444] | +0.170 [+0.108, +0.233] | +0.058 [-0.051, +0.160] | +0.030 [-0.085, +0.134] | +0.012 [-0.093, +0.118] | +0.306 [+0.208, +0.413] | +0.109 [+0.012, +0.229] |
| targeted_informed (all features) | JOINT | +0.285 [+0.106, +0.431] | +0.295 [+0.133, +0.441] | +0.357 [+0.172, +0.467] | -0.076 [-0.236, +0.121] | -0.202 [-0.314, -0.043] | -0.106 [-0.262, +0.089] | -0.080 [-0.243, +0.069] | -0.305 [-0.421, -0.150] | -0.003 [-0.153, +0.155] |

## 3. ctx_rho_L — level-only ranking ρ

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=coverage | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=eps_1px | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=eps_4px | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=mean_nn | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=mean_nn_asym | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| feature-subset=mean_nn_sym | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| mixed | -1.000 | -1.000 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| results_density_idw | -0.306 | -0.306 | -1.000 | +0.764 | +0.764 | +0.755 | -0.755 | -0.755 | -0.755 |
| results_eb_shrunk | -0.926 | -0.926 | -0.926 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| symmetric_informed | +0.020 | +0.087 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |
| symmetric_uninformed | -1.000 | -1.000 | -1.000 | +0.755 | +0.755 | +0.755 | -0.755 | -0.755 | -0.755 |
| targeted_informed (all features) | +0.047 | -0.056 | -1.000 | +0.719 | +0.733 | +0.755 | -0.755 | -0.755 | -0.755 |

## 4. abs_r_Lg — pooled calibration

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=coverage | +0.793 | +0.766 | +0.766 | +0.757 | +0.791 | +0.459 | +0.282 | +0.153 | +0.147 |
| feature-subset=eps_1px | +0.762 | +0.767 | +0.766 | +0.800 | +0.790 | +0.459 | +0.290 | +0.150 | +0.147 |
| feature-subset=eps_4px | +0.786 | +0.767 | +0.766 | +0.787 | +0.790 | +0.459 | +0.468 | +0.152 | +0.147 |
| feature-subset=mean_nn | +0.790 | +0.748 | +0.766 | +0.778 | +0.792 | +0.459 | +0.071 | +0.090 | +0.147 |
| feature-subset=mean_nn_asym | +0.791 | +0.747 | +0.766 | +0.779 | +0.792 | +0.459 | +0.067 | +0.093 | +0.147 |
| feature-subset=mean_nn_sym | +0.793 | +0.755 | +0.766 | +0.782 | +0.790 | +0.459 | +0.116 | +0.086 | +0.147 |
| mixed | +0.796 | +0.672 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| results_density_idw | +0.693 | +0.556 | +0.766 | +0.783 | +0.789 | +0.459 | +0.218 | -0.070 | +0.147 |
| results_eb_shrunk | +0.795 | +0.664 | +0.764 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| symmetric_informed | +0.782 | +0.723 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |
| symmetric_uninformed | +0.796 | +0.672 | +0.766 | +0.471 | +0.434 | +0.459 | +0.218 | -0.070 | +0.147 |
| targeted_informed (all features) | +0.717 | +0.630 | +0.766 | +0.744 | +0.778 | +0.459 | +0.218 | -0.070 | +0.147 |

## 5. motion ridge ctx_rho_g — with 95% CIs

| ablation | LOTO | LOBO | JOINT |
|---|---|---|---|
| feature-subset=coverage | +0.278 [-0.008, +0.554] | +0.370 [+0.284, +0.454] | +0.172 [+0.009, +0.307] |
| feature-subset=eps_1px | +0.157 [-0.179, +0.429] | +0.337 [+0.269, +0.407] | +0.159 [+0.000, +0.295] |
| feature-subset=eps_4px | +0.178 [-0.076, +0.391] | +0.282 [+0.196, +0.356] | +0.169 [+0.004, +0.314] |
| feature-subset=mean_nn | +0.278 [-0.084, +0.521] | +0.404 [+0.318, +0.486] | +0.270 [+0.090, +0.400] |
| feature-subset=mean_nn_asym | +0.286 [-0.094, +0.512] | +0.402 [+0.315, +0.490] | +0.273 [+0.101, +0.401] |
| feature-subset=mean_nn_sym | +0.373 [+0.090, +0.572] | +0.414 [+0.317, +0.504] | +0.322 [+0.149, +0.438] |
| mixed | +0.271 [-0.049, +0.542] | +0.342 [+0.239, +0.456] | +0.166 [-0.019, +0.302] |
| results_density_idw | +0.271 [-0.046, +0.507] | +0.342 [+0.232, +0.469] | +0.166 [+0.011, +0.292] |
| results_eb_shrunk | +0.271 [-0.087, +0.523] | +0.342 [+0.245, +0.478] | +0.166 [-0.002, +0.291] |
| symmetric_informed | +0.271 [-0.069, +0.529] | +0.342 [+0.235, +0.487] | +0.166 [-0.020, +0.301] |
| symmetric_uninformed | +0.271 [-0.020, +0.483] | +0.342 [+0.229, +0.460] | +0.166 [-0.014, +0.307] |
| targeted_informed (all features) | +0.271 [-0.049, +0.546] | +0.342 [+0.229, +0.470] | +0.166 [-0.004, +0.316] |

---

# Target: `peak_pck`


## 1. ridge ctx_rho_g — feature claim (should be L-invariant)

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | +0.513 | -0.273 | -0.054 | +0.449 | +0.080 | -0.019 | +0.324 | -0.195 | -0.012 |
| feature-subset=coverage | +0.321 | -0.151 | -0.126 | +0.387 | -0.216 | -0.077 | +0.240 | -0.287 | -0.097 |
| feature-subset=eps_16px | +0.344 | +0.323 | -0.054 | +0.476 | +0.356 | -0.019 | +0.247 | -0.241 | -0.012 |
| feature-subset=eps_1px | +0.431 | +0.263 | -0.126 | +0.515 | -0.293 | -0.077 | +0.302 | -0.212 | -0.097 |
| feature-subset=eps_4px | +0.224 | -0.060 | -0.126 | +0.290 | -0.134 | -0.077 | +0.200 | -0.191 | -0.097 |
| feature-subset=kl | -0.276 | -0.271 | -0.054 | +0.095 | +0.039 | -0.019 | -0.215 | -0.322 | -0.012 |
| feature-subset=kl_k20 | -0.139 | -0.244 | -0.054 | +0.182 | +0.066 | -0.019 | -0.119 | -0.270 | -0.012 |
| feature-subset=kl_k5 | -0.117 | -0.136 | -0.054 | +0.168 | +0.063 | -0.019 | -0.142 | -0.251 | -0.012 |
| feature-subset=mean_nn | +0.419 | -0.091 | -0.126 | +0.501 | +0.021 | -0.077 | +0.478 | -0.099 | -0.097 |
| feature-subset=mean_nn_asym | +0.447 | -0.087 | -0.126 | +0.511 | +0.023 | -0.077 | +0.475 | -0.134 | -0.097 |
| feature-subset=mean_nn_sym | +0.487 | -0.138 | -0.126 | +0.487 | -0.030 | -0.077 | +0.489 | -0.257 | -0.097 |
| mixed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| results_density_idw | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| results_eb_shrunk | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| symmetric_informed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| symmetric_uninformed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| targeted_informed (all features) | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |

## 2. symmetric / FID / W2 feature families — ridge ctx_rho_g

| ablation | split | motion_sym | motion_fid | motion_w2 | motion_mmd | appearance_sym | appearance_fid | appearance_w2 | appearance_mmd | appearance_nullk |
|---|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | LOTO | — | — | — | — | — | — | — | — | — |
| feature-subset=asym_only | LOBO | — | — | — | — | — | — | — | — | — |
| feature-subset=asym_only | JOINT | — | — | — | — | — | — | — | — | — |
| feature-subset=eps_16px | LOTO | — | — | — | — | — | — | — | — | — |
| feature-subset=eps_16px | LOBO | — | — | — | — | — | — | — | — | — |
| feature-subset=eps_16px | JOINT | — | — | — | — | — | — | — | — | — |
| feature-subset=kl | LOTO | — | — | — | — | — | — | — | — | — |
| feature-subset=kl | LOBO | — | — | — | — | — | — | — | — | — |
| feature-subset=kl | JOINT | — | — | — | — | — | — | — | — | — |
| feature-subset=kl_k20 | LOTO | — | — | — | — | — | — | — | — | — |
| feature-subset=kl_k20 | LOBO | — | — | — | — | — | — | — | — | — |
| feature-subset=kl_k20 | JOINT | — | — | — | — | — | — | — | — | — |
| feature-subset=kl_k5 | LOTO | — | — | — | — | — | — | — | — | — |
| feature-subset=kl_k5 | LOBO | — | — | — | — | — | — | — | — | — |
| feature-subset=kl_k5 | JOINT | — | — | — | — | — | — | — | — | — |
| mixed | LOTO | +0.436 [+0.181, +0.566] | +0.458 [+0.250, +0.579] | +0.474 [+0.273, +0.566] | -0.130 [-0.542, +0.279] | +0.070 [-0.251, +0.350] | -0.164 [-0.429, +0.175] | -0.188 [-0.448, +0.145] | +0.214 [-0.154, +0.479] | -0.067 [-0.346, +0.254] |
| mixed | LOBO | +0.536 [+0.476, +0.593] | +0.471 [+0.376, +0.579] | +0.478 [+0.388, +0.587] | +0.206 [+0.120, +0.285] | +0.144 [+0.001, +0.318] | +0.262 [+0.071, +0.474] | -0.021 [-0.238, +0.244] | +0.342 [+0.237, +0.454] | +0.054 [-0.059, +0.180] |
| mixed | JOINT | +0.432 [+0.253, +0.562] | +0.472 [+0.315, +0.570] | +0.494 [+0.325, +0.586] | +0.070 [-0.146, +0.250] | +0.022 [-0.116, +0.175] | -0.150 [-0.323, +0.030] | -0.312 [-0.418, -0.100] | +0.291 [+0.109, +0.423] | -0.057 [-0.192, +0.099] |
| results_density_idw | LOTO | +0.436 [+0.216, +0.550] | +0.458 [+0.267, +0.567] | +0.474 [+0.245, +0.576] | -0.130 [-0.517, +0.307] | +0.070 [-0.239, +0.358] | -0.164 [-0.424, +0.195] | -0.188 [-0.461, +0.170] | +0.214 [-0.188, +0.522] | -0.067 [-0.387, +0.241] |
| results_density_idw | LOBO | +0.536 [+0.477, +0.595] | +0.471 [+0.390, +0.571] | +0.478 [+0.382, +0.576] | +0.206 [+0.131, +0.289] | +0.144 [+0.007, +0.296] | +0.262 [+0.075, +0.488] | -0.021 [-0.243, +0.209] | +0.342 [+0.233, +0.460] | +0.054 [-0.059, +0.189] |
| results_density_idw | JOINT | +0.432 [+0.240, +0.553] | +0.472 [+0.308, +0.577] | +0.494 [+0.332, +0.592] | +0.070 [-0.117, +0.244] | +0.022 [-0.164, +0.177] | -0.150 [-0.307, +0.041] | -0.312 [-0.449, -0.133] | +0.291 [+0.103, +0.419] | -0.057 [-0.212, +0.120] |
| results_eb_shrunk | LOTO | +0.436 [+0.219, +0.568] | +0.458 [+0.249, +0.566] | +0.474 [+0.255, +0.572] | -0.130 [-0.526, +0.327] | +0.070 [-0.197, +0.359] | -0.164 [-0.412, +0.189] | -0.188 [-0.448, +0.194] | +0.214 [-0.133, +0.544] | -0.067 [-0.341, +0.248] |
| results_eb_shrunk | LOBO | +0.536 [+0.482, +0.597] | +0.471 [+0.389, +0.565] | +0.478 [+0.386, +0.580] | +0.206 [+0.119, +0.285] | +0.144 [-0.000, +0.302] | +0.262 [+0.099, +0.489] | -0.021 [-0.242, +0.221] | +0.342 [+0.224, +0.444] | +0.054 [-0.074, +0.176] |
| results_eb_shrunk | JOINT | +0.432 [+0.242, +0.552] | +0.472 [+0.293, +0.582] | +0.494 [+0.338, +0.581] | +0.070 [-0.119, +0.261] | +0.022 [-0.148, +0.195] | -0.150 [-0.308, +0.036] | -0.312 [-0.427, -0.136] | +0.291 [+0.106, +0.421] | -0.057 [-0.217, +0.112] |
| symmetric_informed | LOTO | +0.436 [+0.175, +0.555] | +0.458 [+0.233, +0.565] | +0.474 [+0.240, +0.573] | -0.130 [-0.517, +0.342] | +0.070 [-0.214, +0.381] | -0.164 [-0.426, +0.188] | -0.188 [-0.452, +0.128] | +0.214 [-0.108, +0.528] | -0.067 [-0.350, +0.245] |
| symmetric_informed | LOBO | +0.536 [+0.482, +0.593] | +0.471 [+0.382, +0.559] | +0.478 [+0.384, +0.578] | +0.206 [+0.131, +0.276] | +0.144 [-0.003, +0.326] | +0.262 [+0.087, +0.461] | -0.021 [-0.259, +0.221] | +0.342 [+0.245, +0.454] | +0.054 [-0.059, +0.180] |
| symmetric_informed | JOINT | +0.432 [+0.256, +0.549] | +0.472 [+0.301, +0.583] | +0.494 [+0.327, +0.575] | +0.070 [-0.152, +0.255] | +0.022 [-0.148, +0.188] | -0.150 [-0.299, +0.051] | -0.312 [-0.434, -0.134] | +0.291 [+0.135, +0.421] | -0.057 [-0.212, +0.116] |
| symmetric_uninformed | LOTO | +0.436 [+0.195, +0.555] | +0.458 [+0.255, +0.557] | +0.474 [+0.262, +0.579] | -0.130 [-0.549, +0.300] | +0.070 [-0.209, +0.412] | -0.164 [-0.413, +0.175] | -0.188 [-0.450, +0.160] | +0.214 [-0.189, +0.470] | -0.067 [-0.357, +0.265] |
| symmetric_uninformed | LOBO | +0.536 [+0.488, +0.593] | +0.471 [+0.380, +0.577] | +0.478 [+0.372, +0.585] | +0.206 [+0.120, +0.274] | +0.144 [+0.005, +0.324] | +0.262 [+0.055, +0.472] | -0.021 [-0.244, +0.224] | +0.342 [+0.248, +0.449] | +0.054 [-0.060, +0.169] |
| symmetric_uninformed | JOINT | +0.432 [+0.250, +0.543] | +0.472 [+0.310, +0.575] | +0.494 [+0.326, +0.582] | +0.070 [-0.101, +0.248] | +0.022 [-0.153, +0.180] | -0.150 [-0.316, +0.035] | -0.312 [-0.418, -0.137] | +0.291 [+0.103, +0.441] | -0.057 [-0.210, +0.113] |
| targeted_informed (all features) | LOTO | +0.436 [+0.197, +0.540] | +0.458 [+0.258, +0.567] | +0.474 [+0.248, +0.573] | -0.130 [-0.528, +0.258] | +0.070 [-0.249, +0.406] | -0.164 [-0.384, +0.160] | -0.188 [-0.441, +0.138] | +0.214 [-0.193, +0.501] | -0.067 [-0.329, +0.294] |
| targeted_informed (all features) | LOBO | +0.536 [+0.478, +0.592] | +0.471 [+0.385, +0.564] | +0.478 [+0.386, +0.586] | +0.206 [+0.119, +0.293] | +0.144 [+0.015, +0.309] | +0.262 [+0.078, +0.473] | -0.021 [-0.279, +0.214] | +0.342 [+0.234, +0.446] | +0.054 [-0.065, +0.184] |
| targeted_informed (all features) | JOINT | +0.432 [+0.243, +0.547] | +0.472 [+0.299, +0.578] | +0.494 [+0.314, +0.594] | +0.070 [-0.104, +0.243] | +0.022 [-0.147, +0.196] | -0.150 [-0.320, +0.056] | -0.312 [-0.431, -0.137] | +0.291 [+0.112, +0.415] | -0.057 [-0.199, +0.113] |

## 3. ctx_rho_L — level-only ranking ρ

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=coverage | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=eps_16px | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=eps_1px | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=eps_4px | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=kl | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=kl_k20 | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=kl_k5 | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=mean_nn | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=mean_nn_asym | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| feature-subset=mean_nn_sym | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| mixed | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_density_idw | -0.056 | -0.056 | -1.000 | +0.673 | +0.673 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_eb_shrunk | -0.947 | -0.947 | -0.947 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| symmetric_informed | +0.014 | +0.386 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| symmetric_uninformed | -1.000 | -1.000 | -1.000 | +0.540 | +0.540 | +0.540 | -0.540 | -0.540 | -0.540 |
| targeted_informed (all features) | +0.171 | +0.064 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |

## 4. abs_r_Lg — pooled calibration

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| feature-subset=asym_only | +0.887 | +0.782 | +0.855 | +0.715 | +0.739 | +0.337 | +0.432 | -0.219 | +0.167 |
| feature-subset=coverage | +0.855 | +0.819 | +0.819 | +0.743 | +0.764 | +0.402 | +0.444 | +0.154 | +0.148 |
| feature-subset=eps_16px | +0.876 | +0.856 | +0.855 | +0.731 | +0.757 | +0.337 | +0.237 | +0.159 | +0.167 |
| feature-subset=eps_1px | +0.841 | +0.820 | +0.819 | +0.772 | +0.770 | +0.402 | +0.422 | +0.149 | +0.148 |
| feature-subset=eps_4px | +0.840 | +0.821 | +0.819 | +0.760 | +0.750 | +0.402 | +0.527 | +0.150 | +0.148 |
| feature-subset=kl | +0.842 | +0.787 | +0.855 | +0.752 | +0.751 | +0.337 | +0.074 | -0.040 | +0.167 |
| feature-subset=kl_k20 | +0.853 | +0.841 | +0.855 | +0.754 | +0.755 | +0.337 | +0.159 | +0.096 | +0.167 |
| feature-subset=kl_k5 | +0.854 | +0.844 | +0.855 | +0.754 | +0.754 | +0.337 | +0.155 | +0.116 | +0.167 |
| feature-subset=mean_nn | +0.822 | +0.805 | +0.819 | +0.758 | +0.769 | +0.402 | +0.059 | +0.037 | +0.148 |
| feature-subset=mean_nn_asym | +0.822 | +0.806 | +0.819 | +0.757 | +0.769 | +0.402 | +0.054 | +0.011 | +0.148 |
| feature-subset=mean_nn_sym | +0.829 | +0.814 | +0.819 | +0.759 | +0.769 | +0.402 | +0.108 | +0.131 | +0.148 |
| mixed | +0.851 | +0.751 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| results_density_idw | +0.802 | +0.697 | +0.819 | +0.789 | +0.783 | +0.402 | +0.379 | -0.216 | +0.148 |
| results_eb_shrunk | +0.850 | +0.747 | +0.818 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| symmetric_informed | +0.838 | +0.811 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| symmetric_uninformed | +0.851 | +0.751 | +0.819 | +0.417 | +0.368 | +0.402 | +0.379 | -0.216 | +0.148 |
| targeted_informed (all features) | +0.777 | +0.740 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |

## 5. motion ridge ctx_rho_g — with 95% CIs

| ablation | LOTO | LOBO | JOINT |
|---|---|---|---|
| feature-subset=asym_only | — | — | — |
| feature-subset=coverage | +0.321 [+0.040, +0.556] | +0.387 [+0.280, +0.493] | +0.240 [+0.063, +0.380] |
| feature-subset=eps_16px | — | — | — |
| feature-subset=eps_1px | +0.431 [+0.171, +0.592] | +0.515 [+0.458, +0.575] | +0.302 [+0.119, +0.410] |
| feature-subset=eps_4px | +0.224 [-0.026, +0.450] | +0.290 [+0.196, +0.377] | +0.200 [+0.037, +0.348] |
| feature-subset=kl | — | — | — |
| feature-subset=kl_k20 | — | — | — |
| feature-subset=kl_k5 | — | — | — |
| feature-subset=mean_nn | +0.419 [+0.205, +0.534] | +0.501 [+0.436, +0.572] | +0.478 [+0.315, +0.585] |
| feature-subset=mean_nn_asym | +0.447 [+0.193, +0.560] | +0.511 [+0.458, +0.573] | +0.475 [+0.309, +0.556] |
| feature-subset=mean_nn_sym | +0.487 [+0.209, +0.595] | +0.487 [+0.407, +0.562] | +0.489 [+0.310, +0.576] |
| mixed | +0.508 [+0.201, +0.647] | +0.448 [+0.364, +0.536] | +0.321 [+0.141, +0.449] |
| results_density_idw | +0.508 [+0.228, +0.661] | +0.448 [+0.367, +0.531] | +0.321 [+0.109, +0.449] |
| results_eb_shrunk | +0.508 [+0.208, +0.649] | +0.448 [+0.362, +0.533] | +0.321 [+0.130, +0.457] |
| symmetric_informed | +0.508 [+0.260, +0.640] | +0.448 [+0.362, +0.534] | +0.321 [+0.130, +0.436] |
| symmetric_uninformed | +0.508 [+0.212, +0.653] | +0.448 [+0.374, +0.526] | +0.321 [+0.130, +0.459] |
| targeted_informed (all features) | +0.508 [+0.187, +0.652] | +0.448 [+0.367, +0.531] | +0.321 [+0.131, +0.451] |

---

## Files referenced

- `scripts/transfer_analysis_v4/results_density_idw/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_density_idw/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_eb_shrunk/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_eb_shrunk/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_asym_only/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_asym_only/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_coverage/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_coverage/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_eps_16px/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_eps_16px/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_eps_1px/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_eps_1px/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_eps_4px/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_eps_4px/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_kl/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_kl/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_kl_k20/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_kl_k20/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_kl_k5/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_kl_k5/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_mean_nn/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_mean_nn/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_mean_nn_asym/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_mean_nn_asym/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_fsub_mean_nn_sym/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_fsub_mean_nn_sym/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_symmetric_informed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_symmetric_informed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_symmetric_uninformed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_symmetric_uninformed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_targeted_informed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_targeted_informed/summary.csv` — long-form metrics