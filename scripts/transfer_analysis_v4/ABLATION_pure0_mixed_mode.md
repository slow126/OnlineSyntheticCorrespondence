# Transfer Analysis v4 — L / Feature Ablation Summary

Combining 17 result directories:

- **results_pure0_density_idw** — `results_pure0_density_idw`
- **results_pure0_eb_shrunk** — `results_pure0_eb_shrunk`
- **results_pure0_fsub_asym_only** — `results_pure0_fsub_asym_only`
- **results_pure0_fsub_coverage** — `results_pure0_fsub_coverage`
- **results_pure0_fsub_eps_16px** — `results_pure0_fsub_eps_16px`
- **results_pure0_fsub_eps_1px** — `results_pure0_fsub_eps_1px`
- **results_pure0_fsub_eps_4px** — `results_pure0_fsub_eps_4px`
- **results_pure0_fsub_kl** — `results_pure0_fsub_kl`
- **results_pure0_fsub_kl_k20** — `results_pure0_fsub_kl_k20`
- **results_pure0_fsub_kl_k5** — `results_pure0_fsub_kl_k5`
- **results_pure0_fsub_mean_nn** — `results_pure0_fsub_mean_nn`
- **results_pure0_fsub_mean_nn_asym** — `results_pure0_fsub_mean_nn_asym`
- **results_pure0_fsub_mean_nn_sym** — `results_pure0_fsub_mean_nn_sym`
- **results_pure0_mixed** — `results_pure0_mixed`
- **results_pure0_symmetric_informed** — `results_pure0_symmetric_informed`
- **results_pure0_symmetric_uninformed** — `results_pure0_symmetric_uninformed`
- **results_pure0_targeted_informed** — `results_pure0_targeted_informed`

---

## Quick read

- **ctx_rho_g (ridge)** should be ~identical across L modes — g is L-invariant. Any drift is just rounding/sampling.
- **ctx_rho_g (ridge)** should change with `--feature-subset` if the dropped features carry signal. If it doesn't move much, ridge was already down-weighting them.
- **ctx_rho_L** is the *level-only* ranking score and changes a lot across L modes — that's the comparison the ablations are actually testing.
- **abs_r_Lg** is calibration. Lower under uniform L; comparable across informed variants.


---

# Target: `peak_pck`


## 1. ridge ctx_rho_g — feature claim (should be L-invariant)

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| results_pure0_density_idw | +0.155 | -0.273 | -0.009 | +0.287 | +0.040 | -0.026 | +0.052 | -0.301 | -0.022 |
| results_pure0_eb_shrunk | +0.155 | -0.273 | -0.009 | +0.287 | +0.040 | -0.026 | +0.052 | -0.301 | -0.022 |
| results_pure0_fsub_asym_only | +0.165 | -0.231 | -0.009 | +0.288 | +0.064 | -0.026 | +0.080 | -0.294 | -0.022 |
| results_pure0_fsub_coverage | +0.059 | -0.023 | -0.009 | +0.257 | -0.115 | -0.026 | +0.073 | -0.212 | -0.022 |
| results_pure0_fsub_eps_16px | +0.288 | +0.286 | -0.009 | +0.334 | +0.081 | -0.026 | +0.147 | +0.007 | -0.022 |
| results_pure0_fsub_eps_1px | +0.180 | -0.003 | -0.009 | +0.341 | -0.162 | -0.026 | +0.085 | -0.138 | -0.022 |
| results_pure0_fsub_eps_4px | +0.069 | +0.158 | -0.009 | +0.187 | -0.078 | -0.026 | +0.060 | +0.162 | -0.022 |
| results_pure0_fsub_kl | -0.291 | -0.268 | -0.009 | +0.066 | +0.022 | -0.026 | -0.216 | -0.338 | -0.022 |
| results_pure0_fsub_kl_k20 | -0.200 | -0.230 | -0.009 | +0.048 | +0.063 | -0.026 | -0.214 | -0.328 | -0.022 |
| results_pure0_fsub_kl_k5 | -0.188 | -0.161 | -0.009 | +0.046 | +0.043 | -0.026 | -0.211 | -0.304 | -0.022 |
| results_pure0_fsub_mean_nn | +0.100 | -0.126 | -0.009 | +0.279 | +0.006 | -0.026 | +0.096 | -0.193 | -0.022 |
| results_pure0_fsub_mean_nn_asym | +0.347 | -0.101 | -0.009 | +0.309 | -0.010 | -0.026 | +0.283 | -0.207 | -0.022 |
| results_pure0_fsub_mean_nn_sym | +0.346 | -0.189 | -0.009 | +0.294 | -0.022 | -0.026 | +0.312 | -0.202 | -0.022 |
| results_pure0_mixed | +0.155 | -0.273 | -0.009 | +0.287 | +0.040 | -0.026 | +0.052 | -0.301 | -0.022 |
| results_pure0_symmetric_informed | +0.155 | -0.273 | -0.009 | +0.287 | +0.040 | -0.026 | +0.052 | -0.301 | -0.022 |
| results_pure0_symmetric_uninformed | +0.155 | -0.273 | -0.009 | +0.287 | +0.040 | -0.026 | +0.052 | -0.301 | -0.022 |
| results_pure0_targeted_informed | +0.155 | -0.273 | -0.009 | +0.287 | +0.040 | -0.026 | +0.052 | -0.301 | -0.022 |

## 2. symmetric / FID / W2 feature families — ridge ctx_rho_g

| ablation | split | motion_sym | motion_fid | motion_w2 | motion_mmd | appearance_sym | appearance_fid | appearance_w2 | appearance_mmd | appearance_nullk |
|---|---|---|---|---|---|---|---|---|---|---|
| results_pure0_density_idw | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_density_idw | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_density_idw | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_eb_shrunk | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_eb_shrunk | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_eb_shrunk | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_asym_only | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_asym_only | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_asym_only | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_coverage | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_coverage | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_coverage | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_eps_16px | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_eps_16px | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_eps_16px | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_eps_1px | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_eps_1px | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_eps_1px | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_eps_4px | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_eps_4px | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_eps_4px | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_kl | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_kl | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_kl | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_kl_k20 | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_kl_k20 | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_kl_k20 | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_kl_k5 | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_kl_k5 | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_kl_k5 | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_mean_nn | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_mean_nn | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_mean_nn | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_mean_nn_asym | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_mean_nn_asym | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_mean_nn_asym | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_fsub_mean_nn_sym | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_fsub_mean_nn_sym | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_fsub_mean_nn_sym | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_mixed | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_mixed | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_mixed | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_symmetric_informed | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_symmetric_informed | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_symmetric_informed | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_symmetric_uninformed | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_symmetric_uninformed | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_symmetric_uninformed | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |
| results_pure0_targeted_informed | LOTO | +0.163 | +0.173 | +0.235 | +0.022 | +0.077 | -0.208 | +0.046 | -0.234 | -0.140 |
| results_pure0_targeted_informed | LOBO | +0.323 | +0.305 | +0.322 | +0.164 | +0.092 | +0.106 | -0.027 | -0.160 | +0.035 |
| results_pure0_targeted_informed | JOINT | +0.108 | +0.129 | +0.154 | +0.080 | -0.014 | -0.200 | -0.148 | -0.234 | -0.170 |

## 3. ctx_rho_L — level-only ranking ρ

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| results_pure0_density_idw | +0.142 | +0.142 | -1.000 | +0.661 | +0.661 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_eb_shrunk | -0.977 | -0.977 | -0.977 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_asym_only | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_coverage | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_eps_16px | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_eps_1px | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_eps_4px | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_kl | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_kl_k20 | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_kl_k5 | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_mean_nn | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_mean_nn_asym | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_fsub_mean_nn_sym | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_mixed | -1.000 | -1.000 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_symmetric_informed | -0.154 | +0.044 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_symmetric_uninformed | -1.000 | -1.000 | -1.000 | +0.531 | +0.531 | +0.531 | -0.531 | -0.531 | -0.531 |
| results_pure0_targeted_informed | -0.121 | -0.172 | -1.000 | +0.473 | +0.569 | +0.531 | -0.531 | -0.531 | -0.531 |

## 4. abs_r_Lg — pooled calibration

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| results_pure0_density_idw | +0.884 | +0.829 | +0.908 | +0.767 | +0.766 | +0.277 | +0.285 | -0.088 | +0.159 |
| results_pure0_eb_shrunk | +0.919 | +0.868 | +0.908 | +0.729 | +0.729 | +0.277 | +0.285 | -0.088 | +0.159 |
| results_pure0_fsub_asym_only | +0.921 | +0.874 | +0.908 | +0.728 | +0.724 | +0.277 | +0.302 | -0.061 | +0.159 |
| results_pure0_fsub_coverage | +0.922 | +0.907 | +0.908 | +0.730 | +0.725 | +0.277 | +0.318 | +0.163 | +0.159 |
| results_pure0_fsub_eps_16px | +0.915 | +0.908 | +0.908 | +0.737 | +0.729 | +0.277 | +0.199 | +0.156 | +0.159 |
| results_pure0_fsub_eps_1px | +0.917 | +0.908 | +0.908 | +0.751 | +0.728 | +0.277 | +0.295 | +0.161 | +0.159 |
| results_pure0_fsub_eps_4px | +0.915 | +0.908 | +0.908 | +0.742 | +0.716 | +0.277 | +0.368 | +0.162 | +0.159 |
| results_pure0_fsub_kl | +0.902 | +0.879 | +0.908 | +0.751 | +0.727 | +0.277 | +0.109 | +0.067 | +0.159 |
| results_pure0_fsub_kl_k20 | +0.906 | +0.901 | +0.908 | +0.752 | +0.728 | +0.277 | +0.155 | +0.121 | +0.159 |
| results_pure0_fsub_kl_k5 | +0.906 | +0.902 | +0.908 | +0.752 | +0.728 | +0.277 | +0.155 | +0.132 | +0.159 |
| results_pure0_fsub_mean_nn | +0.907 | +0.899 | +0.908 | +0.741 | +0.728 | +0.277 | +0.111 | +0.100 | +0.159 |
| results_pure0_fsub_mean_nn_asym | +0.907 | +0.899 | +0.908 | +0.742 | +0.728 | +0.277 | +0.108 | +0.096 | +0.159 |
| results_pure0_fsub_mean_nn_sym | +0.911 | +0.904 | +0.908 | +0.742 | +0.728 | +0.277 | +0.136 | +0.152 | +0.159 |
| results_pure0_mixed | +0.920 | +0.869 | +0.908 | +0.729 | +0.729 | +0.277 | +0.285 | -0.088 | +0.159 |
| results_pure0_symmetric_informed | +0.906 | +0.893 | +0.908 | +0.729 | +0.729 | +0.277 | +0.285 | -0.088 | +0.159 |
| results_pure0_symmetric_uninformed | +0.920 | +0.869 | +0.908 | +0.296 | +0.262 | +0.277 | +0.285 | -0.088 | +0.159 |
| results_pure0_targeted_informed | +0.872 | +0.850 | +0.908 | +0.729 | +0.729 | +0.277 | +0.285 | -0.088 | +0.159 |

## 5. motion ridge ctx_rho_g — with 95% CIs

| ablation | LOTO | LOBO | JOINT |
|---|---|---|---|
| results_pure0_density_idw | +0.155 | +0.287 | +0.052 |
| results_pure0_eb_shrunk | +0.155 | +0.287 | +0.052 |
| results_pure0_fsub_asym_only | +0.165 | +0.288 | +0.080 |
| results_pure0_fsub_coverage | +0.059 | +0.257 | +0.073 |
| results_pure0_fsub_eps_16px | +0.288 | +0.334 | +0.147 |
| results_pure0_fsub_eps_1px | +0.180 | +0.341 | +0.085 |
| results_pure0_fsub_eps_4px | +0.069 | +0.187 | +0.060 |
| results_pure0_fsub_kl | -0.291 | +0.066 | -0.216 |
| results_pure0_fsub_kl_k20 | -0.200 | +0.048 | -0.214 |
| results_pure0_fsub_kl_k5 | -0.188 | +0.046 | -0.211 |
| results_pure0_fsub_mean_nn | +0.100 | +0.279 | +0.096 |
| results_pure0_fsub_mean_nn_asym | +0.347 | +0.309 | +0.283 |
| results_pure0_fsub_mean_nn_sym | +0.346 | +0.294 | +0.312 |
| results_pure0_mixed | +0.155 | +0.287 | +0.052 |
| results_pure0_symmetric_informed | +0.155 | +0.287 | +0.052 |
| results_pure0_symmetric_uninformed | +0.155 | +0.287 | +0.052 |
| results_pure0_targeted_informed | +0.155 | +0.287 | +0.052 |

---

## Files referenced

- `scripts/transfer_analysis_v4/results_pure0_density_idw/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_density_idw/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_eb_shrunk/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_eb_shrunk/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_asym_only/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_asym_only/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_coverage/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_coverage/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_eps_16px/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_eps_16px/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_eps_1px/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_eps_1px/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_eps_4px/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_eps_4px/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_kl/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_kl/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_kl_k20/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_kl_k20/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_kl_k5/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_kl_k5/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_mean_nn/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_mean_nn/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_mean_nn_asym/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_mean_nn_asym/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_fsub_mean_nn_sym/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_fsub_mean_nn_sym/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_symmetric_informed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_symmetric_informed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_symmetric_uninformed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_symmetric_uninformed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_pure0_targeted_informed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_pure0_targeted_informed/summary.csv` — long-form metrics