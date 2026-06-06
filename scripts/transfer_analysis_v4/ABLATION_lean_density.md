# Transfer Analysis v4 — L / Feature Ablation Summary

Combining 6 result directories:

- **results_lean_canon_mixed** — `results_lean_canon_mixed`
- **results_lean_dL1_mixed** — `results_lean_dL1_mixed`
- **results_lean_dL2_mixed** — `results_lean_dL2_mixed`
- **results_lean_dL3_mixed** — `results_lean_dL3_mixed`
- **results_lean_dL4_mixed** — `results_lean_dL4_mixed`
- **results_lean_dL5_mixed** — `results_lean_dL5_mixed`

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
| results_lean_canon_mixed | +0.508 | -0.254 | -0.126 | +0.448 | +0.074 | -0.077 | +0.321 | -0.202 | -0.097 |
| results_lean_dL1_mixed | +0.216 | -0.327 | -0.126 | +0.338 | +0.092 | -0.077 | +0.110 | -0.221 | -0.097 |
| results_lean_dL2_mixed | +0.322 | -0.415 | -0.126 | +0.409 | +0.038 | -0.077 | +0.212 | -0.183 | -0.097 |
| results_lean_dL3_mixed | +0.446 | -0.365 | -0.126 | +0.494 | +0.054 | -0.077 | +0.238 | -0.136 | -0.097 |
| results_lean_dL4_mixed | +0.427 | -0.329 | -0.126 | +0.494 | +0.091 | -0.077 | +0.274 | -0.085 | -0.097 |
| results_lean_dL5_mixed | +0.478 | -0.216 | -0.126 | +0.502 | +0.084 | -0.077 | +0.311 | -0.074 | -0.097 |

## 2. symmetric / FID / W2 feature families — ridge ctx_rho_g

| ablation | split | motion_sym | motion_fid | motion_w2 | motion_mmd | appearance_sym | appearance_fid | appearance_w2 | appearance_mmd | appearance_nullk |
|---|---|---|---|---|---|---|---|---|---|---|
| results_lean_canon_mixed | LOTO | — | — | — | — | — | — | — | — | — |
| results_lean_canon_mixed | LOBO | — | — | — | — | — | — | — | — | — |
| results_lean_canon_mixed | JOINT | — | — | — | — | — | — | — | — | — |
| results_lean_dL1_mixed | LOTO | — | — | — | — | — | — | — | — | — |
| results_lean_dL1_mixed | LOBO | — | — | — | — | — | — | — | — | — |
| results_lean_dL1_mixed | JOINT | — | — | — | — | — | — | — | — | — |
| results_lean_dL2_mixed | LOTO | — | — | — | — | — | — | — | — | — |
| results_lean_dL2_mixed | LOBO | — | — | — | — | — | — | — | — | — |
| results_lean_dL2_mixed | JOINT | — | — | — | — | — | — | — | — | — |
| results_lean_dL3_mixed | LOTO | — | — | — | — | — | — | — | — | — |
| results_lean_dL3_mixed | LOBO | — | — | — | — | — | — | — | — | — |
| results_lean_dL3_mixed | JOINT | — | — | — | — | — | — | — | — | — |
| results_lean_dL4_mixed | LOTO | — | — | — | — | — | — | — | — | — |
| results_lean_dL4_mixed | LOBO | — | — | — | — | — | — | — | — | — |
| results_lean_dL4_mixed | JOINT | — | — | — | — | — | — | — | — | — |
| results_lean_dL5_mixed | LOTO | — | — | — | — | — | — | — | — | — |
| results_lean_dL5_mixed | LOBO | — | — | — | — | — | — | — | — | — |
| results_lean_dL5_mixed | JOINT | — | — | — | — | — | — | — | — | — |

## 3. ctx_rho_L — level-only ranking ρ

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| results_lean_canon_mixed | -1.000 | -1.000 | -1.000 | +0.481 | +0.535 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_lean_dL1_mixed | -1.000 | -1.000 | -1.000 | +0.583 | +0.585 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_lean_dL2_mixed | -1.000 | -1.000 | -1.000 | +0.556 | +0.578 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_lean_dL3_mixed | -1.000 | -1.000 | -1.000 | +0.515 | +0.530 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_lean_dL4_mixed | -1.000 | -1.000 | -1.000 | +0.491 | +0.534 | +0.540 | -0.540 | -0.540 | -0.540 |
| results_lean_dL5_mixed | -1.000 | -1.000 | -1.000 | +0.484 | +0.534 | +0.540 | -0.540 | -0.540 | -0.540 |

## 4. abs_r_Lg — pooled calibration

| ablation | LOTO/motion | LOTO/appearance | LOTO/random | LOBO/motion | LOBO/appearance | LOBO/random | JOINT/motion | JOINT/appearance | JOINT/random |
|---|---|---|---|---|---|---|---|---|---|
| results_lean_canon_mixed | +0.851 | +0.751 | +0.819 | +0.737 | +0.755 | +0.402 | +0.379 | -0.216 | +0.148 |
| results_lean_dL1_mixed | +0.821 | +0.697 | +0.819 | +0.748 | +0.788 | +0.402 | +0.420 | +0.144 | +0.148 |
| results_lean_dL2_mixed | +0.822 | +0.662 | +0.819 | +0.769 | +0.715 | +0.402 | +0.383 | +0.116 | +0.148 |
| results_lean_dL3_mixed | +0.831 | +0.734 | +0.819 | +0.747 | +0.691 | +0.402 | +0.409 | +0.211 | +0.148 |
| results_lean_dL4_mixed | +0.848 | +0.757 | +0.819 | +0.743 | +0.741 | +0.402 | +0.346 | +0.090 | +0.148 |
| results_lean_dL5_mixed | +0.852 | +0.774 | +0.819 | +0.740 | +0.746 | +0.402 | +0.246 | +0.033 | +0.148 |

## 5. motion ridge ctx_rho_g — with 95% CIs

| ablation | LOTO | LOBO | JOINT |
|---|---|---|---|
| results_lean_canon_mixed | — | — | — |
| results_lean_dL1_mixed | — | — | — |
| results_lean_dL2_mixed | — | — | — |
| results_lean_dL3_mixed | — | — | — |
| results_lean_dL4_mixed | — | — | — |
| results_lean_dL5_mixed | — | — | — |

---

## Files referenced

- `scripts/transfer_analysis_v4/results_lean_canon_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_lean_canon_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_lean_dL1_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_lean_dL1_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_lean_dL2_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_lean_dL2_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_lean_dL3_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_lean_dL3_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_lean_dL4_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_lean_dL4_mixed/summary.csv` — long-form metrics
- `scripts/transfer_analysis_v4/results_lean_dL5_mixed/results.md` — full per-mode report
- `scripts/transfer_analysis_v4/results_lean_dL5_mixed/summary.csv` — long-form metrics