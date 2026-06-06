# Transfer Analysis v4 — Density Ablation

Two stability axes:

- **Feature-side**: for each pairwise self-distance metric, at what N does the metric's value match its asymptote (Spearman ρ ≥ 0.9)? Reads from the existing `analysis_v3/density_invariance_pair_sharded/stability_*.csv`.
- **Fitted-side**: for each family, at what N does the within-context ridge ρ_g stop moving (|Δρ_g| ≤ 0.05 from canonical density, monotone)? Reads from the lean density sweep results.

Density diagonal levels:

| code | flow N | dino N |
|---|---|---|
| dL1 | 50,000 | 25,000 |
| dL2 | 200,000 | 100,000 |
| dL3 | 1,000,000 | 500,000 |
| dL4 | 4,000,000 | 2,000,000 |
| dL5 | 8,000,000 | 4,000,000 |

---

## 1. Feature-side stability (Spearman ρ vs baseline)


### FLOW


_pair_type = eval_eval_

| metric | N=50,000 | N=200,000 | N=1,000,000 | N=4,000,000 | N=8,000,000 |
|---|---|---|---|---|---|
| a_covered_by_b_eps16px | 0.937 | 0.975 | 0.994 | 0.999 | 1.000 |
| a_covered_by_b_eps1px | 0.746 | 0.827 | 0.937 | 0.996 | 1.000 |
| a_covered_by_b_eps4px | 0.912 | 0.960 | 0.984 | 0.997 | 0.999 |
| b_covered_by_a_eps16px | 0.937 | 0.975 | 0.994 | 0.999 | 1.000 |
| b_covered_by_a_eps1px | 0.746 | 0.827 | 0.937 | 0.996 | 1.000 |
| b_covered_by_a_eps4px | 0.912 | 0.960 | 0.984 | 0.997 | 0.999 |
| kl_a_to_b_k20 | -0.077 | -0.024 | 0.012 | 0.028 | 0.032 |
| kl_a_to_b_k5 | 0.014 | 0.061 | 0.077 | 0.104 | 0.107 |
| kl_b_to_a_k20 | -0.077 | -0.024 | 0.012 | 0.028 | 0.032 |
| kl_b_to_a_k5 | 0.014 | 0.061 | 0.077 | 0.104 | 0.107 |
| mean_nn_a_to_b | 0.969 | 0.986 | 0.997 | 0.999 | 1.000 |
| mean_nn_b_to_a | 0.969 | 0.986 | 0.997 | 0.999 | 1.000 |
| mean_nn_sym | 0.971 | 0.990 | 0.998 | 1.000 | 1.000 |

_pair_type = train_eval_

| metric | N=50,000 | N=200,000 | N=1,000,000 | N=4,000,000 | N=8,000,000 |
|---|---|---|---|---|---|
| a_covered_by_b_eps16px | 0.905 | 0.959 | 0.989 | 0.998 | 1.000 |
| a_covered_by_b_eps1px | 0.820 | 0.930 | 0.969 | 0.994 | 0.998 |
| a_covered_by_b_eps4px | 0.911 | 0.960 | 0.980 | 0.995 | 0.999 |
| b_covered_by_a_eps16px | 0.847 | 0.917 | 0.977 | 0.997 | 0.999 |
| b_covered_by_a_eps1px | 0.772 | 0.843 | 0.915 | 0.969 | 0.989 |
| b_covered_by_a_eps4px | 0.753 | 0.825 | 0.920 | 0.975 | 0.993 |
| kl_a_to_b_k20 | 0.469 | 0.479 | 0.489 | 0.495 | 0.492 |
| kl_a_to_b_k5 | 0.484 | 0.502 | 0.514 | 0.503 | 0.508 |
| kl_b_to_a_k20 | -0.018 | 0.040 | 0.113 | 0.189 | 0.206 |
| kl_b_to_a_k5 | 0.049 | 0.095 | 0.156 | 0.215 | 0.230 |
| mean_nn_a_to_b | 0.962 | 0.984 | 0.996 | 0.999 | 1.000 |
| mean_nn_b_to_a | 0.940 | 0.957 | 0.980 | 0.996 | 0.999 |
| mean_nn_sym | 0.953 | 0.983 | 0.996 | 0.999 | 1.000 |

_Minimum N for ρ ≥ 0.9 (per pair_type × metric):_

| pair_type | metric | min N | worst ρ |
|---|---|---|---|
| eval_eval | a_covered_by_b_eps16px | 50,000.0 | 0.937 |
| eval_eval | a_covered_by_b_eps1px | 1,000,000.0 | 0.746 |
| eval_eval | a_covered_by_b_eps4px | 50,000.0 | 0.912 |
| eval_eval | b_covered_by_a_eps16px | 50,000.0 | 0.937 |
| eval_eval | b_covered_by_a_eps1px | 1,000,000.0 | 0.746 |
| eval_eval | b_covered_by_a_eps4px | 50,000.0 | 0.912 |
| eval_eval | kl_a_to_b_k20 | *never reaches threshold* | -0.077 |
| eval_eval | kl_a_to_b_k5 | *never reaches threshold* | 0.014 |
| eval_eval | kl_b_to_a_k20 | *never reaches threshold* | -0.077 |
| eval_eval | kl_b_to_a_k5 | *never reaches threshold* | 0.014 |
| eval_eval | mean_nn_a_to_b | 50,000.0 | 0.969 |
| eval_eval | mean_nn_b_to_a | 50,000.0 | 0.969 |
| eval_eval | mean_nn_sym | 50,000.0 | 0.971 |
| train_eval | a_covered_by_b_eps16px | 50,000.0 | 0.905 |
| train_eval | a_covered_by_b_eps1px | 200,000.0 | 0.820 |
| train_eval | a_covered_by_b_eps4px | 50,000.0 | 0.911 |
| train_eval | b_covered_by_a_eps16px | 200,000.0 | 0.847 |
| train_eval | b_covered_by_a_eps1px | 1,000,000.0 | 0.772 |
| train_eval | b_covered_by_a_eps4px | 1,000,000.0 | 0.753 |
| train_eval | kl_a_to_b_k20 | *never reaches threshold* | 0.469 |
| train_eval | kl_a_to_b_k5 | *never reaches threshold* | 0.484 |
| train_eval | kl_b_to_a_k20 | *never reaches threshold* | -0.018 |
| train_eval | kl_b_to_a_k5 | *never reaches threshold* | 0.049 |
| train_eval | mean_nn_a_to_b | 50,000.0 | 0.962 |
| train_eval | mean_nn_b_to_a | 50,000.0 | 0.940 |
| train_eval | mean_nn_sym | 50,000.0 | 0.953 |

### DINO


_pair_type = eval_eval_

| metric | N=25,000 | N=100,000 | N=500,000 | N=2,000,000 | N=4,000,000 |
|---|---|---|---|---|---|
| a_covered_by_b_eps16px | 0.945 | 0.945 | 1.000 | 1.000 | 1.000 |
| a_covered_by_b_eps1px | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| a_covered_by_b_eps4px | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| b_covered_by_a_eps16px | 0.945 | 0.945 | 1.000 | 1.000 | 1.000 |
| b_covered_by_a_eps1px | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| b_covered_by_a_eps4px | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| kl_a_to_b_k20 | 0.877 | 0.920 | 0.983 | 0.998 | 0.999 |
| kl_a_to_b_k5 | 0.881 | 0.930 | 0.988 | 0.997 | 0.999 |
| kl_b_to_a_k20 | 0.877 | 0.920 | 0.983 | 0.998 | 0.999 |
| kl_b_to_a_k5 | 0.881 | 0.930 | 0.988 | 0.997 | 0.999 |
| mean_nn_a_to_b | 0.971 | 0.981 | 0.993 | 0.997 | 0.999 |
| mean_nn_b_to_a | 0.971 | 0.981 | 0.993 | 0.997 | 0.999 |
| mean_nn_sym | 0.977 | 0.988 | 0.996 | 0.999 | 1.000 |

_pair_type = train_eval_

| metric | N=25,000 | N=100,000 | N=500,000 | N=2,000,000 | N=4,000,000 |
|---|---|---|---|---|---|
| a_covered_by_b_eps16px | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| a_covered_by_b_eps1px | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| a_covered_by_b_eps4px | 0.870 | 0.870 | 0.870 | 1.000 | 1.000 |
| b_covered_by_a_eps16px | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |
| b_covered_by_a_eps1px | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| b_covered_by_a_eps4px | 0.870 | 0.870 | 0.870 | 1.000 | 1.000 |
| kl_a_to_b_k20 | 0.574 | 0.605 | 0.813 | 0.973 | 0.995 |
| kl_a_to_b_k5 | 0.535 | 0.587 | 0.921 | 0.995 | 0.999 |
| kl_b_to_a_k20 | 0.850 | 0.893 | 0.974 | 0.996 | 0.998 |
| kl_b_to_a_k5 | 0.828 | 0.880 | 0.955 | 0.988 | 0.998 |
| mean_nn_a_to_b | 0.937 | 0.966 | 0.989 | 0.997 | 0.999 |
| mean_nn_b_to_a | 0.993 | 0.995 | 0.997 | 0.999 | 1.000 |
| mean_nn_sym | 0.979 | 0.985 | 0.994 | 0.999 | 1.000 |

_Minimum N for ρ ≥ 0.9 (per pair_type × metric):_

| pair_type | metric | min N | worst ρ |
|---|---|---|---|
| eval_eval | a_covered_by_b_eps16px | 25,000 | 0.945 |
| eval_eval | a_covered_by_b_eps1px | 25,000 | 1.000 |
| eval_eval | a_covered_by_b_eps4px | 25,000 | 1.000 |
| eval_eval | b_covered_by_a_eps16px | 25,000 | 0.945 |
| eval_eval | b_covered_by_a_eps1px | 25,000 | 1.000 |
| eval_eval | b_covered_by_a_eps4px | 25,000 | 1.000 |
| eval_eval | kl_a_to_b_k20 | 100,000 | 0.877 |
| eval_eval | kl_a_to_b_k5 | 100,000 | 0.881 |
| eval_eval | kl_b_to_a_k20 | 100,000 | 0.877 |
| eval_eval | kl_b_to_a_k5 | 100,000 | 0.881 |
| eval_eval | mean_nn_a_to_b | 25,000 | 0.971 |
| eval_eval | mean_nn_b_to_a | 25,000 | 0.971 |
| eval_eval | mean_nn_sym | 25,000 | 0.977 |
| train_eval | a_covered_by_b_eps16px | 25,000 | 0.999 |
| train_eval | a_covered_by_b_eps1px | 25,000 | 1.000 |
| train_eval | a_covered_by_b_eps4px | 2,000,000 | 0.870 |
| train_eval | b_covered_by_a_eps16px | 25,000 | 0.999 |
| train_eval | b_covered_by_a_eps1px | 25,000 | 1.000 |
| train_eval | b_covered_by_a_eps4px | 2,000,000 | 0.870 |
| train_eval | kl_a_to_b_k20 | 2,000,000 | 0.574 |
| train_eval | kl_a_to_b_k5 | 500,000 | 0.535 |
| train_eval | kl_b_to_a_k20 | 500,000 | 0.850 |
| train_eval | kl_b_to_a_k5 | 500,000 | 0.828 |
| train_eval | mean_nn_a_to_b | 25,000 | 0.937 |
| train_eval | mean_nn_b_to_a | 25,000 | 0.993 |
| train_eval | mean_nn_sym | 25,000 | 0.979 |

---

## 2. Fitted-side stability (ctx_rho_g across densities)


_For each family × split: ρ_g at canonical density and at each of the 5 matched diagonal levels. min_stable_dL = smallest dL whose |ρ_g − canon| ≤ 0.05 and stays ≤ 0.05 for all higher dLs._


### Target: `peak_pck`

| family | split | canon | dL1 | dL2 | dL3 | dL4 | dL5 | min_stable_dL | span |
|---|---|---|---|---|---|---|---|---|---|
| appearance | LOTO | -0.254 | -0.327 | -0.415 | -0.365 | -0.329 | -0.216 | dL5 | 0.199 |
| appearance | LOBO | +0.074 | +0.092 | +0.038 | +0.054 | +0.091 | +0.084 | dL1 | 0.054 |
| appearance | JOINT | -0.202 | -0.221 | -0.183 | -0.136 | -0.085 | -0.074 | — | 0.147 |
| appearance_fid | LOTO | -0.164 | -0.164 | -0.164 | -0.164 | -0.164 | -0.164 | dL1 | 0.000 |
| appearance_fid | LOBO | +0.262 | +0.262 | +0.262 | +0.262 | +0.262 | +0.262 | dL1 | 0.000 |
| appearance_fid | JOINT | -0.150 | -0.150 | -0.150 | -0.150 | -0.150 | -0.150 | dL1 | 0.000 |
| appearance_mmd | LOTO | +0.214 | +0.214 | +0.214 | +0.214 | +0.214 | +0.214 | dL1 | 0.000 |
| appearance_mmd | LOBO | +0.342 | +0.342 | +0.342 | +0.342 | +0.342 | +0.342 | dL1 | 0.000 |
| appearance_mmd | JOINT | +0.291 | +0.291 | +0.291 | +0.291 | +0.291 | +0.291 | dL1 | 0.000 |
| appearance_nullk | LOTO | -0.067 | -0.067 | -0.067 | -0.067 | -0.067 | -0.067 | dL1 | 0.000 |
| appearance_nullk | LOBO | +0.054 | +0.054 | +0.054 | +0.054 | +0.054 | +0.054 | dL1 | 0.000 |
| appearance_nullk | JOINT | -0.057 | -0.057 | -0.057 | -0.057 | -0.057 | -0.057 | dL1 | 0.000 |
| appearance_sym | LOTO | +0.070 | +0.070 | +0.070 | +0.070 | +0.070 | +0.070 | dL1 | 0.000 |
| appearance_sym | LOBO | +0.144 | +0.144 | +0.144 | +0.144 | +0.144 | +0.144 | dL1 | 0.000 |
| appearance_sym | JOINT | +0.022 | +0.022 | +0.022 | +0.022 | +0.022 | +0.022 | dL1 | 0.000 |
| appearance_w2 | LOTO | -0.188 | -0.188 | -0.188 | -0.188 | -0.188 | -0.188 | dL1 | 0.000 |
| appearance_w2 | LOBO | -0.021 | -0.021 | -0.021 | -0.021 | -0.021 | -0.021 | dL1 | 0.000 |
| appearance_w2 | JOINT | -0.312 | -0.312 | -0.312 | -0.312 | -0.312 | -0.312 | dL1 | 0.000 |
| both | LOTO | +0.305 | +0.082 | +0.056 | +0.169 | +0.327 | +0.347 | dL4 | 0.291 |
| both | LOBO | +0.415 | +0.320 | +0.401 | +0.446 | +0.493 | +0.420 | dL5 | 0.174 |
| both | JOINT | +0.187 | -0.132 | +0.031 | +0.207 | +0.212 | +0.149 | dL3 | 0.345 |
| density | LOTO | -0.282 | -0.282 | -0.282 | -0.282 | -0.282 | -0.282 | dL1 | 0.000 |
| density | LOBO | +0.153 | +0.153 | +0.153 | +0.153 | +0.153 | +0.153 | dL1 | 0.000 |
| density | JOINT | -0.213 | -0.213 | -0.213 | -0.213 | -0.213 | -0.213 | dL1 | 0.000 |
| motion | LOTO | +0.508 | +0.216 | +0.322 | +0.446 | +0.427 | +0.478 | dL5 | 0.292 |
| motion | LOBO | +0.448 | +0.338 | +0.409 | +0.494 | +0.494 | +0.502 | — | 0.163 |
| motion | JOINT | +0.321 | +0.110 | +0.212 | +0.238 | +0.274 | +0.311 | dL4 | 0.211 |
| motion_density | LOTO | +0.265 | +0.226 | +0.253 | +0.388 | +0.345 | +0.345 | — | 0.162 |
| motion_density | LOBO | +0.472 | +0.406 | +0.484 | +0.556 | +0.538 | +0.512 | dL5 | 0.151 |
| motion_density | JOINT | +0.050 | +0.200 | +0.191 | +0.157 | +0.156 | +0.244 | — | 0.194 |
| motion_fid | LOTO | +0.458 | +0.458 | +0.458 | +0.458 | +0.458 | +0.458 | dL1 | 0.000 |
| motion_fid | LOBO | +0.471 | +0.471 | +0.471 | +0.471 | +0.471 | +0.471 | dL1 | 0.000 |
| motion_fid | JOINT | +0.472 | +0.472 | +0.472 | +0.472 | +0.472 | +0.472 | dL1 | 0.000 |
| motion_km | LOTO | +0.413 | +0.413 | +0.413 | +0.413 | +0.413 | +0.413 | dL1 | 0.000 |
| motion_km | LOBO | +0.426 | +0.426 | +0.426 | +0.426 | +0.426 | +0.426 | dL1 | 0.000 |
| motion_km | JOINT | +0.322 | +0.322 | +0.322 | +0.322 | +0.322 | +0.322 | dL1 | 0.000 |
| motion_mmd | LOTO | -0.130 | -0.130 | -0.130 | -0.130 | -0.130 | -0.130 | dL1 | 0.000 |
| motion_mmd | LOBO | +0.206 | +0.206 | +0.206 | +0.206 | +0.206 | +0.206 | dL1 | 0.000 |
| motion_mmd | JOINT | +0.070 | +0.070 | +0.070 | +0.070 | +0.070 | +0.070 | dL1 | 0.000 |
| motion_size | LOTO | +0.392 | -0.051 | +0.320 | +0.343 | +0.394 | +0.423 | dL3 | 0.473 |
| motion_size | LOBO | +0.446 | +0.396 | +0.523 | +0.573 | +0.528 | +0.514 | — | 0.177 |
| motion_size | JOINT | -0.018 | +0.002 | +0.284 | +0.292 | +0.252 | +0.240 | — | 0.309 |
| motion_supdensity | LOTO | +0.311 | +0.362 | +0.383 | +0.488 | +0.425 | +0.444 | — | 0.176 |
| motion_supdensity | LOBO | +0.463 | +0.412 | +0.512 | +0.580 | +0.554 | +0.526 | — | 0.168 |
| motion_supdensity | JOINT | +0.260 | +0.361 | +0.342 | +0.377 | +0.399 | +0.356 | — | 0.139 |
| motion_sym | LOTO | +0.436 | +0.436 | +0.436 | +0.436 | +0.436 | +0.436 | dL1 | 0.000 |
| motion_sym | LOBO | +0.536 | +0.536 | +0.536 | +0.536 | +0.536 | +0.536 | dL1 | 0.000 |
| motion_sym | JOINT | +0.432 | +0.432 | +0.432 | +0.432 | +0.432 | +0.432 | dL1 | 0.000 |
| motion_w2 | LOTO | +0.474 | +0.474 | +0.474 | +0.474 | +0.474 | +0.474 | dL1 | 0.000 |
| motion_w2 | LOBO | +0.478 | +0.478 | +0.478 | +0.478 | +0.478 | +0.478 | dL1 | 0.000 |
| motion_w2 | JOINT | +0.494 | +0.494 | +0.494 | +0.494 | +0.494 | +0.494 | dL1 | 0.000 |
| random | LOTO | -0.126 | -0.126 | -0.126 | -0.126 | -0.126 | -0.126 | dL1 | 0.000 |
| random | LOBO | -0.077 | -0.077 | -0.077 | -0.077 | -0.077 | -0.077 | dL1 | 0.000 |
| random | JOINT | -0.097 | -0.097 | -0.097 | -0.097 | -0.097 | -0.097 | dL1 | 0.000 |
| size | LOTO | -0.172 | -0.172 | -0.172 | -0.172 | -0.172 | -0.172 | dL1 | 0.000 |
| size | LOBO | +0.166 | +0.166 | +0.166 | +0.166 | +0.166 | +0.166 | dL1 | 0.000 |
| size | JOINT | -0.038 | -0.038 | -0.038 | -0.038 | -0.038 | -0.038 | dL1 | 0.000 |
| supervision_density | LOTO | -0.164 | -0.164 | -0.164 | -0.164 | -0.164 | -0.164 | dL1 | 0.000 |
| supervision_density | LOBO | +0.132 | +0.132 | +0.132 | +0.132 | +0.132 | +0.132 | dL1 | 0.000 |
| supervision_density | JOINT | -0.331 | -0.331 | -0.331 | -0.331 | -0.331 | -0.331 | dL1 | 0.000 |

### Target: `auc_normalized`

_no rows_
