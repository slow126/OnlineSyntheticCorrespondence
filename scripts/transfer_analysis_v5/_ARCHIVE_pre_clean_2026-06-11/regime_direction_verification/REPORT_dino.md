# Regime-Direction Law — verification report

Feature space: **dino**

Independent recomputation; no pipeline code imported.


## Target: peak_pck  (rows=891)
- orientation: 99/99 pairs forward (a=train), 0 REVERSED
- split_a=='train' for all train_eval rows: True
- duplicate (train, benchmark, variant) rows: 0
- context_id mismatches vs (benchmark,model,pre,frz): 0
- **integrity: PASS**
- feature join missing rate: 0.000%
- features constant across variants per (train,bench): max nunique = 1 (must be 1)

### metric family: mean_nn (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.191 | -0.196 | -0.230 | +0.005 [-0.055, +0.059] |
| catspp|False|True | -0.221 | -0.224 | -0.257 | +0.003 [-0.048, +0.062] |
| catspp|True|False | -0.294 | -0.308 | -0.289 | +0.014 [-0.043, +0.070] |
| catspp|True|True | -0.275 | -0.247 | -0.264 | -0.027 [-0.087, +0.026] |
| glunet|False|False | -0.282 | -0.307 | -0.310 | +0.025 [-0.038, +0.101] |
| glunet|False|True | -0.203 | -0.162 | -0.173 | -0.041 [-0.115, +0.029] |
| glunet|True|False | -0.189 | -0.161 | -0.143 | -0.028 [-0.097, +0.076] |
| glunet|True|True | -0.183 | -0.145 | -0.152 | -0.037 [-0.106, +0.030] |
| raft|True|False | -0.260 | -0.348 | -0.326 | +0.089 [+0.023, +0.160] |

- flip statistic (scratch − pretrained mean d): **+0.018**, exact permutation p = **0.4286** (70 assignments; RAFT excluded)
- RAFT d = +0.089 (scratch-profile predicted: d > 0): CONSISTENT
- leave-one-benchmark-out flip statistic range: [-0.002, +0.047] (CROSSES 0)

### metric family: eps4px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.279 | -0.279 | -0.279 | +0.000 [+0.000, +0.000] |
| catspp|False|True | -0.291 | -0.291 | -0.291 | +0.000 [+0.000, +0.000] |
| catspp|True|False | +0.143 | +0.143 | +0.143 | +0.000 [+0.000, +0.000] |
| catspp|True|True | +0.110 | +0.110 | +0.110 | +0.000 [+0.000, +0.000] |
| glunet|False|False | +0.042 | +0.042 | +0.042 | +0.000 [+0.000, +0.000] |
| glunet|False|True | -0.048 | -0.048 | -0.048 | +0.000 [+0.000, +0.000] |
| glunet|True|False | +0.376 | +0.376 | +0.376 | +0.000 [+0.000, +0.000] |
| glunet|True|True | +0.209 | +0.209 | +0.209 | +0.000 [+0.000, +0.000] |
| raft|True|False | +0.070 | +0.070 | +0.070 | +0.000 [+0.000, +0.000] |

- flip statistic (scratch − pretrained mean d): **+0.000**, exact permutation p = **1.0000** (70 assignments; RAFT excluded)
- RAFT d = +0.000 (scratch-profile predicted: d > 0): INCONSISTENT
- leave-one-benchmark-out flip statistic range: [+0.000, +0.000] (CROSSES 0)

### metric family: eps16px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.277 | -0.277 | -0.277 | +0.000 [+0.000, +0.000] |
| catspp|False|True | -0.243 | -0.243 | -0.243 | +0.000 [+0.000, +0.000] |
| catspp|True|False | +0.167 | +0.167 | +0.167 | +0.000 [+0.000, +0.000] |
| catspp|True|True | +0.142 | +0.142 | +0.142 | +0.000 [+0.000, +0.000] |
| glunet|False|False | -0.003 | -0.003 | -0.003 | +0.000 [+0.000, +0.000] |
| glunet|False|True | -0.056 | -0.056 | -0.056 | +0.000 [+0.000, +0.000] |
| glunet|True|False | +0.449 | +0.449 | +0.449 | +0.000 [+0.000, +0.000] |
| glunet|True|True | +0.146 | +0.146 | +0.146 | +0.000 [+0.000, +0.000] |
| raft|True|False | +0.037 | +0.037 | +0.037 | +0.000 [+0.000, +0.000] |

- flip statistic (scratch − pretrained mean d): **+0.000**, exact permutation p = **1.0000** (70 assignments; RAFT excluded)
- RAFT d = +0.000 (scratch-profile predicted: d > 0): INCONSISTENT
- leave-one-benchmark-out flip statistic range: [+0.000, +0.000] (CROSSES 0)

### metric family: kl_k20 (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.086 | -0.239 |  | +0.154 [+0.032, +0.254] |
| catspp|False|True | -0.033 | -0.260 |  | +0.226 [+0.107, +0.323] |
| catspp|True|False | +0.006 | -0.302 |  | +0.308 [+0.037, +0.507] |
| catspp|True|True | +0.105 | -0.252 |  | +0.357 [+0.104, +0.555] |
| glunet|False|False | +0.142 | -0.347 |  | +0.490 [+0.349, +0.616] |
| glunet|False|True | +0.209 | -0.195 |  | +0.404 [+0.317, +0.501] |
| glunet|True|False | +0.261 | -0.142 |  | +0.403 [+0.304, +0.498] |
| glunet|True|True | +0.384 | -0.194 |  | +0.578 [+0.465, +0.678] |
| raft|True|False | -0.043 | -0.357 |  | +0.313 [+0.102, +0.498] |

- flip statistic (scratch − pretrained mean d): **-0.093**, exact permutation p = **0.4000** (70 assignments; RAFT excluded)
- RAFT d = +0.313 (scratch-profile predicted: d > 0): CONSISTENT
- leave-one-benchmark-out flip statistic range: [-0.118, -0.059] (CROSSES 0)

### self-pair exclusion (36 rows removed)
- mean_nn: flip statistic without self-pairs = +0.024 (holds)

## Target: auc_normalized  (rows=495)
- feature join missing rate: 0.000%
- features constant across variants per (train,bench): max nunique = 1 (must be 1)

### metric family: mean_nn (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.129 | +0.198 | +0.172 | -0.069 [-0.179, +0.023] |
| catspp|False|True | -0.155 | -0.006 | -0.078 | -0.148 [-0.281, -0.024] |
| catspp|True|False | -0.253 | -0.084 | -0.142 | -0.169 [-0.232, -0.104] |
| catspp|True|True | +0.056 | +0.265 | +0.172 | -0.209 [-0.351, -0.098] |
| raft|True|False | -0.149 | -0.098 | -0.113 | -0.052 [-0.161, +0.060] |

### metric family: eps4px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.148 | -0.148 | -0.148 | +0.000 [+0.000, +0.000] |
| catspp|False|True | -0.471 | -0.471 | -0.471 | +0.000 [+0.000, +0.000] |
| catspp|True|False | +0.193 | +0.193 | +0.193 | +0.000 [+0.000, +0.000] |
| catspp|True|True | +0.088 | +0.088 | +0.088 | +0.000 [+0.000, +0.000] |
| raft|True|False | -0.007 | -0.007 | -0.007 | +0.000 [+0.000, +0.000] |

### metric family: eps16px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.231 | -0.231 | -0.231 | +0.000 [+0.000, +0.000] |
| catspp|False|True | -0.407 | -0.407 | -0.407 | +0.000 [+0.000, +0.000] |
| catspp|True|False | +0.141 | +0.141 | +0.141 | +0.000 [+0.000, +0.000] |
| catspp|True|True | +0.049 | +0.049 | +0.049 | +0.000 [+0.000, +0.000] |
| raft|True|False | -0.033 | -0.033 | -0.033 | +0.000 [+0.000, +0.000] |

### metric family: kl_k20 (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | -0.008 | +0.152 |  | -0.160 [-0.255, -0.046] |
| catspp|False|True | -0.013 | -0.037 |  | +0.024 [-0.047, +0.089] |
| catspp|True|False | +0.173 | -0.119 |  | +0.292 [+0.106, +0.422] |
| catspp|True|True | +0.261 | +0.228 |  | +0.032 [-0.117, +0.145] |
| raft|True|False | +0.032 | -0.113 |  | +0.145 [-0.050, +0.324] |

### self-pair exclusion (20 rows removed)
- mean_nn: flip statistic without self-pairs = +0.105 (holds)

## Verdicts — NEGATIVE CONTROL (law predicts NO flip in appearance space)
- integrity: PASS
- features_constant: PASS
- flip_perm_mean_nn: no flip — as predicted for the control
- lobo_stable_mean_nn: no flip — as predicted for the control
- flip_perm_eps4px: no flip — as predicted for the control
- lobo_stable_eps4px: no flip — as predicted for the control
- flip_perm_eps16px: no flip — as predicted for the control
- lobo_stable_eps16px: no flip — as predicted for the control
- flip_perm_kl_k20: no flip — as predicted for the control
- lobo_stable_kl_k20: no flip — as predicted for the control
- selfpair_robust: holds

# OVERALL: CONTROL CONFIRMED — no appearance-space flip (motion-specificity supported)