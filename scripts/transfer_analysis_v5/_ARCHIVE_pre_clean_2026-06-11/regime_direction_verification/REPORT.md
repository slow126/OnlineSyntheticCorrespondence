# Regime-Direction Law — verification report

Feature space: **flow**

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
| catspp|False|False | +0.527 | +0.026 | +0.456 | +0.501 [+0.253, +0.769] |
| catspp|False|True | +0.507 | +0.077 | +0.460 | +0.430 [+0.176, +0.675] |
| catspp|True|False | +0.005 | +0.511 | +0.417 | -0.506 [-0.747, -0.262] |
| catspp|True|True | +0.144 | +0.502 | +0.564 | -0.358 [-0.588, -0.099] |
| glunet|False|False | +0.305 | +0.414 | +0.465 | -0.109 [-0.510, +0.335] |
| glunet|False|True | +0.371 | +0.344 | +0.580 | +0.026 [-0.321, +0.412] |
| glunet|True|False | -0.230 | +0.674 | +0.261 | -0.904 [-1.153, -0.664] |
| glunet|True|True | -0.219 | +0.763 | +0.332 | -0.982 [-1.148, -0.786] |
| raft|True|False | +0.401 | +0.303 | +0.535 | +0.098 [-0.271, +0.478] |

- flip statistic (scratch − pretrained mean d): **+0.899**, exact permutation p = **0.0286** (70 assignments; RAFT excluded)
- RAFT d = +0.098 (scratch-profile predicted: d > 0): CONSISTENT
- leave-one-benchmark-out flip statistic range: [+0.807, +0.987] (all >0)

### metric family: eps4px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.591 | +0.159 | +0.284 | +0.432 [+0.229, +0.656] |
| catspp|False|True | +0.570 | +0.206 | +0.300 | +0.364 [+0.172, +0.569] |
| catspp|True|False | +0.084 | +0.439 | +0.334 | -0.356 [-0.546, -0.157] |
| catspp|True|True | +0.146 | +0.493 | +0.439 | -0.346 [-0.561, -0.113] |
| glunet|False|False | +0.334 | +0.503 | +0.359 | -0.169 [-0.501, +0.214] |
| glunet|False|True | +0.418 | +0.470 | +0.431 | -0.052 [-0.426, +0.330] |
| glunet|True|False | -0.162 | +0.328 | +0.221 | -0.490 [-0.664, -0.334] |
| glunet|True|True | -0.153 | +0.686 | +0.473 | -0.838 [-0.982, -0.681] |
| raft|True|False | +0.422 | +0.382 | +0.368 | +0.040 [-0.298, +0.382] |

- flip statistic (scratch − pretrained mean d): **+0.652**, exact permutation p = **0.0286** (70 assignments; RAFT excluded)
- RAFT d = +0.040 (scratch-profile predicted: d > 0): CONSISTENT
- leave-one-benchmark-out flip statistic range: [+0.577, +0.725] (all >0)

### metric family: eps16px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.569 | -0.075 | +0.217 | +0.643 [+0.363, +0.942] |
| catspp|False|True | +0.554 | -0.077 | +0.205 | +0.630 [+0.359, +0.905] |
| catspp|True|False | +0.048 | +0.368 | +0.338 | -0.319 [-0.558, -0.101] |
| catspp|True|True | +0.156 | +0.346 | +0.412 | -0.191 [-0.399, +0.052] |
| glunet|False|False | +0.339 | +0.271 | +0.283 | +0.069 [-0.349, +0.527] |
| glunet|False|True | +0.418 | +0.203 | +0.409 | +0.215 [-0.179, +0.634] |
| glunet|True|False | -0.148 | +0.546 | +0.358 | -0.695 [-0.974, -0.435] |
| glunet|True|True | -0.154 | +0.741 | +0.502 | -0.895 [-1.084, -0.696] |
| raft|True|False | +0.463 | +0.141 | +0.322 | +0.321 [-0.085, +0.735] |

- flip statistic (scratch − pretrained mean d): **+0.914**, exact permutation p = **0.0286** (70 assignments; RAFT excluded)
- RAFT d = +0.321 (scratch-profile predicted: d > 0): CONSISTENT
- leave-one-benchmark-out flip statistic range: [+0.812, +1.011] (all >0)

### metric family: kl_k20 (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.188 | +0.142 |  | +0.045 [-0.238, +0.324] |
| catspp|False|True | +0.169 | +0.177 |  | -0.008 [-0.285, +0.262] |
| catspp|True|False | +0.138 | +0.492 |  | -0.354 [-0.510, -0.203] |
| catspp|True|True | +0.170 | +0.575 |  | -0.405 [-0.576, -0.216] |
| glunet|False|False | +0.138 | +0.305 |  | -0.167 [-0.359, +0.032] |
| glunet|False|True | +0.273 | +0.365 |  | -0.092 [-0.349, +0.176] |
| glunet|True|False | +0.189 | +0.547 |  | -0.359 [-0.457, -0.247] |
| glunet|True|True | +0.335 | +0.630 |  | -0.295 [-0.406, -0.189] |
| raft|True|False | +0.102 | +0.348 |  | -0.246 [-0.511, +0.038] |

- flip statistic (scratch − pretrained mean d): **+0.298**, exact permutation p = **0.0286** (70 assignments; RAFT excluded)
- RAFT d = -0.246 (scratch-profile predicted: d > 0): INCONSISTENT
- leave-one-benchmark-out flip statistic range: [+0.231, +0.349] (all >0)

### self-pair exclusion (36 rows removed)
- mean_nn: flip statistic without self-pairs = +0.920 (holds)

## Target: auc_normalized  (rows=495)
- feature join missing rate: 0.000%
- features constant across variants per (train,bench): max nunique = 1 (must be 1)

### metric family: mean_nn (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.404 | -0.041 | +0.348 | +0.445 [+0.268, +0.633] |
| catspp|False|True | +0.472 | -0.009 | +0.347 | +0.481 [+0.204, +0.749] |
| catspp|True|False | -0.191 | +0.537 | +0.332 | -0.728 [-0.977, -0.441] |
| catspp|True|True | +0.226 | +0.174 | +0.424 | +0.053 [-0.047, +0.153] |
| raft|True|False | +0.478 | +0.244 | +0.573 | +0.233 [-0.181, +0.669] |

### metric family: eps4px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.494 | +0.103 | +0.275 | +0.391 [+0.207, +0.542] |
| catspp|False|True | +0.530 | +0.118 | +0.198 | +0.412 [+0.241, +0.586] |
| catspp|True|False | -0.148 | +0.454 | +0.277 | -0.602 [-0.818, -0.342] |
| catspp|True|True | +0.151 | +0.134 | +0.177 | +0.016 [-0.094, +0.130] |
| raft|True|False | +0.504 | +0.332 | +0.389 | +0.172 [-0.240, +0.570] |

### metric family: eps16px (better_when=high; sign=+1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.444 | +0.033 | +0.354 | +0.411 [+0.174, +0.633] |
| catspp|False|True | +0.519 | -0.074 | +0.198 | +0.593 [+0.273, +0.919] |
| catspp|True|False | -0.154 | +0.514 | +0.386 | -0.668 [-0.880, -0.438] |
| catspp|True|True | +0.190 | +0.116 | +0.335 | +0.074 [-0.045, +0.201] |
| raft|True|False | +0.535 | +0.132 | +0.424 | +0.403 [-0.027, +0.848] |

### metric family: kl_k20 (better_when=low; sign=-1)

| variant | a→b (precision) | b→a (recall) | sym | d = a→b − b→a [95% CI] |
|---|---|---|---|---|
| catspp|False|False | +0.491 | +0.109 |  | +0.382 [+0.191, +0.553] |
| catspp|False|True | +0.191 | +0.018 |  | +0.173 [-0.072, +0.414] |
| catspp|True|False | +0.274 | +0.443 |  | -0.170 [-0.291, -0.053] |
| catspp|True|True | +0.251 | +0.301 |  | -0.051 [-0.158, +0.071] |
| raft|True|False | +0.184 | +0.309 |  | -0.125 [-0.411, +0.185] |

### self-pair exclusion (20 rows removed)
- mean_nn: flip statistic without self-pairs = +0.816 (holds)

## Verdicts
- integrity: PASS
- features_constant: PASS
- flip_perm_mean_nn: PASS
- lobo_stable_mean_nn: PASS
- flip_perm_eps4px: PASS
- lobo_stable_eps4px: PASS
- flip_perm_eps16px: PASS
- lobo_stable_eps16px: PASS
- flip_perm_kl_k20: PASS
- lobo_stable_kl_k20: PASS
- selfpair_robust: PASS

# OVERALL: VERIFIED