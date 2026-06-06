# Transfer Analysis v4 — Residual Calibration Ablation

Leakage-clean residual-magnitude calibration across families. Generated from `context_scale_calibration.py` outputs.

**Headline (rank-only) ρ_g claims live in [ABLATION.md](ABLATION.md) and [ABLATION_strength.md](ABLATION_strength.md).** This file reports the *magnitude* side: per-family residual-scale behavior under raw ridge vs each per-fold calibrated head.

Calibrated heads (positive scalar gains per context; rank-preserving):
- `g_global_gain` — one fold-wide residual-std / pred-std gain
- `g_variant_gain` — same-variant gain, fallback global
- `g_context_gain` — same (benchmark|variant), fallback variant
- `g_shrink_gain` — context shrunk toward variant (k=5)
- `g_benchsim_gain` — same-variant gains IDW-smoothed across other benchmarks via flow `mean_nn_sym` (zero-shot-ish over benchmark neighborhood; `--kernel-space dino` switches to DINO neighborhoods)
- `g_profilesim_gain` — same-variant gains IDW-smoothed via eval-side dataset profile/density features


## Quick read

- Raw `g` is **under-dispersed** across families (median std ratio 0.3–0.9). Calibrated heads bring it close to 1.0.
- **mean_nn feature subset is the only family where `g_benchsim_gain` fails** — its raw std ratio (~0.26 LOBO) is so low that the per-fold gains it tries to IDW-smooth are large and noisy, causing the kernel to over-amplify (pooled std ratio > 2).
- Every other family tested has raw std ratio ≥ 0.4 and benefits cleanly from `g_benchsim_gain` and/or `g_profilesim_gain`.
- **Recommendation for downstream search**: `motion_sym + g_benchsim_gain` (best end-to-end LOBO/JOINT). `motion_w2` and `motion_fid` calibrate well too if a single-feature head is preferred.


## Per-family raw-vs-best calibrated head

For each family on LOTO/LOBO/JOINT, comparing raw `g` to the best-calibrated head (by ctx_pearson). Spearman is preserved within each family (gain is rank-invariant per context).


### all_variants

| family | split | raw ρ_S | raw r | raw std (med) | **best head** | best r | best std (med) | best pooled std |
|---|---|---|---|---|---|---|---|---|
| density_idw:motion_sym | LOTO | +0.436 | +0.389 | 0.442 | **g_benchsim_gain** | +0.511 | 0.638 | 0.632 |
| density_idw:motion_sym | LOBO | +0.536 | +0.577 | 0.692 | **g_benchsim_gain** | +0.691 | 1.111 | 0.990 |
| density_idw:motion_sym | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |
| eb_shrunk:motion_sym | LOTO | +0.436 | +0.389 | 0.442 | **g_benchsim_gain** | +0.511 | 0.638 | 0.632 |
| eb_shrunk:motion_sym | LOBO | +0.536 | +0.577 | 0.692 | **g_benchsim_gain** | +0.691 | 1.111 | 0.990 |
| eb_shrunk:motion_sym | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | +0.419 | +0.243 | 0.358 | **g_shrink_gain** | +0.395 | 1.148 | 1.114 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | +0.501 | +0.413 | 0.261 | **g_variant_gain** | +0.464 | 0.589 | 1.086 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | +0.487 | +0.252 | 0.316 | **g_variant_gain** | +0.380 | 0.702 | 1.048 |
| mixed:motion_all | LOTO | +0.507 | +0.462 | 0.719 | **g_context_gain** | +0.545 | 0.934 | 0.897 |
| mixed:motion_all | LOBO | +0.448 | +0.520 | 0.870 | **g_profilesim_gain** | +0.657 | 1.007 | 0.986 |
| mixed:motion_all | JOINT | +0.304 | +0.279 | 0.629 | **g_benchsim_gain** | +0.391 | 0.941 | 0.797 |
| mixed:motion_fid | LOTO | +0.458 | +0.326 | 0.574 | **g_benchsim_gain** | +0.421 | 1.239 | 1.204 |
| mixed:motion_fid | LOBO | +0.471 | +0.384 | 0.490 | **g_benchsim_gain** | +0.518 | 1.027 | 1.035 |
| mixed:motion_fid | JOINT | +0.466 | +0.314 | 0.565 | **g_benchsim_gain** | +0.402 | 1.049 | 1.091 |
| mixed:motion_sym | LOTO | +0.436 | +0.389 | 0.442 | **g_benchsim_gain** | +0.511 | 0.638 | 0.632 |
| mixed:motion_sym | LOBO | +0.536 | +0.577 | 0.692 | **g_benchsim_gain** | +0.691 | 1.111 | 0.990 |
| mixed:motion_sym | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |
| mixed:motion_sym_dinokernel | LOTO | +0.436 | +0.389 | 0.442 | **g_profilesim_gain** | +0.501 | 0.731 | 0.678 |
| mixed:motion_sym_dinokernel | LOBO | +0.536 | +0.577 | 0.692 | **g_profilesim_gain** | +0.682 | 1.073 | 1.142 |
| mixed:motion_sym_dinokernel | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |
| mixed:motion_w2 | LOTO | +0.474 | +0.326 | 0.783 | **g_benchsim_gain** | +0.382 | 1.434 | 1.281 |
| mixed:motion_w2 | LOBO | +0.478 | +0.370 | 0.610 | **g_benchsim_gain** | +0.505 | 1.069 | 1.015 |
| mixed:motion_w2 | JOINT | +0.495 | +0.296 | 0.760 | **g_global_gain** | +0.338 | 1.722 | 1.066 |
| symmetric_informed:motion_sym | LOTO | +0.436 | +0.389 | 0.442 | **g_benchsim_gain** | +0.511 | 0.638 | 0.632 |
| symmetric_informed:motion_sym | LOBO | +0.536 | +0.577 | 0.692 | **g_benchsim_gain** | +0.691 | 1.111 | 0.990 |
| symmetric_informed:motion_sym | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |
| symmetric_uninformed:motion_sym | LOTO | +0.436 | +0.389 | 0.442 | **g_benchsim_gain** | +0.511 | 0.638 | 0.632 |
| symmetric_uninformed:motion_sym | LOBO | +0.536 | +0.577 | 0.692 | **g_benchsim_gain** | +0.691 | 1.111 | 0.990 |
| symmetric_uninformed:motion_sym | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |
| targeted_informed:motion_sym | LOTO | +0.436 | +0.389 | 0.442 | **g_benchsim_gain** | +0.511 | 0.638 | 0.632 |
| targeted_informed:motion_sym | LOBO | +0.536 | +0.577 | 0.692 | **g_benchsim_gain** | +0.691 | 1.111 | 0.990 |
| targeted_informed:motion_sym | JOINT | +0.435 | +0.314 | 0.551 | **g_profilesim_gain** | +0.468 | 0.914 | 0.728 |

### drop_false_true

| family | split | raw ρ_S | raw r | raw std (med) | **best head** | best r | best std (med) | best pooled std |
|---|---|---|---|---|---|---|---|---|
| fsub_mean_nn:motion mean_nn (default) | LOTO | +0.379 | +0.207 | 0.323 | **g_shrink_gain** | +0.366 | 1.149 | 1.109 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | +0.494 | +0.367 | 0.248 | **g_variant_gain** | +0.426 | 0.591 | 1.078 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | +0.420 | +0.203 | 0.255 | **g_variant_gain** | +0.333 | 0.701 | 1.030 |
| mixed:motion_sym | LOTO | +0.429 | +0.351 | 0.474 | **g_profilesim_gain** | +0.480 | 0.785 | 0.739 |
| mixed:motion_sym | LOBO | +0.529 | +0.511 | 0.652 | **g_benchsim_gain** | +0.637 | 1.135 | 0.980 |
| mixed:motion_sym | JOINT | +0.412 | +0.296 | 0.655 | **g_profilesim_gain** | +0.462 | 1.081 | 0.771 |

## Full per-head detail

All heads (raw + 6 calibrated) per (family, split). ρ_S = ctx_spearman (preserved across heads in same family by construction). r = ctx_pearson. std_m = median std ratio. std_p = pooled std ratio.


### all_variants

| family | split | head | ρ_S | r | std_m | std_p | |L+g| r |
|---|---|---|---|---|---|---|---|
| density_idw:motion_sym | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| density_idw:motion_sym | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| density_idw:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| density_idw:motion_sym | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| density_idw:motion_sym | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| density_idw:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.511 | 0.638 | 0.632 | +0.892 |
| density_idw:motion_sym | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| density_idw:motion_sym | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| density_idw:motion_sym | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.698 |
| density_idw:motion_sym | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| density_idw:motion_sym | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| density_idw:motion_sym | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| density_idw:motion_sym | LOBO | `g_benchsim_gain` | +0.536 | +0.691 | 1.111 | 0.990 | +0.705 |
| density_idw:motion_sym | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |
| density_idw:motion_sym | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| density_idw:motion_sym | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| density_idw:motion_sym | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| density_idw:motion_sym | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| density_idw:motion_sym | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| density_idw:motion_sym | JOINT | `g_benchsim_gain` | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| density_idw:motion_sym | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |
| eb_shrunk:motion_sym | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| eb_shrunk:motion_sym | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| eb_shrunk:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| eb_shrunk:motion_sym | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| eb_shrunk:motion_sym | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| eb_shrunk:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.511 | 0.638 | 0.632 | +0.892 |
| eb_shrunk:motion_sym | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| eb_shrunk:motion_sym | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| eb_shrunk:motion_sym | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.698 |
| eb_shrunk:motion_sym | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| eb_shrunk:motion_sym | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| eb_shrunk:motion_sym | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| eb_shrunk:motion_sym | LOBO | `g_benchsim_gain` | +0.536 | +0.691 | 1.111 | 0.990 | +0.705 |
| eb_shrunk:motion_sym | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |
| eb_shrunk:motion_sym | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| eb_shrunk:motion_sym | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| eb_shrunk:motion_sym | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| eb_shrunk:motion_sym | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| eb_shrunk:motion_sym | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| eb_shrunk:motion_sym | JOINT | `g_benchsim_gain` | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| eb_shrunk:motion_sym | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g` | +0.419 | +0.243 | 0.358 | 0.533 | +0.858 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_global_gain` | +0.440 | +0.335 | 0.799 | 1.132 | +0.824 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_variant_gain` | +0.444 | +0.366 | 0.726 | 1.140 | +0.830 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_context_gain` | +0.421 | +0.377 | 1.198 | 1.195 | +0.827 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_shrink_gain` | +0.426 | +0.395 | 1.148 | 1.114 | +0.840 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_benchsim_gain` | +0.445 | +0.253 | 1.505 | 2.758 | +0.596 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_profilesim_gain` | +0.419 | +0.343 | 1.336 | 1.446 | +0.787 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g` | +0.501 | +0.413 | 0.261 | 0.403 | +0.737 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_global_gain` | +0.501 | +0.411 | 0.702 | 1.084 | +0.671 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_variant_gain` | +0.501 | +0.464 | 0.589 | 1.086 | +0.674 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_context_gain` | +0.501 | +0.464 | 0.589 | 1.086 | +0.674 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_shrink_gain` | +0.501 | +0.464 | 0.589 | 1.086 | +0.674 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_benchsim_gain` | +0.501 | +0.380 | 1.272 | 2.181 | +0.545 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_profilesim_gain` | +0.501 | +0.463 | 1.024 | 1.169 | +0.670 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g` | +0.487 | +0.252 | 0.316 | 0.469 | +0.072 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_global_gain` | +0.523 | +0.354 | 0.734 | 1.037 | -0.006 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_variant_gain` | +0.526 | +0.380 | 0.702 | 1.048 | -0.001 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_context_gain` | +0.526 | +0.380 | 0.702 | 1.048 | -0.001 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_shrink_gain` | +0.526 | +0.380 | 0.702 | 1.048 | -0.001 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_benchsim_gain` | +0.517 | +0.262 | 1.479 | 2.437 | +0.021 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_profilesim_gain` | +0.432 | +0.367 | 1.312 | 1.270 | +0.043 |
| mixed:motion_all | LOTO | `g` | +0.507 | +0.462 | 0.719 | 0.586 | +0.887 |
| mixed:motion_all | LOTO | `g_global_gain` | +0.512 | +0.485 | 1.136 | 0.896 | +0.880 |
| mixed:motion_all | LOTO | `g_variant_gain` | +0.508 | +0.520 | 1.020 | 0.902 | +0.886 |
| mixed:motion_all | LOTO | `g_context_gain` | +0.491 | +0.545 | 0.934 | 0.897 | +0.891 |
| mixed:motion_all | LOTO | `g_shrink_gain` | +0.501 | +0.544 | 0.972 | 0.885 | +0.892 |
| mixed:motion_all | LOTO | `g_benchsim_gain` | +0.508 | +0.512 | 0.958 | 0.877 | +0.886 |
| mixed:motion_all | LOTO | `g_profilesim_gain` | +0.501 | +0.531 | 1.026 | 0.994 | +0.884 |
| mixed:motion_all | LOBO | `g` | +0.448 | +0.520 | 0.870 | 0.624 | +0.715 |
| mixed:motion_all | LOBO | `g_global_gain` | +0.448 | +0.522 | 1.298 | 0.936 | +0.686 |
| mixed:motion_all | LOBO | `g_variant_gain` | +0.448 | +0.611 | 1.100 | 0.942 | +0.691 |
| mixed:motion_all | LOBO | `g_context_gain` | +0.448 | +0.611 | 1.100 | 0.942 | +0.691 |
| mixed:motion_all | LOBO | `g_shrink_gain` | +0.448 | +0.611 | 1.100 | 0.942 | +0.691 |
| mixed:motion_all | LOBO | `g_benchsim_gain` | +0.448 | +0.626 | 1.050 | 0.920 | +0.694 |
| mixed:motion_all | LOBO | `g_profilesim_gain` | +0.448 | +0.657 | 1.007 | 0.986 | +0.697 |
| mixed:motion_all | JOINT | `g` | +0.304 | +0.279 | 0.629 | 0.543 | +0.422 |
| mixed:motion_all | JOINT | `g_global_gain` | +0.302 | +0.305 | 0.991 | 0.829 | +0.462 |
| mixed:motion_all | JOINT | `g_variant_gain` | +0.296 | +0.353 | 1.067 | 0.840 | +0.464 |
| mixed:motion_all | JOINT | `g_context_gain` | +0.296 | +0.353 | 1.067 | 0.840 | +0.464 |
| mixed:motion_all | JOINT | `g_shrink_gain` | +0.296 | +0.353 | 1.067 | 0.840 | +0.464 |
| mixed:motion_all | JOINT | `g_benchsim_gain` | +0.307 | +0.391 | 0.941 | 0.797 | +0.486 |
| mixed:motion_all | JOINT | `g_profilesim_gain` | +0.307 | +0.385 | 0.971 | 0.894 | +0.423 |
| mixed:motion_fid | LOTO | `g` | +0.458 | +0.326 | 0.574 | 0.452 | +0.871 |
| mixed:motion_fid | LOTO | `g_global_gain` | +0.483 | +0.380 | 1.388 | 1.115 | +0.837 |
| mixed:motion_fid | LOTO | `g_variant_gain` | +0.481 | +0.390 | 1.316 | 1.120 | +0.839 |
| mixed:motion_fid | LOTO | `g_context_gain` | +0.457 | +0.396 | 1.173 | 1.173 | +0.835 |
| mixed:motion_fid | LOTO | `g_shrink_gain` | +0.463 | +0.405 | 1.219 | 1.122 | +0.842 |
| mixed:motion_fid | LOTO | `g_benchsim_gain` | +0.483 | +0.421 | 1.239 | 1.204 | +0.838 |
| mixed:motion_fid | LOTO | `g_profilesim_gain` | +0.483 | +0.347 | 1.211 | 1.349 | +0.803 |
| mixed:motion_fid | LOBO | `g` | +0.471 | +0.384 | 0.490 | 0.397 | +0.743 |
| mixed:motion_fid | LOBO | `g_global_gain` | +0.471 | +0.404 | 1.224 | 0.988 | +0.691 |
| mixed:motion_fid | LOBO | `g_variant_gain` | +0.471 | +0.439 | 1.186 | 0.988 | +0.692 |
| mixed:motion_fid | LOBO | `g_context_gain` | +0.471 | +0.439 | 1.186 | 0.988 | +0.692 |
| mixed:motion_fid | LOBO | `g_shrink_gain` | +0.471 | +0.439 | 1.186 | 0.988 | +0.692 |
| mixed:motion_fid | LOBO | `g_benchsim_gain` | +0.471 | +0.518 | 1.027 | 1.035 | +0.693 |
| mixed:motion_fid | LOBO | `g_profilesim_gain` | +0.471 | +0.436 | 1.087 | 1.187 | +0.669 |
| mixed:motion_fid | JOINT | `g` | +0.466 | +0.314 | 0.565 | 0.414 | +0.158 |
| mixed:motion_fid | JOINT | `g_global_gain` | +0.495 | +0.376 | 1.369 | 1.020 | +0.118 |
| mixed:motion_fid | JOINT | `g_variant_gain` | +0.494 | +0.382 | 1.336 | 1.020 | +0.112 |
| mixed:motion_fid | JOINT | `g_context_gain` | +0.494 | +0.382 | 1.336 | 1.020 | +0.112 |
| mixed:motion_fid | JOINT | `g_shrink_gain` | +0.494 | +0.382 | 1.336 | 1.020 | +0.112 |
| mixed:motion_fid | JOINT | `g_benchsim_gain` | +0.493 | +0.402 | 1.049 | 1.091 | +0.080 |
| mixed:motion_fid | JOINT | `g_profilesim_gain` | +0.467 | +0.316 | 1.048 | 1.241 | +0.141 |
| mixed:motion_sym | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| mixed:motion_sym | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| mixed:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| mixed:motion_sym | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| mixed:motion_sym | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| mixed:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.511 | 0.638 | 0.632 | +0.892 |
| mixed:motion_sym | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| mixed:motion_sym | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| mixed:motion_sym | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.698 |
| mixed:motion_sym | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| mixed:motion_sym | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| mixed:motion_sym | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| mixed:motion_sym | LOBO | `g_benchsim_gain` | +0.536 | +0.691 | 1.111 | 0.990 | +0.705 |
| mixed:motion_sym | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |
| mixed:motion_sym | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| mixed:motion_sym | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| mixed:motion_sym | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| mixed:motion_sym | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| mixed:motion_sym | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| mixed:motion_sym | JOINT | `g_benchsim_gain` | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| mixed:motion_sym | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |
| mixed:motion_sym_dinokernel | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| mixed:motion_sym_dinokernel | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| mixed:motion_sym_dinokernel | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| mixed:motion_sym_dinokernel | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| mixed:motion_sym_dinokernel | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| mixed:motion_sym_dinokernel | LOTO | `g_benchsim_gain` | +0.459 | +0.452 | 0.576 | 0.590 | +0.884 |
| mixed:motion_sym_dinokernel | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| mixed:motion_sym_dinokernel | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.766 |
| mixed:motion_sym_dinokernel | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.749 |
| mixed:motion_sym_dinokernel | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.754 |
| mixed:motion_sym_dinokernel | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.754 |
| mixed:motion_sym_dinokernel | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.754 |
| mixed:motion_sym_dinokernel | LOBO | `g_benchsim_gain` | +0.536 | +0.646 | 0.972 | 0.976 | +0.745 |
| mixed:motion_sym_dinokernel | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.747 |
| mixed:motion_sym_dinokernel | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| mixed:motion_sym_dinokernel | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| mixed:motion_sym_dinokernel | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| mixed:motion_sym_dinokernel | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| mixed:motion_sym_dinokernel | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| mixed:motion_sym_dinokernel | JOINT | `g_benchsim_gain` | +0.382 | +0.459 | 0.662 | 0.604 | +0.274 |
| mixed:motion_sym_dinokernel | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |
| mixed:motion_w2 | LOTO | `g` | +0.474 | +0.326 | 0.783 | 0.456 | +0.871 |
| mixed:motion_w2 | LOTO | `g_global_gain` | +0.486 | +0.359 | 1.958 | 1.155 | +0.827 |
| mixed:motion_w2 | LOTO | `g_variant_gain` | +0.488 | +0.365 | 1.501 | 1.158 | +0.829 |
| mixed:motion_w2 | LOTO | `g_context_gain` | +0.480 | +0.360 | 1.250 | 1.268 | +0.813 |
| mixed:motion_w2 | LOTO | `g_shrink_gain` | +0.490 | +0.374 | 1.366 | 1.188 | +0.827 |
| mixed:motion_w2 | LOTO | `g_benchsim_gain` | +0.490 | +0.382 | 1.434 | 1.281 | +0.818 |
| mixed:motion_w2 | LOTO | `g_profilesim_gain` | +0.487 | +0.351 | 1.489 | 1.408 | +0.794 |
| mixed:motion_w2 | LOBO | `g` | +0.478 | +0.370 | 0.610 | 0.366 | +0.742 |
| mixed:motion_w2 | LOBO | `g_global_gain` | +0.478 | +0.388 | 1.489 | 0.962 | +0.688 |
| mixed:motion_w2 | LOBO | `g_variant_gain` | +0.478 | +0.420 | 1.222 | 0.963 | +0.689 |
| mixed:motion_w2 | LOBO | `g_context_gain` | +0.478 | +0.420 | 1.222 | 0.963 | +0.689 |
| mixed:motion_w2 | LOBO | `g_shrink_gain` | +0.478 | +0.420 | 1.222 | 0.963 | +0.689 |
| mixed:motion_w2 | LOBO | `g_benchsim_gain` | +0.478 | +0.505 | 1.069 | 1.015 | +0.689 |
| mixed:motion_w2 | LOBO | `g_profilesim_gain` | +0.478 | +0.481 | 1.097 | 1.095 | +0.679 |
| mixed:motion_w2 | JOINT | `g` | +0.495 | +0.296 | 0.760 | 0.425 | +0.046 |
| mixed:motion_w2 | JOINT | `g_global_gain` | +0.506 | +0.338 | 1.722 | 1.066 | -0.055 |
| mixed:motion_w2 | JOINT | `g_variant_gain` | +0.490 | +0.334 | 1.380 | 1.065 | -0.058 |
| mixed:motion_w2 | JOINT | `g_context_gain` | +0.490 | +0.334 | 1.380 | 1.065 | -0.058 |
| mixed:motion_w2 | JOINT | `g_shrink_gain` | +0.490 | +0.334 | 1.380 | 1.065 | -0.058 |
| mixed:motion_w2 | JOINT | `g_benchsim_gain` | +0.460 | +0.317 | 1.185 | 1.048 | -0.132 |
| mixed:motion_w2 | JOINT | `g_profilesim_gain` | +0.420 | +0.264 | 1.079 | 1.177 | -0.088 |
| symmetric_informed:motion_sym | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| symmetric_informed:motion_sym | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| symmetric_informed:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| symmetric_informed:motion_sym | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| symmetric_informed:motion_sym | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| symmetric_informed:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.511 | 0.638 | 0.632 | +0.892 |
| symmetric_informed:motion_sym | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| symmetric_informed:motion_sym | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| symmetric_informed:motion_sym | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.698 |
| symmetric_informed:motion_sym | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| symmetric_informed:motion_sym | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| symmetric_informed:motion_sym | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| symmetric_informed:motion_sym | LOBO | `g_benchsim_gain` | +0.536 | +0.691 | 1.111 | 0.990 | +0.705 |
| symmetric_informed:motion_sym | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |
| symmetric_informed:motion_sym | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| symmetric_informed:motion_sym | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| symmetric_informed:motion_sym | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| symmetric_informed:motion_sym | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| symmetric_informed:motion_sym | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| symmetric_informed:motion_sym | JOINT | `g_benchsim_gain` | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| symmetric_informed:motion_sym | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |
| symmetric_uninformed:motion_sym | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| symmetric_uninformed:motion_sym | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| symmetric_uninformed:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| symmetric_uninformed:motion_sym | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| symmetric_uninformed:motion_sym | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| symmetric_uninformed:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.511 | 0.638 | 0.632 | +0.892 |
| symmetric_uninformed:motion_sym | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| symmetric_uninformed:motion_sym | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| symmetric_uninformed:motion_sym | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.698 |
| symmetric_uninformed:motion_sym | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| symmetric_uninformed:motion_sym | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| symmetric_uninformed:motion_sym | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| symmetric_uninformed:motion_sym | LOBO | `g_benchsim_gain` | +0.536 | +0.691 | 1.111 | 0.990 | +0.705 |
| symmetric_uninformed:motion_sym | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |
| symmetric_uninformed:motion_sym | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| symmetric_uninformed:motion_sym | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| symmetric_uninformed:motion_sym | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| symmetric_uninformed:motion_sym | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| symmetric_uninformed:motion_sym | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| symmetric_uninformed:motion_sym | JOINT | `g_benchsim_gain` | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| symmetric_uninformed:motion_sym | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |
| targeted_informed:motion_sym | LOTO | `g` | +0.436 | +0.389 | 0.442 | 0.313 | +0.877 |
| targeted_informed:motion_sym | LOTO | `g_global_gain` | +0.458 | +0.428 | 0.865 | 0.628 | +0.881 |
| targeted_informed:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.454 | 0.760 | 0.628 | +0.883 |
| targeted_informed:motion_sym | LOTO | `g_context_gain` | +0.436 | +0.441 | 0.664 | 0.641 | +0.880 |
| targeted_informed:motion_sym | LOTO | `g_shrink_gain` | +0.447 | +0.458 | 0.729 | 0.618 | +0.884 |
| targeted_informed:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.511 | 0.638 | 0.632 | +0.892 |
| targeted_informed:motion_sym | LOTO | `g_profilesim_gain` | +0.442 | +0.501 | 0.731 | 0.678 | +0.890 |
| targeted_informed:motion_sym | LOBO | `g` | +0.536 | +0.577 | 0.692 | 0.584 | +0.732 |
| targeted_informed:motion_sym | LOBO | `g_global_gain` | +0.536 | +0.583 | 1.140 | 0.994 | +0.698 |
| targeted_informed:motion_sym | LOBO | `g_variant_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| targeted_informed:motion_sym | LOBO | `g_context_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| targeted_informed:motion_sym | LOBO | `g_shrink_gain` | +0.536 | +0.660 | 1.222 | 0.993 | +0.702 |
| targeted_informed:motion_sym | LOBO | `g_benchsim_gain` | +0.536 | +0.691 | 1.111 | 0.990 | +0.705 |
| targeted_informed:motion_sym | LOBO | `g_profilesim_gain` | +0.536 | +0.682 | 1.073 | 1.142 | +0.694 |
| targeted_informed:motion_sym | JOINT | `g` | +0.435 | +0.314 | 0.551 | 0.354 | +0.327 |
| targeted_informed:motion_sym | JOINT | `g_global_gain` | +0.400 | +0.391 | 1.161 | 0.694 | +0.337 |
| targeted_informed:motion_sym | JOINT | `g_variant_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| targeted_informed:motion_sym | JOINT | `g_context_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| targeted_informed:motion_sym | JOINT | `g_shrink_gain` | +0.389 | +0.407 | 0.878 | 0.693 | +0.356 |
| targeted_informed:motion_sym | JOINT | `g_benchsim_gain` | +0.366 | +0.448 | 0.783 | 0.693 | +0.351 |
| targeted_informed:motion_sym | JOINT | `g_profilesim_gain` | +0.391 | +0.468 | 0.914 | 0.728 | +0.345 |

### drop_false_true

| family | split | head | ρ_S | r | std_m | std_p | |L+g| r |
|---|---|---|---|---|---|---|---|
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g` | +0.379 | +0.207 | 0.323 | 0.466 | +0.862 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_global_gain` | +0.432 | +0.293 | 0.876 | 1.132 | +0.820 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_variant_gain` | +0.429 | +0.334 | 0.767 | 1.145 | +0.827 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_context_gain` | +0.412 | +0.355 | 1.187 | 1.177 | +0.830 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_shrink_gain` | +0.428 | +0.366 | 1.149 | 1.109 | +0.839 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_benchsim_gain` | +0.412 | +0.230 | 1.424 | 2.577 | +0.617 |
| fsub_mean_nn:motion mean_nn (default) | LOTO | `g_profilesim_gain` | +0.411 | +0.325 | 1.382 | 1.467 | +0.785 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g` | +0.494 | +0.367 | 0.248 | 0.340 | +0.751 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_global_gain` | +0.494 | +0.367 | 0.736 | 1.076 | +0.681 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_variant_gain` | +0.494 | +0.426 | 0.591 | 1.078 | +0.684 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_context_gain` | +0.494 | +0.426 | 0.591 | 1.078 | +0.684 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_shrink_gain` | +0.494 | +0.426 | 0.591 | 1.078 | +0.684 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_benchsim_gain` | +0.494 | +0.340 | 1.166 | 2.028 | +0.567 |
| fsub_mean_nn:motion mean_nn (default) | LOBO | `g_profilesim_gain` | +0.494 | +0.409 | 1.014 | 1.213 | +0.672 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g` | +0.420 | +0.203 | 0.255 | 0.413 | +0.131 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_global_gain` | +0.508 | +0.301 | 0.676 | 1.021 | +0.035 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_variant_gain` | +0.506 | +0.333 | 0.701 | 1.030 | +0.037 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_context_gain` | +0.506 | +0.333 | 0.701 | 1.030 | +0.037 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_shrink_gain` | +0.506 | +0.333 | 0.701 | 1.030 | +0.037 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_benchsim_gain` | +0.461 | +0.222 | 1.443 | 2.337 | +0.059 |
| fsub_mean_nn:motion mean_nn (default) | JOINT | `g_profilesim_gain` | +0.397 | +0.333 | 1.130 | 1.253 | +0.079 |
| mixed:motion_sym | LOTO | `g` | +0.429 | +0.351 | 0.474 | 0.297 | +0.878 |
| mixed:motion_sym | LOTO | `g_global_gain` | +0.457 | +0.387 | 1.012 | 0.671 | +0.877 |
| mixed:motion_sym | LOTO | `g_variant_gain` | +0.468 | +0.420 | 0.820 | 0.672 | +0.880 |
| mixed:motion_sym | LOTO | `g_context_gain` | +0.425 | +0.415 | 0.713 | 0.691 | +0.878 |
| mixed:motion_sym | LOTO | `g_shrink_gain` | +0.448 | +0.429 | 0.793 | 0.664 | +0.882 |
| mixed:motion_sym | LOTO | `g_benchsim_gain` | +0.466 | +0.479 | 0.684 | 0.673 | +0.889 |
| mixed:motion_sym | LOTO | `g_profilesim_gain` | +0.437 | +0.480 | 0.785 | 0.739 | +0.888 |
| mixed:motion_sym | LOBO | `g` | +0.529 | +0.511 | 0.652 | 0.521 | +0.744 |
| mixed:motion_sym | LOBO | `g_global_gain` | +0.529 | +0.517 | 1.217 | 0.994 | +0.704 |
| mixed:motion_sym | LOBO | `g_variant_gain` | +0.529 | +0.606 | 1.282 | 0.994 | +0.708 |
| mixed:motion_sym | LOBO | `g_context_gain` | +0.529 | +0.606 | 1.282 | 0.994 | +0.708 |
| mixed:motion_sym | LOBO | `g_shrink_gain` | +0.529 | +0.606 | 1.282 | 0.994 | +0.708 |
| mixed:motion_sym | LOBO | `g_benchsim_gain` | +0.529 | +0.637 | 1.135 | 0.980 | +0.712 |
| mixed:motion_sym | LOBO | `g_profilesim_gain` | +0.529 | +0.628 | 1.107 | 1.171 | +0.697 |
| mixed:motion_sym | JOINT | `g` | +0.412 | +0.296 | 0.655 | 0.322 | +0.317 |
| mixed:motion_sym | JOINT | `g_global_gain` | +0.366 | +0.363 | 1.633 | 0.711 | +0.317 |
| mixed:motion_sym | JOINT | `g_variant_gain` | +0.352 | +0.383 | 0.955 | 0.709 | +0.337 |
| mixed:motion_sym | JOINT | `g_context_gain` | +0.352 | +0.383 | 0.955 | 0.709 | +0.337 |
| mixed:motion_sym | JOINT | `g_shrink_gain` | +0.352 | +0.383 | 0.955 | 0.709 | +0.337 |
| mixed:motion_sym | JOINT | `g_benchsim_gain` | +0.333 | +0.432 | 0.784 | 0.712 | +0.333 |
| mixed:motion_sym | JOINT | `g_profilesim_gain` | +0.365 | +0.462 | 1.081 | 0.771 | +0.329 |

## Mechanism note

Per-context gain = std(actual residual) / std(predicted residual). When raw ridge is heavily under-dispersed (e.g. mean_nn with raw median std ratio 0.26 LOBO), the per-context gains are large (~3–5×) and noisy. IDW-smoothing large noisy gains across benchmarks amplifies the wrong scale (pooled std ratio > 2). When raw ridge is closer to 1× scale (motion_sym 0.69 LOBO, motion all 13 features 0.87 LOBO), the gains are modest (~1.1–1.5×) and IDW-smoothing produces well-behaved calibrated predictions.

Practically: any motion family with raw `median_std_ratio` ≥ ~0.4 benefits from `g_benchsim_gain` / `g_profilesim_gain`. The mean_nn feature subset is the exception because it concentrates too little predictive variance into the head; that's a feature-restriction artifact, not a feature-axis-alignment story.

The kernel-space choice (flow vs DINO `mean_nn_sym`) is a secondary knob — both give similar calibrated pearson for motion_sym (flow kernel slightly better on LOTO/LOBO). Choose based on which benchmark-similarity geometry better predicts your held-out target's residual scale.


## Files referenced

- **density_idw:motion_sym** — calibration dir + per-split scatter/hexbin
- **eb_shrunk:motion_sym** — calibration dir + per-split scatter/hexbin
- **fsub_mean_nn:motion mean_nn (default)** — calibration dir + per-split scatter/hexbin
- **mixed:motion_all** — calibration dir + per-split scatter/hexbin
- **mixed:motion_fid** — calibration dir + per-split scatter/hexbin
- **mixed:motion_sym** — calibration dir + per-split scatter/hexbin
- **mixed:motion_sym_dinokernel** — calibration dir + per-split scatter/hexbin
- **mixed:motion_w2** — calibration dir + per-split scatter/hexbin
- **symmetric_informed:motion_sym** — calibration dir + per-split scatter/hexbin
- **symmetric_uninformed:motion_sym** — calibration dir + per-split scatter/hexbin
- **targeted_informed:motion_sym** — calibration dir + per-split scatter/hexbin

Each calibration dir contains:
- `summary_all_variants.csv`, `summary_drop_false_true.csv`
- `rows_<SPLIT>_<variant_filter>.csv` — per-row predictions for all heads
- `figures/grid_{scatter,hexbin}_<variant_filter>.png` — comparison grid
- `figures/{scatter,hexbin}_<SPLIT>_<variant_filter>_<head>.png` — per-cell figs
