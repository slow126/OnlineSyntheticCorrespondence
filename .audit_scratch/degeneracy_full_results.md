# Comprehensive degeneracy probe (2026-06-16) — 37 CATs++ models

Identity-pair & source-swap input-ablation on CATs++ best snapshots (lolr scratch FF/FT
from RC + local pretrained TF/TT). 32 fixed kubric frame-pairs @512, flow@32x32.
Harness: scripts/diagnose_degeneracy.py (full version) ; raw CSV: .audit_scratch/degeneracy_full.csv

Metric 1 — id/real ratio: |flow(img,img)| / |flow(frame1,frame0)|.  ->1 = input-blind (emits
  same field for identical & real pairs); ~0.3 = distinguishes them (healthy).
Metric 2 — source-sensitivity: |flow(t,s) - flow(t,s_swapped)| / |flow(t,s)|.  high = output
  depends on the source image (real matching); ~0 = ignores it.

## Config means
config                 id/real ratio   source-sensitivity
FF (scratch, trained)      0.85             1.95
FT (scratch, random)       0.73             2.77
TF (pretrained, ft)        0.26             9.67
TT (pretrained, frozen)    0.32             7.69

## id/real ratio  (->1 degenerate)
source                     FF    FT    TF    TT
movi_f                    0.71  0.35   --    --
flyingthings              0.84  0.65   --    --
pointodyssey              0.63  0.51  0.02  0.03
sintel                    0.55  0.54   --    --
synthetic                 0.97  0.87  0.35  0.36
synthetic_small_zoom      0.92  0.72  0.16  0.15
synthetic_large_zoom      0.90  0.78  0.23  0.25
synthetic_random_flipping 0.98  0.92  0.31  0.38
synthetic_2d_warp         0.91  0.88  0.37  0.46
imagenet2dwarp            0.93  0.85  0.38  0.39
spair                     1.00  1.00   --   0.53

## source-sensitivity  (high healthy)
source                     FF    FT    TF    TT
movi_f                    1.71  6.12   --    --
flyingthings              2.31  3.61   --    --
pointodyssey              3.20  4.55  9.78 11.28
sintel                    6.48  4.59   --    --
synthetic                 1.12  1.59 11.41  7.89
synthetic_small_zoom      1.36  2.72  9.60  8.22
synthetic_large_zoom      1.98  2.38  9.04  7.92
synthetic_random_flipping 0.53  1.02 15.39  9.84
synthetic_2d_warp         1.39  1.81  5.99  5.27
imagenet2dwarp            0.97  1.51  6.49  6.26
spair                     0.36  0.52   --   4.87

## Findings
- Pretrained (TF/TT) use the input: low id/ratio (~0.3), high sensitivity (~8-10).
- Scratch (FF/FT) near input-blind: high id/ratio (~0.8), low sensitivity (~2).
- FF WORSE than FT on both metrics (more input-blind, less sensitive) -> training the
  scratch encoder degrades it below random -> explains FF<FT (-15 PCK) in tab:degenerate.
- spair (semantic source) FF/FT = 1.00: fully input-blind, the extreme case.
