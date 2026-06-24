# Degeneracy probe: identity-pair flow test (2026-06-16)
CATs++ lolr best snapshots. Identical image pairs (src=trg) -> correct flow is ZERO.
Inputs: 16 kubric frames @512. flow @32x32. Harness: scripts/diagnose_degeneracy.py

model                       identity|flow|   real|flow|   id/real ratio
pretrained TF (HEALTHY)          0.33           0.90          0.36
movi_f FT (random-frozen enc)    0.28           0.74          0.38
movi_f FF (scratch, real)        0.83           1.18          0.70
spair  FF (scratch, semantic)   11.45          11.50          1.00   <- input-blind

Findings:
- spair-FF ratio 1.00: emits same field for identical & real pairs -> prior, not function (~35x healthy id-flow).
- movi_f-FF (0.70) WORSE than random-frozen FT (0.38) -> training the scratch encoder degrades it; explains FF<FT (-15 PCK) in tab:degenerate.
- low ratio is necessary-not-sufficient (identical inputs trivially -> ~0 even random); FF>FT is the load-bearing comparison.
