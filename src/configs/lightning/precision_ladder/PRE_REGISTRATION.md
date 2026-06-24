# Pre-registered: controlled off-target (precision) intervention — 2026-06-11

## Construction (no re-render)
`KubricInterventionDataset(mirror_flip=f)` replaces a fraction f of recovered pairs
with (frame0, horizontal-mirror(frame0)) + the exact mirror flow (W-1-2x, 0). This
injects strongly OFF-TARGET motion at FIXED appearance (mirrored texture = same
distribution) and FIXED coverage (the 1-f real pairs still span KITTI's motion).

Verified BFV ladder to KITTI (real extraction, kitti2012):
  f:        0.00   0.05   0.10   0.25   0.50
  d_TB(prec): 0.042  0.101  0.165  0.348  0.658   (16x ramp, monotone)
  d_BT(rec):  0.0017 0.0017 0.0017 0.0018 0.0018  (pinned)

## Prediction (stated before training)
Training CATs++ on the ladder, eval peak PCK@0.05 on kitti2012/kitti2015:
1. FF (from scratch): transfer DECREASES monotonically with f -- off-target motion
   corrupts a from-scratch model (CATs++/RAFT are precision-sensitive, established
   observationally + on the GLU-Net/FlowFormer cross-arch test).
2. TT (pretrained backbone, frozen): transfer is ~FLAT in f -- a frozen ImageNet
   backbone forgives off-target mass (the two-cost claim).
3. The CONTRAST (FF degradation slope >> TT degradation slope) is the result: it
   causally demonstrates off-target is a real cost AND that it is regime-specific.

## Falsification
- If TT degrades as steeply as FF -> "backbone forgives off-target" is WRONG.
- If FF is flat in f -> off-target is not a cost even from scratch (the whole
  precision half collapses; commit fully to the single coverage cost).

## Honest scope
This off-target is UNNATURAL (mirror motion) by necessity -- for a broad-support
target like KITTI, off-target mass is unconstructible with natural motion. So this
bounds off-target as "sensitivity to degenerate motion in scratch CATs++/RAFT," not
a co-equal natural cost. Coverage (missing support) remains the universal cost.
