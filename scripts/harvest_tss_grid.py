#!/usr/bin/env python3
"""Harvest the TSS design grid: tss_v1 (tuned source) vs MOVi-F, x
{CATs++, GLU-Net, FlowFormer}, pretrained-frozen, eval on the semantic suite.
Prints the make-or-break comparison (does the tuned TSS source beat the generic
MOVi-F baseline on TSS?) and the PF-Pascal / PF-Willow secondaries."""
import glob, sys
import pandas as pd

ROOT = "/mnt/nvme_1tb_a/snapshots"
ARCHS = ["catspp", "glunet", "flowformer"]
BENCHES = ["tss", "pfpascal", "pfwillow"]
# transfer_table MOVi-F TSS (pretrained-frozen) reference, for sanity vs the re-trained cell
REF_MOVIF_TSS = {"catspp": 55.65, "glunet": 19.71, "flowformer": None}


def peak(source, arch, bench):
    dirs = glob.glob(f"{ROOT}/tssgrid_{source}_{arch}_tt/*/validation_results.csv")
    if not dirs:
        return None
    try:
        d = pd.read_csv(sorted(dirs)[-1])
    except Exception:
        return None
    s = d[d.benchmark == bench]
    return float(s.pck.max()) if len(s) else None


def fmt(x):
    return f"{x:6.2f}" if isinstance(x, float) else "   n/a"


print("=" * 72)
print("TSS DESIGN GRID — tss_v1 (tuned) vs MOVi-F  |  pretrained-frozen (TT)")
print("=" * 72)
for bench in BENCHES:
    print(f"\n### {bench.upper()}  (peak PCK@0.05)")
    print(f"{'arch':<12}{'tss_v1':>9}{'MOVi-F':>9}{'Δ':>9}   verdict")
    print("-" * 52)
    for arch in ARCHS:
        t = peak("tss_v1", arch, bench)
        m = peak("movif", arch, bench)
        delta = (t - m) if (t is not None and m is not None) else None
        verdict = ""
        if delta is not None:
            verdict = "tss_v1 WINS" if delta > 0 else "MOVi-F wins"
        elif t is None and m is None:
            verdict = "(both missing)"
        elif t is None:
            verdict = "(tss_v1 missing)"
        else:
            verdict = "(MOVi-F missing)"
        print(f"{arch:<12}{fmt(t):>9}{fmt(m):>9}{fmt(delta):>9}   {verdict}")
    if bench == "tss":
        print("  ref (transfer_table MOVi-F TSS TT): "
              + ", ".join(f"{a}={REF_MOVIF_TSS[a]}" for a in ARCHS if REF_MOVIF_TSS[a]))
print("\n" + "=" * 72)
print("Make-or-break: a positive Δ on TSS for any architecture = the motion-tuned")
print("source beats the generic baseline on a SEMANTIC target = a second")
print("interventional leg beyond KITTI. CATs++ is the strongest semantic matcher.")
print("=" * 72)
