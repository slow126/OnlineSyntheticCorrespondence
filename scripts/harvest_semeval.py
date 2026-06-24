"""Harvest the full-semantic-suite comparison: peak PCK per benchmark for each
source x regime. Rows = source x {TF,TT}, cols = {TSS, PF-Pascal, PF-Willow, SPair}."""
import glob
import pandas as pd

ROOT = "/mnt/nvme_1tb_a/snapshots"
BENCH = ["tss", "pfpascal", "pfwillow", "spair"]
SRC = [("ns4", "clean 4obj near-static"), ("mm4", "clean 4obj ~43px"),
       ("movif", "MOVi-F (~2px) baseline"), ("tssv2", "cluttered 14obj ~50px")]


def peak(s, reg, b):
    d = glob.glob(f"{ROOT}/semeval_{s}_{reg}/*/validation_results.csv")
    if not d:
        return None
    try:
        df = pd.read_csv(sorted(d)[-1])
    except Exception:
        return None
    x = df[df.benchmark == b]
    return float(x.pck.max()) if len(x) else None


print("=" * 78)
print("FULL SEMANTIC SUITE — peak PCK@0.05 (CATs++, 1000 scenes)")
print("=" * 78)
hdr = f"{'source':<24}{'reg':<5}" + "".join(f"{b:>11}" for b in BENCH)
print(hdr)
print("-" * len(hdr))
for s, label in SRC:
    for reg in ["tf", "tt"]:
        row = [peak(s, reg, b) for b in BENCH]
        cells = "".join((f"{v:11.1f}" if v is not None else f"{'n/a':>11}") for v in row)
        tag = label if reg == "tf" else ""
        print(f"{tag:<24}{reg:<5}{cells}")
print("=" * 78)
print("Read: does ns4's near-static win generalize across ALL semantic benchmarks,")
print("or is it TSS-specific? Does mm4 (large motion) crash on all of them?")
