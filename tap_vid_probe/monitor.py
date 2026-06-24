"""Monitor the TAP-Vid probe: per-run, per-benchmark PCK curves + peak epoch.

Reads every validation_results.csv under the probe snapshot base and, for each run,
prints a benchmark x epoch table of PCK plus the PEAK epoch per benchmark. The proxy
question is answered by comparing the peak-epoch ROW: does tapvid_davis peak at the same
epoch as kitti / tss / pf?

Run:  python tap_vid_probe/monitor.py
      watch -n 60 'python tap_vid_probe/monitor.py'
"""
import csv
import glob
import os
from collections import defaultdict

BASE = os.environ.get("TAPVID_PROBE_SNAP", "/mnt/nvme_1tb_a/tapvid_probe_snapshots")
BENCH_ORDER = ["tapvid_davis", "kitti2015", "kitti2012", "tss", "pfpascal", "pfwillow"]


def load(csv_path):
    # epoch -> benchmark -> pck   (validation_results.csv columns incl epoch,benchmark,pck)
    by_epoch = defaultdict(dict)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                e = int(float(row["epoch"])); b = row["benchmark"]; p = float(row["pck"])
            except (KeyError, ValueError):
                continue
            by_epoch[e][b] = p
    return by_epoch


def main():
    csvs = sorted(glob.glob(os.path.join(BASE, "*", "*", "validation_results.csv")))
    if not csvs:
        print(f"(no validation_results.csv yet under {BASE})")
        return
    for csv_path in csvs:
        run = csv_path.split(os.sep)[-3]  # snapshot group dir = model lane name
        by_epoch = load(csv_path)
        if not by_epoch:
            print(f"\n### {run}: (no rows yet)"); continue
        benches = [b for b in BENCH_ORDER if any(b in v for v in by_epoch.values())]
        benches += [b for e in by_epoch.values() for b in e if b not in benches and b not in BENCH_ORDER]
        epochs = sorted(by_epoch)
        peak = {b: max(((by_epoch[e].get(b, float("-inf")), e) for e in epochs)) for b in benches}

        print(f"\n### {run}   (epochs {epochs[0]}..{epochs[-1]}, {len(epochs)} vals)")
        hdr = "epoch " + " ".join(f"{b[:11]:>11}" for b in benches)
        print(hdr)
        for e in epochs:
            print(f"{e:5d} " + " ".join(
                f"{by_epoch[e].get(b, float('nan')):>11.2f}" for b in benches))
        print("PEAK@ " + " ".join(f"{('%.1f@e%d' % (peak[b][0], peak[b][1])):>11}" for b in benches))


if __name__ == "__main__":
    main()
