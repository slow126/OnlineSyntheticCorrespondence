"""
Per-(training source) static-prior probe, swept over EVERY benchmark.

For each scratch model (family x training source), run the id/real probe on every
benchmark frame set we have, and emit one row per (family, source, benchmark). The
table is then built by averaging id/real over benchmarks per source, with rows
grouped by source motion-type. Fast: id/real only (no O(n^2) source-sensitivity).

id/real = mean||flow(A,A)|| / mean||flow(A,B)||  (->1 = input-blind static prior).
"""
import argparse, glob, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')
from scripts.diagnose_degeneracy_byfamily import (
    build_and_load, predict, mag, realmotion_pairs, spair_pairs, diagnostics,
    parse_config, parse_source, DEV)
import torch


def metrics(m, TRG, SRC):
    """Returns (id/real ratio, relative source-sensitivity)."""
    idr, sens, fmag, rough = diagnostics(m, TRG, SRC)
    return idr, sens / (fmag + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=['cats', 'glunet', 'flowformer', 'raft'])
    ap.add_argument('--glob', required=True)
    ap.add_argument('--ckpt-name', default='model_best.pth')
    ap.add_argument('--benchmarks', required=True,
                    help='comma-sep name:dir of real-motion benchmark frame sets')
    ap.add_argument('--spair', default='', help='SPair root (adds a semantic benchmark)')
    ap.add_argument('--n-pairs', type=int, default=12)
    args = ap.parse_args()

    benches = []
    for spec in args.benchmarks.split(','):
        name, d = spec.split(':')
        benches.append((name, realmotion_pairs(d, args.n_pairs)))
    if args.spair:
        benches.append(('spair', spair_pairs(args.spair, args.n_pairs)))
    print(f"# device={DEV} benchmarks={[b[0] for b in benches]}", file=sys.stderr)
    print("family,source,backbone,encoder,benchmark,id_real_ratio,rel_sens")

    for d in sorted(glob.glob(args.glob)):
        cfg = parse_config(d)
        if cfg is None:
            continue
        backbone, encoder = cfg
        source = parse_source(d, args.family)
        ckpt = os.path.join(d, args.ckpt_name)
        if not os.path.exists(ckpt):
            print(f"  [skip] no ckpt {d}", file=sys.stderr); continue
        try:
            m = build_and_load(args.family, ckpt)
        except Exception as e:
            print(f"  [err] {d}: {e}", file=sys.stderr); continue
        for name, (T, S) in benches:
            try:
                idr, rs = metrics(m, T, S)
                print(f"{args.family},{source},{backbone},{encoder},{name},{idr:.4f},{rs:.4f}")
                sys.stdout.flush()
            except Exception as e:
                print(f"  [err] {name} {d}: {e}", file=sys.stderr)
        del m
        if DEV == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
