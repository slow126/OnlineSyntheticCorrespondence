#!/usr/bin/env python3
"""Strip optimizer + scheduler state from saved .pth checkpoints in place.

These tensors (AdamW momentum buffers, etc.) are only needed to RESUME training,
which is not wired in this project. Removing them keeps the model weights intact
-- eval/inference loads identically -- while cutting each checkpoint to ~1/3 size.

Usage:
    # dry run: report how much would be reclaimed, change nothing
    python scripts/strip_optimizer_from_checkpoints.py snapshots --dry-run

    # do it
    python scripts/strip_optimizer_from_checkpoints.py snapshots
"""
import argparse
import glob
import os
import sys
import torch

STRIP_KEYS = ('optimizer', 'scheduler')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='directory to search recursively for *.pth')
    ap.add_argument('--dry-run', action='store_true', help='report only, do not rewrite')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, '**', '*.pth'), recursive=True))
    if not files:
        print(f'No .pth files under {args.root}')
        return

    print(f'Found {len(files)} checkpoint(s) under {args.root}')
    before_total = 0
    after_total = 0
    skipped = 0
    changed = 0

    for i, f in enumerate(files, 1):
        size_before = os.path.getsize(f)
        before_total += size_before
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f'  [skip] {f}: load failed ({e})')
            skipped += 1
            after_total += size_before
            continue

        if not isinstance(ckpt, dict):
            # Bare state_dict -- nothing to strip.
            after_total += size_before
            continue

        had = [k for k in STRIP_KEYS if ckpt.get(k) is not None]
        if not had:
            after_total += size_before
            continue

        if args.dry_run:
            after_total += size_before  # unknown post size in dry run; report by-key below
            print(f'  [{i}/{len(files)}] would strip {had} from {f} '
                  f'({size_before / 1e6:.0f} MB)')
            changed += 1
            continue

        for k in STRIP_KEYS:
            if k in ckpt:
                ckpt[k] = None
        tmp = f + '.tmp'
        torch.save(ckpt, tmp)
        os.replace(tmp, f)  # atomic; never leaves a half-written checkpoint
        size_after = os.path.getsize(f)
        after_total += size_after
        changed += 1
        print(f'  [{i}/{len(files)}] {os.path.basename(f)}: '
              f'{size_before / 1e6:.0f} -> {size_after / 1e6:.0f} MB')

    print('-' * 60)
    if args.dry_run:
        print(f'DRY RUN: {changed} file(s) carry optimizer/scheduler state.')
        print(f'Current total: {before_total / 1e9:.1f} GB. '
              f'Expect ~2/3 reclaimed after a real run.')
    else:
        print(f'Done. Rewrote {changed} file(s), skipped {skipped}.')
        print(f'Total: {before_total / 1e9:.1f} GB -> {after_total / 1e9:.1f} GB '
              f'(reclaimed {(before_total - after_total) / 1e9:.1f} GB)')


if __name__ == '__main__':
    sys.exit(main())
