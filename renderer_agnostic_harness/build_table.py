#!/usr/bin/env python3
"""Assemble the renderer-agnostic LaTeX tables from per-cell transfer_cell_eval.csv.

Reads peak PCK (max over epochs) per (benchmark, alpha) for each cell, pairs
default-vs-tuned per architecture, and emits three tables matching the paper:
  - KITTI   : K2012 & K2015 at alpha in {0.05, 0.03, 0.01}   (tab_interv_fractal style)
  - TSS     : alpha in {0.10, 0.05, 0.03}
  - TAP-Vid : stride in {1, 2, 4, 8, 16}  (alpha 0.05)
"""
import os, csv, glob, argparse
from collections import defaultdict

ARCHS = [('cats', 'CATs++'), ('glunet', 'GLU-Net'), ('flowformer', 'FlowFormer')]


def load_peaks(cell_dir):
    """benchmark -> alpha -> peak pck (max over epochs)."""
    csv_path = os.path.join(cell_dir, 'transfer_cell_eval.csv')
    if not os.path.exists(csv_path):
        return None
    peak = defaultdict(dict)
    for r in csv.DictReader(open(csv_path)):
        b, a, p = r['benchmark'], float(r['alpha']), float(r['pck'])
        peak[b][a] = max(peak[b].get(a, -1.0), p)
    return peak


def find_cell_dir(snap_base, cell):
    dirs = sorted(glob.glob(os.path.join(snap_base, f'{cell}_*')), key=os.path.getmtime)
    return dirs[-1] if dirs else None


def fmt_delta(d):
    if d is None:
        return '--'
    s = f"{d:+.1f}"
    return f"\\textbf{{{s}}}" if d > 0 else f"${s}$"


def cell_val(peaks, bench, alpha):
    if peaks is None or bench not in peaks or alpha not in peaks[bench]:
        return None
    return peaks[bench][alpha]


def row(arch_label, dflt, tuned, columns):
    """columns = list of (bench, alpha). Emits default/tuned/Delta triplets."""
    cells = [arch_label]
    for bench, alpha in columns:
        dv = cell_val(dflt, bench, alpha)
        tv = cell_val(tuned, bench, alpha)
        dvs = f"{dv:.1f}" if dv is not None else '--'
        tvs = f"{tv:.1f}" if tv is not None else '--'
        delta = (tv - dv) if (dv is not None and tv is not None) else None
        cells += [dvs, tvs, fmt_delta(delta)]
    return ' & '.join(cells) + r' \\'


def table(title, columns, group_headers, peaks_by_cell):
    ncol = 1 + 3 * len(columns)
    colspec = 'l' + ' rrr' * len(columns)
    lines = [f"% {title}", r"\begin{tabular}{" + colspec + "}", r"\toprule"]
    # group header row
    gh = [' '] + [f"\\multicolumn{{3}}{{c}}{{{h}}}" for h in group_headers]
    lines.append(' & '.join(gh) + r' \\')
    cmid = ' '.join(f"\\cmidrule(lr){{{2+3*i}-{4+3*i}}}" for i in range(len(columns)))
    lines.append(cmid)
    sub = ['Arch.'] + ['default & tuned & $\\Delta$'] * len(columns)
    lines.append(' & '.join(sub) + r' \\')
    lines.append(r"\midrule")
    for arch, label in ARCHS:
        dflt = peaks_by_cell.get(f'{arch}_default')
        tuned = peaks_by_cell.get(f'{arch}_tuned')
        lines.append(row(label, dflt, tuned, columns))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snap-base', default='/mnt/nvme_1tb_a/renderer_agnostic')
    ap.add_argument('--cells', nargs='+', default=None)
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables'))
    args = ap.parse_args()

    cells = args.cells or [f'{a}_{s}' for a, _ in ARCHS for s in ('default', 'tuned')]
    peaks_by_cell = {}
    for cell in cells:
        d = find_cell_dir(args.snap_base, cell)
        peaks_by_cell[cell] = load_peaks(d) if d else None
        status = 'OK' if peaks_by_cell[cell] else 'MISSING'
        print(f"  {cell:<20} {status}  ({d})")

    os.makedirs(args.out_dir, exist_ok=True)

    kitti = table(
        "KITTI renderer-agnostic (peak PCK; default vs tuned=trial76)",
        columns=[('kitti2012', 0.05), ('kitti2012', 0.03), ('kitti2012', 0.01),
                 ('kitti2015', 0.05), ('kitti2015', 0.03), ('kitti2015', 0.01)],
        group_headers=['K2012@.05', 'K2012@.03', 'K2012@.01',
                       'K2015@.05', 'K2015@.03', 'K2015@.01'],
        peaks_by_cell=peaks_by_cell)

    tss = table(
        "TSS renderer-agnostic (peak PCK)",
        columns=[('tss_a10', 0.10), ('tss_a05', 0.05), ('tss_a03', 0.03)],
        group_headers=['TSS@.10', 'TSS@.05', 'TSS@.03'],
        peaks_by_cell=peaks_by_cell)

    tapvid = table(
        "TAP-Vid-DAVIS renderer-agnostic (peak PCK@0.05 by stride)",
        columns=[(f'tapvid_davis_s{s}', 0.05) for s in (1, 2, 4, 8, 16)],
        group_headers=[f's{s}' for s in (1, 2, 4, 8, 16)],
        peaks_by_cell=peaks_by_cell)

    for name, tbl in [('tab_ra_kitti', kitti), ('tab_ra_tss', tss), ('tab_ra_tapvid', tapvid)]:
        path = os.path.join(args.out_dir, name + '.tex')
        open(path, 'w').write(tbl + '\n')
        print(f"\n===== {name} =====\n{tbl}")
        print(f"wrote {path}")


if __name__ == '__main__':
    main()
