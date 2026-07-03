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


# Wide "Table 8" layout for the paper: rows = arch x {default, tuned, Delta},
# columns = every benchmark/metric. Auto-fills '--' where a cell hasn't finished.
WIDE_COLS = [
    ('kitti2012', 0.05), ('kitti2012', 0.03), ('kitti2012', 0.01),
    ('kitti2015', 0.05), ('kitti2015', 0.03), ('kitti2015', 0.01),
    ('tss_a10', 0.10), ('tss_a05', 0.05), ('tss_a03', 0.03),
    ('tapvid_davis_s1', 0.05), ('tapvid_davis_s2', 0.05), ('tapvid_davis_s4', 0.05),
    ('tapvid_davis_s8', 0.05), ('tapvid_davis_s16', 0.05),
]


def paper_wide_table(peaks_by_cell):
    n = len(WIDE_COLS)
    colspec = 'll ' + 'rrr ' * 2 + 'rrr ' + 'rrrrr'
    L = [r"\begin{tabular}{" + colspec.strip() + "}", r"\toprule"]
    L.append(r" &  & \multicolumn{3}{c}{KITTI-2012} & \multicolumn{3}{c}{KITTI-2015}"
             r" & \multicolumn{3}{c}{TSS} & \multicolumn{5}{c}{TAP-Vid-DAVIS (stride)} \\")
    L.append(r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}\cmidrule(lr){12-16}")
    L.append(r"Arch. & Src & @.05 & @.03 & @.01 & @.05 & @.03 & @.01 & @.10 & @.05 & @.03"
             r" & 1 & 2 & 4 & 8 & 16 \\")
    L.append(r"\midrule")
    for ai, (arch, label) in enumerate(ARCHS):
        d = peaks_by_cell.get(f'{arch}_default')
        t = peaks_by_cell.get(f'{arch}_tuned')
        dvals = [cell_val(d, b, a) for b, a in WIDE_COLS]
        tvals = [cell_val(t, b, a) for b, a in WIDE_COLS]

        def fnum(v):
            return f"{v:.1f}" if v is not None else "--"

        drow = [f"\\multirow{{3}}{{*}}{{{label}}}", "default"] + [fnum(v) for v in dvals]
        trow = ["", "tuned"]
        for dv, tv in zip(dvals, tvals):
            s = fnum(tv)
            if dv is not None and tv is not None and tv > dv:
                s = f"\\textbf{{{s}}}"
            trow.append(s)
        dlt = ["", r"$\Delta$"]
        for dv, tv in zip(dvals, tvals):
            dlt.append(fmt_delta(tv - dv if (dv is not None and tv is not None) else None))
        L.append(' & '.join(drow) + r' \\')
        L.append(' & '.join(trow) + r' \\')
        L.append(' & '.join(dlt) + r' \\')
        L.append(r"\midrule" if ai < len(ARCHS) - 1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    return '\n'.join(L)


# ---------------------------------------------------------------------------
# Two-generator tables (Kubric + SDF-fractal).
# Kubric @0.05 numbers are taken from the existing Table 7 (tab_interv.tex), pending
# a strict-alpha re-eval of the saved kitti-recovered weights; strict alphas + TAP-Vid
# show '?' until that eval lands (or until kubric_* eval CSVs appear under snap-base).
ARCH_LABEL = {'cats': 'CATs++', 'glunet': 'GLU-Net', 'flowformer': 'FlowFormer', 'raft': 'RAFT'}
GENERATORS = [
    {'key': 'kubric', 'label': 'Kubric', 'base_label': 'MOVi-F',
     'archs': ['cats', 'glunet', 'flowformer', 'raft']},   # RAFT = end-to-end, no pretrained backbone
    {'key': 'sdf', 'label': 'SDF-fractal', 'base_label': 'default',
     'archs': ['cats', 'glunet', 'flowformer']},
]
# Kubric numbers: (arch, benchmark, alpha) -> {'tuned':_, 'base':_}.
# KITTI strict-alpha values are re-evaluated from the saved kitti-recovered / MOVi-F
# checkpoints; GLU-Net & RAFT MOVi-F strict-alpha cells are unrecoverable (original
# base checkpoints gone) so only their published @.05 base is given -> '?' elsewhere.
# TSS @.05 is from the original Table 7.
KUBRIC = {
    # CATs++ KITTI
    ('cats', 'kitti2012', 0.05): {'tuned': 98.16, 'base': 95.57}, ('cats', 'kitti2012', 0.03): {'tuned': 89.12, 'base': 89.10}, ('cats', 'kitti2012', 0.01): {'tuned': 37.80, 'base': 45.51},
    ('cats', 'kitti2015', 0.05): {'tuned': 96.47, 'base': 94.89}, ('cats', 'kitti2015', 0.03): {'tuned': 85.96, 'base': 88.56}, ('cats', 'kitti2015', 0.01): {'tuned': 36.97, 'base': 44.40},
    # FlowFormer KITTI
    ('flowformer', 'kitti2012', 0.05): {'tuned': 94.60, 'base': 75.88}, ('flowformer', 'kitti2012', 0.03): {'tuned': 80.36, 'base': 48.54}, ('flowformer', 'kitti2012', 0.01): {'tuned': 29.47, 'base': 13.77},
    ('flowformer', 'kitti2015', 0.05): {'tuned': 73.21, 'base': 63.15}, ('flowformer', 'kitti2015', 0.03): {'tuned': 55.95, 'base': 46.22}, ('flowformer', 'kitti2015', 0.01): {'tuned': 25.78, 'base': 20.81},
    # GLU-Net KITTI. base = surviving MOVi-F TT ckpt tssgrid_movif_glunet_tt (all 3 alphas).
    # NOTE: original Table-7 base @.05 was 73.9/61.8 (that ckpt is gone); @.05 updated to the
    # surviving ckpt so each row is one consistent checkpoint.
    ('glunet', 'kitti2012', 0.05): {'tuned': 96.15, 'base': 79.8}, ('glunet', 'kitti2012', 0.03): {'tuned': 85.82, 'base': 52.8}, ('glunet', 'kitti2012', 0.01): {'tuned': 46.77, 'base': 15.5},
    ('glunet', 'kitti2015', 0.05): {'tuned': 77.56, 'base': 66.9}, ('glunet', 'kitti2015', 0.03): {'tuned': 60.50, 'base': 49.0}, ('glunet', 'kitti2015', 0.01): {'tuned': 30.69, 'base': 22.6},
    # RAFT KITTI (end-to-end, no pretrained backbone). base = surviving raft_movif_ff (all 3 alphas);
    # original Table-7 base @.05 was 81.5/70.5 (gone), so @.05 updated to surviving ckpt.
    ('raft', 'kitti2012', 0.05): {'tuned': 96.63, 'base': 84.5}, ('raft', 'kitti2012', 0.03): {'tuned': 87.08, 'base': 62.2}, ('raft', 'kitti2012', 0.01): {'tuned': 49.40, 'base': 25.9},
    ('raft', 'kitti2015', 0.05): {'tuned': 78.75, 'base': 74.4}, ('raft', 'kitti2015', 0.03): {'tuned': 61.82, 'base': 57.3}, ('raft', 'kitti2015', 0.01): {'tuned': 32.37, 'base': 30.2},
    # TSS + TAP-Vid-DAVIS (Kubric). base = movif_*_tt (snapshots_mm_rc, 2026-06-26 harness,
    # peak PCK over epochs) -> one consistent checkpoint per base row (supersedes the older
    # Table-7 TSS@.05 base 57.0/19.7/14.1, a different eval regime). tuned TSS@.05 still from
    # original Table 7 (kitti-recovered source); tuned TAP-Vid strides pending Track 2 -> '?'.
    ('cats', 'tss_a10', 0.10): {'base': 79.0}, ('cats', 'tss_a05', 0.05): {'tuned': 60.3, 'base': 61.9}, ('cats', 'tss_a03', 0.03): {'base': 44.6},
    ('glunet', 'tss_a10', 0.10): {'base': 61.3}, ('glunet', 'tss_a05', 0.05): {'tuned': 27.3, 'base': 32.8}, ('glunet', 'tss_a03', 0.03): {'base': 18.3},
    ('flowformer', 'tss_a10', 0.10): {'base': 55.0}, ('flowformer', 'tss_a05', 0.05): {'tuned': 16.0, 'base': 28.3}, ('flowformer', 'tss_a03', 0.03): {'base': 15.5},
    ('cats', 'tapvid_davis_s1', 0.05): {'base': 100.0}, ('cats', 'tapvid_davis_s2', 0.05): {'base': 98.5}, ('cats', 'tapvid_davis_s4', 0.05): {'base': 95.4}, ('cats', 'tapvid_davis_s8', 0.05): {'base': 89.6}, ('cats', 'tapvid_davis_s16', 0.05): {'base': 83.9},
    ('glunet', 'tapvid_davis_s1', 0.05): {'base': 99.4}, ('glunet', 'tapvid_davis_s2', 0.05): {'base': 96.2}, ('glunet', 'tapvid_davis_s4', 0.05): {'base': 87.4}, ('glunet', 'tapvid_davis_s8', 0.05): {'base': 73.0}, ('glunet', 'tapvid_davis_s16', 0.05): {'base': 55.3},
    ('flowformer', 'tapvid_davis_s1', 0.05): {'base': 98.7}, ('flowformer', 'tapvid_davis_s2', 0.05): {'base': 94.7}, ('flowformer', 'tapvid_davis_s4', 0.05): {'base': 82.6}, ('flowformer', 'tapvid_davis_s8', 0.05): {'base': 64.4}, ('flowformer', 'tapvid_davis_s16', 0.05): {'base': 47.4},
}


# Swap which trained cell backs an (arch, role) in the SDF block. Per request the
# CATs++ "tuned" row is sourced from small-zoom (label in the table stays "tuned").
SDF_CELL_OVERRIDE = {('cats', 'tuned'): 'smallzoom_cats'}


def gen_val(peaks_by_cell, gen, arch, role, bench, alpha):
    """role in {'base','tuned'}; returns float or None."""
    if gen['key'] == 'sdf':
        cell = f'{arch}_tuned' if role == 'tuned' else f'{arch}_default'
        cell = SDF_CELL_OVERRIDE.get((arch, role), cell)
        return cell_val(peaks_by_cell.get(cell), bench, alpha)
    # kubric: prefer a real eval CSV if present, else the curated dict above
    cell = f'kubric_{arch}_tuned' if role == 'tuned' else f'kubric_{arch}_movif'
    pk = peaks_by_cell.get(cell)
    if pk:
        v = cell_val(pk, bench, alpha)
        if v is not None:
            return v
    return KUBRIC.get((arch, bench, round(alpha, 2)), {}).get(role)


def two_gen_table(groups, peaks_by_cell):
    """Readable layout: rows = arch x {base, tuned} (no Delta), grouped by generator.
    groups: list of {'label':str,'cols':[(bench,alpha)],'subs':[str]}."""
    cols = [c for g in groups for c in g['cols']]
    n = len(cols)
    L = [r"\begin{tabular}{ll " + "r" * n + "}", r"\toprule"]
    gh = ["", ""] + [f"\\multicolumn{{{len(g['cols'])}}}{{c}}{{{g['label']}}}" for g in groups]
    L.append(" & ".join(gh) + r" \\")
    cm, start = [], 3
    for g in groups:
        cm.append(f"\\cmidrule(lr){{{start}-{start + len(g['cols']) - 1}}}")
        start += len(g['cols'])
    L.append("".join(cm))
    subs = ["Model", "Source"] + [s for g in groups for s in g['subs']]
    L.append(" & ".join(subs) + r" \\")
    L.append(r"\midrule")

    def fnum(v):
        return f"{v:.1f}" if v is not None else "?"

    for gi, gen in enumerate(GENERATORS):
        L.append(f"\\multicolumn{{{n + 2}}}{{l}}{{\\emph{{{gen['label']} generator}}}} \\\\")
        L.append(r"\addlinespace[1pt]")
        for arch in gen['archs']:
            label = ARCH_LABEL[arch]
            bvals = [gen_val(peaks_by_cell, gen, arch, 'base', b, a) for b, a in cols]
            tvals = [gen_val(peaks_by_cell, gen, arch, 'tuned', b, a) for b, a in cols]
            # bold the better of base/tuned in EACH column (even when tuned loses), so
            # losses aren't hidden; on an exact tie bold both.
            brow = [f"\\multirow{{2}}{{*}}{{{label}}}", gen['base_label']]
            trow = ["", "tuned"]
            for bv, tv in zip(bvals, tvals):
                bs, ts = fnum(bv), fnum(tv)
                if bv is not None and tv is not None:
                    if tv > bv:
                        ts = f"\\textbf{{{ts}}}"
                    elif bv > tv:
                        bs = f"\\textbf{{{bs}}}"
                    else:
                        bs, ts = f"\\textbf{{{bs}}}", f"\\textbf{{{ts}}}"
                brow.append(bs)
                trow.append(ts)
            L.append(" & ".join(brow) + r" \\")
            L.append(" & ".join(trow) + r" \\")
            if arch != gen['archs'][-1]:
                L.append(r"\addlinespace[2pt]")
        L.append(r"\midrule" if gi < len(GENERATORS) - 1 else r"\bottomrule")
    L.append(r"\end{tabular}")
    return "\n".join(L)


def build_two_gen_tables(peaks_by_cell):
    t1 = two_gen_table([
        {'label': 'KITTI-2012', 'cols': [('kitti2012', .05), ('kitti2012', .03), ('kitti2012', .01)], 'subs': [r'@5\%', r'@3\%', r'@1\%']},
        {'label': 'KITTI-2015', 'cols': [('kitti2015', .05), ('kitti2015', .03), ('kitti2015', .01)], 'subs': [r'@5\%', r'@3\%', r'@1\%']},
    ], peaks_by_cell)
    t2 = two_gen_table([
        {'label': r'TSS ($\alpha$)', 'cols': [('tss_a10', .10), ('tss_a05', .05), ('tss_a03', .03)], 'subs': [r'@10\%', r'@5\%', r'@3\%']},
        {'label': 'TAP-Vid-DAVIS (stride)', 'cols': [(f'tapvid_davis_s{s}', .05) for s in (1, 2, 4, 8, 16)], 'subs': ['1', '2', '4', '8', '16']},
    ], peaks_by_cell)
    return t1, t2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snap-base', default='/mnt/nvme_1tb_a/renderer_agnostic')
    ap.add_argument('--cells', nargs='+', default=None)
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tables'))
    ap.add_argument('--paper-table', default=None, help='(legacy) wide single-generator layout')
    ap.add_argument('--t1-kitti', default=None, help='write Table 1 (KITTI, two generators) here')
    ap.add_argument('--t2-secondary', default=None, help='write Table 2 (TSS + TAP-Vid) here')
    args = ap.parse_args()

    cells = args.cells or [f'{a}_{s}' for a, _ in ARCHS for s in ('default', 'tuned')]
    cells = list(cells) + list(SDF_CELL_OVERRIDE.values())   # incl. smallzoom_cats
    # also look for Kubric eval cells (filled once strict-alpha re-eval lands)
    kubric_cells = [f'kubric_{a}_{r}' for a in ARCH_LABEL for r in ('tuned', 'movif')]
    peaks_by_cell = {}
    for cell in list(cells) + kubric_cells:
        d = find_cell_dir(args.snap_base, cell)
        peaks_by_cell[cell] = load_peaks(d) if d else None
        if cell in cells:
            print(f"  {cell:<20} {'OK' if peaks_by_cell[cell] else 'MISSING'}  ({d})")

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

    if args.paper_table:
        wide = paper_wide_table(peaks_by_cell)
        open(args.paper_table, 'w').write(wide + '\n')
        print(f"wrote {args.paper_table}")

    # --- two-generator tables ---
    t1, t2 = build_two_gen_tables(peaks_by_cell)
    t1_path = args.t1_kitti or os.path.join(args.out_dir, 'tab_ra_t1_kitti.tex')
    t2_path = args.t2_secondary or os.path.join(args.out_dir, 'tab_ra_t2_secondary.tex')
    open(t1_path, 'w').write(t1 + '\n')
    open(t2_path, 'w').write(t2 + '\n')
    print(f"\n===== TABLE 1 (KITTI, two generators) =====\n{t1}")
    print(f"\n===== TABLE 2 (TSS + TAP-Vid) =====\n{t2}")
    print(f"\nwrote {t1_path}\nwrote {t2_path}")


if __name__ == '__main__':
    main()
