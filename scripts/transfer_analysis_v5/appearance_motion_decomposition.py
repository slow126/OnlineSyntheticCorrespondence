"""Appearance x Motion decomposition on the controllable kubric intervention grid.

The intervention sources are procedurally generated, held-out, and re-renderable,
so they let us cross two axes that are confounded in any natural dataset:

  * MOTION STRUCTURE: camera-dominant vs object-dominant vs frozen-camera
    (the camera/object SE(3) split read straight off theta).
  * APPEARANCE:       hq (HDRI + asset materials) vs matte (low-texture).

with ASSETS (gso vs kubasic) as a nuisance axis we can hold fixed within a pair.

Because every hq<->matte pair shares its motion theta bit-identically (same-theta
appearance ablation) and every camera<->object pair shares assets+appearance
(same-generator motion swap), we can read MAIN EFFECTS by marginalizing the
nuisance axis and CONDITIONING on the benchmark+regime -- we must NOT marginalize
over the benchmark, because the motion effect flips sign across benchmarks (the
crossover), so pooling benchmarks would cancel the very signal we want.

Reads from results/intervention_breakdown.csv (peak PCK per source x arm x
benchmark; middlebury already excluded). Writes:
  results/decomp_cells.csv          -- every cell tagged with its factor levels
  results/decomp_appearance.csv     -- appearance main effect (HQ - matte)
  results/decomp_motion.csv         -- motion-structure effect (camera - object)
  results/decomp_2x2.csv            -- balanced KuBasic 2x2 ANOVA main effects + interaction

Usage:
  python scripts/transfer_analysis_v5/appearance_motion_decomposition.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
BREAKDOWN = HERE / "results" / "intervention_breakdown.csv"
OUT = HERE / "results"
TABLES = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence/ACCV_2026/tables")

# source -> (motion_structure, appearance, assets).  Read off each set's theta.
META = {
    "trial19":                   ("frozen", "hq",    "gso"),
    "lowtex_matte":              ("frozen", "matte", "gso"),
    "kitti_recovered_gso_hq":    ("camera", "hq",    "gso"),
    "kitti_recovered_gso_matte": ("camera", "matte", "gso"),
    "kitti_recovered_hq":        ("camera", "hq",    "kubasic"),
    "kitti_recovered_matte":     ("camera", "matte", "kubasic"),
    "kitti_badmotion_ft_gso_hq": ("object", "hq",    "gso"),
    "ft_recovered_hq":           ("object", "hq",    "kubasic"),
    "ft_recovered_matte":        ("object", "matte", "kubasic"),
}

# benchmark -> its physical motion regime (for reading the crossover sign).
BENCH_REGIME = {
    "kitti2012":    "ego (camera)",
    "kitti2015":    "mixed",
    "flyingthings": "object",
}

# matched same-theta hq<->matte pairs: (hq_source, matte_source) sharing motion+assets.
APPEARANCE_PAIRS = [
    ("trial19",                "lowtex_matte"),               # frozen,  gso
    ("kitti_recovered_gso_hq", "kitti_recovered_gso_matte"),  # camera,  gso
    ("kitti_recovered_hq",     "kitti_recovered_matte"),      # camera,  kubasic
    ("ft_recovered_hq",        "ft_recovered_matte"),         # object,  kubasic
]

# matched camera<->object pairs sharing appearance+assets (same generator/render).
MOTION_PAIRS = [
    ("kitti_recovered_gso_hq", "kitti_badmotion_ft_gso_hq", "gso",     "hq"),
    ("kitti_recovered_hq",     "ft_recovered_hq",           "kubasic", "hq"),
    ("kitti_recovered_matte",  "ft_recovered_matte",        "kubasic", "matte"),
]


def load_cells() -> pd.DataFrame:
    df = pd.read_csv(BREAKDOWN)
    df = df[["arm", "benchmark", "source", "peak_pck"]].copy()
    df = df[df.source.isin(META)]
    df["motion"] = df.source.map(lambda s: META[s][0])
    df["appearance"] = df.source.map(lambda s: META[s][1])
    df["assets"] = df.source.map(lambda s: META[s][2])
    df["bench_regime"] = df.benchmark.map(BENCH_REGIME)
    return df


def pck(df, arm, bench, source):
    hit = df[(df.arm == arm) & (df.benchmark == bench) & (df.source == source)]
    return float(hit.peak_pck.iloc[0]) if len(hit) else None


def appearance_effect(df) -> pd.DataFrame:
    """HQ - matte, per same-theta pair, then averaged -> appearance main effect."""
    rows = []
    for arm in sorted(df.arm.unique()):
        for bench in sorted(df.benchmark.unique()):
            for hq, matte in APPEARANCE_PAIRS:
                a, b = pck(df, arm, bench, hq), pck(df, arm, bench, matte)
                if a is None or b is None:
                    continue
                rows.append(dict(arm=arm, benchmark=bench,
                                 motion=META[hq][0], assets=META[hq][2],
                                 pair=f"{hq}|{matte}", delta_hq_minus_matte=a - b))
    pairwise = pd.DataFrame(rows)
    main = (pairwise.groupby(["arm", "benchmark"])
            .delta_hq_minus_matte.agg(["mean", "std", "count"]).reset_index()
            .rename(columns={"mean": "appearance_effect"}))
    return pairwise, main


def motion_effect(df) -> pd.DataFrame:
    """camera - object, per matched pair, per (arm, benchmark). Flips sign by bench."""
    rows = []
    for arm in sorted(df.arm.unique()):
        for bench in sorted(df.benchmark.unique()):
            for cam, obj, assets, appear in MOTION_PAIRS:
                a, b = pck(df, arm, bench, cam), pck(df, arm, bench, obj)
                if a is None or b is None:
                    continue
                rows.append(dict(arm=arm, benchmark=bench, bench_regime=BENCH_REGIME[bench],
                                 assets=assets, appearance=appear,
                                 pair=f"{cam}|{obj}", delta_cam_minus_obj=a - b))
    pairwise = pd.DataFrame(rows)
    main = (pairwise.groupby(["arm", "benchmark", "bench_regime"])
            .delta_cam_minus_obj.agg(["mean", "std", "count"]).reset_index()
            .rename(columns={"mean": "motion_effect"}))
    return pairwise, main


def balanced_2x2(df) -> pd.DataFrame:
    """Fully-balanced KuBasic 2(appearance) x 2(motion) factorial -> ANOVA effects.

    cells: camera/object x hq/matte on kubasic assets.
    main_motion     = mean[(cam) - (obj)]   over appearance
    main_appearance = mean[(hq)  - (matte)] over motion
    interaction     = (cam_hq - cam_matte) - (obj_hq - obj_matte)) / 2
    """
    cells = {
        ("camera", "hq"):    "kitti_recovered_hq",
        ("camera", "matte"): "kitti_recovered_matte",
        ("object", "hq"):    "ft_recovered_hq",
        ("object", "matte"): "ft_recovered_matte",
    }
    rows = []
    for arm in sorted(df.arm.unique()):
        for bench in sorted(df.benchmark.unique()):
            v = {k: pck(df, arm, bench, s) for k, s in cells.items()}
            if any(x is None for x in v.values()):
                continue
            cam = (v[("camera", "hq")] + v[("camera", "matte")]) / 2
            obj = (v[("object", "hq")] + v[("object", "matte")]) / 2
            hq = (v[("camera", "hq")] + v[("object", "hq")]) / 2
            matte = (v[("camera", "matte")] + v[("object", "matte")]) / 2
            inter = ((v[("camera", "hq")] - v[("camera", "matte")])
                     - (v[("object", "hq")] - v[("object", "matte")])) / 2
            rows.append(dict(
                arm=arm, benchmark=bench, bench_regime=BENCH_REGIME[bench],
                main_motion_cam_minus_obj=cam - obj,
                main_appearance_hq_minus_matte=hq - matte,
                interaction=inter,
                cam_hq=v[("camera", "hq")], cam_matte=v[("camera", "matte")],
                obj_hq=v[("object", "hq")], obj_matte=v[("object", "matte")]))
    return pd.DataFrame(rows)


GRID_SNAP = Path("/mnt/nvme_1tb_a/snapshots/transfer_grid")


def harvest_ff(source, benches):
    """Peak from-scratch PCK per benchmark, read straight from the transfer-grid
    snapshot's validation_results.csv (source of truth; the new GSO matte cell is
    not yet in intervention_breakdown.csv)."""
    import glob
    dirs = sorted(glob.glob(str(GRID_SNAP / f"{source}_pt0_fz0_*")))
    if not dirs:
        return {b: None for b in benches}
    d = pd.read_csv(Path(dirs[-1]) / "validation_results.csv")
    return {b: (float(d[d.benchmark == b].pck.max())
                if len(d[d.benchmark == b]) else None) for b in benches}


def write_paper_table(df):
    """Emit ACCV_2026/tables/tab_decomp.tex as the honest 2x2: two stacked
    motion x texture panels (raw from-scratch PCK), one per target, on the
    COMPLETE GSO 2x2 (camera/object x HQ/matte) -- realistic Google-Scanned-Object
    assets, so the HQ->matte texture manipulation is a meaningful appearance
    degradation. Numbers harvested live from the grid snapshots (nothing typed).

    The point the 2x2 makes:
      * The best cell (bold = max of the four) is always good-motion x HQ -- but
        'good motion' is CAMERA for the ego target and OBJECT for the object
        target, so the bold cell moves between physical rows. Motion is
        target-specific; texture (HQ>matte) is a fixed tax in every cell.
      * On KITTI the off-diagonal nearly ties (good-mo/bad-tex ~ bad-mo/good-tex),
        so the two effects are COMPARABLE in size. The claim is targeting, not
        magnitude dominance -- which is why we do NOT show a 'floor' row (only
        wrong-task sources floor KITTI-from-scratch; that is the separate
        appearance-space negative control, not part of this controlled grid).
      * Replicates on basic KuBasic assets (kitti_recovered_*/ft_recovered_*).
    """
    benches = ["kitti2012", "flyingthings", "kitti2015"]
    # complete GSO 2x2 (camera = kitti_recovered_gso, object = kitti_badmotion_ft_gso)
    V = {s: harvest_ff(s, benches) for s in (
        "kitti_recovered_gso_hq", "kitti_recovered_gso_matte",
        "kitti_badmotion_ft_gso_hq", "kitti_badmotion_ft_gso_matte")}

    def panel(bench, good_is_camera):
        cells = {
            ("cam", 0): V["kitti_recovered_gso_hq"][bench],
            ("cam", 1): V["kitti_recovered_gso_matte"][bench],
            ("obj", 0): V["kitti_badmotion_ft_gso_hq"][bench],
            ("obj", 1): V["kitti_badmotion_ft_gso_matte"][bench],
        }
        best = max(cells, key=cells.get)
        f = lambda k: (f"\\textbf{{{cells[k]:.1f}}}" if k == best else f"{cells[k]:.1f}")
        cam_tag = "good" if good_is_camera else "bad"
        obj_tag = "bad" if good_is_camera else "good"
        return (f"\\quad camera motion \\textit{{({cam_tag})}} & {f(('cam', 0))} & {f(('cam', 1))} \\\\\n"
                f"\\quad object motion \\textit{{({obj_tag})}} & {f(('obj', 0))} & {f(('obj', 1))} \\\\")

    tex = "\n".join([
        r"\begin{tabular}{l cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{texture} \\",
        r"\cmidrule(lr){2-3}",
        r"trained on (from scratch) & HQ \textit{(good)} & matte \textit{(bad)} \\",
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textit{target:} KITTI-2012 --- ego motion} \\",
        panel("kitti2012", good_is_camera=True),
        r"\midrule",
        r"\multicolumn{3}{@{}l}{\textit{target:} FlyingThings --- object motion} \\",
        panel("flyingthings", good_is_camera=False),
        r"\bottomrule",
        r"\end{tabular}",
    ])
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "tab_decomp.tex").write_text(tex + "\n")
    print(f"\nwrote {TABLES}/tab_decomp.tex")


def main():
    df = load_cells()
    df.to_csv(OUT / "decomp_cells.csv", index=False)

    app_pairs, app_main = appearance_effect(df)
    app_main.to_csv(OUT / "decomp_appearance.csv", index=False)

    mot_pairs, mot_main = motion_effect(df)
    mot_main.to_csv(OUT / "decomp_motion.csv", index=False)

    tbl = balanced_2x2(df)
    tbl.to_csv(OUT / "decomp_2x2.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 30)
    fmt = lambda x: f"{x:+.2f}"

    print("=" * 78)
    print("APPEARANCE MAIN EFFECT  (HQ - matte; >0 means HQ helps), per arm x benchmark")
    print("  marginalized over motion structure + assets; n same-theta pairs averaged")
    print("=" * 78)
    print(app_main.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("MOTION-STRUCTURE EFFECT  (camera - object; sign flips by benchmark regime)")
    print("  marginalized over appearance + assets; the crossover")
    print("=" * 78)
    print(mot_main.to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("BALANCED 2x2 (KuBasic): ANOVA main effects + interaction, per arm x benchmark")
    print("=" * 78)
    show = ["arm", "benchmark", "bench_regime", "main_motion_cam_minus_obj",
            "main_appearance_hq_minus_matte", "interaction"]
    print(tbl[show].to_string(index=False, float_format=fmt))

    print("\n" + "=" * 78)
    print("HEADLINE  |motion structure effect|  vs  |appearance effect|, by regime")
    print("=" * 78)
    for arm in sorted(df.arm.unique()):
        a = app_main[app_main.arm == arm].appearance_effect.abs()
        # motion effect: use the two pure-regime benchmarks where it does not cancel
        m = mot_main[(mot_main.arm == arm)
                     & (mot_main.benchmark.isin(["kitti2012", "flyingthings"]))].motion_effect.abs()
        if len(a) and len(m):
            print(f"  {arm}:  |appearance| mean {a.mean():.2f} (max {a.max():.2f})   "
                  f"|motion structure| pure-regime mean {m.mean():.2f} (max {m.max():.2f})   "
                  f"ratio {m.mean() / a.mean():.1f}x")
    write_paper_table(df)
    print(f"\nwrote {OUT}/decomp_*.csv")


if __name__ == "__main__":
    main()
