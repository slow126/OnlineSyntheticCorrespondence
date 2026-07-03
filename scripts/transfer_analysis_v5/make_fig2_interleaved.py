"""Figure 2: one interleaved 4-row grid (RGB above its BFV motion splat), grouped
by Training source (rows 1-2) then Target benchmark (rows 3-4).

Row order per group: RGB sample, then the directional BFV splat of the same
dataset in the same column. The first three columns hold the datasets that appear
in BOTH the source and target sets (FlyingThings / PointOdyssey / SPair), tinted
so the reader can see the fingerprint is a property of the dataset.

Output: ACCV_2026/figures/splats/fig2_interleaved.png
    python scripts/transfer_analysis_v5/make_fig2_interleaved.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence")
SPL = ROOT / "ACCV_2026/figures/splats"          # splat tiles + legend live here
RGB = ROOT / "analysis/rgb_samples"              # per-dataset RGB samples

TILE = 300
GAP = 16              # gap between columns
GAP_IN = 6            # gap between the RGB row and its splat row (within a group)
GAP_GRP = 30          # gap between the two groups
LABEL_H = 40          # dataset-name strip under each group's splat row
HEAD_GRP = 46         # rotated group-name strip (spans both rows of a group)
HEAD_MOD = 32         # rotated per-row modality strip (RGB / Motion)
MARGIN = 24
NOTE_H = 26

# (dataset label, rgb file, splat file). "__LEGEND__" cell holds the colorwheel.
GROUPS = [
    ("Training source", [
        ("FlyingThings",  "flyingthings_train.png", "train__flyingthings__directional_splat.png"),
        ("PointOdyssey",  "pointodyssey_train.png", "train__pointodyssey__directional_splat.png"),
        ("SPair",         "spair_train.jpg",        "train__spair__directional_splat.png"),
        ("SDF-Fractal3D", "sdf_fractal.png",        "train__sdf-fractal3d__directional_splat.png"),
        ("Sintel",        "sintel.png",             "train__sintel__directional_splat.png"),
        ("MOVi-F",        "movi_f.png",             "train__movi-f__directional_splat.png"),
        ("ImageNet2DWarp","imagenet2dwarp.png",     "train__imagenet2dwarp__directional_splat.png"),
        ("__LEGEND__",    None,                     None),
    ]),
    ("Target benchmark", [
        ("FlyingThings", "flyingthings_bench.png",  "benchmark__flyingthings_test__directional_splat.png"),
        ("PointOdyssey", "pointodyssey_bench.jpg",  "benchmark__pointodyssey_test__directional_splat.png"),
        ("SPair",        "spair_bench.jpg",         "benchmark__spair_test__directional_splat.png"),
        ("KITTI-2015",   "kitti2015.png",           "benchmark__kitti2015__directional_splat.png"),
        ("KITTI-2012",   "kitti2012.png",           "benchmark__kitti2012__directional_splat.png"),
        ("TSS",          "tss.png",                 "benchmark__tss__directional_splat.png"),
        ("PF-PASCAL",    "pfpascal.jpg",            "benchmark__pfpascal__directional_splat.png"),
        ("PF-WILLOW",    "pfwillow.png",            "benchmark__pfwillow__directional_splat.png"),
    ]),
]

N_SHARED = 3
N_COL = max(len(t) for _, t in GROUPS)

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_I = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
f_label = ImageFont.truetype(FONT, 22)
f_grp = ImageFont.truetype(FONT_B, 26)
f_mod = ImageFont.truetype(FONT, 20)
f_leg = ImageFont.truetype(FONT, 20)
f_note = ImageFont.truetype(FONT_I, 19)

INK = (55, 60, 70)
FRAME = (190, 194, 200)
SHARED = (232, 236, 244)

col_x = [MARGIN + HEAD_GRP + HEAD_MOD + c * (TILE + GAP) for c in range(N_COL)]
W = col_x[-1] + TILE + MARGIN
grp_h = TILE + GAP_IN + TILE + LABEL_H          # rgb row + splat row + names
H = MARGIN + grp_h + GAP_GRP + grp_h + NOTE_H + MARGIN

canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)


def crop_content(im):
    """Trim the uniform black border around a splat tile."""
    g = im.convert("L")
    bbox = g.point(lambda v: 255 if v > 12 else 0).getbbox()
    if bbox:
        pad = 6
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        im = im.crop(bbox)
    return im


def square(im):
    """Center-crop to a square (for photographic RGB samples)."""
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def rot_text(text, strip_w, span_h, font):
    """A left-edge label rotated 90 deg, sized (strip_w x span_h)."""
    im = Image.new("RGB", (span_h, strip_w), "white")
    d = ImageDraw.Draw(im)
    tw = d.textlength(text, font=font)
    d.text(((span_h - tw) / 2, (strip_w - font.size) / 2 - 2), text, fill=INK, font=font)
    return im.rotate(90, expand=True)


def paste_tile(path, x0, y0, is_rgb):
    im = Image.open(path).convert("RGB")
    im = square(im) if is_rgb else crop_content(im)
    im = im.resize((TILE - 2, TILE - 2), Image.LANCZOS)
    draw.rectangle([x0, y0, x0 + TILE, y0 + TILE], outline=FRAME, width=2)
    canvas.paste(im, (x0 + 1, y0 + 1))


# Faint band behind the shared-dataset columns, spanning the whole grid.
band_x0 = col_x[0] - 7
band_x1 = col_x[N_SHARED - 1] + TILE + 7
draw.rectangle([band_x0, MARGIN - 7, band_x1, MARGIN + 2 * grp_h + GAP_GRP - LABEL_H + 4],
               fill=SHARED)

for g, (gname, tiles) in enumerate(GROUPS):
    gy0 = MARGIN + g * (grp_h + GAP_GRP)
    rgb_y = gy0
    spl_y = gy0 + TILE + GAP_IN
    span = 2 * TILE + GAP_IN

    # group header (spans both rows) + per-row modality labels
    canvas.paste(rot_text(gname, HEAD_GRP - 12, span, f_grp), (MARGIN, rgb_y))
    canvas.paste(rot_text("RGB", HEAD_MOD - 8, TILE, f_mod), (MARGIN + HEAD_GRP, rgb_y))
    canvas.paste(rot_text("Motion", HEAD_MOD - 8, TILE, f_mod), (MARGIN + HEAD_GRP, spl_y))

    for c, (name, rgb_fn, spl_fn) in enumerate(tiles):
        x0 = col_x[c]
        if name == "__LEGEND__":
            continue
        paste_tile(RGB / rgb_fn, x0, rgb_y, is_rgb=True)
        paste_tile(SPL / spl_fn, x0, spl_y, is_rgb=False)
        tw = draw.textlength(name, font=f_label)
        draw.text((x0 + (TILE - tw) / 2, spl_y + TILE + 7), name, fill=INK, font=f_label)

# Direction colorwheel legend in the trailing cell of the source Motion row.
lx0 = col_x[len(GROUPS[0][1]) - 1]
ly0 = MARGIN + TILE + GAP_IN                      # source splat (Motion) row
wheel = Image.open(SPL / "legend__direction_colorwheel.png").convert("RGB")
wsz = TILE - 70
wheel = wheel.resize((wsz, wsz), Image.LANCZOS)
canvas.paste(wheel, (lx0 + (TILE - wsz) // 2, ly0 + 6))
for i, line in enumerate(["hue = flow direction", "extent = cluster spread"]):
    tw = draw.textlength(line, font=f_leg)
    draw.text((lx0 + (TILE - tw) / 2, ly0 + 6 + wsz + 10 + i * 26), line, fill=INK, font=f_leg)

note = "first three columns: the same dataset in source and target"
nw = draw.textlength(note, font=f_note)
draw.text((band_x0 + (band_x1 - band_x0 - nw) / 2, H - MARGIN - NOTE_H + 2),
          note, fill=(120, 128, 140), font=f_note)

out = SPL / "fig2_interleaved.png"
canvas.save(out, dpi=(150, 150))
print(f"wrote {out}  ({canvas.size[0]}x{canvas.size[1]})")
