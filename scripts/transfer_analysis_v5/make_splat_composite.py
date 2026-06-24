"""Compose the BFV splat panels into one polished, conference-ready grid.

Layout: two labeled rows (training sets / benchmarks) of up to eight tiles.
The columns are ordered so datasets that exist in BOTH splits line up
vertically -- FlyingThings / PointOdyssey / SPair sit train-above-test in the
same column, so a reader can see at a glance that the train and eval
fingerprints of a dataset agree. The direction colorwheel legend occupies the
empty trailing cell of the training row. White background, thin gray tile
frames, row headers on the left edge, dataset names under tiles.

Output: ACCV_2026/figures/splats/composite_overview.png

    python scripts/transfer_analysis_v5/make_splat_composite.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SPL = Path("/home/spencer/Projects/OnlineSyntheticCorrespondence/ACCV_2026/figures/splats")
TILE = 300          # tile size px
GAP = 16            # gap between tiles
LABEL_H = 40        # label strip under each tile
HEAD_W = 52         # rotated row-header strip
MARGIN = 24

# Columns 0-2 are the datasets that appear in both splits; they are placed in
# the same column in both rows so the train/eval fingerprints sit one above the
# other. Remaining cells are split-only datasets. (Middlebury is excluded
# everywhere per the eval-resolution finding.)
ROWS = [
    ("TRAINING SETS", [
        ("FlyingThings", "train__flyingthings__directional_splat.png"),
        ("PointOdyssey", "train__pointodyssey__directional_splat.png"),
        ("SPair", "train__spair__directional_splat.png"),
        ("SDF-Fractal3D", "train__sdf-fractal3d__directional_splat.png"),
        ("Sintel", "train__sintel__directional_splat.png"),
        ("MOVi-F", "train__movi-f__directional_splat.png"),
        ("ImageNet2DWarp", "train__imagenet2dwarp__directional_splat.png"),
        ("__LEGEND__", None),
    ]),
    ("BENCHMARKS", [
        ("FlyingThings", "benchmark__flyingthings_test__directional_splat.png"),
        ("PointOdyssey", "benchmark__pointodyssey_test__directional_splat.png"),
        ("SPair", "benchmark__spair_test__directional_splat.png"),
        ("KITTI-2015", "benchmark__kitti2015__directional_splat.png"),
        ("KITTI-2012", "benchmark__kitti2012__directional_splat.png"),
        ("TSS", "benchmark__tss__directional_splat.png"),
        ("PF-PASCAL", "benchmark__pfpascal__directional_splat.png"),
        ("PF-WILLOW", "benchmark__pfwillow__directional_splat.png"),
    ]),
]

N_SHARED = 3        # first N columns hold the same dataset in both rows

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_I = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
f_label = ImageFont.truetype(FONT, 22)
f_head = ImageFont.truetype(FONT_B, 24)
f_leg = ImageFont.truetype(FONT, 20)
f_note = ImageFont.truetype(FONT_I, 19)

INK = (55, 60, 70)
FRAME = (190, 194, 200)
SHARED = (232, 236, 244)   # faint tint behind the shared-dataset columns

n = max(len(tiles) for _, tiles in ROWS)
row_h = TILE + LABEL_H + GAP
W = MARGIN + HEAD_W + n * (TILE + GAP) - GAP + MARGIN
H = MARGIN + 2 * row_h + MARGIN - GAP
canvas = Image.new("RGB", (W, H), "white")
draw = ImageDraw.Draw(canvas)


def crop_content(im):
    """Trim the uniform black border around the splat content."""
    g = im.convert("L")
    bbox = g.point(lambda v: 255 if v > 12 else 0).getbbox()
    if bbox:
        pad = 6
        bbox = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
                min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        im = im.crop(bbox)
    return im


col_x = [MARGIN + HEAD_W + c * (TILE + GAP) for c in range(n)]

# Faint vertical band behind the shared-dataset columns to signal the pairing.
band_x0 = col_x[0] - 7
band_x1 = col_x[N_SHARED - 1] + TILE + 7
draw.rectangle([band_x0, MARGIN - 7, band_x1, MARGIN + 2 * row_h - GAP + 4],
               fill=SHARED)
note = "same dataset — train (top) vs. eval (bottom)"
nw = draw.textlength(note, font=f_note)
draw.text((band_x0 + (band_x1 - band_x0 - nw) / 2, H - MARGIN - 6), note,
          fill=(120, 128, 140), font=f_note)

for r, (header, tiles) in enumerate(ROWS):
    y0 = MARGIN + r * row_h
    # rotated row header
    head = Image.new("RGB", (TILE, HEAD_W - 14), "white")
    hd = ImageDraw.Draw(head)
    tw = hd.textlength(header, font=f_head)
    hd.text(((TILE - tw) / 2, 6), header, fill=INK, font=f_head)
    canvas.paste(head.rotate(90, expand=True), (MARGIN, y0))
    for c, (name, fn) in enumerate(tiles):
        x0 = col_x[c]
        if name == "__LEGEND__":
            continue
        im = crop_content(Image.open(SPL / fn).convert("RGB"))
        im = im.resize((TILE - 2, TILE - 2), Image.LANCZOS)
        draw.rectangle([x0, y0, x0 + TILE, y0 + TILE], outline=FRAME, width=2)
        canvas.paste(im, (x0 + 1, y0 + 1))
        tw = draw.textlength(name, font=f_label)
        draw.text((x0 + (TILE - tw) / 2, y0 + TILE + 7), name, fill=INK,
                  font=f_label)

# Legend (colorwheel + caption) in the trailing cell of the training row.
lx0 = col_x[len(ROWS[0][1]) - 1]
ly0 = MARGIN
wheel = Image.open(SPL / "legend__direction_colorwheel.png").convert("RGB")
wsz = TILE - 70
wheel = wheel.resize((wsz, wsz), Image.LANCZOS)
wx = lx0 + (TILE - wsz) // 2
wy = ly0 + 6
canvas.paste(wheel, (wx, wy))
for i, line in enumerate(["hue = flow direction", "extent = cluster spread"]):
    tw = draw.textlength(line, font=f_leg)
    draw.text((lx0 + (TILE - tw) / 2, wy + wsz + 10 + i * 26), line, fill=INK,
              font=f_leg)

out = SPL / "composite_overview.png"
canvas.save(out)
print(f"wrote {out}  ({canvas.size[0]}x{canvas.size[1]})")
