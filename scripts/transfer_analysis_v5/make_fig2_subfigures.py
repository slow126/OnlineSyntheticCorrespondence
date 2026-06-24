"""Split composite_rgb_motion.png into two subfigure files for ACCV Fig 2.

Produces:
  ACCV_2026/figures/splats/fig2a_splats.png  -- motion fingerprints block
  ACCV_2026/figures/splats/fig2b_rgb.png     -- RGB samples block

The 3 shared columns (FlyingThings / PointOdyssey / SPair) are grouped by
coloring the white background/padding pixels in that band — the actual splat and
RGB content is untouched; only the surrounding whitespace gets a light tint.

Crop boundaries:
  TOP_START=56   skips the 56px title+padding region above the splat tiles
  TOP_END=510    includes to the end of the benchmark tile row
  BOT_START=631  bottom block starts with tiles immediately (no title area)
  BOT_END=1084   drops the trailing 9px white margin

Usage:
  python scripts/transfer_analysis_v5/make_fig2_subfigures.py
"""
from pathlib import Path
import numpy as np
from PIL import Image

SRC    = Path("ACCV_2026/figures/splats/composite_rgb_motion.png")
OUT_A  = Path("ACCV_2026/figures/splats/fig2a_splats.png")
OUT_B  = Path("ACCV_2026/figures/splats/fig2b_rgb.png")

# ── Crop boundaries ───────────────────────────────────────────────────────────
# TOP_START=56: skips the header/title rows (0-55) above the tile area.
#   Rows 0-9: top margin; rows 10-25: title annotation above non-shared cols;
#   rows 26-55: padding. Tiles themselves start at row 56.
# BOT_END=1084: removes the 9 trailing white rows (1084-1092) for tight bottom.
TOP_START = 56
TOP_END   = 510
BOT_START = 631
BOT_END   = 1084

# ── Shared-column band ────────────────────────────────────────────────────────
# Tile separators confirmed at cols 27-33 (before tile 0) and 641-647 (after
# tile 2). Cover left edge through the end of the gap after tile 2.
HL_X0 = 0     # left edge of image
HL_X1 = 648   # just past the gap after tile 2 (gap is cols 641-647)

# Background-pixel threshold: pixels where all RGB channels > THRESH are
# treated as whitespace / padding (not figure content).
BG_THRESH = 230

# Highlight colour for the background: soft blue-gray (ACCV-neutral)
HL_COLOR = np.array([218, 229, 245], dtype=np.uint8)   # light cool blue


def _tint_background(arr: np.ndarray, x0: int, x1: int) -> np.ndarray:
    """Replace near-white background pixels in columns [x0:x1] with HL_COLOR.

    All actual figure content (splat pixels, tile borders, label text on
    non-white) is left exactly as-is; only whitespace gets the tint.
    """
    out = arr.copy()
    region = out[:, x0:x1, :3]
    is_bg = np.all(region > BG_THRESH, axis=2)   # True where pixel ≈ white
    region[is_bg] = HL_COLOR
    out[:, x0:x1, :3] = region
    return out


# ── Load ─────────────────────────────────────────────────────────────────────
img = Image.open(SRC).convert("RGB")
arr = np.array(img)

# ── Crop and tint ─────────────────────────────────────────────────────────────
top_arr = _tint_background(arr[TOP_START:TOP_END].copy(), HL_X0, HL_X1)
bot_arr = _tint_background(arr[BOT_START:BOT_END].copy(), HL_X0, HL_X1)

# ── Save ─────────────────────────────────────────────────────────────────────
Image.fromarray(top_arr).save(OUT_A, dpi=(105, 105))
Image.fromarray(bot_arr).save(OUT_B, dpi=(105, 105))
print(f"Saved {OUT_A}  ({top_arr.shape[1]}x{top_arr.shape[0]})")
print(f"Saved {OUT_B}  ({bot_arr.shape[1]}x{bot_arr.shape[0]})")
