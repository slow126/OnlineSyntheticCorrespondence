from pathlib import Path

import numpy as np


def load_curvature():
    p = Path(__file__).parent.joinpath('julia_curvature_13.npz')
    f = np.load(p)
    curve, points = f['curve_grid'], f['grid_points']
    return curve, points