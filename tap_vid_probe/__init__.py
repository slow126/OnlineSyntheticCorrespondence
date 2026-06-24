"""tap_vid_probe — quarantined TAP-Vid-DAVIS eval benchmark + MOVi-F training probe.

Everything new for the TAP-Vid experiment lives here. The only edits to shared code are
two tiny, guarded hooks that point back at this package:
  - src/data/synth/adapters.py : guarded registration of TapVidDavisAdapter
  - train_cats_unified.py       : one `elif benchmark == 'tapvid_davis'` branch

Delete this directory and remove those two hooks to fully revert.
"""
from tap_vid_probe.tapvid_davis_dataset import (  # noqa: F401
    TapVidDavisAdapter,
    TapVidDavisSimpleDataset,
)
