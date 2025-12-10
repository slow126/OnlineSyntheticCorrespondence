# Dataset Addition Analysis: Easiest to Implement

Based on your current `CorrespondenceDataset` architecture, here's an analysis of which datasets from your list are easiest to add:

## Current Architecture

Your codebase uses:
- **`CorrespondenceDataset`** wrapper that delegates to **adapters** in `src/data/synth/adapters.py`
- Adapters return `CommonSample` objects with `src_img`, `trg_img`, `flow_full`, etc.
- Currently supported: `flyingthings`, `pointodyssey`, `kitti`, `synthetic`, `tss`, `middlebury`, `pfpascal`, `pfwillow`, `spair`

---

## Easiest to Implement (⭐⭐⭐)

### 1. **Monkaa (SceneFlow)** - ~30 minutes
- **Why easy**: Uses `torchvision.datasets.SceneFlowStereo` (same as FlyingThings3D)
- **Implementation**: Copy `FlyingThingsAdapter`, change `variant='Monkaa'`
- **Code location**: `src/data/synth/adapters.py` + `src/data/synth/datasets/FlyingThingsDataset.py`
- **Steps**:
  1. Create `MonkaaSimpleDataset` (copy `FlyingThingsSimpleDataset`, use `SceneFlowStereo(variant='Monkaa')`)
  2. Create `MonkaaAdapter` (copy `FlyingThingsAdapter`)
  3. Register in `ADAPTER_REGISTRY`

### 2. **Driving (SceneFlow)** - ~30 minutes
- **Why easy**: Same as Monkaa, just different variant
- **Implementation**: Copy `FlyingThingsAdapter`, change `variant='Driving'`
- **Steps**: Same as Monkaa, use `variant='Driving'`

### 3. **MPI Sintel** - ~1 hour
- **Why easy**: Uses `torchvision.datasets.Sintel` (official torchvision support)
- **Implementation**: Similar to FlyingThings, but uses `Sintel` class
- **Note**: Supports `pass_name='clean'` or `'final'` (you may want both)
- **Steps**:
  1. Create `SintelSimpleDataset` using `datasets.Sintel`
  2. Create `SintelAdapter`
  3. Register in `ADAPTER_REGISTRY`
  4. Handle `pass_name` parameter (clean/final)

---

## Easy to Implement (⭐⭐)

### 4. **Virtual KITTI 2** - ~2-3 hours
- **Why easy**: Similar structure to KITTI (you already have `KittiAdapter`)
- **Implementation**: Follow `KittiSimpleDataset` pattern
- **Note**: May need custom flow loading (check if it uses same PNG format as KITTI)
- **Steps**:
  1. Create `VirtualKitti2SimpleDataset` (similar to `KittiSimpleDataset`)
  2. Create `VirtualKitti2Adapter` (copy `KittiAdapter`)
  3. Register in `ADAPTER_REGISTRY`

### 5. **HD1K** - ~2-3 hours
- **Why easy**: Real driving data, similar to KITTI
- **Implementation**: Follow `KittiSimpleDataset` pattern
- **Note**: Check flow format (may be `.flo` like Middlebury or PNG like KITTI)
- **Steps**: Same as Virtual KITTI 2

---

## Medium Difficulty (⭐)

### 6. **TAP-Vid** - ~4-6 hours
- **Why medium**: There's already a `TapVidDavis` class in `src/data/synth/datasets/pips2/datasets/tapviddataset_fullseq.py`, but it's for tracking, not correspondence
- **Implementation**: Need to adapt for correspondence format (sparse 2D point tracks → flow/keypoints)
- **Note**: TAP-Vid provides sparse tracks, so you'll need to convert to flow or keypoints
- **Steps**:
  1. Create `TapVidSimpleDataset` that loads sparse tracks
  2. Convert tracks to keypoints or flow format
  3. Create `TapVidAdapter`
  4. Register in `ADAPTER_REGISTRY`

---

## Harder to Implement (Requires More Investigation)

### 7. **Spring (Blender movie dataset)** - Unknown
- **Why harder**: Custom format, need to investigate structure
- **Implementation**: Full custom dataset class needed
- **Note**: Check if it provides flow ground truth or just images

### 8. **AutoFlow** - Unknown
- **Why harder**: Custom format, need to investigate structure
- **Implementation**: Full custom dataset class needed
- **Note**: Check dataset format and flow representation

---

## Recommended Implementation Order

1. **Start with Monkaa** (easiest, ~30 min) - validates the pattern
2. **Then Driving** (same pattern, ~30 min)
3. **Then Sintel** (~1 hour) - adds torchvision dataset variety
4. **Then Virtual KITTI 2** (~2-3 hours) - validates KITTI-like pattern
5. **Then HD1K** (~2-3 hours) - completes real driving datasets
6. **Then TAP-Vid** (~4-6 hours) - adds sparse tracking variety
7. **Finally Spring/AutoFlow** (investigate first)

---

## Implementation Pattern (for Monkaa/Driving/Sintel)

Here's the pattern you'll follow:

```python
# In src/data/synth/datasets/MonkaaDataset.py (or similar)
class MonkaaSimpleDataset(Dataset, nn.Module):
    def __init__(self, root: str, split: str, transforms=None, reverse_flow: bool = False):
        Dataset.__init__(self)
        nn.Module.__init__(self)
        self.dataset = datasets.SceneFlowStereo(root=root, variant='Monkaa', split=split, transforms=transforms)
        self.reverse_flow = reverse_flow
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Same logic as FlyingThingsSimpleDataset
        ...

# In src/data/synth/adapters.py
class MonkaaAdapter(BaseAdapter):
    name = "monkaa"
    def __init__(self, datapath: str, split: str, reverse_flow: bool = False, **kwargs):
        self.dataset = MonkaaSimpleDataset(root=datapath, split=split, reverse_flow=reverse_flow)
    # Same as FlyingThingsAdapter
    ...

# Register in ADAPTER_REGISTRY
ADAPTER_REGISTRY = {
    ...
    "monkaa": MonkaaAdapter,
    "driving": DrivingAdapter,
    "sintel": SintelAdapter,
}
```

---

## Next Steps

Would you like me to:
1. Implement Monkaa and Driving first (easiest)?
2. Implement Sintel?
3. Create a template/example for one of them that you can use as a reference?

Let me know which one you'd like to start with!

