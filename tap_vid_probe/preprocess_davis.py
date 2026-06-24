"""One-time preprocessing: TAP-Vid-DAVIS pickle -> compact mmap cache.

The raw `tapvid_davis.pkl` (2.48 GB) holds full-resolution videos. For the probe we
only ever feed 512x512 image pairs to the models, so we pre-resize every frame to
512x512 uint8 and pre-scale the query points into 512-pixel space ONCE, then store:

  <out>/frames.npy   (TotalFrames, 512, 512, 3) uint8  -- memory-mapped at train time
  <out>/index.pkl    list[dict] per video: {name, offset, S, points512 (N,S,2) f32,
                                            occluded (N,S) bool}

Why mmap: we run several training jobs concurrently (one per GPU). With np.load(
mmap_mode='r') every process shares the SAME frames via the OS page cache instead of
each allocating ~1.8 GB. The dataset loader (tapvid_davis_dataset.py) consumes this.

TAP-Vid point convention (verified empirically 2026-06-24): points are float32
normalized to [0,1], last axis = (x, y), x normalized by width, y by height. Resizing
the frame W x H -> 512 x 512 maps x_pix=x_norm*W -> x_norm*512 and likewise for y, so
the 512-space coords are simply points_norm * 512 regardless of original aspect ratio.

Run:
  python tap_vid_probe/preprocess_davis.py \
      --pkl /mnt/nvme_1tb_a/tapvid/tapvid_davis/tapvid_davis.pkl \
      --out /mnt/nvme_1tb_a/tapvid/probe_cache
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

SIZE = 512


def resize_frames(frames_uint8: np.ndarray, size: int = SIZE) -> np.ndarray:
    """(S,H,W,3) uint8 -> (S,size,size,3) uint8 via bilinear (align_corners=False).

    Matches the resize convention used by the collate pipeline (_resize_tensor)."""
    # to (S,3,H,W) float
    t = torch.from_numpy(frames_uint8).permute(0, 3, 1, 2).float()
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    t = t.clamp(0, 255).round().to(torch.uint8)
    return t.permute(0, 2, 3, 1).contiguous().numpy()  # (S,size,size,3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="/mnt/nvme_1tb_a/tapvid/tapvid_davis/tapvid_davis.pkl")
    ap.add_argument("--out", default="/mnt/nvme_1tb_a/tapvid/probe_cache")
    ap.add_argument("--size", type=int, default=SIZE)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    print(f"[preprocess] loading {args.pkl} ...", flush=True)
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print(f"[preprocess] loaded {len(data)} videos in {time.time()-t0:.1f}s", flush=True)

    # First pass: count total frames so we can allocate a single frames.npy
    names = list(data.keys())
    sizes = [int(np.asarray(data[k]["video"]).shape[0]) for k in names]
    total = sum(sizes)
    print(f"[preprocess] total frames={total} -> "
          f"{total*args.size*args.size*3/1e9:.2f} GB at {args.size}px", flush=True)

    frames = np.lib.format.open_memmap(
        os.path.join(args.out, "frames.npy"),
        mode="w+", dtype=np.uint8, shape=(total, args.size, args.size, 3),
    )

    index = []
    offset = 0
    for k, S in zip(names, sizes):
        rec = data[k]
        vid = np.asarray(rec["video"])          # (S,H,W,3) uint8
        pts = np.asarray(rec["points"], np.float32)  # (N,S,2) normalized (x,y)
        occ = np.asarray(rec["occluded"], bool)      # (N,S)
        assert vid.shape[0] == S and pts.shape[1] == S and occ.shape[1] == S, \
            f"{k}: frame-count mismatch vid={vid.shape} pts={pts.shape} occ={occ.shape}"

        frames[offset:offset + S] = resize_frames(vid, args.size)
        points512 = pts * args.size  # (N,S,2) in 512-pixel space
        index.append(dict(name=k, offset=offset, S=S,
                          points512=points512.astype(np.float32),
                          occluded=occ))
        offset += S
        print(f"  [{len(index):2d}/{len(names)}] {k:18s} S={S:3d} N={pts.shape[0]:3d}", flush=True)

    frames.flush()
    with open(os.path.join(args.out, "index.pkl"), "wb") as f:
        pickle.dump(index, f)

    print(f"[preprocess] DONE in {time.time()-t0:.1f}s -> {args.out}", flush=True)
    print(f"[preprocess] frames.npy shape=({total},{args.size},{args.size},3) uint8", flush=True)


if __name__ == "__main__":
    main()
