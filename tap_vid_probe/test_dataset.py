"""Unit/sanity check for the TAP-Vid-DAVIS dataset + collate pipeline (no model).

Validates that a tapvid_davis batch flows through the REAL CorrespondenceDataset collate
exactly like pointodyssey: shapes, kps in-bounds, dense flow built from kps (finite),
pckthres set, ImageNet-normalised images. Also reconstructs the flow at the keypoints to
confirm direction (trg_kp + flow(trg_kp) ~= src_kp).

Run:  python tap_vid_probe/test_dataset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from src.data.synth.adapters import ADAPTER_REGISTRY

CACHE = os.environ.get("TAPVID_CACHE", "/mnt/nvme_1tb_a/tapvid/probe_cache")


def main():
    assert "tapvid_davis" in ADAPTER_REGISTRY, \
        f"tapvid_davis not registered! registry keys: {sorted(ADAPTER_REGISTRY)}"
    print("[ok] tapvid_davis is registered in ADAPTER_REGISTRY")

    ds = CorrespondenceDataset(
        "tapvid_davis",
        cache_dir=CACHE,
        size=[512, 512],
        downsample_flow=32,
        max_kps=None,
        normalize_images=True,
        tapvid_stride=5,
        tapvid_frame_step=10,
        tapvid_min_pts=1,
        reverse_flow=True,
        thres="img",
    )
    n = len(ds)
    print(f"[ok] dataset length = {n}")
    assert n > 0

    samples = [ds[i] for i in range(4)]
    batch = ds.collate_fn(samples)
    print("[ok] batch keys:", sorted(batch.keys()))

    si = batch["src_img"]
    assert si.shape[1:] == (3, 512, 512), si.shape
    assert si.min() < 0, "images should be ImageNet-normalised (have negatives)"
    print(f"[ok] src_img {tuple(si.shape)} range[{si.min():.2f},{si.max():.2f}]")

    sk, tk = batch["src_kps"], batch["trg_kps"]
    print(f"[ok] src_kps {tuple(sk.shape)} trg_kps {tuple(tk.shape)} n_pts={batch['n_pts'].tolist()}")
    assert sk.shape[1] == 2 and tk.shape[1] == 2

    # in-bounds for the valid (non-padded) keypoints
    for b in range(sk.shape[0]):
        m = int(batch["n_pts"][b])
        if m == 0:
            continue
        v = sk[b, :, :m]
        assert v.min() >= -1 and v.max() <= 513, f"kps out of [0,512]: [{v.min()},{v.max()}]"
    print("[ok] keypoints in-bounds [0,512]")

    pth = batch.get("pckthres")
    print(f"[ok] pckthres = {pth.tolist() if torch.is_tensor(pth) else pth} (expect 512)")
    assert pth is not None and float(pth.flatten()[0]) == 512.0

    flow = batch.get("flow", batch.get("flow_full"))
    fd = batch.get("flow_downsampled")
    # Sparse benchmarks (incl. PointOdyssey): the FULL flow is finite ONLY at the
    # keypoints (inf elsewhere). The downsampled feature flow is made finite. Eval PCK
    # uses kps directly, not the flow, so this sparsity is the intended contract.
    assert torch.isfinite(fd).all(), "downsampled flow has non-finite entries"
    print(f"[ok] flow {tuple(flow.shape)} (sparse); flow_downsampled {tuple(fd.shape)} finite")

    # direction check: full-res flow at integer trg_kp must be finite and
    # trg_kp + flow ~= src_kp (confirms flow_from_kps maps trg->src correctly)
    b = 0
    m = int(batch["n_pts"][b])
    f = flow[b]  # (2,H,W) = (dx,dy)
    errs = []
    for k in range(m):
        x = int(round(float(tk[b, 0, k]))); y = int(round(float(tk[b, 1, k])))
        if 0 <= x < f.shape[2] and 0 <= y < f.shape[1]:
            dx, dy = float(f[0, y, x]), float(f[1, y, x])
            assert torch.isfinite(torch.tensor([dx, dy])).all(), \
                f"flow not finite at trg_kp ({x},{y})"
            pred = torch.tensor([x + dx, y + dy])
            errs.append(float((pred - sk[b, :, k]).norm()))
    assert errs, "no in-bounds keypoints to check direction"
    import statistics
    mean_err = statistics.mean(errs)
    print(f"[ok] flow finite at all {len(errs)} keypoints; direction trg+flow vs src "
          f"mean endpoint err = {mean_err:.2f}px (small => maps trg->src correctly)")
    assert mean_err < 2.0, f"flow direction looks wrong (err {mean_err:.2f}px)"

    print("\nALL DATASET CHECKS PASSED")


if __name__ == "__main__":
    main()
