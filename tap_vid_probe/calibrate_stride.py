"""Pick a (stride, alpha) where TAP-Vid-DAVIS PCK is DISCRIMINATIVE.

At small stride the point motion is below the PCK threshold (alpha*512), so a zero-flow
"identity" prediction already scores high -> PCK is saturated and won't move during
training -> useless as a proxy. This finds an operating point where the identity baseline
is low (so the trained model has room to win) and motion is still trackable.

Step 1 (no model): for each stride/alpha, report mean/median point displacement and the
identity-baseline PCK (% of co-visible points whose true displacement < alpha*512).
Step 2 (model): for a few promising points, run the CATs movi_f model normal vs flipped
to confirm model >> identity and normal > flipped.

Run:  python tap_vid_probe/calibrate_stride.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle

CACHE = os.environ.get("TAPVID_CACHE", "/mnt/nvme_1tb_a/tapvid/probe_cache")
SIZE = 512


def identity_stats(stride, frame_step=5):
    """Return (n_pairs, n_pts, mean_disp, median_disp, {alpha: identity_pck})."""
    with open(os.path.join(CACHE, "index.pkl"), "rb") as f:
        index = pickle.load(f)
    disps = []
    for vid in index:
        occ = vid["occluded"]; pts = vid["points512"]; S = vid["S"]
        for t1 in range(0, S - stride, frame_step):
            t2 = t1 + stride
            covis = (~occ[:, t1]) & (~occ[:, t2])
            idx = np.nonzero(covis)[0]
            if idx.size == 0:
                continue
            d = np.linalg.norm(pts[idx, t2, :] - pts[idx, t1, :], axis=1)  # 512-space px
            disps.append(d)
    disps = np.concatenate(disps) if disps else np.array([0.0])
    out = {}
    for a in (0.05, 0.03, 0.02, 0.01):
        out[a] = float((disps < a * SIZE).mean()) * 100
    n_pairs = sum(1 for vid in index for t1 in range(0, vid["S"] - stride, frame_step)
                  if ((~vid["occluded"][:, t1]) & (~vid["occluded"][:, t1 + stride])).any())
    return n_pairs, disps.size, float(disps.mean()), float(np.median(disps)), out


def main():
    print("=== Step 1: identity-baseline PCK (no model) — want LOW so model can win ===")
    print(f"{'stride':>6} {'pairs':>6} {'pts':>6} {'meanD':>7} {'medD':>6} | "
          f"{'id@.05':>7} {'id@.03':>7} {'id@.02':>7} {'id@.01':>7}")
    rows = {}
    for stride in (5, 10, 15, 20, 25, 30, 40):
        np_, npt, mean_d, med_d, ids = identity_stats(stride)
        rows[stride] = (mean_d, ids)
        print(f"{stride:6d} {np_:6d} {npt:6d} {mean_d:7.1f} {med_d:6.1f} | "
              f"{ids[0.05]:6.1f}% {ids[0.03]:6.1f}% {ids[0.02]:6.1f}% {ids[0.01]:6.1f}%")

    # Step 2: confirm a trained model beats identity + flipped at a promising point.
    import torch
    from torch.utils.data import DataLoader
    from train_lightning import create_model
    from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
    from models.CATs_PlusPlus.utils_training.utils import flow2kps
    from models.CATs_PlusPlus.utils_training.eval_instance import EvaluatorInstance

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = create_model({"type": "cats", "backbone": "resnet101", "freeze": False,
                        "pretrained_backbone": True}, {}).to(device)
    ck = torch.load("/mnt/nvme_1tb_a/cats_lolr_ckpts/movi_f_FT/model_best.pth", map_location="cpu")
    net.load_state_dict(ck.get("state_dict", ck), strict=False)
    net.eval()

    @torch.no_grad()
    def model_pck(stride, alpha, flip=False):
        ds = CorrespondenceDataset("tapvid_davis", cache_dir=CACHE, size=[512, 512],
                                   downsample_flow=32, max_kps=None, normalize_images=True,
                                   tapvid_stride=stride, tapvid_frame_step=5, tapvid_min_pts=1,
                                   reverse_flow=True, thres="img", verbose=False)
        dl = DataLoader(ds, batch_size=4, num_workers=4, shuffle=False, collate_fn=ds.collate_fn)
        ev = EvaluatorInstance("tapvid_davis", alpha=alpha)
        pcks = []
        for b in dl:
            b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            si, ti, sk, tk = b["src_img"], b["trg_img"], b["src_kps"], b["trg_kps"]
            if flip:
                si, ti, sk, tk = ti, si, tk, sk
            est = flow2kps(tk, net(ti, si), b["n_pts"])
            pcks.extend(ev.evaluate(est, {"src_kps": sk, "trg_kps": tk,
                                          "pckthres": b["pckthres"], "n_pts": b["n_pts"]})["pck"])
        return float(np.mean(pcks))

    print("\n=== Step 2: CATs movi_f_FT model vs identity vs flipped ===")
    print(f"{'stride':>6} {'alpha':>6} | {'identity':>9} {'model':>7} {'flipped':>8} "
          f"{'mdl-id':>7} {'mdl-flip':>8}")
    for stride, alpha in [(15, 0.05), (20, 0.05), (20, 0.03), (25, 0.03), (30, 0.03), (30, 0.02)]:
        idpck = rows[stride][1][alpha]
        m = model_pck(stride, alpha, flip=False)
        mf = model_pck(stride, alpha, flip=True)
        print(f"{stride:6d} {alpha:6.2f} | {idpck:8.1f}% {m:6.1f}% {mf:7.1f}% "
              f"{m-idpck:+6.1f} {m-mf:+7.1f}")
    print("\nPick the row with the largest (model - identity) AND (model - flipped) gaps:\n"
          "that stride/alpha makes TAP-Vid track training instead of sitting saturated.")


if __name__ == "__main__":
    main()
