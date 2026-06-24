"""End-to-end smoke eval: run ONE trained checkpoint on TAP-Vid-DAVIS.

This is the direction-critical check the handoff flagged: it evaluates PCK the exact way
the training harness does (net(trg,src) -> flow2kps(trg_kps) -> eval_kps_transfer) and
then FLIPS src/trg to confirm PCK collapses (a backwards src/trg convention silently
halves PCK, so the gap between normal and flipped is the proof the wiring is right).

Default checkpoint = CATs++ movi_f_FT (pretrained-frozen, best_avg_pck ~46). Run:
  python tap_vid_probe/smoke_eval.py
  python tap_vid_probe/smoke_eval.py --ckpt /path/to/model_best.pth --model cats
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_lightning import create_model
from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset
from models.CATs_PlusPlus.utils_training.utils import flow2kps
from models.CATs_PlusPlus.utils_training.eval_instance import EvaluatorInstance

CACHE = os.environ.get("TAPVID_CACHE", "/mnt/nvme_1tb_a/tapvid/probe_cache")


def build_loader(stride, frame_step, downsample_flow):
    ds = CorrespondenceDataset(
        "tapvid_davis", cache_dir=CACHE, size=[512, 512], downsample_flow=downsample_flow,
        max_kps=None, normalize_images=True, tapvid_stride=stride,
        tapvid_frame_step=frame_step, tapvid_min_pts=1, reverse_flow=True, thres="img",
    )
    return DataLoader(ds, batch_size=4, num_workers=4, shuffle=False, collate_fn=ds.collate_fn)


@torch.no_grad()
def eval_pck(net, loader, device, alpha=0.05, flip=False):
    ev = EvaluatorInstance("tapvid_davis", alpha=alpha)
    net.eval()
    pcks = []
    for batch in loader:
        b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        src_img, trg_img = b["src_img"], b["trg_img"]
        src_kps, trg_kps = b["src_kps"], b["trg_kps"]
        if flip:                       # backwards convention -> PCK should collapse
            src_img, trg_img = trg_img, src_img
            src_kps, trg_kps = trg_kps, src_kps
        pred = net(trg_img, src_img)
        est = flow2kps(trg_kps, pred, b["n_pts"])
        r = ev.evaluate(est, {"src_kps": src_kps, "trg_kps": trg_kps, "pckthres": b["pckthres"],
                              "n_pts": b["n_pts"]})
        pcks.extend(r["pck"])          # per-sample PCK, already in 0-100
    return float(np.mean(pcks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/nvme_1tb_a/cats_lolr_ckpts/movi_f_FT/model_best.pth")
    ap.add_argument("--model", default="cats", choices=["cats", "glunet", "flowformer", "raft"])
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--frame_step", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = {"cats": {"type": "cats", "backbone": "resnet101", "freeze": False, "pretrained_backbone": True}}[args.model]
    net = create_model(mcfg, {}).to(device)

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("state_dict", ck)
    missing, unexpected = net.load_state_dict(sd, strict=False)
    print(f"[smoke] loaded {args.ckpt}\n  epoch={ck.get('epoch')} best_avg_pck={ck.get('best_avg_pck')}")
    print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")

    # CATs uses downsample_flow=32; dense models use null. cats checkpoint -> 32.
    loader = build_loader(args.stride, args.frame_step, downsample_flow=32)

    pck = eval_pck(net, loader, device, flip=False)
    pck_flip = eval_pck(net, loader, device, flip=True)
    print(f"\n=== TAP-Vid-DAVIS PCK@0.05 (stride={args.stride}) ===")
    print(f"  normal (trg->src):   {pck:.2f}%")
    print(f"  flipped (src<->trg): {pck_flip:.2f}%")
    print(f"  gap:                 {pck - pck_flip:+.2f} pts")
    ok = (2.0 < pck < 99.0) and (pck > pck_flip + 2.0)
    print("\n" + ("PASS: plausible PCK and direction confirmed (normal > flipped)" if ok
                  else "WARNING: check PCK / direction — normal not clearly > flipped"))


if __name__ == "__main__":
    main()
