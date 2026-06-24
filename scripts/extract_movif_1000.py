"""Extract N MOVi-F adjacent-frame pairs into a kubric_intervention-format
dataset, so we can train on a fixed 1000-example MOVi-F set with the SAME loader
/recipe as tss_v2 (the object-diversity control). Encodes flow to match
KubricInterventionDataset._decode_flow_png exactly (verified inline)."""
import sys, os, json
sys.path.insert(0, "/home/spencer/Projects/OnlineSyntheticCorrespondence")
import numpy as np, cv2
import imageio.v2 as imageio
from torch.utils.data import DataLoader
from src.data.synth.datasets.MoviFDataset import MoviFSimpleDataset

N = 1010
OUT = "/mnt/nvme_1tb_a/kubric_interventions/datasets/movif_1000_extracted/train"
os.makedirs(OUT, exist_ok=True)

ds = MoviFSimpleDataset(
    datapath="/home/spencer/Data/kubric_tfds/movi_f/512x512/1.0.0",
    split="train", reverse_flow=False,
    kubric_dir="/home/spencer/Projects/kubric", config="512x512",
    shuffle_buffer=64,
)
dl = DataLoader(ds, batch_size=1, num_workers=2)
it = iter(dl)

def encode_flow(dx, dy):
    lo = float(min(dx.min(), dy.min())); hi = float(max(dx.max(), dy.max()))
    if hi <= lo: hi = lo + 1e-3
    enc_dy = np.clip((dy - lo) / (hi - lo), 0, 1) * 65535.0
    enc_dx = np.clip((dx - lo) / (hi - lo), 0, 1) * 65535.0
    # cv2 BGR array: B=0, G=enc(dx), R=enc(dy) -> on read [...,[2,1]]=(enc_dy,enc_dx)=(Kubric dy,dx)
    png = np.stack([np.zeros_like(enc_dx), enc_dx, enc_dy], axis=-1).astype(np.uint16)
    return png, lo, hi

verified = False
for i in range(N):
    b = next(it)
    src = b["src_img"][0].permute(1, 2, 0).numpy()
    trg = b["trg_img"][0].permute(1, 2, 0).numpy()
    flow = b["flow"][0].numpy()          # (2,H,W) = (dx,dy)
    dx, dy = flow[0], flow[1]
    sd = f"{OUT}/scene_{i:06d}"; os.makedirs(sd, exist_ok=True)
    imageio.imwrite(f"{sd}/rgba_00000.png", (np.clip(src, 0, 1) * 255).astype(np.uint8))
    imageio.imwrite(f"{sd}/rgba_00001.png", (np.clip(trg, 0, 1) * 255).astype(np.uint8))
    png, lo, hi = encode_flow(dx, dy)
    cv2.imwrite(f"{sd}/backward_flow_00001.png", png)
    json.dump({"backward_flow": {"min": lo, "max": hi}, "forward_flow": {"min": lo, "max": hi}},
              open(f"{sd}/data_ranges.json", "w"))

    if not verified:  # round-trip check on scene 0 (must match the kubric loader's decode)
        enc = cv2.imread(f"{sd}/backward_flow_00001.png", cv2.IMREAD_UNCHANGED)[..., [2, 1]].astype(np.float32)
        dec = enc / 65535.0 * (hi - lo) + lo            # (H,W,2) = (dy,dx)  [Kubric order]
        rt_dx, rt_dy = dec[..., 1], dec[..., 0]
        err = max(np.abs(rt_dx - dx).max(), np.abs(rt_dy - dy).max())
        print(f"[verify] flow round-trip max err = {err:.4f} px (range [{lo:.1f},{hi:.1f}])")
        verified = True
    if i % 200 == 0: print("extracted", i)
print("DONE", N, "->", OUT)
