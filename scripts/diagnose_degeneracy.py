import torch, glob, sys
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
sys.path.insert(0, '/home/spencer/Projects/OnlineSyntheticCorrespondence')
from models.CATs_PlusPlus.models.cats_improved import CATsImproved

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN=[0.485,0.456,0.406]; STD=[0.229,0.224,0.225]

def load_img(p):
    im = Image.open(p).convert('RGB').resize((512,512))
    return TF.normalize(TF.to_tensor(im), MEAN, STD)

CK = {
 'movi_f  FF (scratch)':   ('/mnt/nvme_1tb_a/cats_lolr_ckpts/movi_f_FF/model_best.pth'),
 'movi_f  FT (rand-froz)': ('/mnt/nvme_1tb_a/cats_lolr_ckpts/movi_f_FT/model_best.pth'),
 'spair   FF (scratch)':   ('/mnt/nvme_1tb_a/cats_lolr_ckpts/spair_FF/model_best.pth'),
 'pretrained TF (HEALTHY)':('/mnt/nvme_1tb_b/snapshots_synth_2d/synthetic_2d_warp_cats_steps100_pretrainedTrue_freezeFalse_2026_01_14_09_26/model_best.pth'),
}

# real consecutive-frame pairs from varied scenes: f0 (frame0), f1 (frame1)
scenes = sorted(glob.glob('/mnt/nvme_1tb_a/kubric_interventions/datasets/*_5000/train/scene_*'))
scenes = scenes[::max(1,len(scenes)//16)][:16]
F0 = torch.stack([load_img(s+'/rgba_00000.png') for s in scenes])
F1 = torch.stack([load_img(s+'/rgba_00001.png') for s in scenes])
print(f"device={dev}  n_pairs={len(F0)}")

def predict(m, trg, src):
    out=[]
    with torch.no_grad():
        for i in range(0, len(trg), 4):
            f = m(trg[i:i+4].to(dev), src[i:i+4].to(dev))
            out.append(f.float().cpu())
    return torch.cat(out)[:, :2]    # [N,2,H,W]

def mag(f): return f.pow(2).sum(1).sqrt().mean().item()

print(f"\n{'model':26s} {'identity |flow|':>15s} {'real-pair |flow|':>16s} {'id/real ratio':>13s}")
for name, path in CK.items():
    m = CATsImproved(backbone='resnet101', freeze=False, pretrained_backbone=False).to(dev).eval()
    m.load_state_dict(torch.load(path, map_location='cpu')['state_dict'], strict=False)
    id_f   = predict(m, F0, F0)     # identical pair -> answer is ZERO flow
    real_f = predict(m, F1, F0)     # real motion pair
    im, rm = mag(id_f), mag(real_f)
    print(f"{name:26s} {im:15.3f} {rm:16.3f} {im/max(rm,1e-6):13.2f}")
