"""
By-family static-prior diagnostic.

Extends scripts/diagnose_degeneracy.py (CATs++ only) to GLU-Net and FlowFormer,
and adds a SEMANTIC probe condition alongside the real-motion one.

Two input-ablation diagnostics, each computed under TWO conditions:
  * real-motion : consecutive Kubric frame pairs (true frame0 -> frame1 motion)
  * semantic    : SPair-71k image pairs (same category, different instance)

Diagnostics
  id/real ratio    = |flow(A, A)| / |flow(B, A)|   (->1 == input-blind static prior)
  source-sensitiv. = mean abs deviation of flow(A, .) as the source varies
                     (~0 == output ignores the source == static prior; in flow px)

Models trained from scratch that never learn the task emit a (near) constant flow
regardless of input, so id/real -> 1 and sensitivity -> 0. A model with a working
image backbone uses the input: id-flow collapses toward 0 and sensitivity is high.

All models are built scratch/unfrozen and the trained checkpoint is loaded with
strict=False; the saved state_dict contains every parameter (frozen ones too), so
the build-time backbone init is irrelevant.

Run on RC (snapshots + data live there).  One CSV row per (family, source, config,
condition) is printed to stdout; redirect to a file.
"""
import argparse, glob, os, re, sys
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)
from train_lightning import create_model

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_img(p):
    im = Image.open(p).convert('RGB').resize((512, 512))
    return TF.normalize(TF.to_tensor(im), MEAN, STD)


# ----------------------------------------------------------------------------- model
def build_and_load(family, ckpt):
    if family == 'cats':
        cfg = {'type': 'cats', 'backbone': 'resnet101', 'freeze': False,
               'pretrained_backbone': False}
    elif family == 'glunet':
        cfg = {'type': 'glunet', 'pretrained_backbone': False, 'freeze': False,
               'glunet': {'model_name': 'resnet50', 'local_window_size': 9,
                          'decoder_dense_connect': False}}
    elif family == 'flowformer':
        cfg = {'type': 'flowformer', 'pretrain': False, 'freeze': False, 'iters': 12}
    elif family == 'raft':
        # RAFT is always from scratch by construction (no pretrained feature extractor).
        cfg = {'type': 'raft', 'small': False, 'iters': 12}
    else:
        raise ValueError(family)
    m = create_model(cfg, {})
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # report only gross mismatches
    if len(unexpected) > 50:
        print(f"  [warn] {len(unexpected)} unexpected keys loading {ckpt}", file=sys.stderr)
    return m.to(DEV).eval()


@torch.no_grad()
def predict(m, trg, src):
    out = []
    for i in range(0, len(trg), 4):
        f = m(trg[i:i + 4].to(DEV), src[i:i + 4].to(DEV))
        out.append(f.float().cpu())
    return torch.cat(out)[:, :2]   # [N,2,h,w]


def mag(f):
    return f.pow(2).sum(1).sqrt().mean().item()


def roughness(f):
    """Spatial total-variation of the flow field normalised by its magnitude.

    Coherent (piecewise-smooth) correspondence -> low; salt-and-pepper noise from
    a cost volume firing on texture -> high.  Dimensionless.
    """
    dx = (f[:, :, :, 1:] - f[:, :, :, :-1]).abs().mean()
    dy = (f[:, :, 1:, :] - f[:, :, :-1, :]).abs().mean()
    return ((dx + dy) / (mag(f) + 1e-6)).item()


def diagnostics(m, TRG, SRC):
    """TRG[i],SRC[i] are a matched pair (target, source)."""
    id_flow = predict(m, TRG, TRG)          # identical input -> answer is zero flow
    real_flow = predict(m, TRG, SRC)        # real pair
    id_real = mag(id_flow) / max(mag(real_flow), 1e-6)
    flow_mag = mag(real_flow)               # calibration: how big is the predicted flow
    rough = roughness(real_flow)            # spatial coherence of the prediction
    # source-sensitivity: fix each target, swap the source across the batch, measure
    # how much the output moves (mean abs deviation of the flow field, in px).
    n = len(TRG)
    devs = []
    for i in range(n):
        trg_i = TRG[i:i + 1].repeat(n, 1, 1, 1)
        fl = predict(m, trg_i, SRC)         # [n,2,h,w] : same target, all sources
        devs.append((fl - fl.mean(0, keepdim=True)).abs().mean().item())
    return id_real, float(np.mean(devs)), flow_mag, rough


# ----------------------------------------------------------------------------- data
def realmotion_pairs(frames_dir, n=16):
    scenes = sorted(glob.glob(os.path.join(frames_dir, 'scene_*')))
    scenes = scenes[::max(1, len(scenes) // n)][:n]
    F0 = torch.stack([load_img(os.path.join(s, 'rgba_00000.png')) for s in scenes])
    F1 = torch.stack([load_img(os.path.join(s, 'rgba_00001.png')) for s in scenes])
    return F1, F0   # (target, source) = (frame1, frame0)


def spair_pairs(spair_root, n=16):
    """N semantic pairs (trg_img, src_img) from SPair-71k test PairAnnotations.

    Supports both the single ``pair_annotations.json`` dict and per-pair files.
    """
    import json
    jpg = os.path.join(spair_root, 'JPEGImages')
    records = []
    single = os.path.join(spair_root, 'PairAnnotation', 'test', 'pair_annotations.json')
    if os.path.exists(single):
        d = json.load(open(single))
        records = list(d.values())
    else:
        pa = sorted(glob.glob(os.path.join(spair_root, 'PairAnnotation', 'test', '*.json'))) \
            or sorted(glob.glob(os.path.join(spair_root, 'PairAnnotation', 'tst', '*.json')))
        records = [json.load(open(p)) for p in pa]
    records = records[::max(1, len(records) // n)][:n]

    def _path(cat, name):
        name = name if name.lower().endswith('.jpg') else name + '.jpg'
        return os.path.join(jpg, cat, name)

    T, S = [], []
    for r in records:
        cat = r['category']
        T.append(load_img(_path(cat, r['trg_imname'])))
        S.append(load_img(_path(cat, r['src_imname'])))
    return torch.stack(T), torch.stack(S)


# ----------------------------------------------------------------------------- driver
CONFIG_RE = re.compile(r'pretrain(?:ed)?(True|False)_freeze(True|False)', re.I)
# legacy 2-letter codes: 1st = backbone init (T=ImageNet, F=scratch), 2nd = freeze
CODE_RE = re.compile(r'_(FF|FT|TF|TT)$')


def parse_config(path):
    base = os.path.basename(path.rstrip('/'))
    m = CONFIG_RE.search(path)
    if m:
        pre = m.group(1).lower() == 'true'
        frz = m.group(2).lower() == 'true'
        return ('ImageNet' if pre else 'scratch'), ('frozen' if frz else 'trained')
    c = CODE_RE.search(base)
    if c:
        code = c.group(1)
        backbone = 'ImageNet' if code[0] == 'T' else 'scratch'
        encoder = 'frozen' if code[1] == 'T' else 'trained'
        return backbone, encoder
    if 'raft' in base.lower():
        return 'scratch', 'trained'   # RAFT is always scratch by construction
    return None


def parse_source(path, family):
    base = os.path.basename(path.rstrip('/'))
    c = CODE_RE.search(base)
    if c:
        return base[:c.start()]
    tag = {'cats': '_cats', 'glunet': '_glunet', 'flowformer': '_flowformer',
           'raft': '_raft'}[family]
    return base.split(tag)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True, choices=['cats', 'glunet', 'flowformer', 'raft'])
    ap.add_argument('--glob', required=True, help='glob for snapshot dirs')
    ap.add_argument('--ckpt-name', default='model_best.pth')
    ap.add_argument('--realframes', required=True)
    ap.add_argument('--spair', required=True)
    ap.add_argument('--n-pairs', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0, help='smoke-test: only first K dirs')
    args = ap.parse_args()

    TRG_rm, SRC_rm = realmotion_pairs(args.realframes, args.n_pairs)
    TRG_sm, SRC_sm = spair_pairs(args.spair, args.n_pairs)
    print(f"# device={DEV} realmotion={len(TRG_rm)} semantic={len(TRG_sm)}", file=sys.stderr)
    print("family,source,backbone,encoder,condition,id_real_ratio,source_sensitivity,flow_mag,roughness")

    dirs = sorted(glob.glob(args.glob))
    if args.limit:
        dirs = dirs[:args.limit]
    for d in dirs:
        cfg = parse_config(d)
        if cfg is None:
            continue
        backbone, encoder = cfg
        source = parse_source(d, args.family)
        ckpt = os.path.join(d, args.ckpt_name)
        if not os.path.exists(ckpt):
            print(f"  [skip] no {args.ckpt_name} in {d}", file=sys.stderr)
            continue
        try:
            m = build_and_load(args.family, ckpt)
        except Exception as e:
            print(f"  [err] build/load {d}: {e}", file=sys.stderr)
            continue
        for cond, (T, S) in [('realmotion', (TRG_rm, SRC_rm)), ('semantic', (TRG_sm, SRC_sm))]:
            try:
                idr, sens, fmag, rough = diagnostics(m, T, S)
                print(f"{args.family},{source},{backbone},{encoder},{cond},"
                      f"{idr:.4f},{sens:.4f},{fmag:.4f},{rough:.4f}")
                sys.stdout.flush()
            except Exception as e:
                print(f"  [err] {cond} {d}: {e}", file=sys.stderr)
        del m
        if DEV == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
