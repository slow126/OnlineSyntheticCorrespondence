#!/usr/bin/env python3
"""Re-score the synthetic_fractal_trial76 FF checkpoints on K2012/Mid/FT (+K2015 sanity).

Reuses the EXACT training-time eval (validate_epoch_multi_benchmark) so PCK is
comparable to every other transfer-grid cell. K2015 is included as a cross-check:
the rescored K2015 trajectory must reproduce the logged validation_results.csv
(epoch 26 -> 94.43), which proves the harness is faithful.

Usage:
  CUDA_VISIBLE_DEVICES=1 python score_synthetic_fractal.py            # all 50 epochs
  CUDA_VISIBLE_DEVICES=1 python score_synthetic_fractal.py 26 50      # specific epochs
"""
import sys, os, re, csv, glob, yaml
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from train_lightning import create_model
from train_cats_unified import create_validation_datasets
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark

CELL = "/mnt/nvme_1tb_a/snapshots/transfer_grid/synthetic_fractal_trial76_pt0_fz0_2026_06_09_08_37"
BENCH = ['kitti2015', 'kitti2012', 'middlebury', 'flyingthings']
ALPHAS = [0.05, 0.05, 0.05, 0.05]

cfg = yaml.safe_load(open(os.path.join(CELL, "config.yaml")))
cfg['evaluation']['eval_benchmarks'] = BENCH
cfg['evaluation']['eval_alphas'] = ALPHAS
# Inject benchmark root paths absent from the K2015-only config (from the 4-benchmark cells)
cfg['evaluation'].setdefault('kitti_root', '/home/spencer/Data/correspondence/kitti')
cfg['evaluation'].setdefault('middlebury_root', '/home/spencer/Data/middlebury/all')
cfg['evaluation'].setdefault('flyingthings_root', '/home/spencer/Data/FlyingThings3D_tiny')
ev = cfg['evaluation']
use_ma = ev.get('use_motion_aware', False)
min_mp = ev.get('min_motion_pixels', 5.0)
zthr   = ev.get('zero_threshold', 0.5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device={device}  benchmarks={BENCH}", flush=True)

# ---- build val loaders + evaluator + model ONCE ----
multi_evaluator = MultiBenchmarkEvaluator(dict(zip(BENCH, ALPHAS)))
print("evaluator benchmarks:", multi_evaluator.get_available_benchmarks(), flush=True)
val_datasets, val_loaders = create_validation_datasets(cfg, device=device)
for b, dl in val_loaders.items():
    print(f"  val loader {b}: {len(dl)} batches", flush=True)
model = create_model(cfg['model'], cfg['paths'])
model.to(device).eval()

# ---- which epochs ----
all_eps = sorted(int(re.search(r'epoch_(\d+)\.pth', p).group(1))
                 for p in glob.glob(os.path.join(CELL, 'epoch_*.pth')))
epochs = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else all_eps
print(f"scoring {len(epochs)} epochs: {epochs}", flush=True)

best = {b: (-1.0, None) for b in BENCH}
rows = []
for ep in epochs:
    sd = torch.load(os.path.join(CELL, f'epoch_{ep}.pth'), map_location='cpu')['state_dict']
    miss, unexp = model.load_state_dict(sd, strict=False)
    if ep == epochs[0] and (miss or unexp):
        print(f"  [load] missing={len(miss)} unexpected={len(unexp)} (first miss: {miss[:2]})", flush=True)
    model.eval()
    with torch.no_grad():
        res = validate_epoch_multi_benchmark(
            net=model, val_loaders=val_loaders, device=device, epoch=ep,
            multi_evaluator=multi_evaluator, primary_benchmark=BENCH[0],
            use_motion_aware=use_ma, min_motion_pixels=min_mp,
            zero_threshold=zthr, mmd_every_n_epochs=0)
    line = []
    for b in BENCH:
        pck = float(res[b]['pck'])
        rows.append((ep, b, round(pck, 4)))
        if pck > best[b][0]:
            best[b] = (pck, ep)
        line.append(f"{b}={pck:.2f}")
    print(f"ep{ep:>2}: " + "  ".join(line), flush=True)

print("\n==== BEST PER BENCHMARK (max over scored epochs) ====")
for b in BENCH:
    print(f"  {b:<13}: {best[b][0]:.2f}%  (epoch {best[b][1]})")
ks = best['kitti2015'][0]
print(f"\nSANITY: rescored K2015 best = {ks:.2f}  (logged training best = 94.43; should match)")

out = os.path.join(CELL, 'rescored_results.csv')
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['epoch', 'benchmark', 'pck'])
    w.writerows(rows)
print("wrote", out)
