#!/usr/bin/env python3
"""Post-hoc eval for one transfer cell: KITTI (multi-alpha) + TSS (multi-alpha) +
TAP-Vid-DAVIS (multi-stride), all on the saved epoch_*.pth, peak over epochs.

Generalizes score_multi_alpha_kitti.py. Reuses the EXACT training-time eval
(validate_epoch_multi_benchmark + flow2kps + classify_prd) for fidelity, and the
proven benchmark-naming convention the unified trainer already understands:
  - KITTI  : 'kitti2012' / 'kitti2015' at primary alpha 0.05; extra alphas 0.03/0.01
             scored on the SAME predictions in one pass (MultiAlphaEvaluator).
  - TSS    : 'tss_a10' / 'tss_a05' / 'tss_a03'  (same data, alpha via eval_alphas).
  - TAP-Vid: 'tapvid_davis_s{N}' for N in strides (each its own loader/stride).

All benchmarks evaluate in ONE validate() pass per epoch. Per (benchmark, alpha/stride)
peak over the scored epochs is reported and written to a tidy CSV.

Usage:
  CUDA_VISIBLE_DEVICES=1 python score_transfer_cell.py --cell <DIR>
  ... --families kitti tss tapvid --kitti-alphas 0.05 0.03 0.01 \
      --tss-alphas 0.10 0.05 0.03 --tapvid-strides 1 2 4 8 16 --tapvid-alpha 0.05
  ... --ckpt model_best.pth      # single checkpoint instead of epoch sweep
"""
import sys, os, re, csv, glob, yaml, argparse
import torch

HARNESS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HARNESS)
sys.path.insert(0, REPO)
from train_lightning import create_model
from train_cats_unified import create_validation_datasets
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator, EvaluatorInstance
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark

LOCAL_ROOTS = {
    'kitti_root':        '/home/spencer/Data/correspondence/kitti',
    'tss_root':          '/home/spencer/Data/correspondence/TSS_CVPR2016',
    'tapvid_davis_root': '/mnt/nvme_1tb_a/tapvid/probe_cache',
}


class MultiAlphaEvaluator(MultiBenchmarkEvaluator):
    """MultiBenchmarkEvaluator that also scores per-benchmark EXTRA alphas in one pass.

    `extra_map`: {benchmark: [alpha, ...]}. The primary alpha behaves exactly like the
    parent (so validate()'s reported pck is unchanged & reproduces logged @0.05); each
    extra alpha re-runs the same classify_prd on the same predicted keypoints.
    """
    def __init__(self, primary_map, extra_map):
        super().__init__(primary_map)
        self.extra_map = {b: list(a) for b, a in (extra_map or {}).items()}
        self.extra = {b: {a: EvaluatorInstance(b, a) for a in alphas}
                      for b, alphas in self.extra_map.items()}
        self.reset()

    def reset(self):
        self.acc = {b: {a: [] for a in alphas} for b, alphas in self.extra_map.items()}

    def evaluate(self, benchmark, prd_kps, batch):
        if benchmark in self.extra:
            for a, ev in self.extra[benchmark].items():
                self.acc[benchmark][a] += ev.evaluate(prd_kps, batch)['pck']
        return self.evaluators[benchmark].evaluate(prd_kps, batch)

    def means(self):
        return {b: {a: (sum(v) / len(v) if v else 0.0) for a, v in d.items()}
                for b, d in self.acc.items()}


def _tss_name(alpha):   # 0.10 -> tss_a10, 0.05 -> tss_a05, 0.03 -> tss_a03
    return f"tss_a{int(round(alpha * 100)):02d}"


def build_eval_config(cfg, families, kitti_alphas, tss_alphas, tapvid_strides, tapvid_alpha):
    """Assemble eval_benchmarks / eval_alphas / val_datasets + extra-alpha map."""
    benchmarks, alphas, val_datasets = [], [], {}
    extra_map = {}

    if 'kitti' in families:
        primary_k = kitti_alphas[0]
        for b in ('kitti2012', 'kitti2015'):
            benchmarks.append(b); alphas.append(primary_k)
            val_datasets[b] = {'split': 'val', 'normalize_images': True}
            if len(kitti_alphas) > 1:
                extra_map[b] = list(kitti_alphas[1:])

    if 'tss' in families:
        for a in tss_alphas:
            name = _tss_name(a)
            benchmarks.append(name); alphas.append(a)
            val_datasets[name] = {'reverse_flow': False, 'normalize_images': True}

    if 'tapvid' in families:
        for s in tapvid_strides:
            name = f'tapvid_davis_s{s}'
            benchmarks.append(name); alphas.append(tapvid_alpha)
            val_datasets[name] = {'tapvid_stride': s, 'tapvid_frame_step': 5,
                                  'tapvid_min_pts': 1, 'reverse_flow': True,
                                  'normalize_images': True}

    ev = cfg['evaluation']
    for k, v in LOCAL_ROOTS.items():
        ev[k] = v
    ev['kitti_val_use_full_training'] = True
    ev['use_motion_aware'] = False
    ev['eval_benchmarks'] = benchmarks
    ev['eval_alphas'] = alphas
    ev['val_datasets'] = val_datasets
    return benchmarks, alphas, extra_map


def load_state_dict_into(model, path):
    obj = torch.load(path, map_location='cpu')
    sd = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    return model.load_state_dict(sd, strict=False)


def metric_key(benchmark, alpha):
    """Human-readable metric label per (benchmark, alpha)."""
    if benchmark.startswith('tapvid_davis_s'):
        return f"{benchmark}@{alpha}"          # stride encoded in name
    return f"{benchmark}@{alpha}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--families', nargs='+', default=['kitti', 'tss', 'tapvid'],
                    choices=['kitti', 'tss', 'tapvid'])
    ap.add_argument('--kitti-alphas', type=float, nargs='+', default=[0.05, 0.03, 0.01])
    ap.add_argument('--tss-alphas', type=float, nargs='+', default=[0.10, 0.05, 0.03])
    ap.add_argument('--tapvid-strides', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    ap.add_argument('--tapvid-alpha', type=float, default=0.05)
    ap.add_argument('--epochs', type=int, nargs='+', default=None)
    ap.add_argument('--ckpt', default=None, help='single checkpoint file (e.g. model_best.pth)')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(args.cell, 'config.yaml')))
    benchmarks, alphas, extra_map = build_eval_config(
        cfg, args.families, args.kitti_alphas, args.tss_alphas,
        args.tapvid_strides, args.tapvid_alpha)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  cell={args.cell}", flush=True)
    print(f"benchmarks={benchmarks}", flush=True)
    print(f"alphas={alphas}  extra_map={extra_map}", flush=True)

    primary_map = dict(zip(benchmarks, alphas))
    multi_evaluator = MultiAlphaEvaluator(primary_map, extra_map)
    val_datasets, val_loaders = create_validation_datasets(cfg, device=device)
    for b, dl in val_loaders.items():
        print(f"  val loader {b}: {len(dl)} batches", flush=True)
    model = create_model(cfg['model'], cfg['paths'])
    model.to(device).eval()

    # ---- checkpoints ----
    if args.ckpt:
        ckpts = [(args.ckpt, os.path.join(args.cell, args.ckpt))]
    else:
        ep_paths = glob.glob(os.path.join(args.cell, 'epoch_*.pth'))
        if ep_paths:
            all_eps = sorted(int(re.search(r'epoch_(\d+)\.pth', p).group(1)) for p in ep_paths)
            sel = args.epochs if args.epochs else all_eps
            ckpts = [(f'ep{e}', os.path.join(args.cell, f'epoch_{e}.pth')) for e in sel]
        else:
            mb = os.path.join(args.cell, 'model_best.pth')
            if not os.path.exists(mb):
                raise SystemExit(f"No epoch_*.pth and no model_best.pth in {args.cell}")
            ckpts = [('model_best', mb)]
    print(f"scoring {len(ckpts)} checkpoint(s)", flush=True)

    # (benchmark, alpha) -> (pck, ckpt_tag)
    pairs = [(b, a) for b in benchmarks for a in [primary_map[b]] + extra_map.get(b, [])]
    peak = {p: (-1.0, None) for p in pairs}
    rows = []
    for tag, path in ckpts:
        load_state_dict_into(model, path)
        model.eval()
        multi_evaluator.reset()
        with torch.no_grad():
            res = validate_epoch_multi_benchmark(
                net=model, val_loaders=val_loaders, device=device, epoch=-1,
                multi_evaluator=multi_evaluator, primary_benchmark=benchmarks[0],
                use_motion_aware=False, mmd_every_n_epochs=0)
        extra_means = multi_evaluator.means()
        line = []
        for b in benchmarks:
            vals = {primary_map[b]: float(res[b]['pck'])}
            for a in extra_map.get(b, []):
                vals[a] = float(extra_means[b][a])
            for a, p in vals.items():
                rows.append((tag, b, a, round(p, 4)))
                if p > peak[(b, a)][0]:
                    peak[(b, a)] = (p, tag)
                line.append(f"{metric_key(b, a)}={p:.2f}")
        print(f"{tag}: " + "  ".join(line), flush=True)

    print("\n==== PEAK PCK (max over scored checkpoints) ====")
    for (b, a) in pairs:
        pk, tag = peak[(b, a)]
        print(f"  {metric_key(b, a):<22}: {pk:6.2f}%  ({tag})")

    out = args.out or os.path.join(args.cell, 'transfer_cell_eval.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['ckpt', 'benchmark', 'alpha', 'pck'])
        w.writerows(rows)
    print("\nwrote", out)


if __name__ == '__main__':
    main()
