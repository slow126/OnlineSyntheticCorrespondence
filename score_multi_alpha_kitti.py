#!/usr/bin/env python3
"""Re-score a transfer cell on KITTI-2012 / KITTI-2015 at PCK alpha {0.05, 0.03, 0.01}.

Why: KITTI@0.05 is saturated under a frozen pretrained backbone (~96-98 for any decent
source), so the tuned-vs-default SDF-fractal deltas are tiny/mixed. Re-scoring the SAME
checkpoints at stricter alpha opens headroom without retraining.

Fidelity: reuses the EXACT training-time eval (validate_epoch_multi_benchmark + the SAME
flow2kps + classify_prd). A single inference pass per epoch produces all three alphas:
the PRIMARY alpha (0.05) flows through the unmodified harness so the reported pck still
reproduces validation_results.csv (sanity check); the EXTRA alphas (0.03, 0.01) are scored
on the very same predicted keypoints via side EvaluatorInstance objects. No re-inference.

Architecture-general: works for cats / glunet / flowformer (all expose
forward(trg_img, src_img) -> flow). RAFT is out of scope (no fractal-trial76 source).

Usage:
  CUDA_VISIBLE_DEVICES=1 python score_multi_alpha_kitti.py --cell <DIR>            # all epoch_*.pth
  CUDA_VISIBLE_DEVICES=1 python score_multi_alpha_kitti.py --cell <DIR> --epochs 26 30
  CUDA_VISIBLE_DEVICES=1 python score_multi_alpha_kitti.py --cell <DIR> --ckpt model_best.pth
"""
import sys, os, re, csv, glob, yaml, argparse
import torch

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
from train_lightning import create_model
from train_cats_unified import create_validation_datasets
from models.CATs_PlusPlus.utils_training.eval_instance import MultiBenchmarkEvaluator, EvaluatorInstance
from models.CATs_PlusPlus.utils_training.optimize_multi import validate_epoch_multi_benchmark


class MultiAlphaEvaluator(MultiBenchmarkEvaluator):
    """MultiBenchmarkEvaluator that also scores EXTRA alphas on the same predictions.

    The PRIMARY alpha per benchmark behaves exactly like the parent (so the value
    validate_epoch_multi_benchmark returns/prints is unchanged). For every batch we ALSO
    run the same classify_prd at each extra alpha and accumulate per-sample PCK, so after
    one validate() pass we can read mean PCK at all alphas with zero extra inference.
    """
    def __init__(self, primary_map, extra_alphas):
        super().__init__(primary_map)
        self.extra_alphas = list(extra_alphas)
        self.extra = {b: {a: EvaluatorInstance(b, a) for a in self.extra_alphas}
                      for b in primary_map}
        self.reset()

    def reset(self):
        self.acc = {b: {a: [] for a in self.extra_alphas} for b in self.extra}

    def evaluate(self, benchmark, prd_kps, batch):
        for a, ev in self.extra[benchmark].items():
            self.acc[benchmark][a] += ev.evaluate(prd_kps, batch)['pck']
        return self.evaluators[benchmark].evaluate(prd_kps, batch)

    def means(self):
        return {b: {a: (sum(v) / len(v) if v else 0.0) for a, v in d.items()}
                for b, d in self.acc.items()}


def load_state_dict_into(model, path):
    """Load a checkpoint (epoch_*.pth / model_best.pth / lightning .ckpt) into model."""
    obj = torch.load(path, map_location='cpu')
    sd = obj['state_dict'] if isinstance(obj, dict) and 'state_dict' in obj else obj
    miss, unexp = model.load_state_dict(sd, strict=False)
    return miss, unexp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, help='checkpoint dir with config.yaml')
    ap.add_argument('--benchmarks', nargs='+', default=['kitti2012', 'kitti2015'])
    ap.add_argument('--primary-alpha', type=float, default=0.05)
    ap.add_argument('--extra-alphas', type=float, nargs='+', default=[0.03, 0.01])
    ap.add_argument('--epochs', type=int, nargs='+', default=None,
                    help='specific epoch numbers; default = all epoch_*.pth')
    ap.add_argument('--ckpt', default=None,
                    help='explicit checkpoint file (e.g. model_best.pth) instead of epoch series')
    ap.add_argument('--kitti-root', default='/home/spencer/Data/correspondence/kitti')
    ap.add_argument('--out', default=None, help='output CSV (default: <cell>/multi_alpha_kitti.csv)')
    args = ap.parse_args()

    BENCH = args.benchmarks
    PRIMARY = args.primary_alpha
    EXTRA = args.extra_alphas
    ALL_ALPHAS = [PRIMARY] + EXTRA

    cfg = yaml.safe_load(open(os.path.join(args.cell, 'config.yaml')))
    cfg['evaluation']['eval_benchmarks'] = BENCH
    cfg['evaluation']['eval_alphas'] = [PRIMARY] * len(BENCH)
    cfg['evaluation']['kitti_root'] = args.kitti_root   # override any RC path baked into config
    ev = cfg['evaluation']
    use_ma = ev.get('use_motion_aware', False)
    min_mp = ev.get('min_motion_pixels', 5.0)
    zthr = ev.get('zero_threshold', 0.5)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device={device}  cell={args.cell}", flush=True)
    print(f"benchmarks={BENCH}  primary_alpha={PRIMARY}  extra_alphas={EXTRA}", flush=True)

    multi_evaluator = MultiAlphaEvaluator({b: PRIMARY for b in BENCH}, EXTRA)
    val_datasets, val_loaders = create_validation_datasets(cfg, device=device)
    for b, dl in val_loaders.items():
        print(f"  val loader {b}: {len(dl)} batches", flush=True)
    model = create_model(cfg['model'], cfg['paths'])
    model.to(device).eval()

    # ---- determine checkpoints ----
    if args.ckpt:
        ckpts = [(None, os.path.join(args.cell, args.ckpt))]
    else:
        ep_paths = glob.glob(os.path.join(args.cell, 'epoch_*.pth'))
        if ep_paths:
            all_eps = sorted(int(re.search(r'epoch_(\d+)\.pth', p).group(1)) for p in ep_paths)
            sel = args.epochs if args.epochs else all_eps
            ckpts = [(e, os.path.join(args.cell, f'epoch_{e}.pth')) for e in sel]
        else:
            # fall back to model_best.pth
            mb = os.path.join(args.cell, 'model_best.pth')
            if not os.path.exists(mb):
                raise SystemExit(f"No epoch_*.pth and no model_best.pth in {args.cell}")
            ckpts = [(None, mb)]
    print(f"scoring {len(ckpts)} checkpoint(s)", flush=True)

    # peak[(benchmark, alpha)] = (pck, epoch)
    peak = {(b, a): (-1.0, None) for b in BENCH for a in ALL_ALPHAS}
    rows = []
    for ep, path in ckpts:
        miss, unexp = load_state_dict_into(model, path)
        if ep == ckpts[0][0] and (miss or unexp):
            print(f"  [load] missing={len(miss)} unexpected={len(unexp)} (first miss: {miss[:2]})", flush=True)
        model.eval()
        multi_evaluator.reset()
        with torch.no_grad():
            res = validate_epoch_multi_benchmark(
                net=model, val_loaders=val_loaders, device=device,
                epoch=(ep if ep is not None else -1),
                multi_evaluator=multi_evaluator, primary_benchmark=BENCH[0],
                use_motion_aware=use_ma, min_motion_pixels=min_mp,
                zero_threshold=zthr, mmd_every_n_epochs=0)
        extra_means = multi_evaluator.means()
        tag = f"ep{ep}" if ep is not None else os.path.basename(path)
        line = []
        for b in BENCH:
            pck_by_a = {PRIMARY: float(res[b]['pck'])}
            for a in EXTRA:
                pck_by_a[a] = float(extra_means[b][a])
            for a in ALL_ALPHAS:
                p = pck_by_a[a]
                rows.append((tag, b, a, round(p, 4)))
                if p > peak[(b, a)][0]:
                    peak[(b, a)] = (p, tag)
                line.append(f"{b}@{a}={p:.2f}")
        print(f"{tag}: " + "  ".join(line), flush=True)

    print("\n==== PEAK PCK (max over scored checkpoints) ====")
    for b in BENCH:
        for a in ALL_ALPHAS:
            pk, tag = peak[(b, a)]
            print(f"  {b:<10} @{a:<5}: {pk:6.2f}%  ({tag})")

    out = args.out or os.path.join(args.cell, 'multi_alpha_kitti.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['ckpt', 'benchmark', 'alpha', 'pck'])
        w.writerows(rows)
    print("\nwrote", out)


if __name__ == '__main__':
    main()
