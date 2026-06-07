#!/usr/bin/env python3
"""Pre-render the synthetic validation set ONCE to disk (chunked, crash-proof).

Why this exists
---------------
The `synthetic` eval benchmark is the only one rendered live (online moderngl/SDF
shaders) instead of loaded from disk. In the GLU-Net grid that caused:
  1. A native renderer SIGABRT ("Aborted, core dumped") around the ~408th render
     in a single process (batch 50/125) that killed validation mid-run.
  2. ~52 min per validation pass (the dominant validation cost), so jobs never
     reached their step budget.
  3. A drifting, non-reproducible val set: the renderer RNG is seeded once and
     never reset, and there is unseeded global randomness, so the "synthetic val
     set" differs across epochs AND across runs.

This renders a FIXED set once and writes per-sample tensors that
`CachedSyntheticAdapter` (src/data/synth/adapters.py) serves at validation time:
renderer-free, fast disk I/O, identical across every job and architecture.

How it stays crash-proof
------------------------
Whatever the abort's root cause (a bad sample near #408 or a GL resource leak
that accumulates), it only bites PAST ~408 renders in ONE process. So we render
in CHUNKS, each in a fresh subprocess (clean GL context -> leak resets) of
`--chunk-size` (default 200, safely below ~408). Each chunk gets its own seed, so
the chunks are distinct draws from the same distribution. A chunk that somehow
dies is retried; partial progress is always kept (samples are written
immediately and the driver resumes from the on-disk count).

Cache point
-----------
We save the output of `CorrespondenceDataset._process_synthetic_batch` -- textured
`src_img`/`trg_img` and GT `flow_full`, BEFORE resize/keypoints/normalize. The
cached adapter returns these as a CommonSample and the unchanged collate pipeline
applies resize -> kps_from_flow -> downsample -> normalize identically.

Usage (on the RC cluster)
-------------------------
    python scripts/precompute_synthetic_val.py \
        --out /home/slow1/Data/synthetic_val_cache_v1 \
        --num-samples 1000 \
        --chunk-size 200

Re-running resumes from whatever is already on disk. Then the eval config key
`evaluation.val_datasets.synthetic.synthetic_val_cache` points at --out (already
set to /home/slow1/Data/synthetic_val_cache_v1 in eval_base_rc_fastsynth.yaml).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/precompute_synthetic_val.py`
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _identity_collate(batch):
    """Return the raw list of [src_dict, trg_dict] items untouched."""
    return batch


def _count_cached(out_dir: Path) -> int:
    return len(list(out_dir.glob("sample_*.pt")))


def render_chunk(args) -> int:
    """Worker: render `args.worker_count` samples (seed=args.worker_seed) into
    args.out, naming them after the current on-disk count. Returns #written."""
    import torch
    from torch.utils.data import DataLoader
    from src.data.synth.datasets.CorrespondenceDataset import CorrespondenceDataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dtype = torch.float16 if args.img_dtype == "float16" else torch.float32
    offset = _count_cached(out_dir)
    target = args.worker_count

    dataset = CorrespondenceDataset(
        "synthetic",
        split="val",
        size=(args.size, args.size),
        downsample_flow=None,
        max_kps=None,
        normalize_images=False,
        geometry_config_path=args.geometry_config,
        processor_config_path=args.processor_config,
        opengl_device_index=None,
        geometry_config_overrides={"seed": int(args.worker_seed)},
        verbose=False,
        debug=False,
    )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=_identity_collate, drop_last=False,
    )

    print(f"[worker] seed={args.worker_seed} rendering {target} samples "
          f"(offset {offset}) -> {out_dir}", flush=True)
    written = 0
    shapes_logged = False
    for raw_batch in loader:
        if written >= target:
            break
        samples = dataset._process_synthetic_batch(raw_batch)
        for s in samples:
            if written >= target:
                break
            if s.src_img is None or s.trg_img is None or s.flow_full is None:
                raise RuntimeError(f"missing field at sample {offset + written}")
            record = {
                "src_img": s.src_img.detach().to("cpu", img_dtype).contiguous(),
                "trg_img": s.trg_img.detach().to("cpu", img_dtype).contiguous(),
                "flow_full": s.flow_full.detach().to("cpu", torch.float32).contiguous(),
            }
            if not shapes_logged:
                print(f"[worker] shapes src_img={tuple(record['src_img'].shape)} "
                      f"flow_full={tuple(record['flow_full'].shape)}", flush=True)
                shapes_logged = True
            # tmp-then-rename so a crash mid-write never leaves a truncated file
            idx = offset + written
            tmp = out_dir / f".sample_{idx:05d}.pt.tmp"
            torch.save(record, tmp)
            tmp.rename(out_dir / f"sample_{idx:05d}.pt")
            written += 1
        print(f"[worker] {written}/{target}", end="\r", flush=True)
    print(f"\n[worker] done: wrote {written} samples", flush=True)
    return written


def drive(args) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = args.num_samples
    chunk = args.chunk_size if args.chunk_size > 0 else target

    print(f"[driver] target={target} chunk_size={chunk} -> {out_dir}")
    print(f"[driver] geometry={args.geometry_config}")
    consecutive_failures = 0
    max_failures = 5
    while True:
        existing = _count_cached(out_dir)
        if existing >= target:
            break
        this_count = min(chunk, target - existing)
        chunk_idx = existing // chunk  # distinct seed per chunk position
        worker_seed = int(args.base_seed) + chunk_idx
        cmd = [
            sys.executable, os.path.abspath(__file__), "--worker",
            "--out", str(out_dir),
            "--worker-count", str(this_count),
            "--worker-seed", str(worker_seed),
            "--geometry-config", args.geometry_config,
            "--processor-config", args.processor_config,
            "--batch-size", str(args.batch_size),
            "--size", str(args.size),
            "--img-dtype", args.img_dtype,
        ]
        print(f"[driver] {existing}/{target} -> launching chunk "
              f"(seed={worker_seed}, count={this_count})")
        rc = subprocess.run(cmd).returncode
        after = _count_cached(out_dir)
        if after > existing:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            print(f"[driver] WARNING: chunk made no progress (rc={rc}); "
                  f"failure {consecutive_failures}/{max_failures}")
            if consecutive_failures >= max_failures:
                print(f"[driver] ABORTING after {max_failures} stalled chunks. "
                      f"Cached {after}/{target}.")
                break

    final = _count_cached(out_dir)
    manifest = {
        "num_samples": final,
        "geometry_config": args.geometry_config,
        "processor_config": args.processor_config,
        "render_size": args.size,
        "img_dtype": args.img_dtype,
        "chunk_size": chunk,
        "base_seed": args.base_seed,
        "fields": ["src_img", "trg_img", "flow_full"],
        "note": ("Fixed cached synthetic val set. Chunks rendered in fresh "
                 "subprocesses (each < renderer-abort threshold) with per-chunk "
                 "seeds; the online renderer is non-deterministic so exact sample "
                 "identity is not reproducible, but this on-disk set is now frozen."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[driver] DONE. {final}/{target} samples cached + manifest.json -> {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Output cache directory")
    parser.add_argument("--num-samples", type=int, default=1000, help="Total samples to cache")
    parser.add_argument("--chunk-size", type=int, default=200,
                        help="Samples per fresh subprocess (keep < ~400 to avoid the renderer abort)")
    parser.add_argument("--base-seed", type=int, default=987654321,
                        help="Per-chunk seed = base-seed + chunk_index")
    parser.add_argument("--geometry-config", default="src/configs/online_synth_configs/OnlineGeometryConfig_Val.yaml")
    parser.add_argument("--processor-config", default="src/configs/online_synth_configs/OnlineProcessorConfig.yaml")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--img-dtype", choices=["float16", "float32"], default="float16")
    # internal worker mode
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, default=987654321, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        render_chunk(args)
    else:
        drive(args)


if __name__ == "__main__":
    main()
