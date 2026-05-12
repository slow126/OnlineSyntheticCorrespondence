"""
Extract MOVi-F TFDS records to a file tree for training.

Works on a PARTIAL download — reads whatever TFRecord shards are present locally
without requiring dataset_info.json or all 1024 shards.

SETUP:

  Step 1: Download some or all shards from GCS (requires gcloud CLI + auth):

      KUBRIC_TFDS_TARGET=/home/spencer/Data/kubric_tfds \\
          bash /home/spencer/Projects/kubric/interface/download_movi_f_tfds.sh

    Full train split is ~500 GB (1024 shards). You can download a subset by
    cancelling early — any complete shards (no .gstmp suffix) are usable.

  Step 2: Install TF dependencies (separate env if needed):
      pip install tensorflow tensorflow-datasets etils imageio pypng

USAGE:
  python scripts/extract_movi_f.py \\
      --kubric_dir /home/spencer/Projects/kubric \\
      --shard_dir /home/spencer/Data/kubric_tfds/movi_f/512x512/1.0.0 \\
      --output_dir /home/spencer/Data/movi_f \\
      --split train

OUTPUT tree:
  <output_dir>/<split>/<video_id>/
      frame_{t:04d}.png          # uint8 RGB image at frame t
      bflow_{t:04d}.npy          # float32 [H,W,2] backward flow AT t (dy,dx Kubric order)
      fflow_{t:04d}.npy          # float32 [H,W,2] forward  flow AT t (dy,dx Kubric order)
  <output_dir>/<split>/pairs_index.json   # [[video_id, t], ...] for all complete pairs

Flow arrays are stored in Kubric's raw [H,W,2] [dy,dx] convention.
MoviFDataset converts to the project's [2,H,W] [dx,dy] convention on load.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

# Keep TF off the GPU — extraction is I/O + CPU decode, not compute bound.
# Set before any TF import so it takes effect.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def _load_movi_builder(kubric_dir: str, config: str = "512x512"):
    """
    Load the MoviF builder class from the kubric repo and return an instance.
    Uses importlib so we don't need __init__.py in challenges/.
    """
    builder_path = Path(kubric_dir).resolve() / "challenges" / "movi" / "movi_f.py"
    if not builder_path.exists():
        print(f"ERROR: movi_f builder not found at {builder_path}")
        print("  Make sure --kubric_dir points to the kubric repository root.")
        sys.exit(1)

    kubric_root = str(Path(kubric_dir).resolve())
    if kubric_root not in sys.path:
        sys.path.insert(0, kubric_root)

    spec = importlib.util.spec_from_file_location("movi_f_builder", builder_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"ERROR: Failed to load movi_f builder: {e}")
        print("  Required packages: tensorflow tensorflow-datasets etils imageio pypng")
        sys.exit(1)

    # Instantiate with a dummy data_dir — we only need builder.info.features
    # for deserialization; no file I/O happens here.
    builder = mod.MoviF(config=config, data_dir="/tmp/dummy_movi_f_builder")
    print(f"Loaded MoviF builder (config={config}) from {builder_path}")
    return builder


def _list_shards(shard_dir: Path, split: str) -> list[Path]:
    """Return all complete (non-.gstmp) TFRecord shards for the given split."""
    pattern = f"movi_f-{split}.tfrecord-*"
    shards = sorted(
        p for p in shard_dir.glob(pattern)
        if not p.name.endswith(".gstmp")
    )
    return shards


def _decode_flow(flow_uint16: np.ndarray, flow_range: np.ndarray) -> np.ndarray:
    flow = flow_uint16.astype(np.float32)
    min_val, max_val = float(flow_range[0]), float(flow_range[1])
    return flow / 65535.0 * (max_val - min_val) + min_val


def _video_id(example: dict, idx: int) -> str:
    name = example.get("metadata", {}).get("video_name", None)
    if name is None:
        return f"video_{idx:06d}"
    if isinstance(name, np.ndarray):
        name = name.item()
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return str(name)


def extract(
    builder,
    shard_dir: Path,
    output_dir: Path,
    split: str,
) -> None:
    import tensorflow as tf

    shards = _list_shards(shard_dir, split)
    if not shards:
        print(f"ERROR: No complete shards found in {shard_dir} for split='{split}'.")
        print(f"  Expected files matching: movi_f-{split}.tfrecord-* (without .gstmp)")
        sys.exit(1)

    print(f"Found {len(shards)} complete shard(s) in {shard_dir}")

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Use builder.info.features.deserialize_example to decode raw TFRecord bytes.
    # This is the proper TFDS deserialization path and handles all the complex
    # Sequence/Tensor feature types without needing dataset_info.json.
    features = builder.info.features
    raw_ds = tf.data.TFRecordDataset([str(p) for p in shards])
    decoded_ds = raw_ds.map(
        features.deserialize_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    pairs_index = []
    iterator = enumerate(decoded_ds.as_numpy_iterator())
    if HAS_TQDM:
        iterator = tqdm(iterator, desc=f"Extracting {split}", unit="video",
                        total=len(shards))  # rough estimate; ~6 videos/shard

    for video_idx, example in iterator:
        video = np.asarray(example["video"], dtype=np.uint8)   # [T, H, W, 3]
        meta  = example["metadata"]
        num_frames = video.shape[0]
        if num_frames < 2:
            continue

        vid_id  = _video_id(example, video_idx)
        vid_dir = split_dir / vid_id
        vid_dir.mkdir(exist_ok=True)

        fflow_range = np.asarray(meta["forward_flow_range"])
        bflow_range = np.asarray(meta["backward_flow_range"])

        for t in range(num_frames):
            img_path = vid_dir / f"frame_{t:04d}.png"
            if not img_path.exists():
                Image.fromarray(video[t]).save(img_path)

            # Forward flow at t: frame[t] -> frame[t+1]  (valid for t in [0, T-2])
            if t < num_frames - 1:
                fflow = _decode_flow(
                    np.asarray(example["forward_flow"][t]),
                    fflow_range,
                )
                np.save(str(vid_dir / f"fflow_{t:04d}.npy"), fflow.astype(np.float32))

            # Backward flow at t: frame[t] -> frame[t-1] (valid for t in [1, T-1])
            if t > 0:
                bflow = _decode_flow(
                    np.asarray(example["backward_flow"][t]),
                    bflow_range,
                )
                np.save(str(vid_dir / f"bflow_{t:04d}.npy"), bflow.astype(np.float32))

        for t in range(num_frames - 1):
            pairs_index.append([vid_id, t])

    index_path = split_dir / "pairs_index.json"
    with open(index_path, "w") as f:
        json.dump(pairs_index, f)

    print(f"Done. {len(pairs_index)} pairs from {video_idx + 1} videos.")
    print(f"Pairs index: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MOVi-F TFRecord shards to a file tree (partial download OK).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kubric_dir", type=str, default="/home/spencer/Projects/kubric",
        help="Path to the kubric repository root. Default: /home/spencer/Projects/kubric",
    )
    parser.add_argument(
        "--shard_dir", type=str, required=True,
        help="Directory containing the .tfrecord shard files "
             "(e.g. /home/spencer/Data/kubric_tfds/movi_f/512x512/1.0.0).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Root output directory (e.g. /home/spencer/Data/movi_f).",
    )
    parser.add_argument(
        "--config", type=str, default="512x512",
        choices=["512x512", "256x256", "128x128"],
        help="MOVi-F resolution config. Must match the shards you downloaded.",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        choices=["train", "validation", "test"],
        help="Split to extract (must match shard filenames).",
    )
    args = parser.parse_args()

    builder = _load_movi_builder(args.kubric_dir, config=args.config)

    extract(
        builder=builder,
        shard_dir=Path(args.shard_dir),
        output_dir=Path(args.output_dir),
        split=args.split,
    )


if __name__ == "__main__":
    main()
