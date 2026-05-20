#!/usr/bin/env python3
"""Build and merge symmetric self-distances for IDW neighborhoods.

This is an additive helper for transfer_analysis_v3. It does not delete any
existing experiment results. By default it:

1. Reads the active pair list from analysis_v3/pairwise_self_distances.csv.
2. Computes flow FID and flow sliced-W2 for train_train/eval_eval pairs.
3. Merges any matching flow MMD values from flow_mmd_results_fast.csv.
4. Writes analysis_v3/pairwise_symmetric_distances.csv.
5. Backs up pairwise_self_distances.csv.
6. Adds columns flow_fid_self, flow_sliced_w2_self, flow_mmd_self to
   pairwise_self_distances.csv.

The script is resumable: already-computed rows in --output are reused.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from compute_symmetric_distances import (
    compute_all_fid_stats,
    fid_from_stats,
    load_random_n,
    sliced_w2,
    _vec_path,
)


DEFAULT_PAIR_TYPES = ["train_train", "eval_eval"]


def _pair_key(row: pd.Series) -> tuple[str, str, str, str, str, str]:
    return (
        str(row["space"]),
        str(row["pair_type"]),
        str(row["dataset_a"]),
        str(row["split_a"]),
        str(row["dataset_b"]),
        str(row["split_b"]),
    )


def _undirected_key(a: str, sa: str, b: str, sb: str) -> tuple[tuple[str, str], tuple[str, str]]:
    ka = (str(a), str(sa))
    kb = (str(b), str(sb))
    return tuple(sorted((ka, kb)))  # type: ignore[return-value]


def _load_mmd_lookup(path: Path) -> dict[tuple[tuple[str, str], tuple[str, str]], float]:
    if not path.exists():
        print(f"  NOTE: MMD CSV not found: {path}")
        return {}
    df = pd.read_csv(path)
    required = {"dataset1", "split1", "dataset2", "split2", "mmd"}
    if not required.issubset(df.columns):
        print(f"  NOTE: MMD CSV missing required columns {sorted(required)}: {path}")
        return {}
    lookup: dict[tuple[tuple[str, str], tuple[str, str]], list[float]] = {}
    for _, row in df.iterrows():
        val = row.get("mmd")
        if pd.isna(val):
            continue
        key = _undirected_key(row["dataset1"], row["split1"], row["dataset2"], row["split2"])
        lookup.setdefault(key, []).append(float(val))
    return {k: float(np.mean(v)) for k, v in lookup.items()}


def _existing_lookup(path: Path) -> dict[tuple[str, str, str, str, str, str], dict]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    required = {"space", "pair_type", "dataset_a", "split_a", "dataset_b", "split_b"}
    if not required.issubset(df.columns):
        return {}
    out = {}
    for _, row in df.iterrows():
        out[_pair_key(row)] = row.to_dict()
    return out


def _select_pairs(
    self_df: pd.DataFrame,
    spaces: set[str],
    pair_types: set[str],
) -> pd.DataFrame:
    pairs = self_df[
        self_df["space"].isin(spaces)
        & self_df["pair_type"].isin(pair_types)
    ].copy()
    cols = ["space", "pair_type", "dataset_a", "split_a", "dataset_b", "split_b"]
    pairs = pairs[cols].drop_duplicates().reset_index(drop=True)
    return pairs


def compute_symmetric_rows(
    pairs: pd.DataFrame,
    vec_dir: Path,
    existing: dict[tuple[str, str, str, str, str, str], dict],
    mmd_lookup: dict[tuple[tuple[str, str], tuple[str, str]], float],
    fid_samples: int,
    sw_samples: int,
    n_proj: int,
    seed: int,
    output: Path,
    flush_every: int,
) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()

    # This helper currently computes flow-space symmetric distances only. DINO can
    # be added later using the same math, but the current paper runs are flow-only.
    flow_pairs = pairs[pairs["space"] == "flow"].copy()
    if flow_pairs.empty:
        print("No flow pairs selected; nothing to compute.")
        return pd.DataFrame()

    unique_keys = sorted({
        (row["dataset_a"], row["split_a"])
        for _, row in flow_pairs.iterrows()
    } | {
        (row["dataset_b"], row["split_b"])
        for _, row in flow_pairs.iterrows()
    })

    print(f"Selected flow pairs: {len(flow_pairs)}")
    print(f"Unique dataset/split keys: {len(unique_keys)}")

    print(f"\n--- Computing flow FID stats ({len(unique_keys)} datasets) ---")
    fid_stats = compute_all_fid_stats(
        unique_keys, vec_dir, "flow", fid_samples, desc="flow symmetric cov"
    )

    rng = np.random.default_rng(seed)
    flow_projs = rng.standard_normal((n_proj, 4)).astype(np.float32)
    flow_projs /= np.linalg.norm(flow_projs, axis=1, keepdims=True)

    rows: list[dict] = []
    computed = 0
    skipped = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    bar = tqdm(flow_pairs.iterrows(), total=len(flow_pairs), desc="symmetric pairs",
               unit="pair", dynamic_ncols=True)
    for _, pair in bar:
        key = _pair_key(pair)
        if key in existing:
            skipped += 1
            continue

        da, sa, db, sb = (
            pair["dataset_a"], pair["split_a"],
            pair["dataset_b"], pair["split_b"],
        )
        ka, kb = f"{da}_{sa}", f"{db}_{sb}"
        row = {
            "space": "flow",
            "pair_type": pair["pair_type"],
            "dataset_a": da,
            "split_a": sa,
            "dataset_b": db,
            "split_b": sb,
        }

        if ka in fid_stats and kb in fid_stats:
            row["flow_fid_self"] = fid_from_stats(fid_stats[ka], fid_stats[kb])
        else:
            row["flow_fid_self"] = float("nan")

        pa = _vec_path(vec_dir, da, sa, "flow")
        pb = _vec_path(vec_dir, db, sb, "flow")
        va = load_random_n(pa, sw_samples, seed=int(hash(pa.stem) % (2**31)))
        vb = load_random_n(pb, sw_samples, seed=int(hash(pb.stem) % (2**31)))
        if va is not None and vb is not None:
            row["flow_sliced_w2_self"] = sliced_w2(va, vb, flow_projs)
        else:
            row["flow_sliced_w2_self"] = float("nan")
        del va, vb

        mmd = mmd_lookup.get(_undirected_key(da, sa, db, sb))
        row["flow_mmd_self"] = float(mmd) if mmd is not None else float("nan")

        rows.append(row)
        computed += 1
        if len(rows) >= flush_every:
            batch = pd.DataFrame(rows)
            write_header = not output.exists() or output.stat().st_size == 0
            batch.to_csv(output, mode="a", header=write_header, index=False)
            rows = []
        bar.set_postfix(computed=computed, skipped=skipped, refresh=False)

    if rows:
        batch = pd.DataFrame(rows)
        write_header = not output.exists() or output.stat().st_size == 0
        batch.to_csv(output, mode="a", header=write_header, index=False)

    final = pd.read_csv(output) if output.exists() else pd.DataFrame()
    print(f"\n✓ Wrote {len(final)} rows to {output} in {time.time() - t0:.0f}s")
    for col in ["flow_fid_self", "flow_sliced_w2_self", "flow_mmd_self"]:
        if col in final.columns:
            print(f"  {col}: {final[col].notna().sum()}/{len(final)} valid")
    return final


def merge_into_self_dist(self_path: Path, sym_path: Path, backup_path: Path | None) -> None:
    self_df = pd.read_csv(self_path)
    sym_df = pd.read_csv(sym_path)
    if sym_df.empty:
        print("No symmetric rows to merge.")
        return

    key_cols = ["space", "pair_type", "dataset_a", "split_a", "dataset_b", "split_b"]
    metric_cols = ["flow_fid_self", "flow_sliced_w2_self", "flow_mmd_self"]
    present_metrics = [c for c in metric_cols if c in sym_df.columns]
    if not present_metrics:
        print("No symmetric metric columns found to merge.")
        return

    if backup_path is not None:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self_path, backup_path)
        print(f"Backed up {self_path} -> {backup_path}")

    sym_slim = sym_df[key_cols + present_metrics].drop_duplicates(key_cols, keep="last")
    for col in present_metrics:
        if col in self_df.columns:
            self_df = self_df.drop(columns=[col])
    merged = self_df.merge(sym_slim, on=key_cols, how="left")
    merged.to_csv(self_path, index=False)
    print(f"✓ Merged {present_metrics} into {self_path}")
    for col in present_metrics:
        print(f"  {col}: {merged[col].notna().sum()}/{len(merged)} rows populated")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-dist", default="analysis_v3/pairwise_self_distances.csv")
    parser.add_argument("--output", default="analysis_v3/pairwise_symmetric_distances.csv")
    parser.add_argument("--vec-dir", default="/mnt/nvme_1tb_b/coverage_vectors")
    parser.add_argument("--flow-mmd-csv", default="flow_mmd_results_fast.csv")
    parser.add_argument("--spaces", nargs="+", default=["flow"], choices=["flow"])
    parser.add_argument("--pair-types", nargs="+", default=DEFAULT_PAIR_TYPES,
                        choices=["train_train", "eval_eval", "train_eval"])
    parser.add_argument("--fid-samples", type=int, default=200_000)
    parser.add_argument("--sw-samples", type=int, default=100_000)
    parser.add_argument("--n-proj", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--no-merge", action="store_true",
                        help="Only write --output; do not merge into --self-dist.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Do not create a .before_symmetric backup before merge.")
    args = parser.parse_args()

    self_path = Path(args.self_dist)
    output = Path(args.output)
    vec_dir = Path(args.vec_dir)
    if not self_path.exists():
        raise SystemExit(f"self-distance CSV not found: {self_path}")

    self_df = pd.read_csv(self_path)
    pairs = _select_pairs(self_df, set(args.spaces), set(args.pair_types))
    existing = _existing_lookup(output)
    mmd_lookup = _load_mmd_lookup(Path(args.flow_mmd_csv))

    compute_symmetric_rows(
        pairs=pairs,
        vec_dir=vec_dir,
        existing=existing,
        mmd_lookup=mmd_lookup,
        fid_samples=args.fid_samples,
        sw_samples=args.sw_samples,
        n_proj=args.n_proj,
        seed=args.seed,
        output=output,
        flush_every=args.flush_every,
    )

    if not args.no_merge:
        backup = None if args.no_backup else self_path.with_suffix(".before_symmetric.csv")
        merge_into_self_dist(self_path, output, backup)


if __name__ == "__main__":
    main()
