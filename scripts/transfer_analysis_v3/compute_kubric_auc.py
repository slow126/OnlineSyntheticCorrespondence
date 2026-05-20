#!/usr/bin/env python3
"""
Compute AUC from kubric (movi_f) snapshot validation_results.csv files and
append rows to analysis/leakage_free_flow_kmeans_manifold/auc_results.csv.

The kubric runs use 100 steps/epoch with check_val_every_n_epoch=10, so
evaluations land at steps 1000, 2000, 3000, 4000, 5000 — identical to the
first 5 evaluation checkpoints from all other training runs.  AUC is
therefore directly comparable to the rest of the auc_results.csv.

Snapshot naming conventions handled:
  movi_f_movi_f_cats_pretrained{True|False}_freeze{True|False}_{timestamp}
  movi_f_movi_f_raft_baseline_{timestamp}   (pretrained/freeze read from config.yaml)

Usage:
    python scripts/transfer_analysis_v3/compute_kubric_auc.py \\
        [--snapshot-dir /mnt/nvme_1tb_b/kubric_snapshots] \\
        [--auc-csv analysis/leakage_free_flow_kmeans_manifold/auc_results.csv] \\
        [--auc-steps 5000] \\
        [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


AUC_STEPS = 5000
STEPS_PER_EPOCH = 100  # from config: training.steps_per_epoch

# Regex for cats-style run names
_CATS_RE = re.compile(
    r"^(?P<train>movi_f)_(?P<train2>movi_f)_cats"
    r"_pretrained(?P<pretrained>True|False)"
    r"_freeze(?P<freeze>True|False)"
    r"_(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$"
)

# Regex for raft_baseline run names
_RAFT_RE = re.compile(
    r"^(?P<train>movi_f)_(?P<train2>movi_f)_raft_baseline"
    r"_(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$"
)


def parse_snapshot(snap_dir: Path) -> dict | None:
    """Return metadata dict for a snapshot dir, or None if unrecognised."""
    name = snap_dir.name

    m = _CATS_RE.match(name)
    if m:
        return {
            "run_id":       str(snap_dir),
            "run_name":     name,
            "train_dataset": "movi_f",
            "step_tag":     "steps",
            "logsteps":     float(STEPS_PER_EPOCH),
            "pretrained":   m.group("pretrained") == "True",
            "freeze":       m.group("freeze") == "True",
            "timestamp":    m.group("timestamp"),
            "model_family": "catspp",
        }

    m = _RAFT_RE.match(name)
    if m:
        cfg_path = snap_dir / "config.yaml"
        pretrained = False
        freeze = False
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            model_cfg = cfg.get("model", {})
            freeze = bool(model_cfg.get("freeze", False))
            # pretrained here means pre-trained FLOW weights (not just backbone)
            pretrained = cfg.get("paths", {}).get("pretrained") is not None
        return {
            "run_id":       str(snap_dir),
            "run_name":     name,
            "train_dataset": "movi_f",
            "step_tag":     "steps",
            "logsteps":     float(STEPS_PER_EPOCH),
            "pretrained":   pretrained,
            "freeze":       freeze,
            "timestamp":    m.group("timestamp"),
            "model_family": "raft",
        }

    return None


def compute_auc_rows(meta: dict, val_csv: Path, auc_steps: int) -> list[dict]:
    """Compute per-benchmark AUC rows from validation_results.csv."""
    df = pd.read_csv(val_csv)
    df["training_steps"] = df["epoch"] * STEPS_PER_EPOCH
    df = df[df["training_steps"] <= auc_steps]

    rows = []
    for bench, sub in df.groupby("benchmark"):
        sub = sub.sort_values("training_steps")
        if len(sub) < 2:
            print(f"    WARNING: {bench} has <2 eval points in first {auc_steps} steps — skipping")
            continue

        steps_arr = sub["training_steps"].to_numpy(dtype=float)
        pck_arr   = sub["pck"].to_numpy(dtype=float)
        auc = float(np.trapz(pck_arr, steps_arr))
        auc_norm = auc / auc_steps

        peak_idx = pck_arr.argmax()
        rows.append({
            **meta,
            "benchmark":           bench,
            "auc_steps":           auc_steps,
            "auc":                 auc,
            "auc_normalized":      auc_norm,
            "auc_points":          len(sub),
            "peak_pck":            float(pck_arr[peak_idx]),
            "peak_training_steps": int(steps_arr[peak_idx]),
            "peak_epoch":          int(sub["epoch"].iloc[peak_idx]),
            "final_pck":           float(pck_arr[-1]),
            "final_training_steps": int(steps_arr[-1]),
            "final_epoch":         int(sub["epoch"].iloc[-1]),
            "drop_pck":            float(pck_arr[peak_idx] - pck_arr[-1]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", default="/mnt/nvme_1tb_b/kubric_snapshots",
                        help="Directory containing kubric snapshot subdirs.")
    parser.add_argument("--auc-csv",
                        default="analysis/leakage_free_flow_kmeans_manifold/auc_results.csv",
                        help="Path to the existing auc_results.csv to append to.")
    parser.add_argument("--auc-steps", type=int, default=AUC_STEPS,
                        help="Max training steps for AUC window.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print new rows without writing.")
    args = parser.parse_args()

    snap_root = Path(args.snapshot_dir)
    auc_path  = Path(args.auc_csv)

    if not snap_root.exists():
        print(f"ERROR: --snapshot-dir {snap_root} not found")
        sys.exit(1)
    if not auc_path.exists():
        print(f"ERROR: --auc-csv {auc_path} not found")
        sys.exit(1)

    existing = pd.read_csv(auc_path)
    existing_keys = set(zip(existing["run_id"], existing["benchmark"]))
    print(f"Existing rows: {len(existing)}  ({len(existing_keys)} unique run×benchmark pairs)")

    new_rows: list[dict] = []
    for snap_dir in sorted(snap_root.iterdir()):
        if not snap_dir.is_dir():
            continue
        meta = parse_snapshot(snap_dir)
        if meta is None:
            print(f"  SKIP (unrecognised name): {snap_dir.name}")
            continue

        val_csv = snap_dir / "validation_results.csv"
        if not val_csv.exists():
            print(f"  SKIP (no validation_results.csv): {snap_dir.name}")
            continue

        print(f"  Processing: {snap_dir.name}")
        rows = compute_auc_rows(meta, val_csv, args.auc_steps)

        added = 0
        for row in rows:
            key = (row["run_id"], row["benchmark"])
            if key in existing_keys:
                continue
            new_rows.append(row)
            existing_keys.add(key)
            added += 1
        print(f"    {added}/{len(rows)} benchmarks are new (rest already in CSV)")

    if not new_rows:
        print("\nNo new rows to add.")
        return

    new_df = pd.DataFrame(new_rows, columns=existing.columns)
    print(f"\nNew rows: {len(new_df)}")
    print(new_df[["run_name", "benchmark", "model_family", "pretrained", "freeze",
                   "auc_normalized", "auc_points"]].to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(auc_path, index=False)
    print(f"\n✓ Appended {len(new_df)} rows to {auc_path}  (total: {len(combined)})")


if __name__ == "__main__":
    main()
