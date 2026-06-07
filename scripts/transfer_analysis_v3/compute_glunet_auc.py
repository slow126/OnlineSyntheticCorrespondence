#!/usr/bin/env python3
"""
Compute AUC / peak_pck from GLU-Net snapshot validation_results.csv files and
append rows to analysis/leakage_free_flow_kmeans_manifold/auc_results.csv.

Mirrors compute_kubric_auc.py. GLU-Net runs use steps_per_epoch=100 with
check_val_every_n_epoch=20 (eval at steps 2000, 4000, 6000, ...) plus an
eval_initial point (epoch -1 -> step 0). To stay comparable with the rest of
auc_results.csv:

  * auc / auc_normalized  -> integrated over the first `--auc-steps` (default
    5000) training steps only when at least `--min-auc-points` checkpoints
    exist. Otherwise these fields are NaN while peak_pck remains usable.
  * peak_pck              -> peak through `--peak-steps`, or the full available
    run when that option is omitted. Use a common horizon when importing
    partial runs with unequal progress.

Snapshot naming handled:
  {train_dataset}_glunet_steps100_pretrainedTrue_freezeFalse_{timestamp}

Usage:
    python scripts/transfer_analysis_v3/compute_glunet_auc.py \\
        --snapshot-dir /home/spencer/rc_glunet_val_csvs/snapshots \\
        [--auc-csv analysis/leakage_free_flow_kmeans_manifold/auc_results.csv] \\
        [--auc-steps 5000] [--dry-run]
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AUC_STEPS = 5000
STEPS_PER_EPOCH = 100

_GLUNET_RE = re.compile(
    r"^(?P<train>.+)_glunet_steps\d+"
    r"_pretrained(?P<pretrained>True|False)"
    r"_freeze(?P<freeze>True|False)"
    r"_(?P<timestamp>\d{4}_\d{2}_\d{2}_\d{2}_\d{2})$"
)


def parse_snapshot(snap_dir: Path) -> dict | None:
    m = _GLUNET_RE.match(snap_dir.name)
    if not m:
        return None
    return {
        "run_id":       str(snap_dir),
        "run_name":     snap_dir.name,
        "train_dataset": m.group("train"),
        "step_tag":     "steps",
        "logsteps":     float(STEPS_PER_EPOCH),
        "pretrained":   m.group("pretrained") == "True",
        "freeze":       m.group("freeze") == "True",
        "timestamp":    m.group("timestamp"),
        "model_family": "glunet",
    }


def compute_auc_rows(
    meta: dict,
    val_csv: Path,
    auc_steps: int,
    min_auc_points: int = 3,
    peak_steps: int | None = None,
) -> list[dict]:
    df = pd.read_csv(val_csv)
    # Map the eval_initial point (epoch -1) to step 0 so it anchors the AUC window
    # rather than producing a negative training_steps.
    df["epoch_clip"] = df["epoch"].clip(lower=0)
    df["training_steps"] = df["epoch_clip"] * STEPS_PER_EPOCH

    rows = []
    for bench, sub in df.groupby("benchmark"):
        sub = sub.sort_values("training_steps")
        peak_window = (
            sub[sub["training_steps"] <= peak_steps]
            if peak_steps is not None
            else sub
        )
        if peak_window.empty:
            print(f"    WARNING: {bench} has no points through peak horizon "
                  f"{peak_steps} — skipping")
            continue
        peak_pck = peak_window["pck"].to_numpy(dtype=float)
        peak_training_steps = peak_window["training_steps"].to_numpy(dtype=float)
        peak_epoch = peak_window["epoch"].to_numpy(dtype=int)
        peak_idx = peak_pck.argmax()

        # AUC over the first `auc_steps` steps only
        win = sub[sub["training_steps"] <= auc_steps]
        # drop duplicate step-0 rows if eval_initial + epoch0 collide
        win = win.drop_duplicates(subset="training_steps")
        if len(win) >= min_auc_points:
            w_steps = win["training_steps"].to_numpy(dtype=float)
            w_pck = win["pck"].to_numpy(dtype=float)
            auc = float(np.trapz(w_pck, w_steps))
            auc_norm = auc / auc_steps
        else:
            auc = np.nan
            auc_norm = np.nan

        rows.append({
            **meta,
            "benchmark":            bench,
            "auc_steps":            auc_steps,
            "auc":                  auc,
            "auc_normalized":       auc_norm,
            "auc_points":           len(win),
            "peak_pck":             float(peak_pck[peak_idx]),
            "peak_training_steps":  int(peak_training_steps[peak_idx]),
            "peak_epoch":           int(peak_epoch[peak_idx]),
            "final_pck":            float(peak_pck[-1]),
            "final_training_steps": int(peak_training_steps[-1]),
            "final_epoch":          int(peak_epoch[-1]),
            "drop_pck":             float(peak_pck[peak_idx] - peak_pck[-1]),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", default="/home/spencer/rc_glunet_val_csvs/snapshots",
                        help="Directory containing glunet snapshot subdirs.")
    parser.add_argument("--auc-csv",
                        default="analysis/leakage_free_flow_kmeans_manifold/auc_results.csv")
    parser.add_argument("--auc-steps", type=int, default=AUC_STEPS)
    parser.add_argument("--min-auc-points", type=int, default=3,
                        help="Minimum checkpoints required to report AUC. "
                             "Rows with fewer points retain peak_pck.")
    parser.add_argument(
        "--peak-steps",
        type=int,
        default=None,
        help="Restrict peak_pck to this common training-step horizon.",
    )
    parser.add_argument(
        "--train-datasets",
        nargs="+",
        default=None,
        help="Only import these training datasets.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    snap_root = Path(args.snapshot_dir)
    auc_path = Path(args.auc_csv)
    if not snap_root.exists():
        print(f"ERROR: --snapshot-dir {snap_root} not found"); sys.exit(1)
    if not auc_path.exists():
        print(f"ERROR: --auc-csv {auc_path} not found"); sys.exit(1)

    existing = pd.read_csv(auc_path)
    existing_keys = set(zip(existing["run_id"], existing["benchmark"]))
    print(f"Existing rows: {len(existing)}  (families: {sorted(existing['model_family'].unique())})")

    new_rows: list[dict] = []
    for snap_dir in sorted(snap_root.iterdir()):
        if not snap_dir.is_dir():
            continue
        meta = parse_snapshot(snap_dir)
        if meta is None:
            print(f"  SKIP (unrecognised): {snap_dir.name}")
            continue
        if (
            args.train_datasets is not None
            and meta["train_dataset"] not in set(args.train_datasets)
        ):
            continue
        val_csv = snap_dir / "validation_results.csv"
        if not val_csv.exists():
            print(f"  SKIP (no csv): {snap_dir.name}")
            continue
        print(f"  Processing: {meta['train_dataset']}")
        for row in compute_auc_rows(
            meta,
            val_csv,
            args.auc_steps,
            args.min_auc_points,
            args.peak_steps,
        ):
            key = (row["run_id"], row["benchmark"])
            if key in existing_keys:
                continue
            new_rows.append(row)
            existing_keys.add(key)

    if not new_rows:
        print("\nNo new rows to add.")
        return

    new_df = pd.DataFrame(new_rows, columns=existing.columns)
    print(f"\nNew glunet rows: {len(new_df)}")
    print(new_df[["train_dataset", "benchmark", "auc_normalized", "auc_points",
                  "peak_pck", "peak_epoch"]].to_string(index=False))

    if args.dry_run:
        print("\n--dry-run: not writing.")
        return
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(auc_path, index=False)
    print(f"\n✓ Appended {len(new_df)} rows to {auc_path}  (total: {len(combined)})")


if __name__ == "__main__":
    main()
