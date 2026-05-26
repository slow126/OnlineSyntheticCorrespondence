"""One-shot: average spair_long catspp peak_pck into the existing spair rows
of transfer_table.csv. Backs up the original alongside as .pre_spair_long.csv.

The spair_long raft snapshot has no validation_results.csv (training cut short
on the cluster), so only the 4 catspp variants get averaged. spair_long has
only 1 eval point per snapshot, so peak_pck == final_pck for it.

Run once before re-running v4:
    python scripts/transfer_analysis_v4/patch_spair_long.py
"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import pandas as pd

SPAIR_LONG = Path("/mnt/nvme_1tb_b/spair_long")
TABLE = Path("scripts/transfer_analysis_v3/transfer_table.csv")
BACKUP = TABLE.with_suffix(".pre_spair_long.csv")


def variant_from_dirname(name: str) -> tuple[str, bool, bool] | None:
    """Parse e.g. 'spair_cats_steps100_pretrainedTrue_freezeFalse_2026_05_21_21_49'
    -> ('catspp', pretrained=True, freeze=False)."""
    if "_cats_" in name:
        fam = "catspp"
    elif "_raft_" in name:
        fam = "raft"
    else:
        return None
    pretrained = "pretrainedTrue" in name
    freeze = "freezeTrue" in name
    return fam, pretrained, freeze


def main():
    if not TABLE.exists():
        raise SystemExit(f"missing: {TABLE}")
    if not BACKUP.exists():
        shutil.copy(TABLE, BACKUP)
        print(f"backup -> {BACKUP}")
    else:
        print(f"(backup already exists at {BACKUP}, leaving as-is)")

    t = pd.read_csv(TABLE)
    n_updated = 0
    skipped_no_val = []

    for snap in sorted(glob.glob(str(SPAIR_LONG / "*"))):
        name = os.path.basename(snap)
        parsed = variant_from_dirname(name)
        if parsed is None:
            continue
        fam, pretrained, freeze = parsed
        val_path = Path(snap) / "validation_results.csv"
        if not val_path.exists():
            skipped_no_val.append(name)
            continue
        v = pd.read_csv(val_path)
        if v.empty:
            skipped_no_val.append(name)
            continue
        # spair_long has only 1 eval point per benchmark; peak == final.
        # If there are multiple, take the max (matches peak semantics).
        peaks = v.groupby("benchmark")["pck"].max().to_dict()
        for bench, new_pck in peaks.items():
            mask = (
                (t["train_dataset"] == "spair")
                & (t["benchmark"] == bench)
                & (t["model_family"] == fam)
                & (t["pretrained"] == pretrained)
                & (t["freeze"] == freeze)
            )
            n = int(mask.sum())
            if n == 0:
                print(f"  WARN  no existing row for {fam} pt={pretrained} fz={freeze} bench={bench}")
                continue
            if n > 1:
                print(f"  WARN  {n} rows match {fam} pt={pretrained} fz={freeze} bench={bench}; averaging into all")
            old = t.loc[mask, "peak_pck"].values
            avg = (old + float(new_pck)) / 2.0
            t.loc[mask, "peak_pck"] = avg
            n_updated += n
            print(f"  {fam:6s} pt={int(pretrained)} fz={int(freeze)}  {bench:13s}  "
                  f"old={old[0]:6.2f}  new={float(new_pck):6.2f}  avg={avg[0]:6.2f}")

    t.to_csv(TABLE, index=False)
    print()
    print(f"updated {n_updated} rows; wrote {TABLE}")
    if skipped_no_val:
        print(f"skipped (no validation_results.csv): {skipped_no_val}")


if __name__ == "__main__":
    main()
