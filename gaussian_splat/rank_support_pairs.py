#!/usr/bin/env python3
"""
Rank pairwise support summaries to help choose informative train/eval comparisons for figures.

Reads: <summary-root>/<pair>/<pair>__summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

BASE_TRAINS = {
    "synthetic_train",
    "sintel_train",
    "pointodyssey_train",
    "flyingthings_train",
    "imagenet2dwarp_train",
    "spair_train",
}


def load_rows(summary_root: Path, base_only: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in sorted(summary_root.glob("*/*__summary.json")):
        with p.open("r") as f:
            s = json.load(f)
        pair = s["pair_name"]
        train = pair.split("__")[0]
        if base_only and train not in BASE_TRAINS:
            continue
        fr = s.get("fractions", {})
        ein = float(fr.get("eval_in_train", 0.0))
        tie = float(fr.get("train_in_eval", 0.0))
        rows.append(
            {
                "pair": pair,
                "train": train,
                "eval_in_train": ein,
                "train_in_eval": tie,
                "asym": abs(ein - tie),
                "harmonic": 0.0 if (ein + tie) <= 0 else (2.0 * ein * tie / (ein + tie)),
            }
        )
    return rows


def print_block(title: str, rows: List[Dict[str, Any]], k: int) -> None:
    print(f"\n{title}")
    print("pair,eval_in_train,train_in_eval,asym,harmonic")
    for r in rows[:k]:
        print(
            f"{r['pair']},{r['eval_in_train']:.6f},{r['train_in_eval']:.6f},"
            f"{r['asym']:.6f},{r['harmonic']:.6f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank informative support pair summaries")
    ap.add_argument(
        "--summary-root",
        default="gaussian_splat/output_joint_flow_splats/space_joint__a1p0__t2e1p0__e2t1p5",
        help="Root containing pair subdirs with __summary.json files",
    )
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--base-only", action="store_true", help="Restrict to main base training datasets")
    args = ap.parse_args()

    root = Path(args.summary_root)
    rows = load_rows(root, base_only=args.base_only)
    if not rows:
        raise SystemExit(f"No summaries found under: {root}")

    print_block("Top eval_in_train (best eval coverage)", sorted(rows, key=lambda r: r["eval_in_train"], reverse=True), args.topk)
    print_block("Bottom eval_in_train (worst eval coverage)", sorted(rows, key=lambda r: r["eval_in_train"]), args.topk)
    print_block("Top asymmetry |eval_in_train - train_in_eval|", sorted(rows, key=lambda r: r["asym"], reverse=True), args.topk)
    print_block("Top mutual overlap (harmonic mean)", sorted(rows, key=lambda r: r["harmonic"], reverse=True), args.topk)


if __name__ == "__main__":
    main()
