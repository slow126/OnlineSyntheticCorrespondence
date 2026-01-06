#!/usr/bin/env python3
"""
Summarize leakage-free summary_report.txt files into a comparison table.

Usage:
  python scripts/summarize_target_comparison.py \
    --out analysis/target_compare/summary.csv \
    --label modeA path/to/summary_report.txt \
    --label modeB path/to/summary_report.txt
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List


_PRED_RE = re.compile(
    r"^(LOBO|LOTO) pred: .*Pearson=([+-]?\d+\.\d+).+Spearman=([+-]?\d+\.\d+)"
)
_RANK_RE = re.compile(
    r"^(LOBO|LOTO) rank: top1=([+-]?\d+\.\d+), top3=([+-]?\d+\.\d+), "
    r"top20%=([+-]?\d+\.\d+).+spearman=([+-]?\d+\.\d+)"
)


def parse_summary(path: Path) -> Dict[str, float]:
    data: Dict[str, float] = {}
    lines = path.read_text().splitlines()
    for line in lines:
        line = line.strip()
        pred_match = _PRED_RE.match(line)
        if pred_match:
            tag, pearson, spearman = pred_match.groups()
            data[f"{tag.lower()}_pred_pearson"] = float(pearson)
            data[f"{tag.lower()}_pred_spearman"] = float(spearman)
            continue
        rank_match = _RANK_RE.match(line)
        if rank_match:
            tag, top1, top3, top20, spearman = rank_match.groups()
            tag = tag.lower()
            data[f"{tag}_rank_top1"] = float(top1)
            data[f"{tag}_rank_top3"] = float(top3)
            data[f"{tag}_rank_top20"] = float(top20)
            data[f"{tag}_rank_spearman"] = float(spearman)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--label",
        nargs=2,
        action="append",
        metavar=("NAME", "SUMMARY_PATH"),
        required=True,
        help="Label and summary_report.txt path.",
    )
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    fieldnames = [
        "mode",
        "lobo_pred_pearson",
        "lobo_pred_spearman",
        "lobo_rank_top1",
        "lobo_rank_top3",
        "lobo_rank_top20",
        "lobo_rank_spearman",
        "loto_pred_pearson",
        "loto_pred_spearman",
        "loto_rank_top1",
        "loto_rank_top3",
        "loto_rank_top20",
        "loto_rank_spearman",
    ]

    for label, summary_path in args.label:
        path = Path(summary_path)
        if not path.exists():
            raise FileNotFoundError(f"Summary not found: {path}")
        data = parse_summary(path)
        row: Dict[str, object] = {"mode": label}
        for key in fieldnames[1:]:
            row[key] = data.get(key, float("nan"))
        rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote summary to: {out_path}")


if __name__ == "__main__":
    main()
