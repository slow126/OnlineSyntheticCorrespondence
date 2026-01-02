#!/usr/bin/env python3
"""
Aggregate per-run summary reports into a single root summary.
"""

import argparse
from pathlib import Path
from typing import List, Optional


def _extract_line(lines: List[str], prefix: str) -> Optional[str]:
    for line in lines:
        if line.startswith(prefix):
            return line.strip()
    return None


def _extract_block(lines: List[str], header: str, end_markers: List[str]) -> List[str]:
    for idx, line in enumerate(lines):
        if line.strip() == header:
            out = []
            for next_line in lines[idx + 1:]:
                stripped = next_line.strip()
                if not stripped:
                    break
                if any(stripped.startswith(marker) for marker in end_markers):
                    break
                out.append(next_line.rstrip())
            return out
    return []


def _summarize_report(path: Path) -> List[str]:
    lines = path.read_text().splitlines()
    out = []
    target = _extract_line(lines, "Target:")
    pred_target = _extract_line(lines, "Prediction target:")
    predictors = _extract_line(lines, "Predictors:")
    missing_auc = _extract_line(lines, "Missing AUC table:")

    if target:
        out.append(target)
    if pred_target:
        out.append(pred_target)
    if predictors:
        out.append(predictors)
    if missing_auc:
        out.append(missing_auc)

    headline = _extract_block(
        lines,
        "Headline metrics:",
        ["Overall predictor signal", "Standardized", "Prediction validation", "Takeaways", "LEAKAGE-FREE SUMMARY"],
    )
    if headline:
        out.append("Headline metrics:")
        out.extend(f"  {line.strip()}" for line in headline)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate summary_report.txt files.")
    parser.add_argument(
        "--base-dir",
        default="analysis/leakage_free_local_fast_dino_faiss",
        help="Base directory containing target/variant summary reports.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output summary file (default: base_dir/summary_report.txt).",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_path = Path(args.output_file) if args.output_file else base_dir / "summary_report.txt"

    summary_paths = sorted(base_dir.glob("*/*/summary_report.txt"))
    lines: List[str] = []
    lines.append("LEAKAGE-FREE SUMMARY (AGGREGATED)")
    lines.append("=" * 80)
    if not summary_paths:
        lines.append(f"No summary reports found under {base_dir}")
        output_path.write_text("\n".join(lines))
        return

    for path in summary_paths:
        rel = path.relative_to(base_dir)
        parts = rel.parts
        if len(parts) >= 3:
            run_name = f"{parts[0]}/{parts[1]}"
        else:
            run_name = str(rel)
        lines.append("")
        lines.append(f"RUN: {run_name}")
        lines.append("-" * (len(run_name) + 5))
        summary_lines = _summarize_report(path)
        if summary_lines:
            lines.extend(f"  {line}" for line in summary_lines)
        else:
            lines.append(f"  (no summary extracted) {path}")

    output_path.write_text("\n".join(lines))
    print(f"Wrote aggregated summary to {output_path}")


if __name__ == "__main__":
    main()
