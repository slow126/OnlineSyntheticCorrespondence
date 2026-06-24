"""Materialize the paper's recommended selection policy as a rows family.

The policy: rank by the symmetric mean-NN distance for from-scratch contexts
(both directions carry signal there; averaging denoises) and by the matched
direction (d_B->T, missing support) for pretrained-backbone contexts. RAFT is
from-scratch by construction.

Builds rows_{split}_motion_policy.csv in the predictions dir by selecting,
per context, the rows from motion_meannn_sym (scratch) or motion_rule
(pretrained). The motion_rule family already uses the matched direction per
regime, so its pretrained contexts are exactly the d_B->T ranking.

Usage:
    python scripts/transfer_analysis_v5/make_policy_rows.py \
        --rows-dir scripts/transfer_analysis_v4/results_rule_v5core/predictions/peak_pck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def regime_of(context_id: str) -> str:
    _, model, pretrained, _ = context_id.split("|")
    if model == "raft":
        return "scratch"
    return "pretrained" if pretrained == "True" else "scratch"


def combine(sym_path: Path, rule_path: Path, dest: Path, normalize=False):
    sym = pd.read_csv(sym_path)
    rule = pd.read_csv(rule_path)
    if normalize:
        # the two calibration runs differ only in rows excluded everywhere
        # downstream: middlebury (eval-harness defect) and the stray
        # single-source raft|False|False contexts (unrankable)
        sym = sym[(sym.benchmark != "middlebury")
                  & (~sym.context_id.str.endswith("raft|False|False"))]
        rule = rule[(rule.benchmark != "middlebury")
                    & (~rule.context_id.str.endswith("raft|False|False"))]
    sym_part = sym[sym.context_id.map(regime_of) == "scratch"]
    rule_part = rule[rule.context_id.map(regime_of) == "pretrained"]
    out = pd.concat([sym_part, rule_part], ignore_index=True)
    # sanity: the two parts must tile the context set exactly (per-context row
    # counts may differ by one source between runs; ranking is per context)
    assert set(out.context_id) == set(sym.context_id) == set(rule.context_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"wrote {dest}  ({len(sym_part)} scratch rows from "
          f"{sym_path.name}, {len(rule_part)} pretrained rows from "
          f"{rule_path.name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["LOTO", "LOBO", "JOINT"])
    ap.add_argument("--scratch-family", default="motion_meannn_sym")
    ap.add_argument("--pretrained-family", default="motion_rule")
    ap.add_argument("--out-family", default="motion_policy")
    ap.add_argument("--benchsim-sym", default=None,
                    help="dir of calibrated rows for the scratch (sym) arm; "
                    "with --benchsim-rule, also writes benchsim_policy/")
    ap.add_argument("--benchsim-rule", default=None)
    ap.add_argument("--benchsim-out", default=None)
    args = ap.parse_args()

    rows_dir = Path(args.rows_dir)
    for split in args.splits:
        combine(rows_dir / f"rows_{split}_{args.scratch_family}.csv",
                rows_dir / f"rows_{split}_{args.pretrained_family}.csv",
                rows_dir / f"rows_{split}_{args.out_family}.csv")

    if args.benchsim_sym and args.benchsim_rule and args.benchsim_out:
        for split in args.splits:
            combine(Path(args.benchsim_sym) / f"rows_{split}_all_variants.csv",
                    Path(args.benchsim_rule) / f"rows_{split}_all_variants.csv",
                    Path(args.benchsim_out) / f"rows_{split}_all_variants.csv",
                    normalize=True)


if __name__ == "__main__":
    main()
