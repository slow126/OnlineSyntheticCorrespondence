"""Refresh machine-rendered table blocks inside hand-edited markdown docs.

Scans the given markdown files (default: every .md in the vault Project folder
that contains a marker) for blocks of the form

    <!-- tbl:NAME -->
    ...                <- replaced with the freshly rendered table
    <!-- /tbl:NAME -->

and re-renders ONLY the content between markers from the artifact CSVs (see
blocks.py). All prose outside markers is never touched — edit the documents
freely; rerun this after any artifact rerun.

    python scripts/transfer_analysis_v5/update_tables.py            # all vault docs
    python scripts/transfer_analysis_v5/update_tables.py path.md    # specific file(s)
    python scripts/transfer_analysis_v5/update_tables.py --check    # report only
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from blocks import VAULT, build_blocks  # noqa: E402

MARKER = re.compile(
    r"(<!-- tbl:(?P<name>[A-Za-z0-9_]+) -->)\n.*?\n(<!-- /tbl:(?P=name) -->)",
    re.DOTALL)


def refresh(path: Path, blocks: dict[str, str], check: bool) -> tuple[int, int]:
    text = path.read_text()
    updated, unknown = 0, 0

    def sub(m: re.Match) -> str:
        nonlocal updated, unknown
        name = m.group("name")
        if name not in blocks:
            unknown += 1
            print(f"  WARNING: {path.name}: no renderer for block '{name}' — left as-is")
            return m.group(0)
        # blank lines around the content: Obsidian won't render a table glued
        # to the HTML comment line
        new = f"{m.group(1)}\n\n{blocks[name]}\n\n{m.group(3)}"
        if new != m.group(0):
            updated += 1
        return new

    out = MARKER.sub(sub, text)
    if out != text and not check:
        path.write_text(out)
    return updated, unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path,
                    help="markdown files (default: vault Project/*.md with markers)")
    ap.add_argument("--check", action="store_true",
                    help="report what would change without writing")
    args = ap.parse_args()

    files = args.files or [p for p in sorted(VAULT.glob("*.md"))
                           if "<!-- tbl:" in p.read_text()]
    if not files:
        print(f"no files with <!-- tbl:NAME --> markers under {VAULT}")
        return

    blocks = build_blocks()
    for f in files:
        n, bad = refresh(f, blocks, args.check)
        verb = "would update" if args.check else "updated"
        print(f"{f}: {verb} {n} block(s)" + (f", {bad} unknown" if bad else ""))


if __name__ == "__main__":
    main()
