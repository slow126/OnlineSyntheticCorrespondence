#!/usr/bin/env python3
"""
Convert a directory of CSV tables into paper-ready LaTeX tables.

Default behavior:
  - Reads all CSV files in --input-dir (non-recursive).
  - Writes one .tex per CSV in --output-dir.
  - Escapes LaTeX-sensitive characters in text columns.
  - Formats numeric columns with fixed precision.
  - Uses booktabs with an optional resizebox wrapper.
  - Emits:
      - latex_tables_manifest.csv
      - include_all_tables.tex (all table environments concatenated)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


def _split_csv_arg(text: str) -> List[str]:
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _title_from_stem(stem: str) -> str:
    text = stem.replace("_", " ").replace("-", " ").strip()
    if not text:
        return "Table"
    words = [w.capitalize() if not w.isupper() else w for w in text.split()]
    return " ".join(words)


def _sanitize_label(stem: str, prefix: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem).strip("_").lower()
    if not slug:
        slug = "table"
    return f"{prefix}:{slug}"


def _latex_escape(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text == "nan":
        return ""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in text:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _format_float(
    x: float,
    precision: int,
    sci_thresh: int,
    na_token: str,
    strip_trailing_zeros: bool = True,
) -> str:
    if not math.isfinite(x):
        return na_token
    if x == 0:
        return "0" if strip_trailing_zeros else f"{0:.{precision}f}"
    abs_x = abs(x)
    if abs_x >= 10 ** sci_thresh or abs_x < 10 ** (-sci_thresh):
        return f"{x:.{precision}e}"
    text = f"{x:.{precision}f}"
    if strip_trailing_zeros:
        return text.rstrip("0").rstrip(".")
    return text


def _format_dataframe(
    df: pd.DataFrame,
    precision: int,
    sci_thresh: int,
    na_token: str,
    strip_trailing_zeros: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].map(lambda v: "True" if bool(v) else "False")
        elif pd.api.types.is_numeric_dtype(out[col]):
            if _is_int_like_numeric(out[col]):
                out[col] = pd.to_numeric(out[col], errors="coerce").map(
                    lambda v: str(int(round(float(v)))) if pd.notna(v) else na_token
                )
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce").map(
                lambda v: _format_float(
                    float(v),
                    precision,
                    sci_thresh,
                    na_token,
                    strip_trailing_zeros=strip_trailing_zeros,
                )
                if pd.notna(v)
                else na_token
            )
        else:
            out[col] = out[col].map(lambda v: _latex_escape(v) if pd.notna(v) else _latex_escape(na_token))
    out.columns = [_latex_escape(c) for c in out.columns]
    return out


def _column_alignment(
    df: pd.DataFrame,
    numeric_align: str = "r",
    add_vertical_rules: bool = False,
) -> str:
    align = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            align.append(numeric_align)
        else:
            align.append("l")
    if not align:
        return "l"
    if add_vertical_rules:
        return "|" + "|".join(align) + "|"
    return "".join(align)


def _build_table_tex(
    df: pd.DataFrame,
    caption: str,
    label: str,
    precision: int,
    sci_thresh: int,
    na_token: str,
    use_resizebox: bool,
    table_position: str,
    cell_note: str,
    cell_note_raw: bool,
    tabcolsep: str,
    strip_trailing_zeros: bool,
    numeric_align: str,
    add_vertical_rules: bool,
) -> str:
    if df.empty:
        body = "% Empty table\n\\begin{tabular}{l}\n\\toprule\n(Empty) \\\\\n\\bottomrule\n\\end{tabular}\n"
    else:
        fmt_df = _format_dataframe(
            df,
            precision=precision,
            sci_thresh=sci_thresh,
            na_token=na_token,
            strip_trailing_zeros=strip_trailing_zeros,
        )
        col_align = _column_alignment(
            df,
            numeric_align=numeric_align,
            add_vertical_rules=add_vertical_rules,
        )
        tabular = fmt_df.to_latex(
            index=False,
            escape=False,
            column_format=col_align,
            na_rep="",
        )
        body = tabular

    lines: List[str] = []
    lines.append(f"\\begin{{table}}[{table_position}]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    if cell_note.strip():
        if cell_note_raw:
            lines.append(f"\\textit{{{cell_note}}}")
        else:
            lines.append(f"\\textit{{{_latex_escape(cell_note)}}}")
    if str(tabcolsep).strip():
        lines.append(f"\\setlength{{\\tabcolsep}}{{{str(tabcolsep).strip()}}}")
    if use_resizebox:
        lines.append("\\resizebox{\\linewidth}{!}{%")
        lines.append(body.rstrip())
        lines.append("}")
    else:
        lines.append(body.rstrip())
    lines.append("\\end{table}")
    lines.append("")
    return "\n".join(lines)


def _iter_csv_paths(input_dir: Path, recursive: bool) -> List[Path]:
    patt = "**/*.csv" if recursive else "*.csv"
    return sorted(p for p in input_dir.glob(patt) if p.is_file())


def _load_caption_map(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise SystemExit(f"Missing caption map file: {path}")
    df = pd.read_csv(path)
    if "file" not in df.columns or "caption" not in df.columns:
        raise SystemExit("Caption map must have columns: file,caption")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row["file"]).strip()
        cap = str(row["caption"]).strip()
        if key:
            out[key] = cap
    return out


def _is_int_like_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return False
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[vals.notna()]
    if vals.empty:
        return False
    # Values close to integers are rendered as integers (counts/fold ids/etc.).
    frac = (vals - vals.round()).abs()
    return bool((frac <= 1e-9).all())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CSV tables to LaTeX table files.")
    parser.add_argument("--input-dir", required=True, help="Directory containing CSV tables.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated .tex files.")
    parser.add_argument(
        "--include",
        default="",
        help="Optional comma-separated filename stems to include (without .csv).",
    )
    parser.add_argument(
        "--exclude",
        default="",
        help="Optional comma-separated filename stems to exclude (without .csv).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan for CSVs under input dir.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional row cap per table (0 = all rows).",
    )
    parser.add_argument(
        "--float-precision",
        type=int,
        default=4,
        help="Decimal precision for numeric columns.",
    )
    parser.add_argument(
        "--scientific-threshold",
        type=int,
        default=4,
        help="Use scientific notation for |x| >= 10^N or |x| < 10^-N.",
    )
    parser.add_argument(
        "--label-prefix",
        default="tab",
        help="LaTeX label prefix (default: tab).",
    )
    parser.add_argument(
        "--table-position",
        default="t",
        help="LaTeX table position flag (default: t).",
    )
    parser.add_argument(
        "--no-resizebox",
        action="store_true",
        help="Disable wrapping table with resizebox{\\linewidth}{!}{...}.",
    )
    parser.add_argument(
        "--caption-map",
        default="",
        help="Optional CSV map with columns: file,caption where file is filename or stem.",
    )
    parser.add_argument(
        "--cell-note",
        default="",
        help="Optional note inserted under caption (e.g., tuple semantics).",
    )
    parser.add_argument(
        "--cell-note-for",
        default="",
        help="Optional comma-separated filename stems to apply --cell-note to. Empty means all selected tables.",
    )
    parser.add_argument(
        "--cell-note-raw",
        action="store_true",
        help="Treat --cell-note as raw LaTeX (no escaping).",
    )
    parser.add_argument(
        "--tabcolsep",
        default="",
        help="Optional LaTeX tabcolsep value (e.g., 3.6pt).",
    )
    parser.add_argument(
        "--na-token",
        default="--",
        help="Token used for missing values in rendered tables (default: --).",
    )
    parser.add_argument(
        "--fixed-float-format",
        action="store_true",
        help="Keep trailing zeros for non-integer numeric columns (e.g., 10.0 instead of 10).",
    )
    parser.add_argument(
        "--numeric-align",
        choices=("r", "c"),
        default="r",
        help="Alignment for numeric columns (default: r).",
    )
    parser.add_argument(
        "--add-vertical-rules",
        action="store_true",
        help="Add vertical rules between all columns in the LaTeX tabular format.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise SystemExit(f"Missing input dir: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    include_set = set(_split_csv_arg(args.include))
    exclude_set = set(_split_csv_arg(args.exclude))
    cell_note_for = set(_split_csv_arg(args.cell_note_for))
    caption_map: Dict[str, str] = {}
    if args.caption_map.strip():
        caption_map = _load_caption_map(Path(args.caption_map))

    csv_paths = _iter_csv_paths(input_dir, recursive=bool(args.recursive))
    rows: List[Dict[str, object]] = []
    all_tex_chunks: List[str] = []

    for csv_path in csv_paths:
        stem = csv_path.stem
        fname = csv_path.name
        if include_set and stem not in include_set and fname not in include_set:
            continue
        if stem in exclude_set or fname in exclude_set:
            continue

        df = pd.read_csv(csv_path)
        original_rows = len(df)
        if args.max_rows and args.max_rows > 0:
            df = df.head(args.max_rows).copy()

        caption = caption_map.get(fname) or caption_map.get(stem) or _title_from_stem(stem)
        label = _sanitize_label(stem=stem, prefix=args.label_prefix)
        apply_cell_note = bool(args.cell_note.strip()) and (
            not cell_note_for or stem in cell_note_for or fname in cell_note_for
        )
        cell_note = args.cell_note if apply_cell_note else ""
        tex_text = _build_table_tex(
            df=df,
            caption=caption,
            label=label,
            precision=max(0, int(args.float_precision)),
            sci_thresh=max(1, int(args.scientific_threshold)),
            na_token=str(args.na_token),
            use_resizebox=not bool(args.no_resizebox),
            table_position=args.table_position,
            cell_note=cell_note,
            cell_note_raw=bool(args.cell_note_raw),
            tabcolsep=str(args.tabcolsep),
            strip_trailing_zeros=not bool(args.fixed_float_format),
            numeric_align=str(args.numeric_align),
            add_vertical_rules=bool(args.add_vertical_rules),
        )

        out_tex = output_dir / f"{stem}.tex"
        out_tex.write_text(tex_text)
        all_tex_chunks.append(tex_text.rstrip())
        rows.append(
            {
                "csv_file": str(csv_path),
                "tex_file": str(out_tex),
                "caption": caption,
                "label": label,
                "n_rows_original": original_rows,
                "n_rows_written": len(df),
                "n_cols": len(df.columns),
            }
        )

    manifest = pd.DataFrame(rows)
    manifest_path = output_dir / "latex_tables_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    include_all_path = output_dir / "include_all_tables.tex"
    include_all_path.write_text("\n\n".join(all_tex_chunks).strip() + ("\n" if all_tex_chunks else ""))

    print(f"Wrote {len(rows)} LaTeX tables to {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Combined include: {include_all_path}")


if __name__ == "__main__":
    main()
