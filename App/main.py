from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from cover_metadata import read_cover_metadata
from flip_parser import inspect_pdf


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a ForeFlight/TurAF FLIP PDF and create a chart-page manifest."
    )
    parser.add_argument("pdf", type=Path, help="Source FLIP PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "HEDEF" / "Initial",
        help="Folder for the analysis manifest (default: HEDEF/Initial)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.pdf.expanduser().resolve()
    if not source.is_file():
        print(f"ERROR: PDF not found: {source}", file=sys.stderr)
        return 2

    try:
        metadata = read_cover_metadata(source)
        charts, unresolved = inspect_pdf(source, metadata.effective_date, metadata.document_type)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "chart_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("pdf_page", "chart_number", "airport", "category", "procedure", "filename"))
        for chart in charts:
            writer.writerow((
                chart.pdf_page, chart.chart_no, chart.airport,
                chart.category, chart.procedure, chart.filename,
            ))

    print(f"Document type : {metadata.document_type}")
    print(f"Effective     : {metadata.effective_from} TO {metadata.effective_to}")
    print(f"Filename date : {metadata.effective_date}")
    print(f"Chart pages   : {len(charts)}")
    print(f"Unresolved    : {len(unresolved)}")
    print(f"Manifest      : {manifest}")
    for page_number, reason in unresolved:
        print(f"REVIEW page {page_number}: {reason}")
    return 0 if not unresolved else 3


if __name__ == "__main__":
    raise SystemExit(main())
