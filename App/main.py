from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from cover_metadata import read_cover_metadata
from flip_parser import inspect_pdf
from pdf_splitter import split_chart_pages
from roi_orientation_pipeline import run_combined_roi_orientation_stage
from aerodrome_grid_diagnostic import run_aerodrome_grid_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a ForeFlight/TurAF FLIP PDF and export each chart page as a separate PDF."
    )
    parser.add_argument("pdf", type=Path, help="Source FLIP PDF")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "HEDEF" / "00-Initial",
        help="Folder for the manifest and initial chart PDFs (default: HEDEF/00-Initial)",
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Export ROI and orientation diagnostic images",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.pdf.expanduser().resolve()
    if not source.is_file():
        print(f"ERROR: PDF not found: {source}", file=sys.stderr)
        return 2

    hedef_root = Path(__file__).resolve().parent.parent / "HEDEF"
    initial_dir = args.output.expanduser().resolve()
    first_dir = hedef_root / "01-Rotation"
    diagnostic_dir = hedef_root / "01-Rotation-Diagnostic"
    turn_diag_dir = hedef_root / "01-Orientation-Diagnostic"

    try:
        metadata = read_cover_metadata(source)
        charts, unresolved = inspect_pdf(source, metadata.effective_date, metadata.document_type)
        initial_dir.mkdir(parents=True, exist_ok=True)
        manifest = initial_dir / "chart_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(("pdf_page", "chart_number", "airport", "category", "procedure", "filename"))
            for chart in charts:
                writer.writerow((
                    chart.pdf_page, chart.chart_no, chart.airport,
                    chart.category, chart.procedure, chart.filename,
                ))

        generated = split_chart_pages(source, charts, initial_dir)
        (
            first_exports,
            diag_exports,
            turn_diag_exports,
            skipped_first,
            rotated_items,
        ) = run_combined_roi_orientation_stage(
            initial_dir=initial_dir,
            output_dir=first_dir,
            charts=charts,
            diagnostic=args.diagnostic,
            roi_diagnostic_dir=diagnostic_dir if args.diagnostic else None,
            orientation_diagnostic_dir=turn_diag_dir if args.diagnostic else None,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Document type : {metadata.document_type}")
    print(f"Effective     : {metadata.effective_from} TO {metadata.effective_to}")
    print(f"Filename date : {metadata.effective_date}")
    print(f"Chart pages   : {len(charts)}")
    print(f"PDF exports   : {len(generated)}")
    print(f"01-Rotation: {len(first_exports)} (skipped: {len(skipped_first)})")
    for name in skipped_first:
        print(f"00-INITIAL REMAINED (no closed frame): {name}")
    if args.diagnostic:
        print(f"01-Rotation-Diag: {len(diag_exports)}")
    grid_diagnostics = []
    if args.diagnostic:
        grid_diagnostics = run_aerodrome_grid_diagnostics(
            input_dir=first_dir,
            output_dir=hedef_root / "02-GeoPDF-Diagnostic",
            charts=charts,
        )
        print(f"02-GeoPDF-Diagnostic: {len(grid_diagnostics)}")
        for name, counts in grid_diagnostics:
            print(
                f"  GRID {name}: LAT={counts['LAT']}, LONG={counts['LONG']}, "
                f"CALCULATED={counts['CALCULATED']}, COORD-LABELS={counts['labels']}"
            )

    print(f"Orientation rotated: {len(rotated_items)}")
    for fname, rot_deg, method in rotated_items:
        print(f"  ROTATED {rot_deg}° CW [{method}]: {fname}")
    if args.diagnostic:
        print(f"01-Orientation-Diag: {len(turn_diag_exports)}")
    print(f"Unresolved    : {len(unresolved)}")
    print(f"Initial folder: {initial_dir}")
    print(f"Stage 01 (Rotation) folder: {first_dir}")
    print(f"Chart manifest: {manifest.resolve()}")
    print(f"ROI manifest  : {(first_dir / 'roi_manifest.json').resolve()}")
    for page_number, reason in unresolved:
        print(f"REVIEW page {page_number}: {reason}")
    return 0 if not unresolved else 3


if __name__ == "__main__":
    raise SystemExit(main())