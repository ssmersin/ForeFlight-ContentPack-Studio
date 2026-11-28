from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pymupdf


def _segments(page: pymupdf.Page, horizontal: bool):
    found = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", ()):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1], item[2]
            dx, dy = abs(p2.x-p1.x), abs(p2.y-p1.y)
            if horizontal and dy <= 1.25 and dx >= 8:
                found.append(((p1.y+p2.y)/2, min(p1.x,p2.x), max(p1.x,p2.x)))
            elif not horizontal and dx <= 1.25 and dy >= 8:
                found.append(((p1.x+p2.x)/2, min(p1.y,p2.y), max(p1.y,p2.y)))
    return found


def _merge_segments(segments, tolerance=1.8, max_gap=14.0):
    ordered = sorted(segments, key=lambda item: (item[0], item[1]))
    rows = []
    for fixed, start, end in ordered:
        match = next((row for row in reversed(rows)
                      if abs(row[0]-fixed) <= tolerance and start <= row[2]+max_gap
                      and end >= row[1]-max_gap), None)
        if match is None:
            rows.append([fixed, start, end])
        else:
            match[0] = (match[0]+fixed)/2
            match[1] = min(match[1], start)
            match[2] = max(match[2], end)
    return [(r[0], r[1], r[2]) for r in rows]


def detect_page_grid_candidates(page: pymupdf.Page):
    """After Second-Turn: horizontal vectors are LAT; vertical vectors are LONG."""
    width, height = float(page.rect.width), float(page.rect.height)
    lat = _merge_segments(_segments(page, True))
    long = _merge_segments(_segments(page, False))
    lat = [v for v in lat if v[2]-v[1] >= width*0.50 and 4 < v[0] < height-4]
    long = [v for v in long if v[2]-v[1] >= height*0.50 and 4 < v[0] < width-4]
    return {"LAT": sorted(lat), "LONG": sorted(long)}


def save_grid_diagnostic(source_pdf: Path, output_image: Path):
    doc = pymupdf.open(source_pdf)
    try:
        page = doc[0]
        candidates = detect_page_grid_candidates(page)
        for axis, color in (("LAT", (1, 0, 0)), ("LONG", (0, 0.25, 1))):
            for index, (fixed, start, end) in enumerate(candidates[axis], 1):
                if axis == "LAT":
                    p1, p2 = (start, fixed), (end, fixed)
                    label = (max(8, start+3), max(12, fixed-3))
                else:
                    p1, p2 = (fixed, start), (fixed, end)
                    label = (fixed+3, max(12, start+8))
                page.draw_line(p1, p2, color=color, width=1.5, overlay=True)
                page.insert_text(label, f"{axis} candidate {index}",
                                 fontsize=6, color=color, overlay=True)
        output_image.parent.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        pix.save(output_image)
        return {axis: len(lines) for axis, lines in candidates.items()}
    finally:
        doc.close()


def run_aerodrome_grid_diagnostics(input_dir: Path, output_dir: Path,
                                   charts: Iterable[object]):
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for chart in charts:
        if str(getattr(chart, "procedure", "")).strip().upper() != "AERODROME DIAGRAM":
            continue
        source = input_dir / str(getattr(chart, "filename", ""))
        if not source.is_file():
            continue
        target = output_dir / f"{source.stem}_grid-diagnostic.png"
        counts = save_grid_diagnostic(source, target)
        results.append((source.name, counts["LAT"], counts["LONG"]))
    return results
