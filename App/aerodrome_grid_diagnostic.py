from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable

import pymupdf

# Reuse the FLIP custom-font decoder already used by the orientation stage.
from orientation_turn import _collect_text_lines


_COORD_RE = re.compile(
    r"(?<!\\d)(\\d{1,2})(?:\\s*[°º˚]\\s*|\\s+)(\\d{1,2}(?:[.,]\\d+)?)\\s*['′’]?\\s*([NSEW])\\b",
    re.IGNORECASE,
)


def _visible_segments(page: pymupdf.Page, horizontal: bool):
    """Read vector geometry from the final Stage-01 PDF in its visible page axes.

    PyMuPDF exposes drawing points in the PDF's unrotated coordinate system.
    Applying page.rotation_matrix here only maps coordinates to the already-
    rotated page view; it does not rotate, rewrite, or otherwise modify the PDF.
    """
    matrix = page.rotation_matrix
    found = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", ()):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1] * matrix, item[2] * matrix
            dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
            if horizontal and dy <= 1.6 and dx >= 5:
                found.append(((p1.y + p2.y) / 2, min(p1.x, p2.x), max(p1.x, p2.x)))
            elif not horizontal and dx <= 1.6 and dy >= 5:
                found.append(((p1.x + p2.x) / 2, min(p1.y, p2.y), max(p1.y, p2.y)))
    return found


def _merge_segments(segments, tolerance=2.2, max_gap=36.0):
    """Cluster collinear fragments and bridge table/artwork interruptions."""
    clusters = []
    for fixed, start, end in sorted(segments, key=lambda item: (item[0], item[1])):
        match = next(
            (row for row in reversed(clusters)
             if abs(row[0] - fixed) <= tolerance
             and start <= row[2] + max_gap
             and end >= row[1] - max_gap),
            None,
        )
        if match is None:
            clusters.append([fixed, start, end])
        else:
            count = match[3]
            match[0] = (match[0] * count + fixed) / (count + 1)
            match[1] = min(match[1], start)
            match[2] = max(match[2], end)
            match[3] += 1
    return [(r[0], r[1], r[2]) for r in clusters]


def _coordinate_labels(page: pymupdf.Page):
    """Return decoded coordinate labels in visible-page coordinates."""
    labels = []
    # _collect_text_lines handles FLIP fonts without ToUnicode and page rotation.
    for line in _collect_text_lines(page, page.rect):
        text = line["text"]
        for match in _COORD_RE.finditer(text):
            degrees = float(match.group(1)) + float(match.group(2).replace(",", ".")) / 60.0
            hemisphere = match.group(3).upper()
            if hemisphere in ("S", "W"):
                degrees = -degrees
            rect = line["rect"]
            labels.append({
                "text": match.group(0).strip(),
                "axis": "LAT" if hemisphere in ("N", "S") else "LONG",
                "value": degrees,
                "x": (rect.x0 + rect.x1) / 2,
                "y": (rect.y0 + rect.y1) / 2,
            })
    return labels


def _attach_labels(candidates, labels, axis):
    for index, (fixed, start, end) in enumerate(candidates):
        eligible = [label for label in labels if label["axis"] == axis]
        if axis == "LAT":
            nearby = [v for v in eligible if start - 30 <= v["x"] <= end + 30]
            nearby.sort(key=lambda v: abs(v["y"] - fixed))
        else:
            nearby = [v for v in eligible if start - 30 <= v["y"] <= end + 30]
            nearby.sort(key=lambda v: abs(v["x"] - fixed))
        if nearby:
            candidates[index] = (fixed, start, end, nearby[0]["text"], nearby[0]["value"])
        else:
            candidates[index] = (fixed, start, end, "", None)
    return candidates


def _calculated_complement(candidates, width, height, labels):
    """Estimate missing LAT lines from LONG spacing (or vice versa).

    This is diagnostic-only. The estimate uses equal physical grid spacing and
    the cosine correction at the mean labelled latitude. It is emitted only
    when one axis has a single detected line and the other has at least two.
    """
    lat, lon = candidates["LAT"], candidates["LONG"]
    if len(lat) == 1 and len(lon) >= 2:
        intervals = [abs(lon[i + 1][0] - lon[i][0]) for i in range(len(lon) - 1)]
        pixel_spacing = sorted(intervals)[len(intervals) // 2]
        lat_values = [v["value"] for v in labels if v["axis"] == "LAT"]
        latitude = sum(lat_values) / len(lat_values) if lat_values else 0.0
        spacing = pixel_spacing / max(0.15, math.cos(math.radians(abs(latitude))))
        anchor = lat[0][0]
        calculated = []
        y = anchor - spacing
        while y > height * 0.04:
            calculated.append((y, 0.0, width, "CALCULATED", None))
            y -= spacing
        y = anchor + spacing
        while y < height * 0.96:
            calculated.append((y, 0.0, width, "CALCULATED", None))
            y += spacing
        return calculated, {"axis": "LAT", "spacing_pt": spacing, "latitude_used": latitude}
    if len(lon) == 1 and len(lat) >= 2:
        intervals = [abs(lat[i + 1][0] - lat[i][0]) for i in range(len(lat) - 1)]
        pixel_spacing = sorted(intervals)[len(intervals) // 2]
        lat_values = [v["value"] for v in labels if v["axis"] == "LAT"]
        latitude = sum(lat_values) / len(lat_values) if lat_values else 0.0
        spacing = pixel_spacing * max(0.15, math.cos(math.radians(abs(latitude))))
        anchor = lon[0][0]
        calculated = []
        x = anchor - spacing
        while x > width * 0.04:
            calculated.append((x, 0.0, height, "CALCULATED", None))
            x -= spacing
        x = anchor + spacing
        while x < width * 0.96:
            calculated.append((x, 0.0, height, "CALCULATED", None))
            x += spacing
        return calculated, {"axis": "LONG", "spacing_pt": spacing, "latitude_used": latitude}
    return [], None


def detect_page_grid_candidates(page: pymupdf.Page):
    """Stage 02: visible horizontal vectors are LAT; visible vertical are LONG."""
    width, height = float(page.rect.width), float(page.rect.height)
    merge_gap = max(36.0, max(width, height) * 0.08)
    lat = _merge_segments(_visible_segments(page, True), max_gap=merge_gap)
    lon = _merge_segments(_visible_segments(page, False), max_gap=merge_gap)

    # Reject only page-frame neighbourhoods. Do not require a line to span
    # half the page: interrupted and partially obscured grid vectors are valid.
    lat = [v for v in lat if height * 0.035 < v[0] < height * 0.965
           and v[2] - v[1] >= width * 0.18]
    lon = [v for v in lon if width * 0.035 < v[0] < width * 0.965
           and v[2] - v[1] >= height * 0.18]

    labels = _coordinate_labels(page)
    lat = _attach_labels(sorted(lat), labels, "LAT")
    lon = _attach_labels(sorted(lon), labels, "LONG")
    calculated, calculation = _calculated_complement(
        {"LAT": lat, "LONG": lon}, width, height, labels
    )
    return {
        "LAT": lat,
        "LONG": lon,
        "CALCULATED": calculated,
        "labels": labels,
        "calculation": calculation,
    }


def save_grid_diagnostic(source_pdf: Path, output_image: Path):
    doc = pymupdf.open(source_pdf)
    try:
        page = doc[0]
        candidates = detect_page_grid_candidates(page)
        width, height = float(page.rect.width), float(page.rect.height)

        # Drawing APIs use unrotated PDF coordinates. Map visible geometry back
        # for annotation only; this does not rotate or modify the source PDF.
        derotation = page.derotation_matrix

        def raw_point(x, y):
            return pymupdf.Point(x, y) * derotation

        def draw_visible_line(p1, p2, **kwargs):
            page.draw_line(raw_point(*p1), raw_point(*p2), **kwargs)

        def insert_visible_text(point, text, **kwargs):
            page.insert_text(raw_point(*point), text, **kwargs)

        # Extend every accepted grid candidate over the complete visible page.
        for axis, color in (("LAT", (1, 0, 0)), ("LONG", (0, 0.25, 1))):
            for index, item in enumerate(candidates[axis], 1):
                fixed, _start, _end, label, _value = item
                if axis == "LAT":
                    draw_visible_line((0, fixed), (width, fixed), color=color, width=1.25, overlay=True)
                    caption = label or f"LAT candidate {index}"
                    insert_visible_text((5, max(10, fixed - 3)), caption,
                                     fontsize=7, color=color, overlay=True)
                else:
                    draw_visible_line((fixed, 0), (fixed, height), color=color, width=1.25, overlay=True)
                    caption = label or f"LONG candidate {index}"
                    insert_visible_text((fixed + 3, 12), caption,
                                     fontsize=7, color=color, overlay=True)

        for index, item in enumerate(candidates["CALCULATED"], 1):
            fixed, _start, _end, _label, _value = item
            if candidates["calculation"]["axis"] == "LAT":
                draw_visible_line((0, fixed), (width, fixed), color=(0.1, 0.65, 0.1),
                               width=1.5, overlay=True)
                insert_visible_text((5, max(10, fixed - 3)), f"LAT CALCULATED {index}",
                                 fontsize=7, color=(0.1, 0.65, 0.1), overlay=True)
            else:
                draw_visible_line((fixed, 0), (fixed, height), color=(0.1, 0.65, 0.1),
                               width=1.5, overlay=True)
                insert_visible_text((fixed + 3, 12), f"LONG CALCULATED {index}",
                                 fontsize=7, color=(0.1, 0.65, 0.1), overlay=True)

        output_image.parent.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        pix.save(output_image)
        return {
            "LAT": len(candidates["LAT"]),
            "LONG": len(candidates["LONG"]),
            "CALCULATED": len(candidates["CALCULATED"]),
            "labels": len(candidates["labels"]),
        }
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
        results.append((source.name, counts))
    return results
