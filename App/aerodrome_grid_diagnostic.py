from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable

import pymupdf

# Reuse the FLIP custom-font decoder already used by the orientation stage.
from orientation_turn import _collect_text_lines


# Supports both decimal minutes (40° 19.09' N) and DMS
# (40°19'00'' N), including the FLIP font-decoded quote variants.
_COORD_RE = re.compile(
    r"(?<!\d)(\d{1,3})\s*[°º˚]\s*(\d{1,2}(?:[.,]\d+)?)"
    r"(?:\s*['′’]\s*(\d{1,2}(?:[.,]\d+)?)\s*(?:[\"″”]{1,2}|['′’]{1,2})?)?"
    r"\s*['′’]?\s*([NSEW])\b",
    re.IGNORECASE,
)


def _expanded_roi(roi: pymupdf.Rect, page: pymupdf.Page) -> pymupdf.Rect:
    """Include the chart-edge coordinate labels just outside the neatline."""
    margin_x = page.rect.width * 0.045
    margin_y = page.rect.height * 0.045
    return pymupdf.Rect(
        max(page.rect.x0, roi.x0 - margin_x),
        max(page.rect.y0, roi.y0 - margin_y),
        min(page.rect.x1, roi.x1 + margin_x),
        min(page.rect.y1, roi.y1 + margin_y),
    )


def _visible_segments(page: pymupdf.Page, horizontal: bool, roi: pymupdf.Rect):
    """Extract only vector segments crossing the ROI, in visible-page axes."""
    matrix = page.rotation_matrix
    found = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", ()):
            if not item or item[0] != "l":
                continue
            p1, p2 = item[1] * matrix, item[2] * matrix
            dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
            if horizontal and dy <= 1.6:
                fixed = (p1.y + p2.y) / 2
                start, end = min(p1.x, p2.x), max(p1.x, p2.x)
                if roi.y0 <= fixed <= roi.y1 and end >= roi.x0 and start <= roi.x1:
                    found.append((fixed, max(start, roi.x0), min(end, roi.x1)))
            elif not horizontal and dx <= 1.6:
                fixed = (p1.x + p2.x) / 2
                start, end = min(p1.y, p2.y), max(p1.y, p2.y)
                if roi.x0 <= fixed <= roi.x1 and end >= roi.y0 and start <= roi.y1:
                    found.append((fixed, max(start, roi.y0), min(end, roi.y1)))
    return found


def _merge_segments(segments, tolerance=2.2, max_gap=24.0):
    """Merge fragments only when they are collinear and spatially adjacent."""
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
            clusters.append([fixed, start, end, 1])
        else:
            count = match[3]
            match[0] = (match[0] * count + fixed) / (count + 1)
            match[1] = min(match[1], start)
            match[2] = max(match[2], end)
            match[3] += 1
    return [(row[0], row[1], row[2]) for row in clusters]


def _coordinate_labels(page: pymupdf.Page, roi: pymupdf.Rect):
    """Decode coordinate labels only in/along the chart ROI."""
    labels = []
    for line in _collect_text_lines(page, roi):
        for match in _COORD_RE.finditer(line["text"]):
            degrees = float(match.group(1))
            minutes = float(match.group(2).replace(",", "."))
            seconds = float((match.group(3) or "0").replace(",", "."))
            value = degrees + minutes / 60.0 + seconds / 3600.0
            hemisphere = match.group(4).upper()
            if hemisphere in ("S", "W"):
                value = -value
            rect = line["rect"]
            labels.append({
                "text": match.group(0).strip(),
                "axis": "LAT" if hemisphere in ("N", "S") else "LONG",
                "hemisphere": hemisphere,
                "value": value,
                "x": (rect.x0 + rect.x1) / 2,
                "y": (rect.y0 + rect.y1) / 2,
            })
    # Remove duplicate labels emitted by adjacent text spans.
    unique = {}
    for label in labels:
        key = (label["axis"], round(label["value"], 7), round(label["x"], 1), round(label["y"], 1))
        unique[key] = label
    return list(unique.values())


def _format_coordinate(value: float, axis: str, hemisphere_hint: str = "") -> str:
    hemisphere = hemisphere_hint or (
        ("N" if value >= 0 else "S") if axis == "LAT"
        else ("E" if value >= 0 else "W")
    )
    absolute = abs(value)
    degrees = int(absolute)
    minutes_float = (absolute - degrees) * 60.0
    minutes = int(minutes_float)
    seconds = int(round((minutes_float - minutes) * 60.0))
    if seconds >= 60:
        seconds = 0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        degrees += 1
    return f"{degrees:02d} {minutes:02d} {seconds:02d} {hemisphere}"


def _match_grid_lines(segments, labels, axis, roi):
    """Accept a vector line only when an axis-matching coordinate label anchors it."""
    horizontal = axis == "LAT"
    dimension = roi.height if horizontal else roi.width
    min_length = dimension * 0.42
    max_distance = max(22.0, dimension * 0.035)
    candidates = [
        segment for segment in segments
        if segment[2] - segment[1] >= min_length
    ]
    accepted = []
    for label in (item for item in labels if item["axis"] == axis):
        nearby = []
        for segment in candidates:
            fixed, start, end = segment
            along = label["x"] if horizontal else label["y"]
            if start - 28.0 <= along <= end + 28.0:
                distance = abs((label["y"] if horizontal else label["x"]) - fixed)
                if distance <= max_distance:
                    nearby.append((distance, segment))
        if not nearby:
            continue
        distance, segment = min(nearby, key=lambda item: item[0])
        existing = next(
            (item for item in accepted if abs(item[0] - segment[0]) <= 2.5),
            None,
        )
        record = (segment[0], segment[1], segment[2], label["text"], label["value"], label["hemisphere"])
        if existing is None:
            accepted.append(record)
        elif distance < abs(existing[0] - (label["y"] if horizontal else label["x"])):
            accepted[accepted.index(existing)] = record
    accepted.sort(key=lambda item: item[0])
    return accepted


def _median(values):
    values = sorted(values)
    if not values:
        return None
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0


def _calculated_complement(candidates, width, height, roi, labels):
    """Fill a single-line axis using the confirmed spacing of the other axis."""
    lat, lon = candidates["LAT"], candidates["LONG"]
    missing_axis = "LAT" if len(lat) == 1 and len(lon) >= 2 else (
        "LONG" if len(lon) == 1 and len(lat) >= 2 else None
    )
    if missing_axis is None:
        return [], None

    known = lon if missing_axis == "LAT" else lat
    positions = [item[0] for item in known]
    pixel_intervals = [abs(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)]
    pixel_step = _median(pixel_intervals)
    if not pixel_step or pixel_step <= 0:
        return [], None

    value_intervals = [
        abs(known[i + 1][4] - known[i][4])
        for i in range(len(known) - 1)
        if known[i][4] is not None and known[i + 1][4] is not None
    ]
    angular_step = _median(value_intervals)
    if not angular_step or angular_step <= 0:
        # Standard chart graticule interval fallback: one minute.
        angular_step = 1.0 / 60.0

    lat_values = [item["value"] for item in labels if item["axis"] == "LAT"]
    mean_lat = abs(sum(lat_values) / len(lat_values)) if lat_values else 0.0
    cosine = max(0.15, math.cos(math.radians(mean_lat)))

    # At equal angular intervals, longitude spacing contracts by cos(latitude).
    if missing_axis == "LAT":
        spacing = pixel_step / cosine
        anchor = lat[0]
        fixed_min, fixed_max = 0.0, height
    else:
        spacing = pixel_step * cosine
        anchor = lon[0]
        fixed_min, fixed_max = 0.0, width

    if spacing <= 0:
        return [], None
    anchor_value = anchor[4]
    hemisphere = anchor[5]
    calculated = []
    step_count = 1
    while anchor[0] - step_count * spacing > fixed_min:
        position = anchor[0] - step_count * spacing
        value = (anchor_value + step_count * angular_step) if missing_axis == "LAT" else (anchor_value - step_count * angular_step)
        calculated.append((position, 0.0, width if missing_axis == "LAT" else height,
                           _format_coordinate(value, missing_axis, hemisphere), value, hemisphere))
        step_count += 1
    step_count = 1
    while anchor[0] + step_count * spacing < fixed_max:
        position = anchor[0] + step_count * spacing
        value = (anchor_value - step_count * angular_step) if missing_axis == "LAT" else (anchor_value + step_count * angular_step)
        calculated.append((position, 0.0, width if missing_axis == "LAT" else height,
                           _format_coordinate(value, missing_axis, hemisphere), value, hemisphere))
        step_count += 1

    calculated.sort(key=lambda item: item[0])
    return calculated, {
        "axis": missing_axis,
        "spacing_pt": spacing,
        "angular_step_deg": angular_step,
        "latitude_used": mean_lat,
        "derived_from": "LONG" if missing_axis == "LAT" else "LAT",
    }


def detect_page_grid_candidates(page: pymupdf.Page, roi: pymupdf.Rect):
    """Stage 02: ROI-anchored LAT/LONG detection; never classify arbitrary page lines."""
    expanded = _expanded_roi(roi, page)
    labels = _coordinate_labels(page, expanded)
    horizontal = _merge_segments(
        _visible_segments(page, True, roi),
        max_gap=max(18.0, roi.width * 0.07),
    )
    vertical = _merge_segments(
        _visible_segments(page, False, roi),
        max_gap=max(18.0, roi.height * 0.07),
    )
    lat = _match_grid_lines(horizontal, labels, "LAT", roi)
    lon = _match_grid_lines(vertical, labels, "LONG", roi)
    calculated, calculation = _calculated_complement(
        {"LAT": lat, "LONG": lon}, float(page.rect.width), float(page.rect.height), roi, labels
    )
    return {
        "LAT": lat,
        "LONG": lon,
        "CALCULATED": calculated,
        "labels": labels,
        "calculation": calculation,
        "roi": roi,
    }


def save_grid_diagnostic(source_pdf: Path, output_image: Path, roi_data: dict):
    doc = pymupdf.open(source_pdf)
    try:
        page = doc[0]
        roi = pymupdf.Rect(
            float(roi_data["x0"]), float(roi_data["y0"]),
            float(roi_data["x1"]), float(roi_data["y1"]),
        )
        candidates = detect_page_grid_candidates(page, roi)
        width, height = float(page.rect.width), float(page.rect.height)
        derotation = page.derotation_matrix

        def raw_point(x, y):
            return pymupdf.Point(x, y) * derotation

        def draw_line(p1, p2, **kwargs):
            page.draw_line(raw_point(*p1), raw_point(*p2), **kwargs)

        def insert_text(point, text, **kwargs):
            page.insert_text(raw_point(*point), text, **kwargs)

        # Confirmed grid lines: red LAT, blue LONG. Each is extended to page edges.
        for axis, color in (("LAT", (1, 0, 0)), ("LONG", (0, 0.25, 1))):
            for index, item in enumerate(candidates[axis], 1):
                fixed, _start, _end, _label, value, hemisphere = item
                caption = _format_coordinate(value, axis, hemisphere) if value is not None else _label
                if axis == "LAT":
                    draw_line((0, fixed), (width, fixed), color=color, width=1.35, overlay=True)
                    insert_text((5, max(10, fixed - 4)), caption, fontsize=7, color=color, overlay=True)
                else:
                    draw_line((fixed, 0), (fixed, height), color=color, width=1.35, overlay=True)
                    insert_text((fixed + 3, 12), caption, fontsize=7, color=color, overlay=True)

        # Calculated grid lines: green, with extrapolated DMS coordinate values.
        for item in candidates["CALCULATED"]:
            fixed, _start, _end, caption, _value, _hemisphere = item
            if candidates["calculation"]["axis"] == "LAT":
                draw_line((0, fixed), (width, fixed), color=(0.1, 0.65, 0.1), width=1.5, overlay=True)
                insert_text((5, max(10, fixed - 4)), caption, fontsize=7, color=(0.1, 0.65, 0.1), overlay=True)
            else:
                draw_line((fixed, 0), (fixed, height), color=(0.1, 0.65, 0.1), width=1.5, overlay=True)
                insert_text((fixed + 3, 12), caption, fontsize=7, color=(0.1, 0.65, 0.1), overlay=True)

        output_image.parent.mkdir(parents=True, exist_ok=True)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
        pix.save(output_image)
        return {
            "LAT": len(candidates["LAT"]),
            "LONG": len(candidates["LONG"]),
            "CALCULATED": len(candidates["CALCULATED"]),
            "labels": len(candidates["labels"]),
            "calculation": candidates["calculation"],
        }
    finally:
        doc.close()


def run_aerodrome_grid_diagnostics(input_dir: Path, output_dir: Path,
                                   charts: Iterable[object]):
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = input_dir / "roi_manifest.json"
    if not manifest_path.is_file():
        return []
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = []
    for chart in charts:
        if str(getattr(chart, "procedure", "")).strip().upper() != "AERODROME DIAGRAM":
            continue
        filename = str(getattr(chart, "filename", ""))
        source = input_dir / filename
        record = records.get(filename)
        roi_data = (record or {}).get("roi")
        if not source.is_file() or not roi_data:
            continue
        target = output_dir / f"{source.stem}_grid-diagnostic.png"
        counts = save_grid_diagnostic(source, target, roi_data)
        results.append((source.name, counts))
    return results
