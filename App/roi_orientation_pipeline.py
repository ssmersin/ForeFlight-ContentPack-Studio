from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pymupdf

from planview_roi import (
    BANDED_PLANVIEW_CATEGORIES,
    _is_single_frame_chart,
    _save_diagnostic_image,
    _save_roi_manifest,
    detect_planview_roi,
)
from orientation_turn import (
    _save_turn_diagnostic,
    _transform_roi_after_cw_rotation,
    detect_page_orientation,
)


def run_combined_roi_orientation_stage(
    initial_dir: Path,
    output_dir: Path,
    charts: Iterable[object],
    diagnostic: bool = False,
    roi_diagnostic_dir: Path | None = None,
    orientation_diagnostic_dir: Path | None = None,
):
    """Run ROI detection and orientation consecutively on one open PDF document.

    The source PDF is opened once. ROI and orientation remain separate modules;
    this module only coordinates their in-memory page processing and final save.
    """
    selected = []
    for chart in charts:
        category = str(getattr(chart, "category", ""))
        procedure = str(getattr(chart, "procedure", ""))
        if _is_single_frame_chart(category, procedure) or category.upper() in BANDED_PLANVIEW_CATEGORIES:
            selected.append((chart, category, procedure))

    if not selected:
        return [], [], [], [], []

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "roi_manifest.json"
    records: dict[str, dict] = {}
    if manifest_path.is_file():
        records = json.loads(manifest_path.read_text(encoding="utf-8"))

    moved: list[Path] = []
    roi_diagnostics: list[Path] = []
    orientation_diagnostics: list[Path] = []
    skipped: list[str] = []
    rotated_items: list[tuple[str, int, str]] = []

    for chart, category, procedure in selected:
        filename = chart.filename
        src_pdf = initial_dir / filename
        if not src_pdf.is_file():
            continue

        dst_pdf = output_dir / filename
        doc = pymupdf.open(src_pdf)
        try:
            page = doc[0]
            original_width = float(page.rect.width)
            original_height = float(page.rect.height)

            # 01: determine ROI while the original page is open.
            roi = detect_planview_roi(page, category=category, procedure=procedure)
            if roi is None:
                skipped.append(filename)
                continue

            # 02: immediately analyze orientation on the same page object.
            orientation = detect_page_orientation(page, roi, category, procedure)
            effective_rotation = int(orientation.rotation_cw) % 360 if orientation.apply_rotation else 0

            # Diagnostics render copies/in-memory pixels and do not alter the source page.
            if diagnostic and roi_diagnostic_dir is not None:
                roi_path = roi_diagnostic_dir / f"{Path(filename).stem}.jpg"
                _save_diagnostic_image(page, roi, roi_path)
                if roi_path.is_file():
                    roi_diagnostics.append(roi_path)

            if effective_rotation:
                page.set_rotation((int(page.rotation) + effective_rotation) % 360)

            if diagnostic and orientation_diagnostic_dir is not None:
                turn_path = orientation_diagnostic_dir / f"{Path(filename).stem}.jpg"
                _save_turn_diagnostic(
                    page, roi, original_width, original_height, orientation, turn_path
                )
                if turn_path.is_file():
                    orientation_diagnostics.append(turn_path)

            if effective_rotation:
                doc.save(str(dst_pdf), garbage=4, deflate=True)
        finally:
            doc.close()

        if effective_rotation:
            src_pdf.unlink()
            rotated_items.append((
                filename,
                effective_rotation,
                f"{orientation.method}/{orientation.basis}/{orientation.var_check}",
            ))
        else:
            if dst_pdf.exists():
                dst_pdf.unlink()
            shutil.move(str(src_pdf), str(dst_pdf))

        new_roi, new_width, new_height = _transform_roi_after_cw_rotation(
            asdict(roi), original_width, original_height, effective_rotation
        )
        record = {
            "procedure": procedure,
            "category": category,
            "page_width": new_width,
            "page_height": new_height,
            "original_roi": asdict(roi),
            "roi": new_roi,
            "orientation": {
                "rotation_cw": effective_rotation,
                "proposed_rotation_cw": int(orientation.rotation_cw) % 360,
                "applied": bool(effective_rotation),
                "method": orientation.method,
                "basis": orientation.basis,
                "var_text": orientation.var_text,
                "year": orientation.year,
                "var_check": orientation.var_check,
                "north_angle_deg": orientation.north_angle_deg,
                "residual_deg": orientation.residual_deg,
                "text_lines_in_roi": orientation.text_lines,
                "review_required": not bool(effective_rotation) and (
                    "CONFLICT" in orientation.var_check.upper()
                    or "NOT_FOUND" in orientation.var_check.upper()
                    or orientation.method == "none"
                ),
            },
        }
        records[filename] = record
        moved.append(dst_pdf)

    if moved:
        _save_roi_manifest(manifest_path, records)

    return moved, roi_diagnostics, orientation_diagnostics, skipped, rotated_items
