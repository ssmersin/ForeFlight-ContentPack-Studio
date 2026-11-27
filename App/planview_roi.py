from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import pymupdf

from opencv_roi import detect_planview_bounds_opencv

SUPPORTED_DIAGRAM_PROCEDURES = frozenset({
    "AERODROME DIAGRAM",
    "RADAR MINIMUM ALTITUDES",
})

SINGLE_FRAME_TOP_STRIP_CATEGORIES = frozenset({
    "ARRIVAL",
    "STAR",
})

BANDED_PLANVIEW_CATEGORIES = frozenset({
    "APPROACH",
    "DEPARTURE",
    "SID",
})

APPROACH_TOP_KEYWORDS = (
    "ATIS",
    "APP CON",
    "TOWER",
    "GND CON",
    "AD ELEV",
    "TDZE",
    "RWY LDG",
    "MDA",
    "TA :",
)

APPROACH_BOTTOM_KEYWORDS = (
    "MISSED APPROACH",
    "CATEGORY",
    "CIRCLING",
    "MSA ",
    "TCH",
)


@dataclass(frozen=True)
class PlanviewRoi:
    x0: float
    y0: float
    x1: float
    y1: float

    def to_rect(self) -> pymupdf.Rect:
        return pymupdf.Rect(self.x0, self.y0, self.x1, self.y1)


def _is_dark_value(color: Optional[tuple[float, ...]], threshold: float = 0.35) -> bool:
    if not color:
        return False
    if len(color) == 4:
        c, m, y, k = color
        return k >= 0.65 or (max(c, m, y) < 0.35 and k >= 0.40)
    return max(color) <= threshold


def _drawing_is_dark(drawing: dict) -> bool:
    if _is_dark_value(drawing.get("color")):
        return True
    if _is_dark_value(drawing.get("fill")):
        return True
    return False


def _extract_segments(
    page: pymupdf.Page,
    rect_candidates: Optional[list[PlanviewRoi]] = None,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Extract horizontal segments (y, x_min, x_max) and vertical segments (x, y_min, y_max)."""
    width = page.rect.width
    height = page.rect.height
    h_segs: list[tuple[float, float, float]] = []
    v_segs: list[tuple[float, float, float]] = []

    for drawing in page.get_drawings():
        is_dark = _drawing_is_dark(drawing)
        for item in drawing.get("items", []):
            kind = item[0]
            if kind == "re":
                r: pymupdf.Rect = item[1]
                if is_dark and rect_candidates is not None:
                    if r.width >= width * 0.70 and r.height >= height * 0.55:
                        if height * 0.08 <= r.y0 <= height * 0.28 and height * 0.78 <= r.y1 <= height * 0.95:
                            rect_candidates.append(PlanviewRoi(r.x0, r.y0, r.x1, r.y1))
                if r.width > 12.0:
                    h_segs.append((r.y0, r.x0, r.x1))
                    h_segs.append((r.y1, r.x0, r.x1))
                if r.height > 12.0:
                    v_segs.append((r.x0, r.y0, r.y1))
                    v_segs.append((r.x1, r.y0, r.y1))
            elif kind == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) <= 1.2 and abs(p2.x - p1.x) > 12.0:
                    h_segs.append(((p1.y + p2.y) / 2.0, min(p1.x, p2.x), max(p1.x, p2.x)))
                elif abs(p1.x - p2.x) <= 1.2 and abs(p2.y - p1.y) > 12.0:
                    v_segs.append(((p1.x + p2.x) / 2.0, min(p1.y, p2.y), max(p1.y, p2.y)))

    return h_segs, v_segs


def _is_closed_rectangle_frame(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    h_segs: list[tuple[float, float, float]],
    v_segs: list[tuple[float, float, float]],
    tol: float = 3.5,
) -> bool:
    """Verify that top, bottom, left, and right edges form a closed rectangle."""
    top_ok = any(
        abs(y - y0) <= tol and sx0 <= x0 + tol and sx1 >= x1 - tol
        for y, sx0, sx1 in h_segs
    )
    bottom_ok = any(
        abs(y - y1) <= tol and sx0 <= x0 + tol and sx1 >= x1 - tol
        for y, sx0, sx1 in h_segs
    )
    left_ok = any(
        abs(x - x0) <= tol and sy0 <= y0 + tol and sy1 >= y1 - tol
        for x, sy0, sy1 in v_segs
    )
    right_ok = any(
        abs(x - x1) <= tol and sy0 <= y0 + tol and sy1 >= y1 - tol
        for x, sy0, sy1 in v_segs
    )
    if not (top_ok and bottom_ok and left_ok and right_ok):
        return False

    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    for cx, cy in corners:
        has_h = any(
            abs(y - cy) <= tol and min(sx0, sx1) - tol <= cx <= max(sx0, sx1) + tol
            for y, sx0, sx1 in h_segs
        )
        has_v = any(
            abs(x - cx) <= tol and min(sy0, sy1) - tol <= cy <= max(sy0, sy1) + tol
            for x, sy0, sy1 in v_segs
        )
        if not (has_h and has_v):
            return False

    return True


def _rma_top_border_y(
    roi: PlanviewRoi,
    h_segs: list[tuple[float, float, float]],
) -> Optional[float]:
    """Return the y coordinate of the horizontal divider below the RMA top info strip."""
    frame_width = roi.x1 - roi.x0
    frame_height = roi.y1 - roi.y0
    if frame_width <= 0 or frame_height <= 0:
        return None

    min_y = roi.y0 + 5.0
    max_y = roi.y0 + max(25.0, frame_height * 0.08)

    candidates = [
        y for y, sx0, sx1 in h_segs
        if min_y <= y <= max_y
        and (sx1 - sx0) >= frame_width * 0.70
        and sx0 <= roi.x0 + 5.0
        and sx1 >= roi.x1 - 5.0
    ]
    if not candidates:
        return None

    return max(candidates)


def _apply_procedure_roi_refinements(
    page: pymupdf.Page,
    roi: PlanviewRoi,
    h_segs: list[tuple[float, float, float]],
    procedure: Optional[str],
    category: Optional[str] = None,
) -> PlanviewRoi:
    proc_upper = (procedure or "").upper()
    cat_upper = (category or "").upper()
    if "RADAR MINIMUM ALTITUDE" in proc_upper or cat_upper in SINGLE_FRAME_TOP_STRIP_CATEGORIES:
        rma_y0 = _rma_top_border_y(roi, h_segs)
        if rma_y0 is not None and rma_y0 > roi.y0 and (roi.y1 - rma_y0) >= page.rect.height * 0.50:
            return PlanviewRoi(x0=roi.x0, y0=rma_y0, x1=roi.x1, y1=roi.y1)
    return roi


def detect_closed_black_frame(
    page: pymupdf.Page,
    h_segs: Optional[list[tuple[float, float, float]]] = None,
    v_segs: Optional[list[tuple[float, float, float]]] = None,
    procedure: Optional[str] = None,
    category: Optional[str] = None,
) -> Optional[PlanviewRoi]:
    """Find the main closed dark rectangular border of a single-frame planview page.

    Returns None if no closed frame is found; never falls back to guessed bounds.
    """
    rect = page.rect
    width = rect.width
    height = rect.height
    if width <= 0 or height <= 0:
        return None

    rect_candidates: list[PlanviewRoi] = []
    extracted_h, extracted_v = _extract_segments(page, rect_candidates=rect_candidates)
    h_list = h_segs if h_segs is not None else extracted_h
    v_list = v_segs if v_segs is not None else extracted_v

    if rect_candidates:
        rect_candidates.sort(key=lambda c: (c.x1 - c.x0) * (c.y1 - c.y0), reverse=True)
        return _apply_procedure_roi_refinements(
            page, rect_candidates[0], h_list, procedure, category=category
        )

    top_ys = sorted({
        y for y, sx0, sx1 in h_list
        if (sx1 - sx0) >= width * 0.70 and height * 0.08 <= y <= height * 0.28
    })
    bottom_ys = sorted({
        y for y, sx0, sx1 in h_list
        if (sx1 - sx0) >= width * 0.70 and height * 0.80 <= y <= height * 0.94
    })
    left_xs = sorted({
        x for x, sy0, sy1 in v_list
        if (sy1 - sy0) >= height * 0.55 and width * 0.01 <= x <= width * 0.18
    })
    right_xs = sorted({
        x for x, sy0, sy1 in v_list
        if (sy1 - sy0) >= height * 0.55 and width * 0.82 <= x <= width * 0.99
    })

    best_roi: Optional[PlanviewRoi] = None
    best_area = 0.0

    for y0 in top_ys:
        for y1 in bottom_ys:
            if y1 - y0 < height * 0.55:
                continue
            for x0 in left_xs:
                for x1 in right_xs:
                    if x1 - x0 < width * 0.75:
                        continue
                    if _is_closed_rectangle_frame(x0, y0, x1, y1, h_list, v_list):
                        area = (x1 - x0) * (y1 - y0)
                        if area > best_area:
                            best_area = area
                            best_roi = PlanviewRoi(x0, y0, x1, y1)

    if best_roi is None:
        return None
    return _apply_procedure_roi_refinements(page, best_roi, h_list, procedure, category=category)


def _find_horizontal_dividers(
    page: pymupdf.Page,
    h_segs: list[tuple[float, float, float]],
    min_width_ratio: float = 0.65,
) -> list[float]:
    min_len = page.rect.width * min_width_ratio
    ys = [y for y, sx0, sx1 in h_segs if (sx1 - sx0) >= min_len]
    ys.sort()
    merged: list[float] = []
    for y in ys:
        if not merged or abs(y - merged[-1]) > 1.5:
            merged.append(y)
        else:
            merged[-1] = (merged[-1] + y) / 2.0
    return merged


def detect_approach_planview_roi(
    page: pymupdf.Page,
    h_segs: Optional[list[tuple[float, float, float]]] = None,
    v_segs: Optional[list[tuple[float, float, float]]] = None,
) -> Optional[PlanviewRoi]:
    """Detect the planview region between top header tables and bottom profile/minima.

    Returns None if top or bottom boundaries cannot be reliably determined.
    """
    rect = page.rect
    width = rect.width
    height = rect.height
    if width <= 0 or height <= 0:
        return None

    extracted_h, extracted_v = _extract_segments(page)
    h_list = h_segs if h_segs is not None else extracted_h
    v_list = v_segs if v_segs is not None else extracted_v

    dividers = _find_horizontal_dividers(page, h_list, min_width_ratio=0.65)
    blocks = page.get_text("blocks")

    top_table_bottom = 0.0
    bottom_section_top = height

    for b in blocks:
        if len(b) < 5:
            continue
        by0, by1, raw_text = float(b[1]), float(b[3]), str(b[4])
        upper = " ".join(raw_text.upper().split())
        if not upper:
            continue

        if by0 < height * 0.25 and any(k in upper for k in APPROACH_TOP_KEYWORDS):
            top_table_bottom = max(top_table_bottom, by1)

        if by0 > height * 0.45 and any(k in upper for k in APPROACH_BOTTOM_KEYWORDS):
            bottom_section_top = min(bottom_section_top, by0)

    top_candidates = [
        y for y in dividers
        if height * 0.08 <= y <= height * 0.26 and y >= top_table_bottom - 2.0
    ]
    if top_candidates:
        y0 = top_candidates[0]
    elif top_table_bottom > 0.0:
        y0 = top_table_bottom + 2.0
    else:
        return None

    bottom_candidates = [
        y for y in dividers
        if y >= max(y0 + height * 0.35, height * 0.52)
        and y <= height * 0.86
        and y <= bottom_section_top + 4.0
    ]
    if bottom_candidates:
        y1 = bottom_candidates[0]
    elif bottom_section_top < height:
        y1 = bottom_section_top - 2.0
    else:
        return None

    if (y1 - y0) < height * 0.25:
        return None

    left_xs = [
        x for x, sy0, sy1 in v_list
        if (sy1 - sy0) >= height * 0.35 and sy0 <= y0 + 5.0 and sy1 >= y1 - 5.0 and x < width * 0.20
    ]
    right_xs = [
        x for x, sy0, sy1 in v_list
        if (sy1 - sy0) >= height * 0.35 and sy0 <= y0 + 5.0 and sy1 >= y1 - 5.0 and x > width * 0.80
    ]

    x0 = max(left_xs) if left_xs else width * 0.025
    x1 = min(right_xs) if right_xs else width * 0.975

    if (x1 - x0) < width * 0.70:
        return None

    return PlanviewRoi(x0=x0, y0=y0, x1=x1, y1=y1)


def _is_single_frame_chart(category: str, procedure: str) -> bool:
    cat_upper = category.upper()
    proc_upper = procedure.upper()
    if category == "Diagram" and proc_upper in SUPPORTED_DIAGRAM_PROCEDURES:
        return True
    return cat_upper in SINGLE_FRAME_TOP_STRIP_CATEGORIES


def detect_planview_roi(page: pymupdf.Page, category: str, procedure: str) -> Optional[PlanviewRoi]:
    """Dispatch ROI detection by chart type; falls back to OpenCV when rules do not match."""
    h_segs, v_segs = _extract_segments(page)
    cat_upper = category.upper()

    if _is_single_frame_chart(category, procedure):
        roi = detect_closed_black_frame(
            page, h_segs=h_segs, v_segs=v_segs, procedure=procedure, category=category
        )
        if roi is not None:
            return roi

        # Vektör çizim/çerçeve bulunamazsa (ör. farklı format veya Jeppesen düzeni) OpenCV dene
        bounds = detect_planview_bounds_opencv(page, category=category)
        if bounds is not None:
            return PlanviewRoi(*bounds)
        return None

    if cat_upper in BANDED_PLANVIEW_CATEGORIES:
        bounds = detect_planview_bounds_opencv(page, category=category)
        if bounds is not None:
            return PlanviewRoi(*bounds)
        return None

    # Tanımlı kategori dışındaki yeni doküman formatlarında da OpenCV ile şans ver
    bounds = detect_planview_bounds_opencv(page, category=category)
    if bounds is not None:
        return PlanviewRoi(*bounds)
    return None


def _save_diagnostic_image(
    page_or_pdf,
    roi: PlanviewRoi,
    out_jpg: Path,
    dpi: int = 150,
) -> None:
    """Render an open page (or legacy PDF path) with its ROI rectangle."""
    doc = None
    if isinstance(page_or_pdf, pymupdf.Page):
        # Draw on an in-memory copy so diagnostics never contaminate the PDF being saved.
        doc = pymupdf.open(stream=page_or_pdf.parent.tobytes(), filetype="pdf")
        page = doc[page_or_pdf.number]
    else:
        doc = pymupdf.open(page_or_pdf)
        page = doc[0]
    try:
        page.draw_rect(roi.to_rect(), color=(1.0, 0.92, 0.0), width=3.0)
        zoom = dpi / 72.0
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        out_jpg.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_jpg), jpg_quality=90)
    finally:
        if doc is not None:
            doc.close()


def _save_roi_manifest(manifest_path: Path, records: dict[str, dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_first_stage_planview_charts(
    initial_dir: Path,
    first_dir: Path,
    charts: Iterable[object],
    diagnostic: bool = False,
    diagnostic_dir: Path | None = None,
) -> tuple[list[Path], list[Path], list[str]]:
    """Move supported planview charts with detected ROI into 01-First.

    Persists detected ROI bounds in 01-First/roi_manifest.json for later stages.
    Charts without a detected ROI remain in 00-Initial.
    Returns (moved_pdfs, diagnostic_images, skipped_filenames).
    """
    selected = []
    for chart in charts:
        category = str(getattr(chart, "category", ""))
        procedure = str(getattr(chart, "procedure", ""))
        if _is_single_frame_chart(category, procedure) or category.upper() in BANDED_PLANVIEW_CATEGORIES:
            selected.append((chart, category, procedure))

    if not selected:
        return [], [], []

    manifest_path = first_dir / "roi_manifest.json"
    roi_records: dict[str, dict] = {}
    if manifest_path.is_file():
        roi_records = json.loads(manifest_path.read_text(encoding="utf-8"))

    moved: list[Path] = []
    diagnostics: list[Path] = []
    skipped: list[str] = []

    for chart, category, procedure in selected:
        src_pdf = initial_dir / chart.filename
        if not src_pdf.is_file():
            continue

        doc = pymupdf.open(src_pdf)
        try:
            page = doc[0]
            page_rect = page.rect
            roi = detect_planview_roi(page, category=category, procedure=procedure)
        finally:
            doc.close()

        if roi is None:
            skipped.append(chart.filename)
            continue

        first_dir.mkdir(parents=True, exist_ok=True)
        if diagnostic and diagnostic_dir is not None:
            diagnostic_dir.mkdir(parents=True, exist_ok=True)
            diag_path = diagnostic_dir / f"{Path(chart.filename).stem}.jpg"
            _save_diagnostic_image(src_pdf, roi, diag_path)
            diagnostics.append(diag_path)

        dst_pdf = first_dir / chart.filename
        shutil.move(str(src_pdf), str(dst_pdf))
        moved.append(dst_pdf)

        roi_records[chart.filename] = {
            "procedure": procedure,
            "category": category,
            "page_width": page_rect.width,
            "page_height": page_rect.height,
            "roi": asdict(roi),
        }

    if moved:
        _save_roi_manifest(manifest_path, roi_records)

    return moved, diagnostics, skipped