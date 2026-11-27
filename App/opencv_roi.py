from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import pymupdf


def _render_page_gray(page: pymupdf.Page, dpi: int = 180) -> tuple[np.ndarray, float]:
    scale = dpi / 72.0
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray, scale


def _extract_grid_mask(gray: np.ndarray) -> np.ndarray:
    """Extract horizontal and vertical structural border lines from the rendered page."""
    h, w = gray.shape
    # İnce ve orta kalınlıktaki koyu çizgileri yakalamak için adaptif + sabit eşik birleşimi
    _, dark = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)

    h_len = max(30, int(w * 0.22))
    v_len = max(30, int(h * 0.14))

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

    h_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kernel, iterations=1)
    v_lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, v_kernel, iterations=1)

    # Küçük kopuklukları kapatmak için hafif yatay/dikey birleştirme
    h_lines = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 2)), iterations=1)
    v_lines = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9)), iterations=1)

    return cv2.bitwise_or(h_lines, v_lines)


def detect_planview_bounds_opencv(
    page: pymupdf.Page,
    category: str,
    dpi: int = 180,
) -> Optional[tuple[float, float, float, float]]:
    """Detect planview bounding box (x0, y0, x1, y1) in PDF points using OpenCV.

    Returns None when no clear enclosed planview region can be isolated.
    """
    if page.rect.width <= 0 or page.rect.height <= 0:
        return None

    gray, scale = _render_page_gray(page, dpi=dpi)
    h, w = gray.shape
    grid_mask = _extract_grid_mask(gray)

    interior = cv2.bitwise_not(grid_mask)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(interior, connectivity=4)

    best_box: Optional[tuple[int, int, int, int]] = None
    best_area = 0

    is_banded_chart = category.upper() in {"APPROACH", "DEPARTURE", "SID"}
    min_w = int(w * 0.72)
    min_h = int(h * 0.28) if is_banded_chart else int(h * 0.50)
    min_y0 = int(h * 0.08)
    max_y0 = int(h * 0.28)
    max_y1 = int(h * 0.88) if is_banded_chart else int(h * 0.95)

    for label in range(1, num_labels):
        x, y, cw, ch, _ = stats[label]
        x1 = x + cw
        y1 = y + ch

        if x <= 2 or y <= 2 or x1 >= w - 2 or y1 >= h - 2:
            continue
        if cw < min_w or ch < min_h:
            continue
        if not (min_y0 <= y <= max_y0):
            continue
        if y1 > max_y1:
            continue

        area = cw * ch
        if area > best_area:
            best_area = area
            best_box = (x, y, x1, y1)

    if best_box is None:
        return None

    bx0, by0, bx1, by1 = best_box
    return (
        bx0 / scale,
        by0 / scale,
        bx1 / scale,
        by1 / scale,
    )