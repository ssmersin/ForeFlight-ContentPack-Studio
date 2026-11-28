from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orientation_turn import _approach_orientation


class ApproachFeatureDetectionTests(unittest.TestCase):
    def setUp(self):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=500, height=500)
        self.roi = pymupdf.Rect(50, 50, 450, 450)

    def tearDown(self):
        self.doc.close()

    def test_edge_tick_scanning_is_disabled_for_approach_orientation(self):
        self.page.draw_line((46, 100), (54, 100), color=(0, 0, 0), width=1)
        self.page.draw_line((446, 140), (454, 140), color=(0, 0, 0), width=1)
        self.page.draw_line((150, 46), (150, 54), color=(0, 0, 0), width=1)
        self.page.draw_line((250, 446), (250, 454), color=(0, 0, 0), width=1)

        result = _approach_orientation(self.page, self.roi)

        self.assertFalse(any("tick" in reference.kind.lower() for reference in result.references))

    def test_approach_boxes_and_airport_coordinates_are_retained(self):
        # Caution / emergency safe altitude / holding / IAF and airport data boxes.
        boxes = [
            (pymupdf.Rect(70, 70, 190, 120), ["CAUTION", "Minimum circling bank angle"]),
            (pymupdf.Rect(210, 70, 360, 120), ["EMERG SAFE ALT 100 NM 9500"]),
            (pymupdf.Rect(70, 150, 190, 205), ["HOLDING", "MAX HOLDING SPEED"]),
            (pymupdf.Rect(210, 150, 360, 215), ["IAF KEMCA", "N 38°59.46'", "E 027°33.93'"]),
            (pymupdf.Rect(70, 240, 210, 300), ["AKHISAR", "N 38°59.46'", "E 027°33.93'"]),
        ]
        for rect, labels in boxes:
            self.page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            y = rect.y0 + 18
            for label in labels:
                self.page.insert_text((rect.x0 + 7, y), label, fontsize=8)
                y += 13

        result = _approach_orientation(self.page, self.roi)
        kinds = {reference.kind for reference in result.references}

        self.assertIn("approach_box_text", kinds)
        self.assertIn("approach_emg", kinds)
        self.assertIn("approach_holding", kinds)
        self.assertIn("approach_iaf", kinds)
        self.assertIn("approach_airport_info", kinds)
        self.assertFalse(result.apply_rotation)


if __name__ == "__main__":
    unittest.main()
