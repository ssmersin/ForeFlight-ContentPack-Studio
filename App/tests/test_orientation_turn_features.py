from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orientation_turn import (
    _approach_orientation, _collect_roi_edge_ticks,
    _collect_text_lines, _find_rma_latitude_vote,
)


class ApproachFeatureDetectionTests(unittest.TestCase):
    def setUp(self):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=500, height=500)
        self.roi = pymupdf.Rect(50, 50, 450, 450)

    def tearDown(self):
        self.doc.close()

    def test_edge_ticks_are_found_on_all_four_edges_without_direction_assumption(self):
        self.page.draw_line((46, 100), (54, 100), color=(0, 0, 0), width=1)
        self.page.draw_line((446, 140), (454, 140), color=(0, 0, 0), width=1)
        self.page.draw_line((150, 46), (150, 54), color=(0, 0, 0), width=1)
        self.page.draw_line((250, 446), (250, 454), color=(0, 0, 0), width=1)

        ticks = _collect_roi_edge_ticks(self.page, self.roi)

        self.assertEqual({tick.label for tick in ticks}, {"LEFT", "RIGHT", "TOP", "BOTTOM"})
        self.assertTrue(all(tick.kind == "edge_tick" for tick in ticks))

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


class RmaEdgeTickValidationTests(unittest.TestCase):
    def setUp(self):
        self.doc = pymupdf.open()
        self.page = self.doc.new_page(width=500, height=500)
        self.roi = pymupdf.Rect(50, 50, 450, 450)

    def tearDown(self):
        self.doc.close()

    def add_pair(self, y):
        self.page.draw_line((46, y), (54, y), color=(0, 0, 0), width=1)
        self.page.draw_line((446, y), (454, y), color=(0, 0, 0), width=1)

    def add_left_label(self, y, text="35-10N"):
        self.page.insert_text((54, y + 3), text, fontsize=7)

    def vote(self):
        lines = _collect_text_lines(self.page, self.roi)
        return _find_rma_latitude_vote(self.page, lines, self.roi)

    def test_opposite_edge_pair_with_one_label_is_accepted(self):
        self.add_pair(150)
        self.add_left_label(150)

        vote = self.vote()

        self.assertIsNotNone(vote)
        self.assertFalse(vote["ignored"])
        self.assertEqual(vote["n"], 1)
        self.assertEqual(sum(box.kind == "lat_tick" for box in vote["boxes"]), 1)
        self.assertEqual(sum(box.kind == "lat_opposite_tick" for box in vote["boxes"]), 1)

    def test_single_unpaired_tick_requires_a_coordinate_label(self):
        self.page.draw_line((46, 150), (54, 150), color=(0, 0, 0), width=1)
        rejected = self.vote()
        self.assertIsNotNone(rejected)
        self.assertTrue(rejected["ignored"])

        self.add_left_label(150)
        vote = self.vote()
        self.assertIsNotNone(vote)
        self.assertFalse(vote["ignored"])
        self.assertEqual(vote["n"], 1)

    def test_multiple_unpaired_ticks_are_rejected(self):
        self.page.draw_line((46, 100), (54, 100), color=(0, 0, 0), width=1)
        self.page.draw_line((46, 200), (54, 200), color=(0, 0, 0), width=1)
        self.add_left_label(100)

        rejected = self.vote()
        self.assertIsNotNone(rejected)
        self.assertTrue(rejected["ignored"])


    def test_pair_with_labels_on_both_edges_is_rejected(self):
        self.add_pair(150)
        self.add_left_label(150)
        self.page.insert_text((420, 153), "35-10N", fontsize=7)

        rejected = self.vote()
        self.assertIsNotNone(rejected)
        self.assertTrue(rejected["ignored"])

    def test_equal_tick_intervals_are_accepted(self):
        for y, label in ((100, "35-10N"), (200, "35-11N"), (300, "35-12N")):
            self.add_pair(y)
            self.add_left_label(y, label)

        vote = self.vote()
        self.assertIsNotNone(vote)
        self.assertFalse(vote["ignored"])
        self.assertEqual(vote["n"], 3)

    def test_irregular_tick_intervals_are_rejected(self):
        for y, label in ((100, "35-10N"), (200, "35-11N"), (330, "35-12N")):
            self.add_pair(y)
            self.add_left_label(y, label)

        rejected = self.vote()
        self.assertIsNotNone(rejected)
        self.assertTrue(rejected["ignored"])


if __name__ == "__main__":
    unittest.main()
