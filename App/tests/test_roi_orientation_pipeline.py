from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import roi_orientation_pipeline
from orientation_turn import OrientationResult
from planview_roi import PlanviewRoi


class CombinedRoiOrientationPipelineTests(unittest.TestCase):
    def test_roi_and_orientation_use_one_source_open_and_save_rotated_page(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            initial_dir = root / "00-Initial"
            output_dir = root / "01-First"
            initial_dir.mkdir()
            source_pdf = initial_dir / "TEST.pdf"

            source_doc = pymupdf.open()
            source_doc.new_page(width=600, height=400)
            source_doc.save(source_pdf)
            source_doc.close()

            chart = SimpleNamespace(
                filename="TEST.pdf",
                category="Diagram",
                procedure="AERODROME DIAGRAM",
            )
            roi = PlanviewRoi(50, 60, 550, 350)
            orientation = OrientationResult(
                90, "variation_indicator", [], basis="test", apply_rotation=True
            )
            real_open = pymupdf.open

            with (
                patch.object(roi_orientation_pipeline.pymupdf, "open", wraps=real_open) as open_spy,
                patch.object(roi_orientation_pipeline, "detect_planview_roi", return_value=roi),
                patch.object(roi_orientation_pipeline, "detect_page_orientation", return_value=orientation),
            ):
                moved, roi_diags, turn_diags, skipped, rotated = (
                    roi_orientation_pipeline.run_combined_roi_orientation_stage(
                        initial_dir=initial_dir,
                        output_dir=output_dir,
                        charts=[chart],
                    )
                )

            self.assertEqual(open_spy.call_count, 1)
            self.assertEqual(len(moved), 1)
            self.assertEqual(skipped, [])
            self.assertEqual(len(rotated), 1)
            self.assertFalse(source_pdf.exists())
            self.assertTrue((output_dir / "TEST.pdf").is_file())

            output_doc = real_open(output_dir / "TEST.pdf")
            try:
                self.assertEqual(output_doc[0].rotation, 90)
                self.assertEqual(output_doc[0].rect.width, 400)
                self.assertEqual(output_doc[0].rect.height, 600)
            finally:
                output_doc.close()

            manifest = json.loads((output_dir / "roi_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["TEST.pdf"]["orientation"]["rotation_cw"], 90)
            self.assertTrue(manifest["TEST.pdf"]["orientation"]["applied"])
            self.assertIn("original_roi", manifest["TEST.pdf"])


if __name__ == "__main__":
    unittest.main()
