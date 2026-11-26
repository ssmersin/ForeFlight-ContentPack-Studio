from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject

from flip_parser import ChartInfo, _procedure_and_category
from pdf_splitter import split_chart_pages


class PdfSplitterTests(unittest.TestCase):
    def test_radar_instrument_approach_minimums_is_diagram(self) -> None:
        text = "DİYAGRAM                    APPROACH                         SID\n12-8\nRADAR INSTRUMENT APPROACH MINIMUMS\nCIGLI (LTBL)"
        procedure, category = _procedure_and_category(text)
        self.assertEqual(procedure, "RADAR INSTRUMENT APPROACH MINIMUMS")
        self.assertEqual(category, "Diagram")

    def test_navigation_header_does_not_turn_every_chart_into_diagram(self) -> None:
        text = "DİYAGRAM                    APPROACH                         SID\n1-2\nHILO TACAN RWY 34R\nAIRPORT (LTBJ)"
        procedure, category = _procedure_and_category(text)
        self.assertEqual(procedure, "HILO TACAN RWY 34R")
        self.assertEqual(category, "Approach")

    def test_geospatial_page_and_catalog_dictionaries_survive_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=100, height=100)
            measure = DictionaryObject({
                NameObject("/Subtype"): NameObject("/GEO"),
                NameObject("/GPTS"): ArrayObject([NumberObject(1), NumberObject(2)]),
            })
            viewport = DictionaryObject({
                NameObject("/Type"): NameObject("/Viewport"),
                NameObject("/BBox"): ArrayObject([NumberObject(0), NumberObject(0), NumberObject(100), NumberObject(100)]),
                NameObject("/Measure"): measure,
            })
            page[NameObject("/VP")] = ArrayObject([writer._add_object(viewport)])
            writer._root_object[NameObject("/LGIDict")] = writer._add_object(
                DictionaryObject({NameObject("/Version"): NumberObject(1)})
            )
            with source.open("wb") as stream:
                writer.write(stream)

            chart = ChartInfo(1, "1-1", "LTXX", "Diagram", "TEST", "geo-test.pdf")
            exported = split_chart_pages(source, [chart], root / "out")
            result = PdfReader(str(exported[0]))
            self.assertIn("/VP", result.pages[0])
            self.assertIn("/Measure", result.pages[0]["/VP"][0])
            self.assertIn("/LGIDict", result.trailer["/Root"])


if __name__ == "__main__":
    unittest.main()
