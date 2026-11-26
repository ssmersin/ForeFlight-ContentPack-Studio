from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

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

    def test_widget_appearance_is_flattened_into_page_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "widget-source.pdf"
            writer = PdfWriter()
            page = writer.add_blank_page(width=100, height=100)
            appearance = DecodedStreamObject()
            appearance.set_data(b"0 0 1 rg 0 0 50 10 re f")
            appearance[NameObject("/Type")] = NameObject("/XObject")
            appearance[NameObject("/Subtype")] = NameObject("/Form")
            appearance[NameObject("/BBox")] = ArrayObject([
                NumberObject(0), NumberObject(0), NumberObject(50), NumberObject(10)
            ])
            appearance[NameObject("/Resources")] = DictionaryObject()
            widget = DictionaryObject({
                NameObject("/Type"): NameObject("/Annot"),
                NameObject("/Subtype"): NameObject("/Widget"),
                NameObject("/Rect"): ArrayObject([
                    NumberObject(10), NumberObject(10), NumberObject(60), NumberObject(20)
                ]),
                NameObject("/AP"): DictionaryObject({
                    NameObject("/N"): writer._add_object(appearance)
                }),
            })
            page[NameObject("/Annots")] = ArrayObject([writer._add_object(widget)])
            with source.open("wb") as stream:
                writer.write(stream)

            chart = ChartInfo(1, "1-1", "LTXX", "Diagram", "WIDGET", "widget-test.pdf")
            exported = split_chart_pages(source, [chart], root / "out")
            result = PdfReader(str(exported[0]))
            page_content = result.pages[0].get_contents().get_data()
            self.assertIn(b"/FFWidget0 Do", page_content)
            self.assertIn("/FFWidget0", result.pages[0]["/Resources"]["/XObject"])
            self.assertNotIn("/Annots", result.pages[0])

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
