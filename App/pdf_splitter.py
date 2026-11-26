from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject


_GEO_PAGE_KEYS = ("/VP", "/Measure", "/LGIDict")
_GEO_CATALOG_KEYS = ("/LGIDict",)


def _has_geospatial_data(reader: PdfReader) -> bool:
    """Detect GeoPDF dictionaries before choosing a page-copy strategy."""
    root = reader.trailer["/Root"]
    if any(key in root for key in _GEO_CATALOG_KEYS):
        return True
    for page in reader.pages:
        if any(key in page for key in _GEO_PAGE_KEYS):
            return True
    return False


def _copy_page_with_pymupdf(source: pymupdf.Document, page_index: int, target: Path) -> None:
    output = pymupdf.open()
    try:
        # A single-page file cannot retain meaningful links to other source pages.
        # Excluding these widgets also avoids carrying the source document's form tree.
        output.insert_pdf(
            source,
            from_page=page_index,
            to_page=page_index,
            links=False,
            annots=False,
            widgets=False,
        )
        output.set_metadata(source.metadata or {})
        output.save(target, garbage=1, deflate=True)
    finally:
        output.close()


def _copy_geo_page_with_pypdf(reader: PdfReader, page_index: int, target: Path) -> None:
    """Copy GeoPDF page dictionaries without rewriting geospatial structures."""
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    if reader.metadata:
        writer.add_metadata({str(k): str(v) for k, v in reader.metadata.items() if v is not None})

    # GeoPDF implementations may store geospatial dictionaries at catalog level.
    # Clone those objects rather than dropping them when creating a new one-page catalog.
    root = reader.trailer["/Root"]
    for key in _GEO_CATALOG_KEYS:
        if key in root:
            value = root.raw_get(key)
            if hasattr(value, "clone"):
                value = value.clone(writer)
            writer._root_object[NameObject(key)] = value

    with target.open("wb") as stream:
        writer.write(stream)


def split_chart_pages(
    source_pdf: Path,
    charts: Iterable[object],
    output_dir: Path,
) -> list[Path]:
    """Export each parsed chart as a one-page PDF using its manifest filename.

    Ordinary PDFs are optimized with PyMuPDF. GeoPDFs use pypdf object-copy
    so page-level viewports/measure dictionaries and catalog LGIDict data
    are retained rather than flattened or rasterized.
    """
    chart_list = list(charts)
    filenames = [chart.filename for chart in chart_list]
    if len(filenames) != len(set(filenames)):
        raise ValueError("Aynı dosya adı birden fazla sayfaya atanmış; dışa aktarma durduruldu.")

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(source_pdf), strict=False)
    is_geospatial = _has_geospatial_data(reader)
    source_doc = None if is_geospatial else pymupdf.open(source_pdf)
    generated: list[Path] = []

    try:
        for chart in chart_list:
            page_index = int(chart.pdf_page) - 1
            if page_index < 0 or page_index >= len(reader.pages):
                raise ValueError(f"PDF sayfa numarası geçersiz: {chart.pdf_page}")

            target = output_dir / chart.filename
            temporary = target.with_name(f".{target.stem}.tmp.pdf")
            try:
                if is_geospatial:
                    _copy_geo_page_with_pypdf(reader, page_index, temporary)
                else:
                    assert source_doc is not None
                    _copy_page_with_pymupdf(source_doc, page_index, temporary)

                # Validate page count and geometry without reparsing every embedded font.
                check_doc = pymupdf.open(temporary)
                try:
                    if len(check_doc) != 1:
                        raise RuntimeError(f"Tek sayfa doğrulaması başarısız: {chart.filename}")
                finally:
                    check_doc.close()
                if is_geospatial:
                    check = PdfReader(str(temporary), strict=False)
                    source_page = reader.pages[page_index]
                    output_page = check.pages[0]
                    for key in _GEO_PAGE_KEYS:
                        if key in source_page and key not in output_page:
                            raise RuntimeError(
                                f"GeoPDF verisi korunamadı ({key}): {chart.filename}"
                            )
                temporary.replace(target)
                generated.append(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
    finally:
        if source_doc is not None:
            source_doc.close()
        reader.stream.close()

    return generated
