from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pymupdf
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject


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
        # Copy vector page content. Interactive links point into the original FLIP
        # and are not useful in a one-page export; their visible widget appearances
        # are flattened separately from the source page below.
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


def _flatten_widget_appearances(
    source_reader: PdfReader,
    page_index: int,
    base_pdf: Path,
    flattened_pdf: Path,
) -> None:
    """Paint widget /AP streams into page content without retaining form actions."""
    source_page = source_reader.pages[page_index]
    annotations = source_page.get("/Annots", [])
    if not annotations:
        base_pdf.replace(flattened_pdf)
        return

    widgets = [
        annotation.get_object()
        for annotation in annotations
        if annotation.get_object().get("/Subtype") == "/Widget"
    ]
    if not widgets:
        base_pdf.replace(flattened_pdf)
        return

    base_reader = PdfReader(str(base_pdf), strict=False)
    writer = PdfWriter(clone_from=base_reader)
    target_page = writer.pages[0]
    resources = target_page["/Resources"].get_object()
    xobjects = resources.get("/XObject")
    if xobjects is None:
        xobjects = DictionaryObject()
        resources[NameObject("/XObject")] = xobjects
    else:
        xobjects = xobjects.get_object()

    draw_commands: list[str] = []
    widget_index = 0
    for annotation in widgets:
        appearance_entry = annotation.get("/AP", {}).get("/N")
        if appearance_entry is None:
            continue
        if isinstance(appearance_entry, DictionaryObject):
            state = annotation.get("/AS")
            appearance_entry = appearance_entry.get(state)
            if appearance_entry is None:
                continue
        appearance = appearance_entry.get_object()
        if "/BBox" not in appearance:
            continue

        bbox = [float(value) for value in appearance["/BBox"]]
        matrix = [float(value) for value in appearance.get("/Matrix", [1, 0, 0, 1, 0, 0])]
        x0, y0, x1, y1 = bbox
        a, b, c, d, e, f = matrix
        transformed = [
            (a * x + c * y + e, b * x + d * y + f)
            for x in (x0, x1)
            for y in (y0, y1)
        ]
        min_x = min(point[0] for point in transformed)
        max_x = max(point[0] for point in transformed)
        min_y = min(point[1] for point in transformed)
        max_y = max(point[1] for point in transformed)
        rect = [float(value) for value in annotation["/Rect"]]
        llx, lly, urx, ury = rect
        if max_x == min_x or max_y == min_y:
            continue
        scale_x = (urx - llx) / (max_x - min_x)
        scale_y = (ury - lly) / (max_y - min_y)
        translate_x = llx - min_x * scale_x
        translate_y = lly - min_y * scale_y

        # Compose the annotation placement matrix with the appearance's own matrix.
        ctm = (
            scale_x * a,
            scale_y * b,
            scale_x * c,
            scale_y * d,
            scale_x * e + translate_x,
            scale_y * f + translate_y,
        )
        name = f"/FFWidget{widget_index}"
        xobjects[NameObject(name)] = appearance_entry.clone(writer)
        draw_commands.append(
            "q " + " ".join(f"{value:.8f}" for value in ctm)
            + f" cm {name} Do Q\n"
        )
        widget_index += 1

    if draw_commands:
        content = DecodedStreamObject()
        content.set_data("".join(draw_commands).encode("ascii"))
        existing_contents = target_page.raw_get("/Contents") if "/Contents" in target_page else None
        contents = ArrayObject()
        if isinstance(existing_contents, ArrayObject):
            contents.extend(existing_contents)
        elif existing_contents is not None:
            contents.append(existing_contents)
        contents.append(writer._add_object(content))
        target_page[NameObject("/Contents")] = contents

    with flattened_pdf.open("wb") as stream:
        writer.write(stream)
    base_pdf.unlink(missing_ok=True)
    base_reader.stream.close()


def _copy_geo_page_with_pypdf(reader: PdfReader, page_index: int, target: Path) -> None:
    """Copy GeoPDF page dictionaries without rewriting geospatial structures."""
    writer = PdfWriter()
    writer.add_page(reader.pages[page_index])
    if reader.metadata:
        writer.add_metadata({str(k): str(v) for k, v in reader.metadata.items() if v is not None})

    # GeoPDF implementations may store geospatial dictionaries at catalog level.
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

    Ordinary PDFs are copied as optimized vector pages and widget appearance
    streams are flattened into page content. GeoPDFs use pypdf object-copy so
    page viewports/measure dictionaries and catalog LGIDict data are retained.
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
            flattened = target.with_name(f".{target.stem}.flat.pdf")
            try:
                if is_geospatial:
                    _copy_geo_page_with_pypdf(reader, page_index, temporary)
                else:
                    assert source_doc is not None
                    _copy_page_with_pymupdf(source_doc, page_index, temporary)
                    _flatten_widget_appearances(reader, page_index, temporary, flattened)
                    temporary = flattened

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
                if flattened.exists():
                    flattened.unlink()
    finally:
        if source_doc is not None:
            source_doc.close()
        reader.stream.close()

    return generated
