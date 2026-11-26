# ForeFlight Content Pack Studio — Python project

This stage is a **document-analysis core**, not a user-interface implementation.

## Open in Visual Studio 2026

Open `ForeFlight-ContentPack-Studio.sln` from the repository root. The solution contains a Python project (`App/ForeFlightContentPackStudio.pyproj`).

Use Visual Studio's **Python Environments** window to create/select the project's Python environment. The `requirements.txt` file is included in the project and can be installed from its context menu when dependencies are added. The current analysis milestone intentionally has no third-party Python package dependencies.

## PDF text engine

The sample FLIP PDF uses embedded/custom character maps that require Poppler's `pdftotext` for reliable text extraction. The application checks for Poppler and, if missing, automatically downloads the Windows package, verifies its SHA-256 checksum, and installs it under `App/tools/poppler` on first use. No manual PS1 upload or separate installation step is required.

## Current analysis workflow

Set the source PDF path in the Visual Studio project's **Debug > Script Arguments** property, or run the project with a PDF path argument. For example:

```text
D:\\ForeFlight\\Kitap.pdf
```

The application reads the document type and effective date range from page 1, analyzes chart pages, and writes `HEDEF/Initial/chart_manifest.csv`. The filename date is derived from the cover's effective-to date (for example, `01 MAR 2027` becomes `MAR2027`).

The current milestone deliberately stops at analysis and manifest generation. It does not yet split or rewrite the source PDF. This avoids accidentally damaging page-level GeoPDF information while the correct preservation strategy is being validated.
