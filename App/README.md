# ForeFlight Content Pack Studio — Python project

This stage is a **document-analysis core**, not a user-interface implementation.

## Open in Visual Studio 2026

Open `ForeFlight-ContentPack-Studio.sln` from the repository root. The solution contains a Python project (`App/ForeFlightContentPackStudio.pyproj`).

Use the Visual Studio **Python Environments** window to create/select the project's environment. The requirements file is included in the project; use its context menu to install dependencies when the project later gains Python packages. The current analysis milestone intentionally has no third-party Python package dependencies.

## PDF text engine

The FLIP sample uses embedded/custom character maps that require Poppler's `pdftotext` for reliable extraction. The PowerShell script `setup_poppler.ps1` downloads Poppler, verifies its SHA-256 checksum, and places it under `App/tools/poppler`.

You do not upload the PS1 file anywhere. It is source code in the repository. To run it from Visual Studio, open the file and use **Open in PowerShell** / run it in Visual Studio's PowerShell terminal. It is only a one-time local setup step; it does not modify the source PDF.

## Current analysis workflow

Run the project with a source PDF path as the command-line argument:

```text
Kitap.pdf
```

The application reads the document type and effective date range from page 1, analyzes chart pages, and writes `HEDEF/Initial/chart_manifest.csv`. The filename date is derived from the cover's effective-to date (for example, `01 MAR 2027` becomes `MAR2027`).

The current milestone deliberately stops at analysis and manifest generation. It does not yet split or rewrite the source PDF. This avoids accidentally damaging page-level GeoPDF information while the correct preservation strategy is being validated.
