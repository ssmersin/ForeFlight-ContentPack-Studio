# ForeFlight Content Pack Studio — App

Windows desktop application for the first FLIP PDF separation stage.

## First-time setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r App\requirements.txt
powershell -ExecutionPolicy Bypass -File App\setup_poppler.ps1
```

The setup script downloads the Windows Poppler package, verifies its SHA-256 checksum, and installs `pdftotext` under `App\tools\poppler`. The Poppler binaries are not committed to the repository.

Install the 64-bit Ghostscript AGPL release separately from [the official releases page](https://www.ghostscript.com/releases/). The application uses its command-line PDF writer to avoid copying unused shared fonts into every page file. Ghostscript is not bundled with this repository.

## Run

```powershell
py App\main.py
```

The default export directory is `HEDEF\Initial` beside the `App` directory. Select a source FLIP PDF, review the detected chart pages and filenames, then export. Existing files are skipped rather than overwritten.

## Current scope

- Read FLIP page text, including pages with rotated chart content.
- Identify chart page number, ICAO airport code, category, and procedure title.
- Export each detected chart as a compact one-page PDF.
- Keep the original PDF page number visible in the log for traceability.

This is the initial extraction stage only. GeoPDF georeferencing, coordinate extraction, and content-pack generation are intentionally out of scope.
