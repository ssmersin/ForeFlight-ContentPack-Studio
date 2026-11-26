# ForeFlight Content Pack Studio

Windows-first Python project for analyzing aviation PDF publications and preparing ForeFlight content-pack data.

## Open the project

Open `ForeFlight-ContentPack-Studio.sln` in Visual Studio 2026. The solution contains the Python project in `App/`; project dependencies are managed through Visual Studio's Python Environments interface and `App/requirements.txt`.

## Current milestone

The first milestone analyzes FLIP PDFs without asking the user to re-enter information already printed on the cover. It reads the document type and effective date range from page 1, classifies chart pages, and writes a CSV manifest to `HEDEF/Initial/`.

PDF page splitting and content-pack generation will follow after page-level GeoPDF metadata preservation is verified. No branches are created without the repository owner's approval; development proceeds directly on `main`.
