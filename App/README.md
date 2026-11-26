# ForeFlight Content Pack Studio — Python project

This stage is a **document-analysis and chart-page export core**, not a user-interface implementation.

## Open in Visual Studio 2026

Open \`ForeFlight-ContentPack-Studio.sln\` from the repository root. The solution contains the Python project in \`App/\`.

Use Visual Studio's **Python Environments** window to create/select the project's Python environment. Install \`App/requirements.txt\` into that environment. The export engine uses PyMuPDF for ordinary PDF pages and pypdf for widget appearance flattening and GeoPDF-safe page copying.

## PDF text engine

The sample FLIP PDF uses embedded/custom character maps that require Poppler's \`pdftotext\` for reliable text extraction. The application checks for Poppler and, if missing, automatically downloads the Windows package, verifies its SHA-256 checksum, and installs it under \`App/tools/poppler\` on first use. No manual PS1 upload or separate installation step is required.

## Workflow

Set the source PDF path in the Visual Studio project's **Debug > Script Arguments** property, or run the project with a PDF path argument. For example:

\`\`\`text
D:\\ForeFlight\\Kitap.pdf
\`\`\`

The application:

1. Reads the document type and effective date range from page 1.
2. Analyzes chart pages and writes \`HEDEF/Initial/chart_manifest.csv\`.
3. Exports every detected chart page as a separate, one-page PDF into the same \`HEDEF/Initial/\` folder.
4. Replaces an existing export with the same generated filename when rerun.

Filenames are derived from the cover's effective-to date (for example, \`01 MAR 2027\` becomes \`MAR2027\`). The PDF page number in the manifest is the source document's 1-based page number.

## PDF preservation strategy

- **Ordinary PDFs:** PyMuPDF copies vector page content. Visible form/widget appearances (such as FLIP's top navigation tabs and home icon) are flattened into the page as vector form XObjects. Their interactive actions and links are not retained because their destinations refer to the original multi-page publication.
- **GeoPDFs:** Before export, the source is checked for page \`/VP\`, \`/Measure\`, \`/LGIDict\`, and catalog \`/LGIDict\` structures. GeoPDF pages use pypdf object copying rather than page rewriting/rasterization; detected page-level geospatial dictionaries are checked in the exported file. Catalog-level \`/LGIDict\` is copied as well.

The current \`Kitap.pdf\` sample follows the ordinary-PDF path. Its page artwork and visible navigation widget appearances were compared against the source. The GeoPDF path is covered by a synthetic preservation test; representative real-world GeoPDFs are still required for broader validation.

## Output

Default output folder: \`HEDEF/Initial/\`

- \`chart_manifest.csv\`
- One PDF per chart page, using the filename in the manifest

Unresolved pages are listed in the console and cause exit code \`3\`; successful analysis/export with no unresolved pages returns \`0\`.
